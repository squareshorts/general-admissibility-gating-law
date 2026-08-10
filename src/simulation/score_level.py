"""Continuous correlated evidence simulation and decision-strategy controls.

Conditional on truth t, scores follow a bivariate normal model:

    S = mu_S[t] + E_S
    Z = mu_Z[t] + rho E_S + sqrt(1-rho^2) E_Z.

Changing rho represents shared conditional variation/error. Environment shifts
may change rho and either score mean, so binary q1 and q0 remain valid within an
environment while need not transport between environments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from src.theory.gating import Losses


@dataclass(frozen=True)
class ScoreModel:
    prevalence: float = 0.08
    mu_s0: float = 0.0
    mu_s1: float = 1.35
    mu_z0: float = 0.0
    mu_z1: float = 1.20
    rho0: float = 0.15
    rho1: float = 0.15
    z_environment_shift: float = 0.0


def generate_scores(n: int, model: ScoreModel, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    truth = rng.random(n) < model.prevalence
    es, ez = rng.standard_normal((2, n))
    rho = np.where(truth, model.rho1, model.rho0)
    s = np.where(truth, model.mu_s1, model.mu_s0) + es
    z = (
        np.where(truth, model.mu_z1, model.mu_z0)
        + rho*es + np.sqrt(np.maximum(0.0, 1-rho*rho))*ez
        + model.z_environment_shift
    )
    return {"truth": truth, "s": s, "z": z}


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35, 35)
    return 1/(1+np.exp(-x))


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float = 1e-5) -> np.ndarray:
    """Small deterministic Newton solver; columns are standardized internally."""
    means, scales = x.mean(axis=0), x.std(axis=0)
    scales = np.where(scales == 0, 1, scales)
    xs = (x-means)/scales
    design = np.column_stack([np.ones(len(x)), xs])
    beta = np.zeros(design.shape[1])
    penalty = np.eye(len(beta))*ridge
    penalty[0, 0] = 0
    for _ in range(60):
        prob = sigmoid(design@beta)
        weight = np.maximum(prob*(1-prob), 1e-8)
        hessian = design.T@(design*weight[:, None])+penalty
        gradient = design.T@(y-prob)-penalty@beta
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return np.concatenate([beta, means, scales])


def predict_logistic(model: np.ndarray, x: np.ndarray) -> np.ndarray:
    p = x.shape[1]
    beta, means, scales = model[:p+1], model[p+1:2*p+1], model[2*p+1:]
    design = np.column_stack([np.ones(len(x)), (x-means)/scales])
    return sigmoid(design@beta)


def bayes_three_actions(probability: np.ndarray, losses: Losses) -> tuple[np.ndarray, np.ndarray]:
    """Choose positive, negative, or withheld by posterior expected loss."""
    positive_loss = probability*losses.true_positive+(1-probability)*losses.false_positive
    negative_loss = probability*losses.false_negative+(1-probability)*losses.true_negative
    withheld_loss = probability*losses.withheld_true+(1-probability)*losses.withheld_false
    action = (positive_loss <= negative_loss) & (positive_loss <= withheld_loss)
    withheld = (withheld_loss < positive_loss) & (withheld_loss < negative_loss)
    return action, withheld


def known_model_posterior(s: np.ndarray, z: np.ndarray, model: ScoreModel) -> np.ndarray:
    """Exact posterior for models with equal class-conditional covariance."""
    if not math.isclose(model.rho0, model.rho1):
        raise ValueError("Known linear posterior requires rho0 == rho1")
    rho = model.rho0
    inverse = np.array([[1, -rho], [-rho, 1]])/(1-rho*rho)
    mu0 = np.array([model.mu_s0, model.mu_z0+model.z_environment_shift])
    mu1 = np.array([model.mu_s1, model.mu_z1+model.z_environment_shift])
    delta = mu1-mu0
    intercept = math.log(model.prevalence/(1-model.prevalence)) - .5*(mu1@inverse@mu1-mu0@inverse@mu0)
    return sigmoid(intercept+np.column_stack([s,z])@(inverse@delta))


def evaluate_policy(
    truth: np.ndarray, action: np.ndarray, withheld: np.ndarray, losses: Losses,
) -> dict[str, float]:
    truth, action, withheld = truth.astype(bool), action.astype(bool), withheld.astype(bool)
    if np.any(action & withheld):
        raise ValueError("Action and withholding must be disjoint")
    negative = ~(action | withheld)
    tp, fp = np.mean(truth & action), np.mean(~truth & action)
    wt, wf = np.mean(truth & withheld), np.mean(~truth & withheld)
    fn, tn = np.mean(truth & negative), np.mean(~truth & negative)
    pi = np.mean(truth)
    return {
        "prevalence": pi,
        "operational_sensitivity": tp/pi if pi else math.nan,
        "false_action_rate": fp/(1-pi) if pi < 1 else math.nan,
        "ppv": tp/(tp+fp) if tp+fp else math.nan,
        "action_frequency": tp+fp,
        "withholding_frequency": wt+wf,
        "explicit_negative_frequency": fn+tn,
        "unacted_event_probability": fn+wt,
        "expected_loss": tp*losses.true_positive+fp*losses.false_positive
        + fn*losses.false_negative+tn*losses.true_negative
        + wt*losses.withheld_true+wf*losses.withheld_false,
    }


def posterior_expected_risk(
    probability: np.ndarray, action: np.ndarray, withheld: np.ndarray, losses: Losses,
) -> float:
    positive = probability*losses.true_positive+(1-probability)*losses.false_positive
    negative = probability*losses.false_negative+(1-probability)*losses.true_negative
    abstain = probability*losses.withheld_true+(1-probability)*losses.withheld_false
    return float(np.mean(np.where(action, positive, np.where(withheld, abstain, negative))))


def best_threshold(values: np.ndarray, truth: np.ndarray, losses: Losses) -> float:
    candidates = np.quantile(values, np.linspace(0.01, 0.99, 199))
    best = min(candidates, key=lambda t: evaluate_policy(
        truth, values >= t, np.zeros(len(values), dtype=bool), losses
    )["expected_loss"])
    return float(best)


def best_confidence_gate(
    s: np.ndarray, truth: np.ndarray, tau: float, losses: Losses,
) -> float:
    candidates = np.quantile(s[s >= tau], np.linspace(0.02, 0.98, 99))
    def loss(k: float) -> float:
        candidate = s >= tau
        return evaluate_policy(truth, s >= k, candidate & (s < k), losses)["expected_loss"]
    return float(min(candidates, key=loss))


def best_hard_gate(
    s: np.ndarray, z: np.ndarray, truth: np.ndarray, losses: Losses,
) -> tuple[float, float]:
    """Tune a rectangular conjunction on validation data only."""
    s_grid = np.quantile(s, np.linspace(.25, .98, 25))
    z_grid = np.quantile(z, np.linspace(.10, .95, 25))
    best_loss, best_pair = math.inf, (float(s_grid[0]), float(z_grid[0]))
    for tau in s_grid:
        candidate = s >= tau
        for c in z_grid:
            action = candidate & (z >= c)
            value = evaluate_policy(truth, action, candidate & ~action, losses)["expected_loss"]
            if value < best_loss:
                best_loss, best_pair = value, (float(tau), float(c))
    return best_pair


def run_strategy_experiment(
    *, model: ScoreModel, losses: Losses, tau: float, gate_c: float,
    seed: int, n_train: int = 40_000, n_validation: int = 40_000,
    n_test: int = 120_000,
) -> tuple[pd.DataFrame, dict[str, float]]:
    train = generate_scores(n_train, model, seed)
    validation = generate_scores(n_validation, model, seed+1)
    test = generate_scores(n_test, model, seed+2)

    s_model = fit_logistic(train["s"][:, None], train["truth"])
    joint_model = fit_logistic(np.column_stack([train["s"], train["z"]]), train["truth"])
    val_s_prob = predict_logistic(s_model, validation["s"][:, None])
    s_retuned = best_threshold(validation["s"], validation["truth"], losses)
    posterior_cut = losses.false_positive/(losses.false_positive+losses.false_negative)
    # Validation tuning is separate from the theoretical cost cut and test data.
    confidence_k = best_confidence_gate(validation["s"], validation["truth"], tau, losses)
    tuned_tau, tuned_c = best_hard_gate(
        validation["s"], validation["z"], validation["truth"], losses
    )

    candidate = test["s"] >= tau
    hard_action = candidate & (test["z"] >= gate_c)
    hard_withheld = candidate & ~hard_action
    retention = hard_action.sum()/max(candidate.sum(), 1)
    rng = np.random.default_rng(seed+3)
    random_action = candidate & (rng.random(n_test) < retention)
    random_withheld = candidate & ~random_action
    noise = rng.standard_normal(n_test)
    noise_cut = np.quantile(noise[candidate], 1-retention)
    independent_action = candidate & (noise >= noise_cut)
    independent_withheld = candidate & ~independent_action

    test_s_prob = predict_logistic(s_model, test["s"][:, None])
    test_joint_prob = predict_logistic(joint_model, np.column_stack([test["s"], test["z"]]))
    joint_action, joint_withheld = bayes_three_actions(test_joint_prob, losses)
    population_probability = known_model_posterior(test["s"], test["z"], model)
    population_action, population_withheld = bayes_three_actions(population_probability, losses)
    zero = np.zeros(n_test, dtype=bool)
    strategies = {
        "original_threshold": (candidate, zero),
        "hard_conjunctive_gate": (hard_action, hard_withheld),
        "tuned_hard_conjunctive_gate": (
            (test["s"] >= tuned_tau) & (test["z"] >= tuned_c),
            (test["s"] >= tuned_tau) & (test["z"] < tuned_c),
        ),
        "matched_random_gate": (random_action, random_withheld),
        "truth_independent_gate": (independent_action, independent_withheld),
        "retuned_s_threshold": (test["s"] >= s_retuned, zero),
        "detector_confidence_abstention": (
            test["s"] >= confidence_k, candidate & (test["s"] < confidence_k)
        ),
        "cost_sensitive_s_posterior": (test_s_prob >= posterior_cut, zero),
        "joint_s_z_model": (joint_action, joint_withheld),
        "population_bayes_s_z": (population_action, population_withheld),
        "or_rule": ((test["s"] >= tau) | (test["z"] >= gate_c), zero),
    }
    rows = []
    for name, (action, withheld) in strategies.items():
        rows.append({
            "strategy": name, "split": "test", "seed": seed,
            "known_model_expected_risk": posterior_expected_risk(
                population_probability, action, withheld, losses
            ),
            **evaluate_policy(test["truth"], action, withheld, losses),
        })
    details = {
        "tau": tau, "gate_c": gate_c, "hard_candidate_retention": retention,
        "retuned_s_threshold": s_retuned, "confidence_gate_threshold": confidence_k,
        "cost_sensitive_posterior_cut": posterior_cut,
        "tuned_hard_tau": tuned_tau, "tuned_hard_c": tuned_c,
    }
    return pd.DataFrame(rows), details


def same_score_equivalence(s: np.ndarray, tau: float, kappa: float) -> bool:
    return bool(np.array_equal((s >= tau) & (s >= kappa), s >= max(tau, kappa)))


def gate_parameters(data: dict[str, np.ndarray], tau: float, c: float) -> dict[str, float]:
    truth, candidate, accepted = data["truth"], data["s"] >= tau, data["z"] >= c
    t_candidates, f_candidates = candidate & truth, candidate & ~truth
    q1 = np.mean(accepted[t_candidates]) if t_candidates.any() else math.nan
    q0 = np.mean(accepted[f_candidates]) if f_candidates.any() else math.nan
    return {"q1": q1, "q0": q0, "delta_q": q1-q0}
