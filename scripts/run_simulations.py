"""Generate all exact sweeps and seeded stochastic verification artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.analytical import (
    SEED, acquisition_cost_delta, complete_metrics, convention_losses,
    grid_rows, monte_carlo_record, prevalence_boundary,
)
from src.simulation.finite_sample import run_finite_sample
from src.simulation.score_level import (
    ScoreModel, evaluate_policy, gate_parameters, generate_scores,
    run_strategy_experiment, same_score_equivalence,
)
from src.theory.gating import Losses, Parameters


SIM = ROOT / "results" / "simulations"
TABLES = ROOT / "results" / "tables"
CONVENTIONS = ("pure_abstention", "abstention_plus_missed", "treated_as_negative")


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(SIM / name, index=False)


def exact_sweeps() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    q_values = np.linspace(0, 1, 61)
    records = []
    for convention in CONVENTIONS:
        losses = convention_losses(convention, c_fp=10, c_fn=4, c_a=1)
        for q1 in q_values:
            for q0 in q_values:
                records.append((Parameters(.05, .85, .12, q1, q0), losses,
                                {"convention": convention}))
    frames["phase_q1_q0.csv"] = grid_rows(records)

    prevalence = np.unique(np.r_[np.logspace(-4, -1, 55), np.linspace(.1, .9, 55)])
    costs = np.logspace(-1, 2, 75)
    records = []
    for convention in CONVENTIONS:
        for pi in prevalence:
            for c_fp in costs:
                losses = convention_losses(convention, c_fp=c_fp, c_fn=4, c_a=1)
                p = Parameters(pi, .85, .12, .86, .28)
                records.append((p, losses, {"convention": convention,
                                            "analytical_prevalence_boundary": prevalence_boundary(p, losses)}))
    frames["phase_prevalence_cost.csv"] = grid_rows(records)

    records = []
    for convention in CONVENTIONS:
        losses = convention_losses(convention, c_fp=10, c_fn=4, c_a=1)
        for pi in prevalence:
            for q1 in q_values:
                records.append((Parameters(pi, .85, .12, q1, .30), losses,
                                {"convention": convention}))
    frames["phase_prevalence_retention.csv"] = grid_rows(records)

    records = []
    for w1 in np.linspace(0, 15, 76):
        for c_fp in np.logspace(-1, 2, 75):
            losses = Losses(c_fp, 4, w1, 1)
            records.append((Parameters(.05, .85, .12, .85, .30), losses,
                            {"convention": "general_state_dependent"}))
    frames["phase_withholding_cost.csv"] = grid_rows(records)

    records = []
    losses = convention_losses("abstention_plus_missed", c_fp=10, c_fn=4, c_a=1)
    for s in np.linspace(.05, 1, 61):
        for f in np.linspace(0, .5, 61):
            records.append((Parameters(.05, s, f, .85, .30), losses,
                            {"convention": "abstention_plus_missed"}))
    frames["phase_sensitivity_fpr.csv"] = grid_rows(records)

    rows = []
    p = Parameters(.05, .85, .12, .85, .30)
    losses = convention_losses("abstention_plus_missed", c_fp=10, c_fn=4, c_a=1)
    for c_q in np.linspace(0, 5, 101):
        rows.append({"c_q": c_q, "base_delta_loss": complete_metrics(p, losses)["delta_loss"],
                     "modified_delta_loss": acquisition_cost_delta(p, losses, c_q),
                     "candidate_frequency": p.prevalence*p.sensitivity+(1-p.prevalence)*p.false_positive_rate})
    frames["acquisition_cost.csv"] = pd.DataFrame(rows)

    for filename, frame in frames.items():
        save(frame, filename)
    return frames


def monte_carlo_checks() -> pd.DataFrame:
    rows = []
    n = 500_000
    seed_offset = 0
    for convention in CONVENTIONS:
        losses = convention_losses(convention, c_fp=10, c_fn=4, c_a=1)
        template = Parameters(.05, .85, .12, .85, .30)
        boundary = prevalence_boundary(template, losses)
        scenarios = [
            (f"{convention}_beneficial", Parameters(max(.001, boundary*.35), .85, .12, .85, .30)),
            (f"{convention}_near_boundary", Parameters(boundary, .85, .12, .85, .30)),
            (f"{convention}_harmful", Parameters(min(.90, boundary+(1-boundary)*.55), .85, .12, .85, .30)),
        ]
        for name, p in scenarios:
            rows.append(monte_carlo_record(name, p, losses, observations=n,
                                           seed=SEED+seed_offset, convention=convention))
            seed_offset += 1
    extras = [
        ("truth_independent", Parameters(.10,.80,.15,.40,.40), Losses(8,3,2,.5), "general"),
        ("positive_delta_q_harmful", Parameters(.50,.50,.50,.90,.80), Losses(1,2,10,10), "general"),
        ("negative_delta_q_beneficial", Parameters(.001,.50,.50,.20,.30), Losses(100,1,1,0), "general"),
    ]
    for name, p, losses, convention in extras:
        rows.append(monte_carlo_record(name, p, losses, observations=n,
                                       seed=SEED+seed_offset, convention=convention))
        seed_offset += 1
    frame = pd.DataFrame(rows)
    save(frame, "monte_carlo_verification.csv")
    return frame


def score_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    losses = convention_losses("abstention_plus_missed", c_fp=5, c_fn=1, c_a=.2)
    models = {
        "conditionally_independent": ScoreModel(rho0=0, rho1=0),
        "positively_correlated": ScoreModel(rho0=.50, rho1=.50),
        "strongly_redundant": ScoreModel(mu_z1=1.35, rho0=.90, rho1=.90),
        "complementary_information": ScoreModel(mu_z1=1.65, rho0=-.10, rho1=-.10),
        "shared_measurement_error": ScoreModel(rho0=.80, rho1=.80),
    }
    strategy_frames, detail_rows = [], []
    for i, (name, model) in enumerate(models.items()):
        frame, details = run_strategy_experiment(model=model, losses=losses, tau=.55,
                                                 gate_c=.35, seed=SEED+100*i)
        frame["evidence_environment"] = name
        frame["withholding_convention"] = "abstention_plus_missed"
        strategy_frames.append(frame)
        detail_rows.append({"evidence_environment": name, **details})
    strategies = pd.concat(strategy_frames, ignore_index=True)
    details = pd.DataFrame(detail_rows)
    save(strategies, "score_strategy_results.csv")
    save(details, "score_strategy_tuning.csv")

    data = generate_scores(100_000, models["conditionally_independent"], SEED+900)
    equal = same_score_equivalence(data["s"], .55, 1.05)
    equivalence = pd.DataFrame([{
        "observations": len(data["s"]), "tau": .55, "kappa": 1.05,
        "equivalent_threshold": 1.05, "mismatch_count": 0 if equal else int(np.sum(
            ((data["s"] >= .55) & (data["s"] >= 1.05)) != (data["s"] >= 1.05)
        )), "observationwise_equal": equal, "seed": SEED+900,
    }])
    save(equivalence, "same_score_equivalence.csv")

    shift_rows = []
    for i, rho in enumerate(np.linspace(-.6, .9, 16)):
        model = ScoreModel(rho0=float(rho), rho1=float(rho))
        sample = generate_scores(180_000, model, SEED+1000+i)
        gp = gate_parameters(sample, .55, .35)
        truth, candidate = sample["truth"], sample["s"] >= .55
        s_hat = np.mean(candidate[truth]); f_hat = np.mean(candidate[~truth])
        p = Parameters(float(np.mean(truth)), float(s_hat), float(f_hat), gp["q1"], gp["q0"])
        metrics = complete_metrics(p, losses)
        shift_rows.append({"rho": rho, "z_environment_shift": 0, **gp,
                           "ppv_gated": metrics["ppv_gated"],
                           "operational_sensitivity": metrics["operational_sensitivity"],
                           "delta_loss": metrics["delta_loss"], "seed": SEED+1000+i})
    for i, z_shift in enumerate(np.linspace(-1, 1, 11)):
        model = ScoreModel(rho0=.2, rho1=.2, z_environment_shift=float(z_shift))
        sample = generate_scores(180_000, model, SEED+1100+i)
        gp = gate_parameters(sample, .55, .35)
        truth, candidate = sample["truth"], sample["s"] >= .55
        p = Parameters(float(np.mean(truth)), float(np.mean(candidate[truth])),
                       float(np.mean(candidate[~truth])), gp["q1"], gp["q0"])
        metrics = complete_metrics(p, losses)
        shift_rows.append({"rho": .2, "z_environment_shift": z_shift, **gp,
                           "ppv_gated": metrics["ppv_gated"],
                           "operational_sensitivity": metrics["operational_sensitivity"],
                           "delta_loss": metrics["delta_loss"], "seed": SEED+1100+i})
    shifts = pd.DataFrame(shift_rows)
    save(shifts, "distribution_shift.csv")
    return strategies, equivalence, shifts


def finite_sample_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    index = 0
    for scenario in ("strong_benefit", "near_boundary", "rare_false_retention"):
        for pi in (.001, .01, .10):
            for n in (1_000, 5_000, 20_000):
                q0 = .01 if scenario == "rare_false_retention" else .30
                p = Parameters(pi, .85, .12, .85, q0)
                if scenario == "near_boundary":
                    harm = pi*.85*(1-.85)*5
                    denominator = (1-pi)*.12*(1-q0)
                    c_fp = 1+(harm+.005)/denominator
                else:
                    c_fp = 10
                losses = convention_losses(
                    "abstention_plus_missed", c_fp=c_fp, c_fn=4, c_a=1
                )
                frame = run_finite_sample(p, losses, n=n, repeats=400,
                                          seed=SEED+2000+index)
                frame["scenario"] = scenario
                frame["false_positive_cost"] = c_fp
                frames.append(frame)
                index += 1
    replicates = pd.concat(frames, ignore_index=True)
    save(replicates, "finite_sample_replicates.csv")
    summary = replicates.groupby(["scenario","prevalence","n"], as_index=False).agg(
        repeats=("replicate","count"), median_true_candidates=("true_candidates","median"),
        median_false_candidates=("false_candidates","median"),
        median_delta_q=("delta_q_hat","median"), wrong_loss_sign_rate=("wrong_loss_sign","mean"),
        q0_zero_rate=("observed_q0_zero","mean"),
        median_delta_loss=("delta_loss_hat","median"), exact_delta_loss=("exact_delta_loss","first"),
        false_positive_cost=("false_positive_cost","first"),
    )
    save(summary, "finite_sample_summary.csv")
    return replicates, summary


def write_tables(mc: pd.DataFrame, strategies: pd.DataFrame, finite: pd.DataFrame) -> None:
    metric_rows = [
        ("Candidate sensitivity", "s"), ("Candidate specificity", "1-f"),
        ("Operational sensitivity", "s q1"), ("Explicit-negative specificity", "1-f"),
        ("Non-action specificity", "1-f q0"), ("PPV", "pi s q1/[pi s q1+(1-pi)f q0]"),
        ("FDR", "1-PPV"), ("False-action rate", "f q0"),
        ("False-action frequency", "(1-pi)f q0"),
        ("Action frequency", "pi s q1+(1-pi)f q0"),
        ("Withholding frequency", "pi s(1-q1)+(1-pi)f(1-q0)"),
        ("Explicit-negative frequency", "pi(1-s)+(1-pi)(1-f)"),
        ("Unacted-event probability", "pi(1-s q1)"),
        ("Expected-loss change", "pi s(1-q1)(W1-C_TP)+(1-pi)f(1-q0)(W0-C_FP)"),
    ]
    table1 = pd.DataFrame(metric_rows, columns=["metric","analytical_formula"])
    table1.to_csv(TABLES/"table1_metric_definitions.csv", index=False)
    table2 = mc[["scenario","convention","prevalence","sensitivity","false_positive_rate",
                 "retain_true_candidate","retain_false_candidate","delta_q","delta_loss_exact",
                 "delta_loss_mc","delta_loss_mc_ci_low","delta_loss_mc_ci_high"]]
    table2.to_csv(TABLES/"table2_representative_scenarios.csv", index=False)
    table3 = strategies.copy()
    table3.to_csv(TABLES/"table3_strategy_performance.csv", index=False)
    table4 = finite.copy()
    table4.to_csv(TABLES/"table4_finite_sample_reliability.csv", index=False)
    for number, frame, title in [
        (1, table1, "Principal policy metrics"), (2, table2, "Representative scenarios"),
        (3, table3, "Score-level decision strategies"), (4, table4, "Finite-sample reliability"),
    ]:
        (TABLES/f"table{number}.md").write_text(
            f"# Table {number}. {title}\n\n"+frame.to_markdown(index=False)+"\n", encoding="utf-8"
        )


def main() -> None:
    SIM.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    exact = exact_sweeps()
    mc = monte_carlo_checks()
    strategies, equivalence, shifts = score_experiments()
    _, finite = finite_sample_experiments()
    write_tables(mc, strategies, finite)
    q_grid = exact["phase_q1_q0.csv"]
    nondegenerate = q_grid[(q_grid.retain_true_candidate>0) & (q_grid.retain_false_candidate>0)]
    ppv_sign = np.sign(np.where(nondegenerate.ppv_change.abs()<1e-12, 0, nondegenerate.ppv_change))
    delta_q_sign = np.sign(np.where(nondegenerate.delta_q.abs()<1e-12, 0, nondegenerate.delta_q))
    ppv_sign_mismatches = int(np.sum(ppv_sign != delta_q_sign))
    mc_covered = (
        (mc.delta_loss_exact >= mc.delta_loss_mc_ci_low)
        & (mc.delta_loss_exact <= mc.delta_loss_mc_ci_high)
    )
    bayes_risk_violations = 0
    for environment, group in strategies.groupby("evidence_environment"):
        bayes = float(group.loc[group.strategy=="population_bayes_s_z", "known_model_expected_risk"].iloc[0])
        bayes_risk_violations += int(np.sum(group.known_model_expected_risk < bayes-1e-12))
    metadata = {
        "seed": SEED, "analytical_surfaces_use_monte_carlo": False,
        "score_train_validation_test_independent": True,
        "same_score_equivalence_verified": bool(equivalence.iloc[0]["observationwise_equal"]),
        "withholding_conventions": list(CONVENTIONS),
        "ppv_delta_q_sign_mismatches": ppv_sign_mismatches,
        "negative_false_action_reduction_cells": int(np.sum(q_grid.false_action_reduction < -1e-12)),
        "max_loss_identity_error": float((q_grid.delta_loss-q_grid.delta_loss_identity).abs().max()),
        "monte_carlo_95pct_intervals_covering_exact": int(mc_covered.sum()),
        "monte_carlo_scenarios": int(len(mc)),
        "population_bayes_risk_violations": bayes_risk_violations,
        "maximum_near_boundary_wrong_sign_rate": float(finite.wrong_loss_sign_rate.max()),
        "maximum_observed_q0_zero_rate_when_true_q0_point01": float(finite.q0_zero_rate.max()),
    }
    (SIM/"run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Generated {len(list(SIM.glob('*.csv')))} simulation CSV files")
    print(f"Generated {len(list(TABLES.glob('*')))} table artifacts")
    print("Same-score mismatch count:", int(equivalence.iloc[0]["mismatch_count"]))
    print("Maximum Monte Carlo |Delta_L error|:", float((mc.delta_loss_mc-mc.delta_loss_exact).abs().max()))


if __name__ == "__main__":
    main()
