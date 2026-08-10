"""Exact parameter sweeps, withholding conventions, and Monte Carlo checks."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Iterable

import numpy as np
import pandas as pd

from src.theory.gating import Losses, Parameters, analytical_metrics, simulate


SEED = 20260809


def convention_losses(
    convention: str, *, c_fp: float, c_fn: float, c_a: float = 0.0,
    c_tp: float = 0.0, c_tn: float = 0.0,
) -> Losses:
    """Construct losses without mixing withholding conventions."""
    if convention == "pure_abstention":
        w1 = w0 = c_a
    elif convention == "abstention_plus_missed":
        w1, w0 = c_a + c_fn, c_a
    elif convention == "treated_as_negative":
        w1, w0 = c_fn, c_tn
    else:
        raise ValueError(f"Unknown withholding convention: {convention}")
    return Losses(c_fp, c_fn, w1, w0, c_tp, c_tn)


def complete_metrics(p: Parameters, losses: Losses) -> dict[str, float]:
    """Return every required candidate and operational population metric."""
    m = analytical_metrics(p, losses)
    pi, s, f = p.prevalence, p.sensitivity, p.false_positive_rate
    q1, q0 = p.retain_true_candidate, p.retain_false_candidate
    negative_frequency = pi * (1.0 - s) + (1.0 - pi) * (1.0 - f)
    m.update({
        "candidate_sensitivity": s,
        "candidate_specificity": 1.0 - f,
        "operational_sensitivity": s * q1,
        "explicit_negative_specificity": 1.0 - f,
        "non_action_specificity": 1.0 - f * q0,
        "false_action_rate": f * q0,
        "unconditional_false_action_frequency": (1.0 - pi) * f * q0,
        "explicit_negative_frequency": negative_frequency,
        "unacted_event_probability": pi * (1.0 - s * q1),
        "delta_q": q1 - q0,
        "ppv_change": m["ppv_gated"] - m["ppv_ungated"],
        "false_action_reduction": (1.0 - pi) * f * (1.0 - q0),
    })
    return m


def grid_rows(
    parameters: Iterable[tuple[Parameters, Losses, dict[str, object]]]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for p, losses, labels in parameters:
        row = {**labels, **asdict(p), **asdict(losses), **complete_metrics(p, losses)}
        row["loss_region"] = (
            "beneficial" if row["delta_loss"] < -1e-12
            else "harmful" if row["delta_loss"] > 1e-12 else "neutral"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def prevalence_boundary(p: Parameters, losses: Losses) -> float:
    """Exact Delta_L=0 prevalence boundary where it is well defined."""
    benefit = p.false_positive_rate * (1-p.retain_false_candidate) * (
        losses.false_positive-losses.withheld_false
    )
    harm = p.sensitivity * (1-p.retain_true_candidate) * (
        losses.withheld_true-losses.true_positive
    )
    denominator = benefit + harm
    return benefit / denominator if denominator > 0 else math.nan


def acquisition_cost_delta(p: Parameters, losses: Losses, c_q: float) -> float:
    """Loss difference when Q costs c_q for every P=1 candidate.

    This is a labeled extension: Delta_L_Q = Delta_L + C_Q Pr(P=1).
    """
    base = analytical_metrics(p, losses)["delta_loss"]
    candidate_frequency = (
        p.prevalence*p.sensitivity
        + (1-p.prevalence)*p.false_positive_rate
    )
    return base + c_q*candidate_frequency


def monte_carlo_record(
    name: str, p: Parameters, losses: Losses, *, observations: int, seed: int,
    convention: str,
) -> dict[str, object]:
    exact = complete_metrics(p, losses)
    sampled = simulate(p, losses, observations=observations, seed=seed)
    a = p.prevalence*p.sensitivity*(1-p.retain_true_candidate)
    b = (1-p.prevalence)*p.false_positive_rate*(1-p.retain_false_candidate)
    d1 = losses.withheld_true-losses.true_positive
    d0 = losses.withheld_false-losses.false_positive
    variance = max(0.0, a*d1*d1+b*d0*d0-exact["delta_loss"]**2)
    se = math.sqrt(variance/observations)
    return {
        "scenario": name, "convention": convention, "observations": observations,
        "seed": seed, **asdict(p), **asdict(losses),
        "delta_q": p.retain_true_candidate-p.retain_false_candidate,
        "delta_loss_exact": exact["delta_loss"],
        "delta_loss_mc": sampled["delta_loss"], "delta_loss_mc_se": se,
        "delta_loss_mc_ci_low": sampled["delta_loss"]-1.96*se,
        "delta_loss_mc_ci_high": sampled["delta_loss"]+1.96*se,
        "ppv_ungated_exact": exact["ppv_ungated"],
        "ppv_gated_exact": exact["ppv_gated"],
        "ppv_ungated_mc": sampled["ppv_ungated"],
        "ppv_gated_mc": sampled["ppv_gated"],
        "operational_sensitivity_exact": exact["operational_sensitivity"],
        "operational_sensitivity_mc": sampled["sensitivity_gated"],
        "false_action_frequency_exact": exact["unconditional_false_action_frequency"],
        "false_action_frequency_mc": sampled["false_actions_gated"],
        "withholding_frequency_exact": exact["withholding_frequency"],
        "withholding_frequency_mc": sampled["withholding_frequency"],
    }
