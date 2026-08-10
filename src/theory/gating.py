"""Analytical metrics and Monte Carlo checks for binary admissibility gating."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Dict


@dataclass(frozen=True)
class Parameters:
    prevalence: float
    sensitivity: float
    false_positive_rate: float
    retain_true_candidate: float
    retain_false_candidate: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")


@dataclass(frozen=True)
class Losses:
    false_positive: float
    false_negative: float
    withheld_true: float
    withheld_false: float
    true_positive: float = 0.0
    true_negative: float = 0.0


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def analytical_metrics(p: Parameters, losses: Losses) -> Dict[str, float]:
    """Return population metrics for the ungated and gated policies."""
    pi = p.prevalence
    s = p.sensitivity
    f = p.false_positive_rate
    q1 = p.retain_true_candidate
    q0 = p.retain_false_candidate

    tp_u, fp_u = pi * s, (1.0 - pi) * f
    fn_u, tn_u = pi * (1.0 - s), (1.0 - pi) * (1.0 - f)
    tp_g, fp_g = tp_u * q1, fp_u * q0
    wt, wf = tp_u * (1.0 - q1), fp_u * (1.0 - q0)

    loss_u = (
        tp_u * losses.true_positive
        + fp_u * losses.false_positive
        + fn_u * losses.false_negative
        + tn_u * losses.true_negative
    )
    loss_g = (
        tp_g * losses.true_positive
        + fp_g * losses.false_positive
        + fn_u * losses.false_negative
        + tn_u * losses.true_negative
        + wt * losses.withheld_true
        + wf * losses.withheld_false
    )

    return {
        "ppv_ungated": _ratio(tp_u, tp_u + fp_u),
        "ppv_gated": _ratio(tp_g, tp_g + fp_g),
        "fdr_ungated": _ratio(fp_u, tp_u + fp_u),
        "fdr_gated": _ratio(fp_g, tp_g + fp_g),
        "sensitivity_ungated": s,
        "sensitivity_gated": s * q1,
        "explicit_negative_specificity_ungated": 1.0 - f,
        "explicit_negative_specificity_gated": 1.0 - f,
        "non_action_specificity_gated": 1.0 - f * q0,
        "false_actions_ungated": fp_u,
        "false_actions_gated": fp_g,
        "missed_events_explicit_negative": fn_u,
        "unacted_events_gated": fn_u + wt,
        "action_frequency_ungated": tp_u + fp_u,
        "action_frequency_gated": tp_g + fp_g,
        "withholding_frequency": wt + wf,
        "loss_ungated": loss_u,
        "loss_gated": loss_g,
        "delta_loss": loss_g - loss_u,
        "delta_loss_identity": (
            tp_u * (1.0 - q1) * (losses.withheld_true - losses.true_positive)
            + fp_u * (1.0 - q0) * (losses.withheld_false - losses.false_positive)
        ),
    }


def simulate(
    p: Parameters, losses: Losses, *, observations: int, seed: int
) -> Dict[str, float]:
    """Simulate the joint table with independent draws conditional on truth.

    The simulator realizes the exact five-parameter population model. It does
    not impose an additional independence claim because Q is drawn using its
    truth- and candidate-conditional probability.
    """
    if observations <= 0:
        raise ValueError("observations must be positive")

    rng = random.Random(seed)
    tp_u = fp_u = fn_u = tn_u = tp_g = fp_g = wt = wf = 0

    for _ in range(observations):
        truth = rng.random() < p.prevalence
        candidate_probability = (
            p.sensitivity if truth else p.false_positive_rate
        )
        candidate = rng.random() < candidate_probability

        if candidate:
            if truth:
                tp_u += 1
                if rng.random() < p.retain_true_candidate:
                    tp_g += 1
                else:
                    wt += 1
            else:
                fp_u += 1
                if rng.random() < p.retain_false_candidate:
                    fp_g += 1
                else:
                    wf += 1
        elif truth:
            fn_u += 1
        else:
            tn_u += 1

    n = float(observations)
    loss_u = (
        tp_u * losses.true_positive
        + fp_u * losses.false_positive
        + fn_u * losses.false_negative
        + tn_u * losses.true_negative
    ) / n
    loss_g = (
        tp_g * losses.true_positive
        + fp_g * losses.false_positive
        + fn_u * losses.false_negative
        + tn_u * losses.true_negative
        + wt * losses.withheld_true
        + wf * losses.withheld_false
    ) / n

    return {
        "ppv_ungated": _ratio(tp_u, tp_u + fp_u),
        "ppv_gated": _ratio(tp_g, tp_g + fp_g),
        "false_actions_ungated": fp_u / n,
        "false_actions_gated": fp_g / n,
        "sensitivity_gated": _ratio(tp_g, tp_u + fn_u),
        "withholding_frequency": (wt + wf) / n,
        "loss_ungated": loss_u,
        "loss_gated": loss_g,
        "delta_loss": loss_g - loss_u,
    }
