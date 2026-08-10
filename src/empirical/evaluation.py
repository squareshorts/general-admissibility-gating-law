"""Leakage-safe empirical evaluation utilities."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd


def binary_metrics(
    truth: np.ndarray, candidate: np.ndarray, action: np.ndarray
) -> dict[str, float]:
    """Metrics for candidate generation followed by an admissibility gate."""
    truth = np.asarray(truth, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    action = np.asarray(action, dtype=bool)
    if not np.all(~action | candidate):
        raise ValueError("actions must be a subset of candidates")
    false = ~truth
    withheld = candidate & ~action

    def ratio(num: np.ndarray, den: np.ndarray) -> float:
        count = int(den.sum())
        return float(num.sum() / count) if count else math.nan

    return {
        "n": float(len(truth)),
        "prevalence": float(truth.mean()),
        "candidate_frequency": float(candidate.mean()),
        "candidate_ppv": ratio(truth & candidate, candidate),
        "action_ppv": ratio(truth & action, action),
        "candidate_sensitivity": ratio(truth & candidate, truth),
        "operational_sensitivity": ratio(truth & action, truth),
        "candidate_false_action_rate": ratio(false & candidate, false),
        "false_action_rate": ratio(false & action, false),
        "q1": ratio(truth & candidate & action, truth & candidate),
        "q0": ratio(false & candidate & action, false & candidate),
        "delta_q": ratio(truth & candidate & action, truth & candidate)
        - ratio(false & candidate & action, false & candidate),
        "action_frequency": float(action.mean()),
        "withholding_frequency": float(withheld.mean()),
        "true_candidates_withheld": ratio(truth & withheld, truth & candidate),
        "false_candidates_removed": ratio(false & withheld, false & candidate),
    }


def expected_binary_loss(
    truth: np.ndarray,
    action: np.ndarray,
    *,
    false_positive_cost: float,
    false_negative_cost: float = 1.0,
    acquisition_cost: float = 0.0,
    candidates: np.ndarray | None = None,
) -> float:
    truth = np.asarray(truth, dtype=bool)
    action = np.asarray(action, dtype=bool)
    clinical = (
        false_positive_cost * np.mean(~truth & action)
        + false_negative_cost * np.mean(truth & ~action)
    )
    acquired = np.mean(candidates) if candidates is not None else 0.0
    return float(clinical + acquisition_cost * acquired)


def select_youden_threshold(scores: np.ndarray, truth: np.ndarray) -> float:
    """Select a detector threshold on validation data only."""
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    cuts = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 399)))
    tpr = np.array([np.mean(scores[truth] >= cut) for cut in cuts])
    fpr = np.array([np.mean(scores[~truth] >= cut) for cut in cuts])
    best = np.flatnonzero((tpr - fpr) == np.max(tpr - fpr))
    return float(cuts[best[-1]])


def threshold_for_action_rate(scores: np.ndarray, rate: float) -> float:
    rate = float(np.clip(rate, 0.0, 1.0))
    return float(np.quantile(scores, 1.0 - rate))


def cluster_bootstrap_metrics(
    frame: pd.DataFrame,
    *,
    truth: str,
    candidate: str,
    action: str,
    cluster: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Percentile intervals from a deterministic cluster bootstrap."""
    point = binary_metrics(frame[truth].to_numpy(), frame[candidate].to_numpy(), frame[action].to_numpy())
    groups = {key: value.index.to_numpy() for key, value in frame.groupby(cluster, sort=False)}
    keys = np.asarray(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in point if name != "n"}
    for _ in range(repetitions):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        indices = np.concatenate([groups[key] for key in sampled])
        values = binary_metrics(
            frame.loc[indices, truth].to_numpy(),
            frame.loc[indices, candidate].to_numpy(),
            frame.loc[indices, action].to_numpy(),
        )
        for name in draws:
            draws[name].append(values[name])
    rows = []
    for name, values in draws.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        lower, upper = (np.quantile(finite, [0.025, 0.975]) if len(finite) else (math.nan, math.nan))
        rows.append({"metric": name, "estimate": point[name], "lower": lower, "upper": upper})
    return pd.DataFrame(rows)


def matched_random_gate(
    candidate: np.ndarray, retention: float, repetitions: int, seed: int
) -> np.ndarray:
    """Return truth-blind random gates with exactly matched candidate counts."""
    candidate = np.asarray(candidate, dtype=bool)
    rng = np.random.default_rng(seed)
    candidate_indices = np.flatnonzero(candidate)
    keep = int(round(float(retention) * len(candidate_indices)))
    actions = np.zeros((repetitions, len(candidate)), dtype=bool)
    for repetition in range(repetitions):
        selected = rng.choice(candidate_indices, size=keep, replace=False)
        actions[repetition, selected] = True
    return actions


def stratified_metrics(
    frame: pd.DataFrame, stratum: str, minimum_candidates: int = 20
) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(stratum, dropna=False):
        if int(group["candidate"].sum()) < minimum_candidates:
            continue
        metrics = binary_metrics(group["truth"], group["candidate"], group["action"])
        rows.append({"stratum_variable": stratum, "stratum": str(value), **metrics})
    return pd.DataFrame(rows)


__all__ = [
    "binary_metrics",
    "cluster_bootstrap_metrics",
    "expected_binary_loss",
    "matched_random_gate",
    "select_youden_threshold",
    "stratified_metrics",
    "threshold_for_action_rate",
]
