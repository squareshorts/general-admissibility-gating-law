"""Read-only audit of the locked aggregate empirical record."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EMP = ROOT / "results" / "empirical"
TOL = 1e-12


def close(observed: float, expected: float, label: str, failures: list[str]) -> None:
    if abs(float(observed) - float(expected)) > TOL:
        failures.append(f"{label}: observed={observed}, expected={expected}")


def one(frame: pd.DataFrame, **where: str) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in where.items():
        mask &= frame[column].eq(value)
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise AssertionError(f"Expected one row for {where}; found {len(selected)}")
    return selected.iloc[0]


def main() -> None:
    counts = pd.read_csv(EMP / "candidate_counts.csv")
    metrics = pd.read_csv(EMP / "main_metrics.csv")
    controls = pd.read_csv(EMP / "strategy_controls.csv")
    costs = pd.read_csv(EMP / "cost_grid.csv")
    failures: list[str] = []
    checks = 0

    expected = {
        "Challenge 2015": (153, 54, 99, 49, 96),
        "PTB-XL": (887, 392, 495, 300, 386),
    }
    for dataset, (candidate_n, true_n, false_n, kept_true, kept_false) in expected.items():
        r = one(counts, dataset=dataset)
        for column, value in (
            ("candidate_count", candidate_n),
            ("true_candidates", true_n),
            ("false_candidates", false_n),
            ("accepted_true_candidates", kept_true),
            ("accepted_false_candidates", kept_false),
        ):
            close(r[column], value, f"{dataset}:{column}", failures); checks += 1

        gate = one(metrics, dataset=dataset, strategy="Hard gate")
        q1, q0 = kept_true / true_n, kept_false / false_n
        close(gate.q1, q1, f"{dataset}:q1", failures); checks += 1
        close(gate.q0, q0, f"{dataset}:q0", failures); checks += 1
        close(gate.delta_q, q1 - q0, f"{dataset}:delta_q", failures); checks += 1
        close(gate.action_ppv, kept_true / (kept_true + kept_false), f"{dataset}:PPV", failures); checks += 1

    expected_controls = {
        "retuned_s_threshold": (0.5037707390648567, 0.6072727272727273, 0.1996359223300971),
        "same_score_confidence_gate": (0.5037707390648567, 0.6072727272727273, 0.1996359223300971),
        "joint_s_q_model": (0.4354485776805251, 0.7236363636363636, 0.3131067961165048),
    }
    for strategy, values in expected_controls.items():
        r = one(controls, dataset="PTB-XL", strategy=strategy)
        for column, value in zip(("action_ppv", "operational_sensitivity", "false_action_rate"), values):
            close(r[column], value, f"PTB-XL:{strategy}:{column}", failures); checks += 1

    winners = costs.loc[costs.winner].groupby(["dataset", "strategy"]).size().to_dict()
    for key, value in {
        ("PTB-XL", "cost_tuned_joint"): 27,
        ("PTB-XL", "cost_tuned_s"): 16,
        ("PTB-XL", "hard_gate"): 5,
    }.items():
        close(winners.get(key, 0), value, f"cost-grid:{key}", failures); checks += 1

    frozen = EMP / "consistency_audit.json"
    if not frozen.exists():
        failures.append("missing frozen consistency_audit.json")

    print(f"Aggregate audit checks: {checks}")
    print(f"Failures: {len(failures)}")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print("Locked aggregate empirical record passed.")


if __name__ == "__main__":
    main()
