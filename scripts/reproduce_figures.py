"""Reproduce the eight manuscript-facing figures and verify their existence."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)


def main() -> None:
    required = ROOT / "results" / "empirical" / "main_metrics.csv"
    if not required.exists():
        raise FileNotFoundError(
            "Frozen empirical outputs are missing: results/empirical/main_metrics.csv"
        )

    run("run_simulations.py")
    run("make_figures.py")

    stems = (
        "figure1_decision_architecture",
        "figure2_ppv_boundary",
        "figure3_expected_loss_phase",
        "figure4_evidence_policy_comparison",
        "figure5_finite_sample_uncertainty",
        "figure6_empirical_delta_q",
        "figure7_empirical_policy_comparison",
        "figure8_operational_heterogeneity",
    )
    missing = [stem for stem in stems if not (ROOT / "figures" / f"{stem}.pdf").exists()]
    if missing:
        raise RuntimeError(f"Missing reproduced figures: {missing}")

    print("Reproduction complete: eight canonical PDFs written to figures/.")


if __name__ == "__main__":
    main()
