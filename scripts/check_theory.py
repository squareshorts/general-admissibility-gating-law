"""Deterministic numerical and Monte Carlo checks of the theory specification."""

from __future__ import annotations

import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.theory.gating import Losses, Parameters, analytical_metrics, simulate


SCENARIOS = {
    "beneficial_sparse": (
        Parameters(0.02, 0.85, 0.10, 0.92, 0.25),
        Losses(20.0, 5.0, 6.0, 1.0),
    ),
    "harmful_costly_withholding": (
        Parameters(0.50, 0.50, 0.50, 0.90, 0.80),
        Losses(1.0, 2.0, 10.0, 10.0),
    ),
    "beneficial_despite_negative_delta_q": (
        Parameters(0.001, 0.50, 0.50, 0.20, 0.30),
        Losses(100.0, 1.0, 1.0, 0.0),
    ),
    "truth_independent_gate": (
        Parameters(0.10, 0.80, 0.15, 0.40, 0.40),
        Losses(8.0, 3.0, 2.0, 0.5),
    ),
}


def main() -> None:
    observations = 600_000
    print(f"Monte Carlo observations per scenario: {observations:,}")
    print("seed: 20260809")

    for index, (name, (parameters, losses)) in enumerate(SCENARIOS.items()):
        exact = analytical_metrics(parameters, losses)
        sampled = simulate(
            parameters,
            losses,
            observations=observations,
            seed=20260809 + index,
        )
        identity_error = abs(exact["delta_loss"] - exact["delta_loss_identity"])
        mc_error = abs(sampled["delta_loss"] - exact["delta_loss"])
        p_true_withheld = (
            parameters.prevalence
            * parameters.sensitivity
            * (1.0 - parameters.retain_true_candidate)
        )
        p_false_withheld = (
            (1.0 - parameters.prevalence)
            * parameters.false_positive_rate
            * (1.0 - parameters.retain_false_candidate)
        )
        delta_true = losses.withheld_true - losses.true_positive
        delta_false = losses.withheld_false - losses.false_positive
        second_moment = (
            p_true_withheld * delta_true**2
            + p_false_withheld * delta_false**2
        )
        standard_error = math.sqrt(
            max(0.0, second_moment - exact["delta_loss"] ** 2) / observations
        )
        tolerance = 6.0 * standard_error + 1e-6
        if identity_error > 1e-12 or mc_error > tolerance:
            raise AssertionError(
                f"{name}: identity_error={identity_error}, mc_error={mc_error}, "
                f"six_sigma_tolerance={tolerance}"
            )
        print(
            f"{name:38s} "
            f"Delta_Q={parameters.retain_true_candidate - parameters.retain_false_candidate:+.3f} "
            f"PPV_U={exact['ppv_ungated']:.4f} "
            f"PPV_G={exact['ppv_gated']:.4f} "
            f"Delta_L exact={exact['delta_loss']:+.4f} "
            f"MC={sampled['delta_loss']:+.4f}"
        )

    independent = analytical_metrics(*SCENARIOS["truth_independent_gate"])
    if not math.isclose(
        independent["ppv_ungated"], independent["ppv_gated"], abs_tol=1e-12
    ):
        raise AssertionError("Truth-independent gate changed analytical PPV")
    print("All analytical identities and Monte Carlo checks passed.")


if __name__ == "__main__":
    main()
