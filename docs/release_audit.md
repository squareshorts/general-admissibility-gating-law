# Release audit

Audit performed on 2026-08-10 before repository population.

## Source-package completeness

The uploaded frozen package and source patch jointly contained the analytical modules, deterministic simulation pipeline, empirical acquisition/evaluation/reporting code, tests, theory notes, frozen empirical outputs, simulation outputs, and evidence documentation.

## Verification performed

- `python scripts/check_theory.py`: passed.
- `python scripts/audit_empirical_results.py`: 39 numerical checks, 28 artifact/source checks, no failures.
- `python -m unittest discover -s tests -v` in the public no-data layout: 34 discovered, 31 passed, 3 skipped because processed PhysioNet data were intentionally absent.
- A clean rerun of `scripts/run_simulations.py` completed successfully with seed `20260809`.
- Regenerated simulation CSVs were numerically identical to the frozen outputs to absolute tolerance `1e-12`; observed differences were serialization-level only.
- Same-score mismatch count: 0.
- `scripts/make_figures.py` successfully regenerated all canonical vector PDFs and PNG previews from the reconstructed simulations plus frozen empirical outputs.

## Packaging changes that do not alter scientific results

Two release-hygiene changes were made after the evidence freeze:

1. `wfdb` is imported lazily inside waveform-reading functions so theory/simulation and frozen-output tests do not require the waveform dependency.
2. The candidate-count reconstruction test is skipped when its required processed datasets are absent, rather than failing on a missing local data file.

The Figure 1 layout code was revised for typesetting clarity only; no analytical or empirical quantity was changed.
