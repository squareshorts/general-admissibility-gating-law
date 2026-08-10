# Reproducibility guide

## Figure-only reproduction

The manuscript-facing figures can be reconstructed without redistributing the biomedical source datasets. Frozen empirical summaries are stored in `results/empirical/`, while the simulation inputs are regenerated deterministically from the code and fixed seed.

Run:

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_figures.py
```

This executes `scripts/run_simulations.py` followed by `scripts/make_figures.py` and writes canonical vector PDFs to `figures/`.

## Numerical checks

```bash
python -m unittest discover -s tests -v
python scripts/check_theory.py
python scripts/audit_empirical_results.py
```

In a repository clone without processed PhysioNet data, the data-reconstruction tests are skipped. The remaining tests and all checks based on frozen machine-readable outputs remain executable.

## Full empirical reconstruction

Install the additional waveform dependency:

```bash
python -m pip install -r requirements-empirical.txt
```

Obtain the datasets according to `data/README.md`, then run the acquisition and empirical scripts as documented there. The full reconstruction path is intended to validate the locked empirical package, not to retune the detector or gate.

## Frozen-policy rule

The following are considered part of the evidence freeze and must not be changed when reproducing the reported results:

- outcome-independent Challenge split;
- PTB-XL official train/validation/test folds;
- detector operating thresholds;
- hard-gate definitions;
- random-control seeds and acceptance matching;
- bootstrap design;
- prespecified expected-loss grid;
- simulation seed `20260809`.

## Data handling

Raw and processed PhysioNet records are intentionally excluded from the repository. Only provenance instructions and frozen aggregate outputs are distributed.
