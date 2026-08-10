# General Admissibility Gating Law

Reproducible analysis package for candidate-conditioned evaluation of post-positive gating in biomedical decision systems.

## Scope

This repository is intentionally limited to the material needed to reproduce the reported numerical results, figures, tables, and consistency checks. The manuscript LaTeX source is maintained separately and is not part of the reproducibility archive.

The project studies a post-positive decision architecture in which a fixed detector produces candidate-positive events and a downstream gate determines whether those candidates are acted on or withheld. The central candidate-conditioned quantities are

- `q1 = Pr(Q=1 | P=1, T=1)`: retention of true positive candidates;
- `q0 = Pr(Q=1 | P=1, T=0)`: retention of false positive candidates;
- `Delta_Q = q1 - q0`: candidate-conditioned gate discrimination.

For a fixed detector, positive predictive value improves if and only if `q1 > q0`. Expected loss follows a separate criterion that also depends on prevalence, detector performance, action costs, and the consequences of withholding.

## Empirical systems

The locked empirical analyses use two public biomedical datasets:

1. PhysioNet/Computing in Cardiology Challenge 2015.
2. PTB-XL v1.0.3.

Hidden Challenge test labels were not used.

## Repository layout

- `analysis/` — deterministic analysis and figure-generation code.
- `config/` — frozen thresholds, partitions, seeds, and cost definitions.
- `results/` — machine-readable frozen simulation and empirical outputs.
- `figures/` — reproducible figure outputs and figure manifest.
- `tables/` — reproducible table outputs and table manifest.
- `tests/` — numerical reconstruction and consistency checks.
- `reproducibility/` — evidence-freeze record and audit outputs.
- `docs/` — reproducibility and data-acquisition instructions.

## Reproducibility policy

The archived release must reproduce the manuscript-facing results without retuning any empirical gate, detector threshold, split, cost scenario, or reported numerical result. Figures and tables are regenerated only from frozen machine-readable outputs or from deterministic scripts operating under the frozen configuration.

The evidence freeze associated with the manuscript reports 34/34 passing tests, 39 numerical reconstruction checks, and 28 artifact/source checks at tolerance `1e-12`.

## Data availability

The source datasets are not redistributed here. They are publicly available from PhysioNet under their respective terms. Scripts and configuration files operate on locally obtained copies of those datasets.

## Software environment

Python dependencies are listed in `requirements.txt`. Exact package versions must be pinned in the manuscript release before tagging `v1.0.0`.

## Citation

Citation metadata are provided in `CITATION.cff`. The DOI for the archived `v1.0.0` release will be added after deposition in Zenodo.

## License

Code is released under the BSD 3-Clause License. Dataset licenses and terms remain those of the original data providers.
