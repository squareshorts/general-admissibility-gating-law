# General Admissibility Gating Law

Reproducible analysis package for candidate-conditioned evaluation of post-positive gating in biomedical decision systems.

## Overview

This repository supports the manuscript *Candidate-Conditioned Evaluation of Post-Positive Gating in Biomedical Decision Systems: Predictive Value, Utility, and Held-Out Validation*.

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

- `analysis/` — analysis entry points and implementation notes.
- `config/` — frozen analysis settings and configuration records.
- `results/` — machine-readable simulation and empirical outputs.
- `figures/` — generated manuscript figures and figure-generation notes.
- `tables/` — generated manuscript tables and table manifests.
- `tests/` — numerical and consistency checks.
- `reproducibility/` — evidence-freeze and audit records.
- `manuscript/` — manuscript-facing materials.
- `docs/` — project and reproducibility documentation.

## Reproducibility status

The manuscript reports a frozen evidence package with deterministic stochastic seeds, locked empirical definitions, numerical consistency checks, and artifact/source checks. Repository files will be added in a way that preserves the frozen results rather than rerunning or retuning the held-out analyses.

## Data availability

The source datasets are not redistributed here. They are publicly available from PhysioNet under their respective terms. Scripts and configuration files in this repository are intended to operate on locally obtained copies of those datasets.

## Software environment

Python dependencies are listed in `requirements.txt`. Exact versions used for the archived manuscript release should be frozen before tagging the release.

## Citation

Citation metadata are provided in `CITATION.cff`. A DOI will be added after the manuscript release is archived in a permanent repository such as Zenodo.

## License

Code is released under the BSD 3-Clause License. Dataset licenses and terms remain those of the original data providers.
