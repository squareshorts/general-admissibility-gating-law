# General Admissibility Gating Law

Reproducible analyses for candidate-conditioned evaluation of post-positive gating in biomedical decision systems.

This repository accompanies the manuscript **Candidate-Conditioned Evaluation of Post-Positive Gating in Biomedical Decision Systems: Predictive Value, Utility, and Held-Out Validation**.

## Scientific scope

The project evaluates a post-positive decision architecture in which a fixed detector produces candidate-positive events and a downstream gate determines which candidates are allowed to trigger action. The core candidate-conditioned quantities are

- `q1 = Pr(Q=1 | P=1, T=1)`, retention of true positive candidates;
- `q0 = Pr(Q=1 | P=1, T=0)`, retention of false positive candidates;
- `Delta_Q = q1 - q0`, the sign criterion for PPV enrichment under the stated nondegenerate conditions.

The repository is intended to contain the frozen analytical checks, simulation code, empirical analyses, machine-readable outputs, figures, tables, and reproducibility audits used in the manuscript. Empirical evaluation uses the PhysioNet/Computing in Cardiology Challenge 2015 and PTB-XL datasets; raw source datasets are not redistributed here.

## Repository status

**Pre-submission reproducibility repository.** The public repository structure and release documentation are initialized, but the frozen analysis package must be synced from the local research project before the manuscript-linked `v1.0.0` release is created. Do not cite the moving `main` branch as the archival analysis version.

## Planned structure

```text
analysis/              analysis scripts and entry points
config/frozen/         locked analysis configuration
results/               machine-readable simulation and empirical outputs
figures/               generated manuscript figures and figure-generation code
tables/                machine-readable and rendered tables
tests/                 numerical and reproducibility tests
reproducibility/       evidence freeze and consistency-audit material
manuscript/            manuscript-associated source snapshots
docs/                  release and reproducibility documentation
```

Each directory contains a README describing what belongs there and what must be frozen before release.

## Data

The raw datasets remain with their original providers:

- PhysioNet/Computing in Cardiology Challenge 2015
- PTB-XL v1.0.3

Hidden Challenge test labels were not used. Dataset-specific licenses and access conditions apply.

## Reproducibility target

The manuscript-linked archival release should contain, at minimum:

1. the exact analysis scripts used for the reported results;
2. frozen configuration files and deterministic seeds;
3. machine-readable simulation and empirical outputs;
4. figure and table generation code plus manifests;
5. numerical consistency and artifact/source audits;
6. dependency/environment specification;
7. a tagged GitHub release (`v1.0.0`) archived in Zenodo with a persistent DOI.

See `docs/release_checklist.md` and `docs/reproducibility.md`.

## License

Code in this repository is released under the BSD 3-Clause License unless otherwise stated. Dataset licenses remain with the original data providers.

## Citation

A `CITATION.cff` file is included for software citation. The manuscript DOI and archival Zenodo DOI should be added once assigned.
