# Reproducibility notes

This repository is intended to preserve the exact analysis state associated with the manuscript release.

## Principles

- Held-out empirical definitions, thresholds, partitions, and cost scenarios must remain locked.
- Regeneration of figures and tables must use machine-readable results rather than manual editing of reported values.
- Any refactoring performed after the evidence freeze must preserve numerical outputs.
- Dataset files are not redistributed; users must obtain them from the original providers.

## Determinism

The manuscript reports deterministic seed `20260809` for stochastic analyses where applicable.

## Release procedure

Before creating the manuscript release:

1. Freeze exact dependency versions.
2. Verify all analysis tests and numerical reconstruction checks.
3. Regenerate all manuscript figures and tables from repository outputs.
4. Verify figure/table manifests against the manuscript.
5. Tag the exact release used for submission.
6. Archive that release in Zenodo or another DOI-minting repository.

## Archived outputs

The archived release should include machine-readable simulation and empirical results, frozen configuration files, figure and table manifests, and the outputs of the final reproducibility checks.
