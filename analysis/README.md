# Analysis

This directory will contain the executable analysis code used for the manuscript.

Planned organization:

- `simulations/` — analytical sweeps, Monte Carlo verification, evidence-structure simulations, and finite-sample experiments.
- `challenge2015/` — locked PhysioNet/Computing in Cardiology Challenge 2015 gate analysis.
- `ptbxl/` — locked PTB-XL detector, gate, and policy-comparison analysis.

The held-out analyses must not be retuned when these files are imported into the public repository. Any cleanup should be limited to packaging, paths, documentation, and reproducibility without changing locked numerical outputs.
