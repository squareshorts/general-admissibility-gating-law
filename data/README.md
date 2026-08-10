# Dataset provenance

The source biomedical datasets are not redistributed in this repository. They remain available from PhysioNet under their original terms.

## PhysioNet/Computing in Cardiology Challenge 2015

- Version: 1.0.0
- Source: PhysioNet/Computing in Cardiology Challenge 2015
- Public material used: training records and public labels only
- Hidden Challenge test labels: not accessed
- Development acquisition date: 2026-08-09
- Published-manifest SHA-256 recorded at evidence freeze: `7432ea83a178a467d087746d6c1be4b9ea94ea374f30de1079d40d9bd5da0599`

## PTB-XL

- Version: 1.0.3
- Source: PhysioNet PTB-XL
- Public material used: metadata and 100 Hz waveform records
- Development acquisition date: 2026-08-09
- Published-manifest SHA-256 recorded at evidence freeze: `b7224b92b341511ec3ceb13dc6652079b2c36a06504bcb49506f157f51dc695d`

## Public archive policy

The release is designed to reproduce manuscript figures from deterministic simulations plus frozen aggregate empirical outputs. Raw ECG/monitor records and processed waveform arrays are intentionally excluded. `scripts/acquire_empirical_data.py` documents the public acquisition locations and checksum-validation procedure used during development, but downloading the datasets is not required for figure reproduction.
