# Evidence freeze

## Freeze record

- **Freeze date:** 2026-08-09.
- **Repository:** `C:\work\general_admissibility_gating_law`.
- **Git status:** this directory is not a Git repository; no commit hash is
  available and no commit was created.
- **Final test status:** 34 run, 34 passed, 0 failed, 0 errors, 0 skipped.
- **Consistency audit:** 39/39 numerical reconstructions and 28/28 artifact
  checks passed at tolerance `1e-12`.
- **Parameter status:** targets, detector definitions, gates, thresholds,
  partitions, preprocessing, and cost scenarios were locked before held-out
  evaluation. No post-hoc retuning was performed.

## Frozen empirical systems

### Challenge 2015 v1.0.0

- Public records: 750.
- Held-out partition: deterministic SHA-256 record-name test split, `n=153`.
- Candidate architecture: every record is an issued monitor alarm.
- Gate: prespecified final-ten-second pulsatile-channel validity rule.
- Candidate counts: 54 true, 99 false.
- Retained: 49 true, 96 false.
- `q1=0.9074`, `q0=0.9697`, `Delta_Q=-0.0623`, 95% interval
  [-0.1525, 0.0206].
- PPV: 0.3529 to 0.3379.
- Operational sensitivity: 1.0000 to 0.9074.
- False candidates removed: 3.03%; withholding: 5.23%.
- Cost result: under missed-event withholding, the hard gate beats the fixed
  alarm policy only at approximately `C_FP/C_FN>=2`; break-even acquisition
  costs are 0.0065, 0.0458, and 0.1242 loss units per candidate at ratios 2,
  4, and 8.

### PTB-XL v1.0.3

- Records: 21,799.
- Held-out partition: official patient-respecting fold 10, `n=2,198`; folds
  1-8 trained the detector and fold 9 selected thresholds.
- Target: myocardial-infarction diagnostic superclass.
- Detector: ridge logistic waveform model; locked candidate threshold
  0.2208007272.
- Gate: all four technical artifact fields clear.
- Candidates: 887; 392 true and 495 false.
- Retained: 300 true and 386 false.
- `q1=0.7653`, `q0=0.7798`, `Delta_Q=-0.0145`, 95% interval
  [-0.0720, 0.0384].
- PPV: 0.4419 to 0.4373.
- Operational sensitivity: 0.7127 to 0.5455.
- False candidates removed: 22.02%; withholding: 9.14%.
- Acceptance-matched retuned `S`: PPV 0.5038, sensitivity 0.6073,
  false-action rate 0.1996; it dominated the hard gate.
- Cost-grid winners: joint model 27/48, tuned `S` 16/48, hard gate 5/48.

## Frozen artifacts

Figures:

1. Figure E1 — gate discrimination (`q1`, `q0`, bootstrap intervals).
2. Figure E2 — PPV versus operational sensitivity.
3. Figure E3 — expected-loss cost map; displayed convention
   `W1/C_FN=1`, `W0/C_FP=0`.
4. Figure E4 — empirical strategy comparison.
5. Figure E5 — alarm-type, device, and site transport audit.

Tables:

1. Table 5 — dataset characteristics and locked definitions.
2. Table 6 — empirical gate properties and uncertainty.
3. Table 7 — strategy controls and cost-grid dominance.

Machine-readable records:

- `results/empirical/figure_manifest.csv`
- `results/empirical/table_manifest.csv`
- `results/empirical/run_metadata.json`
- `results/empirical/consistency_audit.json`
- all result CSVs listed in `results/empirical/README.md`

## Frozen scientific interpretation

Neither gate achieved `q1>q0`, and both reduced held-out PPV. The empirical
positive-gate hypotheses are therefore contradicted for the locked definitions.
The mathematical PPV and expected-loss criteria are unchanged. The evidence
supports the distinction between quality and conditional admissibility, the
separation of PPV discrimination from utility, the equivalence of monotone
same-score gating and threshold adjustment, the need for comparator policies,
and the possibility of transport heterogeneity.

The project does not support a domain-general “Admissibility Gating Law.”

## Known limitations

- Challenge 2015 exposes only alarm candidates, no original detector score,
  and no patient identifier for grouped resampling.
- PTB-XL technical-quality annotations are retrospective expert labels; a
  prospective gate would require separately validated acquisition and cost.
- `S` and `Q` share an ECG acquisition in PTB-XL.
- Transport analyses use internal strata, not external cohorts.
- Both retained systems are medical; no credible third domain was found.
- Costs are standardized grids rather than validated monetary utilities.

## Readiness and manuscript constraint

- **Original positive-validation manuscript:** **NOT READY / NOT SUPPORTED**.
  The empirical premise of generally useful real-world gates failed.
- **Theory plus empirical falsification manuscript:** **READY WITH
  LIMITATIONS**. Theory, simulation, two locked held-out falsifications,
  comparator coverage, cost analysis, transport analysis, provenance,
  machine-readable artifacts, consistency checks, and tests are complete. The
  limitations above must be prominent and the work must not be presented as
  external or cross-industry replication.

Recommended working title:

> **Admissibility Gating in Positive-Event Decision Systems: Theory, Boundary
> Conditions, and Empirical Falsification**

Recommended terminology: **operational gating framework**, with
**admissibility gating criterion** for the `q1/q0` and expected-loss conditions.
Do not use “law” as publication terminology.

Smallest defensible central claim:

> Admissibility is a conditional decision property, not an intrinsic property
> of a quality measure: a positive-candidate gate improves precision only when
> it preferentially retains true candidates, while utility additionally
> depends on prevalence, action costs, withholding consequences, and acquisition
> cost; plausible quality indicators may fail these conditions and be dominated
> by simpler policies.

Subsequent manuscript drafting must use this frozen package without altering
the locked empirical analyses. Any later exploratory work must be labeled as a
new analysis and must not replace these held-out results.
