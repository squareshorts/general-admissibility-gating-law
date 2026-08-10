"""Finite-sample reliability calculations for binary gated decisions."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from src.theory.gating import Losses, Parameters, analytical_metrics


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    phat = successes/total
    den = 1+z*z/total
    center = (phat+z*z/(2*total))/den
    radius = z*math.sqrt(phat*(1-phat)/total+z*z/(4*total*total))/den
    return max(0.0, center-radius), min(1.0, center+radius)


def run_finite_sample(
    p: Parameters, losses: Losses, *, n: int, repeats: int, seed: int,
) -> pd.DataFrame:
    probs = np.array([
        p.prevalence*p.sensitivity*p.retain_true_candidate,
        p.prevalence*p.sensitivity*(1-p.retain_true_candidate),
        p.prevalence*(1-p.sensitivity),
        (1-p.prevalence)*p.false_positive_rate*p.retain_false_candidate,
        (1-p.prevalence)*p.false_positive_rate*(1-p.retain_false_candidate),
        (1-p.prevalence)*(1-p.false_positive_rate),
    ])
    probs /= probs.sum()
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, probs, size=repeats)
    exact_delta = analytical_metrics(p, losses)["delta_loss"]
    rows = []
    for rep, (tr, tw, fn, fr, fw, tn) in enumerate(counts):
        tc, fc = tr+tw, fr+fw
        q1 = tr/tc if tc else math.nan
        q0 = fr/fc if fc else math.nan
        ppv_u = (tr+tw)/(tr+tw+fr+fw) if tr+tw+fr+fw else math.nan
        ppv_g = tr/(tr+fr) if tr+fr else math.nan
        dl_samples = np.array([
            losses.withheld_true-losses.true_positive,
            losses.withheld_false-losses.false_positive,
        ])
        dl = (tw*dl_samples[0]+fw*dl_samples[1])/n
        second = (tw*dl_samples[0]**2+fw*dl_samples[1]**2)/n
        se = math.sqrt(max(0, second-dl*dl)/n)
        q1_low, q1_high = wilson_interval(int(tr), int(tc))
        q0_low, q0_high = wilson_interval(int(fr), int(fc))
        wrong = (dl < 0) != (exact_delta < 0) if dl != 0 else exact_delta != 0
        rows.append({
            "replicate": rep, "n": n, "prevalence": p.prevalence, "seed": seed,
            "true_candidates": tc, "false_candidates": fc, "q1_hat": q1,
            "q1_ci_low": q1_low, "q1_ci_high": q1_high, "q0_hat": q0,
            "q0_ci_low": q0_low, "q0_ci_high": q0_high,
            "delta_q_hat": q1-q0, "ppv_change_hat": ppv_g-ppv_u,
            "delta_loss_hat": dl, "delta_loss_ci_low": dl-1.96*se,
            "delta_loss_ci_high": dl+1.96*se, "exact_delta_loss": exact_delta,
            "wrong_loss_sign": wrong, "observed_q0_zero": q0 == 0 if not math.isnan(q0) else False,
        })
    return pd.DataFrame(rows)
