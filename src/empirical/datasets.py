"""Dataset-specific, outcome-blind feature and gate construction."""

from __future__ import annotations

import ast
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate


ROOT = Path(__file__).resolve().parents[2]
QUALITY_FIELDS = ("baseline_drift", "static_noise", "burst_noise", "electrodes_problems")


def deterministic_split(identifier: str) -> str:
    """Outcome-independent 60/20/20 split used where no official split exists."""
    bucket = int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "train" if bucket < 6 else ("validation" if bucket < 8 else "test")


def _pulse_channel_quality(values: np.ndarray, fs: float) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    finite_fraction = float(finite.mean())
    if finite.sum() < max(20, int(0.5 * len(values))):
        return {"finite": finite_fraction, "iqr": 0.0, "flat": 1.0, "spike": 1.0, "periodicity": 0.0}
    x = values.copy()
    x[~finite] = np.nanmedian(x)
    iqr = float(np.quantile(x, 0.75) - np.quantile(x, 0.25))
    scale = max(iqr, float(np.std(x)), 1e-12)
    diff = np.diff(x)
    flat = float(np.mean(np.abs(diff) <= 1e-4 * scale))
    typical_step = max(float(np.median(np.abs(diff))), 1e-8 * scale)
    spike = float(np.mean(np.abs(diff) > 12.0 * typical_step))
    centered = x - np.mean(x)
    denominator = float(centered @ centered)
    if denominator <= 1e-12:
        periodicity = 0.0
    else:
        ac = correlate(centered, centered, mode="full", method="fft")[len(centered) - 1 :]
        low = max(1, int(fs * 60 / 240))
        high = min(len(ac), int(fs * 60 / 30) + 1)
        periodicity = float(np.max(ac[low:high]) / denominator) if high > low else 0.0
    return {"finite": finite_fraction, "iqr": iqr, "flat": flat, "spike": spike, "periodicity": periodicity}


def build_challenge_frame(force: bool = False) -> pd.DataFrame:
    import wfdb

    raw = ROOT / "data" / "raw" / "physionet_challenge_2015" / "training"
    processed = ROOT / "data" / "processed" / "physionet_challenge_2015"
    processed.mkdir(parents=True, exist_ok=True)
    cache = processed / "record_quality.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)
    alarms = pd.read_csv(raw / "ALARMS", names=["record", "alarm_type", "truth"])
    rows = []
    for position, row in alarms.iterrows():
        base = raw / str(row.record)
        header = wfdb.rdheader(str(base))
        start = max(0, header.sig_len - int(10 * header.fs))
        record = wfdb.rdrecord(str(base), sampfrom=start, sampto=header.sig_len)
        pulse_indices = [
            index
            for index, name in enumerate(record.sig_name)
            if any(token in name.upper() for token in ("PLETH", "ABP", "ART", "PAP"))
        ]
        channel_results = [_pulse_channel_quality(record.p_signal[:, index], record.fs) for index in pulse_indices]
        if channel_results:
            best = max(channel_results, key=lambda item: item["periodicity"] - item["flat"] - item["spike"])
            admissible = bool(
                best["finite"] >= 0.95
                and best["iqr"] > 1e-6
                and best["flat"] <= 0.20
                and best["spike"] <= 0.05
                and best["periodicity"] >= 0.15
            )
        else:
            best = {"finite": 0.0, "iqr": 0.0, "flat": 1.0, "spike": 1.0, "periodicity": 0.0}
            admissible = False
        rows.append(
            {
                "record": row.record,
                "alarm_type": row.alarm_type,
                "truth": int(row.truth),
                "split": deterministic_split(str(row.record)),
                "candidate": 1,
                "action": int(admissible),
                "pulse_channel_count": len(pulse_indices),
                **{f"quality_{name}": value for name, value in best.items()},
            }
        )
        if (position + 1) % 100 == 0:
            print(f"challenge features: {position + 1}/{len(alarms)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(cache, index=False)
    return frame


def _waveform_features(path: str) -> np.ndarray:
    import wfdb

    signal = wfdb.rdrecord(path).p_signal.astype(float)
    signal = np.nan_to_num(signal, nan=np.nanmedian(signal, axis=0), posinf=0.0, neginf=0.0)
    centered = signal - signal.mean(axis=0)
    std = centered.std(axis=0)
    p10, p90 = np.quantile(signal, [0.10, 0.90], axis=0)
    mad_diff = np.mean(np.abs(np.diff(signal, axis=0)), axis=0)
    crossings = np.mean(centered[1:] * centered[:-1] < 0, axis=0)
    spectrum = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    frequency = np.fft.rfftfreq(len(centered), d=0.01)
    total = np.maximum(spectrum[(frequency >= 0.5) & (frequency <= 40)].sum(axis=0), 1e-12)
    bands = []
    for low, high in ((0.5, 5), (5, 15), (15, 40)):
        bands.append(spectrum[(frequency >= low) & (frequency < high)].sum(axis=0) / total)
    return np.concatenate([std, p90 - p10, mad_diff, crossings, *bands])


def _mi_truth(code_text: str, statement_table: pd.DataFrame) -> bool:
    codes = ast.literal_eval(code_text)
    return any(
        code in statement_table.index
        and bool(statement_table.loc[code, "diagnostic"])
        and statement_table.loc[code, "diagnostic_class"] == "MI"
        for code in codes
    )


def build_ptbxl_data(force: bool = False, workers: int = 8) -> tuple[pd.DataFrame, np.ndarray]:
    raw = ROOT / "data" / "raw" / "ptb_xl_1.0.3"
    processed = ROOT / "data" / "processed" / "ptb_xl_1.0.3"
    processed.mkdir(parents=True, exist_ok=True)
    frame_cache, feature_cache = processed / "records.csv", processed / "waveform_features.npy"
    if frame_cache.exists() and feature_cache.exists() and not force:
        return pd.read_csv(frame_cache), np.load(feature_cache)
    metadata = pd.read_csv(raw / "ptbxl_database.csv")
    statements = pd.read_csv(raw / "scp_statements.csv", index_col=0)
    metadata["truth"] = [int(_mi_truth(value, statements)) for value in metadata.scp_codes]
    for field in QUALITY_FIELDS:
        metadata[f"artifact_{field}"] = metadata[field].fillna("").astype(str).str.strip().ne("").astype(int)
    artifact_columns = [f"artifact_{field}" for field in QUALITY_FIELDS]
    metadata["artifact_count"] = metadata[artifact_columns].sum(axis=1)
    metadata["admissible"] = metadata.artifact_count.eq(0).astype(int)
    metadata["split"] = np.where(
        metadata.strat_fold <= 8, "train", np.where(metadata.strat_fold == 9, "validation", "test")
    )
    features: list[np.ndarray | None] = [None] * len(metadata)

    def task(index: int) -> tuple[int, np.ndarray]:
        return index, _waveform_features(str(raw / metadata.iloc[index].filename_lr))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(task, index) for index in range(len(metadata))]
        for count, future in enumerate(as_completed(futures), 1):
            index, values = future.result()
            features[index] = values
            if count % 1000 == 0 or count == len(metadata):
                print(f"ptb-xl features: {count}/{len(metadata)}", flush=True)
    matrix = np.vstack(features)
    keep = [
        "ecg_id", "patient_id", "truth", "split", "strat_fold", "sex", "age",
        "site", "device", "artifact_count", "admissible", *artifact_columns,
    ]
    frame = metadata[keep].copy()
    frame.to_csv(frame_cache, index=False)
    np.save(feature_cache, matrix)
    return frame, matrix


__all__ = ["QUALITY_FIELDS", "build_challenge_frame", "build_ptbxl_data", "deterministic_split"]
