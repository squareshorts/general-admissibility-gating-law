"""Acquire the two approved public empirical datasets.

Only the public 2015 Challenge training records and the 100 Hz PTB-XL
waveforms are downloaded.  Downloads are resumable at the file level and are
accepted only when their published SHA-256 digest matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "admissibility-gating-law-research/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, expected: str | None = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected is None or sha256(target) == expected):
        return target

    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as source, partial.open("wb") as sink:
                while block := source.read(1024 * 1024):
                    sink.write(block)
            if expected is not None and sha256(partial) != expected:
                raise OSError(f"checksum mismatch for {target.name}")
            os.replace(partial, target)
            return target
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            if attempt == 4:
                raise RuntimeError(f"failed to download {url}") from exc
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def parse_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        entries[name.strip().lstrip("*")] = digest
    return entries


def write_receipt(target: Path, payload: dict[str, object]) -> None:
    (target / "LOCAL_ACQUISITION.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def acquire_challenge(workers: int) -> None:
    target = ROOT / "data" / "raw" / "physionet_challenge_2015"
    file_base = "https://physionet.org/files/challenge-2015/1.0.0"
    training_base = f"{file_base}/training"
    for name in ("LICENSE.txt",):
        download(f"{file_base}/{name}", target / name)
    download(f"{training_base}/SHA256SUMS", target / "SHA256SUMS")
    manifest = parse_manifest(target / "SHA256SUMS")
    wanted = {
        name: digest
        for name, digest in manifest.items()
        if name in {"ALARMS", "RECORDS"} or name.endswith((".hea", ".mat"))
    }

    def task(item: tuple[str, str]) -> Path:
        name, digest = item
        return download(f"{training_base}/{name}", target / "training" / name, digest)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(task, item) for item in sorted(wanted.items())]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 100 == 0 or count == len(futures):
                print(f"challenge: {count}/{len(futures)} files verified", flush=True)
    write_receipt(
        target,
        {
            "accessed_utc_date": "2026-08-09",
            "dataset": "PhysioNet/CinC Challenge 2015",
            "version": "1.0.0",
            "official_source": "https://physionet.org/content/challenge-2015/1.0.0/",
            "downloaded_subset": "public training records and labels only",
            "published_manifest_sha256": sha256(target / "SHA256SUMS"),
            "license_sha256": sha256(target / "LICENSE.txt"),
            "verified_manifest_entries": len(wanted),
        },
    )


def acquire_ptbxl(workers: int) -> None:
    target = ROOT / "data" / "raw" / "ptb_xl_1.0.3"
    # The official PhysioNet public S3 mirror is substantially faster for the
    # many small PTB-XL files than the interactive file server.
    file_base = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3"
    metadata = (
        "LICENSE.txt",
        "RECORDS",
        "SHA256SUMS.txt",
        "ptbxl_database.csv",
        "scp_statements.csv",
        "ptbxl_v102_changelog.txt",
        "ptbxl_v103_changelog.txt",
    )
    for name in metadata:
        download(f"{file_base}/{name}", target / name)
    manifest = parse_manifest(target / "SHA256SUMS.txt")
    # v1.0.3 RECORDS omits the line break between its final records100 entry
    # and first records500 entry; normalize that published formatting defect.
    record_text = (target / "RECORDS").read_text(encoding="utf-8").replace(
        "records500/", "\nrecords500/"
    )
    record_bases = [
        line.strip()
        for line in record_text.splitlines()
        if line.startswith("records100/")
    ]
    wanted = [f"{base}{suffix}" for base in record_bases for suffix in (".hea", ".dat")]

    def task(name: str) -> Path:
        return download(f"{file_base}/{name}", target / name, manifest[name])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(task, name) for name in wanted]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 1000 == 0 or count == len(futures):
                print(f"ptb-xl: {count}/{len(futures)} files verified", flush=True)
    write_receipt(
        target,
        {
            "accessed_utc_date": "2026-08-09",
            "dataset": "PTB-XL",
            "version": "1.0.3",
            "official_source": "https://physionet.org/content/ptb-xl/1.0.3/",
            "downloaded_subset": "100 Hz waveforms and metadata",
            "published_manifest_sha256": sha256(target / "SHA256SUMS.txt"),
            "license_sha256": sha256(target / "LICENSE.txt"),
            "verified_waveform_files": len(wanted),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("challenge", "ptbxl", "all"))
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if args.dataset in {"challenge", "all"}:
        acquire_challenge(args.workers)
    if args.dataset in {"ptbxl", "all"}:
        acquire_ptbxl(args.workers)


if __name__ == "__main__":
    main()
