"""Torch-free digest of a year-sorted Graph V2 CSV through a cutoff year.

Identical to build_s5_validation_snapshot.read_sorted_year_csv's digest (header, then each row up to the cutoff), so
it can be compared with the digests stored in the validation bundle's metadata. Kept separate because the tree part
of S6 must not import torch (torch and scikit-learn's tree load conflicting OpenMP runtimes on macOS).
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def digest_through(path: Path, cutoff: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        year_col = header.index("year")
        digest.update(json.dumps(header, ensure_ascii=False).encode("utf-8"))
        for row in reader:
            if int(row[year_col]) > cutoff:
                break
            digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
