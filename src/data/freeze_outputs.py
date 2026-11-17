"""Freeze existing outputs before an additive rebuild; never overwrite a snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "results/frozen/pre_expansion"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"Snapshot already exists; refusing to overwrite: {out}")
    files = []
    for rel in ("data/processed", "data/graph", "configs", "src"):
        files.extend(p for p in (ROOT / rel).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith("."))
    files.extend(p for p in (ROOT / "results").rglob("*") if p.is_file()
                 and not {"frozen", "expanded"}.intersection(p.relative_to(ROOT / "results").parts)
                 and not p.name.startswith("."))
    files.extend(ROOT.glob("*.md"))
    out.mkdir(parents=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "files": {}}
    for src in sorted(set(files)):
        rel = src.relative_to(ROOT)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        sha = digest(src)
        if digest(dst) != sha:
            raise RuntimeError(f"Snapshot verification failed: {rel}")
        manifest["files"][str(rel)] = {"sha256": sha, "bytes": src.stat().st_size}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified snapshot: {out} ({len(files)} files)")


if __name__ == "__main__":
    main()
