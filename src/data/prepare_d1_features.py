"""D1 (retrospective) features: the history of every source, including the added ones and finance.

Under decision D1 (configs/protocol.yaml) everything observed up to year t-1 may be an input for year t, so the added
sources get the same history features as the original ones (lag 1-3, change, 2/3-year means, trend, prior-observation
count), built by the same function as temporal_features.csv: years < t only, missing years stay missing.

Writes data/processed/d1_temporal_features.csv (read by graph v2). The original temporal features are checked to be
unchanged; the expanded file (publication-filtered, the record of the expanded experiment) is not touched.
Run from the repo root:  python3 src/data/prepare_d1_features.py   (or: make features)
"""
from pathlib import Path
import json

import pandas as pd

from build_master import temporal_features
from new_sources import FEATURES, FINANCE_PER_STUDENT
from validate_additions import require_master_gate, sha

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
SOURCES = FEATURES + FINANCE_PER_STUDENT       # staff, AP/IB, STAAR, graduation, finance per student


def prepare() -> pd.DataFrame:
    require_master_gate()
    m = pd.read_csv(P / "master_school_year.csv", dtype={"campus_id": str})
    old = pd.read_csv(P / "temporal_features.csv", dtype={"campus_id": str})
    out = temporal_features(m, SOURCES)
    pd.testing.assert_frame_equal(out[old.columns], old, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    out[old.columns] = old                       # original columns byte-for-byte
    out.to_csv(P / "d1_temporal_features.csv", index=False)
    (P / "reports" / "d1_feature_receipt.json").write_text(json.dumps(
        {"master_sha256": sha(P / "master_school_year.csv"), "rows": len(out), "columns": len(out.columns),
         "added_sources": SOURCES}, indent=2) + "\n")
    print(f"Wrote {P / 'd1_temporal_features.csv'}: {out.shape}; history for {len(SOURCES)} added variables")
    return out


if __name__ == "__main__":
    prepare()
