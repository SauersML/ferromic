import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phewas import categories
from phewas import models


def test_plan_category_sets_respects_min_k_and_dedup(tmp_path):
    phenos = ["A", "B", "C", "D"]
    name_to_cat = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
    manifest = {"kept": ["A", "B", "C"]}

    core_idx = pd.Index(["p1", "p2", "p3"], name="person_id")
    cache_dir = tmp_path.as_posix()
    fingerprint = models._index_fingerprint(core_idx)
    manifest_path = tmp_path / f"pheno_dedup_manifest_TEST_{fingerprint}.json"
    manifest_path.write_text(json.dumps(manifest))

    loaded = categories.load_dedup_manifest(cache_dir, "TEST", core_idx)
    kept, dropped = categories.plan_category_sets(phenos, name_to_cat, loaded, min_k=2)

    assert "X" in kept
    assert kept["X"] == ["A", "B"]
    assert "Y" in dropped and dropped["Y"] == ["C"]
    assert "Y" not in kept
    assert all(p in {"A", "B", "C"} for plist in kept.values() for p in plist)
