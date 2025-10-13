import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phewas import categories
from phewas import pheno


def test_covariance_and_metrics_basic(monkeypatch):
    core = pd.DataFrame(index=pd.Index([str(i) for i in range(10)], name="person_id"))

    def fake_cases(name, cdr, cache):
        return {
            "ph1": ["0", "1", "2"],
            "ph2": ["0", "1"],
            "ph3": ["7", "8"],
        }.get(name, [])

    monkeypatch.setattr(pheno, "_case_ids_cached", fake_cases)

    cat_sets = {"Cat": ["ph1", "ph2", "ph3"]}
    allowed = {"Cat": np.ones(core.shape[0], dtype=bool)}

    nulls = categories.build_category_null_structure(
        core,
        allowed,
        cat_sets,
        cache_dir=".",
        cdr_codename="TEST",
        method="fast_phi",
        shrinkage="ridge",
        lambda_value=0.05,
        min_k=2,
        global_mask=None,
    )

    assert "Cat" in nulls
    struct = nulls["Cat"]
    assert struct.covariance.shape == (3, 3)

    inv = pd.DataFrame(
        {
            "Phenotype": ["ph1", "ph2", "ph3"],
            "P_EMP": [1e-5, 2e-4, 0.03],
            "Beta": [0.2, 0.1, -0.05],
        }
    )

    out = categories.compute_category_metrics(
        inv,
        p_col="P_EMP",
        beta_col="Beta",
        null_structures=nulls,
        gbj_draws=200,
        z_cap=6.0,
        rng_seed=42,
        min_k=2,
    )

    assert out.shape[0] == 1
    row = out.loc[0]
    assert row["Category"] == "Cat"
    assert np.isfinite(row["P_GBJ"])
    assert np.isfinite(row["P_GLS"])
