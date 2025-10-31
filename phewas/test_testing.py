import json
import pandas as pd

from . import models, testing


def test_consolidate_uses_payload_phenotype(tmp_path):
    inv = "INV1"
    phenotype = "Case+Control"
    safe_inv = models.safe_basename(inv)
    safe_pheno = models.safe_basename(phenotype)

    cache_dir = tmp_path / safe_inv / "lrt_overall"
    cache_dir.mkdir(parents=True)

    payload = {
        "Phenotype": phenotype,
        "P_LRT_Overall": 0.05,
        "P_Value": 0.05,
        "P_Overall_Valid": True,
    }

    payload_path = cache_dir / f"{safe_pheno}.json"
    payload_path.write_text(json.dumps(payload))

    meta_path = cache_dir / f"{safe_pheno}.meta.json"
    meta_path.write_text(json.dumps({"target": inv}))

    df = pd.DataFrame({"Phenotype": [phenotype], "Inversion": [inv]})

    result, _ = testing.consolidate_and_select(
        df.copy(),
        inversions=[inv],
        cache_root=str(tmp_path),
    )

    assert pd.notna(result.loc[0, "P_LRT_Overall"])
    assert bool(result.loc[0, "P_Overall_Valid"])
