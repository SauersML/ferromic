import json
from pathlib import Path

import pytest

from imputation.targets import PHEWAS_TARGET_INVERSIONS
from phewas import aou_within_ancestry_hits as workflow


def test_shortlist_requires_exactly_37_unique_names(tmp_path: Path):
    path = tmp_path / "phenotypes.txt"
    path.write_text(
        "# selected\n" + "\n".join(f"Trait_{index}" for index in range(37)) + "\n",
        encoding="utf-8",
    )
    workflow.validate_shortlist(path)
    path.write_text("Trait_A\nTrait_A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="37 unique"):
        workflow.validate_shortlist(path)


def test_dosage_checkpoint_requires_targets_and_exact_row_count(tmp_path: Path):
    path = tmp_path / "imputed_inversion_dosages.tsv"
    path.write_text(
        "SampleID\t" + "\t".join(PHEWAS_TARGET_INVERSIONS) + "\n"
        "1001\t" + "\t".join("0.5" for _ in PHEWAS_TARGET_INVERSIONS) + "\n",
        encoding="utf-8",
    )
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(
        json.dumps({"targets": list(PHEWAS_TARGET_INVERSIONS), "sample_count": 1}),
        encoding="utf-8",
    )
    assert workflow.validate_dosages(path) == 1
    sidecar.write_text(
        json.dumps({"targets": list(PHEWAS_TARGET_INVERSIONS), "sample_count": 2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="row count"):
        workflow.validate_dosages(path)


def test_phewas_result_checkpoint_requires_analysis_columns(tmp_path: Path):
    path = tmp_path / "result.tsv"
    path.write_text(
        "Phenotype\tInversion\tBeta\tOR\tP_Value\n"
        "Trait_A\tinv-a\t0.1\t1.1\t0.2\n",
        encoding="utf-8",
    )
    assert workflow.validate_phewas_result(path) == 1
