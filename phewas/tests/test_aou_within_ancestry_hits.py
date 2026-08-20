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


def test_production_phewas_runs_one_inversion_pool_at_a_time(
    monkeypatch, tmp_path: Path
):
    commands = []
    monkeypatch.setattr(workflow, "_run", lambda command, **_kwargs: commands.append(command))
    monkeypatch.setattr(workflow, "validate_pcs", lambda *_args: None)
    monkeypatch.setattr(workflow, "validate_phewas_result", lambda *_args: 1)

    paths = workflow.Paths(
        repo=tmp_path,
        local=tmp_path / "local",
        v8=tmp_path / "v8",
    )
    paths.pca_output.mkdir(parents=True)
    paths.results.mkdir(parents=True)
    (paths.pca_output / "within_ancestry_pcs_eur.tsv").touch()
    (paths.pca_output / "within_ancestry_pcs_eur.json").touch()

    workflow.run_population(paths, "eur")

    phewas_command = [str(value) for value in commands[-1]]
    option = phewas_command.index("--max-concurrent-inversions")
    assert phewas_command[option + 1] == "1"
