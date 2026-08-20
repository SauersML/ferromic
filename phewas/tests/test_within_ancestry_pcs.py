"""Tests for the ancestry-specific ("fine-scale") principal-component source.

These cover the toggle itself rather than the fitting step: that ``--pc-source`` is
plumbed across the spawn boundary, that ancestry-specific components replace the global
ones only for single-population runs, and that the two arms cannot share a cache.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from phewas import cli, iox as io, run


@pytest.fixture(autouse=True)
def _restore_run_globals():
    """The pipeline configures itself through module globals plus the environment."""
    saved = {
        "PC_SOURCE": run.PC_SOURCE,
        "POPULATION_FILTER": run.POPULATION_FILTER,
        "PHENOTYPE_FILTER": run.PHENOTYPE_FILTER,
        "WITHIN_ANCESTRY_PCS_URI": run.WITHIN_ANCESTRY_PCS_URI,
        "NUM_PCS": run.NUM_PCS,
        "MASTER_RESULTS_CSV": run.MASTER_RESULTS_CSV,
    }
    saved_env = {
        key: os.environ.get(key)
        for key in (
            "FERROMIC_PC_SOURCE",
            "FERROMIC_POPULATION_FILTER",
            "FERROMIC_PHENOTYPE_FILTER",
            "FERROMIC_WITHIN_ANCESTRY_PCS_URI",
        )
    }
    yield
    for key, value in saved.items():
        setattr(run, key, value)
    for key, value in saved_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _covariates(n=40, num_pcs=16, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.Index([f"p{i}" for i in range(n)], name="person_id")
    frame = pd.DataFrame(
        {f"PC{i}": rng.normal(0, 0.01, n) for i in range(1, num_pcs + 1)},
        index=index,
    )
    frame["AGE"] = rng.integers(40, 80, n).astype(float)
    frame["sex"] = rng.integers(0, 2, n).astype(float)
    return frame


def _within_pcs(index, k=8, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f"WPC{i}": rng.normal(0, 1.0, len(index)) for i in range(1, k + 1)},
        index=pd.Index([str(v) for v in index], name="person_id"),
    )


# --- the toggle ------------------------------------------------------------------


def test_within_ancestry_requires_a_population_label():
    args = cli.parse_args(["--pc-source", "within-ancestry"])
    with pytest.raises(SystemExit) as excinfo:
        cli.apply_cli_configuration(args)
    assert "--pop-label" in str(excinfo.value)


def test_within_ancestry_with_population_label_is_accepted():
    args = cli.parse_args(["--pc-source", "within-ancestry", "--pop-label", "EUR"])
    config = cli.apply_cli_configuration(args)
    assert config["pc_source"] == run.PC_SOURCE_WITHIN_ANCESTRY
    assert config["population_filter"] == "EUR"
    # The pipeline body runs in a spawned subprocess and does not inherit globals.
    assert os.environ["FERROMIC_PC_SOURCE"] == run.PC_SOURCE_WITHIN_ANCESTRY


def test_default_pc_source_leaves_no_environment_footprint():
    args = cli.parse_args([])
    config = cli.apply_cli_configuration(args)
    assert config["pc_source"] == run.PC_SOURCE_GLOBAL
    assert "FERROMIC_PC_SOURCE" not in os.environ


def test_apply_pipeline_config_round_trips_through_the_environment():
    run._apply_pipeline_config(
        {"population_filter": "afr", "pc_source": "within-ancestry"}
    )
    assert run.PC_SOURCE == run.PC_SOURCE_WITHIN_ANCESTRY
    assert os.environ["FERROMIC_PC_SOURCE"] == "within-ancestry"
    # Runs differing only in PC source must be distinguishable on disk.
    assert "pop-afr" in run.MASTER_RESULTS_CSV
    assert "pcs-within-ancestry" in run.MASTER_RESULTS_CSV


def test_explicit_output_path_replaces_timestamped_name(tmp_path):
    output = tmp_path / "eur_within.tsv"
    args = cli.parse_args(
        [
            "--pop-label",
            "eur",
            "--pc-source",
            "within-ancestry",
            "--output",
            str(output),
        ]
    )
    config = cli.apply_cli_configuration(args)
    run._apply_pipeline_config(config)
    assert run.MASTER_RESULTS_CSV == str(output.resolve())


def test_apply_pipeline_config_rejects_pooled_within_ancestry():
    with pytest.raises(ValueError, match="single-population"):
        run._apply_pipeline_config(
            {"population_filter": "all", "pc_source": "within-ancestry"}
        )


def test_unknown_pc_source_is_rejected():
    with pytest.raises(ValueError, match="Unknown principal-component source"):
        run._normalize_pc_source("projected")


def test_uri_template_requires_a_population_placeholder():
    run.WITHIN_ANCESTRY_PCS_URI = "gs://bucket/wpcs.tsv"
    with pytest.raises(ValueError, match=r"\{pop\}"):
        run._resolve_within_ancestry_pcs_uri("eur")


def test_uri_template_substitutes_the_population_label():
    run.WITHIN_ANCESTRY_PCS_URI = "gs://bucket/wpcs_{pop}.tsv"
    assert run._resolve_within_ancestry_pcs_uri("EUR") == "gs://bucket/wpcs_eur.tsv"


# --- the substitution --------------------------------------------------------------


def test_swap_replaces_global_pcs_and_reports_the_component_count():
    covariates = _covariates()
    within = _within_pcs(covariates.index, k=8)

    updated, k = run._apply_within_ancestry_pcs(covariates, within, "eur")

    assert k == 8
    assert [c for c in updated.columns if c.startswith("PC")] == [f"PC{i}" for i in range(1, 9)]
    # Values come from the ancestry-specific table, not the projected global one.
    np.testing.assert_allclose(
        updated["PC1"].to_numpy(), within["WPC1"].to_numpy().astype(np.float32), rtol=1e-6
    )
    # Non-PC covariates are untouched.
    pd.testing.assert_series_equal(updated["AGE"], covariates["AGE"])


def test_swap_refuses_to_silently_drop_participants():
    covariates = _covariates(n=20)
    within = _within_pcs(covariates.index[:18], k=4)
    with pytest.raises(ValueError, match="no ancestry-specific"):
        run._apply_within_ancestry_pcs(covariates, within, "eur")


def test_swap_rejects_non_contiguous_components():
    covariates = _covariates(n=10)
    within = _within_pcs(covariates.index, k=3).rename(columns={"WPC2": "WPC9"})
    with pytest.raises(ValueError, match="contiguous"):
        run._apply_within_ancestry_pcs(covariates, within, "eur")


def test_swap_requires_within_ancestry_columns():
    covariates = _covariates(n=10)
    empty = pd.DataFrame(index=covariates.index)
    with pytest.raises(ValueError, match="no WPC"):
        run._apply_within_ancestry_pcs(covariates, empty, "eur")


# --- the loader ----------------------------------------------------------------------


def test_loader_reads_a_wide_table(tmp_path):
    path = tmp_path / "wpcs_eur.tsv"
    frame = pd.DataFrame(
        {
            "person_id": ["a", "b", "c"],
            "WPC1": [0.1, -0.2, 0.3],
            "WPC2": [1.0, 0.5, -1.5],
        }
    )
    frame.to_csv(path, sep="\t", index=False)

    loaded = io.load_within_ancestry_pcs("project", str(path))

    assert list(loaded.columns) == ["WPC1", "WPC2"]
    assert loaded.index.tolist() == ["a", "b", "c"]
    assert loaded.index.name == "person_id"


def test_loader_rejects_duplicate_participants(tmp_path):
    path = tmp_path / "wpcs_eur.tsv"
    pd.DataFrame({"person_id": ["a", "a"], "WPC1": [0.1, 0.2]}).to_csv(
        path, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="Duplicate person_id"):
        io.load_within_ancestry_pcs("project", str(path))


def test_loader_rejects_missing_values(tmp_path):
    path = tmp_path / "wpcs_eur.tsv"
    pd.DataFrame({"person_id": ["a", "b"], "WPC1": [0.1, None]}).to_csv(
        path, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="non-finite"):
        io.load_within_ancestry_pcs("project", str(path))


# --- cache separation -----------------------------------------------------------------


def test_covariate_cache_key_separates_the_two_pc_sources():
    """An EUR run with fine-scale PCs must not reuse the EUR global-PC covariate cache."""
    shared = ("demographics.parquet", "pcs.parquet", "sex.parquet", "anc.parquet", "proj")
    global_key = run._source_key(*shared, run.PC_SOURCE_GLOBAL, None, 16)
    within_key = run._source_key(*shared, run.PC_SOURCE_WITHIN_ANCESTRY, "wpcs_eur.tsv", 8)
    assert global_key != within_key


# --- fitting inputs (the producer side) ------------------------------------------------


from phewas.extra import within_ancestry_pca as wapca  # noqa: E402


def test_inversion_ids_become_padded_regions():
    regions = wapca.inversion_regions(["chr17-45585160-INV-706887"], flank=1_000_000)
    assert len(regions) == 1
    region = regions[0]
    assert region.chrom == "chr17"
    assert region.start == 44_585_160
    assert region.end == 45_585_160 + 706_887 + 1_000_000


def test_excluded_regions_cover_the_conventional_blocks_and_every_inversion():
    regions = wapca.excluded_regions()
    reasons = " ".join(r.reason for r in regions)
    assert "MHC" in reasons
    assert sum("tested inversion" in r.reason for r in regions) == len(run.TARGET_INVERSIONS)


def test_site_list_drops_excluded_regions_and_non_autosomes(tmp_path):
    bim = tmp_path / "arrays.bim"
    rows = [
        ("1", "keep_1", 0, 1_000_000, "A", "G"),      # retained
        ("6", "mhc", 0, 30_000_000, "A", "G"),        # MHC
        ("17", "inv17", 0, 45_600_000, "A", "G"),     # 17q21.31 inversion
        ("17", "flank17", 0, 44_600_000, "A", "G"),   # inside the 1 Mb flank
        ("22", "keep_22", 0, 20_000_000, "A", "G"),   # retained
        ("X", "sex_chrom", 0, 5_000_000, "A", "G"),   # non-autosomal
    ]
    with open(bim, "w") as handle:
        for row in rows:
            handle.write("\t".join(str(v) for v in row) + "\n")

    out = tmp_path / "sites.tsv"
    sites = wapca.build_site_list(str(bim), str(out))

    positions = set(sites["pos"])
    assert positions == {1_000_000, 20_000_000}
    written = pd.read_csv(out, sep="\t", header=None)
    assert len(written) == 2


def test_fit_defaults_match_the_fixed_phewas_design():
    args = wapca.build_parser().parse_args(
        [
            "fit", "--genotypes", "arrays", "--sites", "sites.tsv",
            "--group", "eur", "--keep", "keep.txt",
        ]
    )
    assert args.components == 16
    assert args.threads == 4
    assert args.maf == 0.01
    assert args.geno == 0.05
    assert args.mind == 0.05
    assert wapca.GNOMON_LD_DEFAULT_MARKERS == 100_000
    assert wapca.GNOMON_LD_DEFAULT_WINDOW_BP == 500_000


def test_fit_rejects_a_group_that_could_escape_its_output_prefix(tmp_path):
    args = wapca.build_parser().parse_args(
        [
            "fit", "--genotypes", "arrays", "--sites", "sites.tsv",
            "--group", "../eur", "--keep", "keep.txt", "--out-dir", str(tmp_path),
        ]
    )
    with pytest.raises(SystemExit, match="--group must start with a letter"):
        wapca.fit_group(args)


def test_keep_list_intersects_cohort_and_drops_related(tmp_path):
    ancestry = tmp_path / "ancestry.tsv"
    pd.DataFrame(
        {
            "person_id": ["a", "b", "c", "d"],
            "ancestry_pred": ["eur", "eur", "eur", "afr"],
        }
    ).to_csv(ancestry, sep="\t", index=False)

    cohort = tmp_path / "cohort.txt"
    cohort.write_text("a\nb\nc\n")
    related = tmp_path / "related.txt"
    related.write_text("c\n")

    ids = wapca.build_keep_list(
        ancestry_path=str(ancestry),
        group="EUR",
        out_path=str(tmp_path / "keep_eur.txt"),
        genotype_ids={"a", "c"},
        cohort_path=str(cohort),
        related_path=str(related),
    )
    assert ids == ["a"]


def test_keep_list_refuses_an_empty_group(tmp_path):
    ancestry = tmp_path / "ancestry.tsv"
    pd.DataFrame({"person_id": ["a"], "ancestry_pred": ["afr"]}).to_csv(
        ancestry, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="No participants remain"):
        wapca.build_keep_list(
            ancestry_path=str(ancestry),
            group="eas",
            out_path=str(tmp_path / "k.txt"),
            genotype_ids={"a"},
        )


# --- gnomon output format ---------------------------------------------------------------


def _pack_scores(scores: np.ndarray, row_ids: list[str]) -> bytes:
    """Build a gnomon score matrix by hand, mirroring map/io.rs."""
    rows, cols = scores.shape
    header = bytearray(wapca._MATRIX_HEADER_LEN)
    header[0:8] = wapca._MATRIX_MAGIC
    header[8:12] = wapca._MATRIX_VERSION.to_bytes(4, "little")
    header[12:20] = rows.to_bytes(8, "little")
    header[20:28] = cols.to_bytes(8, "little")
    header[28:32] = wapca._MATRIX_ELEMENT_KIND_F64_LE.to_bytes(4, "little")
    body = np.asarray(scores, dtype="<f8").T.tobytes(order="C")

    encoded = [rid.encode("utf-8") for rid in row_ids]
    offsets, running = [0], 0
    for chunk in encoded:
        running += len(chunk)
        offsets.append(running)
    id_header = bytearray(wapca._ROW_IDS_HEADER_LEN)
    id_header[0:8] = wapca._ROW_IDS_MAGIC
    id_header[8:12] = wapca._ROW_IDS_VERSION.to_bytes(4, "little")
    id_header[16:24] = len(row_ids).to_bytes(8, "little")
    id_header[24:32] = running.to_bytes(8, "little")
    offset_bytes = b"".join(o.to_bytes(8, "little") for o in offsets)
    return bytes(header) + body + bytes(id_header) + offset_bytes + b"".join(encoded)


def _write_fit_outputs(directory, stem, scores, row_ids):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.hwe_scores.bin").write_bytes(_pack_scores(scores, row_ids))
    (directory / f"{stem}.hwe_scores.metadata.json").write_text(
        json.dumps(
            {
                "version": wapca._MATRIX_VERSION,
                "kind": "scores",
                "layout": "column_major",
                "dtype": "f64_le",
                "element_kind": wapca._MATRIX_ELEMENT_KIND_F64_LE,
                "endianness": "little",
                "row_ids_embedded": True,
                "row_id_field": "IID",
                "rows": scores.shape[0],
                "cols": scores.shape[1],
            }
        )
    )
    (directory / f"{stem}.hwe.json").write_text("{}")
    (directory / f"{stem}.hwe.project.bin").write_bytes(b"projection cache")
    (directory / f"{stem}.hwe_summary.tsv").write_text(
        "metric\tvalue\nn_samples\t3\nconverged\ttrue\n"
    )
    (directory / f"{stem}.samples.tsv").write_text("FID\tIID\n0\ta\n0\tb\n")


def test_native_output_prefix_isolates_each_groups_outputs(tmp_path):
    eur = wapca._artifact_paths(str(tmp_path / "gnomon_eur"))
    afr = wapca._artifact_paths(str(tmp_path / "gnomon_afr"))
    assert eur["hwe.json"] == str(tmp_path / "gnomon_eur.hwe.json")
    assert eur["hwe_scores.bin"] != afr["hwe_scores.bin"]


def test_score_matrix_round_trips(tmp_path):
    scores = np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]])
    row_ids = ["1001", "1002", "1003"]
    _write_fit_outputs(tmp_path, "arrays", scores, row_ids)

    decoded, ids = wapca._read_hwe_scores(
        str(tmp_path / "arrays.hwe_scores.bin"),
        str(tmp_path / "arrays.hwe_scores.metadata.json"),
    )
    np.testing.assert_allclose(decoded, scores)
    assert ids == row_ids


def test_scores_frame_labels_components_and_ids(tmp_path):
    scores = np.array([[0.1, -0.2, 9.9], [0.3, 0.4, 9.9]])
    _write_fit_outputs(tmp_path, "arrays", scores, ["a", "b"])
    collected = {
        "hwe_scores.bin": str(tmp_path / "arrays.hwe_scores.bin"),
        "hwe_scores.metadata.json": str(tmp_path / "arrays.hwe_scores.metadata.json"),
    }
    frame = wapca._scores_frame(collected, components=2)
    assert list(frame.columns) == ["person_id", "WPC1", "WPC2"]
    assert frame["person_id"].tolist() == ["a", "b"]


def test_scores_frame_rejects_too_few_components(tmp_path):
    _write_fit_outputs(tmp_path, "arrays", np.zeros((2, 2)), ["a", "b"])
    collected = {
        "hwe_scores.bin": str(tmp_path / "arrays.hwe_scores.bin"),
        "hwe_scores.metadata.json": str(tmp_path / "arrays.hwe_scores.metadata.json"),
    }
    with pytest.raises(SystemExit, match="Expected 5 components"):
        wapca._scores_frame(collected, components=5)


def test_scores_frame_rejects_duplicate_participants(tmp_path):
    _write_fit_outputs(tmp_path, "arrays", np.zeros((2, 2)), ["a", "a"])
    collected = {
        "hwe_scores.bin": str(tmp_path / "arrays.hwe_scores.bin"),
        "hwe_scores.metadata.json": str(tmp_path / "arrays.hwe_scores.metadata.json"),
    }
    with pytest.raises(SystemExit, match="duplicate participant"):
        wapca._scores_frame(collected, components=2)


def test_fit_artifacts_requires_complete_current_main_output(tmp_path):
    _write_fit_outputs(tmp_path, "pca_eur", np.zeros((2, 2)), ["a", "b"])

    collected = wapca._fit_artifacts(str(tmp_path / "pca_eur"))
    assert collected["hwe.json"] == str(tmp_path / "pca_eur.hwe.json")
    assert collected["hwe.project.bin"] == str(tmp_path / "pca_eur.hwe.project.bin")

    (tmp_path / "pca_eur.samples.tsv").unlink()
    with pytest.raises(SystemExit, match="samples.tsv"):
        wapca._fit_artifacts(str(tmp_path / "pca_eur"))


def test_gnomon_fit_uses_current_main_flags_once(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        stdout = (
            "gnomon 0.1.0\nRelease: development build\n"
            if command[-1] == "version"
            else ""
        )
        return __import__("subprocess").CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(wapca.subprocess, "run", fake_run)
    monkeypatch.setattr(wapca, "_full_digest", lambda _: "abc123")
    provenance = wapca._run_gnomon_fit(
        gnomon="/bin/gnomon",
        genotypes="/data/arrays.bed",
        output_prefix="/results/gnomon_eur",
        keep_path="/data/keep.txt",
        sites_path="/data/sites.tsv",
        components=16,
        threads=4,
        maf=0.01,
        geno=0.05,
        mind=0.05,
        workdir=str(tmp_path),
    )

    fit_command = calls[1][0]
    assert fit_command[:3] == ["/bin/gnomon", "fit", "/data/arrays.bed"]
    assert fit_command.count("/data/arrays.bed") == 1
    for flag in (
        "--out", "--keep", "--list", "--components", "--threads",
        "--mind", "--geno", "--maf", "--ld",
    ):
        assert flag in fit_command
    for obsolete_workaround in ("--markers", "--bp_window", "--allow-unconverged"):
        assert obsolete_workaround not in fit_command
    assert "preexec_fn" not in calls[1][1]
    assert provenance["version"].startswith("gnomon 0.1.0")
    assert provenance["executable_sha256"] == "abc123"


def test_fit_summary_requires_current_schema_and_convergence(tmp_path):
    summary = tmp_path / "hwe_summary.tsv"
    pd.DataFrame(
        {"metric": ["n_samples", "converged"], "value": [55000, "true"]}
    ).to_csv(
        summary, sep="\t", index=False
    )
    parsed = wapca._fit_summary(str(summary))
    assert parsed == {"n_samples": "55000", "converged": "true"}
    json.dumps({"hwe_summary": parsed})

    summary.write_text("metric\tvalue\nconverged\tfalse\n")
    with pytest.raises(SystemExit, match="without a certified converged"):
        wapca._fit_summary(str(summary))
