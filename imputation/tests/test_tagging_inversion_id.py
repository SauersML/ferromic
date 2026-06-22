"""Regression tests for BUG #9: chr17 tag-SNP span emitted as the inversion coordinate.

The tag SNPs span only a ~29 kb window inside the chr17q21 inversion. The writer used
to label the column with the tag-SNP min/max (chr17-45974480-INV-29218), masquerading
as the inversion coordinate. The canonical inversion is chr17-45585160-INV-706887.
The fix emits the canonical ID in the inversion column and records tag-SNP provenance
in separate metadata columns.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tagging_snp_inversion_dosages as tag  # noqa: E402

CANONICAL = "chr17-45585160-INV-706887"
FAKE = "chr17-45974480-INV-29218"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_writer_emits_canonical_inversion_id(tmp_path):
    iids = ["S1", "S2", "S3"]
    calls = np.array([0.0, np.nan, 2.0], dtype=np.float32)
    selected_bps = [45974480, 45996523, 46003698]
    out = tmp_path / "hardcalls.tsv"

    tag.write_single_inversion_hardcalls_tsv(
        iids=iids, calls=calls, selected_bps=selected_bps,
        chr_label="chr17", out_path=str(out),
    )

    with open(out, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    header = rows[0]

    # Inversion column is the canonical ID, NOT the tag-SNP span.
    assert header[1] == CANONICAL
    assert FAKE not in header
    # Tag-SNP provenance lives in separate metadata columns.
    assert "Tag_SNP_Span" in header
    assert "Tag_SNP_Method" in header

    # Data row carries the canonical-id value and metadata.
    first = rows[1]
    assert first[0] == "S1"
    assert first[1] == "0"           # call value
    assert first[2].startswith("chr17:")  # tag span metadata
    assert "unanimity_hardcall" in first[3]
    # No-call row is blank in the dosage column.
    assert rows[2][1] == ""


def test_committed_tsv_uses_canonical_id():
    """The committed all_pop_phewas_tag.tsv must be relabeled to the canonical ID."""
    path = os.path.join(REPO_ROOT, "data", "all_pop_phewas_tag.tsv")
    if not os.path.exists(path):
        return  # data file not present in this checkout; skip
    with open(path, "rb") as f:
        data = f.read()
    assert FAKE.encode() not in data, "fake tag-span inversion ID still present in tsv"
    assert CANONICAL.encode() in data, "canonical inversion ID missing from tsv"
