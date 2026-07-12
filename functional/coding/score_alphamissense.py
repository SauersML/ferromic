"""AlphaMissense scoring for the mapped missense variants, plus a same-gene matched-null
percentile and per-transcript reference protein extraction (input to ESM C).

AlphaMissense_hg38.tsv columns:
``CHROM POS REF ALT genome uniprot_id transcript_id protein_variant am_pathogenicity am_class``.
Variants are matched on genomic plus-strand ``CHROM/POS/REF/ALT``.

The matched null (``percentile_in_gene``) is the exact percentile of an observed variant's
pathogenicity within the full distribution of AlphaMissense scores for *all possible*
missense in the same transcript — the "random same-gene variant" null, computed exactly
rather than sampled. It is what separates a real lesion from background H1/H2 divergence.
"""
from __future__ import annotations

import gzip
import os
from collections import defaultdict

import numpy as np

from .codons import CODON_TABLE, revcomp

AM_PATHOGENIC = 0.564  # published likely-pathogenic cutoff (Cheng et al., Science 2023)


def parse_gtf_cds(gtf_path: str, wanted_tx) -> dict:
    """Collect sorted CDS intervals + strand/chrom per (unversioned) transcript id."""
    wanted = {t.split(".")[0] for t in wanted_tx}
    out: dict = defaultdict(lambda: {"strand": None, "chrom": None, "cds": []})
    opener = gzip.open if gtf_path.endswith(".gz") else open
    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] != "CDS":
                continue
            idx = f[8].find('transcript_id "')
            if idx < 0:
                continue
            tbase = f[8][idx + 15:].split('"', 1)[0].split(".")[0]
            if tbase not in wanted:
                continue
            rec = out[tbase]
            rec["strand"], rec["chrom"] = f[6], f[0]
            rec["cds"].append((int(f[3]), int(f[4])))
    for rec in out.values():
        rec["cds"].sort()
    return dict(out)


def translate_cds(rec: dict, fa) -> str:
    """Translate a transcript's concatenated CDS (strand-aware) to protein."""
    seq = "".join(fa[rec["chrom"]][s - 1:e].seq.upper() for s, e in rec["cds"])
    if rec["strand"] == "-":
        seq = revcomp(seq)
    return "".join(CODON_TABLE.get(seq[i:i + 3], "X") for i in range(0, len(seq) - 2, 3))


def lookup_alphamissense(targets, am_path: str) -> dict:
    """Return ``{(chrom, pos, ref, alt): (am_pathogenicity, am_class)}`` for the target
    variants. Uses a tabix index if present, else a single filtered linear scan.

    ``targets`` is an iterable of ``(chrom, pos, ref, alt)`` tuples.
    """
    want = {(str(c), int(p), r, a) for c, p, r, a in targets}
    scores: dict = {}
    if os.path.exists(am_path + ".tbi"):
        import pysam
        tb = pysam.TabixFile(am_path)
        for chrom, pos, ref, alt in want:
            try:
                for row in tb.fetch(chrom, pos - 1, pos):
                    c = row.split("\t")
                    if int(c[1]) == pos and c[2] == ref and c[3] == alt:
                        scores[(chrom, pos, ref, alt)] = (float(c[8]), c[9])
                        break
            except (ValueError, OSError):
                pass
    else:
        want_pos = {(c, p) for (c, p, _, _) in want}
        with gzip.open(am_path, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                c = line.rstrip("\n").split("\t")
                if (c[0], int(c[1])) in want_pos:
                    key = (c[0], int(c[1]), c[2], c[3])
                    if key in want:
                        scores[key] = (float(c[8]), c[9])
    return scores


def same_gene_null_percentiles(missense_rows: list[dict], am_path: str) -> dict:
    """For each missense variant, its pathogenicity percentile within the full AlphaMissense
    distribution over its own transcript. Returns ``{(gene_name, protein_change): percentile}``.

    One pass over AlphaMissense collects every score for the wanted transcripts, so the
    percentile is exact (not sampled).
    """
    tx_of = {}
    for r in missense_rows:
        tb = str(r["transcript_id"]).split(".")[0]
        tx_of.setdefault(tb, r["gene_name"])
    dist = {t: [] for t in tx_of}
    with gzip.open(am_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            tb = c[6].split(".")[0]
            if tb in dist:
                try:
                    dist[tb].append(float(c[8]))
                except ValueError:
                    pass
    pct = {}
    for r in missense_rows:
        tb = str(r["transcript_id"]).split(".")[0]
        d = np.array(dist.get(tb, []))
        obs = r.get("am_pathogenicity")
        obs = float(obs) if obs not in (None, "") else None
        pct[(r["gene_name"], r.get("protein_change"))] = (
            float((d < obs).mean()) if (len(d) and obs is not None) else None
        )
    return pct
