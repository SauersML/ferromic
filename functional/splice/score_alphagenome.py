"""AlphaGenome API driver: score each inversion for per-gene, per-tissue signed expression
log-fold-change and per-gene splice-site-usage disruption, writing one ``.npz`` per inversion.

Modelling choice (carried from experiment #6): each inversion breakpoint is represented as a
reverse-complement substitution at the breakpoint base(s), scored in a window centred on the
breakpoint, using AlphaGenome's ``RNA_SEQ`` and ``SPLICE_SITE_USAGE`` recommended variant
scorers. ``ads[0]`` is the gene x track RNA-seq LFC matrix; ``ads[1]`` is the gene x track
splice-site-usage disruption, collapsed to one ``splice_abs`` per gene.

Requires ``pip install alphagenome`` and an ``ALPHAGENOME_API_KEY``. This module makes network
calls, so it is not exercised by the reproduction tests; the cached ``.npz`` outputs are the
inputs to :mod:`functional.splice.formulations`.
"""
from __future__ import annotations

import os

import numpy as np

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


def make_client(api_key: str | None = None):
    """Create an AlphaGenome DNA client + the RNA-seq and splice-site-usage scorers."""
    from alphagenome.models import dna_client, variant_scorers

    client = dna_client.create(api_key or os.environ["ALPHAGENOME_API_KEY"])
    scorers = [
        variant_scorers.RECOMMENDED_VARIANT_SCORERS["RNA_SEQ"],
        variant_scorers.RECOMMENDED_VARIANT_SCORERS["SPLICE_SITE_USAGE"],
    ]
    return client, scorers


def score_breakpoint(client, scorers, chrom: str, position: int, ref: str, window: int) -> dict:
    """Score one inversion breakpoint. Returns a dict with ``rna_lfc`` [genes x tracks],
    ``splice_abs`` [genes], ``gene_ids``, ``gene_names``, ``track_names``, ``track_gtex``.

    The breakpoint is modelled as a reverse-complement substitution (``ref`` -> revcomp(ref))
    in a window of ``window`` bp centred on ``position``.
    """
    from alphagenome.data import genome

    start = max(0, position - window // 2)
    interval = genome.Interval(chromosome=chrom, start=start, end=start + window)
    variant = genome.Variant(chromosome=chrom, position=position,
                             reference_bases=ref, alternate_bases=revcomp(ref))
    ads = client.score_variant(interval=interval, variant=variant, variant_scorers=scorers)
    rna = ads[0]
    genes = rna.obs
    if rna.shape[0] == 0 or "gene_id" not in genes.columns:
        return {"rna_lfc": np.zeros((0, 0), np.float32), "splice_abs": np.zeros((0,), np.float32),
                "gene_ids": [], "gene_names": [], "track_names": [], "track_gtex": []}
    X = np.asarray(rna.X, dtype=np.float32)
    sp = ads[1]
    sp_abs = np.abs(np.asarray(sp.X, dtype=np.float32)).max(axis=1) if sp.shape[0] else None
    sp_by_gene = {}
    if sp_abs is not None and "gene_id" in sp.obs.columns:
        for i, gid in enumerate(sp.obs["gene_id"]):
            sp_by_gene[str(gid)] = float(sp_abs[i])
    gene_ids = [str(g) for g in genes["gene_id"]]
    return {
        "rna_lfc": X,
        "splice_abs": np.array([sp_by_gene.get(g, np.nan) for g in gene_ids], np.float32),
        "gene_ids": gene_ids,
        "gene_names": [str(g) for g in genes.get("gene_name", gene_ids)],
        "track_names": list(rna.var["name"]),
        "track_gtex": list(rna.var["gtex_tissue"]) if "gtex_tissue" in rna.var else [""] * X.shape[1],
    }


def save_region(out_dir: str, event_id: str, scored: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{event_id.replace(':', '_')}.npz")
    np.savez(path, event_id=event_id, **scored)
    return path
