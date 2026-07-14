"""Gene-localised AlphaGenome splice-disruption formulation (the validated formulation).

Experiment #6 searched a large space of ways to turn an inversion's per-gene x per-tissue
AlphaGenome signal into a single pathogenicity score. The formulation that beat the
breakpoint baseline and validated against measured GTEx sQTLs is *gene-localised splice
disruption*: for each inversion, take the per-gene splice-site-usage disruption
(``splice_abs``, the max over splice tracks already collapsed by AlphaGenome), and report the
gene with the largest disruption as the inversion's top-splice gene. This module recomputes
that formulation from the cached per-inversion AlphaGenome scores (``.npz``); it does not call
the API (see :mod:`functional.splice.score_alphagenome`).

Each ``.npz`` (one per inversion region) carries:
  ``event_id`` (region id), ``gene_ids``, ``gene_names``, ``rna_lfc`` [genes x tracks] signed
  expression LFC, ``splice_abs`` [genes] splice-site-usage disruption, ``track_gtex`` [tracks]
  GTEx tissue label per track.
"""
from __future__ import annotations

import glob
import os

import numpy as np


def _ensg(x) -> str:
    return str(x).split(".")[0]


def load_region(npz_path: str) -> dict:
    """Per-gene summary for one inversion region from its AlphaGenome ``.npz``.

    Returns ``{gene_id: {"name", "splice", "expr_signed", "expr_absmax"}}``. ``expr_signed``
    is the signed LFC at the gene's most-disrupted GTEx tissue track (max |LFC| over tissues).
    """
    from collections import defaultdict

    d = np.load(npz_path, allow_pickle=True)
    rna = np.asarray(d["rna_lfc"], float)
    sp = np.asarray(d["splice_abs"], float)
    gids = [_ensg(x) for x in d["gene_ids"]]
    names = [str(x) for x in d["gene_names"]]
    tcols: dict = defaultdict(list)
    for j, t in enumerate([str(x) for x in d["track_gtex"]]):
        if t:
            tcols[t].append(j)
    genes = {}
    for i, g in enumerate(gids):
        per_tissue = []
        for cols in tcols.values():
            v = rna[i, cols]
            if not np.all(np.isnan(v)):
                per_tissue.append(v[np.nanargmax(np.abs(v))])
        expr_signed = per_tissue[int(np.nanargmax(np.abs(per_tissue)))] if per_tissue else np.nan
        genes[g] = {
            "name": names[i],
            "splice": float(sp[i]) if i < len(sp) else np.nan,
            "expr_signed": float(expr_signed),
            "expr_absmax": float(np.nanmax(np.abs(rna[i]))) if rna[i].size else np.nan,
        }
    return genes


def top_splice_gene(genes: dict) -> tuple[str | None, str | None, float]:
    """The gene with maximal splice disruption. Returns ``(gene_id, gene_name, ag_max_splice)``.
    ``(None, None, nan)`` if no gene has a finite splice score."""
    items = [(g, d["splice"]) for g, d in genes.items() if not np.isnan(d["splice"])]
    if not items:
        return None, None, float("nan")
    gid, val = max(items, key=lambda x: x[1])
    return gid, genes[gid]["name"], float(val)


def load_all(npz_dir: str) -> dict:
    """Load every region ``.npz`` under ``npz_dir`` -> ``{region_id: genes}``."""
    out = {}
    for p in glob.glob(os.path.join(npz_dir, "*.npz")):
        region = str(np.load(p, allow_pickle=True)["event_id"])
        out[region] = load_region(p)
    return out
