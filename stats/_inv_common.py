"""Utility helpers for mapping inversion IDs to genomic region labels, and the
canonical chimp-polarization (ancestral vs derived orientation) source of truth.

Project orientation convention (post chimp-polarization cutover)
----------------------------------------------------------------
Group/allele **0 == "direct" == ANCESTRAL** orientation (the arrangement shared
with the primate outgroups), and group/allele **1 == "inverted" == DERIVED**
orientation. This replaces the previous convention where "inverted" merely meant
"not the hg38 reference arrangement". The mapping from the raw hg38-reference
encoding to this polarized encoding is given per inversion by
``data/inversion_polarity.tsv`` (column ``flip_ref_polarity``): when the flip
bit is set, the hg38 reference orientation is itself the DERIVED one, so the
mechanical 0/1 labels were swapped to satisfy "inverted == derived".

Use :func:`load_polarity` / :func:`is_flipped` to consult that table.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Optional

import pandas as pd

INV_INFO_PATH = "inv_properties.tsv"
POLARITY_PATH = "inversion_polarity.tsv"


def _to_int(value) -> int | None:
    """Convert inv_info start/end values to integers (returns None on failure)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def load_inv_region_map(inv_info_path: str = INV_INFO_PATH) -> Dict[str, str]:
    """Return a mapping {OrigID -> chr:start-end} using ``inv_properties.tsv``."""
    if not os.path.exists(inv_info_path):
        return {}
    try:
        info_df = pd.read_csv(inv_info_path, sep="\t", dtype=str)
    except Exception:
        return {}

    required = {"OrigID", "Chromosome", "Start", "End"}
    if not required.issubset(info_df.columns):
        return {}

    mapping: Dict[str, str] = {}
    for _, row in info_df.iterrows():
        orig = (row.get("OrigID") or "").strip()
        if not orig:
            continue

        chrom = (row.get("Chromosome") or "").strip()
        if not chrom:
            continue
        chrom_fmt = chrom if chrom.lower().startswith("chr") else f"chr{chrom}"

        start = _to_int(row.get("Start"))
        end = _to_int(row.get("End"))
        if start is None or end is None:
            continue

        mapping[orig] = f"{chrom_fmt}:{start:,}-{end:,}"
    return mapping


def map_inversion_value(value, inv_info_path: str = INV_INFO_PATH):
    """Map a single inversion ID to its region label (falls back to the input)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    key = str(value).strip()
    if not key:
        return value
    mapping = load_inv_region_map(inv_info_path)
    return mapping.get(key, value)


def map_inversion_series(series: pd.Series, inv_info_path: str = INV_INFO_PATH) -> pd.Series:
    """Vectorized helper that applies :func:`map_inversion_value` to a Series."""
    mapping = load_inv_region_map(inv_info_path)
    if not mapping:
        return series

    def convert(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return value
        text = str(value)
        key = text.strip()
        if not key:
            return text
        return mapping.get(key, key)

    return series.apply(convert)


def _norm_chrom(raw) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    c = str(raw).strip()
    while c.lower().startswith("chr"):
        c = c[3:]
    if not c:
        return None
    cl = c.lower()
    if cl == "x":
        core = "X"
    elif cl == "y":
        core = "Y"
    elif cl in {"m", "mt"}:
        core = "M"
    elif cl.isdigit():
        core = str(int(cl))
    else:
        core = c.upper()
    return "chr" + core


def _polarity_candidate_paths(polarity_path: str):
    seen = set()
    for base in ("", os.getcwd(), os.path.dirname(os.path.abspath(__file__)),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
                 "data"):
        p = os.path.join(base, polarity_path) if base else polarity_path
        p = os.path.normpath(p)
        if p not in seen:
            seen.add(p)
            yield p


@lru_cache(maxsize=4)
def load_polarity(polarity_path: str = POLARITY_PATH) -> Dict[tuple, dict]:
    """Load ``inversion_polarity.tsv`` -> {(chrom,start,end): record}.

    Each record carries at least ``flip`` (bool), ``ancestral_orientation``,
    ``derived_orientation``, ``confidence`` and ``orig_id``. Lookups are
    coordinate-keyed; :func:`is_flipped` adds a +/-1 bp tolerance and OrigID
    fallback. Returns ``{}`` if the table is absent (callers then assume the
    identity polarity, i.e. the raw hg38-reference encoding)."""
    path = None
    for cand in _polarity_candidate_paths(polarity_path):
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        return {}
    try:
        df = pd.read_csv(path, sep="\t", dtype=str)
    except Exception:
        return {}
    out: Dict[tuple, dict] = {}
    for _, row in df.iterrows():
        chrom = _norm_chrom(row.get("chrom"))
        s, e = _to_int(row.get("start")), _to_int(row.get("end"))
        if chrom is None or s is None or e is None:
            continue
        rec = {
            "flip": str(row.get("flip_ref_polarity", "0")).strip() in ("1", "True", "true"),
            "ancestral_orientation": (row.get("ancestral_orientation") or "").strip(),
            "derived_orientation": (row.get("derived_orientation") or "").strip(),
            "confidence": (row.get("confidence") or "").strip(),
            "orig_id": (row.get("orig_id") or "").strip(),
            "chrom": chrom, "start": s, "end": e,
        }
        out[(chrom, s, e)] = rec
    return out


@lru_cache(maxsize=4)
def _polarity_by_orig(polarity_path: str = POLARITY_PATH) -> Dict[str, dict]:
    by_orig: Dict[str, dict] = {}
    for rec in load_polarity(polarity_path).values():
        for orig in (rec.get("orig_id") or "").split(";"):
            orig = orig.strip()
            if orig:
                by_orig[orig] = rec
    return by_orig


def polarity_record(chrom=None, start=None, end=None, orig_id=None,
                    polarity_path: str = POLARITY_PATH) -> Optional[dict]:
    """Return the polarity record for an inversion by coordinates (+/-1 bp) or
    by OrigID. ``None`` if not found."""
    table = load_polarity(polarity_path)
    nc = _norm_chrom(chrom)
    s, e = _to_int(start), _to_int(end)
    if nc is not None and s is not None and e is not None:
        for ds in (0, -1, 1):
            for de in (0, -1, 1):
                rec = table.get((nc, s + ds, e + de))
                if rec is not None:
                    return rec
    if orig_id:
        rec = _polarity_by_orig(polarity_path).get(str(orig_id).strip())
        if rec is not None:
            return rec
    return None


def is_flipped(chrom=None, start=None, end=None, orig_id=None,
               polarity_path: str = POLARITY_PATH) -> bool:
    """True if the hg38-reference orientation is the DERIVED one for this
    inversion (so the mechanical 0/1 encoding must be swapped to make
    1 == derived). Unknown loci default to False (identity polarity)."""
    rec = polarity_record(chrom, start, end, orig_id, polarity_path)
    return bool(rec and rec.get("flip"))


__all__ = [
    "INV_INFO_PATH",
    "POLARITY_PATH",
    "load_inv_region_map",
    "map_inversion_value",
    "map_inversion_series",
    "load_polarity",
    "polarity_record",
    "is_flipped",
]
