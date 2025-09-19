"""Utility helpers for mapping inversion IDs to genomic region labels."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict

import pandas as pd

INV_INFO_PATH = "inv_info.tsv"


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
    """Return a mapping {OrigID -> chr:start-end} using ``inv_info.tsv``."""
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


__all__ = [
    "INV_INFO_PATH",
    "load_inv_region_map",
    "map_inversion_value",
    "map_inversion_series",
]
