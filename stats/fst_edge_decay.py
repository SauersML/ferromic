"""One-sided Spearman decay of FST from inversion edges toward the center."""
from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

RE_HEADER = re.compile(
    r">.*?hudson_pairwise_fst.*?_chr_?(?P<chrom>[\w.\-]+)_start_(?P<start>\d+)_end_(?P<end>\d+)",
    re.IGNORECASE,
)


@dataclass
class FstDecayResult:
    chrom: str
    start: int
    end: int
    recurrence_flag: int
    recurrence_label: str
    rho: float | None
    p_one_sided: float | None
    q_value: float | None
    bins_used: int

    @property
    def inv_label(self) -> str:
        return f"chr{self.chrom}:{self.start}-{self.end}"


def _normalize_chrom(chrom: str) -> str:
    chrom = str(chrom).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    if chrom.lower().startswith("chr_"):
        chrom = chrom[4:]
    return chrom


def _parse_values(lines: Iterable[str]) -> np.ndarray:
    if not lines:
        return np.array([], dtype=np.float64)
    seq = "".join(line.strip() for line in lines if line.strip())
    if not seq:
        return np.array([], dtype=np.float64)
    return np.fromstring(seq.replace("NA", "nan"), sep=",", dtype=np.float64)


def _load_inv_properties(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "inv_properties.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing inversion annotation table: {path}")

    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.rename(
        columns={
            "Chromosome": "chromosome",
            "Start": "start",
            "End": "end",
            "OrigID": "inversion_id",
            "0_single_1_recur_consensus": "recurrence_flag",
        }
    )
    df["chromosome"] = df["chromosome"].astype(str).str.replace("^chr", "", regex=True)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df["recurrence_flag"] = pd.to_numeric(df["recurrence_flag"], errors="coerce")
    df = df[df["recurrence_flag"].isin([0, 1])].copy()
    df["recurrence_label"] = df["recurrence_flag"].map({0: "Single-event", 1: "Recurrent"})
    return df


def _one_sided_p(rho: float | None, p_two_sided: float | None) -> float | None:
    if rho is None or p_two_sided is None or not np.isfinite([rho, p_two_sided]).all():
        return None
    if rho < 0:
        return float(p_two_sided / 2.0)
    if rho > 0:
        return float(1.0 - p_two_sided / 2.0)
    return 0.5


def _spearman_folded(values: np.ndarray) -> tuple[float | None, float | None, int]:
    length = values.size
    if length <= 100_000:
        return None, None, 0

    max_len = min(100_000, length // 2)
    usable = (max_len // 2_000) * 2_000
    if usable < 4_000:  # need at least two bins for correlation
        return None, None, 0

    left = values[:usable]
    right = values[-usable:][::-1]

    try:
        left_bins = np.nanmean(left.reshape(-1, 2_000), axis=1)
        right_bins = np.nanmean(right.reshape(-1, 2_000), axis=1)
    except ValueError:
        return None, None, 0

    folded = np.nanmean(np.vstack([left_bins, right_bins]), axis=0)
    bin_centers = np.arange(folded.size, dtype=float) * 2_000 + 1_000

    mask = np.isfinite(folded)
    if mask.sum() < 2:
        return None, None, int(mask.sum())

    rho_val, p_val = stats.spearmanr(bin_centers[mask], folded[mask])
    rho = float(rho_val) if np.isfinite(rho_val) else None
    p = float(p_val) if np.isfinite(p_val) else None
    return rho, p, int(mask.sum())


def compute_fst_edge_decay(data_dir: Path) -> List[FstDecayResult]:
    candidates = [
        data_dir / "per_site_fst_output.falsta",
        data_dir / "per_site_fst_output.falsta.gz",
    ]
    falsta_path = next((p for p in candidates if p.exists()), None)
    if falsta_path is None:
        raise FileNotFoundError("Missing per_site_fst_output.falsta(.gz)")

    inv_df = _load_inv_properties(data_dir)
    recurrence_map = {
        (str(chrom), int(start), int(end)): (int(flag), label)
        for chrom, start, end, flag, label in inv_df[
            ["chromosome", "start", "end", "recurrence_flag", "recurrence_label"]
        ].itertuples(index=False)
    }

    results: List[FstDecayResult] = []

    def handle_record(header: str | None, seq_lines: List[str]) -> None:
        if not header:
            return
        match = RE_HEADER.search(header)
        if not match:
            return

        chrom = _normalize_chrom(match.group("chrom"))
        start = int(match.group("start"))
        end = int(match.group("end"))
        key = (chrom, start, end)
        if key not in recurrence_map:
            return

        values = _parse_values(seq_lines)
        if values.size == 0 or not np.isfinite(values).any():
            return

        rho, p_two_sided, bins_used = _spearman_folded(values)
        if bins_used == 0:
            return

        p_one_sided = _one_sided_p(rho, p_two_sided)
        flag, label = recurrence_map[key]
        results.append(
            FstDecayResult(
                chrom=chrom,
                start=start,
                end=end,
                recurrence_flag=flag,
                recurrence_label=label,
                rho=rho,
                p_one_sided=p_one_sided,
                q_value=None,
                bins_used=bins_used,
            )
        )

    opener = gzip.open if falsta_path.suffix == ".gz" else open
    with opener(falsta_path, "rt", encoding="utf-8", errors="ignore") as handle:
        current_header: str | None = None
        seq_lines: List[str] = []
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                handle_record(current_header, seq_lines)
                current_header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        handle_record(current_header, seq_lines)

    if not results:
        return []

    pvals = [r.p_one_sided for r in results]
    valid_mask = [p is not None and np.isfinite(p) for p in pvals]
    if any(valid_mask):
        valid_p = [p for p, keep in zip(pvals, valid_mask) if keep]
        _, qvals, _, _ = multipletests(valid_p, method="fdr_bh")
        q_iter = iter(qvals)
        for res, keep in zip(results, valid_mask):
            res.q_value = float(next(q_iter)) if keep else None
    return results


__all__ = ["FstDecayResult", "compute_fst_edge_decay"]
