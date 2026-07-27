#!/usr/bin/env python
"""Reconcile the breakpoint-decay Spearman: rho = 0.500 vs the manuscript's 0.451.

`replicate_manuscript_statistics.txt` reports rho = 0.500, p = 2.156e-04 for the
overall decay of nucleotide diversity against distance from the inversion edge;
the manuscript reports rho = 0.451, p = 0.001. Both quote "n = 60".

Two things are going on.

1. **The reported n is not the test's sample size.** In
   `replicate_manuscript_statistics.py::_calc_spearman`, `n` is
   `len(window_data)` -- the number of contributing inversion x orientation
   series -- while the correlation is computed over the 2 kbp *bins* spanning
   0-100 kbp, i.e. at most 50 points. Both published rho values are consistent
   with 50 bins and not with 60 (rho = 0.500 over 50 bins gives p = 2.2e-04;
   rho = 0.451 gives p = 1.0e-03). So the two numbers are the same test at the
   same n, differing only in how per-locus series are combined.

2. **The combination rule is the free parameter.** This script recomputes the
   overall decay from the committed per-site tracks under every defensible
   combination of (within-locus site aggregation) x (across-locus aggregation)
   and reports which reproduces which published value, so the released number can
   be stated with its rule rather than left ambiguous.

Input : data/per_site_diversity_output.falsta.gz (committed, per-base filtered pi)
        data/inv_properties.tsv (locus list + recurrence)
Output: data/decay_spearman_variants.tsv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import re

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FALSTA = os.path.join(REPO, "data", "per_site_diversity_output.falsta.gz")
OUT = os.path.join(REPO, "data", "decay_spearman_variants.tsv")

BIN_SIZE = 2_000
MAX_DIST = 100_000
N_BINS = MAX_DIST // BIN_SIZE          # 50
MIN_INV_PER_BIN = 5
HEADER_RE = re.compile(
    r"^>filtered_pi_chr_(?P<chrom>.+)_start_(?P<start>\d+)_end_(?P<end>\d+)_group_(?P<grp>[01])$")


def consensus_loci():
    """(chrom, start, end) for loci with a consensus recurrence call.

    The log's "Overall (All Consensus 0+1)" line is restricted to these, which is
    why it reports 60 contributing series where the raw track file has 102.
    """
    path = os.path.join(REPO, "data", "inv_properties.tsv")
    keep = set()
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if (r.get("0_single_1_recur_consensus") or "").strip() in ("0", "1"):
                chrom = str(r["Chromosome"]).replace("chr", "")
                keep.add((chrom, int(float(r["Start"])), int(float(r["End"]))))
    return keep


def load_series(restrict=None):
    """Per (locus, orientation) series of per-bin values, folded to both edges.

    Distance for site i is min(i, L-1-i), so both breakpoints map to 0 -- the
    same folding `_distance_fold_binning` uses. Only loci longer than 100 kbp
    contribute, matching the Methods.
    """
    out = []
    with gzip.open(FALSTA, "rt") as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            values_line = fh.readline()
            if not values_line:
                break
            m = HEADER_RE.match(header.strip())
            if not m:
                continue
            start, end = int(m.group("start")), int(m.group("end"))
            if end - start + 1 <= MAX_DIST:
                continue
            if restrict is not None and (m.group("chrom"), start, end) not in restrict:
                continue
            v = np.fromstring(values_line.strip(), sep=",")
            if v.size == 0:
                continue
            idx = np.arange(v.size)
            dist = np.minimum(idx, v.size - 1 - idx)
            keep = dist < MAX_DIST
            if not keep.any():
                continue
            bin_id = (dist[keep] // BIN_SIZE).astype(int)
            vals = v[keep]
            means = np.full(N_BINS, np.nan)
            medians = np.full(N_BINS, np.nan)
            for b in range(N_BINS):
                sel = vals[bin_id == b]
                sel = sel[np.isfinite(sel)]
                if sel.size:
                    means[b] = sel.mean()
                    medians[b] = np.median(sel)
            out.append({
                "locus": f"chr{m.group('chrom')}:{start}-{end}",
                "group": int(m.group("grp")),
                "mean": means, "median": medians,
            })
    return out


def spearman_over_bins(series, within, across):
    mat = np.vstack([s[within] for s in series])
    counts = np.sum(np.isfinite(mat), axis=0)
    agg = (np.nanmedian if across == "median" else np.nanmean)(mat, axis=0)
    agg = np.where(counts >= MIN_INV_PER_BIN, agg, np.nan)
    dist = np.arange(0, MAX_DIST, BIN_SIZE, dtype=float)
    mask = np.isfinite(agg)
    if mask.sum() < 3:
        return None
    rho, p = stats.spearmanr(dist[mask], agg[mask])
    return dict(rho=float(rho), p=float(p), n_bins=int(mask.sum()),
                n_series=len(series))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    keep = consensus_loci()
    series = load_series(restrict=keep)
    if not series:
        raise SystemExit(f"no qualifying loci parsed from {FALSTA}")
    print(f"{len(series)} (locus x orientation) series from loci > {MAX_DIST//1000} kbp\n")

    rows = []
    for within in ("mean", "median"):
        for across in ("median", "mean"):
            r = spearman_over_bins(series, within, across)
            if r is None:
                continue
            label = f"within-locus {within}, across-locus {across}"
            matches = []
            if abs(r["rho"] - 0.500) < 5e-3:
                matches.append("replication log (0.500)")
            if abs(r["rho"] - 0.451) < 5e-3:
                matches.append("manuscript (0.451)")
            rows.append(dict(within_locus=within, across_locus=across,
                             rho=round(r["rho"], 6), p_value=f"{r['p']:.6g}",
                             n_bins=r["n_bins"], n_series=r["n_series"],
                             matches="; ".join(matches)))
            print(f"  {label:44s} rho = {r['rho']:+.4f}  p = {r['p']:.3g}  "
                  f"bins = {r['n_bins']}" + (f"   <-- {'; '.join(matches)}" if matches else ""))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")
    print("\nNote: n in the replication log is the number of contributing series, not the\n"
          "correlation's sample size. The test is over the bins column above.")


if __name__ == "__main__":
    main()
