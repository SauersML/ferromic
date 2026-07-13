"""Per-inversion pop-gen feature extractor for the recurrence classifier.

Input for every inversion (real or simulated):
  G      : (n_hap, n_sites) uint8 0/1 haplotype x biallelic-site matrix
  labels : (n_hap,) int in {0 = direct orientation, 1 = inverted orientation}

Output: an ordered dict of scalar features. The same extractor runs on msprime
simulations (where the single/recurrent label is known ground truth) and on the
real HGSVC haplotypes, so the logistic classifier fit on simulations transfers to
the real inversions.

Feature list (the Porubsky et al. recurrence signals):
  tree_n_events        NJ + Fitch parsimony minimum # of orientation state changes
  log_tree_events      log1p of the above
  one_minus_best_r2    1 - max over sites of r^2(site, orientation)   (tag-SNP; tiSNP)
  one_minus_top3_r2    1 - mean of the top-3 site r^2 with orientation
  log_pi_ratio         log(pi_inverted / pi_direct)      (floored)
  log_pi_inv           log(pi_inverted)                  (floored)
  disp_ratio_inv       inverted-lineage dispersion: mean within-inverted pairwise
                       distance / mean between-orientation pairwise distance
  pdist_sd             sd of all normalized pairwise distances
  pdist_excess         (mean between-group - mean within-group) / mean overall
  log_pdist_ratio      log(mean within-inverted / mean within-direct pairwise distance)
  pdist_q80            80th percentile of within-inverted normalized pairwise distance
  hudson_fst           Hudson's FST between orientation groups
  log_n_sites          log1p of # segregating sites

All definitions are deterministic. Pairwise distances are Hamming counts divided
by n_sites (per-site) unless noted; pi is per-site nucleotide diversity.
"""
from __future__ import annotations

import numpy as np

# reuse the NJ + Fitch parsimony (tree #events); see recurrence/parsimony.py
from .parsimony import classify as _tree_events

PI_FLOOR = 1e-6
FEATURE_NAMES = [
    "tree_n_events", "log_tree_events",
    "one_minus_best_r2", "one_minus_top3_r2",
    "log_pi_ratio", "log_pi_inv",
    "disp_ratio_inv",
    "pdist_sd", "pdist_excess", "log_pdist_ratio", "pdist_q80",
    "hudson_fst", "log_n_sites",
]


def _pi_within(G_sub):
    """Mean per-segregating-site heterozygosity within a set of haplotypes.
    G_sub: (m, s) 0/1. (Used by the full 13-feature classifier.)"""
    m, s = G_sub.shape
    if m < 2 or s == 0:
        return 0.0
    c1 = G_sub.sum(axis=0).astype(np.float64)
    c0 = m - c1
    site_pairs = c0 * c1
    denom = m * (m - 1) / 2.0
    return float(site_pairs.sum() / denom / s)


def _pi_perbp(G_sub, seq_len):
    """Nucleotide diversity per base pair (sum pairwise diffs / n_pairs / seq_len),
    matching ferromic output.csv. G_sub: (m,s) 0/1 over segregating sites in seq_len."""
    m, s = G_sub.shape
    if m < 2:
        return 0.0
    c1 = G_sub.sum(axis=0).astype(np.float64)
    c0 = m - c1
    denom = m * (m - 1) / 2.0
    return float((c0 * c1).sum() / denom / seq_len)


def _theta_perbp(n_hap, n_seg, seq_len):
    """Watterson's theta per bp = S / a_n / seq_len, a_n = sum_{i=1}^{n-1} 1/i."""
    if n_hap < 2 or seq_len <= 0:
        return 0.0
    a_n = float(np.sum(1.0 / np.arange(1, n_hap)))
    return n_seg / a_n / seq_len


def popgen_group_stats(G, labels, seq_len):
    """Per-bp diversity/differentiation stats matching ferromic output.csv columns,
    so the transferable pop-gen classifier uses identical definitions on sims and
    real data. Returns dict with pi_inv/pi_dir/theta_inv/theta_dir/dxy/fst/seg/freq."""
    G = np.asarray(G, dtype=np.uint8)
    labels = np.asarray(labels)
    inv = G[labels == 1]
    dir_ = G[labels == 0]
    n1, n0 = inv.shape[0], dir_.shape[0]
    # segregating within each group = sites polymorphic in that group
    seg_inv = int(((inv.sum(0) > 0) & (inv.sum(0) < n1)).sum()) if n1 >= 2 else 0
    seg_dir = int(((dir_.sum(0) > 0) & (dir_.sum(0) < n0)).sum()) if n0 >= 2 else 0
    pi_inv = _pi_perbp(inv, seq_len)
    pi_dir = _pi_perbp(dir_, seq_len)
    # Hudson dxy per bp = mean between-group per-site diff prob / bp
    dxy = np.nan
    if n1 >= 1 and n0 >= 1 and G.shape[1] > 0:
        p1 = inv.mean(0)
        p0 = dir_.mean(0)
        dxy = float((p1 * (1 - p0) + p0 * (1 - p1)).sum() / seq_len)
    return dict(
        pi_inv_bp=pi_inv, pi_dir_bp=pi_dir,
        theta_inv_bp=_theta_perbp(n1, seg_inv, seq_len),
        theta_dir_bp=_theta_perbp(n0, seg_dir, seq_len),
        dxy_bp=dxy, fst=_hudson_fst(G, labels),
        seg_inv=seg_inv, seg_dir=seg_dir,
        n_inv=n1, n_dir=n0, inv_freq=(n1 / (n1 + n0)) if (n1 + n0) else np.nan,
    )


def _hamming_full(G):
    """(n,n) Hamming count matrix."""
    n = G.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    Gi = G.astype(np.int16)
    for i in range(n):
        D[i] = (Gi[i] != Gi).sum(axis=1)
    return D


def _hudson_fst(G, labels):
    """Hudson's FST between orientation groups (Bhatia et al. 2013 ratio-of-averages)."""
    inv = G[labels == 1]
    dir_ = G[labels == 0]
    n1, n2 = inv.shape[0], dir_.shape[0]
    if n1 < 2 or n2 < 2 or G.shape[1] == 0:
        return np.nan
    p1 = inv.mean(axis=0)
    p2 = dir_.mean(axis=0)
    # per-site numerator/denominator of Hudson FST
    num = (p1 - p2) ** 2 - p1 * (1 - p1) / (n1 - 1) - p2 * (1 - p2) / (n2 - 1)
    den = p1 * (1 - p2) + p2 * (1 - p1)
    ds = den.sum()
    if ds <= 0:
        return np.nan
    return float(num.sum() / ds)


def extract_features(G, labels):
    """Return dict of features. G:(n,s) 0/1, labels:(n,) {0,1}."""
    G = np.asarray(G, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.int64)
    n, s = G.shape
    inv_mask = labels == 1
    dir_mask = labels == 0
    n_inv, n_dir = int(inv_mask.sum()), int(dir_mask.sum())

    f = {k: np.nan for k in FEATURE_NAMES}

    # tree #events (topology) -- always computable
    ev = float(_tree_events(G, labels))
    f["tree_n_events"] = ev
    f["log_tree_events"] = float(np.log1p(ev))
    f["log_n_sites"] = float(np.log1p(s))

    if s == 0 or n_inv < 2 or n_dir < 2:
        return f  # degenerate; leave rest NaN (imputed downstream)

    # tag-SNP r^2 with orientation
    y = inv_mask.astype(np.float64)
    yc = y - y.mean()
    yss = (yc ** 2).sum()
    Gf = G.astype(np.float64)
    gm = Gf.mean(axis=0)
    Gc = Gf - gm
    gss = (Gc ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = (Gc * yc[:, None]).sum(axis=0)
        r2 = np.where((gss > 0) & (yss > 0), (cov ** 2) / (gss * yss), 0.0)
    r2 = np.nan_to_num(r2, nan=0.0)
    r2_sorted = np.sort(r2)[::-1]
    best = float(r2_sorted[0])
    top3 = float(r2_sorted[:3].mean())
    f["one_minus_best_r2"] = 1.0 - best
    f["one_minus_top3_r2"] = 1.0 - top3

    # pi (per-site) within each orientation
    pi_inv = _pi_within(G[inv_mask])
    pi_dir = _pi_within(G[dir_mask])
    pi_inv_fl = max(pi_inv, PI_FLOOR)
    pi_dir_fl = max(pi_dir, PI_FLOOR)
    f["log_pi_inv"] = float(np.log(pi_inv_fl))
    f["log_pi_ratio"] = float(np.log(pi_inv_fl / pi_dir_fl))

    # pairwise distance structure (per-site normalized)
    D = _hamming_full(G) / s
    iu = np.triu_indices(n, k=1)
    all_d = D[iu]
    f["pdist_sd"] = float(all_d.std())
    # within-inverted, within-direct, between
    inv_idx = np.where(inv_mask)[0]
    dir_idx = np.where(dir_mask)[0]
    win_inv = D[np.ix_(inv_idx, inv_idx)][np.triu_indices(n_inv, 1)]
    win_dir = D[np.ix_(dir_idx, dir_idx)][np.triu_indices(n_dir, 1)]
    between = D[np.ix_(inv_idx, dir_idx)].ravel()
    m_win = np.concatenate([win_inv, win_dir]).mean() if (win_inv.size + win_dir.size) else np.nan
    m_bet = between.mean()
    m_all = all_d.mean()
    f["pdist_excess"] = float((m_bet - m_win) / m_all) if m_all > 0 else np.nan
    if win_dir.size and win_dir.mean() > 0 and win_inv.mean() > 0:
        f["log_pdist_ratio"] = float(np.log(win_inv.mean() / win_dir.mean()))
    else:
        f["log_pdist_ratio"] = np.nan
    f["pdist_q80"] = float(np.quantile(win_inv, 0.80)) if win_inv.size else np.nan
    # inverted-lineage dispersion: within-inverted spread relative to between-group
    f["disp_ratio_inv"] = float(win_inv.mean() / m_bet) if (win_inv.size and m_bet > 0) else np.nan

    # Hudson FST
    f["hudson_fst"] = _hudson_fst(G, labels)
    return f
