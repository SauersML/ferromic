"""
Directional distributional tests for frf_delta between single-origin and recurrent inversions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# File paths
FRF_PATH = Path("per_inversion_breakpoint_tests/per_inversion_breakpoint_test_results.tsv")
INV_PATH = Path("inv_properties.tsv")

# Column names
CHR_COL_FRF = "chrom"
START_COL_FRF = "start"
END_COL_FRF = "end"
DELTA_COL = "frf_delta"

CHR_COL_INV = "Chromosome"
START_COL_INV = "Start"
END_COL_INV = "End"
STATUS_COL = "0_single_1_recur_consensus"

N_PERMUTATIONS = 20000
RANDOM_SEED = 123


def normalize_chromosome(series):
    """Normalize chromosome names for matching."""
    return (
        series
        .astype(str)
        .str.lower()
        .str.replace("^chr_?", "", regex=True)
        .str.strip()
    )


def load_and_merge_data():
    """Load FRF results and inversion properties, merge on coordinates."""
    frf = pd.read_csv(FRF_PATH, sep="\t")
    inv = pd.read_csv(INV_PATH, sep="\t")
    
    # Normalize chromosome names
    frf["chrom_norm"] = normalize_chromosome(frf[CHR_COL_FRF])
    inv["chrom_norm"] = normalize_chromosome(inv[CHR_COL_INV])
    
    # Merge on coordinates
    merged = pd.merge(
        frf,
        inv[[CHR_COL_INV, START_COL_INV, END_COL_INV, STATUS_COL, "chrom_norm"]],
        left_on=["chrom_norm", START_COL_FRF, END_COL_FRF],
        right_on=["chrom_norm", START_COL_INV, END_COL_INV],
        how="inner",
        validate="1:1"
    )
    
    # Filter for valid status and delta values
    merged = merged[merged[STATUS_COL].notna()]
    merged = merged[merged[STATUS_COL].isin([0, 1])]
    merged = merged[np.isfinite(merged[DELTA_COL])]
    
    return merged


def extract_groups(merged):
    """Extract frf_delta values for each group."""
    group0 = merged.loc[merged[STATUS_COL] == 0, DELTA_COL].to_numpy()
    group1 = merged.loc[merged[STATUS_COL] == 1, DELTA_COL].to_numpy()
    
    return group0, group1


# ============================================================================
# Directional Anderson-Darling Test
# ============================================================================

def directional_ad_statistic(x, y):
    """Compute directional Anderson-Darling statistic."""
    pooled = np.concatenate([x, y])
    pooled_sorted = np.sort(pooled)
    
    nx = len(x)
    ny = len(y)
    n = nx + ny
    
    # Compute empirical CDFs at each unique pooled value
    fx = np.searchsorted(np.sort(x), pooled_sorted, side='right') / nx
    fy = np.searchsorted(np.sort(y), pooled_sorted, side='right') / ny
    
    # Pooled empirical CDF
    h = np.arange(1, n + 1) / n
    
    # Anderson-Darling weight
    weight = np.where((h > 0) & (h < 1), 1.0 / (h * (1 - h)), 0.0)
    
    # Directional: keep only where fx < fy
    diff_squared = np.where(fx < fy, (fx - fy) ** 2, 0.0)
    
    stat = np.sum(weight * diff_squared) * (nx * ny / n)
    
    return stat


def directional_ad_test(x, y, n_perm=20000, random_state=123):
    """Directional Anderson-Darling test via permutation."""
    rng = np.random.RandomState(random_state)
    
    stat_obs_0gt1 = directional_ad_statistic(x, y)
    stat_obs_1gt0 = directional_ad_statistic(y, x)
    
    nx = len(x)
    ny = len(y)
    data = np.concatenate([x, y])
    labels = np.array([0] * nx + [1] * ny)
    
    perm_stats_0gt1 = np.empty(n_perm)
    perm_stats_1gt0 = np.empty(n_perm)
    
    for i in range(n_perm):
        rng.shuffle(labels)
        a = data[labels == 0]
        b = data[labels == 1]
        perm_stats_0gt1[i] = directional_ad_statistic(a, b)
        perm_stats_1gt0[i] = directional_ad_statistic(b, a)
    
    p_0gt1 = (perm_stats_0gt1 >= stat_obs_0gt1).mean()
    p_1gt0 = (perm_stats_1gt0 >= stat_obs_1gt0).mean()
    
    return {
        "statistic_0gt1": float(stat_obs_0gt1),
        "p_value_0gt1": float(p_0gt1),
        "statistic_1gt0": float(stat_obs_1gt0),
        "p_value_1gt0": float(p_1gt0),
    }


# ============================================================================
# Directional Energy Distance Test
# ============================================================================

def energy_distance(x, y):
    """Compute energy distance between two samples."""
    x = x.ravel()
    y = y.ravel()
    
    xy = np.abs(x[:, None] - y[None, :])
    xx = np.abs(x[:, None] - x[None, :])
    yy = np.abs(y[:, None] - y[None, :])
    
    return 2.0 * xy.mean() - xx.mean() - yy.mean()


def directional_energy_test(x, y, n_perm=20000, random_state=123):
    """Directional energy distance test via signed statistic."""
    rng = np.random.RandomState(random_state)
    
    nx = x.size
    ny = y.size
    data = np.concatenate([x, y])
    labels = np.array([0] * nx + [1] * ny)
    
    def signed_energy(a, b):
        d = energy_distance(a, b)
        
        aa, bb = np.meshgrid(a, b, indexing="ij")
        p_a_gt_b = (aa > bb).mean()
        p_b_gt_a = (bb > aa).mean()
        
        if p_a_gt_b > p_b_gt_a:
            return d
        if p_b_gt_a > p_a_gt_b:
            return -d
        return 0.0
    
    t_obs = signed_energy(x, y)
    
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(labels)
        a = data[labels == 0]
        b = data[labels == 1]
        perm_stats[i] = signed_energy(a, b)
    
    p_0gt1 = (perm_stats >= t_obs).mean()
    p_1gt0 = (perm_stats <= t_obs).mean()
    
    return {
        "statistic": float(t_obs),
        "p_value_0gt1": float(p_0gt1),
        "p_value_1gt0": float(p_1gt0),
    }


# ============================================================================
# Directional Kolmogorov-Smirnov Test
# ============================================================================

def ecdf(values, points):
    """Compute empirical CDF of values at given points."""
    values_sorted = np.sort(values)
    counts = np.searchsorted(values_sorted, points, side="right")
    return counts / len(values_sorted)


def directional_ks_test(x, y, n_perm=20000, random_state=123):
    """One-sided Kolmogorov-Smirnov test via permutation."""
    rng = np.random.RandomState(random_state)
    
    nx = x.size
    ny = y.size
    data = np.concatenate([x, y])
    labels = np.array([0] * nx + [1] * ny)
    
    grid = np.unique(data)
    
    def one_sided_stats(a, b):
        fa = ecdf(a, grid)
        fb = ecdf(b, grid)
        d_plus = np.max(fb - fa)
        d_minus = np.max(fa - fb)
        return d_plus, d_minus
    
    d_plus_obs, d_minus_obs = one_sided_stats(x, y)
    
    perm_d_plus = np.empty(n_perm)
    perm_d_minus = np.empty(n_perm)
    
    for i in range(n_perm):
        rng.shuffle(labels)
        a = data[labels == 0]
        b = data[labels == 1]
        d_plus, d_minus = one_sided_stats(a, b)
        perm_d_plus[i] = d_plus
        perm_d_minus[i] = d_minus
    
    p_0gt1 = (perm_d_plus >= d_plus_obs).mean()
    p_1gt0 = (perm_d_minus >= d_minus_obs).mean()
    
    return {
        "statistic_0gt1": float(d_plus_obs),
        "p_value_0gt1": float(p_0gt1),
        "statistic_1gt0": float(d_minus_obs),
        "p_value_1gt0": float(p_1gt0),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    merged = load_and_merge_data()
    group0, group1 = extract_groups(merged)
    
    print(f"n_group0={len(group0)}")
    print(f"n_group1={len(group1)}")
    print(f"mean_group0={np.mean(group0):.6f}")
    print(f"mean_group1={np.mean(group1):.6f}")
    print(f"median_group0={np.median(group0):.6f}")
    print(f"median_group1={np.median(group1):.6f}")
    print()
    
    ad_results = directional_ad_test(group0, group1, n_perm=N_PERMUTATIONS, random_state=RANDOM_SEED)
    energy_results = directional_energy_test(group0, group1, n_perm=N_PERMUTATIONS, random_state=RANDOM_SEED)
    ks_results = directional_ks_test(group0, group1, n_perm=N_PERMUTATIONS, random_state=RANDOM_SEED)
    
    print("anderson_darling_stat_0gt1={:.4f}".format(ad_results['statistic_0gt1']))
    print("anderson_darling_p_0gt1={:.4f}".format(ad_results['p_value_0gt1']))
    print("anderson_darling_stat_1gt0={:.4f}".format(ad_results['statistic_1gt0']))
    print("anderson_darling_p_1gt0={:.4f}".format(ad_results['p_value_1gt0']))
    print()
    
    print("energy_stat={:.6f}".format(energy_results['statistic']))
    print("energy_p_0gt1={:.4f}".format(energy_results['p_value_0gt1']))
    print("energy_p_1gt0={:.4f}".format(energy_results['p_value_1gt0']))
    print()
    
    print("ks_stat_0gt1={:.4f}".format(ks_results['statistic_0gt1']))
    print("ks_p_0gt1={:.4f}".format(ks_results['p_value_0gt1']))
    print("ks_stat_1gt0={:.4f}".format(ks_results['statistic_1gt0']))
    print("ks_p_1gt0={:.4f}".format(ks_results['p_value_1gt0']))


if __name__ == "__main__":
    main()
