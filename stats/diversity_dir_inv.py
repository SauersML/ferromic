import sys
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg

FNAME = "output.csv"
ALTERNATIVE = "two-sided"  # fixed

# ----------------------------- Utilities -----------------------------

def nf(x, digits=6):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "NA"
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return f"[{nf(x[0], digits)},{nf(x[1], digits)}]"
    if abs(x) > 1e4 or (0 < abs(x) < 1e-4):
        return f"{x:.3e}"
    return f"{x:.{digits}g}"

def summarize(s, name):
    s = pd.Series(s, dtype=float)
    return (
        f"{name}: "
        f"n={s.notna().sum()}  "
        f"median={nf(s.median())}  "
        f"mean={nf(s.mean())}  "
        f"sd={nf(s.std(ddof=1))}  "
        f"IQR=[{nf(s.quantile(0.25))},{nf(s.quantile(0.75))}]  "
        f"min={nf(s.min())}  "
        f"max={nf(s.max())}"
    )

def aggregate_duplicates(df):
    key = ["chr", "region_start", "region_end"]
    counts = df.groupby(key, dropna=False, observed=True).size()
    n_dups = int((counts > 1).sum())
    if n_dups > 0:
        df = (df.groupby(key, dropna=False, observed=True)
                .agg({"pi_inverted": "mean", "pi_direct": "mean"})
                .reset_index())
    return df, n_dups

# ----------------------------- Load & clean -----------------------------

try:
    df = pd.read_csv(FNAME, na_values=["NA", "NaN", ""], low_memory=False)
except Exception as e:
    print(f"ERROR: Could not read {FNAME}: {e}")
    sys.exit(1)

required = ["chr", "region_start", "region_end", "0_pi_filtered", "1_pi_filtered"]
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"ERROR: Missing required columns: {missing}")
    sys.exit(1)

sub = (df[["chr", "region_start", "region_end", "0_pi_filtered", "1_pi_filtered"]]
       .rename(columns={"0_pi_filtered": "pi_direct",
                        "1_pi_filtered": "pi_inverted"}))

n_rows_raw = len(sub)
sub = sub.dropna(subset=["pi_direct", "pi_inverted"])
sub = sub[(sub["region_end"] > sub["region_start"])]
n_rows_clean = len(sub)

sub, n_dups = aggregate_duplicates(sub)

sub = sub.sort_values(["chr", "region_start", "region_end"], kind="mergesort").reset_index(drop=True)

x = sub["pi_inverted"].to_numpy(dtype=float)  # inverted (group 1)
y = sub["pi_direct"].to_numpy(dtype=float)    # direct (group 0)
diff = x - y

n_regions = len(sub)
n_ties = int((diff == 0).sum())
n_pos = int((diff > 0).sum())
n_neg = int((diff < 0).sum())

# ----------------------------- Tests -----------------------------

# 1) Wilcoxon signed-rank (SciPy) + effect sizes (Pingouin)
wilc = stats.wilcoxon(x, y, zero_method="pratt", alternative=ALTERNATIVE, nan_policy="omit")
try:
    w_pg = pg.wilcoxon(x, y, alternative=ALTERNATIVE)
    rbc = float(w_pg["RBC"].iloc[0])
    cles = float(w_pg["CLES"].iloc[0])
    n_wilcox_used = int(w_pg["N"].iloc[0])
except Exception:
    rbc = np.nan
    cles = np.nan
    # approximate n used (non-zero diffs) for info only
    n_wilcox_used = int((diff != 0).sum())

# 2) Paired t-test (SciPy)
tt_sc = stats.ttest_rel(x, y, alternative=ALTERNATIVE, nan_policy="omit")
dof_sc = int(np.count_nonzero(~np.isnan(diff)) - 1)

# 3) Paired t-test (Pingouin) with CI & Cohen's dz
tt_pg = pg.ttest(x, y, paired=True, alternative=ALTERNATIVE)
t_pg = float(tt_pg["T"].iloc[0])
dof_pg = float(tt_pg["dof"].iloc[0])
p_pg = float(tt_pg["p-val"].iloc[0])
ci_pg = tuple(np.asarray(tt_pg["CI95%"].iloc[0]).tolist())  # (low, high)
dz = float(pg.compute_effsize(x, y, paired=True, eftype="cohen"))

# 4) Exact Sign test (binomial) on direction (ties removed)
nz = diff[diff != 0]
n_sign = nz.size
if n_sign > 0:
    k_pos = int((nz > 0).sum())
    prop_pos = k_pos / n_sign
    bt = stats.binomtest(k=k_pos, n=n_sign, p=0.5, alternative=ALTERNATIVE)
    p_sign = bt.pvalue
else:
    k_pos = 0
    prop_pos = np.nan
    p_sign = np.nan

# 5) Paired permutation test on mean difference (SciPy)
def stat_func(a, b):
    return np.mean(a - b)

try:
    perm = stats.permutation_test(
        data=(x, y),
        statistic=stat_func,
        permutation_type="paired",
        alternative=ALTERNATIVE,
        n_resamples=10000,
        vectorized=False,
        random_state=42
    )
    p_perm = float(perm.pvalue)
    stat_mean_diff = float(np.mean(diff))
except Exception:
    p_perm = np.nan
    stat_mean_diff = float(np.mean(diff))

# ----------------------------- Output (data only) -----------------------------

sep = "=" * 72
print(sep)
print("Paired tests on filtered π (inverted − direct)")
print(sep)
print(f"file: {FNAME}")
print(f"regions_used: {n_regions}")
print(f"dropped_rows: {n_rows_raw - n_rows_clean}")
print(f"duplicates_aggregated_groups: {n_dups}")
print(f"ties_count: {n_ties}")
print(f"positive_diffs: {n_pos}")
print(f"negative_diffs: {n_neg}")
print()

print("descriptives")
print("-" * 72)
print(summarize(y, "direct_group0_pi"))
print(summarize(x, "inverted_group1_pi"))
print(summarize(diff, "difference_inv_minus_dir"))
print()

print("wilcoxon_signed_rank_scipy")
print("-" * 72)
print(f"W={nf(wilc.statistic)}  p={nf(wilc.pvalue)}  n_used={n_wilcox_used}  "
      f"rank_biserial_r={nf(rbc)}  CLES={nf(cles)}")
print()

print("paired_t_test_scipy")
print("-" * 72)
print(f"t={nf(tt_sc.statistic)}  dof={dof_sc}  p={nf(tt_sc.pvalue)}")
print()

print("paired_t_test_pingouin")
print("-" * 72)
print(f"t={nf(t_pg)}  dof={nf(dof_pg)}  p={nf(p_pg)}  CI95%={nf(ci_pg)}  cohen_dz={nf(dz)}")
print()

print("exact_sign_test_binomial")
print("-" * 72)
print(f"n_nonzero={n_sign}  k_positive={k_pos}  prop_positive={nf(prop_pos)}  p={nf(p_sign)}")
print()

print("paired_permutation_test_mean_diff_scipy")
print("-" * 72)
print(f"mean_diff={nf(stat_mean_diff)}  p={nf(p_perm)}  resamples=10000")
print(sep)
