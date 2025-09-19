import sys
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg

FNAME = "output.csv"
ALTERNATIVE = "two-sided"

# ----------------------------- Utilities -----------------------------

def nf(x, digits=6):
    # Formatter that avoids "−0" and handles tiny/large values.
    def _scalar(v):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "NA"
        # squash negative zero
        if isinstance(v, (int, float, np.floating)):
            if abs(v) < 1e-15:
                v = 0.0
        if isinstance(v, (int, float, np.floating)):
            if abs(v) > 1e4 or (0 < abs(v) < 1e-4):
                return f"{float(v):.3e}"
            return f"{float(v):.{digits}g}"
        return str(v)

    if isinstance(x, (list, tuple, np.ndarray)) and np.size(x) == 2:
        a, b = x
        return f"[{_scalar(float(a))},{_scalar(float(b))}]"
    return _scalar(x)

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
    counts = df.groupby(key, dropna=False).size()
    n_dups = int((counts > 1).sum())
    if n_dups > 0:
        df = (df.groupby(key, dropna=False)
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

# Group means/medians
mean_inv, mean_dir = float(np.mean(x)), float(np.mean(y))
median_inv, median_dir = float(np.median(x)), float(np.median(y))
mean_diff = float(np.mean(diff))
median_diff = float(np.median(diff))

# ----------------------------- Tests -----------------------------

# MEDIAN-BASED / LOCATION tests
# 1) Exact Sign test (tests median of differences = 0; ties removed) — SciPy
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
    p_sign = np.nan  # zero informative pairs

# 2) Wilcoxon signed-rank (tests symmetric location shift; equals median test if symmetric) — SciPy
wilc = stats.wilcoxon(x, y, zero_method="pratt", alternative=ALTERNATIVE, nan_policy="omit")
# n used (non-zero diffs); pulled from data to avoid version quirks
n_wilcox_used = int((diff != 0).sum())

# MEAN-BASED tests
# 3) Paired t-test — SciPy
tt_sc = stats.ttest_rel(x, y, alternative=ALTERNATIVE, nan_policy="omit")
dof_sc = int(np.count_nonzero(~np.isnan(diff)) - 1)

# 4) Paired t-test — Pingouin (with 95% CI & Cohen's dz; all library-computed)
tt_pg = pg.ttest(x, y, paired=True, alternative=ALTERNATIVE)
t_pg = float(tt_pg["T"].iloc[0])
dof_pg = float(tt_pg["dof"].iloc[0])
p_pg = float(tt_pg["p-val"].iloc[0])
ci_low, ci_high = map(float, np.asarray(tt_pg["CI95%"].iloc[0]).tolist())
dz = float(pg.compute_effsize(x, y, paired=True, eftype="cohen"))

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
print(f"group_means: direct={nf(mean_dir)}  inverted={nf(mean_inv)}")
print(f"group_medians: direct={nf(median_dir)}  inverted={nf(median_inv)}")
print(f"mean_difference_inv_minus_dir: {nf(mean_diff)}")
print(f"median_difference_inv_minus_dir: {nf(median_diff)}")
print()

print("MEDIAN-BASED / LOCATION TESTS")
print("-" * 72)
print("Exact sign test (median of differences) — SciPy")
print(f"  n_nonzero={n_sign}  k_positive={k_pos}  prop_positive={nf(prop_pos)}  p={nf(p_sign)}")
print()
print("Wilcoxon signed-rank (location shift; median if symmetric) — SciPy")
print(f"  W={nf(wilc.statistic)}  p={nf(wilc.pvalue)}  n_used={n_wilcox_used}")
print()

print("MEAN-BASED TESTS")
print("-" * 72)
print("Paired t-test — SciPy (tests mean of differences)")
print(f"  t={nf(tt_sc.statistic)}  dof={dof_sc}  p={nf(tt_sc.pvalue)}")
print()
print("Paired t-test — Pingouin (tests mean of differences)")
print(f"  t={nf(t_pg)}  dof={nf(dof_pg)}  p={nf(p_pg)}  CI95%={nf((ci_low, ci_high))}  cohen_dz={nf(dz)}")
print(sep)
