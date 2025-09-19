import sys
import re
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg

FNAME = "output.csv"
INVINFO = "inv_info.tsv"
ALTERNATIVE = "two-sided"  # fixed

# ----------------------------- Utilities -----------------------------

def nf(x, digits=6):
    # Formatter that avoids "−0" and handles tiny/large values.
    def _scalar(v):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "NA"
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

def norm_chr(v):
    # Normalize chromosome labels by stripping leading 'chr' (case-insensitive)
    # and returning the remainder as a string (e.g., "chr7" -> "7").
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    s = re.sub(r'^chr', '', s, flags=re.IGNORECASE)
    return s

def aggregate_duplicates_by_key(df, key_cols, value_cols):
    counts = df.groupby(key_cols, dropna=False).size()
    n_dups = int((counts > 1).sum())
    if n_dups > 0:
        agg_map = {c: "mean" for c in value_cols}
        df = df.groupby(key_cols, dropna=False, as_index=False).agg(agg_map)
    return df, n_dups

# ----------------------------- Load & prepare inputs -----------------------------

# Load stats CSV
try:
    df = pd.read_csv(FNAME, na_values=["NA", "NaN", ""], low_memory=False)
except Exception as e:
    print(f"ERROR: Could not read {FNAME}: {e}")
    sys.exit(1)

required = ["chr", "region_start", "region_end", "0_pi_filtered", "1_pi_filtered"]
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"ERROR: Missing required columns in {FNAME}: {missing}")
    sys.exit(1)

# Load inversion metadata TSV
try:
    inv = pd.read_csv(INVINFO, sep="\t", na_values=["NA", "NaN", ""], low_memory=False)
except Exception as e:
    print(f"ERROR: Could not read {INVINFO}: {e}")
    sys.exit(1)

inv_required = ["Chromosome", "Start", "End", "0_single_1_recur_consensus"]
missing_inv = [c for c in inv_required if c not in inv.columns]
if missing_inv:
    print(f"ERROR: Missing required columns in {INVINFO}: {missing_inv}")
    sys.exit(1)

# Keep only rows with consensus exactly 0 or 1 (exclude NA/other)
inv["0_single_1_recur_consensus"] = pd.to_numeric(inv["0_single_1_recur_consensus"], errors="coerce")
inv = inv[inv["0_single_1_recur_consensus"].isin([0, 1])].copy()

# ----------------------------- Harmonize keys -----------------------------

# Prepare CSV keys
df_before = len(df)
df = df.rename(columns={"chr": "chr_raw"})
df["chr_key"] = df["chr_raw"].apply(norm_chr)
df["region_start"] = pd.to_numeric(df["region_start"], errors="coerce")
df["region_end"] = pd.to_numeric(df["region_end"], errors="coerce")

# Clean CSV rows
df = df.dropna(subset=["chr_key", "region_start", "region_end", "0_pi_filtered", "1_pi_filtered"])
df = df[(df["region_end"] > df["region_start"])]
df_after = len(df)

# Subset of CSV used for analysis
sub = (df[["chr_key", "chr_raw", "region_start", "region_end", "0_pi_filtered", "1_pi_filtered"]]
       .rename(columns={"0_pi_filtered": "pi_direct", "1_pi_filtered": "pi_inverted"}))

# Aggregate duplicates in CSV on exact region coordinates (if any)
sub, n_dups_csv = aggregate_duplicates_by_key(
    sub,
    key_cols=["chr_key", "region_start", "region_end"],
    value_cols=["pi_direct", "pi_inverted"]
)

# Prepare TSV keys
inv["chr_key"] = inv["Chromosome"].apply(norm_chr)
inv["Start"] = pd.to_numeric(inv["Start"], errors="coerce")
inv["End"] = pd.to_numeric(inv["End"], errors="coerce")
inv = inv.dropna(subset=["chr_key", "Start", "End"])

# If the TSV has duplicate (chr, start, end) keys that would match CSV, fail
inv_key_counts = inv.groupby(["chr_key", "Start", "End"], dropna=False).size().reset_index(name="n")
dup_keys = inv_key_counts[inv_key_counts["n"] > 1]
if not dup_keys.empty:
    # See if any duplicates would intersect CSV regions
    probe = dup_keys.merge(
        sub[["chr_key", "region_start", "region_end"]],
        left_on=["chr_key", "Start", "End"],
        right_on=["chr_key", "region_start", "region_end"],
        how="inner"
    )
    if not probe.empty:
        examples = probe[["chr_key", "Start", "End"]].drop_duplicates().head(5).to_dict(orient="records")
        print("ERROR: Multiple rows in inv_info.tsv share the same (chr,Start,End) that match CSV regions. Examples:", examples)
        sys.exit(1)

# ----------------------------- Intersect by exact coordinates -----------------------------

# Keep only regions present in BOTH files (intersection) AND with consensus ∈ {0,1}
m = pd.merge(
    sub,
    inv[["chr_key", "Start", "End", "0_single_1_recur_consensus"]],
    left_on=["chr_key", "region_start", "region_end"],
    right_on=["chr_key", "Start", "End"],
    how="inner",
    validate="m:1"  # each CSV region maps to at most one TSV row
)

# Extra safety: ensure no CSV region got multiple TSV matches (older pandas)
dup_after_merge = m.duplicated(subset=["chr_key", "region_start", "region_end"], keep=False)
if dup_after_merge.any():
    conflict = m.loc[dup_after_merge, ["chr_key", "region_start", "region_end"]].drop_duplicates().head(5).to_dict(orient="records")
    print("ERROR: More than one TSV row matched a CSV region. Conflicts:", conflict)
    sys.exit(1)

n_common = len(m)

# ----------------------------- Analysis data -----------------------------

x = m["pi_inverted"].to_numpy(dtype=float)  # inverted (group 1)
y = m["pi_direct"].to_numpy(dtype=float)    # direct (group 0)
diff = x - y

n_regions = len(m)
n_ties = int((diff == 0).sum())
n_pos = int((diff > 0).sum())
n_neg = int((diff < 0).sum())

# Group means/medians
mean_inv, mean_dir = float(np.mean(x)), float(np.mean(y))
median_inv, median_dir = float(np.median(x)), float(np.median(y))
mean_diff = float(np.mean(diff))
median_diff = float(np.median(diff))

# ----------------------------- Tests (libraries only) -----------------------------

# MEDIAN-BASED / LOCATION tests
# 1) Exact Sign test (median of differences = 0; ties removed) — SciPy
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
    p_sign = np.nan  # no informative pairs

# 2) Wilcoxon signed-rank (symmetric location shift; median if symmetric) — SciPy
wilc = stats.wilcoxon(x, y, zero_method="pratt", alternative=ALTERNATIVE, nan_policy="omit")
n_wilcox_used = int((diff != 0).sum())

# MEAN-BASED tests
# 3) Paired t-test — SciPy
tt_sc = stats.ttest_rel(x, y, alternative=ALTERNATIVE, nan_policy="omit")
dof_sc = int(np.count_nonzero(~np.isnan(diff)) - 1)

# 4) Paired t-test — Pingouin (with 95% CI & Cohen's dz)
tt_pg = pg.ttest(x, y, paired=True, alternative=ALTERNATIVE)
t_pg = float(tt_pg["T"].iloc[0])
dof_pg = float(tt_pg["dof"].iloc[0])
p_pg = float(tt_pg["p-val"].iloc[0])
ci_low, ci_high = map(float, np.asarray(tt_pg["CI95%"].iloc[0]).tolist())
dz = float(pg.compute_effsize(x, y, paired=True, eftype="cohen"))

# ----------------------------- Output (data only) -----------------------------

sep = "=" * 72
print(sep)
print("Paired tests on filtered π (inverted − direct) — RESTRICTED to inv_info.tsv consensus ∈ {0,1}")
print(sep)
print(f"file_stats_csv: {FNAME}")
print(f"file_invinfo_tsv: {INVINFO}")
print(f"csv_rows_initial: {df_before}")
print(f"csv_rows_after_clean: {df_after}")
print(f"csv_duplicates_aggregated_groups: {n_dups_csv}")
print(f"tsv_rows_with_consensus_0or1: {len(inv)}")
print(f"regions_in_intersection: {n_common}")
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
print(f"  W={nf(wilc.statistic)}  p={nf(wilc.pvalue)}  n_used={n_wilcox_used}  ties={n_ties}  pos={n_pos}  neg={n_neg}")
print()

print("MEAN-BASED TESTS")
print("-" * 72)
print("Paired t-test — SciPy (tests mean of differences)")
print(f"  t={nf(tt_sc.statistic)}  dof={dof_sc}  p={nf(tt_sc.pvalue)}")
print()
print("Paired t-test — Pingouin (tests mean of differences)")
print(f"  t={nf(t_pg)}  dof={nf(dof_pg)}  p={nf(p_pg)}  CI95%={nf((ci_low, ci_high))}  cohen_dz={nf(dz)}")
print(sep)
