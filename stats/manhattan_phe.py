import os, re, sys, math, subprocess
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- config -----------------
INFILE = "phewas_results.tsv"
PHECODE_FILE = "phecodeX.csv"
OUTDIR = "phewas_plots"

PHENO_COL = "Phenotype"
P_COL = "P_Value"
BETA_COL = "Beta"
INV_COL = "Inversion"
SIG_COL = "Sig_Global"

ALPHA = 0.05              # for Bonferroni line
MAX_WIDTH = 18.0          # cap fig width in inches (keeps from being too wide)
MIN_WIDTH = 8.5           # minimum width
WIDTH_PER_100 = 0.25      # +0.25" per 100 phenotypes (gentle scaling)
FIG_HEIGHT = 6.0
POINT_SIZE = 10.0
ANNOTATE_TOP_N = 10       # top phenotypes by smallest p to label
UNCAT_NAME = "Uncategorized"

# ----------------- utilities -----------------

def canonicalize_name(s: str) -> str:
    """
    Make a robust, case/space-insensitive key:
    - replace underscores/hyphens with spaces
    - remove non-word characters
    - collapse spaces
    - lowercase
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^\w\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def pretty_text(s: str) -> str:
    """Underscores → spaces for display."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return str(s).replace("_", " ")

def truthy_series(s: pd.Series) -> pd.Series:
    """Interpret typical truthy forms."""
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})

def open_file(path: str) -> None:
    """Open a file using the OS default viewer."""
    try:
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            os.startfile(path)  # type: ignore
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass

def compute_width(n_points: int) -> float:
    """
    Keep plots compact. Width grows gently with sample size and is capped.
    """
    width = MIN_WIDTH + WIDTH_PER_100 * (n_points / 100.0)
    return float(max(MIN_WIDTH, min(MAX_WIDTH, width)))

def sanitize_filename(s: str) -> str:
    s = str(s) if s is not None else "NA"
    s = s.strip()
    s = re.sub(r"[^\w.\-]+", "_", s)
    return s[:200] if s else "NA"

# ----------------- category mapping -----------------

def load_category_map(phecode_csv: str) -> pd.DataFrame:
    """
    Build a mapping from phecode_string -> (category, category_num).
    If there are multiple rows per phecode_string, choose the most frequent (mode).
    """
    if not os.path.exists(phecode_csv):
        sys.exit(f"ERROR: Cannot find category file {phecode_csv}")

    pc = pd.read_csv(phecode_csv, dtype=str)
    needed = {"phecode_string", "phecode_category", "category_num"}
    if not needed.issubset(set(pc.columns)):
        sys.exit(f"ERROR: {phecode_csv} must contain columns: {sorted(needed)}")

    pc["clean_name"] = pc["phecode_string"].map(canonicalize_name)

    # For each clean_name, pick the most common category/category_num combo
    grp = pc.groupby("clean_name", dropna=False)[["phecode_category", "category_num"]]
    rows = []
    for key, sub in grp:
        # find mode of pair (category, category_num)
        pairs = list(zip(sub["phecode_category"], sub["category_num"]))
        if not pairs:
            continue
        most = Counter(pairs).most_common(1)[0][0]
        rows.append({"clean_name": key, "phecode_category": most[0], "category_num": most[1]})
    cmap = pd.DataFrame(rows)
    # ensure numeric sortability
    cmap["category_num_num"] = pd.to_numeric(cmap["category_num"], errors="coerce")
    return cmap

# ----------------- plotting core -----------------

def plot_one_inversion(df_group: pd.DataFrame, inversion_label: str) -> str | None:
    """
    Make a single PheWAS Manhattan per inversion.
    X-axis: phenotype index grouped by CATEGORY (ticks at category centers).
    Colors: alternate by CATEGORY.
    Shape: triangle-up for Beta >= 0, triangle-down for Beta < 0.
    Stars: outline star for Sig_Global hits.
    """
    g = df_group.copy()

    # numeric conversions
    g[P_COL] = pd.to_numeric(g[P_COL], errors="coerce")
    g[BETA_COL] = pd.to_numeric(g[BETA_COL], errors="coerce")

    # keep rows with phenotype + p-value
    g = g[g[PHENO_COL].notna() & g[P_COL].notna()]
    if g.empty:
        return None

    # avoid log10(0)
    tiny = np.nextafter(0, 1)
    g.loc[g[P_COL] <= 0, P_COL] = tiny

    # clean display text and category
    g["Phen_pretty"] = g[PHENO_COL].map(pretty_text)
    g["minuslog10p"] = -np.log10(g[P_COL])
    g["Beta_sign"] = np.where(g[BETA_COL].fillna(0) >= 0, "pos", "neg")

    # Order categories by category_num (numeric), then by name
    # (columns `phecode_category` and `category_num_num` were already merged in main())
    g["cat_name"] = g["phecode_category"].fillna(UNCAT_NAME)
    g["cat_num"]  = g["category_num_num"].fillna(9999)

    cat_order = (
        g[["cat_name", "cat_num"]]
        .drop_duplicates()
        .sort_values(["cat_num", "cat_name"], kind="mergesort")
        .reset_index(drop=True)
    )["cat_name"].tolist()

    # build x-positions per category, with centers for ticks
    x_positions = []
    centers = []
    ticklabels = []
    start = 0
    pieces = []
    for i, cat in enumerate(cat_order):
        block = g[g["cat_name"] == cat].sort_values([P_COL, "Phen_pretty"], kind="mergesort").copy()
        n = len(block)
        block["x"] = np.arange(start, start + n, dtype=float)
        pieces.append(block)
        centers.append(start + (n - 1) / 2.0)
        ticklabels.append(cat)
        start += n

    g = pd.concat(pieces, ignore_index=True)
    m = len(g)

    # Figure sizing—compact and capped
    fig_w = compute_width(m)
    fig, ax = plt.subplots(figsize=(fig_w, FIG_HEIGHT))

    # alternating colors per CATEGORY (C0/C1)
    # build a color for each row based on its category’s parity in cat_order
    cat_to_parity = {c: (idx % 2) for idx, c in enumerate(cat_order)}
    colors = np.where([cat_to_parity[c] == 0 for c in g["cat_name"]], "C0", "C1")

    # scatter, split by sign for triangle up/down
    pos_mask = g["Beta_sign"] == "pos"
    neg_mask = ~pos_mask

    if pos_mask.any():
        ax.scatter(
            g.loc[pos_mask, "x"], g.loc[pos_mask, "minuslog10p"],
            s=POINT_SIZE, marker="^", c=colors[pos_mask],
            edgecolors="none", alpha=0.9, zorder=2, label="β ≥ 0"
        )
    if neg_mask.any():
        ax.scatter(
            g.loc[neg_mask, "x"], g.loc[neg_mask, "minuslog10p"],
            s=POINT_SIZE, marker="v", c=colors[neg_mask],
            edgecolors="none", alpha=0.9, zorder=2, label="β < 0"
        )

    # Stars for Sig_Global hits (truthy)
    if SIG_COL in g.columns:
        sig = truthy_series(g[SIG_COL])
        if sig.any():
            ax.scatter(
                g.loc[sig, "x"], g.loc[sig, "minuslog10p"],
                s=80, marker="*", facecolors="none", edgecolors="black",
                linewidths=1.0, zorder=3, label="Sig_Global (0.05)"
            )

    # Bonferroni and nominal lines
    bonf_y = -math.log10(ALPHA / m)
    nom05_y = -math.log10(ALPHA)
    ax.axhline(bonf_y, color="red", linestyle="--", linewidth=1.2, label=f"Bonferroni (m={m})")
    ax.axhline(nom05_y, color="gray", linestyle=":", linewidth=1.0, label="p = 0.05")

    # y-grid for readability
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    # Category tick labels (not every phenotype)
    ax.set_xticks(centers)
    ax.set_xticklabels(ticklabels, rotation=45, ha="right", fontsize=9)
    ax.tick_params(axis="x", pad=2)  # tighter spacing
    ax.set_xlim(-0.5, g["x"].max() + 0.5)

    # vertical separators between categories
    cum_sizes = np.cumsum([len(g[g["cat_name"] == c]) for c in cat_order])
    for x0 in cum_sizes[:-1]:
        ax.axvline(x=x0 - 0.5, color="#cccccc", linestyle="-", linewidth=0.6, zorder=1)

    # annotate top hits
    tops = g.nsmallest(min(ANNOTATE_TOP_N, m), P_COL)
    for _, r in tops.iterrows():
        ax.annotate(
            r["Phen_pretty"],
            xy=(r["x"], r["minuslog10p"]),
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=8, rotation=30, ha="left", va="bottom", zorder=4
        )

    # labels & title
    inv_txt = str(inversion_label)
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title(f"PheWAS Manhattan — {inv_txt}", pad=10)

    # tidy spines & legend
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()

    os.makedirs(OUTDIR, exist_ok=True)
    outpath = os.path.join(OUTDIR, f"phewas_{sanitize_filename(inv_txt)}.pdf")
    fig.savefig(outpath, format="pdf")
    plt.close(fig)
    return outpath

# ----------------- main -----------------

def main():
    # read inputs
    if not os.path.exists(INFILE):
        sys.exit(f"ERROR: Cannot find {INFILE}")
    if not os.path.exists(PHECODE_FILE):
        sys.exit(f"ERROR: Cannot find {PHECODE_FILE}")

    df = pd.read_csv(INFILE, sep="\t", dtype=str)
    for col in [PHENO_COL, P_COL, INV_COL]:
        if col not in df.columns:
            sys.exit(f"ERROR: {INFILE} missing required column '{col}'")

    # build category map and merge onto phewas rows
    cmap = load_category_map(PHECODE_FILE)
    df["Phen_clean"] = df[PHENO_COL].map(canonicalize_name)
    df = df.merge(
        cmap[["clean_name", "phecode_category", "category_num", "category_num_num"]],
        how="left",
        left_on="Phen_clean",
        right_on="clean_name"
    )

    # drop rows without an inversion label
    inv_mask = df[INV_COL].astype(str).str.strip() != ""
    df = df[inv_mask].copy()
    if df.empty:
        sys.exit("No rows with a non-empty Inversion value.")

    # generate one PDF per inversion; auto-open only if any Sig_Global==True in that inversion
    made = []
    to_open = []
    for inv, group in df.groupby(INV_COL, dropna=False):
        out = plot_one_inversion(group, inversion_label=inv)
        if out:
            made.append(out)
            has_sig = (SIG_COL in group.columns) and truthy_series(group[SIG_COL]).any()
            if has_sig:
                to_open.append(out)

    if not made:
        print("No plots produced (no valid phenotypes/p-values).")
        return

    print(f"Wrote {len(made)} PDF(s) to: {OUTDIR}")
    for p in made:
        print("  -", p)

    if to_open:
        print(f"Auto-opening {len(to_open)} plot(s) with ≥1 Sig_Global (0.05) hit:")
        for p in to_open:
            print("    *", p)
            open_file(p)
    else:
        print("No inversions had a Sig_Global (0.05) hit — nothing auto-opened.")

if __name__ == "__main__":
    main()
