import os
import sys
import math
from collections import Counter, defaultdict
from typing import List, Tuple, Set, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

# -------------------------
# Config / constants
# -------------------------
CDS_SUMMARY_TSV = "cds_identical_proportions.tsv"
PAIRS_PREFIX = "pairs_CDS__"   # actual filename: f"{PAIRS_PREFIX}{summary_filename}.tsv"

# Minimum required haplotypes *per group* for an alignment to be considered.
# Increasing this value enforces stricter filtering before performing
# Direct vs Inverted comparisons. Values below 3 are not supported because the
# jackknife estimates require at least three haplotypes.
MIN_HAPLOTYPES_PER_GROUP = 3

def cat_label(cons: int, grp: int) -> str:
    return f"{'Recurrent' if cons==1 else 'Single'}/{ 'Inverted' if grp==1 else 'Direct'}"

# -------------------------
# Utilities
# -------------------------
def comb2(k: int) -> int:
    return k*(k-1)//2

def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and (math.isnan(p) or p < 0)):
        return "NA"
    if p == 0.0 or p < 1e-300:
        return "<1e-300"
    if p >= 0.001:
        return f"{p:.3f}"
    return f"{p:.2e}"

def bh_fdr(valid_pvals: List[float]) -> List[float]:
    p = np.asarray(valid_pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(m) + 1)
    for i in range(m-2, -1, -1):
        q[i] = min(q[i], q[i+1])
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out.tolist()

# -------------------------
# Alignment-level: p and SE(p) with leave-one-haplotype-out jackknife
# -------------------------
def compute_alignment_stats(pairs_path: str) -> Tuple[float, float, int, int, Set[str], int, int, Dict[str,int]]:
    """
    Returns:
      p, se, k, n_sites, H, y, n, ident_counts (per-haplotype identical-pair count)
    Raises: FileNotFoundError, ValueError
    """
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"pairs TSV not found: {pairs_path}")

    df = pd.read_csv(pairs_path, sep="\t", dtype=str)
    need = {"sample1","sample2","n_sites","n_diff_sites"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"Missing required columns in {pairs_path}; need {need}")

    # haplotypes and counts
    H = set(df["sample1"].astype(str)).union(set(df["sample2"].astype(str)))
    k = len(H)
    if k < 3:
        raise ValueError(f"k <= 2 (k={k}); omitted by rule")

    # pairs and identicals
    n_diff = pd.to_numeric(df["n_diff_sites"], errors="coerce")
    n = int(df.shape[0])
    y = int((n_diff == 0).sum())

    # n_sites
    try:
        n_sites = int(df["n_sites"].dropna().iloc[0])
    except Exception:
        n_sites = -1

    # identical pairs per haplotype (for LOO recompute)
    df_ident = df.loc[n_diff == 0, ["sample1","sample2"]].copy()
    ident_counts = Counter()
    if not df_ident.empty:
        ident_counts.update(df_ident["sample1"].astype(str).tolist())
        ident_counts.update(df_ident["sample2"].astype(str).tolist())

    # alignment p
    p_full = y / n if n > 0 else float("nan")

    # leave-one-haplotype-out SE for p
    p_lo = []
    for h in H:
        y_h = ident_counts.get(str(h), 0)
        n_minus = comb2(k - 1)
        if n_minus <= 0:
            raise ValueError(f"n_minus <= 0 for k={k} in {pairs_path}")
        y_minus = y - y_h
        p_minus = y_minus / n_minus
        p_lo.append(p_minus)

    if len(p_lo) != k:
        raise ValueError(f"leave-one-out count mismatch (got {len(p_lo)} != k={k}) in {pairs_path}")

    mean_lo = sum(p_lo) / k
    var_jk = (k - 1) / k * sum((v - mean_lo) ** 2 for v in p_lo)
    se = math.sqrt(max(var_jk, 0.0))

    if not math.isfinite(se):
        raise ValueError(f"SE is not finite for {pairs_path}")

    return p_full, se, k, n_sites, H, y, n, dict(ident_counts)

# Recompute alignment p when dropping haplotype h
def alignment_p_minus_h(k: int, y: int, p: float, ident_counts: Dict[str,int], h: str) -> float:
    # k is the original haplotype count of this alignment (>=3 guaranteed at input).
    # After removing h, denominator becomes comb2(k-1); caller ensures (k-1) >= 2.
    y_minus = y - ident_counts.get(h, 0)
    n_minus = comb2(k - 1)
    return y_minus / n_minus

# -------------------------
# Main
# -------------------------
def main():
    if not os.path.exists(CDS_SUMMARY_TSV):
        print(f"ERROR: {CDS_SUMMARY_TSV} not found.", file=sys.stderr)
        sys.exit(1)

    print(">>> Loading cds_identical_proportions.tsv ...")
    print(f"    Minimum haplotypes per group required: {MIN_HAPLOTYPES_PER_GROUP}")
    df0 = pd.read_csv(CDS_SUMMARY_TSV, sep="\t", dtype=str)

    needed = {
        "dataset","consensus","phy_group","filename",
        "gene_name","transcript_id",
        "chr","inv_start","inv_end",
        "n_sequences","n_pairs","n_identical_pairs","prop_identical_pairs",
        "inv_exact_match"
    }
    missing = needed - set(df0.columns)
    if missing:
        print("ERROR: Missing columns in cds_identical_proportions.tsv:", ", ".join(sorted(missing)), file=sys.stderr)
        sys.exit(1)

    # cast types
    for c in ["consensus","phy_group","n_sequences","n_pairs","n_identical_pairs","inv_start","inv_end","inv_exact_match"]:
        df0[c] = pd.to_numeric(df0[c], errors="coerce")

    # Filter: CDS only, exact match only, consensus in {0,1}, groups in {0,1}, k>=3
    df = df0[
        (df0["dataset"] == "CDS") &
        (df0["inv_exact_match"] == 1) &
        (df0["consensus"].isin([0,1])) &
        (df0["phy_group"].isin([0,1])) &
        (df0["n_sequences"] >= 3)
    ].copy()

    if df.empty:
        print("No alignments remain after filtering (CDS, exact-match, consensus ∈ {0,1}, k ≥ 3).")
        sys.exit(0)

    # normalize/strings
    df["filename"] = df["filename"].astype(str)
    df["gene_name"] = df["gene_name"].astype(str)
    df["transcript_id"] = df["transcript_id"].astype(str)
    df["chr"] = df["chr"].astype(str)

    # inversion locus id
    df["inv_id"] = df.apply(lambda r: f"{r['chr']}:{int(r['inv_start'])}-{int(r['inv_end'])}", axis=1)

    print(f"    Alignments after filtering: {len(df)}")

    # -------------------------
    # Alignment-level stats
    # -------------------------
    alignment_rows = []
    skips = []

    print("\n>>> Computing alignment-level p and SE (sequence leave-one-out) ...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Alignments", unit="aln"):
        fn = row["filename"]
        pairs_path = f"{PAIRS_PREFIX}{fn}.tsv"
        if not os.path.exists(pairs_path):
            skips.append({"level":"alignment","filename":fn,"reason":"MISSING_PAIRS_TSV","detail":pairs_path})
            continue

        try:
            p, se, k, n_sites, H, y, n, ident_counts = compute_alignment_stats(pairs_path)
        except Exception as e:
            skips.append({"level":"alignment","filename":fn,"reason":"ALIGNMENT_FAIL","detail":repr(e)})
            continue

        if k < MIN_HAPLOTYPES_PER_GROUP:
            skips.append({
                "level": "alignment",
                "filename": fn,
                "reason": "ALIGNMENT_MIN_HAPLOTYPES",
                "detail": f"k={k} < MIN_HAPLOTYPES_PER_GROUP={MIN_HAPLOTYPES_PER_GROUP}"
            })
            continue

        if not math.isfinite(se):
            skips.append({"level":"alignment","filename":fn,"reason":"SE_NOT_FINITE","detail":"alignment SE invalid"})
            continue

        alignment_rows.append({
            "filename": fn,
            "gene_name": row["gene_name"],
            "transcript_id": row["transcript_id"],
            "phy_group": int(row["phy_group"]),
            "consensus": int(row["consensus"]),
            "inv_id": row["inv_id"],
            "k": int(k),
            "n_sites": int(n_sites),
            "p": float(p),
            "se": float(se),
            "y": int(y),
            "n": int(n),
            "H": set(H),
            "ident_counts": dict(ident_counts),
        })

    print("\n>>> Skips during alignment processing:")
    if not skips:
        print("    None.")
    else:
        for s in skips:
            print(f"    SKIP {s['level']} :: {s['filename']} -> {s['reason']} :: {s['detail']}")

    if not alignment_rows:
        print("No valid alignments after processing; exiting.")
        sys.exit(0)

    aln = pd.DataFrame(alignment_rows)

    # -------------------------
    # Per-inversion REGION medians and median-of-medians (diagnostic)
    # -------------------------
    print("\n>>> Computing per-inversion medians and median-of-medians for each group ...")
    inv_meds = (
        aln.groupby(["consensus","phy_group","inv_id"], as_index=False)["p"]
           .median()
           .rename(columns={"p":"inv_median_p"})
    )
    group_median_of_medians = (
        inv_meds.groupby(["consensus","phy_group"], as_index=False)["inv_median_p"]
                .median()
                .rename(columns={"inv_median_p":"median_of_inversion_medians"})
    )

    print("=== Median CDS identical proportion *per inversion region* (grouped) ===")
    for _, r in inv_meds.sort_values(["consensus","phy_group","inv_id"]).iterrows():
        print(f"    {cat_label(int(r['consensus']), int(r['phy_group'])):<22} {r['inv_id']:<16} median_CDS_p={r['inv_median_p']:.6f}")

    print("\n=== Median *across inversion medians* (each of the four groups) ===")
    for _, r in group_median_of_medians.sort_values(["consensus","phy_group"]).iterrows():
        print(f"    {cat_label(int(r['consensus']), int(r['phy_group'])):<22} median_of_medians={r['median_of_inversion_medians']:.6f}")

    # -------------------------
    # Index data for testing & ENFORCE: at most ONE alignment per (gene, inversion, group)
    # -------------------------
    key_to_idxs: Dict[Tuple[str,str,int], List[int]] = defaultdict(list)
    for idx, row in aln.iterrows():
        key = (row["gene_name"], row["inv_id"], int(row["phy_group"]))
        key_to_idxs[key].append(idx)

    # Crash the program if any (gene, inversion, group) has multiple alignments
    violations = [(g, inv, grp, len(idxs)) for (g, inv, grp), idxs in key_to_idxs.items() if len(idxs) > 1]
    if violations:
        msg_lines = [
            "ASSERTION FAILED: Multiple CDS alignments/files detected for the same (gene, inversion, group).",
            "This program forbids averaging and aborts when duplicates exist.",
            "Examples (up to 10):"
        ]
        for (g, inv, grp, cnt) in violations[:10]:
            msg_lines.append(f"  gene={g}  inv={inv}  group={'Inverted' if grp==1 else 'Direct'}  count={cnt}")
        msg_lines.append("Please deduplicate upstream so that each (gene, inversion, group) has exactly one CDS file.")
        raise AssertionError("\n".join(msg_lines))

    # Also map (gene, inv) to present groups
    gi_groups: Dict[Tuple[str,str], Set[int]] = defaultdict(set)
    for (g, inv, grp), idxs in key_to_idxs.items():
        gi_groups[(g, inv)].add(grp)

    # -------------------------
    # Build (gene, inversion) tests
    # -------------------------
    print("\n>>> Testing Direct vs Inverted within each (gene, inversion) using Δ-jackknife over haplotypes ...")
    tests = []
    all_gi = sorted(gi_groups.keys())
    for (gname, inv_id) in tqdm(all_gi, total=len(all_gi), desc="Gene×Inversion tests", unit="test"):
        groups_present = gi_groups[(gname, inv_id)]
        if groups_present != {0,1}:
            tests.append({
                "gene_name": gname, "inv_id": inv_id,
                "status": "SKIP_NO_BOTH_GROUPS",
                "detail": f"groups_present={sorted(list(groups_present))}"
            })
            continue

        # Fetch the single required alignment for each group (asserted above)
        idx_dir = key_to_idxs[(gname, inv_id, 0)][0]
        idx_inv = key_to_idxs[(gname, inv_id, 1)][0]
        rec_dir = aln.loc[idx_dir]
        rec_inv = aln.loc[idx_inv]

        # Ensure both groups have enough haplotypes for testing
        if int(rec_dir["k"]) < MIN_HAPLOTYPES_PER_GROUP or int(rec_inv["k"]) < MIN_HAPLOTYPES_PER_GROUP:
            tests.append({
                "gene_name": gname, "inv_id": inv_id,
                "status": "SKIP_MIN_HAPLOTYPES",
                "detail": f"direct_k={int(rec_dir['k'])}, inverted_k={int(rec_inv['k'])}, min={MIN_HAPLOTYPES_PER_GROUP}"
            })
            continue

        # Group-level p = the single alignment p (no averaging)
        p_dir = float(rec_dir["p"])
        p_inv = float(rec_inv["p"])

        # Build union of haplotypes across BOTH groups
        H_dir = set(rec_dir["H"])
        H_inv = set(rec_inv["H"])
        union_H = H_dir | H_inv

        if len(union_H) == 0:
            tests.append({
                "gene_name": gname, "inv_id": inv_id,
                "status": "SKIP_NO_HAPLOTYPES",
                "detail": "Union of haplotypes across groups is empty"
            })
            continue

        # Δ full
        delta_full = p_inv - p_dir

        # Prepare alignment rec dicts
        dir_rec = {
            "k": int(rec_dir["k"]),
            "p": float(rec_dir["p"]),
            "y": int(rec_dir["y"]),
            "H": set(rec_dir["H"]),
            "ident_counts": dict(rec_dir["ident_counts"]),
        }
        inv_rec = {
            "k": int(rec_inv["k"]),
            "p": float(rec_inv["p"]),
            "y": int(rec_inv["y"]),
            "H": set(rec_inv["H"]),
            "ident_counts": dict(rec_inv["ident_counts"]),
        }

        # Leave-one-haplotype-out over the UNION
        delta_lo = []
        fail_flag = False
        for h in union_H:
            # Direct group p^{(-h)}
            if h in dir_rec["H"]:
                if dir_rec["k"] - 1 < 2:
                    fail_flag = True
                    break
                p_dir_minus = alignment_p_minus_h(dir_rec["k"], dir_rec["y"], dir_rec["p"], dir_rec["ident_counts"], h)
            else:
                p_dir_minus = dir_rec["p"]

            # Inverted group p^{(-h)}
            if h in inv_rec["H"]:
                if inv_rec["k"] - 1 < 2:
                    fail_flag = True
                    break
                p_inv_minus = alignment_p_minus_h(inv_rec["k"], inv_rec["y"], inv_rec["p"], inv_rec["ident_counts"], h)
            else:
                p_inv_minus = inv_rec["p"]

            if not (math.isfinite(p_dir_minus) and math.isfinite(p_inv_minus)):
                fail_flag = True
                break

            delta_lo.append(p_inv_minus - p_dir_minus)

        if fail_flag or len(delta_lo) != len(union_H):
            tests.append({
                "gene_name": gname, "inv_id": inv_id,
                "status": "SKIP_DELTA_JK_FAIL",
                "detail": "Failed LOO recompute for one or more haplotypes"
            })
            continue

        # Jackknife variance for Δ
        K = len(union_H)
        mean_lo = sum(delta_lo) / K
        var_jk = (K - 1) / K * sum((d - mean_lo) ** 2 for d in delta_lo)
        se_delta = math.sqrt(max(var_jk, 0.0))

        # p-value logic including boundary case
        delta = delta_full
        note = ""
        if se_delta > 0 and math.isfinite(se_delta):
            z = delta / se_delta
            pval = 2.0 * (1.0 - norm_cdf(abs(z)))
        else:
            if abs(delta) < 1e-15:
                z = 0.0
                pval = 1.0
                note = "boundary_zero_diff"
            else:
                z = float("inf") if delta > 0 else float("-inf")
                pval = 1e-300
                note = "boundary_nonzero_diff"

        tests.append({
            "gene_name": gname,
            "transcript_id": rec_dir["transcript_id"],
            "inv_id": inv_id,
            "p_direct": float(p_dir),
            "p_inverted": float(p_inv),
            "delta": float(delta),
            "se_delta": float(se_delta),
            "z_value": float(z) if math.isfinite(z) else z,
            "p_value": float(pval),
            "note": note,
            "status": "OK"
        })

    tests_df = pd.DataFrame(tests)

    # Print skips
    print("\n>>> Skips during (gene, inversion) testing:")
    skipped = tests_df[tests_df["status"] != "OK"]
    if skipped.empty:
        print("    None.")
    else:
        for _, r in skipped.iterrows():
            print(f"    SKIP gene={r.get('gene_name','?')} inv={r.get('inv_id','?')} -> {r['status']} :: {r.get('detail','')}")

    ok = tests_df[tests_df["status"] == "OK"].copy()
    if ok.empty:
        print("No valid (gene, inversion) tests to report. Exiting.")
        sys.exit(0)

    # FDR across VALID p-values only
    valid_pvals = ok["p_value"].astype(float).tolist()
    qvals = bh_fdr(valid_pvals)
    ok["q_value"] = qvals

    # Order & print ALL results
    ok = ok.sort_values(["q_value","gene_name","inv_id"]).reset_index(drop=True)

    print("\n=== (Gene, Inversion) Direct vs Inverted — ALL RESULTS ===")
    header = [
        "gene_name","transcript_id","inv_id",
        "p_direct","p_inverted",
        "delta","se_delta","z","p","q","note"
    ]
    print("\t".join(header))
    for _, r in ok.iterrows():
        print("\t".join([
            str(r["gene_name"]),
            str(r.get("transcript_id","NA")),
            str(r["inv_id"]),
            f"{r['p_direct']:.6f}",
            f"{r['p_inverted']:.6f}",
            f"{r['delta']:.6f}",
            f"{r['se_delta']:.6f}",
            (f"{r['z_value']:.3f}" if math.isfinite(r["z_value"]) else ("+inf" if r["z_value"]>0 else "-inf")),
            fmt_p(float(r["p_value"])),
            fmt_p(float(r["q_value"])),
            r.get("note","")
        ]))

    # Save TSV
    out_tests = "gene_inversion_direct_inverted.tsv"
    save_cols = [
        "gene_name","transcript_id","inv_id",
        "p_direct","p_inverted",
        "delta","se_delta","z_value","p_value","q_value","note"
    ]
    ok[save_cols].to_csv(out_tests, sep="\t", index=False)
    print(f"\nWrote: {out_tests}")

    # Significant list (q < 0.05)
    alpha = 0.05
    sig = ok[ok["q_value"] < alpha].copy()

    print("\n=== Significant (q < 0.05) ===")
    if sig.empty:
        print("  (none)")
    else:
        for _, r in sig.sort_values("q_value").iterrows():
            print(f"  {r['gene_name']}  ({r.get('transcript_id','NA')})  @ {r['inv_id']}:  "
                  f"Δ={r['delta']:.4f},  p={fmt_p(float(r['p_value']))},  q={fmt_p(float(r['q_value']))}"
                  + (f"  [{r['note']}]" if r.get('note') else ""))

    # Summary
    print("\n=== Summary ===")
    print(f"Valid (gene, inversion) tests: {len(ok)}")
    print(f"Significant at FDR 0.05: {len(sig)}")
    print("\nDone.")

if __name__ == "__main__":
    main()
