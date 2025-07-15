import os
import sys

import pandas as pd
import numpy as np
from cyvcf2 import VCF
from scipy.stats import pearsonr
from joblib import Parallel, delayed, cpu_count

# --- Configuration ---
ground_truth_file = "../variants_freeze4inv_sv_inv_hg38_processed_arbigent_filtered_manualDotplot_filtered_PAVgenAdded_withInvCategs_syncWithWH.fixedPH.simpleINV.mod.tsv"
vcf_dir           = "../vcfs"
flank_size        = 50000
output_file       = "perfect_tagged_snps.tsv"

# These will be set in main()
sample_cols = None
TOTAL       = None


def process_inv(idx, row):
    inv   = row["orig_ID"]
    chrom = row["seqnames"]
    if chrom == "chrY":
        return None

    # parse coordinates
    try:
        start = int(row["start"])
        end   = int(row["end"])
    except ValueError:
        return None

    # open VCF
    vcf_path = os.path.join(
        vcf_dir,
        f"{chrom}.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz"
    )
    if not os.path.exists(vcf_path):
        print(f"{idx+1}/{TOTAL} {inv} SKIP VCF_MISSING", file=sys.stderr)
        return None
    try:
        vcf = VCF(vcf_path)
    except Exception:
        print(f"{idx+1}/{TOTAL} {inv} SKIP VCF_OPEN_ERROR", file=sys.stderr)
        return None

    # map TSV sample names → VCF samples
    smap = {
        tsv: vcf_s
        for tsv in sample_cols
        for vcf_s in vcf.samples
        if tsv in vcf_s
    }
    if not chrom.endswith("X") and len(smap) / len(sample_cols) < 0.5:
        sys.exit(f"ERROR sample mapping failure for inversion {inv}: "
                 f"only {len(smap)}/{len(sample_cols)} samples mapped")
    if not smap:
        print(f"{idx+1}/{TOTAL} {inv} SKIP NOMAP", file=sys.stderr)
        return None

    # build per‐sample inversion dosage vector y
    y, valid = [], []
    for tsv, vcf_name in smap.items():
        gt = row[tsv]
        if "|" not in gt:
            continue
        a, b = gt.split("|")
        a = a.split("_")[0]; b = b.split("_")[0]
        if not (a.isdigit() and b.isdigit()):
            continue
        y.append(int(a) + int(b))
        valid.append(vcf_name)

    if len(y) < 2 or len(set(y)) < 2:
        print(f"{idx+1}/{TOTAL} {inv} SKIP NO_HAPS {len(y)}", file=sys.stderr)
        return None

    y = np.array(y, dtype=int)
    non_missing = y.size
    inv_count   = int((y > 0).sum())
    dir_count   = int((y == 0).sum())

    # ** upfront filter: require more than one of each genotype **
    if inv_count <= 1 or dir_count <= 1:
        print(f"{idx+1}/{TOTAL} {inv} SKIP LOW_COUNTS invs={inv_count} directs={dir_count}", file=sys.stderr)
        return None

    # scan SNPs in ±flank_size around the inversion
    left  = max(1, start - flank_size)
    right = end + flank_size
    region = f"{chrom}:{left}-{right}"
    sub = VCF(vcf_path, samples=valid)

    tested = []
    for v in sub(region):
        if not v.is_snp or v.is_indel or len(v.ALT) != 1:
            continue

        x = []
        missing = False
        for a0, a1, *_ in v.genotypes:
            if a0 < 0 or a1 < 0:
                missing = True
                break
            x.append(a0 + a1)
        if missing:
            continue

        x_arr = np.array(x, dtype=int)
        if x_arr.std() == 0 or x_arr.size != y.size:
            continue

        r, p = pearsonr(x_arr, y)
        tested.append((v, r, r*r, p))

    if not tested:
        print(f"{idx+1}/{TOTAL} {inv} SKIP NO_SNPS", file=sys.stderr)
        return None

    # report the single best SNP
    v_best, r_best, r2_best, p_best = max(tested, key=lambda t: t[2])
    sid_best = v_best.ID or f"{v_best.CHROM}:{v_best.POS}"
    dist_best = min(abs(v_best.POS - start), abs(v_best.POS - end))
    print(
        f"{idx+1}/{TOTAL} {inv} BEST {sid_best} "
        f"R2={r2_best:.3f} p={p_best:.2e} dist={dist_best} "
        f"samples={non_missing} invs={inv_count} directs={dir_count}",
        file=sys.stderr
    )

    # find all perfect‐tag SNPs (R^2 ≈ 1)
    perfect_hits = [
        (v, r, r2, p) for (v, r, r2, p) in tested
        if np.isclose(r2, 1.0)
    ]
    if not perfect_hits:
        return None

    records = []
    for v_p, r_p, r2_p, p_p in perfect_hits:
        sid = v_p.ID or f"{v_p.CHROM}:{v_p.POS}"
        dist = min(abs(v_p.POS - start), abs(v_p.POS - end))
        print(
            f"{idx+1}/{TOTAL} {inv} PERFECT {sid} "
            f"R2={r2_p:.3f} p={p_p:.2e} dist={dist} "
            f"samples={non_missing} invs={inv_count} directs={dir_count}",
            file=sys.stderr
        )
        records.append({
            "inversion_id": inv,
            "snp_id":       sid,
            "ref":          v_p.REF,
            "alt":          v_p.ALT[0],
            "r":            r_p,
            "r_squared":    r2_p,
            "p_value":      p_p,
            "distance":     dist,
            "non_missing":  non_missing,
            "inv_count":    inv_count,
            "dir_count":    dir_count
        })

    return records


if __name__ == "__main__":
    # load inversions table
    df = pd.read_csv(ground_truth_file, sep="\t", dtype=str)
    required = {"orig_ID", "seqnames", "start", "end"}
    if not required.issubset(df.columns):
        sys.exit("ERROR missing required columns in ground truth TSV")

    sample_cols = [c for c in df.columns if c not in required]
    TOTAL       = len(df)
    records     = df.to_dict("records")

    # process in parallel
    raw = Parallel(
        n_jobs=min(cpu_count(), TOTAL),
        backend="loky"
    )(delayed(process_inv)(i, r) for i, r in enumerate(records))

    # flatten results
    all_perfects = []
    for item in raw:
        if item:
            all_perfects.extend(item)

    if not all_perfects:
        sys.exit("ERROR no perfect tags found after filtering")

    # final summary with p-value
    print(f"\nSummary: found {len(all_perfects)} perfect tag(s)", file=sys.stderr)
    for rec in all_perfects:
        print(
            f"{rec['inversion_id']} → {rec['snp_id']} "
            f"R2={rec['r_squared']:.3f} p={rec['p_value']:.2e} "
            f"samples={rec['non_missing']} "
            f"invs={rec['inv_count']} directs={rec['dir_count']}",
            file=sys.stderr
        )

    # write output TSV
    out_df = pd.DataFrame(all_perfects)
    out_df.to_csv(
        output_file,
        sep="\t",
        index=False,
        columns=[
            "inversion_id", "snp_id", "ref", "alt",
            "r", "r_squared", "p_value", "distance",
            "non_missing", "inv_count", "dir_count"
        ]
    )
