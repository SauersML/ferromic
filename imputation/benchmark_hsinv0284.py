#!/usr/bin/env python3
"""
Benchmark the ferromic imputation model for the 6q24.1 inversion against the
experimental MLPA/iMLPA genotypes published by Giner-Delgado et al. 2019
(Nat. Commun. 10:4222; InvFEST id HsInv0284; dbVar study nstd169).

Reviewer 2, comment 3.

Our inversion id (coordinate match against data/inv_properties.tsv):
    chr6-141867315-INV-29159   (GRCh38 chr6:141,866,310-141,898,728)

Pipeline
--------
1.  Truth: dbVar nstd169 per-sample genotype VCF (GRCh37). Pull the HsInv0284
    record, decode GT -> inverted-allele dosage {0,1,2} keyed by 1000G sample id.
2.  Predictors: the trained PLS model uses 334 GRCh38 SNPs in chr6:141.8-141.9 Mb.
    Slice those SNPs out of the 1000 Genomes 30x high-coverage GRCh38 *phased*
    panel (NYGC, 3202 samples), restricted to the samples that also have an
    experimental call, and write a PLINK bed (prefix `subset`).
3.  prepare_data_for_infer.py routes the bed into the model genotype matrix
    (chr:pos + effect-allele matching, with allele flipping).  Run UNCHANGED.
4.  Apply the model exactly as infer_dosage.py does: impute any missing SNP with
    the per-ancestry mean (here ancestry = 1000G superpopulation), then predict.
5.  Compare predicted vs experimental dosage: Pearson r, r^2, RMSE, and hard-call
    concordance (round prediction to 0/1/2), overall and per superpopulation.

All inputs are public.  Nothing here touches the private T2T callset, All of Us,
or any non-public resource.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# Our model / inversion under test.
MODEL_NAME = "chr6-141867315-INV-29159"

# dbVar nstd169 per-sample genotype VCF (GRCh37).  HsInv0284 is a named record.
TRUTH_VCF_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/dbVar/data/Homo_sapiens/"
    "by_study/genotype/nstd169/45inversion_genotypes_v4.8.vcf.gz"
)
TRUTH_INV_ID = "HsInv0284"

# 1000G 30x high-coverage GRCh38 phased panel (chr6) + 3202-sample pedigree.
KGP_DIR = (
    "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
    "1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV"
)
KGP_CHR6_VCF = f"{KGP_DIR}/1kGP_high_coverage_Illumina.chr6.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
KGP_PED_URL = (
    "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
    "1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt"
)

MISSING_VALUE_CODE = -127

# Superpopulation -> integer code, mirroring infer_dosage.load_ancestry_map().
SUPERPOP_CODES = {"EUR": 0, "AFR": 1, "AMR": 2, "EAS": 3, "SAS": 4}
UNKNOWN_CODE = 7


def run(cmd: List[str], **kw) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


# --------------------------------------------------------------------------- #
# Step 1: experimental truth
# --------------------------------------------------------------------------- #
def load_truth(truth_vcf: str) -> Dict[str, int]:
    """Return {sample_id: inverted-allele dosage 0/1/2} for HsInv0284.

    Missing experimental calls (./.) are dropped.
    """
    import gzip

    opener = gzip.open if truth_vcf.endswith(".gz") else open
    samples: List[str] = []
    calls: Dict[str, int] = {}
    with opener(truth_vcf, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            fields = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                samples = fields[9:]
                continue
            if fields[2] != TRUTH_INV_ID:
                continue
            for sid, cell in zip(samples, fields[9:]):
                gt = cell.split(":")[0].replace("|", "/")
                alleles = [a for a in gt.split("/") if a not in (".", "")]
                if not alleles:
                    continue  # missing experimental call
                calls[sid] = sum(int(a) for a in alleles)
            break
    if not calls:
        sys.exit(f"[FATAL] {TRUTH_INV_ID} not found in {truth_vcf}")
    print(
        f"Truth: {len(calls)} called samples for {TRUTH_INV_ID} "
        f"(dist: { {d: sum(v == d for v in calls.values()) for d in (0, 1, 2)} })"
    )
    return calls


# --------------------------------------------------------------------------- #
# Step 2: predictor genotypes -> PLINK bed
# --------------------------------------------------------------------------- #
def model_snp_region(model_dir: str) -> Tuple[int, int]:
    with open(os.path.join(model_dir, f"{MODEL_NAME}.snps.json")) as fh:
        spec = json.load(fh)
    pos = [int(s["pos"]) for s in spec]
    return min(pos), max(pos)


def build_subset_bed(
    kgp_vcf: str,
    truth_samples: List[str],
    region_start: int,
    region_end: int,
    out_prefix: str,
    tmp_dir: str,
) -> None:
    """Slice chr6 model region from the 1000G panel for the overlapping samples
    and write a PLINK bed (prefix out_prefix) with GRCh38 chr:pos ids."""
    keep_path = os.path.join(tmp_dir, "keep_samples.txt")
    with open(keep_path, "w") as fh:
        fh.write("\n".join(truth_samples) + "\n")

    # Region slice (tabix) -> keep overlapping samples -> biallelic SNPs only.
    region = f"chr6:{region_start}-{region_end}"
    sliced = os.path.join(tmp_dir, "kgp_region.vcf.gz")
    run(
        [
            "bcftools", "view",
            "-r", region,
            "-S", keep_path, "--force-samples",
            "-m2", "-M2", "-v", "snps",
            "-Oz", "-o", sliced,
            kgp_vcf,
        ]
    )
    run(["bcftools", "index", "-t", sliced])

    # PLINK2 -> bed.  set-all-var-ids gives chr:pos ids the router matches on
    # via chrom_aliases (it strips 'chr').  --max-alleles 2 already enforced.
    run(
        [
            "plink2",
            "--vcf", sliced,
            "--set-all-var-ids", "@:#",
            "--new-id-max-allele-len", "200", "missing",
            "--make-bed",
            "--out", out_prefix,
        ]
    )


# --------------------------------------------------------------------------- #
# Step 4: apply model (mirrors infer_dosage.py imputation + predict)
# --------------------------------------------------------------------------- #
def compute_ancestry_means(X: np.ndarray, anc: np.ndarray, n_codes: int) -> np.ndarray:
    """Per-ancestry mean of each SNP over non-missing entries, with global
    fallback.  Identical logic to infer_dosage.compute_ancestry_means."""
    n_samples, n_snps = X.shape
    sums = np.zeros((n_codes + 1, n_snps))
    counts = np.zeros((n_codes + 1, n_snps))
    valid = X != MISSING_VALUE_CODE
    safe = np.where(valid, X, 0).astype(np.float64)
    sums[n_codes] = safe.sum(axis=0)
    counts[n_codes] = valid.sum(axis=0)
    for code in range(n_codes):
        m = anc == code
        if not np.any(m):
            continue
        sums[code] = safe[m].sum(axis=0)
        counts[code] = valid[m].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = sums / counts
    gmean = np.nan_to_num(means[n_codes], nan=0.0)
    means[n_codes] = gmean
    for code in range(n_codes):
        nan = np.isnan(means[code])
        means[code, nan] = gmean[nan]
    return means.astype(np.float32)


def predict_dosage(
    model_dir: str, geno_dir: str, sample_ids: List[str], anc: np.ndarray
) -> np.ndarray:
    clf = joblib.load(os.path.join(model_dir, f"{MODEL_NAME}.model.joblib"))
    X = np.array(
        np.load(os.path.join(geno_dir, f"{MODEL_NAME}.genotypes.npy"), mmap_mode="r"),
        dtype=np.float32,
    )
    if X.shape[0] != len(sample_ids):
        sys.exit(f"[FATAL] matrix rows {X.shape[0]} != samples {len(sample_ids)}")

    miss = X == MISSING_VALUE_CODE
    print(f"Predictor matrix: {X.shape}; missing entries: {int(miss.sum())}")
    if miss.any():
        means = compute_ancestry_means(X, anc, len(SUPERPOP_CODES))
        X[miss] = means[anc][miss]
    pred = np.ravel(clf.predict(X.astype(np.float32)))
    # Constrain to the valid dosage range [0, 2], exactly as infer_dosage.py does. The model
    # can extrapolate slightly outside it; impossible dosages must never be reported.
    return np.clip(pred, 0.0, 2.0)


# --------------------------------------------------------------------------- #
# Step 5: metrics
# --------------------------------------------------------------------------- #
def metrics(truth: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    n = len(truth)
    if n < 2 or np.std(truth) == 0 or np.std(pred) == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(truth, pred)[0, 1])
    hard = np.clip(np.rint(pred), 0, 2).astype(int)
    return {
        "n": int(n),
        "pearson_r": r,
        "r2": float(r * r) if r == r else float("nan"),
        "rmse": float(np.sqrt(np.mean((truth - pred) ** 2))),
        "concordance": float(np.mean(hard == truth)),
        "n_carriers_truth": int(np.sum(truth > 0)),
        "n_carriers_pred_hard": int(np.sum(hard > 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "models"))
    ap.add_argument("--truth-vcf", required=True, help="local dbVar nstd169 genotype vcf(.gz)")
    ap.add_argument("--kgp-vcf", default=KGP_CHR6_VCF, help="1000G chr6 phased panel (url or local)")
    ap.add_argument("--ped", required=True, help="local 1000G 3202-sample ped/population file")
    ap.add_argument("--work-dir", default="bench_work")
    ap.add_argument("--out-tsv", default="data/imputation_benchmark_HsInv0284.tsv")
    ap.add_argument("--out-summary", default="data/imputation_benchmark_HsInv0284_summary.tsv")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    work = os.path.abspath(args.work_dir)
    os.makedirs(work, exist_ok=True)

    # 1) truth
    truth = load_truth(args.truth_vcf)

    # superpopulation map
    ped = pd.read_csv(args.ped, sep=r"\s+")
    sp_map = dict(zip(ped["SampleID"].astype(str), ped["Superpopulation"].astype(str)))

    # samples we can benchmark = experimental call AND present in 1000G panel ped
    bench_samples = sorted(s for s in truth if s in sp_map)
    print(
        f"Benchmark samples (experimental ∩ 1000G panel): {len(bench_samples)} "
        f"of {len(truth)} experimentally called"
    )

    # 2) predictor bed
    rstart, rend = model_snp_region(model_dir)
    print(f"Model SNP region (GRCh38): chr6:{rstart}-{rend} (334 SNPs)")
    subset_prefix = os.path.join(work, "subset")
    build_subset_bed(args.kgp_vcf, bench_samples, rstart, rend, subset_prefix, work)

    # 3) route bed -> model genotype matrix (prepare_data_for_infer.py, unchanged)
    geno_dir = os.path.join(work, "genotype_matrices")
    env = dict(os.environ)
    env.update(
        PLINK_PREFIX=subset_prefix,
        OUTPUT_DIR=geno_dir,
        MODEL_SOURCE_DIR=model_dir,
    )
    run([sys.executable, os.path.join(os.path.dirname(__file__), "prepare_data_for_infer.py")], env=env)

    # bed sample order (fam) drives matrix row order
    fam = pd.read_csv(subset_prefix + ".fam", sep=r"\s+", header=None, dtype=str)
    sample_ids = fam[1].tolist()
    anc = np.array(
        [SUPERPOP_CODES.get(sp_map.get(s, ""), UNKNOWN_CODE) for s in sample_ids],
        dtype=np.int8,
    )

    # 4) predict
    pred = predict_dosage(model_dir, geno_dir, sample_ids, anc)

    # 5) align truth & predictions, write per-sample table
    rows = []
    for sid, p in zip(sample_ids, pred):
        if sid not in truth:
            continue
        rows.append(
            {
                "SampleID": sid,
                "Superpopulation": sp_map.get(sid, "NA"),
                "experimental_dosage": int(truth[sid]),
                "imputed_dosage": round(float(p), 4),
                "imputed_hardcall": int(np.clip(round(float(p)), 0, 2)),
            }
        )
    df = pd.DataFrame(rows).sort_values(["Superpopulation", "SampleID"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out_tsv)), exist_ok=True)
    df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"Wrote {len(df)} per-sample comparisons -> {args.out_tsv}")

    # summary: overall + per superpopulation
    summary = []
    t = df["experimental_dosage"].to_numpy(float)
    p = df["imputed_dosage"].to_numpy(float)
    summary.append({"group": "ALL", **metrics(t, p)})
    for sp in sorted(df["Superpopulation"].unique()):
        sub = df[df["Superpopulation"] == sp]
        summary.append(
            {
                "group": sp,
                **metrics(
                    sub["experimental_dosage"].to_numpy(float),
                    sub["imputed_dosage"].to_numpy(float),
                ),
            }
        )
    sdf = pd.DataFrame(summary)
    sdf.to_csv(args.out_summary, sep="\t", index=False)
    print(f"Wrote summary -> {args.out_summary}")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
