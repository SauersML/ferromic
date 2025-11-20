import os
import sys
import gzip
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set
from joblib import Parallel, delayed, cpu_count
from cyvcf2 import VCF

# -------------------- Configuration --------------------
VCF_DIR = "../vcfs"
VCF_TEMPLATE = "{chrom}.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz"  # chrom like 'chr1'
INVERSION_TSV = "../variants_freeze4inv_sv_inv_hg38_processed_arbigent_filtered_manualDotplot_filtered_PAVgenAdded_withInvCategs_syncWithWH.fixedPH.simpleINV.mod.tsv"
OUTPUT_FILE = "vcf_list.txt"
FLANK_BP = 50_000
EXAMPLE_LIMIT = 5
N_JOBS = max(1, cpu_count() // 2)  # adjust as you like
VALID_BASES: Set[str] = {"A", "C", "G", "T"}
# -------------------------------------------------------

def unify_chr(chrom: str) -> Optional[str]:
    """Normalize chromosome to 'chrN' (N=1..22,X,Y,MT). Return None if unsupported."""
    c = chrom.strip()
    if c.startswith("chr"): c = c[3:]
    c = c.upper()
    if c in {"X", "Y"}:
        return f"chr{c}"
    if c in {"M", "MT"}:
        return "chrMT"
    try:
        n = int(c)
        return f"chr{n}" if 1 <= n <= 22 else None
    except ValueError:
        return None

def chrom_sort_key(chr_label: str) -> int:
    c = chr_label[3:] if chr_label.startswith("chr") else chr_label
    if c == "X": return 23
    if c == "Y": return 24
    if c == "MT": return 25
    return int(c)

def natural_sort_key(snp_id: str) -> Tuple[int, int]:
    # snp_id like "chr10:12345"
    chrom_part, pos_part = snp_id.split(":")
    return chrom_sort_key(chrom_part), int(pos_part)

def is_biallelic_snp(ref: str, alts: List[str]) -> Tuple[bool, Optional[str]]:
    if len(alts) != 1:
        return False, "not_biallelic"
    ref = ref.strip().upper()
    alt = alts[0].strip().upper()
    if len(ref) != 1 or len(alt) != 1:
        return False, "indel_or_len!=1"
    if ref not in VALID_BASES or alt not in VALID_BASES:
        return False, "non_ACGT"
    return True, None

def variant_has_full_callrate(var) -> bool:
    """
    100% call rate: no missing GT. With cyvcf2:
      gt_types codes: 0=HOM_REF, 1=HET, 2=HOM_ALT, 3=UNKNOWN
    """
    gt = var.gt_types  # numpy array
    if gt is None:
        return False
    # Missing if any UNKNOWN present
    return not np.any(gt == 3)

def compute_major_allele(var) -> Optional[str]:
    """
    Count allele copies from gt_types:
      ref copies = 2*hom_ref + 1*het
      alt copies = 2*hom_alt + 1*het
    Return the allele letter (REF or ALT[0]) with higher count (ties -> REF).
    """
    gt = var.gt_types
    if gt is None: return None
    hom_ref = np.sum(gt == 0)
    het     = np.sum(gt == 1)
    hom_alt = np.sum(gt == 2)
    n_ref = 2 * hom_ref + het
    n_alt = 2 * hom_alt + het
    major_is_ref = (n_ref >= n_alt)
    return (var.REF if major_is_ref else var.ALT[0]).upper()

def vcf_path_for_chrom(chrom: str) -> str:
    return os.path.join(VCF_DIR, VCF_TEMPLATE.format(chrom=chrom))

def process_one_inversion(job: dict) -> Tuple[str, Dict[str, int], List[Tuple[str, str]]]:
    """
    Process a single inversion window and return:
      (inversion_id, stats_dict, kept_pairs)
    where kept_pairs = [(snp_id, major_allele), ...], snp_id like 'chrN:POS'
    """
    inv_id = job.get("orig_ID", f"{job.get('seqnames','chr?')}-{job.get('start','?')}-{job.get('end','?')}")
    chrom = job["seqnames"]
    start = int(job["start"])
    end   = int(job["end"])
    padded_start = max(1, start - FLANK_BP)
    padded_end   = end + FLANK_BP
    region_str = f"{chrom}:{padded_start}-{padded_end}"
    vcf_path = vcf_path_for_chrom(chrom)

    stats = Counter()
    kept_pairs: List[Tuple[str, str]] = []
    examples = defaultdict(list)

    if not os.path.exists(vcf_path):
        print(f"[WARN] [{inv_id}] VCF not found: {vcf_path}")
        stats["vcf_missing"] += 1
        return inv_id, stats, kept_pairs

    try:
        vcf = VCF(vcf_path)  # all samples
        n_samples = len(vcf.samples)
        print(f"[INV] {inv_id}  chrom={chrom}  region={region_str}  samples={n_samples}  vcf={Path(vcf_path).name}")

        for var in vcf(region_str):
            stats["variants_total"] += 1

            # normalize chr
            norm_chr = unify_chr(var.CHROM)
            if norm_chr is None:
                stats["unsupported_chrom"] += 1
                if len(examples["unsupported_chrom"]) < EXAMPLE_LIMIT:
                    examples["unsupported_chrom"].append(var.CHROM)
                continue

            # fast checks
            if not var.is_snp:
                stats["not_snp"] += 1
                if len(examples["not_snp"]) < EXAMPLE_LIMIT:
                    examples["not_snp"].append(f"{var.CHROM}:{var.POS}")
                continue
            if var.is_indel:
                stats["indel"] += 1
                if len(examples["indel"]) < EXAMPLE_LIMIT:
                    examples["indel"].append(f"{var.CHROM}:{var.POS}")
                continue

            ok, reason = is_biallelic_snp(var.REF, var.ALT)
            if not ok:
                stats[reason] += 1
                if len(examples[reason]) < EXAMPLE_LIMIT:
                    examples[reason].append(f"{var.CHROM}:{var.POS} {var.REF}>{','.join(var.ALT)}")
                continue

            # 100% call rate
            if not variant_has_full_callrate(var):
                stats["failed_callrate"] += 1
                if len(examples["failed_callrate"]) < EXAMPLE_LIMIT:
                    examples["failed_callrate"].append(f"{var.CHROM}:{var.POS}")
                continue

            # major allele
            maj = compute_major_allele(var)
            if maj is None or maj not in VALID_BASES:
                stats["bad_major"] += 1
                continue

            snp_id = f"{norm_chr}:{var.POS}"
            kept_pairs.append((snp_id, maj))
            stats["kept"] += 1

        # file-level summary
        print(f"  [SUMMARY {inv_id}] seen={stats.get('variants_total',0)} kept={stats.get('kept',0)} "
              f"fail_call={stats.get('failed_callrate',0)} not_snp={stats.get('not_snp',0)} "
              f"indel={stats.get('indel',0)} not_bial={stats.get('not_biallelic',0)} "
              f"len!=1={stats.get('indel_or_len!=1',0)} nonACGT={stats.get('non_ACGT',0)}")

    except Exception as e:
        print(f"[ERROR] [{inv_id}] {type(e).__name__}: {e}")
        stats["vcf_error"] += 1

    return inv_id, stats, kept_pairs

def main():
    t0 = time.time()
    print("=== MAJOR-ALLELE SNP LISTER (LOUD MODE) ===")
    print(f"[CONFIG] INVERSION_TSV={INVERSION_TSV}")
    print(f"[CONFIG] VCF_DIR={VCF_DIR}")
    print(f"[CONFIG] TEMPLATE={VCF_TEMPLATE}")
    print(f"[CONFIG] OUTPUT_FILE={OUTPUT_FILE}")
    print(f"[CONFIG] FLANK_BP={FLANK_BP}")
    print(f"[CONFIG] N_JOBS={N_JOBS}")
    print()

    if not os.path.exists(INVERSION_TSV):
        print(f"[FATAL] Inversion TSV not found: {INVERSION_TSV}")
        sys.exit(1)

    # Load loci to process (keep it similar to your modeling setup)
    cfg = pd.read_csv(INVERSION_TSV, sep="\t", dtype={"seqnames": str}, on_bad_lines="warn")
    # If you want the exact same filtering as modeling, uncomment:
    # cfg = cfg[(cfg["verdict"] == "pass") & (~cfg["seqnames"].isin(["chrY", "chrM"]))].copy()

    if cfg.empty:
        print("[WARN] No inversions to process. Exiting.")
        Path(OUTPUT_FILE).write_text("")
        print(f"[RESULT] Output written to: {OUTPUT_FILE}")
        return

    jobs = cfg[["orig_ID", "seqnames", "start", "end"]].to_dict("records")
    print(f"[DISCOVERY] Loaded {len(jobs)} inversion(s). Overlaps are allowed.")

    # Parallel processing of inversions
    print("[PROCESS] Launching parallel workers...")
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(process_one_inversion)(job) for job in jobs
    )

    # Collect and deduplicate
    print("[COLLECT] Merging results across inversions and removing duplicates...")
    global_stats = Counter()
    per_inv_kept = {}
    pairs_all: List[Tuple[str, str]] = []

    for inv_id, stats, pairs in results:
        global_stats.update(stats)
        per_inv_kept[inv_id] = stats.get("kept", 0)
        pairs_all.extend(pairs)

    # Deduplicate by snp_id, keeping the first allele (should agree anyway)
    dedup_map: Dict[str, str] = {}
    conflicts = 0
    for snp_id, allele in pairs_all:
        if snp_id in dedup_map and dedup_map[snp_id] != allele:
            conflicts += 1  # extremely unlikely; log and keep the first
        else:
            dedup_map.setdefault(snp_id, allele)

    sorted_snps = sorted(dedup_map.keys(), key=natural_sort_key)

    # Write output
    print(f"[WRITE] Writing {len(sorted_snps)} unique SNPs to {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w") as out:
        for snp_id in sorted_snps:
            out.write(f"{snp_id} {dedup_map[snp_id]}\n")
    print("[WRITE] Done.\n")

    # Global summary
    print("=== GLOBAL SUMMARY ===")
    print(f"Inversions processed     : {len(jobs)}")
    print(f"Total variants seen      : {global_stats.get('variants_total', 0)}")
    print(f"Kept (pre-dedup)         : {global_stats.get('kept', 0)}")
    print(f"Unique after dedup       : {len(sorted_snps)}")
    print(f"Conflicting major alleles: {conflicts}")
    print(f"Failed callrate (<100%)  : {global_stats.get('failed_callrate', 0)}")
    print(f"Not SNP                  : {global_stats.get('not_snp', 0)}")
    print(f"Indels                   : {global_stats.get('indel', 0)}")
    print(f"Not biallelic            : {global_stats.get('not_biallelic', 0)}")
    print(f"len!=1 / non-ACGT        : {global_stats.get('indel_or_len!=1', 0)} / {global_stats.get('non_ACGT', 0)}")
    print(f"Unsupported chrom        : {global_stats.get('unsupported_chrom', 0)}")
    print(f"VCF missing/errors       : {global_stats.get('vcf_missing', 0)} / {global_stats.get('vcf_error', 0)}")
    print()

    # Per-inversion kept counts (optional noise)
    print("[DETAIL] Per-inversion kept counts:")
    for inv_id in sorted(per_inv_kept):
        print(f"  {inv_id:40s} kept={per_inv_kept[inv_id]}")

    dt = time.time() - t0
    print(f"\n[RESULT] Process complete in {dt:.2f}s.")
    print(f"[RESULT] Output written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
