# Finds the tagging SNP via K-Fold CV.
# Reports detailed allele correspondence for high-confidence SNPs.
# Prints the final summary and saves the high-confidence results to a TSV file.

import pandas as pd
import numpy as np
from cyvcf2 import VCF
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from joblib import Parallel, delayed, cpu_count
import os
import time
import logging
import sys
import subprocess
from collections import Counter
import warnings

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="An input array is constant; the correlation coefficient is not defined.")

class FatalSampleMappingError(Exception):
    pass

# --- Core Worker and Helper Functions ---

def analyze_locus_correlation(inversion_job: dict, n_folds: int = 5):
    """
    Uses K-Fold CV to select the most robust tag SNP, evaluates its unbiased correlation,
    and identifies which specific allele corresponds to the inversion.
    """
    inversion_id = inversion_job.get('orig_ID', 'Unknown_ID')
    log_prefix = f"[{inversion_id}]"

    try:
        # --- A. Data Preparation ---
        chrom = inversion_job['seqnames']
        flank_size = 50000
        region_start = max(0, inversion_job['start'] - flank_size)
        region_end = inversion_job['end'] + flank_size
        vcf_path = f"../vcfs/{chrom}.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz"
        
        vcf_for_samples = VCF(vcf_path, lazy=True)
        vcf_samples = vcf_for_samples.samples
        
        tsv_samples = list(inversion_job.keys())[7:]
        
        sample_map = {tsv_s: vcf_s for tsv_s in tsv_samples for vcf_s in vcf_samples if tsv_s in vcf_s}
        
        if not (str(chrom).endswith('X') or str(chrom).endswith('Y')) and (len(sample_map) / len(tsv_samples)) < 0.5:
            raise FatalSampleMappingError(f"Autosomal sample mapping rate for {chrom} was below 50%.")
        if not sample_map:
            raise ValueError("0% of TSV samples mapped to VCF header.")

        # --- B. Create Haplotype-Level Dataset ---
        haplotype_inv_status = []
        valid_vcf_samples_in_order = []
        
        def parse_hap_gt(gt_str: any) -> (int, int):
            if not isinstance(gt_str, str) or '|' not in gt_str: return None, None
            parts = gt_str.split('|')
            if len(parts) != 2: return None, None
            h1_str, h2_str = parts[0].split('_')[0], parts[1].split('_')[0]
            if not h1_str.isdigit() or not h2_str.isdigit(): return None, None
            # Inversion status: 0=Direct, 1=Inverted
            return int(h1_str), int(h2_str)

        for tsv_s, vcf_s in sample_map.items():
            h1_label, h2_label = parse_hap_gt(inversion_job[tsv_s])
            if h1_label is not None:
                haplotype_inv_status.extend([h1_label, h2_label])
                valid_vcf_samples_in_order.append(vcf_s)

        if len(haplotype_inv_status) < n_folds * 2:
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Too few haplotypes ({len(haplotype_inv_status)}) for {n_folds}-Fold CV.'}
        
        y_haplotypes = np.array(haplotype_inv_status, dtype=int)
        
        vcf_subset = VCF(vcf_path, samples=valid_vcf_samples_in_order)
        region_str = f"{chrom}:{region_start}-{region_end}"
        
        snp_data_list, snp_metadata = [], []
        
        for variant in vcf_subset(region_str):
             if variant.is_snp and not variant.is_indel and len(variant.ALT) == 1:
                haplotypes_for_snp = [allele for gt in variant.genotypes for allele in gt[0:2]]
                snp_data_list.append(haplotypes_for_snp)
                
                snp_id = variant.ID if variant.ID is not None else f"{variant.CHROM}:{variant.POS}"
                snp_metadata.append({'id': snp_id, 'ref': variant.REF, 'alt': variant.ALT[0]})

        if not snp_metadata:
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'No valid biallelic SNPs in region'}
        
        X_haplotypes = np.array(snp_data_list, dtype=int).T
        logging.info(f"{log_prefix} Built matrix with {X_haplotypes.shape[0]} haplotypes and {X_haplotypes.shape[1]} SNPs.")

        # --- C1. K-Fold CV for Robust SNP Selection ---
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        best_snp_indices_from_folds = []

        for train_indices, _ in kf.split(X_haplotypes):
            X_train, y_train = X_haplotypes[train_indices], y_haplotypes[train_indices]
            fold_best_snp_idx, fold_min_p_val = -1, 1.0

            for i in range(X_train.shape[1]):
                snp_alleles = X_train[:, i]
                valid_indices = np.where(snp_alleles >= 0)[0]
                if len(valid_indices) < 3: continue
                
                filtered_snp_alleles, filtered_y = snp_alleles[valid_indices], y_train[valid_indices]
                if np.all(filtered_snp_alleles == filtered_snp_alleles[0]): continue
                
                _, p = pearsonr(filtered_snp_alleles, filtered_y)
                if p < fold_min_p_val: fold_min_p_val, fold_best_snp_idx = p, i
            
            if fold_best_snp_idx != -1: best_snp_indices_from_folds.append(fold_best_snp_idx)

        if not best_snp_indices_from_folds:
            return {'status': 'FAILED', 'id': inversion_id, 'reason': 'Could not determine a best SNP in any CV fold.'}

        final_best_snp_index = Counter(best_snp_indices_from_folds).most_common(1)[0][0]

        # --- C2. Evaluate chosen SNP on FULL dataset & Determine Allele Correspondence ---
        best_snp_meta = snp_metadata[final_best_snp_index]
        final_best_snp_id, ref_allele, alt_allele = best_snp_meta['id'], best_snp_meta['ref'], best_snp_meta['alt']

        final_snp_alleles = X_haplotypes[:, final_best_snp_index]
        valid_indices = np.where(final_snp_alleles >= 0)[0]

        if len(valid_indices) < 3:
             return {'status': 'FAILED', 'id': inversion_id, 'reason': f'CV-selected SNP {final_best_snp_id} insufficient data for final eval.'}

        final_filtered_alleles, final_filtered_y = final_snp_alleles[valid_indices], y_haplotypes[valid_indices]
        
        if np.all(final_filtered_alleles == final_filtered_alleles[0]):
             return {'status': 'FAILED', 'id': inversion_id, 'reason': f'CV-selected SNP {final_best_snp_id} has no variation.'}

        final_r, final_p = pearsonr(final_filtered_alleles, final_filtered_y)

        allele_direct, allele_inverted = (f"{alt_allele} (ALT)", f"{ref_allele} (REF)") if final_r < 0 else (f"{ref_allele} (REF)", f"{alt_allele} (ALT)")

        return {
            'status': 'SUCCESS', 'id': inversion_id, 'best_snp': final_best_snp_id,
            'r_squared': final_r ** 2, 'p_value': final_p, 'n_haplotypes': len(y_haplotypes),
            'Allele_Direct': allele_direct, 'Allele_Inverted': allele_inverted
        }

    except FatalSampleMappingError as e:
        return {'status': 'FATAL', 'id': inversion_id, 'reason': str(e)}
    except Exception as e:
        return {'status': 'FAILED', 'id': inversion_id, 'reason': f"{type(e).__name__}: {e}"}

# --- Main Orchestrator ---
if __name__ == '__main__':
    logging.info("--- Starting Tag SNP Correlation Analysis (CV & Allele Specific) ---")
    start_time = time.time()
    
    ground_truth_file = "../variants_freeze4inv_sv_inv_hg38_processed_arbigent_filtered_manualDotplot_filtered_PAVgenAdded_withInvCategs_syncWithWH.fixedPH.simpleINV.mod.tsv"

    if not os.path.exists(ground_truth_file):
        logging.critical(f"FATAL: GT file not found at '{ground_truth_file}'"); sys.exit(1)

    try:
        config_df = pd.read_csv(ground_truth_file, sep='\t', on_bad_lines='warn')
    except Exception as e:
        logging.critical(f"FATAL: Could not parse TSV file. Error: {e}"); sys.exit(1)

    initial_rows = len(config_df)
    config_df = config_df[(config_df['verdict'] == 'pass') & (config_df['seqnames'] != 'chrY')].copy()
    logging.info(f"Loaded {initial_rows} rows. Filtered down to {len(config_df)} inversions for processing.")
    
    # --- VCF Index Check ---
    logging.info("--- Checking for VCF indexes... ---")
    unique_chroms = config_df['seqnames'].unique()
    for chrom in unique_chroms:
        vcf_path = f"../vcfs/{chrom}.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz"
        if os.path.exists(vcf_path) and not os.path.exists(vcf_path + ".tbi"):
            logging.info(f"Indexing {vcf_path} with tabix...")
            try:
                subprocess.run(['tabix', '-p', 'vcf', vcf_path], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as err:
                logging.critical(f"FATAL: tabix failed for {vcf_path}: {err.stderr}"); sys.exit(1)
    
    jobs = config_df.to_dict('records')
    if not jobs:
        logging.warning("No valid jobs to run. Exiting."); sys.exit(0)

    num_jobs = len(jobs)
    num_procs = min(num_jobs, cpu_count())
    logging.info(f"Initializing joblib with {num_procs} parallel processes for {num_jobs} jobs...")
    
    results_list = Parallel(n_jobs=num_procs, backend='loky')(delayed(analyze_locus_correlation)(job) for job in jobs)
        
    logging.info(f"--- Analysis Complete in {time.time() - start_time:.2f} seconds ---")
    
    # --- Process Results ---
    successful_runs, failed_runs = [], []
    for result in results_list:
        if result and result.get('status') == 'FATAL':
            logging.critical(f"FATAL ERROR in worker for {result['id']}: {result['reason']}. Aborting.")
            sys.exit(1)
        elif result and result.get('status') == 'SUCCESS':
            successful_runs.append(result)
        elif result:
            failed_runs.append(result)

    print("\n\n" + "="*100)
    print("---                                    FINAL SUMMARY REPORT                                    ---")
    print("="*100)

    print("\n--- FAILED OR SKIPPED LOCI SUMMARY ---")
    if failed_runs:
        reason_counts = Counter(f"({res.get('status', 'N/A')}) {res.get('reason', 'Unknown')}" for res in failed_runs)
        for reason, count in sorted(reason_counts.items()):
            print(f"  - ({count: >3} loci): {reason}")
    else:
        print("  None.")

    if successful_runs:
        results_df = pd.DataFrame(successful_runs).set_index('id')
        
        print("\n\n--- AGGREGATED PERFORMANCE SUMMARY (ALL SUCCESSFUL LOCI) ---")
        print(f"\nProportion of Regions Successfully Analyzed: {len(successful_runs)} / {num_jobs} ({len(successful_runs)/num_jobs:.2%})")
        print("\n--- Unbiased R-squared (Mean/Median based on CV selection) ---")
        print(f"  Mean R-squared:   {results_df['r_squared'].mean():.4f}")
        print(f"  Median R-squared: {results_df['r_squared'].median():.4f}")

        # --- HIGH-CONFIDENCE HITS WITH ALLELE DETAILS & FILE OUTPUT ---
        print("\n\n" + "="*100)
        print("--- HIGH-CONFIDENCE TAG SNPS (CV-Selected, R² > 0.3 AND p < 0.05) ---")
        print("="*100)
        
        high_conf_df = results_df[
            (results_df['r_squared'] > 0.3) & (results_df['p_value'] < 0.05)
        ].sort_values('r_squared', ascending=False)

        if not high_conf_df.empty:
            pd.set_option('display.width', 160)
            pd.set_option('display.max_rows', None)
            pd.options.display.float_format = '{:.4g}'.format
            
            display_cols = ['best_snp', 'r_squared', 'p_value', 'n_haplotypes', 'Allele_Direct', 'Allele_Inverted']
            print(high_conf_df[display_cols])

            # --- SAVE TO TSV FILE ---
            output_tsv_path = "high_confidence_tag_snps.tsv"
            try:
                high_conf_df[display_cols].to_csv(output_tsv_path, sep='\t', index=True, float_format='%.4g')
                print(f"\n[SUCCESS] Saved {len(high_conf_df)} high-confidence results to '{output_tsv_path}'")
            except Exception as e:
                print(f"\n[WARNING] Could not save high-confidence results to TSV: {e}")
        else:
            print("  No high-confidence tag SNPs found meeting the criteria.")

    print("\n" + "="*100)
