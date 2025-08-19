import pandas as pd
import numpy as np
from cyvcf2 import VCF
from collections import Counter
import warnings
import os
import time
import logging
import sys
import subprocess
from joblib import Parallel, delayed, cpu_count, dump
import random
import itertools
import re
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import wilcoxon, pearsonr
import traceback

warnings.filterwarnings("ignore", category=FutureWarning)
rng = np.random.default_rng(seed=42)

def create_synthetic_data(X_hap1: np.ndarray, X_hap2: np.ndarray, raw_gts: pd.Series, sample_indices: np.ndarray, X_existing: np.ndarray, target_counts: dict):
    hap_pool_0, hap_pool_1 = [], []
    for i, gt in enumerate(raw_gts):
        original_index = sample_indices[i]
        if gt == '0|0':
            hap_pool_0.extend([(X_hap1[i], original_index), (X_hap2[i], original_index)])
        elif gt == '1|1':
            hap_pool_1.extend([(X_hap1[i], original_index), (X_hap2[i], original_index)])
        elif gt == '0|1':
            hap_pool_0.append((X_hap1[i], original_index))
            hap_pool_1.append((X_hap2[i], original_index))
        elif gt == '1|0':
            hap_pool_1.append((X_hap1[i], original_index))
            hap_pool_0.append((X_hap2[i], original_index))
    
    if not hap_pool_0 and not hap_pool_1: return None, None, None

    existing_genomes_set = {tuple(genome) for genome in X_existing}
    X_synth, y_synth, parent_map = [], [], []

    for class_label, num_needed in target_counts.items():
        if num_needed <= 0: continue
        for _ in range(num_needed):
            new_diploid, parents = None, None
            if class_label == 2: # Homozygous Inverted (1|1)
                if len(hap_pool_1) < 2: return None, None, None
                h1_idx, h2_idx = rng.choice(len(hap_pool_1), 2, replace=True)
                h1, p1 = hap_pool_1[h1_idx]
                h2, p2 = hap_pool_1[h2_idx]
                new_diploid = h1 + h2
                parents = [p1, p2]
            elif class_label == 0: # Homozygous Standard (0|0)
                if len(hap_pool_0) < 2: return None, None, None
                h1_idx, h2_idx = rng.choice(len(hap_pool_0), 2, replace=True)
                h1, p1 = hap_pool_0[h1_idx]
                h2, p2 = hap_pool_0[h2_idx]
                new_diploid = h1 + h2
                parents = [p1, p2]
            elif class_label == 1: # Heterozygous (0|1)
                if len(hap_pool_0) < 1 or len(hap_pool_1) < 1: return None, None, None
                h0_idx = rng.choice(len(hap_pool_0))
                h1_idx = rng.choice(len(hap_pool_1))
                h0, p0 = hap_pool_0[h0_idx]
                h1, p1 = hap_pool_1[h1_idx]
                new_diploid = h0 + h1
                parents = [p0, p1]

            if new_diploid is not None and tuple(new_diploid) not in existing_genomes_set:
                X_synth.append(new_diploid)
                y_synth.append(class_label)
                parent_map.append(parents)
                existing_genomes_set.add(tuple(new_diploid))

    if not X_synth: return None, None, None
    return np.array(X_synth), np.array(y_synth), parent_map

def extract_haplotype_data_for_locus(inversion_job: dict, allowed_snps_dict: dict):
    inversion_id = inversion_job.get('orig_ID', 'Unknown_ID')
    vcf_path = None
    try:
        chrom, start, end = inversion_job['seqnames'], inversion_job['start'], inversion_job['end']
        vcf_path = f"../vcfs/{chrom}.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz"
        if not os.path.exists(vcf_path):
            reason = f"VCF file not found: {vcf_path}"
            logging.error(f"[{inversion_id}] FAILED: {reason}")
            return {'status': 'FAILED', 'id': inversion_id, 'reason': reason}

        vcf_reader = VCF(vcf_path, lazy=True)
        vcf_samples = vcf_reader.samples
        tsv_samples = [col for col in inversion_job.keys() if col.startswith(('HG', 'NA'))]
        sample_map = {ts: p[0] for ts in tsv_samples if len(p := [vs for vs in vcf_samples if ts in vs]) == 1}
        if not sample_map:
            reason = "No TSV samples could be mapped to VCF samples."
            logging.warning(f"[{inversion_id}] SKIPPING: {reason}")
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': reason}

        def parse_gt_for_synth(gt_str: any):
            high_conf_map = {"0|0": 0, "1|0": 1, "0|1": 1, "1|1": 2}
            low_conf_map = {"0|0_lowconf": 0, "1|0_lowconf": 1, "0|1_lowconf": 1, "1|1_lowconf": 2}
            if gt_str in high_conf_map: return (high_conf_map[gt_str], True, gt_str)
            if gt_str in low_conf_map: return (low_conf_map[gt_str], False, gt_str.replace("_lowconf", ""))
            return (None, None, None)

        gt_data = {vcf_s: {'dosage': d, 'is_high_conf': hc, 'raw_gt': rgt} for tsv_s, vcf_s in sample_map.items() for d, hc, rgt in [parse_gt_for_synth(inversion_job.get(tsv_s))] if d is not None}
        if not gt_data:
            reason = 'No samples with a valid inversion dosage.'
            logging.warning(f"[{inversion_id}] SKIPPING: {reason}")
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': reason}

        gt_df = pd.DataFrame.from_dict(gt_data, orient='index')
        flank_size = 50000
        region_str = f"{chrom}:{max(0, start - flank_size)}-{end + flank_size}"
        vcf_subset = VCF(vcf_path, samples=list(gt_df.index))

        h1_data, h2_data, snp_meta, processed_pos = [], [], [], set()

        for var in vcf_subset(region_str):
            if var.POS in processed_pos: continue
            normalized_chrom = var.CHROM.replace('chr', '')
            snp_id_str = f"{normalized_chrom}:{var.POS}"
            if snp_id_str not in allowed_snps_dict: continue
            if not (var.is_snp and not var.is_indel and len(var.ALT) == 1 and all(-1 not in gt[:2] for gt in var.genotypes)): continue
            effect_allele = allowed_snps_dict[snp_id_str]
            ref_allele, alt_allele = var.REF, var.ALT[0]
            genotypes = np.array([gt[:2] for gt in var.genotypes], dtype=int)
            if effect_allele == alt_allele: encoded_gts = genotypes
            elif effect_allele == ref_allele: encoded_gts = 1 - genotypes
            else: continue
            h1_data.append(encoded_gts[:, 0]); h2_data.append(encoded_gts[:, 1])
            snp_meta.append({'id': snp_id_str, 'pos': var.POS, 'effect_allele': effect_allele})
            processed_pos.add(var.POS)

        if not snp_meta:
            reason = 'No suitable SNPs from the whitelist were found in the region.'
            logging.warning(f"[{inversion_id}] SKIPPING: {reason}")
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': reason}
        df_H1 = pd.DataFrame(np.array(h1_data).T, index=vcf_subset.samples)
        df_H2 = pd.DataFrame(np.array(h2_data).T, index=vcf_subset.samples)
        common_samples = df_H1.index.intersection(gt_df.index)
        if len(common_samples) < 20:
            reason = f'Insufficient overlapping samples ({len(common_samples)}) for modeling.'
            logging.warning(f"[{inversion_id}] SKIPPING: {reason}")
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': reason}
            
        return {'status': 'PREPROCESSED', 'id': inversion_id, 'X_hap1': df_H1.loc[common_samples].values, 'X_hap2': df_H2.loc[common_samples].values, 'y_diploid': gt_df.loc[common_samples, 'dosage'].values.astype(int), 'raw_gts': gt_df.loc[common_samples, 'raw_gt'], 'snp_metadata': snp_meta}
    except Exception as e:
        reason = f"Data Extraction Error: {type(e).__name__}: {e}. Problem VCF: '{vcf_path}'"
        logging.error(f"[{inversion_id}] FAILED: {reason}")
        return {'status': 'FAILED', 'id': inversion_id, 'reason': reason}

def get_effective_max_components(X_train, y_train, max_components):
    if X_train.shape[1] == 0: return 0
    max_components = min(max_components, X_train.shape[1])
    if max_components <= 1: return max_components
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PLSRegression(n_components=max_components).fit(X_train, y_train)
        for msg in w:
            if issubclass(msg.category, UserWarning) and "y residual is constant" in str(msg.message):
                if match := re.search(r'iteration (\d+)', str(msg.message)): return min(max_components, int(match.group(1)))
    return max_components

def analyze_and_model_locus_pls(preloaded_data: dict, n_jobs_inner: int, output_dir: str):
    inversion_id = preloaded_data['id']
    try:
        y_full = preloaded_data['y_diploid']
        X_hap1, X_hap2 = preloaded_data['X_hap1'], preloaded_data['X_hap2']
        X_full = X_hap1 + X_hap2
        
        # --- Start of New Rescue and Data Assembly Logic ---
        
        total_counts = Counter(y_full)
        needed_counts = {c: max(0, 2 - total_counts.get(c, 0)) for c in [0, 1, 2]}
        
        use_group_kfold = False
        X_combined, y_combined, groups = X_full, y_full, None

        if any(v > 0 for v in needed_counts.values()):
            logging.info(f"[{inversion_id}] Minority class count < 2. Attempting rescue. Needed: {needed_counts}")
            all_sample_indices = np.arange(len(y_full))
            X_synth, y_synth, parent_map = create_synthetic_data(
                X_hap1, X_hap2, preloaded_data['raw_gts'], 
                all_sample_indices, X_full, needed_counts
            )
            
            if X_synth is None or sum(needed_counts.values()) > len(y_synth):
                reason = "Minority class count < 2 and could not be rescued with synthetic data (insufficient haplotypes)."
                logging.warning(f"[{inversion_id}] SKIPPING: {reason}")
                return {'status': 'SKIPPED', 'id': inversion_id, 'reason': reason}
            
            logging.info(f"[{inversion_id}] Rescue successful. Generated {len(y_synth)} synthetic sample(s).")
            use_group_kfold = True
            
            X_combined = np.vstack([X_full, X_synth])
            y_combined = np.concatenate([y_full, y_synth])
            
            # Create group IDs for StratifiedGroupKFold
            group_id_map = {i: i for i in range(len(y_full))}
            synth_group_ids = []
            for child_parents in parent_map:
                parent_group_id = group_id_map[child_parents[0]]
                for parent_idx in child_parents:
                    group_id_map[parent_idx] = parent_group_id
                synth_group_ids.append(parent_group_id)
            groups = np.array([group_id_map[i] for i in range(len(y_full))] + synth_group_ids)

        # --- End of New Rescue and Data Assembly Logic ---

        val_min_class_count = min(Counter(y_combined).values())
        n_outer_splits = min(5, val_min_class_count)
        
        if use_group_kfold:
            outer_cv = StratifiedGroupKFold(n_splits=n_outer_splits, shuffle=True, random_state=42)
        else:
            outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=42)

        pipeline = Pipeline([('pls', PLSRegression())])
        y_true_pooled, y_pred_pls_pooled, y_pred_dummy_pooled = [], [], []

        cv_iterator = outer_cv.split(X_combined, y_combined, groups=groups) if use_group_kfold else outer_cv.split(X_combined, y_combined)
        for train_idx, test_idx in cv_iterator:
            X_train, y_train = X_combined[train_idx], y_combined[train_idx]
            X_test, y_test = X_combined[test_idx], y_combined[test_idx]
            
            if len(X_train) == 0 or len(np.unique(y_train)) < 2: continue
            sample_weights = compute_sample_weight("balanced", y=y_train)
            resampled_indices = rng.choice(len(X_train), size=len(X_train), replace=True, p=sample_weights / np.sum(sample_weights))
            X_train_resampled, y_train_resampled = X_train[resampled_indices], y_train[resampled_indices]
            
            train_min_class_count = min(Counter(y_train_resampled).values())
            if train_min_class_count < 2: continue
            
            max_components = min(100, X_train_resampled.shape[0] - 1)
            effective_max_components = get_effective_max_components(X_train_resampled, y_train_resampled, max_components)
            if effective_max_components < 1: continue
            
            inner_cv = StratifiedKFold(n_splits=min(3, train_min_class_count), shuffle=True, random_state=123)
            grid_search = GridSearchCV(estimator=pipeline, param_grid={'pls__n_components': range(1, effective_max_components + 1)}, scoring='neg_mean_squared_error', cv=inner_cv, n_jobs=n_jobs_inner, error_score='raise')
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "y residual is constant", UserWarning); grid_search.fit(X_train_resampled, y_train_resampled)
            
            dummy_model_fold = DummyRegressor(strategy='mean').fit(X_train, y_train)
            y_true_pooled.extend(y_test); y_pred_pls_pooled.extend(grid_search.best_estimator_.predict(X_test).flatten()); y_pred_dummy_pooled.extend(dummy_model_fold.predict(X_test))
        
        if not y_true_pooled:
            reason = "Nested CV failed on all folds, possibly due to data structure after rescue."
            logging.error(f"[{inversion_id}] FAILED: {reason}")
            return {'status': 'FAILED', 'id': inversion_id, 'reason': reason}
        
        y_true_arr, y_pred_arr = np.array(y_true_pooled), np.array(y_pred_pls_pooled)
        if y_true_arr.size < 2 or np.std(y_true_arr) == 0 or np.std(y_pred_arr) == 0: corr = 0.0
        else: corr, _ = pearsonr(y_true_arr, y_pred_arr)
        
        pearson_r2 = (corr if not np.isnan(corr) else 0.0)**2
        _, p_value = wilcoxon(np.abs(y_true_arr - y_pred_arr), np.abs(y_true_arr - np.array(y_pred_dummy_pooled)), alternative='less', zero_method='zsplit')
        
        final_sample_weights = compute_sample_weight("balanced", y=y_combined)
        final_resampled_indices = rng.choice(len(y_combined), size=len(y_combined), replace=True, p=final_sample_weights / np.sum(final_sample_weights))
        X_final_train, y_final_train = X_combined[final_resampled_indices], y_combined[final_resampled_indices]
        
        final_min_class_count = min(Counter(y_final_train).values())
        if final_min_class_count < 2:
            reason = "Final balanced training set lacks class diversity."
            logging.error(f"[{inversion_id}] FAILED: {reason}")
            return {'status': 'FAILED', 'id': inversion_id, 'reason': reason}
        
        final_max_components = min(100, X_final_train.shape[0] - 1)
        final_effective_max = get_effective_max_components(X_final_train, y_final_train, final_max_components)
        if final_effective_max < 1:
            reason = "Final model training range is invalid."
            logging.error(f"[{inversion_id}] FAILED: {reason}")
            return {'status': 'FAILED', 'id': inversion_id, 'reason': reason}
        
        final_cv = StratifiedKFold(n_splits=min(3, final_min_class_count), shuffle=True, random_state=42)
        final_pipeline = Pipeline([('pls', PLSRegression())])
        final_grid_search = GridSearchCV(estimator=final_pipeline, param_grid={'pls__n_components': range(1, final_effective_max + 1)}, scoring='neg_mean_squared_error', cv=final_cv, n_jobs=n_jobs_inner, refit=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "y residual is constant", UserWarning); final_grid_search.fit(X_final_train, y_final_train)
        
        os.makedirs(output_dir, exist_ok=True)
        model_filename = os.path.join(output_dir, f"{inversion_id}.model.joblib")
        dump(final_grid_search.best_estimator_, model_filename)
        pd.DataFrame(preloaded_data['snp_metadata']).to_json(os.path.join(output_dir, f"{inversion_id}.snps.json"), orient='records')
        
        return {'status': 'SUCCESS', 'id': inversion_id, 'unbiased_pearson_r2': pearson_r2, 'unbiased_rmse': np.sqrt(mean_squared_error(y_true_pooled, y_pred_pls_pooled)), 'model_p_value': p_value, 'best_n_components': final_grid_search.best_params_['pls__n_components'], 'num_snps_in_model': X_full.shape[1], 'model_path': model_filename}
    except Exception as e:
        reason = f"Analysis Error: {type(e).__name__}: {e}"
        logging.error(f"[{inversion_id}] FAILED: {reason}")
        return {'status': 'FAILED', 'id': inversion_id, 'reason': reason, 'traceback': traceback.format_exc()}

def process_locus_end_to_end(job: dict, n_jobs_inner: int, allowed_snps_dict: dict, output_dir: str):
    inversion_id = job.get('orig_ID', 'Unknown_ID')
    success_receipt = os.path.join(output_dir, f"{inversion_id}.model.joblib")
    if os.path.exists(success_receipt):
        return {'status': 'CACHED', 'id': inversion_id, 'reason': 'SUCCESS receipt found.'}
    result = extract_haplotype_data_for_locus(job, allowed_snps_dict)
    if result.get('status') == 'PREPROCESSED':
        result = analyze_and_model_locus_pls(result, n_jobs_inner, output_dir)
    return result

def load_and_normalize_snp_list(filepath: str):
    if not os.path.exists(filepath):
        logging.critical(f"FATAL: SNP whitelist file not found: '{filepath}'"); sys.exit(1)
    allowed_snps_dict = {}
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) == 2:
                snp_id, effect_allele = parts
                chrom, pos = snp_id.split(':', 1)
                normalized_chrom = chrom.replace('chr', '')
                normalized_id = f"{normalized_chrom}:{pos}"
                allowed_snps_dict[normalized_id] = effect_allele.upper()
    if not allowed_snps_dict:
        logging.critical(f"FATAL: SNP whitelist file '{filepath}' was empty."); sys.exit(1)
    logging.info(f"Successfully loaded and normalized {len(allowed_snps_dict)} SNPs from '{filepath}'.")
    return allowed_snps_dict

def check_snp_availability_for_locus(job: dict, allowed_snps_dict: dict):
    inversion_id = job.get('orig_ID', 'Unknown_ID')
    try:
        chrom, start, end = job['seqnames'], job['start'], job['end']
        vcf_path = f"../vcfs/{chrom}.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz"
        if not os.path.exists(vcf_path):
            return {'status': 'VCF_NOT_FOUND', 'id': inversion_id, 'reason': f"VCF file not found: {vcf_path}"}
        flank_size = 50000
        region_str = f"{chrom}:{max(0, start - flank_size)}-{end + flank_size}"
        vcf_reader = VCF(vcf_path, lazy=True)
        for var in vcf_reader(region_str):
            normalized_chrom = var.CHROM.replace('chr', '')
            snp_id_str = f"{normalized_chrom}:{var.POS}"
            if snp_id_str in allowed_snps_dict:
                return {'status': 'FOUND', 'id': inversion_id}
        return {'status': 'NOT_FOUND', 'id': inversion_id, 'region': region_str}
    except Exception as e:
        return {'status': 'PRECHECK_FAILED', 'id': inversion_id, 'reason': str(e)}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(message)s]',
                        datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging.FileHandler("log.txt", mode='w'), logging.StreamHandler(sys.stdout)])

    script_start_time = time.time()
    logging.info("--- Starting Idempotent Imputation Pipeline ---")
    
    output_dir = "final_imputation_models"
    ground_truth_file = "../variants_freeze4inv_sv_inv_hg38_processed_arbigent_filtered_manualDotplot_filtered_PAVgenAdded_withInvCategs_syncWithWH.fixedPH.simpleINV.mod.tsv"
    snp_whitelist_file = "passed_snvs.txt"
    
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(ground_truth_file):
        logging.critical(f"FATAL: Ground-truth file not found: '{ground_truth_file}'"); sys.exit(1)
    
    allowed_snps = load_and_normalize_snp_list(snp_whitelist_file)
    
    config_df = pd.read_csv(ground_truth_file, sep='\t', on_bad_lines='warn', dtype={'seqnames': str})
    config_df = config_df[(config_df['verdict'] == 'pass') & (~config_df['seqnames'].isin(['chrY', 'chrM']))].copy()
    all_jobs = config_df.to_dict('records')
    if not all_jobs:
        logging.warning("No valid inversions to process. Exiting."); sys.exit(0)

    total_cores = cpu_count()
    N_INNER_JOBS = 8 
    if total_cores < N_INNER_JOBS: N_INNER_JOBS = total_cores
    N_OUTER_JOBS = max(1, total_cores // N_INNER_JOBS)
    
    logging.info(f"Loaded {len(all_jobs)} inversions to process or verify.")
    logging.info(f"Using {N_OUTER_JOBS} parallel 'outer' jobs, each with up to {N_INNER_JOBS} 'inner' cores for model training.")
    
    logging.info("\n--- Running Pre-flight SNP Availability Check ---")
    precheck_generator = (delayed(check_snp_availability_for_locus)(job, allowed_snps) for job in all_jobs)
    precheck_results = Parallel(n_jobs=N_OUTER_JOBS, backend='loky')(
        tqdm(precheck_generator, total=len(all_jobs), desc="Pre-checking SNP availability", unit="locus")
    )
    
    loci_without_snps = [r for r in precheck_results if r and r.get('status') == 'NOT_FOUND']
    if loci_without_snps:
        logging.warning("\n" + "="*80)
        logging.warning(f"--- PRE-CHECK WARNING: Found {len(loci_without_snps)} loci that will fail due to no suitable SNPs ---")
        for failed_locus in sorted(loci_without_snps, key=lambda x: x['id']):
            logging.warning(f"  - [{failed_locus['id']}] will fail: No SNPs from whitelist found in region [{failed_locus['region']}]")
        logging.warning("="*80 + "\n")
    else:
        logging.info("--- Pre-flight SNP Availability Check Passed: All loci have at least one potential SNP. ---\n")
    
    logging.info("--- Starting Main Processing Pipeline ---")
    job_generator = (delayed(process_locus_end_to_end)(job, N_INNER_JOBS, allowed_snps, output_dir) for job in all_jobs)
    
    all_results = Parallel(n_jobs=N_OUTER_JOBS, backend='loky')(
        tqdm(job_generator, total=len(all_jobs), desc="Processing Loci", unit="locus")
    )

    logging.info(f"--- All Processing Complete in {time.time() - script_start_time:.2f} seconds ---")
    
    valid_results = [r for r in all_results if r is not None]
    
    successful_runs = [r for r in valid_results if r.get('status') == 'SUCCESS']
    cached_runs = [r for r in valid_results if r.get('status') == 'CACHED']
    
    completed_ids = {r['id'] for r in valid_results}
    all_initial_ids = {job['orig_ID'] for job in all_jobs}
    crashed_ids = all_initial_ids - completed_ids

    other_runs = [r for r in valid_results if r.get('status') not in ['SUCCESS', 'CACHED']]
    
    logging.info("\n\n" + "="*100 + "\n---      FINAL PIPELINE REPORT      ---\n" + "="*100)
    
    summary_counts = Counter()
    for r in other_runs:
        summary_counts[f"({r.get('status')}) {r.get('reason', 'No reason provided.')}"] += 1
    for _ in crashed_ids:
        summary_counts["(CRASHED) Worker process terminated (likely RAM/resource issue)."] += 1

    logging.info(f"Total jobs in initial set: {len(all_jobs)}")
    logging.info(f"  - Newly Successful this run: {len(successful_runs)}")
    logging.info(f"  - Found previously completed (Cached): {len(cached_runs)}")
    if summary_counts:
        logging.info("\n--- Details of Incomplete Loci From This Run ---")
        for reason, count in sorted(summary_counts.items(), key=lambda item: item[1], reverse=True): 
            logging.info(f"  - ({count: >3} loci): {reason}")

    final_models = [f for f in os.listdir(output_dir) if f.endswith('.model.joblib')]
    logging.info(f"\n--- Total Successful Models in Output Directory: {len(final_models)} ---")

    if successful_runs:
        results_df = pd.DataFrame(successful_runs).set_index('id').sort_values('unbiased_pearson_r2', ascending=False)
        logging.info(f"\n--- Performance of {len(successful_runs)} NEWLY Successful Models ---")
        high_perf_df = results_df[(results_df['unbiased_pearson_r2'] > 0.5) & (results_df['model_p_value'] < 0.05)]
        if not high_perf_df.empty:
            summary_cols = ['unbiased_pearson_r2', 'unbiased_rmse', 'model_p_value', 'best_n_components', 'num_snps_in_model']
            with pd.option_context('display.max_rows', None, 'display.width', 120):
                logging.info("\n" + high_perf_df[summary_cols].to_string())
        else:
            logging.info("\n--- No new high-performance models were generated in this run. ---")
    
    logging.info("\n" + "="*100)
