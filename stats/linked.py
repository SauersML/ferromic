# --- IMPORTS ---
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
from multiprocessing import Manager
import threading

# Scikit-learn for modeling, evaluation, and preprocessing
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, get_scorer
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import wilcoxon, pearsonr, ConstantInputWarning

# --- SETUP ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)
rng = np.random.default_rng(seed=42)

# --- WORKER FUNCTIONS ---

def create_synthetic_data(X_hap1: np.ndarray, X_hap2: np.ndarray, raw_gts: pd.Series, confidence_mask: np.ndarray, X_real_train_fold: np.ndarray):
    if not np.any(confidence_mask):
        return None, None
    X_h1_hc, X_h2_hc, gts_hc = X_hap1[confidence_mask], X_hap2[confidence_mask], raw_gts[confidence_mask]
    hap_pool_0, hap_pool_1 = [], []
    for i, gt in enumerate(gts_hc):
        if gt == '0|0': hap_pool_0.extend([X_h1_hc[i], X_h2_hc[i]])
        elif gt == '1|1': hap_pool_1.extend([X_h1_hc[i], X_h2_hc[i]])
        elif gt == '0|1': hap_pool_0.append(X_h1_hc[i]); hap_pool_1.append(X_h2_hc[i])
        elif gt == '1|0': hap_pool_1.append(X_h1_hc[i]); hap_pool_0.append(X_h2_hc[i])
    if not hap_pool_0 and not hap_pool_1: return None, None
    unique_hap_pool_0 = np.unique(np.array(hap_pool_0), axis=0) if hap_pool_0 else np.array([])
    unique_hap_pool_1 = np.unique(np.array(hap_pool_1), axis=0) if hap_pool_1 else np.array([])
    n_unique_0, n_unique_1 = len(unique_hap_pool_0), len(unique_hap_pool_1)
    if n_unique_0 < 1 and n_unique_1 < 1: return None, None
    existing_genomes_set = {tuple(genome) for genome in X_real_train_fold}
    X_synth, y_synth = [], []
    if n_unique_1 >= 2:
        for i, j in itertools.combinations_with_replacement(range(n_unique_1), 2):
            new_diploid = unique_hap_pool_1[i] + unique_hap_pool_1[j]
            if tuple(new_diploid) not in existing_genomes_set: X_synth.append(new_diploid); y_synth.append(2)
    if n_unique_0 >= 2:
        for i, j in itertools.combinations_with_replacement(range(n_unique_0), 2):
            new_diploid = unique_hap_pool_0[i] + unique_hap_pool_0[j]
            if tuple(new_diploid) not in existing_genomes_set: X_synth.append(new_diploid); y_synth.append(0)
    if n_unique_0 >= 1 and n_unique_1 >= 1:
        for i, j in itertools.product(range(n_unique_0), range(n_unique_1)):
            new_diploid = unique_hap_pool_0[i] + unique_hap_pool_1[j]
            if tuple(new_diploid) not in existing_genomes_set: X_synth.append(new_diploid); y_synth.append(1)
    if not X_synth: return None, None
    return np.array(X_synth), np.array(y_synth)

def extract_haplotype_data_for_locus(inversion_job: dict):
    inversion_id = inversion_job.get('orig_ID', 'Unknown_ID')
    try:
        chrom, start, end = inversion_job['seqnames'], inversion_job['start'], inversion_job['end']
        vcf_path = f"../vcfs/{chrom}.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz"
        if not os.path.exists(vcf_path):
            return {'status': 'FAILED', 'id': inversion_id, 'reason': f"VCF file not found: {vcf_path}"}
        vcf_reader = VCF(vcf_path, lazy=True)
        vcf_samples = vcf_reader.samples
        tsv_samples = [col for col in inversion_job.keys() if col.startswith(('HG', 'NA'))]

        sample_map = {ts: p[0] for ts in tsv_samples if len(p := [vs for vs in vcf_samples if ts in vs]) == 1}
        
        if not sample_map:
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': "No TSV samples could be mapped to VCF samples."}
        def parse_gt_for_synth(gt_str: any):
            high_conf_map = {"0|0": 0, "1|0": 1, "0|1": 1, "1|1": 2}
            low_conf_map = {"0|0_lowconf": 0, "1|0_lowconf": 1, "0|1_lowconf": 1, "1|1_lowconf": 2}
            if gt_str in high_conf_map: return (high_conf_map[gt_str], True, gt_str)
            if gt_str in low_conf_map: return (low_conf_map[gt_str], False, gt_str.replace("_lowconf", ""))
            return (None, None, None)
        gt_data = {vcf_s: {'dosage': d, 'is_high_conf': hc, 'raw_gt': rgt} for tsv_s, vcf_s in sample_map.items() if (d := parse_gt_for_synth(inversion_job.get(tsv_s))[0]) is not None for _, hc, rgt in [parse_gt_for_synth(inversion_job.get(tsv_s))]}
        if not gt_data:
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'No samples with a valid inversion dosage.'}
        gt_df = pd.DataFrame.from_dict(gt_data, orient='index')
        region_str = f"{chrom}:{max(0, start - flank_size)}-{end + flank_size}" if (flank_size := 50000) else f"{chrom}:{start}-{end}"
        vcf_subset = VCF(vcf_path, samples=list(gt_df.index))
        h1_data, h2_data, snp_meta, processed_pos = [], [], [], set()
        for var in vcf_subset(region_str):
            if var.POS in processed_pos: continue
            if var.is_snp and not var.is_indel and len(var.ALT) == 1 and all(-1 not in gt[:2] for gt in var.genotypes):
                h1_data.append([gt[0] for gt in var.genotypes]); h2_data.append([gt[1] for gt in var.genotypes])
                snp_meta.append({'id': var.ID or f"{var.CHROM}:{var.POS}", 'pos': var.POS}); processed_pos.add(var.POS)
        if not snp_meta:
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'No suitable SNPs (with 100% call rate) found.'}
        df_H1 = pd.DataFrame(np.array(h1_data).T, index=vcf_subset.samples); df_H2 = pd.DataFrame(np.array(h2_data).T, index=vcf_subset.samples)
        common_samples = df_H1.index.intersection(gt_df.index)
        if len(common_samples) < 20: return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Insufficient overlapping samples ({len(common_samples)}) for modeling.'}
        if gt_df.loc[common_samples, 'is_high_conf'].sum() < 3: return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Insufficient high-confidence samples for pooling.'}
        return {'status': 'PREPROCESSED', 'id': inversion_id, 'X_hap1': df_H1.loc[common_samples].values, 'X_hap2': df_H2.loc[common_samples].values, 'y_diploid': gt_df.loc[common_samples, 'dosage'].values.astype(int), 'confidence_mask': gt_df.loc[common_samples, 'is_high_conf'].values.astype(bool), 'raw_gts': gt_df.loc[common_samples, 'raw_gt'], 'snp_metadata': snp_meta}
    except Exception as e:
        return {'status': 'FAILED', 'id': inversion_id, 'reason': f"Data Extraction Error: {type(e).__name__}: {e}"}

def get_effective_max_components(X_train, y_train, max_components):
    if max_components <= 1: return max_components
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PLSRegression(n_components=max_components).fit(X_train, y_train)
        for msg in w:
            if issubclass(msg.category, UserWarning) and "y residual is constant" in str(msg.message):
                if match := re.search(r'iteration (\d+)', str(msg.message)): return min(max_components, int(match.group(1)))
    return max_components

def make_progress_scorer(scoring, progress_queue):
    scorer = get_scorer(scoring)
    def progress_scoring_func(estimator, X, y):
        progress_queue.put(('UPDATE', 1))
        return scorer(estimator, X, y)
    return progress_scoring_func

def analyze_and_model_locus_pls(preloaded_data: dict, n_jobs_inner: int = 1, progress_scorer=None):
    inversion_id = preloaded_data['id']
    try:
        y_full, confidence_mask = preloaded_data['y_diploid'], preloaded_data['confidence_mask']
        X_hap1, X_hap2 = preloaded_data['X_hap1'], preloaded_data['X_hap2']
        X_full = X_hap1 + X_hap2
        X_val, y_val = X_full[confidence_mask], y_full[confidence_mask]
        X_lowconf, y_lowconf = X_full[~confidence_mask], y_full[~confidence_mask]

        val_min_class_count = min(Counter(y_val).values()) if y_val.size > 0 else 0
        if val_min_class_count < 2: return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Validation set minority class count ({val_min_class_count}) is < 2.'}
        
        n_outer_splits = min(5, val_min_class_count)
        outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=42)
        pipeline = Pipeline([('pls', PLSRegression())])
        y_true_pooled, y_pred_pls_pooled, y_pred_dummy_pooled = [], [], []
        original_hc_indices = np.where(confidence_mask)[0]

        for train_val_idx, test_val_idx in outer_cv.split(X_val, y_val):
            X_test, y_test = X_val[test_val_idx], y_val[test_val_idx]
            X_train_val_fold, y_train_val_fold = X_val[train_val_idx], y_val[train_val_idx]
            X_real_train_fold = np.vstack([p for p in [X_train_val_fold, X_lowconf] if p.shape[0] > 0])
            y_real_train_fold = np.concatenate([p for p in [y_train_val_fold, y_lowconf] if p.shape[0] > 0])
            fold_specific_training_mask = np.zeros_like(confidence_mask, dtype=bool)
            fold_specific_training_mask[original_hc_indices[train_val_idx]] = True
            X_synth_fold, y_synth_fold = create_synthetic_data(X_hap1, X_hap2, preloaded_data['raw_gts'], fold_specific_training_mask, X_real_train_fold)
            train_parts_X, train_parts_y = [X_real_train_fold], [y_real_train_fold]
            if X_synth_fold is not None: train_parts_X.append(X_synth_fold); train_parts_y.append(y_synth_fold)
            X_train_full, y_train_full = np.vstack(train_parts_X), np.concatenate(train_parts_y)
            if len(X_train_full) == 0 or len(np.unique(y_train_full)) < 2: continue
            sample_weights = compute_sample_weight("balanced", y=y_train_full)
            resampled_indices = rng.choice(len(X_train_full), size=len(X_train_full), replace=True, p=sample_weights / np.sum(sample_weights))
            X_train, y_train = X_train_full[resampled_indices], y_train_full[resampled_indices]
            train_min_class_count = min(Counter(y_train).values()) if len(y_train) > 0 else 0
            if train_min_class_count < 2: continue
            max_components = min(100, X_train.shape[1], X_train.shape[0] - 1)
            effective_max_components = get_effective_max_components(X_train, y_train, max_components)
            if effective_max_components < 1: continue
            inner_cv = StratifiedKFold(n_splits=min(3, train_min_class_count), shuffle=True, random_state=123)
            grid_search = GridSearchCV(estimator=pipeline, param_grid={'pls__n_components': range(1, effective_max_components + 1)}, scoring=progress_scorer, cv=inner_cv, n_jobs=n_jobs_inner, error_score='raise')
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "y residual is constant", UserWarning); grid_search.fit(X_train, y_train)
            dummy_model_fold = DummyRegressor(strategy='mean').fit(X_train_full, y_train_full)
            y_true_pooled.extend(y_test); y_pred_pls_pooled.extend(grid_search.best_estimator_.predict(X_test).flatten()); y_pred_dummy_pooled.extend(dummy_model_fold.predict(X_test))
        
        if not y_true_pooled: return {'status': 'FAILED', 'id': inversion_id, 'reason': "Nested CV failed on all folds."}
        y_true_arr, y_pred_arr = np.array(y_true_pooled), np.array(y_pred_pls_pooled)
        corr = 0.0 if (y_true_arr.size > 1 and np.all(y_true_arr == y_true_arr[0])) or (y_pred_arr.size > 1 and np.all(y_pred_arr == y_pred_arr[0])) else pearsonr(y_true_arr, y_pred_arr)[0]
        pearson_r2 = (corr if not np.isnan(corr) else 0.0)**2
        _, p_value = wilcoxon(np.abs(y_true_arr - y_pred_arr), np.abs(y_true_arr - np.array(y_pred_dummy_pooled)), alternative='less', zero_method='zsplit')

        X_synth_final, y_synth_final = create_synthetic_data(X_hap1, X_hap2, preloaded_data['raw_gts'], confidence_mask, X_full)
        final_train_X_parts, final_train_y_parts = [X_full], [y_full]
        if X_synth_final is not None: final_train_X_parts.append(X_synth_final); final_train_y_parts.append(y_synth_final)
        X_final_train_full, y_final_train_full = np.vstack(final_train_X_parts), np.concatenate(final_train_y_parts)
        final_sample_weights = compute_sample_weight("balanced", y=y_final_train_full)
        final_resampled_indices = rng.choice(len(X_final_train_full), size=len(X_final_train_full), replace=True, p=final_sample_weights / np.sum(final_sample_weights))
        X_final_train, y_final_train = X_final_train_full[final_resampled_indices], y_final_train_full[final_resampled_indices]
        final_min_class_count = min(Counter(y_final_train).values()) if y_final_train.size > 0 else 0
        if final_min_class_count < 2: return {'status': 'FAILED', 'id': inversion_id, 'reason': "Final balanced training set lacks class diversity."}
        final_max_components = min(100, X_final_train.shape[1], X_final_train.shape[0] - 1)
        final_effective_max = get_effective_max_components(X_final_train, y_final_train, final_max_components)
        if final_effective_max < 1: return {'status': 'FAILED', 'id': inversion_id, 'reason': "Final model training range is invalid."}
        final_cv = StratifiedKFold(n_splits=min(3, final_min_class_count), shuffle=True, random_state=42)
        final_pipeline = Pipeline([('pls', PLSRegression())])
        final_grid_search = GridSearchCV(estimator=final_pipeline, param_grid={'pls__n_components': range(1, final_effective_max + 1)}, scoring='neg_mean_squared_error', cv=final_cv, n_jobs=n_jobs_inner, refit=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "y residual is constant", UserWarning); final_grid_search.fit(X_final_train, y_final_train)
        output_dir = "final_imputation_models"
        os.makedirs(output_dir, exist_ok=True)
        model_filename = os.path.join(output_dir, f"{inversion_id}.model.joblib")
        dump(final_grid_search.best_estimator_, model_filename)
        pd.DataFrame(preloaded_data['snp_metadata']).to_json(os.path.join(output_dir, f"{inversion_id}.snps.json"), orient='records')
        return {'status': 'SUCCESS', 'id': inversion_id, 'unbiased_pearson_r2': pearson_r2, 'unbiased_rmse': np.sqrt(mean_squared_error(y_true_pooled, y_pred_pls_pooled)), 'model_p_value': p_value, 'best_n_components': final_grid_search.best_params_['pls__n_components'], 'num_snps_in_model': X_full.shape[1], 'model_path': model_filename}
    except Exception as e:
        import traceback
        return {'status': 'FAILED', 'id': inversion_id, 'reason': f"Analysis Error: {type(e).__name__}: {e}\n{traceback.format_exc()}"}

def process_locus_end_to_end(job, progress_queue):
    """
    Complete end-to-end processing for one locus.
    Handles data extraction and, if successful, analysis with progress updates.
    """
    # Stage 1: Extraction
    preloaded_data = extract_haplotype_data_for_locus(job)
    if preloaded_data.get('status') != 'PREPROCESSED':
        return preloaded_data # Return failure/skip result immediately

    # Stage 2: Analysis
    # Dynamically calculate the number of steps for THIS locus and update the global total
    try:
        y_full, confidence_mask = preloaded_data['y_diploid'], preloaded_data['confidence_mask']
        X_full = preloaded_data['X_hap1'] + preloaded_data['X_hap2']
        y_val = y_full[confidence_mask]
        val_min_class_count = min(Counter(y_val).values()) if y_val.size > 0 else 0
        if val_min_class_count < 2:
            return analyze_and_model_locus_pls(preloaded_data) # Let the main func handle the skip

        n_outer_splits = min(5, val_min_class_count)
        # This is a good faith estimate of inner loop iterations
        train_min_class_count_est = 3 # Assume at least 3 for a robust estimate
        n_inner_splits = min(3, train_min_class_count_est)
        n_params_est = min(50, X_full.shape[1]) # Rough estimate of n_components
        
        total_locus_steps = n_outer_splits * n_inner_splits * n_params_est
        progress_queue.put(('ADD_TOTAL', total_locus_steps))
    except Exception:
        # If calculation fails, just add 0 and let the analysis function fail formally
        progress_queue.put(('ADD_TOTAL', 0))

    progress_scorer = make_progress_scorer('neg_mean_squared_error', progress_queue)
    return analyze_and_model_locus_pls(preloaded_data, progress_scorer=progress_scorer)

def progress_listener(queue, pbar):
    """Listens for messages and updates the progress bar total and value."""
    while True:
        message = queue.get()
        if message == 'STOP': break
        
        msg_type, value = message
        if msg_type == 'ADD_TOTAL':
            pbar.total += value
            pbar.refresh()
        elif msg_type == 'UPDATE':
            pbar.update(value)

if __name__ == '__main__':
    script_start_time = time.time()
    logging.info("--- Starting PLS-Based Inversion Imputation Model Pipeline ---")
    
    output_dir = "final_imputation_models"
    ground_truth_file = "../variants_freeze4inv_sv_inv_hg38_processed_arbigent_filtered_manualDotplot_filtered_PAVgenAdded_withInvCategs_syncWithWH.fixedPH.simpleINV.mod.tsv"
    
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(ground_truth_file): 
        logging.critical(f"FATAL: Ground-truth file not found: '{ground_truth_file}'"); sys.exit(1)
    
    config_df = pd.read_csv(ground_truth_file, sep='\t', on_bad_lines='warn', dtype={'seqnames': str})
    config_df = config_df[(config_df['verdict'] == 'pass') & (~config_df['seqnames'].isin(['chrY', 'chrM']))].copy()
    all_jobs = config_df.to_dict('records')
    if not all_jobs: 
        logging.warning("No valid inversions to process. Exiting."); sys.exit(0)

    total_cores = cpu_count()
    outer_jobs = total_cores
    logging.info(f"Loaded {len(all_jobs)} inversions. Using {outer_jobs} cores for parallel processing.")
    
    with Manager() as manager:
        progress_queue = manager.Queue()
        # Initialize pbar with total=0; it will be built dynamically.
        with tqdm(total=0, desc="Processing loci", unit="iter") as pbar:
            listener_thread = threading.Thread(target=progress_listener, args=(progress_queue, pbar))
            listener_thread.daemon = True
            listener_thread.start()

            try:
                # SINGLE parallel call with the end-to-end wrapper function
                all_results = Parallel(n_jobs=outer_jobs, backend='loky')(
                    delayed(process_locus_end_to_end)(job, progress_queue) for job in all_jobs
                )
            finally:
                # Clean shutdown
                progress_queue.put('STOP')
                listener_thread.join()

    logging.info(f"--- All Processing Complete in {time.time() - script_start_time:.2f} seconds ---")
    successful_runs = [r for r in all_results if r and r.get('status') == 'SUCCESS']
    
    print("\n\n" + "="*100 + "\n---      FINAL PLS IMPUTATION MODEL REPORT      ---\n" + "="*100)
    reason_counts = Counter(f"({r.get('status', 'N/A')}) {r.get('reason', 'N/A').splitlines()[0]}" for r in all_results if r.get('status') != 'SUCCESS')
    if reason_counts:
        print("\n--- FAILED OR SKIPPED LOCI SUMMARY ---")
        for reason, count in sorted(reason_counts.items(), key=lambda item: item[1], reverse=True): 
            print(f"  - ({count: >3} loci): {reason}")

    if successful_runs:
        results_df = pd.DataFrame(successful_runs).set_index('id').sort_values('unbiased_pearson_r2', ascending=False)
        print(f"\n--- Aggregate Performance ({len(successful_runs)} Successful Models) ---")
        print(f"  Mean Unbiased Pearson r²: {results_df['unbiased_pearson_r2'].mean():.4f}")
        print(f"  Models with Est. r² > 0.5: {(results_df['unbiased_pearson_r2'] > 0.5).sum()}")
        print(f"  Models with p < 0.05 (vs. baseline): {(results_df['model_p_value'] < 0.05).sum()}")
        high_perf_df = results_df[(results_df['unbiased_pearson_r2'] > 0.5) & (results_df['model_p_value'] < 0.05)]
        if not high_perf_df.empty:
            summary_cols = ['unbiased_pearson_r2', 'unbiased_rmse', 'model_p_value', 'best_n_components', 'num_snps_in_model']
            summary_filename = os.path.join(output_dir, "high_performance_pls_models_summary.tsv")
            high_perf_df[summary_cols].to_csv(summary_filename, sep='\t', float_format='%.4g')
            print(f"\n--- High-Performance Models (Est. Unbiased r² > 0.5 & p < 0.05) ---")
            with pd.option_context('display.max_rows', None, 'display.width', 120): print(high_perf_df[summary_cols])
            print(f"\n[SUCCESS] Saved summary for {len(high_perf_df)} models to '{summary_filename}'")
        else:
            print("\n--- No high-performance models found meeting the r² > 0.5 criteria ---")
    else:
        print("\n--- No successful models were generated across all jobs. ---")
    print("\n" + "="*100)
