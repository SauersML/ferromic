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

# Scikit-learn for modeling, evaluation, and preprocessing
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from scipy.stats import wilcoxon, pearsonr

# --- SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Use a modern random number generator
rng = np.random.default_rng(seed=42)

class FatalSampleMappingError(Exception):
    """Custom exception for critical, unrecoverable errors in sample mapping."""
    pass

# --- SYNTHETIC DATA GENERATION FUNCTION ---
def create_synthetic_data(X_hap1: np.ndarray, X_hap2: np.ndarray, raw_gts: pd.Series, confidence_mask: np.ndarray, num_total_real_samples: int):
    """
    Creates synthetic diploid genomes by mixing existing high-confidence haplotypes.
    
    Args:
        X_hap1: Numpy array of first haplotypes for all real samples (samples x SNPs).
        X_hap2: Numpy array of second haplotypes for all real samples (samples x SNPs).
        raw_gts: Pandas Series of raw genotype strings ('0|0', '0|1', etc.) for real samples.
        confidence_mask: Boolean array indicating high-confidence real samples to be used for pooling.
        num_total_real_samples: The total number of real samples to match for the synthetic set size.

    Returns:
        A tuple (X_synth, y_synth) of numpy arrays for synthetic data, or (None, None) if creation fails.
    """
    # Use only haplotypes from the provided confidence_mask to build the pools
    if not np.any(confidence_mask):
        return None, None
    
    X_h1_hc = X_hap1[confidence_mask]
    X_h2_hc = X_hap2[confidence_mask]
    gts_hc = raw_gts[confidence_mask]

    hap_pool_0, hap_pool_1 = [], []
    for i, gt in enumerate(gts_hc):
        if gt == '0|0':
            hap_pool_0.append(X_h1_hc[i])
            hap_pool_0.append(X_h2_hc[i])
        elif gt == '1|1':
            hap_pool_1.append(X_h1_hc[i])
            hap_pool_1.append(X_h2_hc[i])
        elif gt == '0|1':
            hap_pool_0.append(X_h1_hc[i])
            hap_pool_1.append(X_h2_hc[i])
        elif gt == '1|0':
            hap_pool_1.append(X_h1_hc[i])
            hap_pool_0.append(X_h2_hc[i])

    # Check if pools are sufficient to create all genotype classes
    if len(hap_pool_0) < 2 or len(hap_pool_1) < 2:
        logging.warning(f"Insufficient haplotype diversity to create balanced synthetic data. "
                        f"Pool sizes: Non-Inverted={len(hap_pool_0)}, Inverted={len(hap_pool_1)}. Skipping augmentation.")
        return None, None

    n_synth_total = num_total_real_samples
    n_per_class = n_synth_total // 3
    
    X_synth, y_synth = [], []
    
    # Generate Dosage 2 (1|1)
    for _ in range(n_per_class):
        indices = rng.choice(len(hap_pool_1), size=2, replace=False)
        new_diploid_dosage = hap_pool_1[indices[0]] + hap_pool_1[indices[1]]
        X_synth.append(new_diploid_dosage)
        y_synth.append(2)

    # Generate Dosage 0 (0|0)
    for _ in range(n_per_class):
        indices = rng.choice(len(hap_pool_0), size=2, replace=False)
        new_diploid_dosage = hap_pool_0[indices[0]] + hap_pool_0[indices[1]]
        X_synth.append(new_diploid_dosage)
        y_synth.append(0)

    # Generate Dosage 1 (0|1) - Remainder goes here to match total
    n_rem = n_synth_total - (2 * n_per_class)
    for _ in range(n_rem):
        idx0 = rng.choice(len(hap_pool_0))
        idx1 = rng.choice(len(hap_pool_1))
        new_diploid_dosage = hap_pool_0[idx0] + hap_pool_1[idx1]
        X_synth.append(new_diploid_dosage)
        y_synth.append(1)

    logging.info(f"Successfully generated {len(y_synth)} synthetic samples for this fold.")
    return np.array(X_synth), np.array(y_synth)


# --- DATA EXTRACTION FUNCTION ---
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

        # --- Robust Sample Mapping ---
        sample_map = {ts: p[0] for ts in tsv_samples if len(p := [vs for vs in vcf_samples if ts in vs]) == 1}
        if not sample_map:
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': "No TSV samples could be mapped to VCF samples."}

        # --- Parse Ground Truth Genotypes and Confidence ---
        def parse_gt_for_synth(gt_str: any):
            high_conf_map = {"0|0": 0, "1|0": 1, "0|1": 1, "1|1": 2}
            low_conf_map = {"0|0_lowconf": 0, "1|0_lowconf": 1, "0|1_lowconf": 1, "1|1_lowconf": 2}
            
            if gt_str in high_conf_map:
                return (high_conf_map[gt_str], True, gt_str) # Dosage, IsHighConf, RawGT
            if gt_str in low_conf_map:
                return (low_conf_map[gt_str], False, None) # Low conf samples don't contribute to pools
            return (None, None, None)

        gt_data = {}
        for tsv_s, vcf_s in sample_map.items():
            dosage, is_high_conf, raw_gt = parse_gt_for_synth(inversion_job.get(tsv_s))
            if dosage is not None:
                gt_data[vcf_s] = {'dosage': dosage, 'is_high_conf': is_high_conf, 'raw_gt': raw_gt}
        
        if not gt_data:
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'No samples with a valid inversion dosage.'}
        gt_df = pd.DataFrame.from_dict(gt_data, orient='index')

        # --- Extract Phased SNP Haplotypes ---
        flank_size = 50000
        regions = [f"{chrom}:{max(0, start - flank_size)}-{end + flank_size}"]

        vcf_subset = VCF(vcf_path, samples=list(gt_df.index))
        vcf_samples_ordered = vcf_subset.samples
        
        h1_data, h2_data, snp_meta, processed_pos = [], [], [], set()

        for region_str in regions:
            for var in vcf_subset(region_str):
                if var.POS in processed_pos: continue
                if var.is_snp and not var.is_indel and len(var.ALT) == 1:
                    # Skip any SNP with even one missing genotype call
                    if any(-1 in gt[:2] for gt in var.genotypes):
                        continue
                    
                    h1_data.append([gt[0] for gt in var.genotypes])
                    h2_data.append([gt[1] for gt in var.genotypes])
                    snp_meta.append({'id': var.ID or f"{var.CHROM}:{var.POS}", 'pos': var.POS})
                    processed_pos.add(var.POS)

        if not snp_meta:
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'No suitable SNPs (with 100% call rate) found.'}

        # --- Align and Finalize Data ---
        df_H1 = pd.DataFrame(np.array(h1_data).T, index=vcf_samples_ordered)
        df_H2 = pd.DataFrame(np.array(h2_data).T, index=vcf_samples_ordered)
        
        common_samples = df_H1.index.intersection(gt_df.index)
        if len(common_samples) < 20:
             return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Insufficient overlapping samples ({len(common_samples)}) for modeling.'}
        
        num_high_conf = gt_df.loc[common_samples, 'is_high_conf'].sum()
        if num_high_conf < 3: # Need enough high-conf samples to build pools and test folds
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Insufficient high-confidence samples ({num_high_conf}) for haplotype pooling.'}

        return {
            'status': 'PREPROCESSED', 'id': inversion_id,
            'X_hap1': df_H1.loc[common_samples].values,
            'X_hap2': df_H2.loc[common_samples].values,
            'y_diploid': gt_df.loc[common_samples, 'dosage'].values.astype(int),
            'confidence_mask': gt_df.loc[common_samples, 'is_high_conf'].values.astype(bool),
            'raw_gts': gt_df.loc[common_samples, 'raw_gt'],
            'snp_metadata': snp_meta
        }
    except Exception as e:
        return {'status': 'FAILED', 'id': inversion_id, 'reason': f"Data Extraction Error: {type(e).__name__}: {e}"}

# --- CORE MODELING AND EVALUATION FUNCTION ---
def analyze_and_model_locus_pls(preloaded_data: dict, n_jobs_inner: int = 1):
    inversion_id = preloaded_data['id']
    try:
        y_full, confidence_mask = preloaded_data['y_diploid'], preloaded_data['confidence_mask']
        X_hap1, X_hap2 = preloaded_data['X_hap1'], preloaded_data['X_hap2']
        X_full = X_hap1 + X_hap2 # Recreate diploid dosages
        
        # --- Create Datasets ---
        X_val, y_val = X_full[confidence_mask], y_full[confidence_mask] # For testing
        X_lowconf, y_lowconf = X_full[~confidence_mask], y_full[~confidence_mask] # For training

        # --- Sanity Checks on Validation Set ---
        if X_val.shape[0] < 20 or X_val.shape[1] < 1:
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Insufficient high-conf data (samples={X_val.shape[0]})'}
        val_min_class_count = min(Counter(y_val).values())
        if val_min_class_count < 2:
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Validation set minority class count ({val_min_class_count}) is < 2.'}

        # --- Nested Cross-Validation for Unbiased Performance Estimation ---
        outer_cv = StratifiedKFold(n_splits=min(5, val_min_class_count), shuffle=True, random_state=42)
        # Pipeline no longer needs an imputer since data extraction guarantees no missing values.
        pipeline = Pipeline([('pls', PLSRegression())])
        y_true_pooled, y_pred_pls_pooled, y_pred_dummy_pooled = [], [], []

        original_hc_indices = np.where(confidence_mask)[0]

        for i, (train_val_idx, test_val_idx) in enumerate(outer_cv.split(X_val, y_val)):
            # Test set is PURELY real, high-confidence, held-out samples
            X_test, y_test = X_val[test_val_idx], y_val[test_val_idx]

            # Assemble the training set for this fold
            X_train_val, y_train_val = X_val[train_val_idx], y_val[train_val_idx]
            

            # so not leak
            # Create a mask that is True only for the high-confidence samples in this training fold
            fold_training_original_indices = original_hc_indices[train_val_idx]
            fold_specific_training_mask = np.zeros_like(confidence_mask, dtype=bool)
            fold_specific_training_mask[fold_training_original_indices] = True
            
            X_synth_fold, y_synth_fold = create_synthetic_data(
                X_hap1, X_hap2, preloaded_data['raw_gts'], fold_specific_training_mask, num_total_real_samples=len(y_full)
            )
            use_synth_fold = X_synth_fold is not None
            
            # Combine real high-conf, real low-conf, and fold-specific synthetic data for training
            train_parts_X = [X_train_val, X_lowconf]
            train_parts_y = [y_train_val, y_lowconf]
            if use_synth_fold:
                train_parts_X.append(X_synth_fold)
                train_parts_y.append(y_synth_fold)
            
            X_train = np.vstack([p for p in train_parts_X if p.shape[0] > 0])
            y_train = np.concatenate([p for p in train_parts_y if p.shape[0] > 0])

            train_min_class_count = min(Counter(y_train).values()) if len(y_train) > 0 else 0
            if train_min_class_count < 2: continue
            
            max_components = min(1000, X_train.shape[1], X_train.shape[0] - 1)
            if max_components < 1: continue

            inner_cv = StratifiedKFold(n_splits=min(3, train_min_class_count), shuffle=True, random_state=123)
            
            # Note: The 'pls__' prefix is still needed because the model is inside a Pipeline object
            grid_search = GridSearchCV(
                estimator=pipeline, 
                param_grid={'pls__n_components': range(1, max_components + 1)}, 
                scoring='neg_mean_squared_error',
                cv=inner_cv, 
                n_jobs=n_jobs_inner, 
                error_score='raise'
            )
            grid_search.fit(X_train, y_train)
            
            dummy_model_fold = DummyRegressor(strategy='mean').fit(X_train, y_train)
            
            y_true_pooled.extend(y_test)
            y_pred_pls_pooled.extend(grid_search.best_estimator_.predict(X_test).flatten())
            y_pred_dummy_pooled.extend(dummy_model_fold.predict(X_test))

        if not y_true_pooled:
            return {'status': 'FAILED', 'id': inversion_id, 'reason': "Nested CV failed on all folds."}

        # --- Calculate Unbiased Performance Metrics ---
        corr, _ = pearsonr(y_true_pooled, y_pred_pls_pooled)
        pearson_r2 = corr**2 if not np.isnan(corr) else 0.0
        errors_pls = np.abs(np.array(y_true_pooled) - np.array(y_pred_pls_pooled))
        errors_dummy = np.abs(np.array(y_true_pooled) - np.array(y_pred_dummy_pooled))
        _, p_value = wilcoxon(errors_pls, errors_dummy, alternative='less', zero_method='zsplit')

        # --- Train and Save Final Model on ALL available data ---
        # Regenerate synthetic data using ALL high-confidence samples for the final model
        X_synth_final, y_synth_final = create_synthetic_data(
            X_hap1, X_hap2, preloaded_data['raw_gts'], confidence_mask, num_total_real_samples=len(y_full)
        )
        use_synth_final = X_synth_final is not None
        
        final_train_X_parts = [X_full]
        final_train_y_parts = [y_full]
        if use_synth_final:
            final_train_X_parts.append(X_synth_final)
            final_train_y_parts.append(y_synth_final)

        X_final_train = np.vstack(final_train_X_parts)
        y_final_train = np.concatenate(final_train_y_parts)
        
        final_min_class_count = min(Counter(y_final_train).values())
        final_cv = StratifiedKFold(n_splits=min(3, final_min_class_count), shuffle=True, random_state=42)
        final_max_components = min(1000, X_final_train.shape[1], X_final_train.shape[0] - 1)
        
        # Use a new GridSearchCV instance for the final model
        final_pipeline = Pipeline([('pls', PLSRegression())])
        final_grid_search = GridSearchCV(
            estimator=final_pipeline, 
            param_grid={'pls__n_components': range(1, final_max_components + 1)},
            scoring='neg_mean_squared_error', 
            cv=final_cv, 
            n_jobs=n_jobs_inner, 
            refit=True
        )
        final_grid_search.fit(X_final_train, y_final_train)
        
        output_dir = "final_imputation_models"
        os.makedirs(output_dir, exist_ok=True)
        model_filename = os.path.join(output_dir, f"{inversion_id}.model.joblib")
        dump(final_grid_search.best_estimator_, model_filename)
        pd.DataFrame(preloaded_data['snp_metadata']).to_json(os.path.join(output_dir, f"{inversion_id}.snps.json"), orient='records')

        return {
            'status': 'SUCCESS', 'id': inversion_id,
            'unbiased_pearson_r2': pearson_r2,
            'unbiased_rmse': np.sqrt(mean_squared_error(y_true_pooled, y_pred_pls_pooled)),
            'model_p_value': p_value,
            'best_n_components': final_grid_search.best_params_['pls__n_components'],
            'num_snps_in_model': X_full.shape[1], 'model_path': model_filename
        }
    except Exception as e:
        import traceback
        return {'status': 'FAILED', 'id': inversion_id, 'reason': f"Analysis Error: {type(e).__name__}: {e}\n{traceback.format_exc()}"}

# --- MAIN ORCHESTRATOR ---
if __name__ == '__main__':
    logging.info("--- Starting PLS-Based Inversion Imputation Model Pipeline (with Synthetic Data Augmentation) ---")
    start_time = time.time()
    
    # --- Configuration ---
    output_dir = "final_imputation_models"
    ground_truth_file = "../variants_freeze4inv_sv_inv_hg38_processed_arbigent_filtered_manualDotplot_filtered_PAVgenAdded_withInvCategs_syncWithWH.fixedPH.simpleINV.mod.tsv"
    
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(ground_truth_file): 
        logging.critical(f"FATAL: Ground-truth file not found: '{ground_truth_file}'"); sys.exit(1)
    
    config_df = pd.read_csv(ground_truth_file, sep='\t', on_bad_lines='warn', dtype={'seqnames': str})
    config_df = config_df[(config_df['verdict'] == 'pass') & (~config_df['seqnames'].isin(['chrY', 'chrM']))].copy()
    
    all_jobs = config_df.to_dict('records')
    num_jobs = len(all_jobs)
    if not all_jobs: 
        logging.warning("No valid inversions to process. Exiting."); sys.exit(0)

    num_procs = max(1, cpu_count() // 2)
    batch_size = num_procs * 2
    logging.info(f"Loaded {num_jobs} inversions. Processing in batches using {num_procs} cores.")
    
    all_results = []
    for i in range(0, num_jobs, batch_size):
        batch_jobs = all_jobs[i:min(i + batch_size, num_jobs)]
        logging.info(f"--- Starting Batch {i//batch_size + 1}/{(num_jobs + batch_size - 1) // batch_size} ---")
        
        # Phase 1: Data extraction.
        with Parallel(n_jobs=num_procs, backend='loky') as parallel:
            preloaded_data_batch = parallel(delayed(extract_haplotype_data_for_locus)(job) for job in batch_jobs)
        
        successful_loads = [d for d in preloaded_data_batch if d.get('status') == 'PREPROCESSED']
        all_results.extend([d for d in preloaded_data_batch if d.get('status') != 'PREPROCESSED'])

        # Phase 2: Analysis of successfully loaded data.
        if successful_loads:
            with Parallel(n_jobs=num_procs, backend='loky') as parallel:
                analysis_results = parallel(delayed(analyze_and_model_locus_pls)(data) for data in successful_loads)
                all_results.extend(analysis_results)
        
        logging.info(f"--- Finished Batch {i//batch_size + 1} ---")

    # --- Final Report ---
    logging.info(f"--- All Batches Complete in {time.time() - start_time:.2f} seconds ---")
    successful_runs = [r for r in all_results if r and r.get('status') == 'SUCCESS']
    print("\n\n" + "="*100 + "\n---      FINAL PLS IMPUTATION MODEL REPORT (with Synthetic Data)      ---\n" + "="*100)
    
    reason_counts = Counter(f"({r.get('status', 'N/A')}) {r.get('reason', 'N/A').splitlines()[0]}" for r in all_results if r.get('status') != 'SUCCESS')
    if reason_counts:
        print("\n--- FAILED OR SKIPPED LOCI SUMMARY ---")
        for reason, count in sorted(reason_counts.items(), key=lambda item: item[1], reverse=True): 
            print(f"  - ({count: >3} loci): {reason}")

    if successful_runs:
        results_df = pd.DataFrame(successful_runs).set_index('id').sort_values('unbiased_pearson_r2', ascending=False)
        
        print(f"\n--- Aggregate Performance ({len(successful_runs)} Successful Models) ---")
        print(f"  Mean Unbiased Pearson r²: {results_df['unbiased_pearson_r2'].mean():.4f} (estimated from nested CV on real data)")
        print(f"  Models with Est. r² > 0.5: {(results_df['unbiased_pearson_r2'] > 0.5).sum()}")
        print(f"  Models with p < 0.05 (vs. baseline): {(results_df['model_p_value'] < 0.05).sum()}")

        high_perf_df = results_df[(results_df['unbiased_pearson_r2'] > 0.5) & (results_df['model_p_value'] < 0.05)]
        
        if not high_perf_df.empty:
            summary_cols = ['unbiased_pearson_r2', 'unbiased_rmse', 'model_p_value', 'best_n_components', 'num_snps_in_model']
            summary_filename = os.path.join(output_dir, "high_performance_pls_models_summary.tsv")
            high_perf_df[summary_cols].to_csv(summary_filename, sep='\t', float_format='%.4g')
            print(f"\n--- High-Performance Models (Est. Unbiased r² > 0.5 & p < 0.05) ---")
            with pd.option_context('display.max_rows', None, 'display.width', 120):
                print(high_perf_df[summary_cols])
            print(f"\n[SUCCESS] Saved summary for {len(high_perf_df)} models to '{summary_filename}'")
        else: 
            print("\n--- No high-performance models found meeting the r² > 0.5 criteria ---")
    else:
        print("\n--- No successful models were generated across all jobs. ---")
    print("\n" + "="*100)
