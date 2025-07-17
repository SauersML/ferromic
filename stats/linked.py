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

# Scikit-learn for modeling, evaluation, and preprocessing
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from scipy.stats import wilcoxon, pearsonr

# --- SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class FatalSampleMappingError(Exception):
    """Custom exception for critical, unrecoverable errors in sample mapping."""
    pass

# --- DATA EXTRACTION FUNCTION ---
def extract_diploid_data_for_locus(inversion_job: dict):
    inversion_id = inversion_job.get('orig_ID', 'Unknown_ID')
    try:
        chrom, start, end = inversion_job['seqnames'], inversion_job['start'], inversion_job['end']
        vcf_path = f"../vcfs/{chrom}.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz"
        if not os.path.exists(vcf_path):
            return {'status': 'FAILED', 'id': inversion_id, 'reason': f"VCF file not found: {vcf_path}"}

        vcf_reader = VCF(vcf_path, lazy=True)
        vcf_samples = vcf_reader.samples
        tsv_samples = [col for col in inversion_job.keys() if col.startswith(('HG', 'NA'))]

        # --- Robust Sample Mapping with Ambiguity Detection ---
        sample_map = {}
        ambiguous_mappings = {}
        for tsv_s in tsv_samples:
            # Find all VCF samples where the TSV ID is a substring.
            potential_matches = [vcf_s for vcf_s in vcf_samples if tsv_s in vcf_s]
            if len(potential_matches) == 1:
                sample_map[tsv_s] = potential_matches[0]
            elif len(potential_matches) > 1:
                # This is the ambiguous case we must prevent.
                ambiguous_mappings[tsv_s] = potential_matches

        if ambiguous_mappings:
            error_report = [f"'{tsv_id}' matched {vcf_ids}" for tsv_id, vcf_ids in ambiguous_mappings.items()]
            raise FatalSampleMappingError(f"Ambiguous sample IDs found for {inversion_id}. "
                                          f"Cannot proceed due to risk of data corruption. Details: {'; '.join(error_report)}")

        # Check mapping rate.
        mapping_rate = len(sample_map) / len(tsv_samples) if tsv_samples else 0
        if not str(chrom).endswith('X') and mapping_rate < 0.5:
            raise FatalSampleMappingError(f"Autosomal sample mapping rate ({mapping_rate:.1%}) for {chrom} was below 50%.")
        if not sample_map:
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': "No TSV samples could be mapped to VCF samples."}

        def parse_hap_gt_with_confidence(gt_str: any):
            """
            Parses a genotype string, identifying high and low confidence calls.
            High confidence (for validation & training): '0|0', '0|1', '1|0', '1|1'.
            Low confidence (for training only): '0|0_lowconf', '1|0_lowconf', '0|1_lowconf', '1|1_lowconf'.
            Returns a tuple (dosage, is_high_confidence) or (None, None) if invalid.
            """
            high_conf_map = {
                "0|0": 0, "1|0": 1, "0|1": 1, "1|1": 2
            }
            low_conf_map = {
                "0|0_lowconf": 0, "1|0_lowconf": 1, "0|1_lowconf": 1, "1|1_lowconf": 2
            }
            
            if gt_str in high_conf_map:
                return (high_conf_map[gt_str], True)
            if gt_str in low_conf_map:
                return (low_conf_map[gt_str], False)
            
            return (None, None)

        # Create a map from VCF sample ID to its ground truth dosage and confidence level.
        gt_data = {}
        for tsv_s, vcf_s in sample_map.items():
            dosage, is_high_conf = parse_hap_gt_with_confidence(inversion_job.get(tsv_s))
            if dosage is not None:
                gt_data[vcf_s] = {'dosage': dosage, 'is_high_conf': is_high_conf}
        
        if not gt_data:
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'No samples with a valid inversion dosage.'}

        gt_df = pd.DataFrame.from_dict(gt_data, orient='index')

        # Define flanking regions for SNP extraction.
        flank_size = 50000
        if (start + flank_size) >= (end - flank_size): # Overlapping flanks
            regions = [f"{chrom}:{max(0, start - flank_size)}-{end + flank_size}"]
        else:
            regions = [f"{chrom}:{max(0, start - flank_size)}-{start + flank_size}",
                       f"{chrom}:{end - flank_size}-{end + flank_size}"]

        # Extract SNP data for all mapped samples.
        vcf_subset = VCF(vcf_path, samples=list(gt_df.index))
        vcf_samples_ordered = vcf_subset.samples # This order defines the columns of our SNP matrix.
        snp_data_list, snp_metadata, processed_positions = [], [], set()

        for region_str in regions:
            for variant in vcf_subset(region_str):
                if variant.POS in processed_positions: continue
                if variant.is_snp and not variant.is_indel and len(variant.ALT) == 1:
                    dosages = [(gt[0] + gt[1]) if -1 not in gt[:2] else np.nan for gt in variant.genotypes]
                    if (np.isnan(dosages).sum() / len(dosages)) < 0.05:
                        snp_data_list.append(dosages)
                        snp_metadata.append({'id': variant.ID or f"{variant.CHROM}:{variant.POS}", 'pos': variant.POS})
                        processed_positions.add(variant.POS)

        if not snp_metadata:
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'No suitable SNPs found in flanking regions.'}

        # Align X and y using pandas for safety and clarity.
        df_X = pd.DataFrame(np.array(snp_data_list).T, index=vcf_samples_ordered)
        
        # Filter both X (SNP data) and y (dosages/confidence) to the common set of samples.
        y_series = gt_df['dosage']
        conf_series = gt_df['is_high_conf'].astype(bool)
        common_samples = df_X.index.intersection(y_series.index)

        if len(common_samples) < 20:
             return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Insufficient overlapping samples ({len(common_samples)}) for modeling.'}

        # Check for minimum number of high-confidence samples needed for validation folds.
        num_high_conf = conf_series.loc[common_samples].sum()
        if num_high_conf < 20:
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Insufficient high-confidence samples ({num_high_conf}) for validation.'}

        df_X_final = df_X.loc[common_samples]
        y_final = y_series.loc[common_samples].values
        confidence_final = conf_series.loc[common_samples].values

        return {
            'status': 'PREPROCESSED', 'id': inversion_id,
            'y_diploid': y_final.astype(int),
            'X_diploid': df_X_final.values,
            'confidence_mask': confidence_final, # Boolean array: True=HighConf, False=LowConf
            'snp_metadata': snp_metadata
        }
    except Exception as e:
        return {'status': 'FAILED', 'id': inversion_id, 'reason': f"Data Extraction Error: {type(e).__name__}: {e}"}
        
# --- CORE MODELING AND EVALUATION FUNCTION ---
def analyze_and_model_locus_pls(preloaded_data: dict, n_jobs_inner: int = 1):
    """
    Performs nested cross-validation, model evaluation, and final model training.
    Uses high-confidence data for validation and all data (high+low conf) for training.
    """
    inversion_id = preloaded_data['id']
    try:
        y_full, X_full = preloaded_data['y_diploid'], preloaded_data['X_diploid']
        confidence_mask = preloaded_data['confidence_mask']
        snp_meta = preloaded_data['snp_metadata']

        # --- Define Datasets ---
        # Validation set (high-confidence only) for unbiased performance estimation.
        X_val, y_val = X_full[confidence_mask], y_full[confidence_mask]
        # Low-confidence set to be added to training data in each fold.
        X_lowconf, y_lowconf = X_full[~confidence_mask], y_full[~confidence_mask]

        # --- Pre-computation Sanity Checks (on the validation set) ---
        if X_val.shape[0] < 20 or X_val.shape[1] < 1:
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Insufficient high-conf data (samples={X_val.shape[0]}, snps={X_val.shape[1]})'}
        if len(np.unique(y_val)) < 2:
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'Only one inversion dosage class present in validation set.'}
        
        val_min_class_count = min(Counter(y_val).values())
        if val_min_class_count < 2:
            return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Validation set minority class count ({val_min_class_count}) is less than 2.'}

        # --- Nested Cross-Validation for Unbiased Performance Estimation ---
        outer_cv_folds = min(5, val_min_class_count)
        outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=42)
        
        pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('pls', PLSRegression())])

        y_true_pooled, y_pred_pls_pooled, y_pred_dummy_pooled = [], [], []
        fold_failures = []

        # The outer loop splits the VALIDATION data to create unbiased test sets.
        for i, (train_val_idx, test_val_idx) in enumerate(outer_cv.split(X_val, y_val)):
            # Test set is purely high-confidence, held-out samples.
            X_test, y_test = X_val[test_val_idx], y_val[test_val_idx]

            # The training set for this fold combines the high-confidence training split
            # with ALL available low-confidence data to improve model robustness.
            X_train_val, y_train_val = X_val[train_val_idx], y_val[train_val_idx]
            
            if X_lowconf.shape[0] > 0:
                X_train = np.vstack((X_train_val, X_lowconf))
                y_train = np.concatenate((y_train_val, y_lowconf))
            else:
                X_train, y_train = X_train_val, y_train_val

            train_min_class_count = min(Counter(y_train).values()) if len(y_train) > 0 else 0
            if train_min_class_count < 2:
                fold_failures.append(f"Fold {i+1}: Combined training split has < 2 samples in minority class.")
                continue
            
            max_components = min(30, X_train.shape[1], X_train.shape[0] - 1)
            if max_components < 1:
                fold_failures.append(f"Fold {i+1}: Not enough training samples to determine PLS components.")
                continue

            inner_cv_folds = min(3, train_min_class_count)
            inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=123)
            param_grid = {'pls__n_components': range(1, max_components + 1)}
            
            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='neg_mean_squared_error',
                                       cv=inner_cv, n_jobs=n_jobs_inner, error_score='raise')
            try:
                grid_search.fit(X_train, y_train)
            except ValueError as e:
                fold_failures.append(f"Fold {i+1}: GridSearchCV failed: {e}")
                continue

            # Dummy model is trained on the same augmented data for a fair baseline.
            dummy_model_fold = DummyRegressor(strategy='mean').fit(X_train, y_train)
            
            # Pool out-of-sample predictions (on the high-confidence test set).
            y_true_pooled.extend(y_test)
            y_pred_pls_pooled.extend(grid_search.best_estimator_.predict(X_test).flatten())
            y_pred_dummy_pooled.extend(dummy_model_fold.predict(X_test))

        if not y_true_pooled:
            reason = "; ".join(list(set(fold_failures))) or "Unknown CV failure."
            return {'status': 'FAILED', 'id': inversion_id, 'reason': f"Nested CV failed on all folds. Reasons: {reason}"}

        # --- Calculate Unbiased Performance Metrics from Pooled Predictions ---
        y_true_pooled = np.array(y_true_pooled)
        y_pred_pls_pooled = np.array(y_pred_pls_pooled)
        
        if np.std(y_pred_pls_pooled) < 1e-6 or np.std(y_true_pooled) < 1e-6:
            pearson_r2 = 0.0
        else:
            corr, _ = pearsonr(y_true_pooled, y_pred_pls_pooled)
            pearson_r2 = corr**2
        
        unbiased_rmse = np.sqrt(mean_squared_error(y_true_pooled, y_pred_pls_pooled))

        errors_pls = np.abs(y_true_pooled - y_pred_pls_pooled)
        errors_dummy = np.abs(np.array(y_pred_dummy_pooled) - y_true_pooled)
        try:
            _, p_value = wilcoxon(errors_pls, errors_dummy, alternative='less', zero_method='zsplit')
        except ValueError:
            p_value = 1.0

        # --- Train and Save the Final Model on All Data (High + Low Confidence) ---
        output_dir = "final_imputation_models"
        full_data_min_class_count = min(Counter(y_full).values())
        if full_data_min_class_count < 2:
            return {'status': 'FAILED', 'id': inversion_id, 'reason': f'Full dataset minority class count ({full_data_min_class_count}) is < 2.'}

        final_cv = StratifiedKFold(n_splits=min(3, full_data_min_class_count), shuffle=True, random_state=42)
        final_max_components = min(30, X_full.shape[1], X_full.shape[0] - 1)
        if final_max_components < 1:
            return {'status': 'FAILED', 'id': inversion_id, 'reason': 'Not enough data for final model build.'}
        
        final_param_grid = {'pls__n_components': range(1, final_max_components + 1)}
        final_grid_search = GridSearchCV(estimator=pipeline, param_grid=final_param_grid, scoring='neg_mean_squared_error',
                                         cv=final_cv, n_jobs=n_jobs_inner, refit=True)
        final_grid_search.fit(X_full, y_full)
        
        os.makedirs(output_dir, exist_ok=True)
        model_filename = os.path.join(output_dir, f"{inversion_id}.model.joblib")
        snp_list_filename = os.path.join(output_dir, f"{inversion_id}.snps.json")
        
        dump(final_grid_search.best_estimator_, model_filename)
        pd.DataFrame(snp_meta).to_json(snp_list_filename, orient='records', indent=4)

        return {
            'status': 'SUCCESS', 'id': inversion_id,
            'unbiased_rmse': unbiased_rmse,
            'unbiased_pearson_r2': pearson_r2,
            'model_p_value': p_value,
            'best_n_components': final_grid_search.best_params_['pls__n_components'],
            'num_snps_in_model': X_full.shape[1], 'model_path': model_filename
        }
    except Exception as e:
        import traceback
        return {'status': 'FAILED', 'id': inversion_id, 'reason': f"Analysis Error: {type(e).__name__}: {e}\n{traceback.format_exc()}"}

# --- MAIN ORCHESTRATOR ---
if __name__ == '__main__':
    logging.info("--- Starting PLS-Based Inversion Imputation Model Pipeline ---")
    start_time = time.time()
    
    # --- Hardcoded Configuration ---
    output_dir = "final_imputation_models"
    ground_truth_file = "../variants_freeze4inv_sv_inv_hg38_processed_arbigent_filtered_manualDotplot_filtered_PAVgenAdded_withInvCategs_syncWithWH.fixedPH.simpleINV.mod.tsv"
    vcf_base_dir = "../vcfs/"

    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(ground_truth_file): 
        logging.critical(f"FATAL: Ground-truth file not found: '{ground_truth_file}'")
        sys.exit(1)
    
    config_df = pd.read_csv(ground_truth_file, sep='\t', on_bad_lines='warn', dtype={'seqnames': str})
    config_df = config_df[(config_df['verdict'] == 'pass') & (~config_df['seqnames'].isin(['chrY', 'chrM']))].copy()
    
    # Pre-flight check: Ensure VCF files are indexed with tabix.
    for chrom in config_df['seqnames'].unique():
        vcf_path = os.path.join(vcf_base_dir, f"{chrom}.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz")
        if os.path.exists(vcf_path) and not os.path.exists(f"{vcf_path}.tbi"):
            logging.info(f"Indexing VCF file: {vcf_path}")
            try: 
                subprocess.run(['tabix', '-p', 'vcf', vcf_path], check=True, capture_output=True, text=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e: 
                logging.warning(f"Could not index {vcf_path}. 'tabix' may not be in PATH or file may be invalid. Error: {e}")

    all_jobs = config_df.to_dict('records')
    num_jobs = len(all_jobs)
    if not all_jobs: 
        logging.warning("No valid inversions to process after filtering. Exiting.")
        sys.exit(0)

    num_procs = max(1, cpu_count() // 2)
    batch_size = num_procs * 4
    logging.info(f"Loaded {num_jobs} inversions. Processing in batches of {batch_size} using {num_procs} cores.")
    
    all_results = []
    for i in range(0, num_jobs, batch_size):
        batch_jobs = all_jobs[i:min(i + batch_size, num_jobs)]
        logging.info(f"--- Starting Batch {i//batch_size + 1}/{(num_jobs + batch_size - 1) // batch_size} (Inversions {i+1}-{min(i + batch_size, num_jobs)}) ---")
        
        # Phase 1: Data extraction.
        logging.info(f"Phase 1: Parallel extraction of data for {len(batch_jobs)} inversions...")
        with Parallel(n_jobs=num_procs, backend='loky') as parallel:
            preloaded_data_batch = parallel(delayed(extract_diploid_data_for_locus)(job) for job in batch_jobs)
        
        successful_loads = [d for d in preloaded_data_batch if d.get('status') == 'PREPROCESSED']
        all_results.extend([d for d in preloaded_data_batch if d.get('status') != 'PREPROCESSED'])

        # Phase 2: Analysis of successfully loaded data.
        if successful_loads:
            logging.info(f"Phase 2: Parallel analysis of {len(successful_loads)} pre-loaded inversions...")
            with Parallel(n_jobs=num_procs, backend='loky') as parallel:
                analysis_results = parallel(delayed(analyze_and_model_locus_pls)(data) for data in successful_loads)
                all_results.extend(analysis_results)
        
        logging.info(f"--- Finished Batch {i//batch_size + 1} ---")

    logging.info(f"--- All Batches Complete in {time.time() - start_time:.2f} seconds ---")
    
    # --- Final Report Generation ---
    successful_runs = [r for r in all_results if r and r.get('status') == 'SUCCESS']
    print("\n\n" + "="*100 + "\n---                  FINAL PLS IMPUTATION MODEL REPORT                  ---\n" + "="*100)
    
    reason_counts = Counter(f"({res.get('status', 'N/A')}) {res.get('reason', 'N/A').splitlines()[0]}" for res in all_results if res.get('status') != 'SUCCESS')
    if reason_counts:
        print("\n--- FAILED OR SKIPPED LOCI SUMMARY ---")
        for reason, count in sorted(reason_counts.items(), key=lambda item: item[1], reverse=True): 
            print(f"  - ({count: >3} loci): {reason}")

    if successful_runs:
        results_df = pd.DataFrame(successful_runs).set_index('id')
        results_df.sort_values('unbiased_pearson_r2', ascending=False, inplace=True)
        
        print(f"\n--- Aggregate Performance ({len(successful_runs)} Successful Models) ---")
        print(f"  Mean Unbiased Pearson r²: {results_df['unbiased_pearson_r2'].mean():.4f} (estimated from nested cross-validation)")
        print(f"  Median Unbiased Pearson r²: {results_df['unbiased_pearson_r2'].median():.4f}")
        print(f"  Models with Est. r² > 0.3: {(results_df['unbiased_pearson_r2'] > 0.3).sum()}")
        print(f"  Models with p < 0.05 (vs. baseline): {(results_df['model_p_value'] < 0.05).sum()}")

        high_perf_df = results_df[(results_df['unbiased_pearson_r2'] > 0.3) & (results_df['model_p_value'] < 0.05)]
        
        if not high_perf_df.empty:
            summary_cols = ['unbiased_pearson_r2', 'unbiased_rmse', 'model_p_value', 'best_n_components', 'num_snps_in_model']
            summary_df = high_perf_df[summary_cols]
            summary_filename = os.path.join(output_dir, "high_performance_pls_models_summary.tsv")
            summary_df.to_csv(summary_filename, sep='\t', float_format='%.4g')
            print(f"\n--- High-Performance Models (Est. Unbiased r² > 0.3 & p < 0.05) ---")
            with pd.option_context('display.max_rows', None, 'display.width', 120):
                print(summary_df)
            print(f"\n[SUCCESS] Saved summary for {len(high_perf_df)} models to '{summary_filename}'")
            print(f"[SUCCESS] Full model objects and SNP lists saved in '{output_dir}/' directory.")
        else: 
            print("\n--- No high-performance models found meeting the specified criteria ---")
    else:
        print("\n--- No successful models were generated across all jobs. ---")
    print("\n" + "="*100)
