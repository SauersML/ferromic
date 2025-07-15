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
from joblib import Parallel, delayed, cpu_count

# Scikit-learn for modeling and evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, ttest_ind

# TOML for saving detailed model parameters
try:
    import toml
except ImportError:
    print("The 'toml' library is not installed. Please install it using: pip install toml")
    sys.exit(1)


# --- SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class FatalSampleMappingError(Exception):
    pass

# --- CORE HELPER FUNCTIONS ---

def _safe_pearsonr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 1.0, 0.0
    r, p = pearsonr(x, y)
    return p, r

def _safe_ttest_ind(data, group_indicator):
    if np.std(group_indicator) == 0: return 0.0, 1.0
    group0, group1 = data[group_indicator == 0], data[group_indicator == 1]
    if len(group0) < 2 or len(group1) < 2: return 0.0, 1.0
    if np.std(group0) == 0 and np.std(group1) == 0 and np.mean(group0) != np.mean(group1): return np.inf, 0.0
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=RuntimeWarning, message=".*Precision loss.*")
        try:
            stat, p = ttest_ind(group0, group1, equal_var=False, nan_policy='omit')
            return (0.0, 1.0) if np.isnan(stat) else (stat, p)
        except RuntimeWarning:
            return 0.0, 1.0

def calculate_auc_p_value(y_true, y_pred_proba, n_permutations=1000):
    if len(np.unique(y_true)) < 2: return 0.5, 1.0
    observed_auc = roc_auc_score(y_true, y_pred_proba)
    null_aucs = [roc_auc_score(np.random.permutation(y_true), y_pred_proba) for _ in range(n_permutations)]
    p_value = (np.sum(np.array(null_aucs) >= observed_auc) + 1) / (n_permutations + 1)
    return observed_auc, p_value

# --- REVISED CORE MODELING FUNCTION (FIXED & ROBUST) ---

def cv_forward_selection(X, y, snp_indices_to_consider, n_inner_folds=3, shortlist_k=20):
    """
    Performs robust, cross-validation-based forward feature selection.
    This version is robust to perfect separation and adaptive to class imbalance.
    """
    selected_indices = []
    remaining_indices = list(snp_indices_to_consider)
    best_overall_auc = 0.5
    
    if len(y) == 0: return []
    class_counts = np.bincount(y)
    if len(class_counts) < 2 or np.any(class_counts < 2): return []
    min_class_count = min(class_counts)

    while True:
        target_for_test = y
        if selected_indices:
            try:
                temp_model = LogisticRegression(penalty='l2', solver='lbfgs', class_weight='balanced')
                temp_model.fit(X[:, selected_indices], y)
                predictions = temp_model.predict_proba(X[:, selected_indices])[:, 1]
                target_for_test = y - predictions
            except Exception: break

        shortlist_candidates = [(abs(_safe_ttest_ind(data=target_for_test, group_indicator=X[:, idx])[0]), idx) for idx in remaining_indices]
        if not shortlist_candidates: break
        
        shortlist_candidates.sort(key=lambda x: x[0], reverse=True)
        shortlist_indices = [idx for t_stat, idx in shortlist_candidates[:shortlist_k]]
        if not shortlist_indices: break

        n_splits_for_inner_cv = min(n_inner_folds, min_class_count)
        if n_splits_for_inner_cv < 2: break
        
        inner_cv = StratifiedKFold(n_splits=n_splits_for_inner_cv, shuffle=True, random_state=123)
        auc_scores_for_shortlist = {}
        
        for candidate_index in shortlist_indices:
            temp_indices = selected_indices + [candidate_index]
            fold_aucs = []
            for inner_train_idx, inner_test_idx in inner_cv.split(X, y):
                if len(np.unique(y[inner_test_idx])) < 2: continue
                model = LogisticRegression(penalty='l2', solver='lbfgs', class_weight='balanced')
                model.fit(X[inner_train_idx][:, temp_indices], y[inner_train_idx])
                fold_aucs.append(roc_auc_score(y[inner_test_idx], model.predict_proba(X[inner_test_idx][:, temp_indices])[:, 1]))
            
            if fold_aucs: auc_scores_for_shortlist[candidate_index] = np.mean(fold_aucs)
        
        if not auc_scores_for_shortlist: break
        
        best_candidate_index = max(auc_scores_for_shortlist, key=auc_scores_for_shortlist.get)
        if auc_scores_for_shortlist[best_candidate_index] > best_overall_auc:
            selected_indices.append(best_candidate_index)
            remaining_indices.remove(best_candidate_index)
            best_overall_auc = auc_scores_for_shortlist[best_candidate_index]
        else:
            break
            
    return selected_indices

# --- RESTRUCTURED WORKER FUNCTIONS ---

def extract_data_for_locus(inversion_job: dict):
    """
    SEQUENTIAL I/O-HEAVY PHASE: Extracts data from VCF for one locus.
    MODIFIED: Now captures reference and alternate allele characters for each SNP.
    """
    inversion_id = inversion_job.get('orig_ID', 'Unknown_ID')
    try:
        chrom, start, end = inversion_job['seqnames'], inversion_job['start'], inversion_job['end']
        vcf_path = f"../vcfs/{chrom}.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz"
        
        vcf_for_samples = VCF(vcf_path, lazy=True)
        sample_map = {tsv_s: vcf_s for tsv_s in list(inversion_job.keys())[7:] for vcf_s in vcf_for_samples.samples if tsv_s in vcf_s}
        
        if not (str(chrom).endswith('X') or str(chrom).endswith('Y')) and (len(sample_map) / len(list(inversion_job.keys())[7:])) < 0.5:
            raise FatalSampleMappingError(f"Autosomal sample mapping rate for {chrom} was below 50%.")
        if not sample_map: raise ValueError("0% of TSV samples mapped.")

        haplotype_inv_status, valid_vcf_samples_in_order = [], []
        def parse_hap_gt(gt_str: any):
            if not isinstance(gt_str, str) or '|' not in gt_str: return None, None
            parts=gt_str.split('|'); h1,h2 = parts[0].split('_')[0], parts[1].split('_')[0]
            return (int(h1), int(h2)) if h1.isdigit() and h2.isdigit() else (None, None)

        for tsv_s, vcf_s in sample_map.items():
            h1, h2 = parse_hap_gt(inversion_job[tsv_s]);
            if h1 is not None: haplotype_inv_status.extend([h1, h2]); valid_vcf_samples_in_order.append(vcf_s)
        
        regions_to_process = []
        if (start + 50000) >= (end - 50000):
            regions_to_process.append(f"{chrom}:{max(0, start - 50000)}-{end + 50000}")
        else:
            regions_to_process.append(f"{chrom}:{max(0, start - 50000)}-{start + 50000}")
            regions_to_process.append(f"{chrom}:{end - 50000}-{end + 50000}")

        vcf_subset = VCF(vcf_path, samples=valid_vcf_samples_in_order)
        snp_data_list, snp_metadata, processed_positions = [], [], set()

        for region_str in regions_to_process:
            for variant in vcf_subset(region_str):
                if variant.POS in processed_positions: continue
                if variant.is_snp and not variant.is_indel and len(variant.ALT) == 1:
                    snp_data_list.append([allele for gt in variant.genotypes for allele in gt[0:2]])
                    # MODIFICATION: Store actual allele characters
                    snp_metadata.append({'id': variant.ID or f"{variant.CHROM}:{variant.POS}", 'ref_allele': variant.REF, 'alt_allele': variant.ALT[0]})
                    processed_positions.add(variant.POS)
        
        if not snp_metadata: return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'No biallelic SNPs in breakpoint regions'}

        return {
            'status': 'PREPROCESSED', 'id': inversion_id, 'y_haplotypes': np.array(haplotype_inv_status, dtype=int),
            'X_haplotypes': np.array(snp_data_list, dtype=int).T, 'snp_metadata': snp_metadata
        }
    except Exception as e:
        import traceback
        return {'status': 'FAILED', 'id': inversion_id, 'reason': f"Data Extraction Error: {type(e).__name__}: {e}\n{traceback.format_exc()}"}

def analyze_and_model_locus(preloaded_data: dict, outer_cv_folds: int = 5):
    """
    PARALLEL CPU-HEAVY PHASE: Analyzes pre-loaded data using a principled "Audit then Build" workflow.
    MODIFIED: Now extracts the intercept (bias term) and detailed SNP info for the final model.
    """
    inversion_id = preloaded_data['id']
    try:
        y_hap, X_hap, snp_meta = preloaded_data['y_haplotypes'], preloaded_data['X_haplotypes'], preloaded_data['snp_metadata']
        
        if len(y_hap) == 0 or len(np.unique(y_hap)) < 2: return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'Only one inversion class present.'}
        class_counts = np.bincount(y_hap)
        if np.any(class_counts < 3): return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Minority class has < 3 samples ({class_counts}).'}
        
        hq_indices = np.where(np.std(X_hap, axis=0) > 0)[0]
        if len(hq_indices) == 0: return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'No variant SNPs.'}
        n_splits_outer = min(outer_cv_folds, min(class_counts))
        if n_splits_outer < 2: return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Minority class too small for CV.'}
        
        # --- TAG SNP ANALYSIS ---
        skf_tag = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        p_vals_per_fold = [hq_indices[np.nanargmin([_safe_pearsonr(X_hap[train_idx, i], y_hap[train_idx])[0] for i in hq_indices])] for train_idx, _ in skf_tag.split(X_hap, y_hap) if len(train_idx) > 1]
        best_tag_idx = Counter(p_vals_per_fold).most_common(1)[0][0] if p_vals_per_fold else None
        tag_snp_results = {}
        if best_tag_idx:
            _, r = _safe_pearsonr(X_hap[:, best_tag_idx], y_hap)
            meta = snp_meta[best_tag_idx]
            tag_snp_results = {'best_tag_snp': meta['id'], 'tag_snp_r_squared': r**2}

        # --- MULTI-SNP MODEL: PHASE 1 - THE AUDIT (Nested Cross-Validation) ---
        skf_model_outer = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        outer_scores, all_selected_indices = [], []
        for train_idx, test_idx in skf_model_outer.split(X_hap, y_hap):
            selected_in_fold = cv_forward_selection(X_hap[train_idx], y_hap[train_idx], hq_indices)
            all_selected_indices.extend(selected_in_fold)
            if selected_in_fold and len(np.unique(y_hap[test_idx])) > 1:
                try:
                    model = LogisticRegression(penalty=None, solver='newton-cg', class_weight='balanced', max_iter=2000).fit(X_hap[train_idx][:, selected_in_fold], y_hap[train_idx])
                    outer_scores.append(roc_auc_score(y_hap[test_idx], model.predict_proba(X_hap[test_idx][:, selected_in_fold])[:, 1]))
                except Exception: outer_scores.append(0.5)
        
        unbiased_auc = np.mean(outer_scores) if outer_scores else 0.5
        stability_counts = Counter(all_selected_indices)

        # --- MULTI-SNP MODEL: PHASE 2 - THE FINAL BUILD ---
        final_indices = cv_forward_selection(X_hap, y_hap, hq_indices)
        model_results = {}
        if final_indices:
            try:
                final_model = LogisticRegression(penalty=None, solver='newton-cg', class_weight='balanced', max_iter=2000).fit(X_hap[:, final_indices], y_hap)
                _, p_val = calculate_auc_p_value(y_hap, final_model.predict_proba(X_hap[:, final_indices])[:, 1])
                
                # MODIFICATION: Capture all required parameters for a complete model
                snp_details = []
                for idx, coef in sorted(zip(final_indices, final_model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
                    meta = snp_meta[idx]
                    snp_details.append({
                        'id': meta['id'],
                        'ref_allele': meta['ref_allele'],
                        'alt_allele': meta['alt_allele'],
                        'coefficient': coef,
                        'stability': stability_counts.get(idx, 0) / n_splits_outer
                    })
                model_parameters = {
                    'intercept': final_model.intercept_[0],
                    'snps': snp_details
                }
                model_results = {'model_auc': unbiased_auc, 'model_p_value': p_val, 'num_snps': len(snp_details), 'model_parameters': model_parameters}
            except Exception:
                model_results = {'model_auc': unbiased_auc, 'model_p_value': 1.0, 'num_snps': len(final_indices), 'model_parameters': {}}
        else:
            model_results = {'model_auc': unbiased_auc, 'model_p_value': 1.0, 'num_snps': 0, 'model_parameters': {}}
            
        return {'status': 'SUCCESS', 'id': inversion_id, **tag_snp_results, **model_results}

    except Exception as e:
        import traceback
        return {'status': 'FAILED', 'id': inversion_id, 'reason': f"Analysis Error: {type(e).__name__}: {e}\n{traceback.format_exc()}"}

# --- MAIN ORCHESTRATOR ---
if __name__ == '__main__':
    logging.info("--- Starting Combined Tag SNP & Imputation Model Analysis (Batch-Processing Mode) ---")
    start_time = time.time()
    
    ground_truth_file = "../variants_freeze4inv_sv_inv_hg38_processed_arbigent_filtered_manualDotplot_filtered_PAVgenAdded_withInvCategs_syncWithWH.fixedPH.simpleINV.mod.tsv"
    if not os.path.exists(ground_truth_file): logging.critical(f"FATAL: GT file not found: '{ground_truth_file}'"); sys.exit(1)
    
    config_df = pd.read_csv(ground_truth_file, sep='\t', on_bad_lines='warn')
    config_df = config_df[(config_df['verdict'] == 'pass') & (config_df['seqnames'] != 'chrY')].copy()
    
    for chrom in config_df['seqnames'].unique():
        vcf_path = f"../vcfs/{chrom}.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz"
        if os.path.exists(vcf_path) and not os.path.exists(f"{vcf_path}.tbi"):
            try: subprocess.run(['tabix', '-p', 'vcf', vcf_path], check=True, capture_output=True, text=True)
            except: pass

    all_jobs = config_df.to_dict('records')
    num_jobs = len(all_jobs)
    if not all_jobs: logging.warning("No valid jobs to run. Exiting."); sys.exit(0)

    num_procs, batch_size = cpu_count(), cpu_count()
    logging.info(f"Loaded {num_jobs} inversions. Processing in batches of {batch_size} using {num_procs} cores.")
    
    all_results = []
    for i in range((num_jobs + batch_size - 1) // batch_size):
        batch_start_index, batch_end_index = i * batch_size, min((i + 1) * batch_size, num_jobs)
        current_batch_jobs = all_jobs[batch_start_index:batch_end_index]
        logging.info(f"--- Starting Batch {i+1}/{(num_jobs + batch_size - 1) // batch_size} (Inversions {batch_start_index+1}-{batch_end_index}) ---")
        
        logging.info(f"Phase 1: Sequentially extracting data for {len(current_batch_jobs)} inversions...")
        preloaded_data_for_batch = []
        for job_idx, job in enumerate(current_batch_jobs):
            data_result = extract_data_for_locus(job)
            if data_result.get('status') == 'PREPROCESSED':
                preloaded_data_for_batch.append(data_result)
            else:
                all_results.append(data_result)
                progress = (batch_start_index + job_idx + 1) / num_jobs
                sys.stdout.write(f"\rProgress: [{batch_start_index + job_idx + 1}/{num_jobs}] ({progress:.1%}) | ❌ {data_result.get('id', 'N/A'):<25} | {data_result.get('status', 'N/A')}: {data_result.get('reason', 'N/A').splitlines()[0][:50]}...\n")
        
        if preloaded_data_for_batch:
            logging.info(f"Phase 2: Parallel analysis of {len(preloaded_data_for_batch)} pre-loaded inversions...")
            with Parallel(n_jobs=num_procs, backend='loky') as parallel:
                batch_results = parallel(delayed(analyze_and_model_locus)(data) for data in preloaded_data_for_batch)
                for res_idx, result in enumerate(batch_results):
                    all_results.append(result)
                    progress = (batch_start_index + len(current_batch_jobs) - len(preloaded_data_for_batch) + res_idx + 1) / num_jobs
                    if result.get('status') == 'SUCCESS':
                        sys.stdout.write(f"\rProgress: [{len(all_results)}/{num_jobs}] ({progress:.1%}) | ✅ {result['id']:<25} | Unbiased Test AUC={result.get('model_auc', 0):.3f} p={result.get('model_p_value', 1):.3f} ({result.get('num_snps', 0)} SNPs) | Tag R²={result.get('tag_snp_r_squared', 0):.3f}\n")
                    else:
                        sys.stdout.write(f"\rProgress: [{len(all_results)}/{num_jobs}] ({progress:.1%}) | ❌ {result.get('id', 'N/A'):<25} | {result.get('status', 'N/A')}: {result.get('reason', 'N/A').splitlines()[0][:50]}...\n")
        sys.stdout.flush()
        logging.info(f"--- Finished Batch {i+1}/{(num_jobs + batch_size - 1) // batch_size} ---")

    logging.info(f"--- All Batches Complete in {time.time() - start_time:.2f} seconds ---")
    
    successful_runs = [r for r in all_results if r and r.get('status') == 'SUCCESS']
    
    print("\n\n" + "="*100 + "\n---                           FINAL COMBINED ANALYSIS REPORT                           ---\n" + "="*100)
    
    reason_counts = Counter(f"({res.get('status', 'N/A')}) {res.get('reason', 'N/A').splitlines()[0]}" for res in all_results if res.get('status') != 'SUCCESS')
    if reason_counts:
        print("\n--- FAILED OR SKIPPED LOCI SUMMARY ---")
        for reason, count in sorted(reason_counts.items()): print(f"  - ({count: >3} loci): {reason}")

    if successful_runs:
        results_df = pd.DataFrame(successful_runs).set_index('id')
        pd.set_option('display.width', 200); pd.set_option('display.max_rows', None); pd.options.display.float_format = '{:.4g}'.format

        print("\n\n" + "="*50 + " MULTI-SNP IMPUTATION MODEL ANALYSIS " + "="*50)
        model_df = results_df[results_df['model_auc'].notna()]
        if not model_df.empty:
            print(f"\n--- Aggregate Performance (Multi-SNP Model) ---")
            print(f"  Mean Unbiased Test AUC: {model_df['model_auc'].mean():.4f} (estimated from nested cross-validation)")
            print(f"  Models with Est. AUC > 0.70: {(model_df['model_auc'] > 0.70).sum()} / {len(model_df)}")
            
            high_perf_df = model_df[(model_df['model_auc'] > 0.70) & (model_df['model_p_value'] < 0.01)].sort_values('model_auc', ascending=False)
            
            if not high_perf_df.empty:
                # MODIFICATION: Create separate summary TSV and detailed TOML files
                
                # 1. Create and save the summary TSV
                summary_cols = ['model_auc', 'model_p_value', 'num_snps', 'tag_snp_r_squared']
                summary_df = high_perf_df[summary_cols]
                summary_filename = "high_performance_models_summary.tsv"
                summary_df.to_csv(summary_filename, sep='\t', float_format='%.4g')
                print(f"\n--- High-Performance Models Summary (Est. Test AUC > 0.70 and p < 0.01) ---")
                print(summary_df)
                print(f"\n[SUCCESS] Saved summary for {len(high_perf_df)} models to '{summary_filename}'")

                # 2. Create and save the detailed TOML model file
                models_for_toml = {}
                for inv_id, row in high_perf_df.iterrows():
                    params = row.get('model_parameters')
                    if params and 'snps' in params:
                        model_entry = {
                            'model_auc': row['model_auc'],
                            'model_p_value': row['model_p_value'],
                            'num_snps': row['num_snps'],
                            'intercept': params.get('intercept', 'N/A'),
                            'snps': params['snps']
                        }
                        models_for_toml[inv_id] = model_entry
                
                toml_filename = "high_performance_imputation_models.toml"
                try:
                    with open(toml_filename, 'w') as f:
                        toml.dump(models_for_toml, f)
                    print(f"[SUCCESS] Saved full model specifications for {len(models_for_toml)} models to '{toml_filename}'")
                except Exception as e:
                    print(f"\n[ERROR] Could not save TOML model file: {e}")

            else: 
                print("\n--- High-Performance Imputation Models (Est. Test AUC > 0.70 and Final Model p < 0.01) ---")
                print("  None found meeting criteria.")
        else: 
            print("  No valid models were built.")
    print("\n" + "="*100)
