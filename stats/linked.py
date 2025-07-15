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

# --- SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# Suppress expected warnings to keep output clean. The code now handles these cases.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class FatalSampleMappingError(Exception):
    pass

# --- CORE HELPER FUNCTIONS ---

def _safe_pearsonr(x, y):
    """A robust pearsonr function that handles constant inputs, preventing crashes."""
    if np.std(x) == 0 or np.std(y) == 0:
        return 1.0, 0.0 # p-value, r-value
    r, p = pearsonr(x, y)
    return p, r

def _safe_ttest_ind(data, group_indicator):
    """
    Performs a robust two-sample t-test, handling common data issues including
    catastrophic cancellation warnings by treating them as uninformative tests.
    """
    if np.std(group_indicator) == 0:
        return 0.0, 1.0

    group0 = data[group_indicator == 0]
    group1 = data[group_indicator == 1]

    if len(group0) < 2 or len(group1) < 2:
        return 0.0, 1.0

    if np.std(group0) == 0 and np.std(group1) == 0 and np.mean(group0) != np.mean(group1):
        return np.inf, 0.0

    # FIX: Catch the specific "Precision loss" warning and treat it as a non-significant result.
    with warnings.catch_warnings():
        # Temporarily treat this specific warning as an error so we can catch it.
        warnings.filterwarnings('error', category=RuntimeWarning, message=".*Precision loss.*")
        try:
            stat, p = ttest_ind(group0, group1, equal_var=False, nan_policy='omit')
            if np.isnan(stat):
                return 0.0, 1.0
            return stat, p
        except RuntimeWarning:
            # If the warning was caught, it means the data groups were nearly identical.
            # This SNP is uninformative, so we return a t-statistic of 0.
            return 0.0, 1.0

def calculate_auc_p_value(y_true, y_pred_proba, n_permutations=1000):
    """Calculates the p-value of an AUC score via permutation testing."""
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

    The process is iterative:
    1. T-test Pre-Filter: At each step, it uses a t-test to find SNPs that are
       most associated with the model's unexplained signal (residuals). This is
       fast and robustly handles perfect predictors.
    2. CV-AUC Selection: It then performs an inner cross-validation loop on the
       shortlisted SNPs. The number of folds is adapted to the class size to
       prevent errors. The SNP providing the highest average AUC is added.
    3. Termination: The process stops when no candidate SNP can improve the AUC.
    """
    selected_indices = []
    remaining_indices = list(snp_indices_to_consider)
    best_overall_auc = 0.5 # Baseline AUC for a random classifier
    
    if len(y) == 0: return []
    class_counts = np.bincount(y)
    if len(class_counts) < 2 or np.any(class_counts < 2):
        return []
    min_class_count = min(class_counts)

    while True:
        # --- Stage 1: Robust t-test pre-filter to create a shortlist ---
        target_for_test = y
        if selected_indices:
            try:
                temp_model = LogisticRegression(penalty='l2', solver='lbfgs', class_weight='balanced')
                temp_model.fit(X[:, selected_indices], y)
                predictions = temp_model.predict_proba(X[:, selected_indices])[:, 1]
                target_for_test = y - predictions
            except Exception:
                break

        shortlist_candidates = []
        for candidate_index in remaining_indices:
            t_stat, p_val = _safe_ttest_ind(data=target_for_test, group_indicator=X[:, candidate_index])
            shortlist_candidates.append((abs(t_stat), candidate_index))
        
        if not shortlist_candidates: break

        shortlist_candidates.sort(key=lambda x: x[0], reverse=True)
        shortlist_indices = [idx for t_stat, idx in shortlist_candidates[:shortlist_k]]
        
        if not shortlist_indices: break

        # --- Stage 2: Robust CV-AUC selection on the shortlist ---
        n_splits_for_inner_cv = min(n_inner_folds, min_class_count)
        if n_splits_for_inner_cv < 2:
            break

        inner_cv = StratifiedKFold(n_splits=n_splits_for_inner_cv, shuffle=True, random_state=123)
        auc_scores_for_shortlist = {}
        
        for candidate_index in shortlist_indices:
            temp_indices = selected_indices + [candidate_index]
            fold_aucs = []
            
            for inner_train_idx, inner_test_idx in inner_cv.split(X, y):
                X_inner_train = X[inner_train_idx][:, temp_indices]
                X_inner_test = X[inner_test_idx][:, temp_indices]
                y_inner_train, y_inner_test = y[inner_train_idx], y[inner_test_idx]

                if len(np.unique(y_inner_test)) < 2: continue
                
                model = LogisticRegression(penalty='l2', solver='lbfgs', class_weight='balanced')
                model.fit(X_inner_train, y_inner_train)
                fold_aucs.append(roc_auc_score(y_inner_test, model.predict_proba(X_inner_test)[:, 1]))
            
            if fold_aucs: auc_scores_for_shortlist[candidate_index] = np.mean(fold_aucs)
        
        if not auc_scores_for_shortlist: break

        best_candidate_index = max(auc_scores_for_shortlist, key=auc_scores_for_shortlist.get)
        best_auc_this_step = auc_scores_for_shortlist[best_candidate_index]
        
        if best_auc_this_step > best_overall_auc:
            selected_indices.append(best_candidate_index)
            remaining_indices.remove(best_candidate_index)
            best_overall_auc = best_auc_this_step
        else:
            break
            
    return selected_indices

# --- RESTRUCTURED WORKER FUNCTIONS ---

def extract_data_for_locus(inversion_job: dict):
    """
    SEQUENTIAL I/O-HEAVY PHASE: Extracts data from VCF for one locus.
    """
    inversion_id = inversion_job.get('orig_ID', 'Unknown_ID')
    try:
        chrom = inversion_job['seqnames']
        start = inversion_job['start']
        end = inversion_job['end']
        vcf_path = f"../vcfs/{chrom}.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz"
        
        vcf_for_samples = VCF(vcf_path, lazy=True)
        vcf_samples = vcf_for_samples.samples
        tsv_samples = list(inversion_job.keys())[7:]
        sample_map = {tsv_s: vcf_s for tsv_s in tsv_samples for vcf_s in vcf_samples if tsv_s in vcf_s}
        
        if not (str(chrom).endswith('X') or str(chrom).endswith('Y')) and (len(sample_map) / len(tsv_samples)) < 0.5:
            raise FatalSampleMappingError(f"Autosomal sample mapping rate for {chrom} was below 50%.")
        if not sample_map: raise ValueError("0% of TSV samples mapped.")

        haplotype_inv_status, valid_vcf_samples_in_order = [], []
        def parse_hap_gt(gt_str: any):
            if not isinstance(gt_str, str) or '|' not in gt_str: return None, None
            parts=gt_str.split('|'); h1_str,h2_str = parts[0].split('_')[0], parts[1].split('_')[0]
            if not h1_str.isdigit() or not h2_str.isdigit(): return None, None
            return int(h1_str), int(h2_str)

        for tsv_s, vcf_s in sample_map.items():
            h1, h2 = parse_hap_gt(inversion_job[tsv_s]);
            if h1 is not None: haplotype_inv_status.extend([h1, h2]); valid_vcf_samples_in_order.append(vcf_s)
        
        y_haplotypes = np.array(haplotype_inv_status, dtype=int)
        
        regions_to_process = []
        left_bp_start = max(0, start - 50000)
        left_bp_end = start + 50000
        right_bp_start = end - 50000
        right_bp_end = end + 50000

        if left_bp_end >= right_bp_start:
            regions_to_process.append(f"{chrom}:{left_bp_start}-{right_bp_end}")
        else:
            regions_to_process.append(f"{chrom}:{left_bp_start}-{left_bp_end}")
            regions_to_process.append(f"{chrom}:{right_bp_start}-{right_bp_end}")

        vcf_subset = VCF(vcf_path, samples=valid_vcf_samples_in_order)
        snp_data_list, snp_metadata = [], []
        processed_positions = set()

        for region_str in regions_to_process:
            for variant in vcf_subset(region_str):
                if variant.POS in processed_positions: continue
                if variant.is_snp and not variant.is_indel and len(variant.ALT) == 1:
                    snp_data_list.append([allele for gt in variant.genotypes for allele in gt[0:2]])
                    snp_metadata.append({'id': variant.ID or f"{variant.CHROM}:{variant.POS}", 'ref': variant.REF, 'alt': variant.ALT[0]})
                    processed_positions.add(variant.POS)
        
        if not snp_metadata: 
             return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'No biallelic SNPs in breakpoint regions'}

        X_haplotypes = np.array(snp_data_list, dtype=int).T
        return {
            'status': 'PREPROCESSED', 'id': inversion_id, 'y_haplotypes': y_haplotypes,
            'X_haplotypes': X_haplotypes, 'snp_metadata': snp_metadata
        }
    except Exception as e:
        import traceback
        return {'status': 'FAILED', 'id': inversion_id, 'reason': f"Data Extraction Error: {type(e).__name__}: {e}\n{traceback.format_exc()}"}

def analyze_and_model_locus(preloaded_data: dict, outer_cv_folds: int = 5):
    """
    PARALLEL CPU-HEAVY PHASE: Analyzes pre-loaded data using a principled "Audit then Build" workflow.
    """
    inversion_id = preloaded_data['id']
    try:
        y_haplotypes = preloaded_data['y_haplotypes']
        X_haplotypes = preloaded_data['X_haplotypes']
        snp_metadata = preloaded_data['snp_metadata']
        
        if len(y_haplotypes) == 0: return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'No samples with valid genotypes.'}
        if len(np.unique(y_haplotypes)) < 2: return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'Only one inversion class present.'}
        class_counts = np.bincount(y_haplotypes)
        if np.any(class_counts < 3): return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Minority class has < 3 samples ({class_counts}).'}

        high_quality_indices = np.where(np.std(X_haplotypes, axis=0) > 0)[0]
        if len(high_quality_indices) == 0: return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'No variant SNPs.'}

        n_splits_outer = min(outer_cv_folds, min(class_counts))
        if n_splits_outer < 2: return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Minority class too small ({min(class_counts)}) for CV.'}
        
        # --- TAG SNP ANALYSIS ---
        tag_snp_results = {}
        skf_tag = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        p_vals_per_fold = []
        for train_idx, _ in skf_tag.split(X_haplotypes, y_haplotypes):
            if len(train_idx) > 1:
                p_vals = [_safe_pearsonr(X_haplotypes[train_idx, i], y_haplotypes[train_idx])[0] for i in high_quality_indices]
                p_vals_per_fold.append(high_quality_indices[np.nanargmin(p_vals)])
        
        if p_vals_per_fold:
            best_tag_idx = Counter(p_vals_per_fold).most_common(1)[0][0]
            _, r = _safe_pearsonr(X_haplotypes[:, best_tag_idx], y_haplotypes)
            meta = snp_metadata[best_tag_idx]
            tag_snp_results = {'best_tag_snp': meta['id'], 'tag_snp_r_squared': r**2, 'tag_snp_allele_direct': (f"{meta['alt']}(A)" if r<0 else f"{meta['ref']}(R)"), 'tag_snp_allele_inverted': (f"{meta['ref']}(R)" if r<0 else f"{meta['alt']}(A)")}

        # --- MULTI-SNP MODEL: PHASE 1 - THE AUDIT (Nested Cross-Validation) ---
        skf_model_outer = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        outer_loop_scores, all_selected_indices_from_cv = [], []
        for train_idx, test_idx in skf_model_outer.split(X_haplotypes, y_haplotypes):
            selected_in_fold = cv_forward_selection(X_haplotypes[train_idx], y_haplotypes[train_idx], high_quality_indices)
            all_selected_indices_from_cv.extend(selected_in_fold)
            
            if selected_in_fold and len(np.unique(y_haplotypes[test_idx])) > 1:
                try:
                    fold_model = LogisticRegression(penalty=None, solver='newton-cg', class_weight='balanced', max_iter=2000).fit(X_haplotypes[train_idx][:, selected_in_fold], y_haplotypes[train_idx])
                    outer_loop_scores.append(roc_auc_score(y_haplotypes[test_idx], fold_model.predict_proba(X_haplotypes[test_idx][:, selected_in_fold])[:, 1]))
                except Exception:
                    outer_loop_scores.append(0.5)

        unbiased_auc = np.mean(outer_loop_scores) if outer_loop_scores else 0.5
        snp_stability_counts = Counter(all_selected_indices_from_cv)

        # --- MULTI-SNP MODEL: PHASE 2 - THE FINAL BUILD ---
        final_indices = cv_forward_selection(X_haplotypes, y_haplotypes, high_quality_indices)
        
        model_results = {}
        if final_indices:
            try:
                final_model = LogisticRegression(penalty=None, solver='newton-cg', class_weight='balanced', max_iter=2000).fit(X_haplotypes[:, final_indices], y_haplotypes)
                _, p_val = calculate_auc_p_value(y_haplotypes, final_model.predict_proba(X_haplotypes[:, final_indices])[:, 1])
                
                details = []
                for idx, coef in sorted(zip(final_indices, final_model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
                    stability = snp_stability_counts.get(idx, 0) / n_splits_outer
                    details.append({'snp_id': snp_metadata[idx]['id'], 'coefficient': coef, 'stability': stability})

                model_results = {'model_auc': unbiased_auc, 'model_p_value': p_val, 'num_snps_in_model': len(details), 'model_details': details}
            except Exception:
                model_results = {'model_auc': unbiased_auc, 'model_p_value': 1.0, 'num_snps_in_model': len(final_indices), 'model_details': [], 'reason': 'Final unpenalized model failed to converge'}
        else:
            model_results = {'model_auc': unbiased_auc, 'model_p_value': 1.0, 'num_snps_in_model': 0, 'model_details': []}

        return {'status': 'SUCCESS', 'id': inversion_id, 'n_haplotypes': len(y_haplotypes), **tag_snp_results, **model_results}

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
    
    unique_chroms = config_df['seqnames'].unique()
    for chrom in unique_chroms:
        vcf_path = f"../vcfs/{chrom}.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz"
        if os.path.exists(vcf_path) and not os.path.exists(vcf_path + ".tbi"):
            try: subprocess.run(['tabix', '-p', 'vcf', vcf_path], check=True, capture_output=True, text=True)
            except: pass

    all_jobs = config_df.to_dict('records')
    num_jobs = len(all_jobs)
    if not all_jobs: logging.warning("No valid jobs to run. Exiting."); sys.exit(0)

    num_procs = cpu_count()
    batch_size = num_procs
    logging.info(f"Loaded {num_jobs} inversions. Processing in batches of {batch_size} using {num_procs} cores.")
    
    all_results = []
    num_batches = (num_jobs + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        batch_start_index = i * batch_size
        batch_end_index = min((i + 1) * batch_size, num_jobs)
        current_batch_jobs = all_jobs[batch_start_index:batch_end_index]
        
        logging.info(f"--- Starting Batch {i+1}/{num_batches} (Inversions {batch_start_index+1}-{batch_end_index}) ---")
        
        logging.info(f"Phase 1: Sequentially extracting data for {len(current_batch_jobs)} inversions...")
        preloaded_data_for_batch = []
        for job in current_batch_jobs:
            data_result = extract_data_for_locus(job)
            if data_result.get('status') == 'PREPROCESSED':
                preloaded_data_for_batch.append(data_result)
            else:
                all_results.append(data_result)
                reason = data_result.get('reason', 'Unknown').splitlines()[0]
                id_str = data_result.get('id', 'Unknown')
                progress = (len(all_results)) / num_jobs
                sys.stdout.write(f"\rProgress: [{len(all_results)}/{num_jobs}] ({progress:.1%}) | ❌ {id_str:<25} | {data_result.get('status', 'N/A')}: {reason[:50]}...\n")
                sys.stdout.flush()

        if preloaded_data_for_batch:
            logging.info(f"Phase 2: Parallel analysis of {len(preloaded_data_for_batch)} pre-loaded inversions...")
            with Parallel(n_jobs=num_procs, backend='loky') as parallel:
                batch_results = parallel(delayed(analyze_and_model_locus)(data) for data in preloaded_data_for_batch)
                
                for result in batch_results:
                    all_results.append(result)
                    progress = (len(all_results)) / num_jobs
                    id_str = result.get('id', 'Unknown')
                    
                    if result.get('status') == 'SUCCESS':
                        auc = result.get('model_auc', 0)
                        pval = result.get('model_p_value', 1)
                        r2 = result.get('tag_snp_r_squared', 0)
                        n_snps = result.get('num_snps_in_model', 0)
                        sys.stdout.write(f"\rProgress: [{len(all_results)}/{num_jobs}] ({progress:.1%}) | ✅ {id_str:<25} | Unbiased Test AUC={auc:.3f} p={pval:.3f} ({n_snps} SNPs) | Tag R²={r2:.3f}\n")
                    else:
                        reason = result.get('reason', 'Unknown').splitlines()[0]
                        sys.stdout.write(f"\rProgress: [{len(all_results)}/{num_jobs}] ({progress:.1%}) | ❌ {id_str:<25} | {result.get('status', 'N/A')}: {reason[:50]}...\n")
                    sys.stdout.flush()
        logging.info(f"--- Finished Batch {i+1}/{num_batches} ---")

    logging.info(f"--- All Batches Complete in {time.time() - start_time:.2f} seconds ---")
    
    successful_runs = [r for r in all_results if r and r.get('status') == 'SUCCESS']
    failed_runs = [r for r in all_results if r and r.get('status') != 'SUCCESS']

    print("\n\n" + "="*100)
    print("---                           FINAL COMBINED ANALYSIS REPORT                           ---")
    print("="*100)

    if failed_runs:
        print("\n--- FAILED OR SKIPPED LOCI SUMMARY ---")
        reason_counts = Counter(f"({res.get('status', 'N/A')}) {res.get('reason', 'Unknown').splitlines()[0]}" for res in failed_runs)
        for reason, count in sorted(reason_counts.items()): print(f"  - ({count: >3} loci): {reason}")
    
    if successful_runs:
        results_df = pd.DataFrame(successful_runs).set_index('id')
        pd.set_option('display.width', 200); pd.set_option('display.max_rows', None); pd.options.display.float_format = '{:.4g}'.format

        print("\n\n" + "="*50 + " SINGLE BEST TAG SNP ANALYSIS " + "="*50)
        tag_snp_df = results_df.dropna(subset=['best_tag_snp'])
        if not tag_snp_df.empty:
            print("\n--- High-Confidence Tag SNPs (R² > 0.7) ---")
            high_conf_tag_df = tag_snp_df[tag_snp_df['tag_snp_r_squared'] > 0.7].sort_values('tag_snp_r_squared', ascending=False)
            if not high_conf_tag_df.empty:
                display_cols = ['best_tag_snp', 'tag_snp_r_squared', 'n_haplotypes', 'tag_snp_allele_direct', 'tag_snp_allele_inverted']
                print(high_conf_tag_df[display_cols])
            else: print("  None found meeting criteria.")
        else: print("  No valid tag SNPs found.")

        print("\n\n" + "="*50 + " MULTI-SNP IMPUTATION MODEL ANALYSIS " + "="*50)
        model_df = results_df[results_df['model_auc'].notna()]
        if not model_df.empty:
            print(f"\n--- Aggregate Performance (Multi-SNP Model) ---")
            print(f"  Mean Unbiased Test AUC: {model_df['model_auc'].mean():.4f} (estimated from nested cross-validation)")
            print(f"  Models with Est. AUC > 0.75: {(model_df['model_auc'] > 0.75).sum()} / {len(model_df)}")
            
            print("\n--- High-Performance Imputation Models (Est. Test AUC > 0.75 and Final Model p < 0.01) ---")
            high_perf_df = model_df[(model_df['model_auc'] > 0.75) & (model_df['model_p_value'] < 0.01)].sort_values('model_auc', ascending=False)
            if not high_perf_df.empty:
                all_snp_details = []
                for inv_id, row in high_perf_df.iterrows():
                    print(f"\n--- Inversion: {inv_id} | Est. Test AUC: {row['model_auc']:.4f} | Final Model p-value: {row['model_p_value']:.4g} | #SNPs: {row['num_snps_in_model']} ---")
                    if 'model_details' in row and row['model_details']:
                        details_df = pd.DataFrame(row['model_details'])
                        print(details_df.to_string(index=False))
                        for snp_detail in row['model_details']: snp_detail['inversion_id'] = inv_id; snp_detail['model_auc'] = row['model_auc']; snp_detail['model_p_value'] = row['model_p_value']; all_snp_details.append(snp_detail)
                
                if all_snp_details:
                    try: 
                        out_df = pd.DataFrame(all_snp_details)[['inversion_id', 'model_auc', 'model_p_value', 'snp_id', 'coefficient', 'stability']]
                        out_df.to_csv("high_performance_imputation_models.tsv", sep='\t', index=False, float_format='%.4g')
                        print(f"\n[SUCCESS] Saved details for {len(high_perf_df)} models to 'high_performance_imputation_models.tsv'")
                    except Exception as e: print(f"\n[WARNING] Could not save model results: {e}")
            else: print("  None found meeting criteria.")
        else: print("  No valid models were built.")
    print("\n" + "="*100)
