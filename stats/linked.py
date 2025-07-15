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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# --- SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
warnings.filterwarnings("ignore")

class FatalSampleMappingError(Exception):
    pass

# --- CORE HELPER FUNCTIONS (UNCHANGED) ---

def _safe_pearsonr(x, y):
    """A robust pearsonr function that handles constant inputs, preventing crashes."""
    if np.std(x) == 0 or np.std(y) == 0:
        return 1.0, 0.0 # p-value, r-value
    r, p = pearsonr(x, y)
    return p, r

def calculate_auc_p_value(y_true, y_pred_proba, n_permutations=1000):
    """Calculates the p-value of an AUC score via permutation testing."""
    if len(np.unique(y_true)) < 2: return 0.5, 1.0
    observed_auc = roc_auc_score(y_true, y_pred_proba)
    null_aucs = [roc_auc_score(np.random.permutation(y_true), y_pred_proba) for _ in range(n_permutations)]
    p_value = (np.sum(np.array(null_aucs) >= observed_auc) + 1) / (n_permutations + 1)
    return observed_auc, p_value

def hybrid_forward_selection(X_train, y_train, snp_indices_to_consider, inner_cv_folds=5, top_n_candidates=30):
    """A faster, hybrid forward selection using a robust correlation filter."""
    selected_indices = []
    best_overall_cv_score = 0.5
    model = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=2000)

    min_class_count = min(np.bincount(y_train))
    n_splits = min(inner_cv_folds, min_class_count)
    if n_splits < 2: return []
    
    inner_cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

    while True:
        remaining_indices = [idx for idx in snp_indices_to_consider if idx not in selected_indices]
        if not remaining_indices: break
        
        if not selected_indices:
            residuals = y_train
        else:
            temp_model = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=2000).fit(X_train[:, selected_indices], y_train)
            residuals = y_train - temp_model.predict_proba(X_train[:, selected_indices])[:, 1]
        
        p_values = [_safe_pearsonr(X_train[:, idx], residuals)[0] for idx in remaining_indices]
        promising_indices = [idx for _, idx in sorted(zip(p_values, remaining_indices))[:top_n_candidates]]
        
        scores = []
        for candidate_index in promising_indices:
            temp_indices = selected_indices + [candidate_index]
            try:
                cv_scores = cross_val_score(model, X_train[:, temp_indices], y_train, cv=inner_cv_splitter, scoring='roc_auc', n_jobs=1)
                scores.append(np.nanmean(cv_scores))
            except ValueError: scores.append(0.5)

        scores = [s if not np.isnan(s) else 0.5 for s in scores]
        if not scores: break

        best_score_this_step = max(scores)
        if best_score_this_step > best_overall_cv_score:
            best_candidate_index = promising_indices[np.argmax(scores)]
            selected_indices.append(best_candidate_index)
            best_overall_cv_score = best_score_this_step
        else: break
            
    return selected_indices

# --- RESTRUCTURED WORKER FUNCTIONS ---

def extract_data_for_locus(inversion_job: dict):
    """
    SEQUENTIAL I/O-HEAVY PHASE: Extracts data from VCF for one locus.
    This function is run sequentially to prevent VCF library deadlocks.
    """
    inversion_id = inversion_job.get('orig_ID', 'Unknown_ID')
    try:
        chrom = inversion_job['seqnames']
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
        
        region_str = f"{chrom}:{max(0, inversion_job['start'] - 50000)}-{inversion_job['end'] + 50000}"
        vcf_subset = VCF(vcf_path, samples=valid_vcf_samples_in_order)
        
        snp_data_list, snp_metadata = [], []
        for variant in vcf_subset(region_str):
             if variant.is_snp and not variant.is_indel and len(variant.ALT) == 1:
                snp_data_list.append([allele for gt in variant.genotypes for allele in gt[0:2]])
                snp_metadata.append({'id': variant.ID or f"{variant.CHROM}:{variant.POS}", 'ref': variant.REF, 'alt': variant.ALT[0]})
        
        if not snp_metadata: 
             return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'No biallelic SNPs in region'}

        X_haplotypes = np.array(snp_data_list, dtype=int).T
        return {
            'status': 'PREPROCESSED', 'id': inversion_id, 'y_haplotypes': y_haplotypes,
            'X_haplotypes': X_haplotypes, 'snp_metadata': snp_metadata
        }
    except Exception as e:
        import traceback
        return {'status': 'FAILED', 'id': inversion_id, 'reason': f"Data Extraction Error: {type(e).__name__}: {e}\n{traceback.format_exc()}"}

def analyze_and_model_locus(preloaded_data: dict, outer_cv_folds: int = 5, stability_threshold: float = 0.6):
    """
    PARALLEL CPU-HEAVY PHASE: Analyzes pre-loaded in-memory data.
    This function is safe to run in parallel as it performs no file I/O.
    """
    inversion_id = preloaded_data['id']
    try:
        y_haplotypes = preloaded_data['y_haplotypes']
        X_haplotypes = preloaded_data['X_haplotypes']
        snp_metadata = preloaded_data['snp_metadata']
        
        # --- Robust Pre-filtering on the loaded data ---
        if len(y_haplotypes) == 0: return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'No samples with valid genotypes.'}
        if len(np.unique(y_haplotypes)) < 2: return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'Only one inversion class present.'}
        class_counts = np.bincount(y_haplotypes)
        if np.any(class_counts < 3): return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Minority class has < 3 samples ({class_counts}).'}

        high_quality_indices = np.where(np.std(X_haplotypes, axis=0) > 0)[0]
        if len(high_quality_indices) == 0: return {'status': 'SKIPPED', 'id': inversion_id, 'reason': 'No variant SNPs.'}

        # --- Analysis with StratifiedKFold ---
        n_splits_outer = min(outer_cv_folds, min(class_counts))
        if n_splits_outer < 2: return {'status': 'SKIPPED', 'id': inversion_id, 'reason': f'Minority class too small ({min(class_counts)}) for CV.'}
        skf = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)

        # 1. Tag SNP Analysis
        tag_snp_results = {}
        p_vals_per_fold = []
        for train_idx, _ in skf.split(X_haplotypes, y_haplotypes):
            if len(train_idx) > 1:
                p_vals = [_safe_pearsonr(X_haplotypes[train_idx, i], y_haplotypes[train_idx])[0] for i in high_quality_indices]
                p_vals_per_fold.append(high_quality_indices[np.nanargmin(p_vals)])
        
        if p_vals_per_fold:
            best_tag_idx = Counter(p_vals_per_fold).most_common(1)[0][0]
            _, r = _safe_pearsonr(X_haplotypes[:, best_tag_idx], y_haplotypes)
            meta = snp_metadata[best_tag_idx]
            tag_snp_results = {'best_tag_snp': meta['id'], 'tag_snp_r_squared': r**2, 'tag_snp_allele_direct': (f"{meta['alt']}(A)" if r<0 else f"{meta['ref']}(R)"), 'tag_snp_allele_inverted': (f"{meta['ref']}(R)" if r<0 else f"{meta['alt']}(A)")}

        # 2. Multi-SNP Model Analysis (Nested CV)
        outer_loop_scores, all_selected_indices = [], []
        for train_idx, test_idx in skf.split(X_haplotypes, y_haplotypes):
            selected = hybrid_forward_selection(X_haplotypes[train_idx], y_haplotypes[train_idx], high_quality_indices)
            all_selected_indices.extend(selected)
            if selected and len(np.unique(y_haplotypes[test_idx])) > 1:
                fold_model = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=2000).fit(X_haplotypes[train_idx][:, selected], y_haplotypes[train_idx])
                outer_loop_scores.append(roc_auc_score(y_haplotypes[test_idx], fold_model.predict_proba(X_haplotypes[test_idx][:, selected])[:, 1]))

        unbiased_auc = np.mean(outer_loop_scores) if outer_loop_scores else 0.5
        snp_stability_counts = Counter(all_selected_indices)
        final_indices = [idx for idx, count in snp_stability_counts.items() if (count / n_splits_outer) >= stability_threshold]
        
        model_results = {}
        if final_indices:
            model = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=2000).fit(X_haplotypes[:, final_indices], y_haplotypes)
            _, p_val = calculate_auc_p_value(y_haplotypes, model.predict_proba(X_haplotypes[:, final_indices])[:, 1])
            details = [{'snp_id': snp_metadata[idx]['id'], 'coefficient': coef, 'stability': snp_stability_counts.get(idx, 0) / n_splits_outer} for idx, coef in sorted(zip(final_indices, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)]
            model_results = {'model_auc': unbiased_auc, 'model_p_value': p_val, 'num_snps_in_model': len(details), 'model_details': details}
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
    batch_size = num_procs # Process a batch of jobs equal to the number of cores.
    logging.info(f"Loaded {num_jobs} inversions. Processing in batches of {batch_size} using {num_procs} cores.")
    
    all_results = []
    num_batches = (num_jobs + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        batch_start_index = i * batch_size
        batch_end_index = min((i + 1) * batch_size, num_jobs)
        current_batch_jobs = all_jobs[batch_start_index:batch_end_index]
        
        logging.info(f"--- Starting Batch {i+1}/{num_batches} (Inversions {batch_start_index+1}-{batch_end_index}) ---")
        
        # --- PHASE 1: SEQUENTIAL Data Extraction for the current batch ---
        logging.info(f"Phase 1: Sequentially extracting data for {len(current_batch_jobs)} inversions...")
        preloaded_data_for_batch = []
        for job in current_batch_jobs:
            data_result = extract_data_for_locus(job)
            if data_result.get('status') == 'PREPROCESSED':
                preloaded_data_for_batch.append(data_result)
            else:
                # If extraction fails, it's a final result. Add it and print.
                all_results.append(data_result)
                reason = data_result.get('reason', 'Unknown').splitlines()[0]
                id_str = data_result.get('id', 'Unknown')
                progress = (len(all_results)) / num_jobs
                sys.stdout.write(f"\rProgress: [{len(all_results)}/{num_jobs}] ({progress:.1%}) | ❌ {id_str:<25} | {data_result.get('status', 'N/A')}: {reason[:50]}...\n")
                sys.stdout.flush()

        # --- PHASE 2: PARALLEL Analysis on the in-memory data for the batch ---
        if preloaded_data_for_batch:
            logging.info(f"Phase 2: Parallel analysis of {len(preloaded_data_for_batch)} pre-loaded inversions...")
            with Parallel(n_jobs=num_procs, backend='loky') as parallel:
                batch_results = parallel(delayed(analyze_and_model_locus)(data) for data in preloaded_data_for_batch)
                
                # --- Immediate printing of batch results ---
                for result in batch_results:
                    all_results.append(result)
                    progress = (len(all_results)) / num_jobs
                    id_str = result.get('id', 'Unknown')
                    
                    if result.get('status') == 'SUCCESS':
                        auc = result.get('model_auc', 0)
                        pval = result.get('model_p_value', 1)
                        r2 = result.get('tag_snp_r_squared', 0)
                        n_snps = result.get('num_snps_in_model', 0)
                        sys.stdout.write(f"\rProgress: [{len(all_results)}/{num_jobs}] ({progress:.1%}) | ✅ {id_str:<25} | Test AUC={auc:.3f} p={pval:.3f} ({n_snps} SNPs) | Tag R²={r2:.3f}\n")
                    else:
                        reason = result.get('reason', 'Unknown').splitlines()[0]
                        sys.stdout.write(f"\rProgress: [{len(all_results)}/{num_jobs}] ({progress:.1%}) | ❌ {id_str:<25} | {result.get('status', 'N/A')}: {reason[:50]}...\n")
                    sys.stdout.flush()
        logging.info(f"--- Finished Batch {i+1}/{num_batches} ---")

    logging.info(f"--- All Batches Complete in {time.time() - start_time:.2f} seconds ---")
    
    # --- FINAL REPORTING (uses all_results list) ---
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
            print(f"  Mean Unbiased Test AUC: {model_df['model_auc'].mean():.4f}")
            print(f"  Models with AUC > 0.95: {(model_df['model_auc'] > 0.95).sum()} / {len(model_df)}")
            
            print("\n--- High-Performance Imputation Models (Test AUC > 0.95 and p < 0.01) ---")
            high_perf_df = model_df[(model_df['model_auc'] > 0.95) & (model_df['model_p_value'] < 0.01)].sort_values('model_auc', ascending=False)
            if not high_perf_df.empty:
                all_snp_details = []
                for inv_id, row in high_perf_df.iterrows():
                    print(f"\n--- Inversion: {inv_id} | Test AUC: {row['model_auc']:.4f} | p-value: {row['model_p_value']:.4g} | #SNPs: {row['num_snps_in_model']} ---")
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
