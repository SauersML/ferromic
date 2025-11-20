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
from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedGroupKFold, KFold
from sklearn.metrics import mean_squared_error
from pls_patch import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import wilcoxon, pearsonr
import traceback

# TARGET="$HOME/.pytargets/sklearn171" && python3 -m pip install --upgrade --no-cache-dir --ignore-installed   --target "$TARGET" "scikit-learn>=1.6,<2" && PYTHONPATH="$TARGET:$PYTHONPATH" python3 -c 'import sklearn,sys; print(sklearn.__version__, "->", sklearn.__file__)' && PYTHONPATH="$TARGET:$PYTHONPATH" python3 /home/hsiehph/sauer354/di/ferromic/linked.py

# PYTHONPATH="$TARGET:$PYTHONPATH" nohup setsid python3 -u /home/hsiehph/sauer354/di/ferromic/linked.py > linked.log 2>&1 & tail -f linked.log

warnings.filterwarnings("ignore", category=FutureWarning)
rng = np.random.default_rng(seed=42)

# chr8-7301025-INV-5297356, chr9-102565835-INV-4446, chr9-30951702-INV-5595

def _ensure_worker_logging():
    """
    If we're inside a joblib 'loky' worker, the root logger has no handlers.
    Attach a FileHandler (append) and a StreamHandler so INFO logs show up.
    """
    root = logging.getLogger()
    if not root.handlers:  # worker process
        fmt = '[%(asctime)s] [%(levelname)s] [%(message)s]'
        datefmt = '%Y-%m-%d %H:%M:%S'
        root.setLevel(logging.INFO)
        fh = logging.FileHandler("log.txt", mode='a')
        sh = logging.StreamHandler(sys.stdout)
        for h in (fh, sh):
            h.setFormatter(logging.Formatter(fmt, datefmt))
            root.addHandler(h)

def create_synthetic_data(X_hap1: np.ndarray, X_hap2: np.ndarray, raw_gts: pd.Series,
                          sample_indices: np.ndarray, confidence_mask: np.ndarray,
                          X_existing: np.ndarray, target_counts: dict = None):
    """
    Creates synthetic diploid genomes.

    - The boolean mask argument (confidence_mask) selects which *real samples* are allowed
      to contribute haplotypes as parents. To allow BOTH high- and low-confidence parents,
      pass a mask of all True for the desired parent set. To restrict to a subset (e.g.,
      only high-conf or only train-fold parents), pass a mask with True for allowed parents.

    Modes:
    - "Rescue Mode" (target_counts provided): create the requested number of synthetic
      samples for each dosage class and track parent indices.
    - "Augmentation Mode" (target_counts is None): generate all novel combinations to enrich
      the training set (deduplicated against X_existing).
    """
    hap_pool_0, hap_pool_1 = [], []
    for i in range(len(raw_gts)):
        if not confidence_mask[i]:
            continue
        original_index = sample_indices[i]
        gt = raw_gts.iloc[i]
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

    if not hap_pool_0 and not hap_pool_1:
        return None, None, None

    # Deduplicate haplotypes (arrays -> hashable tuples)
    seen_haps_0, seen_haps_1 = set(), set()
    unique_hap_pool_0, unique_hap_pool_1 = [], []
    for hap, p_idx in hap_pool_0:
        hap_tuple = tuple(hap)
        if hap_tuple not in seen_haps_0:
            seen_haps_0.add(hap_tuple)
            unique_hap_pool_0.append((hap, p_idx))
    for hap, p_idx in hap_pool_1:
        hap_tuple = tuple(hap)
        if hap_tuple not in seen_haps_1:
            seen_haps_1.add(hap_tuple)
            unique_hap_pool_1.append((hap, p_idx))

    existing_genomes_set = {tuple(genome) for genome in X_existing}
    X_synth, y_synth, parent_map = [], [], []

    if target_counts:  # Rescue Mode
        for class_label, num_needed in target_counts.items():
            if num_needed <= 0:
                continue
            for _ in range(num_needed):
                new_diploid, parents = None, None
                if class_label == 2:  # 1|1
                    if not unique_hap_pool_1:
                        return None, None, None
                    h1_idx, h2_idx = rng.choice(len(unique_hap_pool_1), 2, replace=True)
                    h1, p1 = unique_hap_pool_1[h1_idx]; h2, p2 = unique_hap_pool_1[h2_idx]
                    new_diploid, parents = h1 + h2, [p1, p2]
                elif class_label == 0:  # 0|0
                    if not unique_hap_pool_0:
                        return None, None, None
                    h1_idx, h2_idx = rng.choice(len(unique_hap_pool_0), 2, replace=True)
                    h1, p1 = unique_hap_pool_0[h1_idx]; h2, p2 = unique_hap_pool_0[h2_idx]
                    new_diploid, parents = h1 + h2, [p1, p2]
                elif class_label == 1:  # 0|1
                    if not unique_hap_pool_0 or not unique_hap_pool_1:
                        return None, None, None
                    h0_idx, h1_idx = rng.choice(len(unique_hap_pool_0)), rng.choice(len(unique_hap_pool_1))
                    h0, p0 = unique_hap_pool_0[h0_idx]; h1, p1 = unique_hap_pool_1[h1_idx]
                    new_diploid, parents = h0 + h1, [p0, p1]

                if new_diploid is not None:
                    X_synth.append(new_diploid)
                    y_synth.append(class_label)
                    parent_map.append(parents)
    else:  # Augmentation Mode
        n_unique_0, n_unique_1 = len(unique_hap_pool_0), len(unique_hap_pool_1)
        if n_unique_1 >= 2:
            for i, j in itertools.combinations_with_replacement(range(n_unique_1), 2):
                h1, _ = unique_hap_pool_1[i]; h2, _ = unique_hap_pool_1[j]
                new_diploid = h1 + h2
                if tuple(new_diploid) not in existing_genomes_set:
                    X_synth.append(new_diploid); y_synth.append(2)
        if n_unique_0 >= 2:
            for i, j in itertools.combinations_with_replacement(range(n_unique_0), 2):
                h1, _ = unique_hap_pool_0[i]; h2, _ = unique_hap_pool_0[j]
                new_diploid = h1 + h2
                if tuple(new_diploid) not in existing_genomes_set:
                    X_synth.append(new_diploid); y_synth.append(0)
        if n_unique_0 >= 1 and n_unique_1 >= 1:
            for i, j in itertools.product(range(n_unique_0), range(n_unique_1)):
                h0, _ = unique_hap_pool_0[i]; h1, _ = unique_hap_pool_1[j]
                new_diploid = h0 + h1
                if tuple(new_diploid) not in existing_genomes_set:
                    X_synth.append(new_diploid); y_synth.append(1)

    if not X_synth:
        return None, None, None
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

        gt_data = {vcf_s: {'dosage': d, 'is_high_conf': hc, 'raw_gt': rgt}
                   for tsv_s, vcf_s in sample_map.items()
                   for d, hc, rgt in [parse_gt_for_synth(inversion_job.get(tsv_s))]
                   if d is not None}
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
            if var.POS in processed_pos:
                continue
            normalized_chrom = var.CHROM.replace('chr', '')
            snp_id_str = f"{normalized_chrom}:{var.POS}"
            if snp_id_str not in allowed_snps_dict:
                continue
            if not (var.is_snp and not var.is_indel and len(var.ALT) == 1 and all(-1 not in gt[:2] for gt in var.genotypes)):
                continue
            effect_allele = allowed_snps_dict[snp_id_str]
            ref_allele, alt_allele = var.REF, var.ALT[0]
            genotypes = np.array([gt[:2] for gt in var.genotypes], dtype=int)
            if effect_allele == alt_allele:
                encoded_gts = genotypes
            elif effect_allele == ref_allele:
                encoded_gts = 1 - genotypes
            else:
                continue
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

        return {'status': 'PREPROCESSED',
                'id': inversion_id,
                'X_hap1': df_H1.loc[common_samples].values,
                'X_hap2': df_H2.loc[common_samples].values,
                'y_diploid': gt_df.loc[common_samples, 'dosage'].values.astype(int),
                'confidence_mask': gt_df.loc[common_samples, 'is_high_conf'].values.astype(bool),
                'raw_gts': gt_df.loc[common_samples, 'raw_gt'],
                'snp_metadata': snp_meta}
    except Exception as e:
        reason = f"Data Extraction Error: {type(e).__name__}: {e}. Problem VCF: '{vcf_path}'"
        logging.error(f"[{inversion_id}] FAILED: {reason}")
        return {'status': 'FAILED', 'id': inversion_id, 'reason': reason}

def get_effective_max_components(X_train, y_train, max_components):
    """
    Determine a safe per-fold upper bound for PLS components that will not fail.
    Uses dimensional caps, numerical rank under LD, and early-stop signals.
    """
    n_samples = int(X_train.shape[0])
    n_features = int(X_train.shape[1]) if X_train.ndim == 2 else 0
    logging.info(f"[PLS-BOUND] start: n={n_samples}, p={n_features}, requested_max={int(max_components)}")

    if n_features == 0:
        logging.warning("[PLS-BOUND] no features available; returning 0 components")
        return 0

    hard_bound_dim = min(n_samples, n_features)
    logging.info(f"[PLS-BOUND] dimensional cap min(n,p)={hard_bound_dim}")

    hard_bound = hard_bound_dim
    try:
        numeric_rank = int(np.linalg.matrix_rank(X_train))
        logging.info(f"[PLS-BOUND] numeric matrix rank={numeric_rank}")
        hard_bound = min(hard_bound, numeric_rank)
        logging.info(f"[PLS-BOUND] cap after rank={hard_bound}")
    except Exception as rank_err:
        logging.warning(f"[PLS-BOUND] rank computation failed ({type(rank_err).__name__}); using dimensional cap={hard_bound_dim}")

    bound = min(int(max_components), hard_bound)
    logging.info(f"[PLS-BOUND] requested_max clipped to bound={bound}")

    if bound <= 1:
        logging.info(f"[PLS-BOUND] bound ≤ 1; returning {bound}")
        return bound

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PLSRegression(n_components=bound).fit(X_train, y_train)

        earliest_stop = None
        for msg in w:
            text = str(msg.message)
            logging.info(f"[PLS-BOUND] captured warning during probe-fit: {text}")
            m = re.search(r"iteration (\d+)", text)
            if m and ("y residual is constant" in text or "x_scores are null" in text):
                k = int(m.group(1))
                earliest_stop = k if earliest_stop is None else min(earliest_stop, k)

        if earliest_stop is not None:
            new_bound = max(1, min(bound, earliest_stop))
            logging.info(f"[PLS-BOUND] early-stop at iteration={earliest_stop}; returning tightened bound={new_bound}")
            return new_bound

    except ValueError as e:
        logging.warning(f"[PLS-BOUND] ValueError during probe-fit: {e}")
        m = re.search(r"upper bound is (\d+)", str(e))
        if m:
            recovered = max(1, int(m.group(1)))
            logging.info(f"[PLS-BOUND] recovered feasible bound from error={recovered}")
            return recovered
        logging.exception("[PLS-BOUND] unrecoverable error during probe-fit")
        raise

    logging.info(f"[PLS-BOUND] returning final bound={bound}")
    return bound

def analyze_and_model_locus_pls(preloaded_data: dict, n_jobs_inner: int, output_dir: str):
    """
    Policy enforced here:
    - TEST folds are composed of high-confidence REAL samples only.
    - TRAIN folds include all remaining REAL samples (high + low).
    - Synthetic data (rescue + augmentation) are generated per fold using TRAIN parents only,
      unconditionally allowing low-confidence parents.
    - No synthetic data ever enters TEST.
    - Final model is fit on all REAL samples (high + low) plus global augmentation
      generated from all parents.
    """
    _ensure_worker_logging()
    inversion_id = preloaded_data['id']

    logging.info(f"[{inversion_id}] START: modeling begins "
                 f"(n_samples={len(preloaded_data['y_diploid'])}, "
                 f"n_snps={len(preloaded_data.get('snp_metadata', []))})")

    try:
        y_full = preloaded_data['y_diploid']
        X_hap1, X_hap2 = preloaded_data['X_hap1'], preloaded_data['X_hap2']
        confidence_mask = preloaded_data['confidence_mask']
        X_full = X_hap1 + X_hap2

        # --- NEW: High-conf-only TEST; synth from TRAIN parents (high+low) ---
        num_real_samples = len(y_full)
        all_real_idx = np.arange(num_real_samples)

        # TEST pool = strictly high-confidence real samples
        high_mask = confidence_mask
        test_pool_idx = np.where(high_mask)[0]
        y_high = y_full[test_pool_idx]

        if len(test_pool_idx) < 2:
            reason = "Too few high-confidence samples to create a test fold."
            logging.error(f"[{inversion_id}] FAILED: {reason}")
            return {'status': 'FAILED', 'id': inversion_id, 'reason': reason}

        # Prefer stratification on high-conf labels; fall back to KFold if necessary
        high_counts = Counter(y_high)
        min_high_class = min(high_counts.values()) if high_counts else 0
        if min_high_class >= 2:
            n_outer_splits = min(5, min_high_class)
            outer_splitter = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=42)
            fold_iter = outer_splitter.split(test_pool_idx, y_high)
        else:
            n_outer_splits = min(5, len(test_pool_idx))
            if n_outer_splits < 2:
                reason = "Not enough high-confidence samples for KFold."
                logging.error(f"[{inversion_id}] FAILED: {reason}")
                return {'status': 'FAILED', 'id': inversion_id, 'reason': reason}
            outer_splitter = KFold(n_splits=n_outer_splits, shuffle=True, random_state=42)
            fold_iter = outer_splitter.split(test_pool_idx)

        y_true_pooled, y_pred_pls_pooled, y_pred_dummy_pooled = [], [], []

        for fold_idx, (train_hi_unused, test_hi_local) in enumerate(fold_iter, start=1):
            # TEST = high-conf only (real)
            test_idx = test_pool_idx[test_hi_local]

            # TRAIN (real) = everyone else (high + low)
            train_real_idx = np.setdiff1d(all_real_idx, test_idx, assume_unique=True)

            # Helper: per-fold synth from TRAIN parents ONLY (allow lowconf unconditionally)
            def _build_fold_synth(target_counts=None):
                parent_mask = np.isin(all_real_idx, train_real_idx)  # parents restricted to train fold
                return create_synthetic_data(
                    X_hap1, X_hap2, preloaded_data['raw_gts'],
                    all_real_idx, parent_mask,
                    X_full, target_counts=target_counts
                )

            # Per-fold RESCUE (top-up any class in train to at least 2)
            class_counts_train = Counter(y_full[train_real_idx])
            needed_counts = {c: max(0, 2 - class_counts_train.get(c, 0)) for c in [0, 1, 2]}
            X_rescue, y_rescue, _ = (None, None, None)
            if any(v > 0 for v in needed_counts.values()):
                X_rescue, y_rescue, _ = _build_fold_synth(target_counts=needed_counts)

            # Per-fold AUGMENTATION (novel combos from train parents)
            X_aug, y_aug, _ = _build_fold_synth(target_counts=None)

            # Assemble TRAIN/TEST sets (no synth in TEST)
            X_train = X_full[train_real_idx]; y_train = y_full[train_real_idx]
            if X_rescue is not None:
                X_train = np.vstack([X_train, X_rescue]); y_train = np.concatenate([y_train, y_rescue])
            if X_aug is not None:
                X_train = np.vstack([X_train, X_aug]);    y_train = np.concatenate([y_train, y_aug])

            X_test = X_full[test_idx]; y_test = y_full[test_idx]

            # Skip folds without sufficient class diversity
            if len(X_train) == 0 or len(np.unique(y_train)) < 2:
                logging.info(f"[{inversion_id}] EVAL fold {fold_idx}/{n_outer_splits}: skipped (train lacks >=2 classes).")
                continue

            # Balanced bootstrap on TRAIN
            sample_weights = compute_sample_weight("balanced", y=y_train)
            resampled_indices = rng.choice(
                len(X_train), size=len(X_train), replace=True,
                p=sample_weights / np.sum(sample_weights)
            )
            X_train_resampled = X_train[resampled_indices]; y_train_resampled = y_train[resampled_indices]
            train_min_class_count = min(Counter(y_train_resampled).values())
            if train_min_class_count < 2:
                logging.info(f"[{inversion_id}] EVAL fold {fold_idx}/{n_outer_splits}: skipped (post-resample class <2).")
                continue

            # Tune PLS components safely
            max_components = min(100, X_train_resampled.shape[0] - 1)
            effective_max_components = get_effective_max_components(X_train_resampled, y_train_resampled, max_components)
            if effective_max_components < 1:
                logging.info(f"[{inversion_id}] EVAL fold {fold_idx}/{n_outer_splits}: skipped (no valid n_components).")
                continue

            pipeline = Pipeline([('pls', PLSRegression())])
            inner_cv = StratifiedKFold(n_splits=min(3, train_min_class_count), shuffle=True, random_state=123)
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid={'pls__n_components': range(1, effective_max_components + 1)},
                scoring='neg_mean_squared_error',
                cv=inner_cv, n_jobs=n_jobs_inner, error_score='raise'
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "y residual is constant", UserWarning)
                grid_search.fit(X_train_resampled, y_train_resampled)

            dummy_model_fold = DummyRegressor(strategy='mean').fit(X_train, y_train)
            y_pred_pls_fold = grid_search.best_estimator_.predict(X_test).flatten()
            y_pred_dummy_fold = dummy_model_fold.predict(X_test)

            # Fold metrics + pooling
            if len(y_test) < 2 or np.std(y_test) == 0 or np.std(y_pred_pls_fold) == 0:
                fold_corr, fold_r2 = np.nan, 0.0
            else:
                fold_corr, _ = pearsonr(y_test, y_pred_pls_fold)
                fold_r2 = (fold_corr if not np.isnan(fold_corr) else 0.0) ** 2
            fold_rmse = np.sqrt(mean_squared_error(y_test, y_pred_pls_fold))
            fold_rmse_dummy = np.sqrt(mean_squared_error(y_test, y_pred_dummy_fold))
            best_ncomp = grid_search.best_params_.get('pls__n_components', 'NA')

            logging.info(f"[{inversion_id}] EVAL fold {fold_idx}/{n_outer_splits}: "
                         f"n_train_real={len(train_real_idx)} n_test={len(test_idx)} "
                         f"best_ncomp={best_ncomp} r2={fold_r2:.4f} rmse={fold_rmse:.4f} "
                         f"dummy_rmse={fold_rmse_dummy:.4f}")

            y_true_pooled.extend(y_test)
            y_pred_pls_pooled.extend(y_pred_pls_fold)
            y_pred_dummy_pooled.extend(y_pred_dummy_fold)

        if not y_true_pooled:
            reason = "Cross-validation produced no evaluable folds (class structure too sparse)."
            logging.error(f"[{inversion_id}] FAILED: {reason}")
            return {'status': 'FAILED', 'id': inversion_id, 'reason': reason}

        y_true_arr, y_pred_arr = np.array(y_true_pooled), np.array(y_pred_pls_pooled)
        if y_true_arr.size < 2 or np.std(y_true_arr) == 0 or np.std(y_pred_arr) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(y_true_arr, y_pred_arr)

        pearson_r2 = (corr if not np.isnan(corr) else 0.0) ** 2
        _, p_value = wilcoxon(np.abs(y_true_arr - y_pred_arr),
                              np.abs(y_true_arr - np.array(y_pred_dummy_pooled)),
                              alternative='less', zero_method='zsplit')

        pooled_rmse = np.sqrt(mean_squared_error(y_true_pooled, y_pred_pls_pooled))
        logging.info(f"[{inversion_id}] EVAL summary (pooled over outer folds): "
                     f"unbiased_r2={pearson_r2:.4f} pooled_rmse={pooled_rmse:.4f} "
                     f"p_wilcoxon={p_value:.3g} (PLS vs Dummy)")

        # -------------------- FINAL MODEL (global) --------------------
        # Augment the final training set as well (allow lowconf parents unconditionally)
        X_final_aug, y_final_aug, _ = create_synthetic_data(
            X_hap1, X_hap2, preloaded_data['raw_gts'],
            np.arange(num_real_samples),
            np.ones_like(confidence_mask, dtype=bool),  # allow both high+low as parents
            X_full, target_counts=None
        )
        X_final_train_full = X_full
        y_final_train_full = y_full
        if X_final_aug is not None:
            X_final_train_full = np.vstack([X_full, X_final_aug])
            y_final_train_full = np.concatenate([y_full, y_final_aug])

        # Balanced bootstrap for final training
        final_sample_weights = compute_sample_weight("balanced", y=y_final_train_full)
        final_resampled_indices = rng.choice(
            len(y_final_train_full), size=len(y_final_train_full), replace=True,
            p=final_sample_weights / np.sum(final_sample_weights)
        )
        X_final_train, y_final_train = X_final_train_full[final_resampled_indices], y_final_train_full[final_resampled_indices]

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
        final_grid_search = GridSearchCV(
            estimator=final_pipeline,
            param_grid={'pls__n_components': range(1, final_effective_max + 1)},
            scoring='neg_mean_squared_error',
            cv=final_cv, n_jobs=n_jobs_inner, refit=True
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "y residual is constant", UserWarning)
            final_grid_search.fit(X_final_train, y_final_train)

        os.makedirs(output_dir, exist_ok=True)
        model_filename = os.path.join(output_dir, f"{inversion_id}.model.joblib")
        dump(final_grid_search.best_estimator_, model_filename)
        pd.DataFrame(preloaded_data['snp_metadata']).to_json(
            os.path.join(output_dir, f"{inversion_id}.snps.json"), orient='records'
        )

        unbiased_rmse_val = np.sqrt(mean_squared_error(y_true_pooled, y_pred_pls_pooled))
        logging.info(f"[{inversion_id}] SUCCESS: saved model -> {model_filename} | "
                     f"unbiased_r2={pearson_r2:.4f} unbiased_rmse={unbiased_rmse_val:.4f} "
                     f"best_ncomp={final_grid_search.best_params_['pls__n_components']} "
                     f"snps={X_full.shape[1]}")
        return {
            'status': 'SUCCESS',
            'id': inversion_id,
            'unbiased_pearson_r2': pearson_r2,
            'unbiased_rmse': unbiased_rmse_val,
            'model_p_value': p_value,
            'best_n_components': final_grid_search.best_params_['pls__n_components'],
            'num_snps_in_model': X_full.shape[1],
            'model_path': model_filename
        }
    except Exception as e:
        reason = f"Analysis Error: {type(e).__name__}: {e}"
        logging.error(f"[{inversion_id}] FAILED: {reason}")
        return {'status': 'FAILED', 'id': inversion_id, 'reason': reason, 'traceback': traceback.format_exc()}

def process_locus_end_to_end(job: dict, n_jobs_inner: int, allowed_snps_dict: dict, output_dir: str):
    _ensure_worker_logging()
    inversion_id = job.get('orig_ID', 'Unknown_ID')

    chrom = job.get('seqnames', 'NA'); start = job.get('start', 'NA'); end = job.get('end', 'NA')
    logging.info(f"[{inversion_id}] START: processing locus chr={chrom} start={start} end={end}")

    success_receipt = os.path.join(output_dir, f"{inversion_id}.model.joblib")
    if os.path.exists(success_receipt):
        logging.info(f"[{inversion_id}] CACHED: model already exists at {success_receipt}")
        return {'status': 'CACHED', 'id': inversion_id, 'reason': 'SUCCESS receipt found.'}

    result = extract_haplotype_data_for_locus(job, allowed_snps_dict)
    if result.get('status') == 'PREPROCESSED':
        result = analyze_and_model_locus_pls(result, n_jobs_inner, output_dir)

    # Loud completion status
    if isinstance(result, dict):
        status = result.get('status', 'UNKNOWN')
        reason = result.get('reason', '')
        if status == 'SUCCESS':
            logging.info(f"[{inversion_id}] COMPLETE: SUCCESS")
        elif status == 'SKIPPED':
            logging.info(f"[{inversion_id}] COMPLETE: SKIPPED ({reason})")
        elif status == 'FAILED':
            logging.info(f"[{inversion_id}] COMPLETE: FAILED ({reason})")
        else:
            logging.info(f"[{inversion_id}] COMPLETE: {status}")
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
    if total_cores < N_INNER_JOBS:
        N_INNER_JOBS = total_cores
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
        logging.warning("\n" + "=" * 80)
        logging.warning(f"--- PRE-CHECK WARNING: Found {len(loci_without_snps)} loci that will fail due to no suitable SNPs ---")
        for failed_locus in sorted(loci_without_snps, key=lambda x: x['id']):
            logging.warning(f"  - [{failed_locus['id']}] will fail: No SNPs from whitelist found in region [{failed_locus['region']}]")
        logging.warning("=" * 80 + "\n")
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

    logging.info("\n\n" + "=" * 100 + "\n---      FINAL PIPELINE REPORT      ---\n" + "=" * 100)

    # Aggregate incomplete outcomes by reason.
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

        # Build a mapping from inversion id to original job for coordinate lookup.
        id_to_job = {job['orig_ID']: job for job in all_jobs}

        # Loud per-locus printout for all non-successful runs with coordinates.
        logging.info("\n--- Per-locus details for incomplete runs ---")
        for r in other_runs:
            j = id_to_job.get(r.get('id'), {})
            chrom = j.get('seqnames', 'NA')
            start = j.get('start', 'NA')
            end = j.get('end', 'NA')
            status = r.get('status')
            rid = r.get('id')
            reason = r.get('reason', 'No reason provided.')
            logging.info(f"[LOCI] [{status}] {rid} chr={chrom} start={start} end={end} reason={reason}")

        # Highlight component-bound failures explicitly.
        comp_bound = [r for r in other_runs if "upper bound" in str(r.get('reason', '')).lower() or "n_components" in str(r.get('reason', '')).lower()]
        if comp_bound:
            logging.info("\n--- Component-bound failures (with coordinates) ---")
            for r in comp_bound:
                j = id_to_job.get(r.get('id'), {})
                chrom = j.get('seqnames', 'NA')
                start = j.get('start', 'NA')
                end = j.get('end', 'NA')
                rid = r.get('id')
                reason = r.get('reason', 'No reason provided.')
                logging.info(f"[COMP-BOUND] {rid} chr={chrom} start={start} end={end} reason={reason}")

        # Print crashed workers with coordinates.
        if crashed_ids:
            logging.info("\n--- Crashed worker loci (with coordinates) ---")
            for rid in sorted(crashed_ids):
                j = id_to_job.get(rid, {})
                chrom = j.get('seqnames', 'NA')
                start = j.get('start', 'NA')
                end = j.get('end', 'NA')
                logging.info(f"[CRASHED] {rid} chr={chrom} start={start} end={end}")

    final_models = [f for f in os.listdir(output_dir) if f.endswith('.model.joblib')]
    logging.info(f"\n--- Total Successful Models in Output Directory: {len(final_models)} ---")

    if successful_runs:
        results_df = pd.DataFrame(successful_runs).set_index('id').sort_values('unbiased_pearson_r2', ascending=False)
        newly_successful_ids = list(results_df.index)
        logging.info("\n--- Newly Successful IDs ---\n" + "\n".join(f"  - {i}" for i in newly_successful_ids))

        # Always echo per-locus metrics from parent so they can't get lost
        for _id, r in results_df.iterrows():
            logging.info(
                f"[{_id}] METRICS: r2={r['unbiased_pearson_r2']:.3f} "
                f"rmse={r['unbiased_rmse']:.3f} p={r['model_p_value']:.3g} "
                f"ncomp={int(r['best_n_components'])} snps={int(r['num_snps_in_model'])} "
                f"path={r['model_path']}"
            )

        ids_txt_path = os.path.join(output_dir, "newly_successful_ids.txt")
        with open(ids_txt_path, "w") as fh:
            for i in newly_successful_ids:
                fh.write(f"{i}\n")
        results_csv_path = os.path.join(output_dir, "newly_successful_models.csv")
        results_df.to_csv(results_csv_path)
        logging.info(f"\nSaved IDs to: {ids_txt_path}\nSaved metrics to: {results_csv_path}")
        logging.info(f"\n--- Performance of {len(successful_runs)} NEWLY Successful Models ---")
        high_perf_df = results_df[(results_df['unbiased_pearson_r2'] > 0.5) & (results_df['model_p_value'] < 0.05)]
        if not high_perf_df.empty:
            summary_cols = ['unbiased_pearson_r2', 'unbiased_rmse', 'model_p_value', 'best_n_components', 'num_snps_in_model']
            with pd.option_context('display.max_rows', None, 'display.width', 120):
                logging.info("\n" + high_perf_df[summary_cols].to_string())
        else:
            logging.info("\n--- No new high-performance models were generated in this run. ---")

    logging.info("\n" + "=" * 100)
