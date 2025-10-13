"""Category-level omnibus and directional tests.

This module builds on the per-phenotype PheWAS results to compute two
category-level metrics per inversion:

1. A dependence-aware omnibus p-value using the Generalized Berk–Jones (GBJ)
   statistic calibrated with correlated null draws.
2. A correlation-weighted directional meta z-score (generalised least squares).

The implementation follows the design notes in the project documentation and
is intentionally conservative: we expose a ``fast_phi`` mode that derives the
correlation structure from case overlap counts and applies shrinkage to keep
matrices well-conditioned.  The heavier ``exact_wild`` mode described in the
notes can be added later without changing the public API.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from . import models
from . import pheno


@dataclass
class CategoryNull:
    """Container for per-category null correlation information."""

    phenotypes: List[str]
    covariance: np.ndarray
    method: str
    shrinkage: str
    lambda_value: float
    n_individuals: int
    dropped: List[str] = field(default_factory=list)


def load_dedup_manifest(cache_dir: str, cdr_codename: str, core_index: pd.Index) -> Mapping[str, object]:
    """Load the cohort-specific phenotype deduplication manifest if present."""

    try:
        cohort_fp = models._index_fingerprint(core_index)
    except Exception:
        return {}
    manifest_path = os.path.join(cache_dir, f"pheno_dedup_manifest_{cdr_codename}_{cohort_fp}.json")
    if not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, "r") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def plan_category_sets(
    phenotype_names: Sequence[str],
    name_to_cat: Mapping[str, str],
    dedup_manifest: Optional[Mapping[str, object]] = None,
    *,
    min_k: int = 3,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Determine the phenotypes to keep per category after deduplication."""

    kept_candidates = {str(p) for p in phenotype_names if p}
    allowed_from_manifest: Optional[set] = None
    if dedup_manifest and isinstance(dedup_manifest.get("kept"), list):
        allowed_from_manifest = {str(p) for p in dedup_manifest["kept"] if p}
        kept_candidates &= allowed_from_manifest

    plan: Dict[str, List[str]] = {}
    dropped: Dict[str, List[str]] = {}
    for pheno_name in sorted(kept_candidates):
        cat = name_to_cat.get(pheno_name, "uncategorized") or "uncategorized"
        plan.setdefault(cat, []).append(pheno_name)

    filtered_plan: Dict[str, List[str]] = {}
    for cat, phenos in plan.items():
        if len(phenos) >= max(1, int(min_k)):
            filtered_plan[cat] = phenos
        else:
            dropped[cat] = phenos
    return filtered_plan, dropped


def _apply_shrinkage(matrix: np.ndarray, *, method: str = "ridge", lambda_value: float = 0.05) -> np.ndarray:
    """Apply a simple shrinkage procedure to keep covariance matrices PD."""

    matrix = np.asarray(matrix, dtype=np.float64)
    p = matrix.shape[0]
    if p == 0:
        return matrix

    if not np.allclose(matrix, matrix.T, atol=1e-8):
        matrix = (matrix + matrix.T) / 2.0

    lam = float(lambda_value)
    if method.lower() == "ridge":
        lam = min(max(lam, 0.0), 1.0)
        shrunk = (1.0 - lam) * matrix + lam * np.eye(p, dtype=np.float64)
    else:
        lam = max(lam, 1e-6)
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.clip(eigvals, lam, None)
        shrunk = (eigvecs @ np.diag(eigvals)) @ eigvecs.T
    np.fill_diagonal(shrunk, 1.0)
    return shrunk


def _phi_covariance_for_category(
    category: str,
    phenotypes: Sequence[str],
    case_indices: Mapping[str, np.ndarray],
    core_index_size: int,
    allowed_mask: Optional[np.ndarray],
    global_mask: Optional[np.ndarray],
    *,
    shrinkage: str,
    lambda_value: float,
    min_k: int,
) -> Optional[CategoryNull]:
    allowed = None
    if allowed_mask is not None:
        allowed = allowed_mask.astype(bool, copy=True)
    elif global_mask is not None:
        allowed = global_mask.astype(bool, copy=True)
    else:
        allowed = np.ones(core_index_size, dtype=bool)

    dropped: List[str] = []
    candidate_names: List[str] = []
    candidate_indices: Dict[str, np.ndarray] = {}

    for name in phenotypes:
        idx = case_indices.get(name)
        if idx is None or idx.size == 0:
            dropped.append(name)
            continue
        allowed[idx] = True
        candidate_names.append(name)
        candidate_indices[name] = idx

    used_idx = np.flatnonzero(allowed)
    if used_idx.size == 0:
        return None

    used_vectors: List[np.ndarray] = []
    used_names: List[str] = []
    for name in candidate_names:
        idx = candidate_indices.get(name)
        if idx is None:
            continue
        case_indicator = np.in1d(used_idx, idx, assume_unique=False)
        n1 = int(case_indicator.sum())
        if n1 == 0 or n1 == used_idx.size:
            dropped.append(name)
            continue
        used_vectors.append(case_indicator)
        used_names.append(name)

    if len(used_vectors) < max(1, min_k):
        return None
    n_people = int(used_idx.size)
    K = len(used_vectors)
    Sigma = np.eye(K, dtype=np.float64)

    for i in range(K):
        vec_i = used_vectors[i]
        n1_i = int(vec_i.sum())
        for j in range(i + 1, K):
            vec_j = used_vectors[j]
            n1_j = int(vec_j.sum())
            n11 = int(np.sum(vec_i & vec_j))
            n10 = n1_i - n11
            n01 = n1_j - n11
            n00 = n_people - (n11 + n10 + n01)
            phi_val = pheno.phi_from_2x2(n11, n10, n01, n00)
            if not np.isfinite(phi_val):
                phi_val = 0.0
            Sigma[i, j] = Sigma[j, i] = float(phi_val)

    Sigma = _apply_shrinkage(Sigma, method=shrinkage, lambda_value=lambda_value)
    return CategoryNull(
        phenotypes=used_names,
        covariance=Sigma,
        method="fast_phi",
        shrinkage=shrinkage,
        lambda_value=lambda_value,
        n_individuals=n_people,
        dropped=dropped,
    )


def build_category_null_structure(
    core_df_with_const: pd.DataFrame,
    allowed_mask_by_cat: Mapping[str, np.ndarray],
    category_sets: Mapping[str, Sequence[str]],
    *,
    cache_dir: str,
    cdr_codename: str,
    method: str = "fast_phi",
    shrinkage: str = "ridge",
    lambda_value: float = 0.05,
    min_k: int = 3,
    global_mask: Optional[np.ndarray] = None,
) -> Dict[str, CategoryNull]:
    """Build correlation-aware null structures for each category."""

    if not category_sets:
        return {}

    core_index = core_df_with_const.index
    case_indices: Dict[str, np.ndarray] = {}
    for name in sorted({p for phenos in category_sets.values() for p in phenos}):
        try:
            case_ids = pheno._case_ids_cached(name, cdr_codename, cache_dir)
        except Exception:
            continue
        if not case_ids:
            continue
        pos = core_index.get_indexer(pd.Index(case_ids))
        pos = pos[pos >= 0]
        if pos.size == 0:
            continue
        case_indices[name] = pos.astype(np.int32, copy=False)

    structures: Dict[str, CategoryNull] = {}
    for cat, phenos in category_sets.items():
        if method != "fast_phi":
            raise NotImplementedError("Only the fast_phi mode is currently implemented.")
        allowed_mask = allowed_mask_by_cat.get(cat)
        struct = _phi_covariance_for_category(
            cat,
            phenos,
            case_indices,
            len(core_index),
            allowed_mask,
            global_mask,
            shrinkage=shrinkage,
            lambda_value=lambda_value,
            min_k=min_k,
        )
        if struct is not None:
            structures[cat] = struct
    return structures


def _gbj_statistic(p_values: np.ndarray) -> float:
    """Compute the Berk–Jones statistic for a vector of p-values."""

    if p_values.size == 0:
        return 0.0
    p_sorted = np.sort(np.clip(p_values.astype(float), 1e-300, 1 - 1e-16))
    m = p_sorted.size
    best = 0.0
    for k, p in enumerate(p_sorted, start=1):
        threshold = k / m
        if p > threshold:
            continue
        term1 = k * math.log(max(k / (m * p), 1e-12))
        if k == m:
            term2 = 0.0
        else:
            term2 = (m - k) * math.log(max((m - k) / (m * (1.0 - p)), 1e-12))
        stat = term1 + term2
        if stat > best:
            best = stat
    return float(best)


def _simulate_gbj_pvalue(
    observed_stat: float,
    covariance: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> float:
    if draws <= 0:
        return float("nan")
    p = covariance.shape[0]
    if p == 0:
        return float("nan")
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(covariance)
        eigvals = np.clip(eigvals, 1e-6, None)
        cov_pd = (eigvecs @ np.diag(eigvals)) @ eigvecs.T
        chol = np.linalg.cholesky(cov_pd)
    stats_obs = 0
    for _ in range(draws):
        sample = rng.standard_normal(p) @ chol.T
        pvals = 2.0 * stats.norm.sf(np.abs(sample))
        stat = _gbj_statistic(pvals)
        if stat >= observed_stat:
            stats_obs += 1
    return float((stats_obs + 1) / (draws + 1))


def _directional_meta_z(z_scores: np.ndarray, covariance: np.ndarray) -> Tuple[float, float]:
    if z_scores.size == 0:
        return float("nan"), float("nan")
    ones = np.ones(z_scores.size, dtype=np.float64)
    try:
        weights = np.linalg.solve(covariance, ones)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(covariance) @ ones
    denom = float(np.dot(ones, weights))
    if denom <= 0:
        return float("nan"), float("nan")
    numerator = float(np.dot(weights, z_scores))
    t_stat = numerator / math.sqrt(denom)
    p_value = float(2.0 * stats.norm.sf(abs(t_stat)))
    return t_stat, p_value


def compute_category_metrics(
    per_pheno_results: pd.DataFrame,
    *,
    p_col: str,
    beta_col: str,
    null_structures: Mapping[str, CategoryNull],
    gbj_draws: int = 5000,
    z_cap: float = 8.0,
    rng_seed: Optional[int] = None,
    min_k: int = 3,
) -> pd.DataFrame:
    """Compute GBJ and directional GLS metrics per category."""

    if per_pheno_results.empty or not null_structures:
        return pd.DataFrame(columns=[
            "Category",
            "K_Total",
            "K_GBJ",
            "K_GLS",
            "P_GBJ",
            "T_GLS",
            "P_GLS",
            "Direction",
            "Method",
            "Shrinkage",
            "Lambda",
            "N_Individuals",
            "Z_Cap",
            "GBJ_Draws",
            "Dropped",
            "Phenotypes",
        ])

    df = per_pheno_results.copy()
    if "Phenotype" in df.columns:
        df = df.set_index("Phenotype")
    rng = np.random.default_rng(rng_seed)

    records: List[MutableMapping[str, object]] = []
    for cat, struct in null_structures.items():
        phenos = list(struct.phenotypes)
        if not phenos:
            continue
        gbj_indices: List[int] = []
        gbj_z: List[float] = []
        gbj_names: List[str] = []
        dir_indices: List[int] = []
        dir_z: List[float] = []
        dir_names: List[str] = []
        missing: List[str] = []

        for idx, name in enumerate(phenos):
            if name not in df.index:
                missing.append(name)
                continue
            row = df.loc[name]
            pval = float(row.get(p_col, float("nan")))
            if not np.isfinite(pval):
                missing.append(name)
                continue
            pval = float(np.clip(pval, 1e-300, 1.0 - 1e-16))
            z_abs = float(stats.norm.isf(pval / 2.0))
            if not np.isfinite(z_abs):
                missing.append(name)
                continue
            if z_cap > 0:
                z_abs = float(min(abs(z_abs), z_cap))
            gbj_indices.append(idx)
            gbj_z.append(z_abs)
            gbj_names.append(name)
            beta = row.get(beta_col)
            if beta is None or (isinstance(beta, float) and (math.isnan(beta) or math.isinf(beta))):
                continue
            beta_val = float(beta)
            sign = 0.0
            if beta_val > 0:
                sign = 1.0
            elif beta_val < 0:
                sign = -1.0
            dir_indices.append(idx)
            dir_z.append(sign * z_abs)
            dir_names.append(name)

        if len(gbj_indices) < max(1, min_k):
            continue

        Sigma_full = struct.covariance
        gbj_cov = Sigma_full[np.ix_(gbj_indices, gbj_indices)]
        gbj_pvals = 2.0 * stats.norm.sf(np.asarray(gbj_z, dtype=np.float64))
        gbj_stat = _gbj_statistic(gbj_pvals)
        gbj_p = _simulate_gbj_pvalue(gbj_stat, gbj_cov, int(gbj_draws), rng)

        gls_stat = float("nan")
        gls_p = float("nan")
        direction_label = "neutral"
        if dir_indices:
            dir_cov = Sigma_full[np.ix_(dir_indices, dir_indices)]
            dir_z_arr = np.asarray(dir_z, dtype=np.float64)
            gls_stat, gls_p = _directional_meta_z(dir_z_arr, dir_cov)
            if np.isfinite(gls_stat):
                direction_label = "increase" if gls_stat > 0 else "decrease"

        records.append({
            "Category": cat,
            "K_Total": len(struct.phenotypes),
            "K_GBJ": len(gbj_indices),
            "K_GLS": len(dir_indices),
            "P_GBJ": gbj_p,
            "T_GLS": gls_stat,
            "P_GLS": gls_p,
            "Direction": direction_label,
            "Method": struct.method,
            "Shrinkage": struct.shrinkage,
            "Lambda": struct.lambda_value,
            "N_Individuals": struct.n_individuals,
            "Z_Cap": z_cap,
            "GBJ_Draws": int(gbj_draws),
            "Dropped": ";".join(struct.dropped + missing),
            "Phenotypes": ";".join(gbj_names),
        })

    if not records:
        return pd.DataFrame(columns=[
            "Category",
            "K_Total",
            "K_GBJ",
            "K_GLS",
            "P_GBJ",
            "T_GLS",
            "P_GLS",
            "Direction",
            "Method",
            "Shrinkage",
            "Lambda",
            "N_Individuals",
            "Z_Cap",
            "GBJ_Draws",
            "Dropped",
            "Phenotypes",
        ])

    df_out = pd.DataFrame(records)
    df_out.sort_values("P_GBJ", inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out


__all__ = [
    "CategoryNull",
    "load_dedup_manifest",
    "plan_category_sets",
    "build_category_null_structure",
    "compute_category_metrics",
]
