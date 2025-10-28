"""Custom follow-up runner with per-phenotype polygenic score controls.

This module mirrors the shared setup performed by :mod:`phewas.run` but trims it
down to a minimal, targeted workflow:

* Only phenotypes whose names match ``PHENOTYPE_PATTERNS`` are analysed.
* Only the inversions listed in ``TARGET_INVERSIONS`` are considered.
* Each phenotype category draws two additional polygenic score controls from
  ``scores.tsv``. These are selected according to ``CATEGORY_PGS_IDS`` and use
  ``<PGS_ID>_AVG`` columns matched case-insensitively.
* No multiple-testing or FDR correction is applied – raw p-values are reported.

All configuration is expressed as module-level globals; there is no CLI entry
point by design.

The output table ``custom_control_follow_ups.tsv`` is written to the current
working directory.
"""

from __future__ import annotations

import os
import sys
import math
import warnings
from dataclasses import dataclass
from decimal import Context, Decimal, localcontext
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import log_ndtr
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from .. import iox as io
from .. import pheno
from ..run import (
    CACHE_DIR as PIPELINE_CACHE_DIR,
    INVERSION_DOSAGES_FILE,
    LOCK_DIR as PIPELINE_LOCK_DIR,
    NUM_PCS as PIPELINE_NUM_PCS,
    PCS_URI as PIPELINE_PCS_URI,
    PHENOTYPE_DEFINITIONS_URL,
    RELATEDNESS_URI as PIPELINE_RELATEDNESS_URI,
    SEX_URI as PIPELINE_SEX_URI,
    _find_upwards as pipeline_find_upwards,
)

# ---------------------------------------------------------------------------
# Configuration (edit in-place; no CLI)
# ---------------------------------------------------------------------------

TARGET_INVERSIONS: Sequence[str] = (
    "chr17-45974480-INV-29218",
)

@dataclass(frozen=True)
class PhenotypePattern:
    pattern: str
    category: str


PHENOTYPE_PATTERNS: Sequence[PhenotypePattern] = (
    PhenotypePattern("Abnormal_mammogram", "breast_cancer"),
    PhenotypePattern(
        "Lump_or_mass_in_breast_or_nonspecific_abnormal_breast_exam",
        "breast_cancer",
    ),
    PhenotypePattern("Malignant_neoplasm_of_the_breast", "breast_cancer"),
    PhenotypePattern("Diastolic_heart_failure", "obesity"),
    PhenotypePattern("Mild_cognitive_impairment", "alzheimers"),
    PhenotypePattern("*Obesity*", "obesity"),
    PhenotypePattern("Overweight and obesity", "obesity"),
    PhenotypePattern("Alzheimer*", "alzheimers"),
    PhenotypePattern("cognitive decline", "alzheimers"),
    PhenotypePattern("cognitive_decline", "alzheimers"),
    PhenotypePattern("Dementias", "alzheimers"),
)

CATEGORY_PGS_IDS: dict[str, Sequence[str]] = {
    "breast_cancer": ("PGS004869", "PGS000507"),
    "alzheimers": ("PGS004146", "PGS004229"),
    "obesity": ("PGS004378", "PGS005198"),
}

SCORES_FILE = Path("scores.tsv")
CUSTOM_CONTROL_PREFIX = "PGS"
OUTPUT_PATH = Path("custom_control_follow_ups.tsv")

NUM_PCS = PIPELINE_NUM_PCS

CONFIDENCE_Z = 1.959963984540054  # scipy.stats.norm.ppf(0.975)
MIN_LOG_FLOAT = math.log(sys.float_info.min)
MAX_LOG_FLOAT = math.log(sys.float_info.max)
DECIMAL_CONTEXT = Context(prec=50, Emin=-999999, Emax=999999)


def _base_covariate_columns(num_pcs: int = NUM_PCS) -> list[str]:
    """Return the baseline covariate columns used in the main PheWAS pipeline."""

    return ["sex", *[f"PC{i}" for i in range(1, num_pcs + 1)], "AGE_c", "AGE_c_sq"]

# ---------------------------------------------------------------------------
# Data source configuration (mirrors ``phewas.run`` defaults)
# ---------------------------------------------------------------------------

CACHE_DIR = Path(PIPELINE_CACHE_DIR)
LOCK_DIR = Path(PIPELINE_LOCK_DIR)
PCS_URI = PIPELINE_PCS_URI
SEX_URI = PIPELINE_SEX_URI
RELATEDNESS_URI = PIPELINE_RELATEDNESS_URI

# ---------------------------------------------------------------------------
# Simple logging helpers
# ---------------------------------------------------------------------------


def info(message: str) -> None:
    print(f"[custom-followup] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[custom-followup][WARN] {message}", flush=True)


def die(message: str) -> None:
    warn(message)
    sys.exit(1)


def _compute_condition_number(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return float("nan")
    with np.errstate(all="ignore"):
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    finite = singular_values[np.isfinite(singular_values)]
    if finite.size == 0:
        return float("nan")
    smallest = finite.min(initial=0.0)
    largest = finite.max(initial=0.0)
    if smallest == 0.0:
        return float("inf")
    return float(largest / smallest)


def _estimate_rank(matrix: np.ndarray) -> int:
    if matrix.size == 0:
        return 0
    with np.errstate(all="ignore"):
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return 0
    eps = np.finfo(matrix.dtype).eps if matrix.dtype.kind == "f" else np.finfo(np.float64).eps
    tolerance = singular_values.max(initial=0.0) * max(matrix.shape) * eps
    return int(np.sum(singular_values > tolerance))


def _summarise_column(values: pd.Series) -> dict[str, object]:
    finite = pd.to_numeric(values, errors="coerce")
    stats: dict[str, object] = {
        "dtype": str(values.dtype),
        "unique": int(values.nunique(dropna=False)),
    }
    if finite.notna().any():
        stats.update(
            {
                "min": float(finite.min(skipna=True)),
                "max": float(finite.max(skipna=True)),
                "mean": float(finite.mean(skipna=True)),
                "std": float(finite.std(skipna=True)),
            }
        )
    return stats


def _summarise_design_matrix(X: pd.DataFrame, y: pd.Series | None = None) -> str:
    matrix = X.to_numpy(dtype=np.float64, copy=False)
    diagnostics = X.attrs.get("diagnostics", {})
    condition_number = diagnostics.get("condition_number", _compute_condition_number(matrix))
    rank = diagnostics.get("matrix_rank", _estimate_rank(matrix))
    lines = [
        f"Design matrix shape: rows={X.shape[0]}, columns={X.shape[1]}",
        f"Estimated rank: {rank}",
        f"Condition number: {condition_number:.3e}" if np.isfinite(condition_number) else f"Condition number: {condition_number}",
    ]
    if y is not None:
        y_cases = int(pd.to_numeric(y, errors="coerce").sum())
        lines.append(f"Outcome summary: cases={y_cases}, controls={len(y) - y_cases}")

    if diagnostics:
        dropped_non_finite = diagnostics.get("dropped_non_finite", 0)
        if dropped_non_finite:
            lines.append(f"Rows dropped for non-finite covariates: {dropped_non_finite}")
        for key in ("dropped_constant", "dropped_duplicates", "dropped_collinear"):
            values = diagnostics.get(key)
            if values:
                label = key.replace("_", " ")
                lines.append(f"{label.title()}: {', '.join(values)}")
        if diagnostics.get("ancestry_constants"):
            lines.append(
                "Ancestry dummies constant after alignment: "
                + ", ".join(diagnostics["ancestry_constants"])
            )


def _decimal_to_string(value: Decimal) -> str:
    normalized = value.normalize() if value != 0 else value
    if normalized == 0:
        return "0"
    exponent = normalized.adjusted()
    if -6 <= exponent <= 6:
        return format(normalized, "f")
    return format(normalized, "E")


def _exp_decimal(value: float) -> Decimal | None:
    if not math.isfinite(value):
        return None
    with localcontext(DECIMAL_CONTEXT):
        return Decimal(str(value)).exp()


def _log_value_to_decimal(log_value: float) -> Decimal | None:
    if math.isnan(log_value):
        return None
    if log_value == float("-inf"):
        return Decimal(0)
    if not math.isfinite(log_value):
        return None
    with localcontext(DECIMAL_CONTEXT):
        return Decimal(str(log_value)).exp()


def _format_probability_from_log(log_value: float | None) -> float | str:
    if log_value is None or math.isnan(log_value):
        return math.nan
    if log_value >= MIN_LOG_FLOAT:
        return math.exp(log_value)
    decimal_value = _log_value_to_decimal(log_value)
    if decimal_value is None:
        return math.nan
    return _decimal_to_string(decimal_value)


def _format_exp(value: float) -> float | str:
    if not math.isfinite(value):
        return math.nan
    if MIN_LOG_FLOAT <= value <= MAX_LOG_FLOAT:
        return math.exp(value)
    decimal_value = _exp_decimal(value)
    if decimal_value is None:
        return math.nan
    return _decimal_to_string(decimal_value)


def _get_term_index(result, term: str) -> int | None:
    params = getattr(result, "params", None)
    if isinstance(params, pd.Series):
        try:
            return int(params.index.get_loc(term))
        except KeyError:
            pass
    exog_names = getattr(getattr(result, "model", None), "exog_names", None)
    if exog_names and term in exog_names:
        return int(exog_names.index(term))
    return None


def _extract_parameter(result, term: str) -> float:
    params = getattr(result, "params", None)
    if isinstance(params, pd.Series):
        value = params.get(term)
        if value is None:
            return math.nan
        return float(value)
    if isinstance(params, np.ndarray):
        index = _get_term_index(result, term)
        if index is not None and 0 <= index < len(params):
            return float(params[index])
    return math.nan


def _compute_standard_error(result, term: str) -> float:
    se = math.nan
    bse = getattr(result, "bse", None)
    if isinstance(bse, pd.Series):
        candidate = bse.get(term)
        if candidate is not None:
            se = float(candidate)
    elif isinstance(bse, np.ndarray):
        index = _get_term_index(result, term)
        if index is not None and 0 <= index < len(bse):
            se = float(bse[index])
    if math.isfinite(se) and se > 0:
        return se

    try:
        cov = result.cov_params()
    except Exception:
        cov = None
    index = _get_term_index(result, term)
    if cov is not None and index is not None:
        if isinstance(cov, pd.DataFrame):
            if term in cov.index and term in cov.columns:
                variance = float(cov.loc[term, term])
                if variance >= 0:
                    return math.sqrt(variance)
        else:
            matrix = np.asarray(cov, dtype=np.float64)
            if 0 <= index < matrix.shape[0]:
                variance = float(matrix[index, index])
                if variance >= 0:
                    return math.sqrt(variance)

    if index is not None:
        try:
            params = getattr(result, "params", None)
            vector = np.asarray(params, dtype=np.float64)
            hessian = getattr(result.model, "hessian")
            hess_matrix = np.asarray(hessian(vector), dtype=np.float64)
            if hess_matrix.size:
                cov = np.linalg.pinv(-hess_matrix)
                variance = float(cov[index, index])
                if variance >= 0:
                    return math.sqrt(variance)
        except Exception:
            pass

    return math.nan


def _summarise_term(result, term: str) -> dict[str, float | str | None]:
    beta = _extract_parameter(result, term)
    se = _compute_standard_error(result, term)
    log_p: float | None = None
    z_value = float("nan")
    if math.isfinite(beta) and math.isfinite(se) and se > 0:
        z_value = beta / se
        if math.isfinite(z_value):
            log_sf = float(log_ndtr(-abs(z_value)))
            log_p = math.log(2.0) + log_sf

    summary: dict[str, float | str | None] = {
        "beta": beta,
        "se": se,
        "p_value": _format_probability_from_log(log_p),
        "odds_ratio": _format_exp(beta) if math.isfinite(beta) else math.nan,
        "ci_lower": math.nan,
        "ci_upper": math.nan,
        "log_p": log_p,
    }

    if math.isfinite(beta) and math.isfinite(se) and se > 0:
        half_width = CONFIDENCE_Z * se
        summary["ci_lower"] = _format_exp(beta - half_width)
        summary["ci_upper"] = _format_exp(beta + half_width)

    return summary


def _empty_term_summary() -> dict[str, float | str | None]:
    return {
        "beta": math.nan,
        "se": math.nan,
        "p_value": math.nan,
        "odds_ratio": math.nan,
        "ci_lower": math.nan,
        "ci_upper": math.nan,
        "log_p": None,
    }

    preview_columns = []
    for col in X.columns:
        preview_columns.append(f"{col} -> {_summarise_column(X[col])}")
    lines.extend(preview_columns)
    return "\n".join(lines)


def _remove_collinear_columns(
    df: pd.DataFrame,
    *,
    tolerance: float = 1e-8,
    protected: Sequence[str] = ("dosage",),
) -> tuple[pd.DataFrame, list[str]]:
    if df.shape[1] <= 1:
        return df, []

    values = df.to_numpy(dtype=np.float64, copy=False)
    columns = list(df.columns)
    removed: list[str] = []

    for idx, column in enumerate(columns):
        if column in removed or column == "const":
            continue

        others = [i for i, col in enumerate(columns) if col != column and col not in removed]
        if not others:
            continue

        predictors = values[:, others]
        target = values[:, idx]
        if predictors.size == 0:
            continue

        try:
            solution, *_ = np.linalg.lstsq(predictors, target, rcond=None)
        except np.linalg.LinAlgError:
            continue

        approximation = predictors @ solution
        target_norm = np.linalg.norm(target)
        residual_norm = np.linalg.norm(target - approximation)
        relative_error = residual_norm / max(target_norm, 1.0)

        if relative_error <= tolerance:
            if column in protected:
                raise RuntimeError(
                    f"{column!r} covariate is a linear combination of other predictors after alignment."
                )
            removed.append(column)

    if removed:
        df = df.drop(columns=removed)

    return df, removed


def _format_logistic_failure(exc: Exception, X: pd.DataFrame, y: pd.Series) -> str:
    summary_lines = _summarise_design_matrix(X, y).splitlines()
    indented = "\n    ".join(summary_lines)
    return f"Logistic regression failed: {exc}\n    {indented}"

def _normalize_label(value: str) -> str:
    return " ".join(str(value).lower().replace("_", " ").split())


def _matches_pattern(value: str, pattern: PhenotypePattern) -> bool:
    candidate = _normalize_label(value)
    matcher = _normalize_label(pattern.pattern)
    return fnmatch(candidate, matcher)


def _resolve_target_runs(definitions: pd.DataFrame) -> list["PhenotypeRun"]:
    matches: dict[str, PhenotypePattern] = {}
    matched_patterns: set[str] = set()
    for _, row in definitions.iterrows():
        sanitized = str(row.get("sanitized_name", ""))
        disease = str(row.get("disease", ""))
        for pattern in PHENOTYPE_PATTERNS:
            if not pattern.pattern:
                continue
            if _matches_pattern(sanitized, pattern) or _matches_pattern(disease, pattern):
                existing = matches.get(sanitized)
                if existing and existing.category != pattern.category:
                    raise RuntimeError(
                        "Conflicting category assignments for phenotype "
                        f"'{sanitized}': {existing.category} vs {pattern.category}."
                    )
                matches[sanitized] = pattern
                matched_patterns.add(pattern.pattern)
    if not matches:
        warn("No phenotypes matched the configured patterns.")
        return []

    for pattern in PHENOTYPE_PATTERNS:
        if pattern.pattern not in matched_patterns:
            warn(f"Pattern '{pattern.pattern}' did not match any phenotypes.")

    runs = [PhenotypeRun(phenotype=name, category=pattern.category) for name, pattern in matches.items()]
    runs.sort(key=lambda run: (run.category, run.phenotype))
    return runs


def _load_scores_table() -> tuple[pd.DataFrame, Path]:
    resolved = SCORES_FILE
    if not resolved.exists():
        resolved = Path(pipeline_find_upwards(str(SCORES_FILE)))
    if not resolved.exists():
        raise FileNotFoundError(f"Scores file not found: {SCORES_FILE}")
    info(f"Loading shared PGS controls from {resolved}")
    df = pd.read_csv(resolved, sep="\t")
    if df.empty:
        raise RuntimeError(f"Scores file '{resolved}' is empty.")

    person_col = df.columns[0]
    df = df.rename(columns={person_col: "person_id"})
    df["person_id"] = df["person_id"].astype(str)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if not df["person_id"].is_unique:
        dupes = df[df["person_id"].duplicated()]["person_id"].unique()
        warn(
            "Scores file contains duplicate person IDs; keeping first occurrence "
            f"for {len(dupes):,} duplicates."
        )
        df = df.drop_duplicates(subset="person_id", keep="first")

    df = df.dropna(subset=["person_id"])
    df = df.set_index("person_id")
    return df, resolved


def _build_category_controls(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    controls: dict[str, pd.DataFrame] = {}
    for category, pgs_ids in CATEGORY_PGS_IDS.items():
        columns: list[str] = []
        for pgs_id in pgs_ids:
            target = f"{pgs_id}_AVG"
            match = next((c for c in scores.columns if c.lower() == target.lower()), None)
            if match is None:
                raise KeyError(
                    f"Could not locate column '{target}' for category '{category}' in scores file."
                )
            columns.append(match)
        controls[category] = scores[columns].copy()
    return controls


def _fit_logistic(X: pd.DataFrame, y: pd.Series):
    model = sm.Logit(y, X)
    try:
        result = model.fit(disp=False, maxiter=100)
        return result, "logit_mle"
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError) as initial_error:
        warn(
            "Logit MLE failed – retrying with L2-regularised fit. "
            f"(reason: {initial_error})"
        )
        try:
            # ``fit_regularized`` only accepts L1-oriented solvers.  Setting
            # ``L1_wt`` to 0 switches the penalty to pure L2 while keeping the
            # solver happy.  ``method="l2"`` previously used here triggers a
            # ``ValueError`` because statsmodels does not implement such an
            # option.
            result = model.fit_regularized(alpha=1e-4, L1_wt=0.0, maxiter=200)
            return result, "logit_l2"
        except Exception as exc:
            message = _format_logistic_failure(exc, X, y)
            message += f"\n    Initial MLE failure: {initial_error}"
            raise RuntimeError(message) from exc
    except Exception as exc:
        raise RuntimeError(_format_logistic_failure(exc, X, y)) from exc


@dataclass
class PhenotypeRun:
    phenotype: str
    category: str


def _load_shared_covariates() -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    project_id = os.getenv("GOOGLE_PROJECT")
    cdr_id = os.getenv("WORKSPACE_CDR")
    if not project_id or not cdr_id:
        die(
            "Both GOOGLE_PROJECT and WORKSPACE_CDR environment variables must be set"
        )

    from google.cloud import bigquery

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)

    client = bigquery.Client(project=project_id)
    cdr_codename = cdr_id.split(".")[-1]

    demographics_cache = CACHE_DIR / f"demographics_{cdr_codename}.parquet"
    demographics_df = io.get_cached_or_generate(
        str(demographics_cache),
        io.load_demographics_with_stable_age,
        bq_client=client,
        cdr_id=cdr_id,
        lock_dir=str(LOCK_DIR),
    )

    pcs_cache = CACHE_DIR / f"pcs_{NUM_PCS}_{io.stable_hash((project_id, PCS_URI, NUM_PCS))}.parquet"
    pcs_df = io.get_cached_or_generate(
        str(pcs_cache),
        io.load_pcs,
        project_id,
        PCS_URI,
        NUM_PCS,
        validate_num_pcs=NUM_PCS,
        lock_dir=str(LOCK_DIR),
    )

    sex_cache = CACHE_DIR / f"genetic_sex_{io.stable_hash((project_id, SEX_URI))}.parquet"
    sex_df = io.get_cached_or_generate(
        str(sex_cache),
        io.load_genetic_sex,
        project_id,
        SEX_URI,
        lock_dir=str(LOCK_DIR),
    )

    related_ids = io.load_related_to_remove(project_id, RELATEDNESS_URI)

    for df in (demographics_df, pcs_df, sex_df):
        df.index = df.index.astype(str)

    shared = demographics_df.join(pcs_df, how="inner").join(sex_df, how="inner")
    shared = shared[~shared.index.isin(related_ids)]
    if not shared.index.is_unique:
        dupes = shared.index[shared.index.duplicated()].unique()
        raise RuntimeError(
            "Duplicate person_id entries detected in shared covariates: "
            + ", ".join(map(str, dupes[:5]))
        )

    ancestry_cache = CACHE_DIR / f"ancestry_labels_{io.stable_hash((project_id, PCS_URI))}.parquet"
    ancestry_df = io.get_cached_or_generate(
        str(ancestry_cache),
        io.load_ancestry_labels,
        project_id,
        LABELS_URI=PCS_URI,
        lock_dir=str(LOCK_DIR),
    )
    anc_series = ancestry_df.reindex(shared.index)["ANCESTRY"].astype(str)
    anc_cat = pd.Categorical(anc_series)
    anc_dummies = pd.get_dummies(
        anc_cat,
        prefix="ANC",
        drop_first=True,
        dtype=np.float32,
    )
    # ``pd.get_dummies`` uses a fresh ``RangeIndex`` by default, which discards the
    # participant identifiers associated with ``anc_series``.  The downstream design
    # matrix logic relies on aligning ancestry dummies by ``person_id``; without
    # restoring the original index, the subsequent ``reindex`` call treats every row
    # as missing and fills the ancestry covariates with zeros.  This manifested as the
    # pipeline claiming that all participants belonged to the reference ancestry
    # stratum, even when other ancestries were present.  Reapply the ``shared`` index
    # so that ancestry indicators stay aligned with the rest of the covariates.
    anc_dummies.index = anc_series.index.astype(str)

    reference_ancestry: str | None
    if anc_cat.categories.size:
        reference_ancestry = str(anc_cat.categories[0])
    else:
        reference_ancestry = None

    return shared, anc_dummies, reference_ancestry


def _load_inversion(target: str) -> pd.DataFrame:
    dosages_path = pipeline_find_upwards(INVERSION_DOSAGES_FILE)
    inversion_df = io.get_cached_or_generate(
        str(CACHE_DIR / f"inversion_{io.stable_hash(target)}.parquet"),
        io.load_inversions,
        target,
        dosages_path,
        validate_target=target,
        lock_dir=str(LOCK_DIR),
    )
    inversion_df.index = inversion_df.index.astype(str)
    return inversion_df[[target]].rename(columns={target: "dosage"})


def _build_design_matrix(
    core: pd.DataFrame,
    ancestry_dummies: pd.DataFrame,
    custom_controls: pd.DataFrame,
    reference_ancestry: str | None,
) -> pd.DataFrame:
    df = core.copy()
    age_mean = df["AGE"].mean()
    df["AGE_c"] = df["AGE"] - age_mean
    df["AGE_c_sq"] = df["AGE_c"] ** 2
    base_covars = _base_covariate_columns()
    covar_cols: list[str] = ["dosage", *base_covars]
    missing = [c for c in covar_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required covariate columns: {missing}")

    design = df[covar_cols].astype(np.float32, copy=False)
    design["const"] = np.float32(1.0)

    anc_slice = ancestry_dummies.reindex(design.index).fillna(0.0).astype(np.float32)
    out = pd.concat([design, anc_slice], axis=1)

    diagnostics: dict[str, object] = {
        "dropped_non_finite": 0,
        "dropped_constant": [],
        "dropped_duplicates": [],
        "dropped_collinear": [],
        "ancestry_constants": [],
    }

    if not custom_controls.empty:
        aligned_custom = custom_controls.reindex(out.index)
        aligned_custom = aligned_custom.dropna(axis=0, how="any")
        out = out.loc[aligned_custom.index]
        out = pd.concat([out, aligned_custom.astype(np.float32)], axis=1)

    finite_mask = np.isfinite(out.to_numpy(dtype=np.float64)).all(axis=1)
    if not finite_mask.all():
        before = len(out)
        out = out.loc[finite_mask]
        dropped = before - len(out)
        diagnostics["dropped_non_finite"] = dropped
        warn(f"Dropped {dropped:,} participants with non-finite covariates.")

    # Statsmodels fails with a ``Singular matrix`` error when any covariate column is
    # constant (all zeros/ones) after aligning the custom controls and ancestry
    # dummies.  This situation arose in production when every participant belonged to
    # the same ancestry stratum, leaving the corresponding dummy columns as all
    # zeros.  The original code surfaced as a logistic regression failure for every
    # affected phenotype, obscuring the underlying data issue.  We proactively drop
    # non-informative constant columns so that the model has full rank.  If the
    # dosage column itself is constant then the analysis cannot produce a meaningful
    # estimate, so we abort early to avoid misleading output.
    constant_counts = out.nunique(dropna=False)
    constant_cols = [col for col, count in constant_counts.items() if count <= 1]

    if "dosage" in constant_cols:
        raise RuntimeError(
            "Inversion dosage is constant after filtering; cannot fit logistic model."
        )

    ancestry_columns = [col for col in out.columns if col.startswith("ANC_")]
    ancestry_constants = [col for col in ancestry_columns if col in constant_cols]

    if ancestry_columns and ancestry_constants:
        ancestry_details: list[str] = []
        if all(out[col].eq(0.0).all() for col in ancestry_columns):
            if reference_ancestry:
                warn(
                    "All participants fall into the reference ancestry stratum "
                    f"('{reference_ancestry}') after filtering; ancestry covariates will be dropped."
                )
            else:
                warn(
                    "All participants fall into a single ancestry stratum after filtering; ancestry covariates will be dropped."
                )
            ancestry_details.append("all ancestry dummies = 0")
        else:
            details: list[str] = []
            for col in ancestry_constants:
                label = col.split("ANC_", 1)[-1]
                value = float(out[col].iloc[0]) if len(out[col]) else float("nan")
                if value == 0.0:
                    details.append(f"{label}=0 (no participants)")
                elif value == 1.0:
                    details.append(f"{label}=1 (all participants)")
                else:
                    details.append(f"{label} constant at {value}")
                ancestry_details.append(f"{col} -> {details[-1]}")
            warn(
                "One or more ancestry dummy covariates became constant after aligning "
                "with the phenotype/custom controls: "
                + "; ".join(details)
            )
        diagnostics["ancestry_constants"] = ancestry_details or [col for col in ancestry_constants]

    removable = [col for col in constant_cols if col != "const"]
    if removable:
        out = out.drop(columns=removable)
        diagnostics["dropped_constant"] = sorted(removable)
        warn(
            "Dropped constant covariate columns to avoid singular design: "
            + ", ".join(sorted(removable))
        )

    signature_owner: dict[bytes, str] = {}
    duplicate_cols: list[str] = []
    duplicate_descriptions: list[str] = []
    for column in out.columns:
        signature = out[column].to_numpy(dtype=np.float64, copy=False).tobytes()
        owner = signature_owner.get(signature)
        if owner is None:
            signature_owner[signature] = column
            continue
        duplicate_cols.append(column)
        duplicate_descriptions.append(f"{column} == {owner}")

    if duplicate_cols:
        if "dosage" in duplicate_cols:
            raise RuntimeError(
                "Dosage covariate duplicates another predictor after alignment; cannot fit logistic model."
            )
        out = out.drop(columns=duplicate_cols)
        diagnostics["dropped_duplicates"] = sorted(duplicate_cols)
        warn(
            "Dropped duplicate covariate columns to avoid singular design: "
            + ", ".join(duplicate_descriptions)
        )

    try:
        out, removed_collinear = _remove_collinear_columns(out)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    if removed_collinear:
        diagnostics["dropped_collinear"] = sorted(removed_collinear)
        warn(
            "Dropped linearly dependent covariates detected via least-squares check: "
            + ", ".join(sorted(removed_collinear))
        )

    final_matrix = out.to_numpy(dtype=np.float64, copy=False)
    diagnostics["matrix_rank"] = _estimate_rank(final_matrix)
    diagnostics["condition_number"] = _compute_condition_number(final_matrix)
    diagnostics["final_columns"] = list(out.columns)
    out.attrs["diagnostics"] = diagnostics

    return out


def _ensure_pheno_cache(
    definitions: pd.DataFrame,
    phenotype: str,
    cdr_id: str,
    core_index: pd.Index,
    project_id: str,
) -> None:
    match = definitions.loc[definitions["sanitized_name"] == phenotype]
    if match.empty:
        raise KeyError(f"Phenotype '{phenotype}' not present in definitions table.")
    if len(match) > 1:
        raise RuntimeError(f"Multiple definition rows found for phenotype '{phenotype}'.")

    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    cdr_codename = cdr_id.split(".")[-1]
    row = match.iloc[0].to_dict()
    row.update({"cdr_codename": cdr_codename, "cache_dir": str(CACHE_DIR)})

    pheno_path = CACHE_DIR / f"pheno_{phenotype}_{cdr_codename}.parquet"
    if not pheno_path.exists():
        info(f"Caching phenotype '{phenotype}' via BigQuery fetch…")
        pheno._query_single_pheno_bq(
            row,
            cdr_id,
            core_index,
            str(CACHE_DIR),
            cdr_codename,
            bq_client=client,
            non_blocking=False,
        )
    return None


def _load_case_status(
    phenotype: str,
    cdr_codename: str,
    participants: Iterable[str],
) -> pd.Series:
    case_ids = io.load_pheno_cases_from_cache(phenotype, str(CACHE_DIR), cdr_codename)
    case_set = set(case_ids)
    data = [1 if pid in case_set else 0 for pid in participants]
    series = pd.Series(data, index=pd.Index(participants, name="person_id"), dtype=np.int8)
    return series


def run() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    project_id = os.getenv("GOOGLE_PROJECT")
    cdr_id = os.getenv("WORKSPACE_CDR")
    if not project_id or not cdr_id:
        die("GOOGLE_PROJECT and WORKSPACE_CDR must be defined in the environment.")

    shared_covariates, anc_dummies, reference_ancestry = _load_shared_covariates()
    definitions = pheno.load_definitions(PHENOTYPE_DEFINITIONS_URL)
    target_runs = _resolve_target_runs(definitions)
    if not target_runs:
        warn("No phenotypes selected for analysis; exiting.")
        return

    scores_table, scores_path = _load_scores_table()
    category_controls = _build_category_controls(scores_table)

    results = []
    cdr_codename = cdr_id.split(".")[-1]

    for inv in TARGET_INVERSIONS:
        info(f"Preparing inversion {inv}")
        inversion_df = _load_inversion(inv)
        core = shared_covariates.join(inversion_df, how="inner")
        core = core.rename(columns={inv: "dosage"}) if inv in core.columns else core

        if "dosage" not in core.columns:
            raise KeyError(f"Inversion column '{inv}' missing after join")

        for cfg in target_runs:
            info(f"Running phenotype '{cfg.phenotype}' (category: {cfg.category})")
            custom_controls = category_controls.get(cfg.category)
            if custom_controls is None:
                warn(
                    f"No custom controls found for category '{cfg.category}'; skipping {cfg.phenotype}."
                )
                continue

            _ensure_pheno_cache(
                definitions,
                cfg.phenotype,
                cdr_id,
                core.index.astype(str),
                project_id,
            )

            try:
                design = _build_design_matrix(
                    core,
                    anc_dummies,
                    custom_controls,
                    reference_ancestry,
                )
            except RuntimeError as exc:
                warn(f"Skipping {cfg.phenotype}: {exc}")
                continue
            phenotype_status = _load_case_status(
                cfg.phenotype,
                cdr_codename,
                design.index,
            )

            n_cases = int(phenotype_status.sum())
            n_total = int(len(phenotype_status))
            n_ctrls = n_total - n_cases

            if n_cases == 0 or n_ctrls == 0:
                warn(
                    f"Skipping {cfg.phenotype}: insufficient cases ({n_cases}) or controls ({n_ctrls})."
                )
                continue

            y = phenotype_status.loc[design.index].astype(np.int8)
            X = design.copy()

            intercept = "const" in X.columns
            ordered_terms = [col for col in X.columns if col != "const"]
            if "dosage" in ordered_terms:
                ordered_terms.insert(0, ordered_terms.pop(ordered_terms.index("dosage")))
            formula_terms: list[str] = []
            if intercept:
                formula_terms.append("1")
            formula_terms.extend(ordered_terms)
            formula_repr = " + ".join(formula_terms) if formula_terms else "<empty design>"
            info(
                "Model specification: logit(P("
                f"{cfg.phenotype}=1)) ~ {formula_repr}"
            )

            try:
                fit, fit_method = _fit_logistic(X, y)
            except RuntimeError as exc:
                warn(f"Logistic regression failed for {cfg.phenotype}: {exc}")
                continue

            term_summary = _summarise_term(fit, "dosage")
            beta = float(term_summary.get("beta", math.nan))
            if math.isnan(beta):
                warn(f"Model for {cfg.phenotype} did not estimate a dosage coefficient.")
                continue
            se = float(term_summary.get("se", math.nan))
            p_value = term_summary.get("p_value", math.nan)
            odds_ratio = term_summary.get("odds_ratio", math.nan)
            ci_lower = term_summary.get("ci_lower", math.nan)
            ci_upper = term_summary.get("ci_upper", math.nan)
            log_p_value = term_summary.get("log_p")
            if log_p_value is None or not math.isfinite(log_p_value):
                log_p_value = math.nan

            custom_cols = [c for c in X.columns if c.upper().startswith(CUSTOM_CONTROL_PREFIX.upper())]

            baseline_summary = _empty_term_summary()
            baseline_method: str | None = None
            baseline_design = X.drop(columns=custom_cols, errors="ignore")
            try:
                baseline_fit, baseline_method = _fit_logistic(baseline_design, y.loc[baseline_design.index])
                baseline_summary = _summarise_term(baseline_fit, "dosage")
            except RuntimeError as exc:
                warn(
                    "Unadjusted logistic regression failed for "
                    f"{cfg.phenotype}: {exc}"
                )

            baseline_log_p = baseline_summary.get("log_p")
            if baseline_log_p is None or not math.isfinite(baseline_log_p):
                baseline_log_p = math.nan

            results.append(
                {
                    "Phenotype": cfg.phenotype,
                    "Category": cfg.category,
                    "Inversion": inv,
                    "N_Total": n_total,
                    "N_Cases": n_cases,
                    "N_Controls": n_ctrls,
                    "Beta": beta,
                    "SE": se,
                    "OR": odds_ratio,
                    "OR_95CI_Lower": ci_lower,
                    "OR_95CI_Upper": ci_upper,
                    "P_Value": p_value,
                    "Log_P_Value": log_p_value,
                    "Fit_Method": fit_method,
                    "Control_File": str(scores_path),
                    "Custom_Covariates": ",".join(custom_cols),
                    "OR_Unadjusted": baseline_summary.get("odds_ratio", math.nan),
                    "OR_Unadjusted_95CI_Lower": baseline_summary.get("ci_lower", math.nan),
                    "OR_Unadjusted_95CI_Upper": baseline_summary.get("ci_upper", math.nan),
                    "P_Value_Unadjusted": baseline_summary.get("p_value", math.nan),
                    "Log_P_Value_Unadjusted": baseline_log_p,
                    "Fit_Method_Unadjusted": baseline_method,
                }
            )

    if not results:
        warn("No successful analyses were produced; skipping output file.")
        return

    output_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, sep="\t", index=False)
    info(f"Wrote {len(output_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":  # pragma: no cover - manual execution only
    run()
