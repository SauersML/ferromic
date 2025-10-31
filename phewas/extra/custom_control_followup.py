"""Custom follow-up runner with per-phenotype polygenic score controls.

This module mirrors the shared setup performed by :mod:`phewas.run` but trims it
down to a minimal, targeted workflow:

* Only phenotypes whose names match ``PHENOTYPE_PATTERNS`` are analysed.
* Only the inversions listed in ``TARGET_INVERSIONS`` are considered.
* Each phenotype category draws two additional polygenic score controls from
  ``scores.tsv``. These are selected according to ``CATEGORY_PGS_IDS`` and use
  ``<PGS_ID>_AVG`` columns matched case-insensitively.
* P-values are adjusted for multiple testing via the Benjamini–Hochberg FDR procedure.

All configuration is expressed as module-level globals; invoke :func:`run` via
``python -m phewas.extra.custom_control_followup`` for an in-repo CLI entry
point.

The output table ``custom_control_follow_ups.tsv`` is written to the current
working directory.
"""

from __future__ import annotations

import os
import re

# ---------------------------------------------------------------------------
# Cap native BLAS/OpenMP threads before importing NumPy/SciPy.
# ---------------------------------------------------------------------------

_THREAD_ENV_DEFAULTS = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}

for _env_var, _default in _THREAD_ENV_DEFAULTS.items():
    if not os.environ.get(_env_var):
        os.environ[_env_var] = _default

import sys
import math
import warnings
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
import multiprocessing
from dataclasses import dataclass
from decimal import Context, Decimal, localcontext
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import log_ndtr
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError

from google.cloud import bigquery

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

CUSTOM_PHENOTYPE_BLACKLIST: frozenset[str] = frozenset(
    {
        "Obesity_hypoventilation_syndrome_OHS",
    }
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

LRT_BOOTSTRAP_REPLICATES = 200
LRT_BOOTSTRAP_REDUCED_REPLICATES = 50
LRT_BOOTSTRAP_REDUCED_THRESHOLD = 50_000
LRT_BOOTSTRAP_DISABLE_THRESHOLD = 10
LRT_BOOTSTRAP_MAX_ABS_ETA_DISABLE = 12.0
LRT_BOOTSTRAP_SEED = 13579

RIDGE_PARAMETRIC_BOOTSTRAP_REPLICATES = 100
RIDGE_PARAMETRIC_BOOTSTRAP_REDUCED_REPLICATES = 25
RIDGE_PARAMETRIC_BOOTSTRAP_REDUCED_THRESHOLD = 50_000
RIDGE_PARAMETRIC_BOOTSTRAP_DISABLE_THRESHOLD = 100_000
RIDGE_PARAMETRIC_BOOTSTRAP_SEED = 424242

# Logistic models occasionally reach extremely large coefficient estimates when the
# outcome is nearly separated.  ``MAX_ABS_DOSAGE_BETA`` bounds what we consider a
# numerically stable dosage coefficient; larger values trigger detailed
# diagnostics so that the root cause can be investigated instead of applying
# ever-stronger penalties.
MAX_ABS_DOSAGE_BETA = 15.0
RIDGE_ALPHA = 1e-6
GRADIENT_NORM_THRESHOLD = 1e-6
HESSIAN_CONDITION_THRESHOLD = 1e8


def _resolve_lrt_bootstrap_replicates(n_obs: int, max_abs_eta: float | None = None) -> int:
    replicates = LRT_BOOTSTRAP_REPLICATES
    if n_obs >= LRT_BOOTSTRAP_DISABLE_THRESHOLD:
        return 0
    if max_abs_eta is not None and math.isfinite(max_abs_eta):
        if max_abs_eta >= LRT_BOOTSTRAP_MAX_ABS_ETA_DISABLE:
            return 0
    if n_obs >= LRT_BOOTSTRAP_REDUCED_THRESHOLD:
        replicates = min(replicates, LRT_BOOTSTRAP_REDUCED_REPLICATES)
    return replicates


def _resolve_penalized_bootstrap_replicates(n_obs: int) -> int:
    replicates = RIDGE_PARAMETRIC_BOOTSTRAP_REPLICATES
    if n_obs >= RIDGE_PARAMETRIC_BOOTSTRAP_DISABLE_THRESHOLD:
        return 0
    if n_obs >= RIDGE_PARAMETRIC_BOOTSTRAP_REDUCED_THRESHOLD:
        replicates = min(replicates, RIDGE_PARAMETRIC_BOOTSTRAP_REDUCED_REPLICATES)
    return replicates

# ---------------------------------------------------------------------------
# Data source configuration (mirrors ``phewas.run`` defaults)
# ---------------------------------------------------------------------------

CACHE_DIR = Path(PIPELINE_CACHE_DIR)
LOCK_DIR = Path(PIPELINE_LOCK_DIR)
PCS_URI = PIPELINE_PCS_URI
SEX_URI = PIPELINE_SEX_URI
RELATEDNESS_URI = PIPELINE_RELATEDNESS_URI

# Directory for phenotype-specific logs relative to the working directory.
LOG_OUTPUT_DIR = Path("logs")
_LOG_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_CLEARED_LOG_PATHS: set[Path] = set()

# ---------------------------------------------------------------------------
# Simple logging helpers
# ---------------------------------------------------------------------------


def _phenotype_log_path(prefix: str) -> Path:
    LOG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sanitized = _LOG_SANITIZE_RE.sub("_", prefix)
    sanitized = sanitized.strip("._-") or "phenotype"
    if len(sanitized) > 128:
        sanitized = sanitized[:128]
    return LOG_OUTPUT_DIR / f"{sanitized}.log"


def _initialise_pheno_log(prefix: str) -> None:
    path = _phenotype_log_path(prefix)
    if path in _CLEARED_LOG_PATHS:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    _CLEARED_LOG_PATHS.add(path)


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
    try:
        with np.errstate(all="ignore"):
            value = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return float("inf")
    if not np.isfinite(value):
        return float("inf")
    return value


def _estimate_rank(matrix: np.ndarray) -> int:
    if matrix.size == 0:
        return 0
    try:
        with np.errstate(all="ignore"):
            return int(np.linalg.matrix_rank(matrix))
    except np.linalg.LinAlgError:
        return 0


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
        dropped_missing_ancestry = diagnostics.get("dropped_missing_ancestry", 0)
        if dropped_missing_ancestry:
            lines.append(
                "Rows dropped for missing ancestry covariates: "
                f"{dropped_missing_ancestry}"
            )
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
        details: Mapping[str, dict[str, object]] | None = diagnostics.get("non_finite_details")  # type: ignore[assignment]
        if details:
            for column, info_dict in details.items():
                pieces: list[str] = []
                nan_count = info_dict.get("nan")
                if nan_count:
                    pieces.append(f"NaN={nan_count}")
                pos_inf = info_dict.get("pos_inf")
                if pos_inf:
                    pieces.append(f"+inf={pos_inf}")
                neg_inf = info_dict.get("neg_inf")
                if neg_inf:
                    pieces.append(f"-inf={neg_inf}")
                examples = info_dict.get("examples")
                if examples:
                    pieces.append("examples=" + ", ".join(map(str, examples)))
                if pieces:
                    lines.append(f"Non-finite values in {column}: " + "; ".join(pieces))
    return "\n".join(lines)


def _format_non_finite_value(value: float) -> str:
    if pd.isna(value):
        return "NaN"
    if np.isposinf(value):
        return "+inf"
    if np.isneginf(value):
        return "-inf"
    return f"{float(value):.6g}"


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


def _extract_series_value(result, attribute: str, term: str) -> float:
    values = getattr(result, attribute, None)
    if isinstance(values, pd.Series):
        candidate = values.get(term)
        if candidate is None:
            return math.nan
        return float(candidate)
    if isinstance(values, np.ndarray):
        index = _get_term_index(result, term)
        if index is not None and 0 <= index < len(values):
            return float(values[index])
    return math.nan
def _logit_result_converged(result) -> bool:
    converged = getattr(result, "converged", None)
    if converged is None:
        mle_retvals = getattr(result, "mle_retvals", {})
        if isinstance(mle_retvals, dict):
            converged = mle_retvals.get("converged")
    if converged is None:
        return True
    return bool(converged)


def _evaluate_result_stability(
    result, term: str = "dosage"
) -> tuple[bool, list[str], dict[str, object]]:
    issues: list[str] = []
    metrics: dict[str, object] = {}

    if not _logit_result_converged(result):
        issues.append("optimizer reported non-convergence")

    beta = _extract_parameter(result, term)
    metrics["beta"] = beta
    if not math.isfinite(beta):
        issues.append("dosage beta is not finite")
    elif abs(beta) > MAX_ABS_DOSAGE_BETA:
        issues.append(
            f"dosage beta magnitude {beta:.6g} exceeds stability threshold {MAX_ABS_DOSAGE_BETA}"
        )

    se = _compute_standard_error(result, term)
    metrics["se"] = se
    if not math.isfinite(se) or se <= 0:
        issues.append("dosage standard error is not positive and finite")

    p_value = _extract_series_value(result, "pvalues", term)
    if math.isfinite(p_value) and p_value > 0:
        metrics["p_value"] = float(p_value)

    llf = getattr(result, "llf", None)
    if llf is not None and math.isfinite(llf):
        metrics["loglike"] = float(llf)

    nobs = getattr(result, "nobs", None)
    if nobs is not None and math.isfinite(nobs):
        metrics["nobs"] = float(nobs)

    retvals = getattr(result, "mle_retvals", {})
    if isinstance(retvals, dict):
        iterations = retvals.get("iterations")
        if iterations is not None:
            metrics["iterations"] = float(iterations)

        warnflag_issue = _interpret_warnflag(retvals.get("warnflag"))
        if warnflag_issue:
            issues.append(warnflag_issue)
            warnflag_value = retvals.get("warnflag")
            if warnflag_value is not None:
                try:
                    metrics["warnflag"] = float(warnflag_value)
                except (TypeError, ValueError):
                    metrics["warnflag"] = warnflag_value

        for key in ("score_norm", "grad", "score"):
            if key in retvals and f"{key}_norm" not in metrics:
                norm_value = _safe_vector_norm(retvals.get(key))
                if norm_value is not None:
                    metrics[f"{key}_norm"] = norm_value
                    if norm_value > 1e-4:
                        issues.append(f"{key} norm {norm_value:.6g} remains above tolerance 1e-4")

        determinant = retvals.get("determinant")
        if determinant is not None:
            try:
                determinant_value = float(determinant)
            except (TypeError, ValueError):
                determinant_value = math.nan
            if math.isfinite(determinant_value):
                metrics["determinant"] = determinant_value
                if abs(determinant_value) < 1e-12:
                    issues.append("observed information matrix determinant near zero")

    try:
        params = np.asarray(getattr(result, "params", []), dtype=np.float64)
        if params.size:
            gradient = result.model.score(params)
        else:
            gradient = None
    except Exception:
        gradient = None

    gradient_norm = _safe_vector_norm(gradient)
    if gradient_norm is not None:
        metrics.setdefault("gradient_norm", gradient_norm)
        if gradient_norm > 1e-4:
            issues.append(
                f"gradient norm {gradient_norm:.6g} indicates estimates may not be at optimum"
            )

    try:
        params = np.asarray(getattr(result, "params", []), dtype=np.float64)
        if params.size:
            hessian = np.asarray(result.model.hessian(params), dtype=np.float64)
        else:
            hessian = np.array([])
    except Exception:
        hessian = np.array([])

    if hessian.size:
        cond = _compute_condition_number(hessian)
        if math.isfinite(cond):
            metrics["hessian_cond"] = cond
            if cond > 1e12:
                issues.append(
                    f"model Hessian condition number {cond:.6g} suggests severe ill-conditioning"
                )

    return not issues, issues, metrics


def _logit_result_is_stable(result, term: str = "dosage") -> bool:
    stable, _, _ = _evaluate_result_stability(result, term)
    return stable


def _format_warning_records(records: Sequence[warnings.WarningMessage]) -> list[str]:
    messages: list[str] = []
    for record in records:
        category = getattr(record.category, "__name__", str(record.category))
        messages.append(f"{category}: {record.message}")
    return messages


def _safe_vector_norm(value: object) -> float | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if array.size == 0:
        return 0.0
    if not np.isfinite(array).all():
        return None
    return float(np.linalg.norm(array))


def _interpret_warnflag(flag: int | float | None) -> str | None:
    if flag in (None, 0):
        return None
    try:
        code = int(flag)
    except (TypeError, ValueError):
        return f"optimizer warnflag={flag}"

    mapping = {
        1: "iteration limit reached before convergence",
        2: "optimizer detected approximate singularity in the Hessian",
        3: "line search failed to improve objective",
    }
    message = mapping.get(code)
    if message is None:
        return f"optimizer warnflag={code}"
    return message


def _format_value_counts(series: pd.Series, *, limit: int = 5, digits: int = 6) -> str:
    if series.empty:
        return "<empty>"

    rounded = series.round(digits)
    counts = rounded.value_counts(dropna=False).head(limit)
    total = len(series)
    pieces: list[str] = []
    for value, count in counts.items():
        if pd.isna(value):
            label = "NaN"
        else:
            label = f"{float(value):.6g}"
        proportion = (count / total) * 100 if total else float("nan")
        pieces.append(f"{label} ({count}/{total} = {proportion:.2f}%)")

    if series.nunique(dropna=True) > limit:
        pieces.append("…")

    return "; ".join(pieces)


def _detect_dosage_separation(cases: pd.Series, controls: pd.Series) -> str | None:
    case_vals = cases.replace([np.inf, -np.inf], np.nan).dropna()
    control_vals = controls.replace([np.inf, -np.inf], np.nan).dropna()

    if case_vals.empty or control_vals.empty:
        return None

    case_min = float(case_vals.min())
    case_max = float(case_vals.max())
    control_min = float(control_vals.min())
    control_max = float(control_vals.max())

    if case_min >= control_max:
        return (
            "All case dosages are greater than or equal to the maximum control dosage; "
            "perfect separation detected."
        )
    if case_max <= control_min:
        return (
            "All case dosages are less than or equal to the minimum control dosage; "
            "perfect separation detected."
        )

    case_p05 = float(case_vals.quantile(0.05))
    case_p95 = float(case_vals.quantile(0.95))
    control_p05 = float(control_vals.quantile(0.05))
    control_p95 = float(control_vals.quantile(0.95))

    if case_p05 > control_p95:
        return (
            "95% of control dosages are below the lowest 5% of case dosages; "
            "near-perfect separation suspected."
        )
    if case_p95 < control_p05:
        return (
            "95% of case dosages are below the lowest 5% of control dosages; "
            "near-perfect separation suspected."
        )

    return None


def _describe_dosage_distribution(X: pd.DataFrame, y: pd.Series) -> list[str]:
    if "dosage" not in X.columns:
        return ["Design matrix missing 'dosage' column during instability diagnostics."]

    dosage = pd.to_numeric(X["dosage"], errors="coerce")
    total = int(len(dosage))
    finite_count = int(dosage.replace([np.inf, -np.inf], np.nan).notna().sum())
    lines: list[str] = [
        f"Dosage finite observations after filtering: {finite_count}/{total}; unique={dosage.nunique(dropna=False)}",
        "Dosage summary (all participants): " + _format_summary_dict(_summarise_column(dosage)),
    ]

    y_bool = y.astype(bool)
    case_vals = dosage.loc[y_bool]
    control_vals = dosage.loc[~y_bool]

    if not case_vals.empty:
        lines.append(
            "Dosage summary (cases): " + _format_summary_dict(_summarise_column(case_vals))
        )
    if not control_vals.empty:
        lines.append(
            "Dosage summary (controls): " + _format_summary_dict(_summarise_column(control_vals))
        )

    non_zero_cases = int((case_vals != 0).sum()) if not case_vals.empty else 0
    non_zero_controls = int((control_vals != 0).sum()) if not control_vals.empty else 0
    lines.append(
        "Non-zero dosage counts: "
        f"cases={non_zero_cases}/{len(case_vals)}; controls={non_zero_controls}/{len(control_vals)}"
    )

    separation_message = _detect_dosage_separation(case_vals, control_vals)
    if separation_message:
        lines.append(separation_message)

    if not case_vals.empty:
        lines.append(
            "Most common case dosages (rounded): " + _format_value_counts(case_vals, limit=5)
        )
    if not control_vals.empty:
        lines.append(
            "Most common control dosages (rounded): "
            + _format_value_counts(control_vals, limit=5)
        )

    dosage_vector = dosage.to_numpy(dtype=np.float64, copy=False)
    outcome_vector = y.to_numpy(dtype=np.float64, copy=False)
    if dosage_vector.size and outcome_vector.size and np.std(dosage_vector) > 0:
        corr_matrix = np.corrcoef(dosage_vector, outcome_vector)
        if corr_matrix.size == 4 and np.isfinite(corr_matrix[0, 1]):
            lines.append(
                "Point-biserial correlation between dosage and outcome: "
                f"{float(corr_matrix[0, 1]):.6g}"
            )

    collinear: list[str] = []
    dosage_series = dosage
    for column in X.columns:
        if column in {"dosage", "const"}:
            continue
        other = pd.to_numeric(X[column], errors="coerce")
        if other.nunique(dropna=True) <= 1:
            continue
        corr = dosage_series.corr(other)
        if pd.notna(corr) and abs(corr) > 0.99:
            collinear.append(f"{column} (corr={corr:.3f})")

    if collinear:
        lines.append(
            "Covariates nearly collinear with dosage: " + ", ".join(sorted(collinear))
        )

    return lines


def _describe_covariate_columns(X: pd.DataFrame) -> list[str]:
    total = len(X)
    if total == 0:
        return ["Design matrix is empty; no covariates available for diagnostics."]

    lines: list[str] = []
    for column in X.columns:
        series = pd.to_numeric(X[column], errors="coerce")
        finite_series = series.replace([np.inf, -np.inf], np.nan)
        finite_mask = finite_series.notna()
        finite_count = int(finite_mask.sum())
        finite_pct = (finite_count / total) * 100 if total else float("nan")
        nan_count = int(series.isna().sum())
        pos_inf_count = int(np.isposinf(series.to_numpy(dtype=np.float64, copy=False)).sum())
        neg_inf_count = int(np.isneginf(series.to_numpy(dtype=np.float64, copy=False)).sum())
        zero_count = int((finite_series == 0).sum())
        non_zero_count = int(((finite_series != 0) & finite_mask).sum())
        unique_finite = int(finite_series.nunique(dropna=True))

        summary_stats = _summarise_column(series)
        metrics: dict[str, object] = {
            "finite": f"{finite_count}/{total}",
            "finite_pct": finite_pct,
            "nan": nan_count,
            "+inf": pos_inf_count,
            "-inf": neg_inf_count,
            "zeros": zero_count,
            "non_zero": non_zero_count,
            "unique_finite": unique_finite,
        }
        metrics.update(summary_stats)
        lines.append(f"{column}: " + _format_summary_dict(metrics))

    return lines


def _summarise_term(
    result,
    term: str,
    *,
    design: pd.DataFrame | None = None,
    response: pd.Series | None = None,
) -> dict[str, float | str | None]:
    beta = _extract_parameter(result, term)
    se = _extract_series_value(result, "bse", term)
    if not math.isfinite(se) or se <= 0:
        se = _compute_standard_error(result, term)

    p_value = _extract_series_value(result, "pvalues", term)
    z_value = _extract_series_value(result, "tvalues", term)
    log_p: float | None
    ci_low = math.nan
    ci_high = math.nan

    if math.isfinite(p_value) and p_value > 0:
        log_p = math.log(p_value)
    else:
        log_p = None

    if (log_p is None or not math.isfinite(log_p)) and math.isfinite(z_value):
        log_sf = float(log_ndtr(-abs(z_value)))
        log_p = math.log(2.0) + log_sf
        if log_p > MIN_LOG_FLOAT:
            p_value = math.exp(log_p)
        else:
            p_value = math.nan

    if not math.isfinite(z_value) and math.isfinite(beta) and math.isfinite(se) and se > 0:
        z_value = beta / se
        if math.isfinite(z_value):
            log_sf = float(log_ndtr(-abs(z_value)))
            log_p = math.log(2.0) + log_sf
            if log_p > MIN_LOG_FLOAT:
                p_value = math.exp(log_p)
            else:
                p_value = math.nan

    summary: dict[str, float | str | None] = {
        "beta": beta,
        "se": se,
        "p_value": _format_probability_from_log(log_p),
        "odds_ratio": _format_exp(beta) if math.isfinite(beta) else math.nan,
        "ci_lower": math.nan,
        "ci_upper": math.nan,
        "log_p": log_p if log_p is not None and math.isfinite(log_p) else None,
    }

    bootstrap_meta: dict[str, float | int | None] | None = None
    penalized_replicates: int | None = None
    if _is_penalized_result(result) and design is not None and response is not None:
        penalized_replicates = _resolve_penalized_bootstrap_replicates(len(response))
        if penalized_replicates > 0:
            try:
                bootstrap_meta = _compute_penalized_parametric_bootstrap(
                    result,
                    design,
                    response,
                    term,
                    replicates=penalized_replicates,
                )
            except Exception:
                bootstrap_meta = None
        if bootstrap_meta:
            se = float(bootstrap_meta.get("se", se))
            if math.isfinite(se) and se > 0:
                summary["se"] = se
            p_value_candidate = bootstrap_meta.get("p_value", math.nan)
            log_p_candidate = bootstrap_meta.get("log_p")
            if math.isfinite(p_value_candidate):
                summary["p_value"] = p_value_candidate
            if log_p_candidate is not None and math.isfinite(log_p_candidate):
                summary["log_p"] = log_p_candidate

            ci_low_candidate = bootstrap_meta.get("ci_low", math.nan)
            ci_high_candidate = bootstrap_meta.get("ci_high", math.nan)
            if math.isfinite(ci_low_candidate):
                ci_low = float(ci_low_candidate)
            if math.isfinite(ci_high_candidate):
                ci_high = float(ci_high_candidate)
            summary["penalized_bootstrap_successes"] = bootstrap_meta.get("successes")
            summary["penalized_bootstrap_attempts"] = bootstrap_meta.get("attempts")
            summary["penalized_bootstrap_replicates"] = bootstrap_meta.get("replicates")
        elif penalized_replicates is not None and penalized_replicates <= 0:
            summary["penalized_bootstrap_replicates"] = penalized_replicates

    conf_int = getattr(result, "conf_int", None)
    if callable(conf_int) and not bootstrap_meta:
        try:
            interval = conf_int(alpha=0.05)
        except Exception:
            interval = None
        if isinstance(interval, pd.DataFrame):
            if term in interval.index:
                ci_low = float(interval.loc[term, 0])
                ci_high = float(interval.loc[term, 1])
        elif isinstance(interval, np.ndarray):
            index = _get_term_index(result, term)
            if index is not None and 0 <= index < interval.shape[0]:
                ci_low = float(interval[index, 0])
                ci_high = float(interval[index, 1])

    if not math.isfinite(ci_low) or not math.isfinite(ci_high):
        if math.isfinite(beta) and math.isfinite(se) and se > 0:
            half_width = CONFIDENCE_Z * se
            ci_low = beta - half_width
            ci_high = beta + half_width

    if math.isfinite(ci_low):
        summary["ci_lower"] = _format_exp(ci_low)
    if math.isfinite(ci_high):
        summary["ci_upper"] = _format_exp(ci_high)

    if (
        penalized_replicates is not None
        and "penalized_bootstrap_replicates" not in summary
    ):
        summary["penalized_bootstrap_replicates"] = penalized_replicates

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
        if sanitized in CUSTOM_PHENOTYPE_BLACKLIST:
            continue
        disease = str(row.get("disease", ""))
        if disease in CUSTOM_PHENOTYPE_BLACKLIST:
            continue
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


def _collect_fit_diagnostics(
    result,
    X: pd.DataFrame,
) -> dict[str, float]:
    diagnostics: dict[str, float] = {}
    params = np.asarray(getattr(result, "params", []), dtype=np.float64)
    if params.size:
        try:
            gradient = np.asarray(result.model.score(params), dtype=np.float64)
        except Exception:
            gradient = np.array([])
        if gradient.size:
            diagnostics["gradient_norm"] = float(np.linalg.norm(gradient))
    eta = X.to_numpy(dtype=np.float64, copy=False) @ params
    if eta.size:
        diagnostics["max_abs_eta"] = float(np.max(np.abs(eta)))
    try:
        hessian = np.asarray(result.model.hessian(params), dtype=np.float64)
    except Exception:
        hessian = np.array([])
    if hessian.size:
        diagnostics["hessian_condition"] = _compute_condition_number(-hessian)
    retvals = getattr(result, "mle_retvals", {})
    if isinstance(retvals, dict):
        iterations = retvals.get("iterations")
        if iterations is not None:
            diagnostics["iterations"] = float(iterations)
    return diagnostics


def _predict_probabilities(result, X: pd.DataFrame) -> np.ndarray | None:
    predict = getattr(result, "predict", None)
    try:
        if callable(predict):
            probs = predict(X)
        else:
            model = getattr(result, "model", None)
            params = getattr(result, "params", None)
            if model is None or params is None:
                return None
            probs = model.predict(np.asarray(params, dtype=np.float64), exog=np.asarray(X, dtype=np.float64))
    except Exception:
        return None

    if isinstance(probs, pd.Series):
        probs_array = probs.to_numpy(dtype=np.float64, copy=False)
    else:
        probs_array = np.asarray(probs, dtype=np.float64)

    if probs_array.shape[0] != len(X):
        return None

    return probs_array


def _compute_lrt_bootstrap_pvalue(
    full_design: pd.DataFrame,
    reduced_design: pd.DataFrame,
    y: pd.Series,
    reduced_fit,
    observed_stat: float,
    *,
    replicates: int | None = None,
    seed: int | None = LRT_BOOTSTRAP_SEED,
) -> tuple[float, int, int]:
    if replicates is None:
        replicates = LRT_BOOTSTRAP_REPLICATES
    if replicates <= 0:
        return math.nan, 0, 0

    probabilities = _predict_probabilities(reduced_fit, reduced_design)
    if probabilities is None:
        return math.nan, 0, 0

    probabilities = np.clip(probabilities, 1e-9, 1 - 1e-9)
    rng = np.random.default_rng(seed)

    successes = 0
    exceedances = 0
    attempts = 0

    for _ in range(replicates):
        attempts += 1
        sampled = rng.binomial(1, probabilities, size=probabilities.shape[0])
        y_boot = pd.Series(sampled, index=y.index, dtype=np.float64)
        try:
            full_fit_boot, _, _ = _fit_logistic(full_design, y_boot)
            reduced_fit_boot, _, _ = _fit_logistic(reduced_design, y_boot.loc[reduced_design.index])
        except RuntimeError:
            continue

        full_ll = getattr(full_fit_boot, "llf", math.nan)
        reduced_ll = getattr(reduced_fit_boot, "llf", math.nan)
        if not (math.isfinite(full_ll) and math.isfinite(reduced_ll)):
            continue

        stat = max(0.0, 2.0 * (float(full_ll) - float(reduced_ll)))
        successes += 1
        if stat >= observed_stat:
            exceedances += 1

    if successes == 0:
        return math.nan, 0, attempts

    p_value = (exceedances + 1) / (successes + 1)
    return p_value, successes, attempts


def _is_penalized_result(result) -> bool:
    return bool(getattr(result, "penalized", False))


def _compute_penalized_parametric_bootstrap(
    result,
    X: pd.DataFrame,
    y: pd.Series,
    term: str,
    *,
    replicates: int | None = None,
    seed: int | None = RIDGE_PARAMETRIC_BOOTSTRAP_SEED,
):
    if replicates is None:
        replicates = RIDGE_PARAMETRIC_BOOTSTRAP_REPLICATES
    if replicates <= 0:
        return None

    probabilities = _predict_probabilities(result, X)
    if probabilities is None:
        return None

    probabilities = np.clip(probabilities, 1e-9, 1 - 1e-9)
    rng = np.random.default_rng(seed)

    sampled_betas: list[float] = []
    attempts = 0
    successes = 0

    for _ in range(replicates):
        attempts += 1
        sampled = rng.binomial(1, probabilities, size=probabilities.shape[0])
        y_boot = pd.Series(sampled, index=y.index, dtype=np.float64)
        try:
            boot_fit, _, _ = _fit_logistic(X, y_boot)
        except RuntimeError:
            continue

        beta = _extract_parameter(boot_fit, term)
        if not math.isfinite(beta):
            continue

        sampled_betas.append(float(beta))
        successes += 1

    if not sampled_betas:
        return None

    beta_array = np.asarray(sampled_betas, dtype=np.float64)
    try:
        ci_low = float(np.percentile(beta_array, 2.5))
        ci_high = float(np.percentile(beta_array, 97.5))
    except Exception:
        ci_low = math.nan
        ci_high = math.nan

    if beta_array.size > 1:
        se = float(np.std(beta_array, ddof=1))
    else:
        se = math.nan

    beta_hat = _extract_parameter(result, term)
    if math.isfinite(beta_hat) and math.isfinite(se) and se > 0:
        z_value = beta_hat / se
        p_value = 2.0 * float(norm.sf(abs(z_value)))
        log_p = math.log(p_value) if p_value > 0 else None
    else:
        z_value = math.nan
        p_value = math.nan
        log_p = None

    return {
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "log_p": log_p if log_p is not None and math.isfinite(log_p) else None,
        "z_value": z_value,
        "successes": successes,
        "attempts": attempts,
        "replicates": replicates,
    }


class PenalizedLogitResult:
    """Minimal result wrapper exposing Wald diagnostics for penalized fits."""

    def __init__(self, model: sm.Logit, params: np.ndarray, cov: np.ndarray) -> None:
        names = list(model.exog_names)
        self.model = model
        self.params = pd.Series(np.asarray(params, dtype=np.float64), index=names)
        self._cov = pd.DataFrame(np.asarray(cov, dtype=np.float64), index=names, columns=names)
        self.bse = pd.Series(np.sqrt(np.diag(self._cov.to_numpy())), index=names)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_values = self.params.to_numpy(dtype=np.float64) / self.bse.to_numpy(dtype=np.float64)
        self.tvalues = pd.Series(z_values, index=names)
        sqrt_two = math.sqrt(2.0)
        pvals = [
            math.erfc(abs(z) / sqrt_two) if math.isfinite(z) else math.nan
            for z in z_values
        ]
        self.pvalues = pd.Series(pvals, index=names)
        try:
            self.llf = float(model.loglike(self.params.to_numpy(dtype=np.float64)))
        except Exception:
            self.llf = math.nan
        try:
            self.nobs = float(model.endog.shape[0])
        except Exception:
            self.nobs = math.nan
        self.mle_retvals = {"converged": True, "method": "ridge"}
        self.converged = True
        self.penalized = True

    def cov_params(self) -> pd.DataFrame:
        return self._cov

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        z_value = float(norm.ppf(1 - alpha / 2))
        lower = self.params - z_value * self.bse
        upper = self.params + z_value * self.bse
        return pd.DataFrame({0: lower, 1: upper})

    def predict(self, exog):
        matrix = np.asarray(exog, dtype=np.float64)
        params = self.params.to_numpy(dtype=np.float64)
        return self.model.predict(params, exog=matrix)


def _fit_logistic(X: pd.DataFrame, y: pd.Series):
    if X.shape[0] != y.shape[0]:
        raise RuntimeError("Design matrix and response vector have incompatible shapes.")

    if X.dtypes.ne(np.float64).any():
        raise RuntimeError("Design matrix must be float64 before fitting.")

    X_matrix = X.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(X_matrix).all():
        raise RuntimeError("Design matrix contains non-finite values before fitting.")

    y_vector = y.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(y_vector).all():
        raise RuntimeError("Response vector contains non-finite values before fitting.")

    glm_result = None
    with warnings.catch_warnings(record=True) as glm_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        try:
            glm_model = sm.GLM(y, X, family=sm.families.Binomial())
            glm_result = glm_model.fit(maxiter=200, tol=1e-8)
        except Exception as exc:
            warn(f"Initial GLM fit failed: {exc}")
            glm_result = None
        else:
            if glm_warnings:
                warn(
                    "GLM emitted warnings: " + "; ".join(_format_warning_records(glm_warnings))
                )

    start_params = getattr(glm_result, "params", None)

    logit_model = sm.Logit(y, X)
    with warnings.catch_warnings(record=True) as logit_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        try:
            result = logit_model.fit(
                start_params=start_params,
                method="lbfgs",
                maxiter=200,
                tol=1e-8,
                disp=False,
            )
            method = "logit_lbfgs"
        except (PerfectSeparationError, np.linalg.LinAlgError, ValueError) as exc:
            warn(f"Logit fit failed: {exc}; retrying without start parameters")
            try:
                result = logit_model.fit(
                    method="lbfgs",
                    maxiter=200,
                    tol=1e-8,
                    disp=False,
                )
                method = "logit_lbfgs"
            except Exception as inner_exc:  # pragma: no cover - defensive branch
                raise RuntimeError(
                    _format_logistic_failure(inner_exc, X, y)
                ) from inner_exc
        except Exception as exc:  # pragma: no cover - defensive branch
            raise RuntimeError(_format_logistic_failure(exc, X, y)) from exc

    if logit_warnings:
        warn("Logit fit emitted warnings: " + "; ".join(_format_warning_records(logit_warnings)))

    diagnostics = _collect_fit_diagnostics(result, X)
    need_ridge = False
    gradient_norm = diagnostics.get("gradient_norm")
    if gradient_norm is not None and gradient_norm > GRADIENT_NORM_THRESHOLD:
        need_ridge = True
    hessian_cond = diagnostics.get("hessian_condition")
    if hessian_cond is not None and hessian_cond > HESSIAN_CONDITION_THRESHOLD:
        need_ridge = True
    max_abs_eta = diagnostics.get("max_abs_eta")
    if max_abs_eta is not None and max_abs_eta > 30.0:
        need_ridge = True
    if not _logit_result_converged(result):
        need_ridge = True

    if need_ridge:
        warn(
            "Triggering ridge fallback due to convergence diagnostics "
            f"(gradient_norm={gradient_norm}, hessian_condition={hessian_cond}, max_abs_eta={max_abs_eta})."
        )
        pen_weight = np.ones(X.shape[1], dtype=np.float64)
        pen_weight[0] = 0.0  # do not penalize intercept
        penalized = logit_model.fit_regularized(
            start_params=start_params,
            method="lbfgs",
            maxiter=200,
            alpha=RIDGE_ALPHA,
            L1_wt=0.0,
            pen_weight=pen_weight,
        )
        params = np.asarray(penalized.params, dtype=np.float64)
        try:
            hessian = np.asarray(logit_model.hessian(params), dtype=np.float64)
        except Exception:
            hessian = np.array([], dtype=np.float64)
        if hessian.size:
            penalty_matrix = np.diag(pen_weight * RIDGE_ALPHA)
            try:
                adjusted = -hessian + penalty_matrix
                cov = np.linalg.pinv(adjusted)
            except np.linalg.LinAlgError:
                cov = np.full((len(params), len(params)), np.nan, dtype=np.float64)
        else:
            cov = np.full((len(params), len(params)), np.nan, dtype=np.float64)
        result = PenalizedLogitResult(logit_model, params, cov)
        method = "logit_lbfgs (ridge penalty)"
        diagnostics = _collect_fit_diagnostics(result, X)
        diagnostics["penalized"] = True
    else:
        setattr(result, "penalized", False)

    return result, method, diagnostics


@dataclass
class PhenotypeRun:
    phenotype: str
    category: str


def _load_shared_covariates(
    *,
    client: bigquery.Client,
    project_id: str,
    cdr_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
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
    anc_series = ancestry_df.reindex(shared.index)["ANCESTRY"]
    missing_ancestry = anc_series.isna()
    if missing_ancestry.any():
        dropped = int(missing_ancestry.sum())
        warn(
            "Dropping participants lacking ancestry labels; unable to adjust "
            f"population structure for {dropped:,} individuals."
        )
        anc_series = anc_series.loc[~missing_ancestry]
        shared = shared.loc[anc_series.index]

    anc_cat = pd.Categorical(anc_series)
    anc_dummies = pd.get_dummies(
        anc_cat,
        prefix="ANC",
        drop_first=True,
        dtype=np.float64,
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

    if anc_dummies.shape[1] > 0:
        # Rows corresponding to previously dropped participants are no longer
        # present, but ``anc_series`` may still contain sporadic missing values if
        # the underlying cache was incomplete.  Propagate those NaNs into the dummy
        # matrix so that downstream logic can detect and remove them instead of
        # silently assigning individuals to the reference ancestry stratum.
        missing_rows = anc_series.isna()
        if missing_rows.any():
            anc_dummies.loc[missing_rows] = np.nan

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


@dataclass
class DesignPreconditioner:
    index: pd.Index
    dosage: pd.Series
    age_terms: pd.DataFrame
    sex: pd.Series | None
    pcs_resid: pd.DataFrame
    scaled_controls: pd.DataFrame
    ancestry_dummies: pd.DataFrame
    ancestry_labels: pd.Series
    pc_projection_basis: np.ndarray
    pc_projection_pinv: np.ndarray
    diagnostics: dict[str, object]
    control_order: list[str]
    age_columns: list[str]
    pc_columns: list[str]
    ancestry_columns: list[str]
    sex_column: str | None

    def build_design(
        self,
        control_columns: Sequence[str],
        *,
        include_sex: bool = True,
        restrict_index: pd.Index | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        selected_controls = [col for col in self.control_order if col in control_columns]

        if restrict_index is not None:
            subset_index = self.index.intersection(pd.Index(restrict_index))
        else:
            subset_index = self.index

        if subset_index.empty:
            raise RuntimeError("No participants remain after applying analysis subset.")

        frames: list[pd.DataFrame] = [
            self.dosage.loc[subset_index].to_frame(name="dosage"),
            self.age_terms.loc[subset_index],
        ]
        if include_sex and self.sex is not None:
            frames.append(self.sex.loc[subset_index].to_frame(name=self.sex.name))
        if not self.pcs_resid.empty:
            frames.append(self.pcs_resid.loc[subset_index])
        if selected_controls:
            frames.append(self.scaled_controls.loc[subset_index, selected_controls])
        if not self.ancestry_dummies.empty:
            frames.append(self.ancestry_dummies.loc[subset_index])

        design = pd.concat(frames, axis=1)
        order: list[str] = ["dosage", *self.age_columns]
        if include_sex and self.sex_column:
            order.append(self.sex_column)
        order.extend(self.pc_columns)
        order.extend(selected_controls)
        order.extend(self.ancestry_columns)

        design = design[order]
        design = design.astype(np.float64)
        design.insert(0, "const", 1.0)

        matrix = design.iloc[:, 1:].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(matrix).all():
            raise RuntimeError("Design matrix contains non-finite values after conditioning.")

        diagnostics = {k: v for k, v in self.diagnostics.items() if k != "control_scalers"}
        rank = _estimate_rank(matrix)
        diagnostics.update(
            {
                "condition_number": _compute_condition_number(matrix),
                "matrix_rank": rank,
                "effective_rank": rank,
                "final_columns": ["const", *order],
                "selected_custom_controls": selected_controls,
                "analysis_rows": len(subset_index),
            }
        )
        design.attrs["diagnostics"] = diagnostics

        if design.dtypes.ne(np.float64).any():
            raise RuntimeError("Design matrix contains non-float64 columns after conditioning.")

        return design


def _infer_ancestry_labels(
    ancestry_dummies: pd.DataFrame,
    reference_ancestry: str | None,
) -> pd.Series:
    if ancestry_dummies.empty:
        label = reference_ancestry if reference_ancestry is not None else "REFERENCE"
        return pd.Series(label, index=ancestry_dummies.index, dtype="object")

    labels = pd.Series(
        reference_ancestry if reference_ancestry is not None else "REFERENCE",
        index=ancestry_dummies.index,
        dtype="object",
    )
    nan_mask = ancestry_dummies.isna().any(axis=1)
    labels.loc[nan_mask] = np.nan
    for column in ancestry_dummies.columns:
        values = ancestry_dummies[column]
        active = values >= 0.5
        labels.loc[active] = column.split("ANC_", 1)[-1]
    return labels


def _zscore_controls_within_ancestry(
    controls: pd.DataFrame,
    ancestry_labels: pd.Series,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, float]]]]:
    if controls.empty:
        return controls.astype(np.float64), {}

    scaled = pd.DataFrame(index=controls.index, columns=controls.columns, dtype=np.float64)
    scalers: dict[str, dict[str, dict[str, float]]] = {}

    unique_labels = ancestry_labels.dropna().unique()
    for column in controls.columns:
        scalers[column] = {}
        col_values = controls[column].astype(np.float64)
        for label in unique_labels:
            mask = ancestry_labels == label
            if not mask.any():
                continue
            segment = col_values.loc[mask]
            mean = float(segment.mean()) if len(segment) else 0.0
            std = float(segment.std(ddof=0)) if len(segment) else 0.0
            if not math.isfinite(std) or std == 0.0:
                scaled.loc[mask, column] = 0.0
            else:
                scaled.loc[mask, column] = (segment - mean) / std
            scalers[column][str(label)] = {"mean": mean, "std": std}

        remaining = ancestry_labels.isna()
        if remaining.any():
            segment = col_values.loc[remaining]
            mean = float(segment.mean()) if len(segment) else 0.0
            std = float(segment.std(ddof=0)) if len(segment) else 0.0
            if not math.isfinite(std) or std == 0.0:
                scaled.loc[remaining, column] = 0.0
            else:
                scaled.loc[remaining, column] = (segment - mean) / std
            scalers[column]["nan"] = {"mean": mean, "std": std}

    scaled = scaled.fillna(0.0).astype(np.float64)
    return scaled, scalers


def _fit_design_preconditioner(
    core: pd.DataFrame,
    ancestry_dummies: pd.DataFrame,
    reference_ancestry: str | None,
    all_custom_controls: pd.DataFrame,
) -> DesignPreconditioner:
    df = core.copy()
    if not all_custom_controls.empty:
        df = df.join(all_custom_controls, how="left")

    ancestry_slice = ancestry_dummies.reindex(df.index)
    pc_columns = [f"PC{i}" for i in range(1, NUM_PCS + 1)]
    missing_pcs = [col for col in pc_columns if col not in df.columns]
    if missing_pcs:
        raise KeyError(f"Missing required principal components: {missing_pcs}")

    required_columns = ["dosage", "AGE", "sex", *pc_columns]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required covariate columns: {missing}")

    control_columns = list(dict.fromkeys(all_custom_controls.columns)) if not all_custom_controls.empty else []

    mask = pd.Series(True, index=df.index, dtype=bool)
    mask &= df[required_columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    if control_columns:
        mask &= df[control_columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    if ancestry_slice.shape[1] > 0:
        mask &= ~ancestry_slice.isna().any(axis=1)

    dropped = int((~mask).sum())
    if dropped:
        warn(f"Dropped {dropped:,} participants due to missing covariates during conditioning.")

    df = df.loc[mask]
    ancestry_slice = ancestry_slice.loc[mask]
    if control_columns:
        df = df.dropna(subset=control_columns)

    if df.empty:
        raise RuntimeError("No participants remain after conditioning filters.")

    ancestry_labels = _infer_ancestry_labels(ancestry_slice.fillna(0.0), reference_ancestry)

    dosage = df["dosage"].astype(np.float64)
    sex = df["sex"].astype(np.float64)

    pcs = df[pc_columns].astype(np.float64)
    pc_means = pcs.mean(axis=0)
    pc_stds = pcs.std(axis=0, ddof=0)
    pc_stds_replaced = pc_stds.replace(0.0, np.nan)
    pcs_z = (pcs - pc_means) / pc_stds_replaced
    pcs_z = pcs_z.fillna(0.0)

    ancestry_matrix = ancestry_slice.fillna(0.0).astype(np.float64)
    intercept = np.ones((len(ancestry_matrix), 1), dtype=np.float64)
    projection_basis = intercept if ancestry_matrix.empty else np.hstack((intercept, ancestry_matrix.to_numpy(dtype=np.float64)))
    projection_pinv = np.linalg.pinv(projection_basis)
    pcs_residual = pcs_z.to_numpy(dtype=np.float64) - projection_basis @ (projection_pinv @ pcs_z.to_numpy(dtype=np.float64))
    pcs_resid_df = pd.DataFrame(pcs_residual, index=df.index, columns=pc_columns)

    age_years = df["AGE"].astype(np.float64)
    age_centered = (age_years - age_years.mean()) / 10.0
    age_basis = pd.DataFrame(
        {"AGE_c": age_centered, "AGE_c_sq": age_centered ** 2},
        index=df.index,
        dtype=np.float64,
    )

    if control_columns:
        controls = df[control_columns].astype(np.float64)
        scaled_controls, control_scalers = _zscore_controls_within_ancestry(controls, ancestry_labels)
    else:
        scaled_controls = pd.DataFrame(index=df.index, dtype=np.float64)
        control_scalers = {}

    ancestry_final = ancestry_matrix.astype(np.float64)

    components = [
        dosage.to_frame(name="dosage"),
        age_basis,
        sex.to_frame(name="sex"),
        pcs_resid_df,
        scaled_controls,
        ancestry_final,
    ]
    combined = pd.concat(components, axis=1)
    combined = combined.astype(np.float64)

    if not np.isfinite(combined.to_numpy(dtype=np.float64, copy=False)).all():
        raise RuntimeError("Non-finite values detected in covariates during conditioning.")

    variances = combined.var(axis=0, ddof=0)
    zero_variance = variances[variances == 0.0].index.tolist()
    if "dosage" in zero_variance:
        raise RuntimeError("Inversion dosage is constant after conditioning; cannot fit logistic model.")

    if "sex" in zero_variance:
        sex_series: pd.Series | None = None
    else:
        sex_series = sex.astype(np.float64)

    age_basis = age_basis.drop(columns=[col for col in age_basis.columns if col in zero_variance])
    pcs_resid_df = pcs_resid_df.drop(columns=[col for col in pcs_resid_df.columns if col in zero_variance])
    scaled_controls = scaled_controls.drop(columns=[col for col in scaled_controls.columns if col in zero_variance])
    ancestry_final = ancestry_final.drop(columns=[col for col in ancestry_final.columns if col in zero_variance])

    control_order = [col for col in control_columns if col in scaled_controls.columns]

    final_components = [
        dosage.to_frame(name="dosage"),
        age_basis,
        sex_series.to_frame(name="sex") if sex_series is not None else None,
        pcs_resid_df,
        scaled_controls,
        ancestry_final,
    ]
    final_components = [frame for frame in final_components if frame is not None and not frame.empty]
    final_matrix = pd.concat(final_components, axis=1).astype(np.float64)

    matrix = final_matrix.to_numpy(dtype=np.float64, copy=False)
    rank = _estimate_rank(matrix)
    diagnostics: dict[str, object] = {
        "conditioning_rows": len(df),
        "dropped_rows_missing": dropped,
        "zero_variance_columns": zero_variance,
        "condition_number": _compute_condition_number(matrix),
        "matrix_rank": rank,
        "effective_rank": rank,
        "control_scalers": control_scalers,
    }

    return DesignPreconditioner(
        index=df.index,
        dosage=dosage.astype(np.float64),
        age_terms=age_basis.astype(np.float64),
        sex=sex_series.astype(np.float64) if sex_series is not None else None,
        pcs_resid=pcs_resid_df.astype(np.float64),
        scaled_controls=scaled_controls.astype(np.float64),
        ancestry_dummies=ancestry_final.astype(np.float64),
        ancestry_labels=ancestry_labels,
        pc_projection_basis=projection_basis,
        pc_projection_pinv=projection_pinv,
        diagnostics=diagnostics,
        control_order=control_order,
        age_columns=list(age_basis.columns),
        pc_columns=list(pcs_resid_df.columns),
        ancestry_columns=list(ancestry_final.columns),
        sex_column="sex" if sex_series is not None else None,
    )


def _ensure_pheno_cache(
    definitions: pd.DataFrame,
    phenotype: str,
    cdr_id: str,
    core_index: pd.Index,
    project_id: str,
    *,
    client: bigquery.Client,
) -> None:
    match = definitions.loc[definitions["sanitized_name"] == phenotype]
    if match.empty:
        raise KeyError(f"Phenotype '{phenotype}' not present in definitions table.")
    if len(match) > 1:
        raise RuntimeError(f"Multiple definition rows found for phenotype '{phenotype}'.")
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


def _log_lines(prefix: str, message: str, *, level: str = "info") -> None:
    lines = message.splitlines()
    if not lines:
        lines = [""]
    log_path = _phenotype_log_path(prefix)
    with log_path.open("a", encoding="utf-8") as handle:
        for line in lines:
            text = f"[{prefix}] {line}"
            if level == "info":
                info(text)
            else:
                warn(text)
            handle.write(text + "\n")


def _format_summary_dict(stats: Mapping[str, object]) -> str:
    pieces: list[str] = []
    for key, value in stats.items():
        if isinstance(value, (float, np.floating)):
            pieces.append(f"{key}={float(value):.6g}")
        else:
            pieces.append(f"{key}={value}")
    return ", ".join(pieces)


def _analyse_single_phenotype(
    *,
    cfg: "PhenotypeRun",
    inv: str,
    preconditioner: DesignPreconditioner,
    custom_control_names: Sequence[str],
    definitions: pd.DataFrame,
    cdr_id: str,
    scores_path: Path,
) -> tuple[dict[str, object] | None, list[str]]:
    prefix = f"{cfg.phenotype}"
    _initialise_pheno_log(prefix)
    _log_lines(prefix, f"Initialising analysis for inversion {inv} (category: {cfg.category})")

    analysis_index = preconditioner.index
    include_sex = True
    if cfg.category == "breast_cancer":
        include_sex = False
        sex_series = preconditioner.sex
        if sex_series is not None:
            total_available = len(analysis_index)
            female_mask = sex_series == 0
            female_index = sex_series.index[female_mask]
            filtered_index = analysis_index.intersection(female_index)
            _log_lines(
                prefix,
                "Restricting to genetically inferred females; "
                f"retained {len(filtered_index)} of {total_available} participants ("
                f"dropped {total_available - len(filtered_index)}).",
            )
            if filtered_index.empty:
                _log_lines(
                    prefix,
                    "No genetically inferred female participants remain after filtering; skipping.",
                    level="warn",
                )
                return None, []
            analysis_index = filtered_index
        else:
            _log_lines(
                prefix,
                "Sex covariate absent after conditioning; assuming cohort already restricted to a single sex.",
            )

    try:
        design = preconditioner.build_design(
            custom_control_names,
            include_sex=include_sex,
            restrict_index=analysis_index,
        )
    except RuntimeError as exc:
        _log_lines(prefix, f"Skipping phenotype due to design matrix issue: {exc}", level="warn")
        return None, []

    cdr_codename = cdr_id.split(".")[-1]
    phenotype_status = _load_case_status(
        cfg.phenotype,
        cdr_codename,
        analysis_index,
    )

    n_cases = int(phenotype_status.sum())
    n_total = int(len(phenotype_status))
    n_ctrls = n_total - n_cases

    if n_cases == 0 or n_ctrls == 0:
        _log_lines(
            prefix,
            f"Insufficient cases ({n_cases}) or controls ({n_ctrls}); skipping.",
            level="warn",
        )
        return None, []

    _log_lines(
        prefix,
        f"Participants available after filtering: {n_total} (cases={n_cases}, controls={n_ctrls})",
    )

    y = phenotype_status.loc[design.index].astype(np.int8)
    X = design.copy()

    design_diag = X.attrs.get("diagnostics", {})
    if design_diag:
        _log_lines(prefix, "Design diagnostics: " + _format_summary_dict(design_diag))

    dosage_stats = _summarise_column(X["dosage"])
    _log_lines(prefix, "Dosage summary: " + _format_summary_dict(dosage_stats))

    custom_cols = [c for c in X.columns if c.upper().startswith(CUSTOM_CONTROL_PREFIX.upper())]
    if custom_cols:
        _log_lines(prefix, f"Custom covariates in design: {', '.join(custom_cols)}")
        for covar in custom_cols:
            covar_stats = _summarise_column(X[covar])
            _log_lines(prefix, f"{covar} summary: " + _format_summary_dict(covar_stats))
    else:
        _log_lines(prefix, "No custom covariates present after alignment.")

    _log_lines(
        prefix,
        _summarise_design_matrix(X, y),
    )

    for line in _describe_covariate_columns(X):
        _log_lines(prefix, f"Covariate diagnostic: {line}")

    intercept = "const" in X.columns
    ordered_terms = [col for col in X.columns if col != "const"]
    if "dosage" in ordered_terms:
        ordered_terms.insert(0, ordered_terms.pop(ordered_terms.index("dosage")))
    formula_terms: list[str] = []
    if intercept:
        formula_terms.append("1")
    formula_terms.extend(ordered_terms)
    formula_repr = " + ".join(formula_terms) if formula_terms else "<empty design>"
    _log_lines(
        prefix,
        f"Model specification: logit(P({cfg.phenotype}=1)) ~ {formula_repr}",
    )

    try:
        fit, fit_method, fit_diagnostics = _fit_logistic(X, y)
    except RuntimeError as exc:
        _log_lines(prefix, f"Logistic regression failed: {exc}", level="warn")
        return None, []

    if fit_diagnostics:
        _log_lines(prefix, "Fit diagnostics: " + _format_summary_dict(fit_diagnostics))

    term_summary = _summarise_term(fit, "dosage", design=X, response=y)
    beta = float(term_summary.get("beta", math.nan))
    if math.isnan(beta):
        _log_lines(prefix, "Dosage coefficient was not estimated; skipping.", level="warn")
        return None, []
    if "penalized_bootstrap_successes" in term_summary:
        successes = term_summary.get("penalized_bootstrap_successes")
        attempts = term_summary.get("penalized_bootstrap_attempts")
        replicates = term_summary.get("penalized_bootstrap_replicates")
        _log_lines(
            prefix,
            "Penalized fit used parametric bootstrap: "
            f"successes={successes}/{attempts} (requested={replicates}).",
        )
    se = float(term_summary.get("se", math.nan))
    p_value = term_summary.get("p_value", math.nan)
    odds_ratio = term_summary.get("odds_ratio", math.nan)
    ci_lower = term_summary.get("ci_lower", math.nan)
    ci_upper = term_summary.get("ci_upper", math.nan)
    log_p_value = term_summary.get("log_p")
    if log_p_value is None or not math.isfinite(log_p_value):
        log_p_value = math.nan

    lrt_bootstrap_pvalue = math.nan
    lrt_bootstrap_successes = 0
    lrt_bootstrap_attempts = 0
    lrt_bootstrap_ran = False
    lrt_statistic = math.nan
    try:
        reduced_design = X.drop(columns=["dosage"])
    except KeyError:
        _log_lines(prefix, "Reduced design without dosage could not be constructed.", level="warn")
        reduced_design = None
    if reduced_design is not None and not reduced_design.empty:
        try:
            reduced_fit, _, reduced_diag = _fit_logistic(
                reduced_design,
                y.loc[reduced_design.index],
            )
        except RuntimeError as exc:
            _log_lines(
                prefix,
                f"Reduced logistic regression (without dosage) failed: {exc}",
                level="warn",
            )
        else:
            if reduced_diag:
                _log_lines(prefix, "Reduced fit diagnostics: " + _format_summary_dict(reduced_diag))
            full_ll = getattr(fit, "llf", math.nan)
            reduced_ll = getattr(reduced_fit, "llf", math.nan)
            if math.isfinite(full_ll) and math.isfinite(reduced_ll):
                lrt_statistic = max(0.0, 2.0 * (float(full_ll) - float(reduced_ll)))
                raw_max_abs_eta = (
                    reduced_diag.get("max_abs_eta")
                    if isinstance(reduced_diag, Mapping)
                    else None
                )
                try:
                    max_abs_eta = (
                        float(raw_max_abs_eta)
                        if raw_max_abs_eta is not None
                        else None
                    )
                except (TypeError, ValueError):
                    max_abs_eta = None
                effective_replicates = _resolve_lrt_bootstrap_replicates(
                    len(reduced_design),
                    max_abs_eta,
                )
                if effective_replicates <= 0:
                    _log_lines(
                        prefix,
                        "Skipping LRT bootstrap (cohort too large or predictions saturated).",
                    )
                else:
                    if effective_replicates < LRT_BOOTSTRAP_REPLICATES:
                        _log_lines(
                            prefix,
                            "Running LRT bootstrap with "
                            f"{effective_replicates} replicates (down from {LRT_BOOTSTRAP_REPLICATES}).",
                        )
                    (
                        lrt_bootstrap_pvalue,
                        lrt_bootstrap_successes,
                        lrt_bootstrap_attempts,
                    ) = _compute_lrt_bootstrap_pvalue(
                        X,
                        reduced_design,
                        y,
                        reduced_fit,
                        lrt_statistic,
                        replicates=effective_replicates,
                    )
                    lrt_bootstrap_ran = True
            else:
                _log_lines(
                    prefix,
                    "Log-likelihoods for full or reduced model were not finite; skipping LRT bootstrap.",
                    level="warn",
                )
    elif reduced_design is not None and reduced_design.empty:
        _log_lines(prefix, "Reduced design is empty after removing dosage; skipping LRT bootstrap.", level="warn")

    baseline_summary = _empty_term_summary()
    baseline_method: str | None = None
    baseline_control_names: list[str] = []
    try:
        baseline_design = preconditioner.build_design(
            baseline_control_names,
            include_sex=include_sex,
            restrict_index=analysis_index,
        )
    except RuntimeError as exc:
        _log_lines(
            prefix,
            f"Baseline design construction failed: {exc}",
            level="warn",
        )
        baseline_design = X.drop(columns=custom_cols, errors="ignore")
        baseline_design.attrs["diagnostics"] = {}
    else:
        baseline_design_diag = baseline_design.attrs.get("diagnostics", {})
        if baseline_design_diag:
            _log_lines(prefix, "Baseline design diagnostics: " + _format_summary_dict(baseline_design_diag))
    try:
        baseline_fit, baseline_method, baseline_diag = _fit_logistic(
            baseline_design,
            y.loc[baseline_design.index],
        )
        baseline_summary = _summarise_term(
            baseline_fit,
            "dosage",
            design=baseline_design,
            response=y.loc[baseline_design.index],
        )
        if "penalized_bootstrap_successes" in baseline_summary:
            successes = baseline_summary.get("penalized_bootstrap_successes")
            attempts = baseline_summary.get("penalized_bootstrap_attempts")
            replicates = baseline_summary.get("penalized_bootstrap_replicates")
            _log_lines(
                prefix,
                "Baseline penalized fit used parametric bootstrap: "
                f"successes={successes}/{attempts} (requested={replicates}).",
            )
    except RuntimeError as exc:
        _log_lines(
            prefix,
            f"Baseline logistic regression (without custom PGS covariates) failed: {exc}",
            level="warn",
        )
        baseline_diag = {}
    else:
        if baseline_diag:
            _log_lines(prefix, "Baseline fit diagnostics: " + _format_summary_dict(baseline_diag))

    baseline_log_p = baseline_summary.get("log_p")
    if baseline_log_p is None or not math.isfinite(baseline_log_p):
        baseline_log_p = math.nan

    summary_lines = [
        f"Analysis complete using {fit_method}; observations={n_total}, cases={n_cases}, controls={n_ctrls}, case_prevalence={n_cases / n_total:.6g}",
        (
            f"Dosage OR={odds_ratio} (95% CI: {ci_lower}, {ci_upper}); "
            f"beta={beta:.6g}, SE={se:.6g}"
        ),
        f"p-value={p_value} (log_p={log_p_value})",
    ]
    if math.isfinite(lrt_bootstrap_pvalue):
        summary_lines.append(
            "LRT bootstrap p-value="
            f"{lrt_bootstrap_pvalue} (successes={lrt_bootstrap_successes}/{lrt_bootstrap_attempts})"
        )
    elif lrt_bootstrap_ran:
        summary_lines.append("LRT bootstrap p-value unavailable (no successful replicates)")
    else:
        summary_lines.append("LRT bootstrap p-value unavailable (reduced model not fitted)")
    if baseline_method:
        summary_lines.append(
            (
                f"Baseline ({baseline_method}) OR={baseline_summary.get('odds_ratio', math.nan)} "
                f"(95% CI: {baseline_summary.get('ci_lower', math.nan)}, {baseline_summary.get('ci_upper', math.nan)}); "
                f"p-value={baseline_summary.get('p_value', math.nan)}"
            )
        )

    result: dict[str, object] = {
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
        "P_Value_LRT_Bootstrap": lrt_bootstrap_pvalue,
        "Fit_Method": fit_method,
        "Control_File": str(scores_path),
        "Custom_Covariates": ",".join(custom_cols),
        "OR_NoCustomControls": baseline_summary.get("odds_ratio", math.nan),
        "OR_NoCustomControls_95CI_Lower": baseline_summary.get("ci_lower", math.nan),
        "OR_NoCustomControls_95CI_Upper": baseline_summary.get("ci_upper", math.nan),
        "P_Value_NoCustomControls": baseline_summary.get("p_value", math.nan),
        "Log_P_Value_NoCustomControls": baseline_log_p,
        "Fit_Method_NoCustomControls": baseline_method,
    }

    return result, summary_lines



def run() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    project_id = os.getenv("GOOGLE_PROJECT")
    cdr_id = os.getenv("WORKSPACE_CDR")
    if not project_id or not cdr_id:
        die("GOOGLE_PROJECT and WORKSPACE_CDR must be defined in the environment.")

    client = bigquery.Client(project=project_id)
    try:
        shared_covariates, anc_dummies, reference_ancestry = _load_shared_covariates(
            client=client,
            project_id=project_id,
            cdr_id=cdr_id,
        )
        definitions = pheno.load_definitions(PHENOTYPE_DEFINITIONS_URL)
        target_runs = _resolve_target_runs(definitions)
        if not target_runs:
            warn("No phenotypes selected for analysis; exiting.")
            return

        scores_table, scores_path = _load_scores_table()
        category_controls = _build_category_controls(scores_table)

        results: list[dict[str, object]] = []
        for inv in TARGET_INVERSIONS:
            info(f"Preparing inversion {inv}")
            inversion_df = _load_inversion(inv)
            core = shared_covariates.join(inversion_df, how="inner")
            core = core.rename(columns={inv: "dosage"}) if inv in core.columns else core

            if "dosage" not in core.columns:
                raise KeyError(f"Inversion column '{inv}' missing after join")

            control_frames: list[pd.DataFrame] = []
            category_control_names: dict[str, list[str]] = {}
            for category, controls in category_controls.items():
                aligned = controls.reindex(core.index)
                control_frames.append(aligned)
                category_control_names[category] = list(aligned.columns)

            if control_frames:
                all_controls = pd.concat(control_frames, axis=1)
                all_controls = all_controls.loc[:, ~all_controls.columns.duplicated()]
                all_controls = all_controls.reindex(core.index)
            else:
                all_controls = pd.DataFrame(index=core.index)

            try:
                preconditioner = _fit_design_preconditioner(
                    core,
                    anc_dummies,
                    reference_ancestry,
                    all_controls,
                )
            except RuntimeError as exc:
                warn(f"Skipping inversion {inv} due to conditioning failure: {exc}")
                continue

            info(
                "Preconditioned cohort: "
                f"{len(preconditioner.index):,} participants (condition_number={preconditioner.diagnostics.get('condition_number', float('nan')):.6g})"
            )

            for category, columns in category_control_names.items():
                category_control_names[category] = [
                    col for col in columns if col in preconditioner.scaled_controls.columns
                ]

            ready_runs: list[tuple[PhenotypeRun, list[str]]] = []
            for cfg in target_runs:
                control_names = category_control_names.get(cfg.category)
                if not control_names:
                    warn(
                        f"No custom controls available for category '{cfg.category}'; skipping {cfg.phenotype}."
                    )
                    continue
                try:
                    _ensure_pheno_cache(
                        definitions,
                        cfg.phenotype,
                        cdr_id,
                        preconditioner.index.astype(str),
                        project_id,
                        client=client,
                    )
                except Exception as exc:
                    warn(
                        f"Failed to prepare phenotype cache for {cfg.phenotype}: {exc}; skipping."
                    )
                    continue
                ready_runs.append((cfg, control_names))

            if not ready_runs:
                warn(f"No analysable phenotypes remained for inversion {inv}; skipping.")
                continue

            max_workers_env = os.getenv("CUSTOM_FOLLOWUP_MAX_WORKERS")
            try:
                configured_workers = int(max_workers_env) if max_workers_env else 1
            except (TypeError, ValueError):
                configured_workers = 1
            configured_workers = max(1, configured_workers)
            cpu_cap = os.cpu_count() or 1
            max_workers = max(1, min(configured_workers, len(ready_runs), cpu_cap, 4))

            if max_workers == 1:
                for cfg, control_names in ready_runs:
                    try:
                        result, summary_lines = _analyse_single_phenotype(
                            cfg=cfg,
                            inv=inv,
                            preconditioner=preconditioner,
                            custom_control_names=control_names,
                            definitions=definitions,
                            cdr_id=cdr_id,
                            scores_path=scores_path,
                        )
                    except Exception as exc:  # pragma: no cover - defensive guard
                        warn(f"Unhandled error in {cfg.phenotype} analysis: {exc}")
                        continue

                    if not result:
                        continue

                    results.append(result)
                    for line in summary_lines:
                        _log_lines(cfg.phenotype, line)
            else:
                mp_context = multiprocessing.get_context("spawn")
                pending: dict[Future, PhenotypeRun] = {}
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=mp_context,
                ) as executor:
                    for cfg, control_names in ready_runs:
                        future = executor.submit(
                            _analyse_single_phenotype,
                            cfg=cfg,
                            inv=inv,
                            preconditioner=preconditioner,
                            custom_control_names=control_names,
                            definitions=definitions,
                            cdr_id=cdr_id,
                            scores_path=scores_path,
                        )
                        pending[future] = cfg

                    for future in as_completed(pending):
                        cfg = pending[future]
                        try:
                            result, summary_lines = future.result()
                        except Exception as exc:  # pragma: no cover - defensive guard
                            warn(f"Unhandled error in {cfg.phenotype} analysis: {exc}")
                            continue

                        if not result:
                            continue

                        results.append(result)
                        for line in summary_lines:
                            _log_lines(cfg.phenotype, line)

        if not results:
            warn("No successful analyses were produced; skipping output file.")
            return

        output_df = pd.DataFrame(results)
        pvals = pd.to_numeric(output_df.get("P_Value"), errors="coerce")
        output_df["BH_FDR_Q"] = np.nan
        mask = pvals.notna() & np.isfinite(pvals)
        if mask.any():
            _, qvals, _, _ = multipletests(pvals.loc[mask], alpha=0.05, method="fdr_bh")
            output_df.loc[mask, "BH_FDR_Q"] = qvals

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".tmp")
        output_df.to_csv(tmp_path, sep="\t", index=False)
        tmp_path.replace(OUTPUT_PATH)
        info(f"Wrote {len(output_df)} rows to {OUTPUT_PATH}")
    finally:
        client.close()


if __name__ == "__main__":  # pragma: no cover - manual execution only
    run()
