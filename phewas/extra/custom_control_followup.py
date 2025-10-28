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
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Context, Decimal, localcontext
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import log_ndtr
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError

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

# Logistic models occasionally reach extremely large coefficient estimates when the
# outcome is nearly separated.  ``MAX_ABS_DOSAGE_BETA`` bounds what we consider a
# numerically stable dosage coefficient; larger values trigger detailed
# diagnostics so that the root cause can be investigated instead of applying
# ever-stronger penalties.
MAX_ABS_DOSAGE_BETA = 15.0

# Optimisation strategies attempted when fitting logistic regressions.  The
# ``label`` is used for logging, while ``options`` are forwarded to
# :meth:`statsmodels.discrete.discrete_model.Logit.fit`.
LOGIT_SOLVER_CANDIDATES: Sequence[tuple[str, dict[str, object]]] = (
    ("logit_newton", {"method": "newton", "maxiter": 200, "tol": 1e-8, "warn_convergence": True}),
    ("logit_bfgs", {"method": "bfgs", "maxiter": 500, "tol": 1e-8, "warn_convergence": True}),
    ("logit_lbfgs", {"method": "lbfgs", "maxiter": 500, "tol": 1e-8, "warn_convergence": True}),
    ("logit_powell", {"method": "powell", "maxiter": 500, "warn_convergence": True}),
)


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


def _report_model_instability(
    label: str,
    issues: list[str],
    metrics: dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    warning_records: Sequence[warnings.WarningMessage],
    *,
    include_design: bool = True,
) -> None:
    if issues:
        warn(f"{label} logistic fit unstable: {'; '.join(issues)}")
    else:
        warn(f"{label} logistic fit unstable for unknown reasons")

    if metrics:
        metric_summary = _format_summary_dict(metrics)
        warn(f"{label} dosage coefficient diagnostics: {metric_summary}")

    if warning_records:
        warn(
            f"{label} emitted convergence warnings: "
            + "; ".join(_format_warning_records(warning_records))
        )

    if include_design:
        for line in _describe_dosage_distribution(X, y):
            warn(f"{label} dosage diagnostic: {line}")

        design_lines = _summarise_design_matrix(X, y).splitlines()
        for line in design_lines:
            warn(f"{label} design diagnostic: {line}")

        for line in _describe_covariate_columns(X):
            warn(f"{label} covariate diagnostic: {line}")


def _summarise_term(result, term: str) -> dict[str, float | str | None]:
    beta = _extract_parameter(result, term)
    se = _extract_series_value(result, "bse", term)
    if not math.isfinite(se) or se <= 0:
        se = _compute_standard_error(result, term)

    p_value = _extract_series_value(result, "pvalues", term)
    z_value = _extract_series_value(result, "tvalues", term)
    log_p: float | None

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

    conf_int = getattr(result, "conf_int", None)
    ci_low = ci_high = math.nan
    if callable(conf_int):
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


def _collect_convergence_metadata(result) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for attr in ("mle_retvals", "fit_history"):
        value = getattr(result, attr, None)
        if isinstance(value, Mapping):
            for key, item in value.items():
                metadata.setdefault(str(key), item)
    converged_attr = getattr(result, "converged", None)
    if converged_attr is not None:
        metadata.setdefault("converged", bool(converged_attr))
    return metadata


def _fit_has_converged(result) -> bool:
    metadata = _collect_convergence_metadata(result)
    converged = metadata.get("converged")
    if isinstance(converged, (bool, np.bool_)):
        return bool(converged)
    return True


def _report_fit_diagnostics(result, label: str) -> None:
    metadata = _collect_convergence_metadata(result)
    converged = metadata.get("converged")
    if isinstance(converged, (bool, np.bool_)) and converged:
        return

    warnflag = metadata.get("warnflag")
    reason_parts: list[str] = []
    if isinstance(warnflag, (int, np.integer)):
        warnflag_map = {
            1: "iteration limit reached before convergence",
            2: "parameter change below tolerance but gradient not close to zero",
            3: "likelihood failed to increase; possible separation or collinearity",
        }
        reason_parts.append(warnflag_map.get(int(warnflag), f"warnflag={warnflag}"))
    for key in ("iterations", "criterion", "deviance", "score", "score_norm", "step"):
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, (float, np.floating)):
            reason_parts.append(f"{key}={float(value):.3e}")
        else:
            reason_parts.append(f"{key}={value}")

    try:
        score_vec = result.model.score(result.params)  # type: ignore[call-arg]
        score_norm = float(np.linalg.norm(np.asarray(score_vec, dtype=np.float64)))
        if math.isfinite(score_norm):
            reason_parts.append(f"score_norm={score_norm:.3e}")
    except Exception:
        pass

    if not reason_parts:
        reason_parts.append("no convergence diagnostics available")

    warn(f"{label} logistic fit did not converge ({'; '.join(reason_parts)})")

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


def _fit_logistic(X: pd.DataFrame, y: pd.Series):
    model = sm.Logit(y, X)
    attempts: list[tuple[str, Exception]] = []
    diagnostics_reported = False

    for label, solver_options in LOGIT_SOLVER_CANDIDATES:
        fit_kwargs = dict(solver_options)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            warnings.simplefilter("always", RuntimeWarning)
            try:
                result = model.fit(disp=False, **fit_kwargs)
            except (PerfectSeparationError, np.linalg.LinAlgError, ValueError) as exc:
                warn(f"{label} solver failed: {exc}")
                include_design = not diagnostics_reported
                _report_model_instability(
                    label,
                    [f"solver raised {exc}"],
                    {},
                    X,
                    y,
                    caught,
                    include_design=include_design,
                )
                if include_design:
                    diagnostics_reported = True
                attempts.append((label, exc))
                continue
            except Exception as exc:  # pragma: no cover - defensive branch
                warn(f"{label} solver failed with unexpected error: {exc}")
                include_design = not diagnostics_reported
                _report_model_instability(
                    label,
                    [f"unexpected solver error: {exc}"],
                    {},
                    X,
                    y,
                    caught,
                    include_design=include_design,
                )
                if include_design:
                    diagnostics_reported = True
                attempts.append((label, exc))
                continue

        stable, issues, metrics = _evaluate_result_stability(result)
        if stable:
            if caught:
                warn(
                    f"{label} emitted convergence warnings despite stability: "
                    + "; ".join(_format_warning_records(caught))
                )
            return result, label

        include_design = not diagnostics_reported
        _report_model_instability(
            label,
            issues or ["dosage estimate flagged as unstable"],
            metrics,
            X,
            y,
            caught,
            include_design=include_design,
        )
        if include_design:
            diagnostics_reported = True
        issue_message = "; ".join(issues) if issues else "unstable dosage estimate"
        attempts.append((label, RuntimeError(issue_message)))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        warnings.simplefilter("always", RuntimeWarning)
        try:
            glm_model = sm.GLM(y, X, family=sm.families.Binomial())
            glm_result = glm_model.fit(maxiter=500, tol=1e-8)
        except Exception as exc:
            warn(f"glm_binomial solver failed: {exc}")
            include_design = not diagnostics_reported
            _report_model_instability(
                "glm_binomial",
                [f"solver raised {exc}"],
                {},
                X,
                y,
                caught,
                include_design=include_design,
            )
            if include_design:
                diagnostics_reported = True
            attempts.append(("glm_binomial", exc))
        else:
            stable, issues, metrics = _evaluate_result_stability(glm_result)
            if stable:
                if caught:
                    warn(
                        "glm_binomial emitted convergence warnings despite stability: "
                        + "; ".join(_format_warning_records(caught))
                    )
                warn("Falling back to GLM binomial fit for converged estimates.")
                return glm_result, "glm_binomial"

            include_design = not diagnostics_reported
            _report_model_instability(
                "glm_binomial",
                issues or ["dosage estimate flagged as unstable"],
                metrics,
                X,
                y,
                caught,
                include_design=include_design,
            )
            if include_design:
                diagnostics_reported = True
            issue_message = "; ".join(issues) if issues else "unstable dosage estimate"
            attempts.append(("glm_binomial", RuntimeError(issue_message)))

    if attempts:
        failure_lines = [f"{label}: {exc}" for label, exc in attempts]
        message = "All logistic solvers failed:\n    " + "\n    ".join(failure_lines)
    else:
        message = "Logistic solver failed for unknown reasons"

    raise RuntimeError(_format_logistic_failure(RuntimeError(message), X, y))


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

    diagnostics: dict[str, object] = {
        "dropped_non_finite": 0,
        "dropped_constant": [],
        "dropped_duplicates": [],
        "dropped_collinear": [],
        "ancestry_constants": [],
        "dropped_missing_ancestry": 0,
    }

    design = df[covar_cols].astype(np.float32, copy=False)
    design["const"] = np.float32(1.0)

    anc_slice = ancestry_dummies.reindex(design.index)
    if anc_slice.shape[1] > 0:
        missing_mask = anc_slice.isna().all(axis=1)
        if missing_mask.any():
            count_missing = int(missing_mask.sum())
            diagnostics["dropped_missing_ancestry"] = count_missing
            design = design.loc[~missing_mask]
            anc_slice = anc_slice.loc[~missing_mask]
            warn(
                "Dropped participants lacking ancestry covariates after alignment: "
                f"{count_missing:,}"
            )
    anc_slice = anc_slice.fillna(0.0).astype(np.float32)
    out = pd.concat([design, anc_slice], axis=1)

    if not custom_controls.empty:
        aligned_custom = custom_controls.reindex(out.index)
        aligned_custom = aligned_custom.dropna(axis=0, how="any")
        out = out.loc[aligned_custom.index]
        out = pd.concat([out, aligned_custom.astype(np.float32)], axis=1)

    finite_mask = np.isfinite(out.to_numpy(dtype=np.float64)).all(axis=1)
    if not finite_mask.all():
        failing_rows = out.loc[~finite_mask]
        non_finite_details: dict[str, dict[str, object]] = {}

        for column in out.columns:
            column_values = failing_rows[column]
            if column_values.empty:
                continue
            details: dict[str, object] = {}
            nan_count = int(column_values.isna().sum())
            if nan_count:
                details["nan"] = nan_count
            pos_inf_count = int(np.isposinf(column_values.to_numpy(dtype=np.float64, copy=False)).sum())
            if pos_inf_count:
                details["pos_inf"] = pos_inf_count
            neg_inf_count = int(np.isneginf(column_values.to_numpy(dtype=np.float64, copy=False)).sum())
            if neg_inf_count:
                details["neg_inf"] = neg_inf_count
            if details:
                examples: list[str] = []
                for value in column_values.iloc[:3]:
                    examples.append(_format_non_finite_value(value))
                details["examples"] = examples
                non_finite_details[column] = details

        before = len(out)
        out = out.loc[finite_mask]
        dropped = before - len(out)
        diagnostics["dropped_non_finite"] = dropped
        if non_finite_details:
            diagnostics["non_finite_details"] = non_finite_details
        message_lines = [
            f"Dropped {dropped:,} participants with non-finite covariates.",
        ]
        if non_finite_details:
            column_summaries: list[str] = []
            for column, details in non_finite_details.items():
                pieces: list[str] = []
                if details.get("nan"):
                    pieces.append(f"NaN={details['nan']}")
                if details.get("pos_inf"):
                    pieces.append(f"+inf={details['pos_inf']}")
                if details.get("neg_inf"):
                    pieces.append(f"-inf={details['neg_inf']}")
                if pieces:
                    column_summaries.append(f"{column} ({', '.join(pieces)})")
            if column_summaries:
                message_lines.append("Problem columns: " + "; ".join(column_summaries))
        warn("\n".join(message_lines))

        if out.empty:
            raise RuntimeError(
                "All participants removed after filtering non-finite covariates."
            )

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


def _log_lines(prefix: str, message: str, *, level: str = "info") -> None:
    for line in message.splitlines():
        text = f"[{prefix}] {line}"
        if level == "info":
            info(text)
        else:
            warn(text)


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
    core: pd.DataFrame,
    anc_dummies: pd.DataFrame,
    reference_ancestry: str | None,
    custom_controls: pd.DataFrame,
    definitions: pd.DataFrame,
    cdr_id: str,
    project_id: str,
    scores_path: Path,
) -> tuple[dict[str, object] | None, list[str]]:
    prefix = f"{cfg.phenotype}"
    _log_lines(prefix, f"Initialising analysis for inversion {inv} (category: {cfg.category})")

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
        _log_lines(prefix, f"Skipping phenotype due to design matrix issue: {exc}", level="warn")
        return None, []

    cdr_codename = cdr_id.split(".")[-1]
    phenotype_status = _load_case_status(
        cfg.phenotype,
        cdr_codename,
        design.index,
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
        fit, fit_method = _fit_logistic(X, y)
    except RuntimeError as exc:
        _log_lines(prefix, f"Logistic regression failed: {exc}", level="warn")
        return None, []

    term_summary = _summarise_term(fit, "dosage")
    beta = float(term_summary.get("beta", math.nan))
    if math.isnan(beta):
        _log_lines(prefix, "Dosage coefficient was not estimated; skipping.", level="warn")
        return None, []
    se = float(term_summary.get("se", math.nan))
    p_value = term_summary.get("p_value", math.nan)
    odds_ratio = term_summary.get("odds_ratio", math.nan)
    ci_lower = term_summary.get("ci_lower", math.nan)
    ci_upper = term_summary.get("ci_upper", math.nan)
    log_p_value = term_summary.get("log_p")
    if log_p_value is None or not math.isfinite(log_p_value):
        log_p_value = math.nan

    baseline_summary = _empty_term_summary()
    baseline_method: str | None = None
    baseline_design = X.drop(columns=custom_cols, errors="ignore")
    try:
        baseline_fit, baseline_method = _fit_logistic(baseline_design, y.loc[baseline_design.index])
        baseline_summary = _summarise_term(baseline_fit, "dosage")
    except RuntimeError as exc:
        _log_lines(
            prefix,
            f"Baseline logistic regression (without custom PGS covariates) failed: {exc}",
            level="warn",
        )

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

    shared_covariates, anc_dummies, reference_ancestry = _load_shared_covariates()
    definitions = pheno.load_definitions(PHENOTYPE_DEFINITIONS_URL)
    target_runs = _resolve_target_runs(definitions)
    if not target_runs:
        warn("No phenotypes selected for analysis; exiting.")
        return

    scores_table, scores_path = _load_scores_table()
    category_controls = _build_category_controls(scores_table)

    results = []
    for inv in TARGET_INVERSIONS:
        info(f"Preparing inversion {inv}")
        inversion_df = _load_inversion(inv)
        core = shared_covariates.join(inversion_df, how="inner")
        core = core.rename(columns={inv: "dosage"}) if inv in core.columns else core

        if "dosage" not in core.columns:
            raise KeyError(f"Inversion column '{inv}' missing after join")

        pending: dict[Future, PhenotypeRun] = {}
        with ThreadPoolExecutor(
            max_workers=max(1, min(len(target_runs), os.cpu_count() or 1))
        ) as executor:
            for cfg in target_runs:
                custom_controls = category_controls.get(cfg.category)
                if custom_controls is None:
                    warn(
                        f"No custom controls found for category '{cfg.category}'; skipping {cfg.phenotype}."
                    )
                    continue

                future = executor.submit(
                    _analyse_single_phenotype,
                    cfg=cfg,
                    inv=inv,
                    core=core,
                    anc_dummies=anc_dummies,
                    reference_ancestry=reference_ancestry,
                    custom_controls=custom_controls,
                    definitions=definitions,
                    cdr_id=cdr_id,
                    project_id=project_id,
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
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, sep="\t", index=False)
    info(f"Wrote {len(output_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":  # pragma: no cover - manual execution only
    run()
