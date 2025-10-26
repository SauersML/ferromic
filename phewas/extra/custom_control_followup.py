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
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
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
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError):
        warn("Logit MLE failed – retrying with L2-regularised fit.")
        try:
            result = model.fit_regularized(method="l2", alpha=1e-4, maxiter=200)
            return result, "logit_l2"
        except Exception as exc:
            raise RuntimeError(f"Logistic regression failed: {exc}") from exc


@dataclass
class PhenotypeRun:
    phenotype: str
    category: str


def _load_shared_covariates() -> tuple[pd.DataFrame, pd.DataFrame]:
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
    anc_dummies.index = anc_dummies.index.astype(str)

    return shared, anc_dummies


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

    if not custom_controls.empty:
        aligned_custom = custom_controls.reindex(out.index)
        aligned_custom = aligned_custom.dropna(axis=0, how="any")
        out = out.loc[aligned_custom.index]
        out = pd.concat([out, aligned_custom.astype(np.float32)], axis=1)

    finite_mask = np.isfinite(out.to_numpy(dtype=np.float64)).all(axis=1)
    if not finite_mask.all():
        before = len(out)
        out = out.loc[finite_mask]
        warn(f"Dropped {before - len(out):,} participants with non-finite covariates.")

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

    shared_covariates, anc_dummies = _load_shared_covariates()
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

            design = _build_design_matrix(core, anc_dummies, custom_controls)
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

            try:
                fit, fit_method = _fit_logistic(X, y)
            except RuntimeError as exc:
                warn(f"Logistic regression failed for {cfg.phenotype}: {exc}")
                continue

            params = fit.params
            beta = float(params.get("dosage", np.nan))
            se = float(fit.bse.get("dosage", np.nan)) if hasattr(fit, "bse") else np.nan
            if math.isnan(beta):
                warn(f"Model for {cfg.phenotype} did not estimate a dosage coefficient.")
                continue
            p_value = float(fit.pvalues.get("dosage", np.nan)) if hasattr(fit, "pvalues") else np.nan
            odds_ratio = float(math.exp(beta)) if math.isfinite(beta) else np.nan

            custom_cols = [c for c in X.columns if c.upper().startswith(CUSTOM_CONTROL_PREFIX.upper())]

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
                    "P_Value": p_value,
                    "Fit_Method": fit_method,
                    "Control_File": str(scores_path),
                    "Custom_Covariates": ",".join(custom_cols),
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
