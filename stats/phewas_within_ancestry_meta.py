"""Meta-analyse the ancestry-stratified PheWAS runs and compare the two PC sources.

Reviewer 3 asked for evidence that the reported associations are not driven by subtle
population structure. The pooled model adjusts for the reference-projected components
published with the callset, which separate continental groups but carry almost no
variance inside one of them, so they cannot control fine-scale structure. The answer is
to re-estimate every association inside each ancestry group, once with those global
components and once with components fit within the group, and meta-analyse each arm.

Running *both* arms is the point. If only the fine-scale arm were run, a shifted estimate
could not be attributed to the components rather than to stratification itself, because
stratifying also costs power and changes the model. The two arms differ in exactly one
thing, so their contrast isolates the principal components.

Inputs are the per-run result tables written by the pipeline, whose filenames carry the
population and PC source (``phewas_results_<stamp>_pop-eur_pcs-within-ancestry.tsv``).

    python stats/phewas_within_ancestry_meta.py --results-dir <dir>

Outputs, in data/:
    within_ancestry_meta_by_arm.tsv     per association, per arm: fixed and random effects
    within_ancestry_lambda_by_arm.tsv   per inversion, per arm: genomic control factor
    within_ancestry_arm_comparison.tsv  the contrast that answers the reviewer
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phewas_lambda_gc import INV_LABEL, _num, bh, lambda_gc  # noqa: E402

_STATS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_STATS_DIR)
_DATA = os.path.join(_REPO_DIR, "data")

IN_POOLED = os.path.join(_DATA, "phewas_results.tsv")
OUT_META = os.path.join(_DATA, "within_ancestry_meta_by_arm.tsv")
OUT_LAMBDA = os.path.join(_DATA, "within_ancestry_lambda_by_arm.tsv")
OUT_COMPARE = os.path.join(_DATA, "within_ancestry_arm_comparison.tsv")

ARM_GLOBAL = "global"
ARM_WITHIN = "within-ancestry"

_RESULT_NAME = re.compile(
    r"phewas_results_(?P<stamp>\d+)_pop-(?P<pop>[a-z0-9]+)(?:_pcs-(?P<arm>[a-z-]+))?\.tsv$"
)


def discover_runs(results_dir: str) -> pd.DataFrame:
    """Index the result tables by population and PC source.

    A run with no ``pcs-`` segment used the global components, which is the pipeline
    default and therefore the control arm.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "phewas_results_*.tsv"))):
        match = _RESULT_NAME.search(os.path.basename(path))
        if match is None:
            continue
        rows.append(
            {
                "path": path,
                "population": match.group("pop"),
                "arm": match.group("arm") or ARM_GLOBAL,
                "stamp": match.group("stamp"),
            }
        )
    if not rows:
        raise SystemExit(
            f"No stratified result tables found in {results_dir}. Expected filenames like "
            "phewas_results_<stamp>_pop-eur_pcs-within-ancestry.tsv."
        )
    runs = pd.DataFrame(rows)

    # Re-running a group leaves several stamps behind; keep the newest of each pair so a
    # stale run cannot quietly enter the meta-analysis.
    runs = runs.sort_values("stamp").groupby(["population", "arm"], as_index=False).last()
    return runs


def _standard_error(row) -> float:
    """Standard error of the log odds ratio, from the interval the pipeline reports.

    Preferring the confidence interval over a p-value inversion matters: the interval is
    what the model actually produced, including the profile and Firth cases where the
    Wald approximation the p-value inversion assumes does not hold.
    """
    lo, hi = _num(row.get("CI_LO_OR")), _num(row.get("CI_HI_OR"))
    if np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > 0 and hi > lo:
        return (np.log(hi) - np.log(lo)) / (2 * 1.959963984540054)
    beta, pval = _num(row.get("Beta")), _num(row.get("P_Value"))
    if np.isfinite(beta) and np.isfinite(pval) and 0 < pval < 1 and beta != 0:
        z = stats.norm.isf(pval / 2)
        if z > 0:
            return abs(beta) / z
    return np.nan


def load_arm(runs: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Stack every population's estimates for one arm into a long table."""
    frames = []
    for _, run in runs[runs["arm"] == arm].iterrows():
        table = pd.read_csv(run["path"], sep="\t", low_memory=False)
        missing = {"Phenotype", "Inversion"} - set(table.columns)
        if missing:
            raise SystemExit(f"{run['path']} is missing {sorted(missing)}.")
        beta = table.get("Beta")
        if beta is None:
            beta = np.log(pd.to_numeric(table.get("OR"), errors="coerce"))
        frames.append(
            pd.DataFrame(
                {
                    "Phenotype": table["Phenotype"],
                    "Inversion": table["Inversion"],
                    "population": run["population"],
                    "beta": pd.to_numeric(beta, errors="coerce"),
                    "se": [_standard_error(r) for _, r in table.iterrows()],
                    "p": pd.to_numeric(table.get("P_Value"), errors="coerce"),
                    "n_cases": pd.to_numeric(table.get("N_Cases"), errors="coerce"),
                }
            )
        )
    long = pd.concat(frames, ignore_index=True)
    long["arm"] = arm
    return long[np.isfinite(long["beta"]) & np.isfinite(long["se"]) & (long["se"] > 0)]


def meta_analyse(long: pd.DataFrame) -> pd.DataFrame:
    """Inverse-variance meta-analysis across populations, per association."""
    records = []
    for (inversion, phenotype, arm), group in long.groupby(
        ["Inversion", "Phenotype", "arm"], sort=False
    ):
        beta = group["beta"].to_numpy(float)
        se = group["se"].to_numpy(float)
        k = len(beta)
        if k < 2:
            continue
        w = 1.0 / se**2
        beta_fe = float(np.sum(w * beta) / np.sum(w))
        se_fe = float(np.sqrt(1.0 / np.sum(w)))
        p_fe = 2 * stats.norm.sf(abs(beta_fe / se_fe))

        Q = float(np.sum(w * (beta - beta_fe) ** 2))
        p_Q = float(stats.chi2.sf(Q, k - 1))
        I2 = max(0.0, 100 * (Q - (k - 1)) / Q) if Q > 0 else 0.0

        # Between-group heterogeneity is expected here -- allele frequencies and disease
        # prevalences genuinely differ -- so the random-effects estimate is the honest
        # summary when it is present. DerSimonian-Laird tau-squared.
        c_dl = float(np.sum(w) - np.sum(w**2) / np.sum(w))
        tau2 = max(0.0, (Q - (k - 1)) / c_dl) if c_dl > 0 else 0.0
        w_re = 1.0 / (se**2 + tau2)
        beta_re = float(np.sum(w_re * beta) / np.sum(w_re))
        se_re = float(np.sqrt(1.0 / np.sum(w_re)))
        p_re = 2 * stats.norm.sf(abs(beta_re / se_re))

        records.append(
            {
                "Inversion": inversion,
                "Locus": INV_LABEL.get(inversion, inversion),
                "Phenotype": phenotype,
                "arm": arm,
                "n_populations": k,
                "populations": ",".join(group["population"]),
                "beta_fixed": beta_fe,
                "se_fixed": se_fe,
                "p_fixed": p_fe,
                "or_fixed": float(np.exp(beta_fe)),
                "beta_random": beta_re,
                "se_random": se_re,
                "p_random": p_re,
                "cochran_Q": Q,
                "p_heterogeneity": p_Q,
                "I2": I2,
                "tau2": tau2,
            }
        )
    return pd.DataFrame(records)


def lambda_by_arm(long: pd.DataFrame) -> pd.DataFrame:
    """Genomic control factor per inversion, per population, per arm.

    If the inflation seen in the pooled analysis were structure, adjusting for fine-scale
    components would reduce it. If it is pleiotropy -- one locus genuinely associated with
    a correlated block of phecodes -- it will not move, and that is the reportable result.
    """
    rows = []
    for (inversion, population, arm), group in long.groupby(
        ["Inversion", "population", "arm"], sort=False
    ):
        lam, n = lambda_gc(group["p"])
        rows.append(
            {
                "Inversion": inversion,
                "Locus": INV_LABEL.get(inversion, inversion),
                "population": population,
                "arm": arm,
                "lambda_gc": lam,
                "n_phecodes": n,
            }
        )
    return pd.DataFrame(rows)


def compare_arms(meta: pd.DataFrame, pooled_path: str) -> pd.DataFrame:
    """Contrast the two arms against each other and against the pooled model."""
    wide = meta.pivot_table(
        index=["Inversion", "Locus", "Phenotype"],
        columns="arm",
        values=["beta_fixed", "se_fixed", "p_fixed", "I2"],
    )
    wide.columns = [f"{stat}_{arm}" for stat, arm in wide.columns]
    wide = wide.reset_index()

    for arm in (ARM_GLOBAL, ARM_WITHIN):
        if f"beta_fixed_{arm}" not in wide.columns:
            raise SystemExit(
                f"No results for the '{arm}' arm. Both arms are required: without the "
                "control arm a shifted estimate cannot be attributed to the components."
            )

    if os.path.exists(pooled_path):
        pooled = pd.read_csv(pooled_path, sep="\t", low_memory=False)
        beta_col = "Beta" if "Beta" in pooled.columns else None
        pooled_beta = (
            pd.to_numeric(pooled[beta_col], errors="coerce")
            if beta_col
            else np.log(pd.to_numeric(pooled["OR"], errors="coerce"))
        )
        pooled_small = pd.DataFrame(
            {
                "Inversion": pooled["Inversion"],
                "Phenotype": pooled["Phenotype"],
                "beta_pooled": pooled_beta,
                "p_pooled": pd.to_numeric(pooled.get("P_Value"), errors="coerce"),
            }
        ).dropna(subset=["beta_pooled"])
        wide = wide.merge(pooled_small, on=["Inversion", "Phenotype"], how="left")
    else:
        wide["beta_pooled"] = np.nan
        wide["p_pooled"] = np.nan

    # The headline quantity: how much of the effect survives adjusting for structure the
    # global components cannot see. A ratio near one means stratification is not driving
    # the association.
    with np.errstate(divide="ignore", invalid="ignore"):
        wide["ratio_within_over_pooled"] = wide[f"beta_fixed_{ARM_WITHIN}"] / wide["beta_pooled"]
        wide["ratio_within_over_global_stratified"] = (
            wide[f"beta_fixed_{ARM_WITHIN}"] / wide[f"beta_fixed_{ARM_GLOBAL}"]
        )
    wide["direction_preserved"] = np.sign(wide[f"beta_fixed_{ARM_WITHIN}"]) == np.sign(
        wide["beta_pooled"]
    )
    # A shift of more than one standard error of the control arm is worth a second look.
    wide["shift_in_control_ses"] = (
        wide[f"beta_fixed_{ARM_WITHIN}"] - wide[f"beta_fixed_{ARM_GLOBAL}"]
    ) / wide[f"se_fixed_{ARM_GLOBAL}"]

    finite = np.isfinite(wide[f"p_fixed_{ARM_WITHIN}"])
    wide.loc[finite, "q_within"] = bh(wide.loc[finite, f"p_fixed_{ARM_WITHIN}"].to_numpy())
    return wide.sort_values(f"p_fixed_{ARM_WITHIN}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        default=_DATA,
        help="Directory holding the per-population result tables.",
    )
    parser.add_argument("--pooled", default=IN_POOLED, help="Pooled multi-ancestry results.")
    args = parser.parse_args(argv)

    runs = discover_runs(args.results_dir)
    print("Runs discovered:")
    for _, run in runs.iterrows():
        print(f"  {run['population']:<5} {run['arm']:<16} {os.path.basename(run['path'])}")

    arms = sorted(runs["arm"].unique())
    long = pd.concat([load_arm(runs, arm) for arm in arms], ignore_index=True)

    meta = meta_analyse(long)
    meta.to_csv(OUT_META, sep="\t", index=False)

    lam = lambda_by_arm(long)
    lam.to_csv(OUT_LAMBDA, sep="\t", index=False)

    comparison = compare_arms(meta, args.pooled)
    comparison.to_csv(OUT_COMPARE, sep="\t", index=False)

    kept = comparison[np.isfinite(comparison["ratio_within_over_pooled"])]
    if len(kept):
        print(
            f"\nMedian within-ancestry / pooled effect ratio: "
            f"{np.nanmedian(kept['ratio_within_over_pooled']):.3f} "
            f"({int(kept['direction_preserved'].sum())}/{len(kept)} keep direction)"
        )
    print(f"\nWrote:\n  {OUT_META}\n  {OUT_LAMBDA}\n  {OUT_COMPARE}")


if __name__ == "__main__":
    main()
