#!/usr/bin/env python3
"""Robust reanalysis of the CDS pair-identity GLM (Reviewer 2 #1).

Why this exists
---------------
``CDS_identical_model.py`` fits a binomial GLM whose outcome is the fraction of
identical haplotype pairs per CDS, weighted by ``freq_weights = n_pairs`` =
C(k, 2). That weight is not the effective sample size:

* ``prop_identical_pairs`` is a U-statistic over k haplotypes. Pairs overlap --
  A-B and A-C share A -- so C(k, 2) overlapping comparisons carry roughly k
  units of information, not C(k, 2). Summed over the data the model believes it
  has 486,275 observations; there are 26 inversions, and the single-event
  coefficient is identified off 7 of them.
* CDSs within one inversion are measured on the *same* haplotypes. In
  17:16823490-18384190, 24 genes rest on the same two inverted sequences.
* ``log_k`` is adjusted for across groups whose support does not overlap
  (median k is 2 in Single/Inverted against 72-78 in Single/Direct), so the
  adjusted Single/Inverted marginal mean is an extrapolation to a mean log_k
  that cell never attains.

Cluster-robust SEs do not repair the first two, and a normal Wald reference
needs dozens of clusters rather than seven.

What this does instead
----------------------
1. reproduces the original weighted binomial GLM, to six digits, so the same
   model is being tested rather than a different one substituted;
2. recalibrates that same GLM by flipping orientation labels for whole
   inversions -- all 2^7 exact assignments -- instead of per CDS;
3. pairs within gene (inverted minus direct), which cancels CDS length, site
   count, inversion length and gene-level constraint exactly, so no covariate
   adjustment and no extrapolation are needed;
4. collapses genes to one value per inversion, the independent unit, and uses
   exact randomisation inference: 2^7 sign flips for the single-event
   orientation effect, all C(26, 7) = 657,800 recurrence-label assignments with
   a Welch-studentised statistic for the interaction;
5. adjusts for the identity difference predicted by local background diversity
   alone -- under a Poisson approximation P(identical) ~ exp(-m*pi), with pi
   taken from the per-base tracks *outside* every analysed transcript span, so
   no CDS adjusts itself;
6. computes power at the real unit count. With 7 inversions the exact sign-flip
   test has a hard floor of 2/128 = 0.0156, so no p below that is reachable
   whatever the effect size.

Outputs (to data/):
  robust_cds_reanalysis_results.tsv
  robust_cds_reanalysis_inversion_effects.tsv
  robust_cds_reanalysis_power.tsv
  robust_cds_reanalysis_report.md
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
ORIENTATION_TERM = "C(phy_group)[T.1]"
INTERACTION_TERM = "C(consensus)[T.1]:C(phy_group)[T.1]"
RNG_SEED = 20260726


# Exact permutation tests count how many rearrangements are "at least as extreme"
# as the observed one. That is a threshold comparison, so a statistic sitting on
# the boundary can fall either side of it depending on floating-point noise -- and
# the GLM refits underneath differ at ~1e-9 between BLAS implementations, so the
# same data gave p = 7/128 on one platform and 8/128 on another. An absolute 1e-12
# tolerance is far tighter than that noise. Scale the tolerance to the statistic
# instead, so a tie is resolved as a tie everywhere.
def _tol(x: float, rel: float = 1e-9) -> float:
    return rel * max(1.0, abs(float(x)))


@dataclass(frozen=True)
class ExactTest:
    estimate: float
    statistic: float
    p_two_sided: float
    p_one_sided_greater: float
    n_permutations: int


def load_matched_data(repo: Path) -> tuple[pd.DataFrame, int]:
    """Load exactly the matched CDS rows the original GLM uses."""
    path = repo / "data" / "cds_identical_proportions.tsv"
    df = pd.read_csv(path, sep="\t")

    numeric = ["consensus", "phy_group", "n_sequences", "n_pairs",
               "n_identical_pairs", "inv_start", "inv_end", "cds_start",
               "cds_end", "n_sites", "inv_exact_match"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["consensus"].isin([0, 1])].copy()
    df["cds_id"] = df["transcript_id"].astype(str)
    df["inv_id"] = df.apply(
        lambda r: f"{r['chr']}:{int(r['inv_start'])}-{int(r['inv_end'])}", axis=1)
    df = df[(df["n_pairs"] > 0) & df["n_identical_pairs"].notna()
            & (df["n_sequences"] >= 2)].copy()

    orient = (df.groupby(["inv_id", "cds_id", "consensus"])["phy_group"]
                .nunique().rename("n_orientations").reset_index())
    valid = orient.loc[orient["n_orientations"] == 2,
                       ["inv_id", "cds_id", "consensus"]]
    before = len(df)
    df = df.merge(valid.assign(_keep=1), on=["inv_id", "cds_id", "consensus"])
    removed = before - len(df)
    df.drop(columns="_keep", inplace=True)

    df["y"] = df["n_identical_pairs"].astype(int)
    df["n"] = df["n_pairs"].astype(int)
    df["prop"] = df["y"] / df["n"]
    df["inv_len"] = (df["inv_end"] - df["inv_start"]).abs() + 1
    df["log_m"] = np.log(df["n_sites"].astype(float))
    df["log_L"] = np.log(df["inv_len"].astype(float))
    df["log_k"] = np.log(df["n_sequences"].astype(float))

    sizes = df.groupby(["inv_id", "cds_id", "consensus"]).size()
    if not (sizes == 2).all():
        raise ValueError("Matched strata do not contain exactly two orientation rows.")
    sums = df.groupby(["inv_id", "cds_id", "consensus"])["phy_group"].sum()
    if not (sums == 1).all():
        raise ValueError("Matched strata do not contain one row per orientation.")
    return df, removed


def fit_original_glm(df: pd.DataFrame, adjusted: bool):
    formula = "prop ~ C(consensus) * C(phy_group)"
    if adjusted:
        formula += " + log_m + log_L + log_k"
    model = smf.glm(formula, data=df, family=sm.families.Binomial(),
                    freq_weights=df["n"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(cov_type="cluster", cov_kwds={"groups": df["inv_id"]})


def exact_glm_single_block_permutation(df: pd.DataFrame, adjusted: bool) -> ExactTest:
    """Exact 2^7 block recalibration of the original single-event coefficient.

    Every CDS label inside a single-event inversion flips together, because
    orientation is a property of the haplotype set, not of the gene. The
    studentised Wald z is permuted rather than the raw coefficient, since the
    inversion blocks are strongly heteroskedastic.
    """
    observed = fit_original_glm(df, adjusted)
    beta_obs = float(observed.params[ORIENTATION_TERM])
    z_obs = float(beta_obs / observed.bse[ORIENTATION_TERM])
    single_ids = sorted(df.loc[df["consensus"] == 0, "inv_id"].unique())

    z_null: list[float] = []
    for flips in itertools.product((0, 1), repeat=len(single_ids)):
        perm = df.copy()
        for inv_id, flip in zip(single_ids, flips):
            if flip:
                mask = (perm["consensus"] == 0) & (perm["inv_id"] == inv_id)
                perm.loc[mask, "phy_group"] = 1 - perm.loc[mask, "phy_group"]
        res = fit_original_glm(perm, adjusted)
        z_null.append(float(res.params[ORIENTATION_TERM] / res.bse[ORIENTATION_TERM]))

    z = np.asarray(z_null)
    tol = _tol(z_obs)
    return ExactTest(beta_obs, z_obs,
                     float(np.mean(np.abs(z) >= abs(z_obs) - tol)),
                     float(np.mean(z >= z_obs - tol)), len(z))


def build_paired_cds(df: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["inv_id", "cds_id", "gene_name", "consensus", "chr",
                  "inv_start", "inv_end", "cds_start", "cds_end", "n_sites",
                  "inv_len"]
    paired = df.pivot_table(index=index_cols, columns="phy_group",
                            values=["prop", "n_sequences", "n"], aggfunc="first")
    paired.columns = [f"{a}_{b}" for a, b in paired.columns]
    paired = paired.reset_index()
    paired["delta"] = paired["prop_1"] - paired["prop_0"]
    paired["log_harmonic_k"] = np.log(
        2.0 / (1.0 / paired["n_sequences_0"] + 1.0 / paired["n_sequences_1"]))
    return paired


def inversion_summary(paired: pd.DataFrame) -> pd.DataFrame:
    return (paired.groupby(["inv_id", "consensus"], as_index=False)
            .agg(delta=("delta", "mean"), median_delta=("delta", "median"),
                 n_cds=("delta", "size"), inv_len=("inv_len", "first"),
                 k_direct_median=("n_sequences_0", "median"),
                 k_inverted_median=("n_sequences_1", "median"))
            .sort_values(["consensus", "inv_id"]).reset_index(drop=True))


def exact_sign_flip(values: Sequence[float]) -> ExactTest:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("Sign-flip test requires a non-empty vector.")
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else math.nan
    t_obs = mean / se if se > 0 else math.inf
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(x))))
    perm = signs * x[None, :]
    means = perm.mean(axis=1)
    ses = perm.std(axis=1, ddof=1) / math.sqrt(len(x))
    t = np.divide(means, ses, out=np.sign(means) * np.full_like(means, np.inf),
                  where=ses > 0)
    tol = _tol(t_obs)
    return ExactTest(mean, t_obs, float(np.mean(np.abs(t) >= abs(t_obs) - tol)),
                     float(np.mean(t >= t_obs - tol)), len(t))


def exact_studentized_group_permutation(values, group, target_group=0) -> ExactTest:
    """Exact Welch-studentised permutation over all recurrence-label assignments."""
    x = np.asarray(values, dtype=float)
    g = np.asarray(group, dtype=int)
    a, b = x[g == target_group], x[g != target_group]
    n_a, n_b = len(a), len(b)
    obs = float(a.mean() - b.mean())
    se_obs = math.sqrt(a.var(ddof=1) / n_a + b.var(ddof=1) / n_b)
    t_obs = obs / se_obs

    total, total_ss = float(x.sum()), float(np.dot(x, x))
    two = one = n_perm = 0
    tol = _tol(t_obs)
    for inds in itertools.combinations(range(len(x)), n_a):
        sel = x[list(inds)]
        sa, ssa = float(sel.sum()), float(np.dot(sel, sel))
        sb, ssb = total - sa, total_ss - ssa
        va = max((ssa - sa * sa / n_a) / (n_a - 1), 0.0)
        vb = max((ssb - sb * sb / n_b) / (n_b - 1), 0.0)
        se = math.sqrt(va / n_a + vb / n_b)
        t = (sa / n_a - sb / n_b) / se if se > 0 else 0.0
        two += abs(t) >= abs(t_obs) - tol
        one += t >= t_obs - tol
        n_perm += 1
    return ExactTest(obs, t_obs, two / n_perm, one / n_perm, n_perm)


def exact_nested_block_sign_flip(single_inv: pd.DataFrame) -> ExactTest:
    """Treat the two nested chr2 inversion records as one biological block."""
    x = single_inv.set_index("inv_id")["delta"]
    nested = {"2:130138212-131200602", "2:130138212-131530534"}
    if not nested.issubset(set(x.index)):
        raise ValueError("Expected nested chr2 inversion pair is missing.")
    blocks = [[i] for i in x.index if i not in nested] + [sorted(nested)]
    mean = float(x.mean())
    t_obs = float(mean / (x.std(ddof=1) / math.sqrt(len(x))))
    t_null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(blocks)):
        z = x.copy()
        for block, sign in zip(blocks, signs):
            z.loc[block] *= sign
        se = float(z.std(ddof=1) / math.sqrt(len(z)))
        t_null.append(float(z.mean() / se) if se > 0 else 0.0)
    arr = np.asarray(t_null)
    tol = _tol(t_obs)
    return ExactTest(mean, t_obs, float(np.mean(np.abs(arr) >= abs(t_obs) - tol)),
                     float(np.mean(arr >= t_obs - tol)), len(arr))


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(a, b) for a, b in merged]


def background_pi_outside_transcript_spans(repo: Path, paired: pd.DataFrame) -> pd.DataFrame:
    """Filtered pi outside every analysed transcript span.

    The whole [cds_start, cds_end] span is removed, introns included, so no CDS
    contributes to the background used to adjust it.
    """
    regions: dict[tuple[str, int, int], tuple[str, np.ndarray, float]] = {}
    for inv_id, sub in paired.groupby("inv_id"):
        chrom = str(sub["chr"].iloc[0])
        start, end = int(sub["inv_start"].iloc[0]), int(sub["inv_end"].iloc[0])
        mask = np.ones(end - start + 1, dtype=bool)
        spans = [(max(start, int(a)), min(end, int(b)))
                 for a, b in sub[["cds_start", "cds_end"]].drop_duplicates().itertuples(index=False)]
        for a, b in merge_intervals([(a, b) for a, b in spans if a <= b]):
            mask[a - start: b - start + 1] = False
        if not mask.any():
            raise ValueError(f"No background bases remain for {inv_id}.")
        regions[(chrom, start, end)] = (inv_id, mask, float(mask.mean()))

    falsta = repo / "data" / "per_site_diversity_output.falsta.gz"
    if not falsta.exists():
        raise FileNotFoundError(f"Missing background-diversity file: {falsta}")
    pattern = re.compile(r"^>filtered_pi_chr_(.+)_start_(\d+)_end_(\d+)_group_([01])$")
    rows: dict[tuple[str, int], dict[str, float]] = {}
    with gzip.open(falsta, "rt") as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            values_line = handle.readline()
            if not values_line:
                raise ValueError("Malformed FALSTA: header without values line.")
            m = pattern.match(header.strip())
            if not m:
                continue
            key = (m.group(1), int(m.group(2)), int(m.group(3)))
            if key not in regions:
                continue
            inv_id, mask, frac = regions[key]
            values = np.fromstring(values_line.strip(), sep=",")
            if len(values) != len(mask):
                raise ValueError(
                    f"Length mismatch for {inv_id}: {len(values)} values vs {len(mask)} bases.")
            rows[(inv_id, int(m.group(4)))] = {
                "pi_background": float(np.nanmean(values[mask])),
                "pi_whole_track": float(np.nanmean(values)),
                "background_fraction": frac,
            }

    expected = {(i, g) for i in paired["inv_id"].unique() for g in (0, 1)}
    missing = expected - set(rows)
    if missing:
        raise ValueError(f"Missing {len(missing)} filtered pi tracks: {sorted(missing)[:5]}")

    return pd.DataFrame([{
        "inv_id": inv_id,
        "pi_direct_background": rows[(inv_id, 0)]["pi_background"],
        "pi_inverted_background": rows[(inv_id, 1)]["pi_background"],
        "pi_direct_whole_track": rows[(inv_id, 0)]["pi_whole_track"],
        "pi_inverted_whole_track": rows[(inv_id, 1)]["pi_whole_track"],
        "background_fraction": rows[(inv_id, 0)]["background_fraction"],
    } for inv_id in sorted(paired["inv_id"].unique())])


def add_background_predictor(paired: pd.DataFrame, background: pd.DataFrame):
    p = paired.merge(background, on="inv_id", how="left", validate="many_to_one")
    # Poisson approximation: P(two sequences identical over m sites) ~ exp(-m*pi).
    p["q_direct_bg"] = np.exp(-p["n_sites"] * p["pi_direct_background"])
    p["q_inverted_bg"] = np.exp(-p["n_sites"] * p["pi_inverted_background"])
    p["qdelta_bg"] = p["q_inverted_bg"] - p["q_direct_bg"]
    p["q_direct_whole"] = np.exp(-p["n_sites"] * p["pi_direct_whole_track"])
    p["q_inverted_whole"] = np.exp(-p["n_sites"] * p["pi_inverted_whole_track"])
    p["qdelta_whole"] = p["q_inverted_whole"] - p["q_direct_whole"]

    inv = (p.groupby(["inv_id", "consensus"], as_index=False)
           .agg(delta=("delta", "mean"), qdelta_bg=("qdelta_bg", "mean"),
                qdelta_whole=("qdelta_whole", "mean"), n_cds=("delta", "size"),
                inv_len=("inv_len", "first"),
                k_direct_median=("n_sequences_0", "median"),
                k_inverted_median=("n_sequences_1", "median"),
                background_fraction=("background_fraction", "first"),
                pi_direct_background=("pi_direct_background", "first"),
                pi_inverted_background=("pi_inverted_background", "first"))
           .sort_values(["consensus", "inv_id"]).reset_index(drop=True))
    inv["single"] = (inv["consensus"] == 0).astype(float)
    inv["recurrent"] = (inv["consensus"] == 1).astype(float)
    inv["log_inv_len_c"] = np.log(inv["inv_len"]) - np.log(inv["inv_len"]).mean()
    hk = 2.0 / (1.0 / inv["k_direct_median"] + 1.0 / inv["k_inverted_median"])
    inv["log_harmonic_k_c"] = np.log(hk) - np.log(hk).mean()
    return p, inv


def restricted_wild_bootstrap_hc3(y, x, coefficient: int, n_boot=200_000,
                                  seed=RNG_SEED) -> dict[str, float]:
    """Restricted Rademacher wild bootstrap-t with HC3 studentisation."""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    n, p = x_arr.shape
    keep = [j for j in range(p) if j != coefficient]
    xr = x_arr[:, keep]

    inv_r = np.linalg.inv(xr.T @ xr)
    a_r = inv_r @ xr.T
    fitted_r = xr @ (a_r @ y_arr)
    lev_r = np.einsum("ij,ji->i", xr, inv_r @ xr.T)
    u_r = (y_arr - fitted_r) / (1.0 - lev_r)

    inv_full = np.linalg.inv(x_arr.T @ x_arr)
    a_full = inv_full @ x_arr.T
    beta_obs = a_full @ y_arr
    resid = y_arr - x_arr @ beta_obs
    lev = np.einsum("ij,ji->i", x_arr, inv_full @ x_arr.T)
    se_obs = math.sqrt(float(np.sum((a_full[coefficient, :] * resid / (1.0 - lev)) ** 2)))
    t_obs = float(beta_obs[coefficient] / se_obs)

    rng = np.random.default_rng(seed)
    two = one = done = 0
    while done < n_boot:
        size = min(10_000, n_boot - done)
        w = rng.choice((-1.0, 1.0), size=(n, size))
        y_star = fitted_r[:, None] + u_r[:, None] * w
        beta_star = a_full @ y_star
        resid_star = y_star - x_arr @ beta_star
        se_star = np.sqrt(np.sum((a_full[coefficient, :, None] * resid_star
                                  / (1.0 - lev)[:, None]) ** 2, axis=0))
        t_star = beta_star[coefficient, :] / se_star
        two += int(np.sum(np.abs(t_star) >= abs(t_obs) - _tol(t_obs)))
        one += int(np.sum(t_star >= t_obs - _tol(t_obs)))
        done += size

    return {"estimate": float(beta_obs[coefficient]), "se_hc3": se_obs,
            "t_hc3": t_obs, "p_two_sided": (two + 1.0) / (done + 1.0),
            "p_one_sided_greater": (one + 1.0) / (done + 1.0), "n_boot": float(done)}


def high_k_sensitivity(paired: pd.DataFrame, minimum_k: int = 3):
    kept = paired[(paired["n_sequences_0"] >= minimum_k)
                  & (paired["n_sequences_1"] >= minimum_k)].copy()
    inv = inversion_summary(kept)
    return inv, exact_sign_flip(inv.loc[inv["consensus"] == 0, "delta"].to_numpy())


def exact_signflip_power(n, sd, effects, n_sim=200_000, seed=RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_sim, n))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)))
    rows = []
    for effect in effects:
        samples = effect + sd * z
        observed = samples.mean(axis=1)
        rej_two = rej_one = 0
        for start in range(0, n_sim, 5_000):
            stop = min(start + 5_000, n_sim)
            pm = samples[start:stop] @ signs.T / n
            obs_chunk = observed[start:stop, None]
            tol = 1e-9 * np.maximum(1.0, np.abs(obs_chunk))
            rej_two += int(np.sum(np.mean(np.abs(pm) >= np.abs(obs_chunk) - tol,
                                          axis=1) <= 0.05))
            rej_one += int(np.sum(np.mean(pm >= obs_chunk - tol,
                                          axis=1) <= 0.05))
        rows.append({"true_effect_probability_points": effect,
                     "power_two_sided": rej_two / n_sim,
                     "power_one_sided": rej_one / n_sim})
    return pd.DataFrame(rows)


def t_confidence_interval(values, alpha=0.05) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(len(x)))
    crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=len(x) - 1))
    return mean - crit * se, mean + crit * se


def add_result(rows, *, method, hypothesis, estimate=None, scale="",
               statistic=None, p_two=None, p_one=None, n_units=None, notes=""):
    rows.append({"method": method, "hypothesis": hypothesis, "estimate": estimate,
                 "scale": scale, "statistic": statistic, "p_two_sided": p_two,
                 "p_one_sided_greater": p_one, "n_independent_units": n_units,
                 "notes": notes})


def fmt_p(value) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.2e}" if value < 0.0001 else f"{value:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=str(REPO_ROOT))
    ap.add_argument("--outdir", default=str(REPO_ROOT / "data"))
    ap.add_argument("--bootstrap", type=int, default=200_000)
    ap.add_argument("--power-sim", type=int, default=200_000)
    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df, removed = load_matched_data(repo)
    paired = build_paired_cds(df)
    inv_raw = inversion_summary(paired)
    single_values = inv_raw.loc[inv_raw["consensus"] == 0, "delta"].to_numpy()
    recurrent_values = inv_raw.loc[inv_raw["consensus"] == 1, "delta"].to_numpy()

    results: list[dict] = []

    original = fit_original_glm(df, adjusted=True)
    for term, label in ((ORIENTATION_TERM, "Inverted vs Direct within single-event inversions"),
                        (INTERACTION_TERM, "Recurrence x orientation interaction")):
        add_result(results, method="Original weighted binomial GLM (reproduced)",
                   hypothesis=label, estimate=float(original.params[term]),
                   scale="log-odds coefficient",
                   statistic=float(original.params[term] / original.bse[term]),
                   p_two=float(original.pvalues[term]),
                   n_units=int(df["inv_id"].nunique()),
                   notes="freq_weights=C(k,2); normal Wald reference; pair dependence ignored")

    glm_block = exact_glm_single_block_permutation(df, adjusted=True)
    add_result(results, method="Same adjusted GLM; exact inversion-block permutation-t",
               hypothesis="Inverted vs Direct within single-event inversions",
               estimate=glm_block.estimate, scale="log-odds coefficient",
               statistic=glm_block.statistic, p_two=glm_block.p_two_sided,
               p_one=glm_block.p_one_sided_greater, n_units=7,
               notes=f"All CDS labels flipped together within each inversion; {glm_block.n_permutations} exact assignments")

    glm_block_nocov = exact_glm_single_block_permutation(df, adjusted=False)
    add_result(results, method="Unadjusted GLM; exact inversion-block permutation-t",
               hypothesis="Inverted vs Direct within single-event inversions",
               estimate=glm_block_nocov.estimate, scale="log-odds coefficient",
               statistic=glm_block_nocov.statistic, p_two=glm_block_nocov.p_two_sided,
               p_one=glm_block_nocov.p_one_sided_greater, n_units=7,
               notes=f"Sensitivity without log CDS length, inversion length or haplotype count; {glm_block_nocov.n_permutations} assignments")

    single_test = exact_sign_flip(single_values)
    recurrent_test = exact_sign_flip(recurrent_values)
    interaction_test = exact_studentized_group_permutation(
        inv_raw["delta"], inv_raw["consensus"], target_group=0)
    single_ci = t_confidence_interval(single_values)
    recurrent_ci = t_confidence_interval(recurrent_values)

    add_result(results, method="Primary: paired inversion-level mean; exact sign-flip t",
               hypothesis="Inverted vs Direct within single-event inversions",
               estimate=single_test.estimate,
               scale="probability difference (Inverted - Direct)",
               statistic=single_test.statistic, p_two=single_test.p_two_sided,
               p_one=single_test.p_one_sided_greater, n_units=len(single_values),
               notes=f"Each inversion equal weight; 95% t CI [{single_ci[0]:.6f}, {single_ci[1]:.6f}]; exact floor is 2/128 = 0.0156")
    add_result(results, method="Paired inversion-level mean; exact sign-flip t",
               hypothesis="Inverted vs Direct within recurrent inversions",
               estimate=recurrent_test.estimate,
               scale="probability difference (Inverted - Direct)",
               statistic=recurrent_test.statistic, p_two=recurrent_test.p_two_sided,
               p_one=recurrent_test.p_one_sided_greater, n_units=len(recurrent_values),
               notes=f"Each inversion equal weight; 95% t CI [{recurrent_ci[0]:.6f}, {recurrent_ci[1]:.6f}]")
    add_result(results, method="Exact Welch-studentised recurrence-label permutation",
               hypothesis="Single-event minus recurrent orientation effect",
               estimate=interaction_test.estimate,
               scale="difference in probability differences",
               statistic=interaction_test.statistic, p_two=interaction_test.p_two_sided,
               p_one=interaction_test.p_one_sided_greater, n_units=len(inv_raw),
               notes=f"All C(26,7)={interaction_test.n_permutations} assignments enumerated")

    nested_test = exact_nested_block_sign_flip(inv_raw[inv_raw["consensus"] == 0])
    add_result(results, method="Nested-locus block sensitivity",
               hypothesis="Inverted vs Direct within single-event inversions",
               estimate=nested_test.estimate, scale="probability difference",
               statistic=nested_test.statistic, p_two=nested_test.p_two_sided,
               p_one=nested_test.p_one_sided_greater, n_units=6,
               notes="The two overlapping chr2 inversion records share one sign-flip block")

    high_k_inv, high_k_test = high_k_sensitivity(paired, minimum_k=3)
    add_result(results, method="Haplotype-count sensitivity (k>=3 in both orientations)",
               hypothesis="Inverted vs Direct within single-event inversions",
               estimate=high_k_test.estimate, scale="probability difference",
               statistic=high_k_test.statistic, p_two=high_k_test.p_two_sided,
               p_one=high_k_test.p_one_sided_greater,
               n_units=int((high_k_inv["consensus"] == 0).sum()),
               notes="Avoids cells resting on a single inverted pair; few single-event inversions remain")

    background = background_pi_outside_transcript_spans(repo, paired)
    _paired_bg, inv_bg = add_background_predictor(paired, background)
    corr_p = stats.pearsonr(inv_bg["delta"], inv_bg["qdelta_bg"])
    corr_s = stats.spearmanr(inv_bg["delta"], inv_bg["qdelta_bg"])

    bg_boot = restricted_wild_bootstrap_hc3(
        inv_bg["delta"], inv_bg[["single", "recurrent", "qdelta_bg"]],
        coefficient=0, n_boot=args.bootstrap)
    add_result(results,
               method="Background-diversity-adjusted inversion OLS; restricted HC3 wild bootstrap-t",
               hypothesis="Residual single-event CDS identity effect after local background diversity",
               estimate=bg_boot["estimate"], scale="residual probability difference",
               statistic=bg_boot["t_hc3"], p_two=bg_boot["p_two_sided"],
               p_one=bg_boot["p_one_sided_greater"], n_units=len(inv_bg),
               notes=("Predictor is mean exp(-m*pi_inverted)-exp(-m*pi_direct) with filtered pi "
                      "computed outside analysed transcript spans; "
                      f"Pearson r={corr_p.statistic:.3f}, Spearman rho={corr_s.statistic:.3f}"))

    bg_arch_boot = restricted_wild_bootstrap_hc3(
        inv_bg["delta"],
        inv_bg[["single", "recurrent", "qdelta_bg", "log_inv_len_c", "log_harmonic_k_c"]],
        coefficient=0, n_boot=args.bootstrap, seed=RNG_SEED + 1)
    add_result(results,
               method="Background + architecture-adjusted inversion OLS; restricted HC3 wild bootstrap-t",
               hypothesis="Residual single-event CDS identity effect",
               estimate=bg_arch_boot["estimate"], scale="residual probability difference",
               statistic=bg_arch_boot["t_hc3"], p_two=bg_arch_boot["p_two_sided"],
               p_one=bg_arch_boot["p_one_sided_greater"], n_units=len(inv_bg),
               notes="Also adjusts for log inversion length and log harmonic haplotype count")

    sd_single = float(np.std(single_values, ddof=1))
    power = exact_signflip_power(
        n=len(single_values), sd=sd_single,
        effects=[0.05, float(single_test.estimate), 0.10, 0.125, 0.135, 0.14, 0.15, 0.20],
        n_sim=args.power_sim)
    observed_power = power.iloc[int(np.argmin(
        np.abs(power["true_effect_probability_points"] - single_test.estimate)))]
    above80 = power[power["power_two_sided"] >= 0.80]
    mde80 = float(above80.iloc[0]["true_effect_probability_points"]) if len(above80) else math.nan

    pd.DataFrame(results).to_csv(outdir / "robust_cds_reanalysis_results.tsv",
                                 sep="\t", index=False)
    inv_bg.merge(inv_raw[["inv_id", "median_delta"]], on="inv_id", how="left",
                 validate="one_to_one").to_csv(
        outdir / "robust_cds_reanalysis_inversion_effects.tsv", sep="\t", index=False)
    power.to_csv(outdir / "robust_cds_reanalysis_power.tsv", sep="\t", index=False)

    n_inv_cells = int(((df.consensus == 0) & (df.phy_group == 1)).sum())
    n_ceiling = int(((df.consensus == 0) & (df.phy_group == 1) & (df.prop == 1)).sum())
    report = f"""# Robust reanalysis of CDS sequence identity

## Data structure

- Matched input: **{len(df)} orientation rows = {len(paired)} paired CDS strata** from **{df['inv_id'].nunique()} inversions**.
- Single-event: **{int((inv_raw['consensus'] == 0).sum())} inversions / {int((paired['consensus'] == 0).sum())} paired CDSs**. Recurrent: **{int((inv_raw['consensus'] == 1).sum())} inversions / {int((paired['consensus'] == 1).sum())} paired CDSs**.
- Rows dropped because the opposite orientation was unavailable: **{removed}**.
- Nominal N implied by `freq_weights=n_pairs`: **{int(df['n'].sum()):,}**. Independent units: **{df['inv_id'].nunique()}**.
- Single-event inverted cells: median haplotype count **{df.loc[(df.consensus == 0) & (df.phy_group == 1), 'n_sequences'].median():.0f}**; **{n_ceiling}/{n_inv_cells}** CDS estimates equal exactly 1.0.

## Results

1. **Original adjusted GLM reproduced:** single-event orientation Wald p = **{fmt_p(float(original.pvalues[ORIENTATION_TERM]))}**; interaction p = **{fmt_p(float(original.pvalues[INTERACTION_TERM]))}**.
2. **Same GLM, exact inversion-block calibration:** two-sided p = **{fmt_p(glm_block.p_two_sided)}** (directional {fmt_p(glm_block.p_one_sided_greater)}); unadjusted {fmt_p(glm_block_nocov.p_two_sided)}.
3. **Primary paired inversion-level analysis:** single-event mean difference = **{100 * single_test.estimate:.2f} percentage points**, exact two-sided p = **{fmt_p(single_test.p_two_sided)}** (directional {fmt_p(single_test.p_one_sided_greater)}), 95% t CI **{100 * single_ci[0]:.2f} to {100 * single_ci[1]:.2f}**.
4. **Recurrent inversions:** **{100 * recurrent_test.estimate:.2f} points**, p = **{fmt_p(recurrent_test.p_two_sided)}**.
5. **Recurrence interaction:** **{100 * interaction_test.estimate:.2f} points**, exact studentised permutation p = **{fmt_p(interaction_test.p_two_sided)}**.
6. **After background-diversity adjustment:** residual **{100 * bg_boot['estimate']:.2f} points**, p = **{fmt_p(bg_boot['p_two_sided'])}**; adding length and haplotype count, **{100 * bg_arch_boot['estimate']:.2f} points**, p = **{fmt_p(bg_arch_boot['p_two_sided'])}**.
7. **k>=3 sensitivity:** **{int((high_k_inv['consensus'] == 0).sum())}** single-event inversions remain, p = **{fmt_p(high_k_test.p_two_sided)}**.
8. **Nested chr2 loci as one block:** p = **{fmt_p(nested_test.p_two_sided)}**.

## Power

Seven single-event blocks, between-inversion SD {sd_single:.3f}, exact sign-flip rule:
power for the observed {100 * single_test.estimate:.2f}-point effect is **{100 * float(observed_power['power_two_sided']):.1f}% two-sided**
({100 * float(observed_power['power_one_sided']):.1f}% directional); about **{100 * mde80:.1f} points** are needed for 80%.
The exact test cannot return p below 2/128 = 0.0156 at this unit count, whatever the effect size.

## Interpretation

The raw direction is compatible with higher CDS identity among single-event inverted
haplotypes, but the evidence is not robustly two-sided significant once the inversion is
the independent unit, and there is no support for a recurrence-by-orientation
interaction. The observed difference tracks what local background diversity alone
predicts (Pearson r = {corr_p.statistic:.3f}, p = {fmt_p(corr_p.pvalue)}; Spearman rho = {corr_s.statistic:.3f}, p = {fmt_p(corr_s.pvalue)}),
and the residual CDS-specific effect is null after that adjustment. The defensible
conclusion is descriptive: single-event inverted haplotypes show higher raw CDS pair
identity here, consistent with their lower background diversity rather than with a
CDS-specific conservation or selection effect.
"""
    (outdir / "robust_cds_reanalysis_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
