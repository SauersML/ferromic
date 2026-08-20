"""What does the classifier's error rate do to the recurrence contrast?

The gene-flux sweep shows that at realistic human recombination rates the
reference recurrence classifier calls a single-origin locus recurrent a
substantial fraction of the time, and that the rate depends strongly on how
recently the arrangements diverged. The obvious worry is that this manufactures
the recurrent-versus-single-event differences the manuscript reports.

It does the opposite, and this script quantifies by how much. Misclassifying a
single-event locus as recurrent moves a low-diversity-inverted locus into the
recurrent group, which drags the recurrent group's mean toward the single-event
group's mean and shrinks the contrast. The observed interaction is therefore an
underestimate of the true one, and the error rate sets how large the
underestimate is.

Given an observed group mean in each class and a false-positive rate f (the
probability that a truly single-origin locus is labelled recurrent), the labelled
recurrent group is a mixture:

    mean_labelled_recurrent = (1 - w) * mean_true_recurrent + w * mean_true_single

where w is the share of the labelled recurrent group that is actually
single-origin, obtained from f and the class sizes. Solving for
mean_true_recurrent gives the deattenuated contrast. The single-event group is
also contaminated in principle, but in the other direction the classifier's error
is the false-negative rate, which the sweep measures as 1 - power; both are
propagated here.

A parametric bootstrap over loci gives intervals, and the whole curve is traced
across the plausible range of f so the conclusion does not rest on one number.

Inputs:  data/recurrence_controls_covariates.tsv   (per-locus pi by orientation)
         simulations/refsim/fluxsweep2_results.csv (or _partial_)
Outputs: data/misclassification_attenuation.tsv
         data/misclassification_attenuation.pdf / .png
"""

import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_STATS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_STATS)
_DATA = os.path.join(_REPO, "data")
_SIM = os.path.join(_REPO, "simulations", "refsim")

COV = os.path.join(_DATA, "recurrence_controls_covariates.tsv")
SWEEP_CANDIDATES = [os.path.join(_SIM, "fluxsweep2_results.csv"),
                    os.path.join(_SIM, "fluxsweep2_partial_results.csv")]
OUT_TSV = os.path.join(_DATA, "misclassification_attenuation.tsv")
OUT_PDF = os.path.join(_DATA, "misclassification_attenuation.pdf")
OUT_PNG = os.path.join(_DATA, "misclassification_attenuation.png")

N_BOOT = 5000
RNG_SEED = 2026
EPS_Q = 0.01

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def sweep_rates(path, rho_target=1e-8):
    """(false-positive rate, power) at the recombination rate of the real loci."""
    sw = pd.read_csv(path)
    single = sw[sw["scenario"].str.startswith("single")]
    rec = sw[~sw["scenario"].str.startswith("single")]
    rhos = sorted(sw["rho"].unique())
    rho = min(rhos, key=lambda r: abs(np.log10(max(r, 1e-12))
                                      - np.log10(rho_target)))
    s = single[single["rho"] == rho]
    r = rec[rec["rho"] == rho]
    fpr = s["n_called"].sum() / s["reps"].sum()
    power = r["n_called"].sum() / r["reps"].sum()
    return rho, float(fpr), float(power), int(s["reps"].sum()), int(r["reps"].sum())


def deattenuate(delta_lab_rec, delta_lab_sin, n_rec, n_sin, fpr, power):
    """True group means given labelling error rates.

    Let p be the true proportion of loci that are recurrent. Of the labelled
    recurrent loci, a share w_r are truly single-origin; of the labelled
    single-event loci, a share w_s are truly recurrent. Both follow from the
    confusion matrix implied by (fpr, power) and the observed label counts.
    """
    n = n_rec + n_sin
    # Solve for the true recurrent proportion p from the labelled counts:
    #   n_rec / n = p * power + (1 - p) * fpr
    obs = n_rec / n
    denom = power - fpr
    if abs(denom) < 1e-9:
        return np.nan, np.nan, np.nan
    p = (obs - fpr) / denom
    p = float(np.clip(p, 0.0, 1.0))
    # confusion shares
    lab_rec = p * power + (1 - p) * fpr
    lab_sin = p * (1 - power) + (1 - p) * (1 - fpr)
    if lab_rec <= 0 or lab_sin <= 0:
        return np.nan, np.nan, np.nan
    w_r = (1 - p) * fpr / lab_rec           # labelled recurrent, truly single
    w_s = p * (1 - power) / lab_sin         # labelled single, truly recurrent
    # Invert the 2x2 mixture:
    #   delta_lab_rec = (1 - w_r) * T_rec + w_r * T_sin
    #   delta_lab_sin = w_s      * T_rec + (1 - w_s) * T_sin
    A = np.array([[1 - w_r, w_r], [w_s, 1 - w_s]])
    if abs(np.linalg.det(A)) < 1e-9:
        return np.nan, np.nan, np.nan
    t_rec, t_sin = np.linalg.solve(A, np.array([delta_lab_rec, delta_lab_sin]))
    return float(t_rec), float(t_sin), p


def main():
    sweep_path = next((p for p in SWEEP_CANDIDATES if os.path.exists(p)), None)
    if sweep_path is None or not os.path.exists(COV):
        sys.exit("need the sweep results and recurrence_controls_covariates.tsv")

    rho, fpr, power, n_s, n_r = sweep_rates(sweep_path)
    print(f"sweep: {os.path.basename(sweep_path)}")
    print(f"at rho = {rho:.0e} (the rate of the real loci): "
          f"false-positive rate = {fpr:.3f} ({n_s} replicates), "
          f"power = {power:.3f} ({n_r} replicates)")

    cov = pd.read_csv(COV, sep="\t")
    cov = cov[np.isfinite(cov["pi_direct"]) & np.isfinite(cov["pi_inverted"])]
    all_pi = np.r_[cov["pi_direct"], cov["pi_inverted"]]
    eps = float(np.quantile(all_pi[all_pi > 0], EPS_Q)) if (all_pi > 0).any() else 1e-6
    cov["delta"] = (np.log(cov["pi_inverted"] + eps)
                    - np.log(cov["pi_direct"] + eps))
    rec = cov.loc[cov["Recurrence"] == "Recurrent", "delta"].to_numpy()
    sin = cov.loc[cov["Recurrence"] == "Single-event", "delta"].to_numpy()
    print(f"\n{len(rec)} recurrent and {len(sin)} single-event loci; "
          f"detection floor eps = {eps:.3g}")

    d_rec, d_sin = rec.mean(), sin.mean()
    obs_int = d_rec - d_sin
    print(f"observed mean delta-log pi: recurrent {d_rec:+.3f}, "
          f"single-event {d_sin:+.3f}, interaction {obs_int:+.3f} "
          f"({np.exp(obs_int):.2f}-fold)")

    rows = []
    # Stop short of the power: as f approaches it the mixture becomes
    # singular and the deconvolution is not identified.
    grid = np.linspace(0.0, min(0.45, 0.85 * power), 30)
    for f in grid:
        t_rec, t_sin, p_true = deattenuate(d_rec, d_sin, len(rec), len(sin),
                                           f, power)
        # p_true pinned at a bound means the label counts are inconsistent with
        # that error rate; the correction there is an extrapolation, not an
        # estimate, so it is left out of the curve.
        if not np.isfinite(t_rec) or p_true <= 1e-6 or p_true >= 1 - 1e-6:
            continue
        rows.append(dict(false_positive_rate=f, power=power,
                         true_recurrent_proportion=p_true,
                         corrected_delta_recurrent=t_rec,
                         corrected_delta_single=t_sin,
                         corrected_interaction=t_rec - t_sin,
                         corrected_fold=np.exp(t_rec - t_sin)))
    tab = pd.DataFrame(rows)

    # bootstrap the interval at the sweep's own false-positive rate
    rng = np.random.default_rng(RNG_SEED)
    boots = []
    for _ in range(N_BOOT):
        r = rng.choice(rec, len(rec), replace=True)
        s = rng.choice(sin, len(sin), replace=True)
        t_rec, t_sin, _ = deattenuate(r.mean(), s.mean(), len(rec), len(sin),
                                      fpr, power)
        if np.isfinite(t_rec):
            boots.append(t_rec - t_sin)
    boots = np.asarray(boots)
    at = tab.iloc[(tab["false_positive_rate"] - fpr).abs().argmin()]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nat the measured error rates the interaction corrects from "
          f"{np.exp(obs_int):.2f}-fold to {at['corrected_fold']:.2f}-fold "
          f"(95% CI {np.exp(lo):.2f}-{np.exp(hi):.2f})")
    print(f"implied true recurrent proportion: "
          f"{at['true_recurrent_proportion']:.3f} "
          f"(labelled {len(rec) / (len(rec) + len(sin)):.3f})")
    print("\nMisclassification shrinks the contrast, so the reported effect is a "
          "lower bound on the true one.")

    tab.to_csv(OUT_TSV, sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax = axes[0]
    ax.plot(tab["false_positive_rate"], tab["corrected_fold"], lw=2,
            color="#3B5BA5")
    ax.axhline(np.exp(obs_int), color="#8C8F99", ls="--", lw=1.2)
    ax.text(0.01, np.exp(obs_int) * 1.03, "observed, taking labels at face value",
            fontsize=8.5, color="#555555")
    ax.axvline(fpr, color="#C2601F", ls=":", lw=1.5)
    ax.annotate(f"measured\nFPR = {fpr:.2f}", xy=(fpr, at["corrected_fold"]),
                xytext=(fpr * 0.35, at["corrected_fold"] * 1.15), fontsize=9,
                color="#C2601F")
    ax.scatter([fpr], [at["corrected_fold"]], s=55, color="#C2601F", zorder=5,
               edgecolor="white")
    ax.set_yscale("log")
    ax.set_ylim(np.exp(obs_int) * 0.6,
                min(tab["corrected_fold"].max() * 1.4, 1e3))
    ax.set_xlabel("false-positive rate of the recurrence classifier")
    ax.set_ylabel("orientation-by-recurrence interaction (fold, log scale)")
    ax.set_title("A  Label error shrinks the contrast", loc="left", fontsize=11)

    ax = axes[1]
    ax.hist(np.exp(boots), bins=np.linspace(0, np.percentile(np.exp(boots), 99), 45), color="#7fb1d3", edgecolor="white")
    ax.axvline(np.exp(obs_int), color="#8C8F99", ls="--", lw=1.4,
               label=f"observed {np.exp(obs_int):.2f}x")
    ax.axvline(at["corrected_fold"], color="#C2601F", lw=1.8,
               label=f"corrected {at['corrected_fold']:.2f}x")
    ax.set_xlabel("corrected interaction (fold)")
    ax.set_ylabel("bootstrap replicates")
    ax.set_title(f"B  Corrected estimate\n95% CI {np.exp(lo):.2f}-{np.exp(hi):.2f}",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"\nWrote {OUT_TSV}\n      {OUT_PDF} / {OUT_PNG}")


if __name__ == "__main__":
    main()
