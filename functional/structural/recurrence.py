"""Handle 2 — recurrence as a natural experiment.

Recurrent inversions flip the same segment on independent SNV backgrounds. Two data-driven
readouts of "does the effect track orientation (structural) or the background (haplotype)":

(a) Population-genetic signature of the inverted haplotype class (from bg_stats):
    - between_bg_divergence and n_diff_sites: how tightly a distinct SNV haplotype co-segregates
      with orientation. A SINGLE-origin inversion carries one ancient, highly diverged haplotype
      (tight linkage, large divergence -> hard to separate structure from SNVs). A RECURRENT
      inversion re-flips on multiple backgrounds, so orientation is less tied to any one haplotype
      (lower divergence / higher within-inverted diversity) -> a consistent signal is more
      attributable to structure.
    - pi_inv / pi_dir: within-background nucleotide diversity.
(b) In-silico structural fraction (Handle 1) contrasted by recurrence status.

Reports recurrent-vs-single contrasts with Mann-Whitney U (small-n, honest about power)."""
import json, os, sys
import numpy as np
from scipy import stats

try:
    from . import _data
except ImportError:
    import _data  # type: ignore

_RESULTS = os.environ.get("STRUCTURAL_RESULTS_DIR", _data.RESULTS_DIR)


def _load(name):
    with open(os.path.join(_RESULTS, name)) as fh:
        return json.load(fh)


bg = {r["locus"]: r for r in _load("bg_stats.json") if "error" not in r}
ag = {r["locus"]: r for r in _load("ag_decomp_full.json")}

def locus_fraction_structural(r):
    """Disruption-weighted mean structural fraction over windows (flank-restricted)."""
    num = den = 0.0
    for w in r["windows"]:
        s = w["disruption_flank"]["struct"]; n = w["disruption_flank"]["snv"]
        num += s; den += s + n
    return (num / den) if den > 0 else float("nan")

rows = []
for loc, b in bg.items():
    fs = locus_fraction_structural(ag[loc]) if loc in ag else float("nan")
    rows.append({
        "locus": loc, "recur_consensus": b["recur_consensus"],
        "between_bg_divergence": b["between_bg_divergence"], "n_diff_sites_050": b["n_diff_sites_050"],
        "pi_inv": b["pi_inv"], "pi_dir": b["pi_dir"],
        "fraction_structural": fs, "measured_any": b["measured_any"],
        "tag_max_r2_cis": b["tag_max_r2_cis"], "n_cis_r2_ge095": b["n_cis_r2_ge095"],
        "inv_AF": b["inv_AF"],
    })

def grp(key):
    R = [r for r in rows if r["recur_consensus"] == "1" and np.isfinite(r.get(key, np.nan))]
    S = [r for r in rows if r["recur_consensus"] == "0" and np.isfinite(r.get(key, np.nan))]
    return [r[key] for r in R], [r[key] for r in S]

summary = {"n_recurrent": sum(1 for r in rows if r["recur_consensus"] == "1"),
           "n_single": sum(1 for r in rows if r["recur_consensus"] == "0"),
           "contrasts": {}}
for key in ["between_bg_divergence", "n_diff_sites_050", "pi_inv", "fraction_structural", "tag_max_r2_cis"]:
    Rv, Sv = grp(key)
    entry = {"recurrent_median": float(np.median(Rv)) if Rv else None,
             "single_median": float(np.median(Sv)) if Sv else None,
             "n_recurrent": len(Rv), "n_single": len(Sv)}
    if len(Rv) >= 2 and len(Sv) >= 2:
        u, p = stats.mannwhitneyu(Rv, Sv, alternative="two-sided")
        entry["mannwhitney_p"] = float(p)
    summary["contrasts"][key] = entry

json.dump({"summary": summary, "rows": rows},
          open(os.path.join(_RESULTS, "handle2_recurrence.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
print("\nper-locus (sorted by recurrence, divergence):")
for r in sorted(rows, key=lambda x: (x["recur_consensus"] or "z", -x["between_bg_divergence"])):
    print(f"  {r['locus']:<28} rec={r['recur_consensus']} div={r['between_bg_divergence']:.3f} "
          f"nDiff.5={r['n_diff_sites_050']:>4} fracS={r['fraction_structural']:.3f} maxr2={r['tag_max_r2_cis']:.3f}")
print("wrote results/handle2_recurrence.json")
