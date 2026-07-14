"""Integrate the three handles + structural anchor into a per-locus verdict:
'structural (breakpoint-mediated)', 'consistent with structural', 'consistent with linked-SNV',
'unresolved (collinear)', or 'mixed/partial'. Honest: this bounds structure-vs-haplotype; it
does not prove the inversion causal."""
import csv, json, os
import numpy as np

try:
    from . import _data
except ImportError:
    import _data  # type: ignore

_RESULTS = os.environ.get("STRUCTURAL_RESULTS_DIR", _data.RESULTS_DIR)


def _load(name):
    with open(os.path.join(_RESULTS, name)) as fh:
        return json.load(fh)


# PAML branch dN/dS (direct vs inverted lineage) per region/gene — a MEASURED orientation-
# specific coding-selection signal (reflects linked coding SNV divergence between backgrounds).
# Optional: point FUNCTIONAL_PAML_RESULTS at ferromic's GRAND_PAML_RESULTS.tsv to include it.
paml = {}
pf = os.environ.get("FUNCTIONAL_PAML_RESULTS", "")
if pf and os.path.exists(pf):
    with open(pf) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            reg = row["region"]  # chrN_start_end
            parts = reg.rsplit("_", 2)
            if len(parts) != 3:
                continue
            loc = f"{parts[0]}:{parts[1]}-{parts[2]}"
            try:
                od = float(row.get("winner_omega2_direct", "") or "nan")
                oi = float(row.get("winner_omega2_inverted", "") or "nan")
                q = float(row.get("overall_q_value", "") or "nan")
            except ValueError:
                od = oi = q = float("nan")
            paml.setdefault(loc, []).append({"gene": row["gene"], "omega_direct": od,
                                             "omega_inverted": oi, "q": q, "status": row["status"]})

anchor = {r["locus"]: r for r in _load("structural_anchor.json")}
ag = {r["locus"]: r for r in _load("ag_decomp_full.json") if "windows" in r}
h2 = {r["locus"]: r for r in _load("handle2_recurrence.json")["rows"]}
try:
    h3 = {r["locus"]: r for r in _load("handle3_conditional.json")}
except FileNotFoundError:
    h3 = {}

def frac_struct(loc):
    if loc not in ag:
        return float("nan")
    num = den = 0.0
    for w in ag[loc]["windows"]:
        s = w["disruption_flank"]["struct"]; n = w["disruption_flank"]["snv"]
        num += s; den += s + n
    return num / den if den > 0 else float("nan")

rows = []
for loc in ag:
    a = anchor.get(loc, {}); b = h2.get(loc, {}); c = h3.get(loc, {})
    fs = frac_struct(loc)
    maxr2 = b.get("tag_max_r2_cis", float("nan"))
    bp_med = a.get("measured_qtl_gene_is_breakpoint_mediated", False)
    bp_in_gene = a.get("breakpoint_in_gene", False)
    conc = ag[loc]["background"]["tag_porubsky_concordance"]
    tag_reliable = conc >= 0.90
    # verdict
    if not tag_reliable:
        verdict = "tag-unreliable (background assignment uncertain, concordance %.2f)" % conc
    elif bp_med:
        verdict = "structural (breakpoint-mediated, model-free)"
    elif np.isfinite(fs) and fs >= 0.66:
        verdict = "consistent with structural (in-silico structure-dominant)"
    elif np.isfinite(fs) and fs <= 0.34:
        verdict = "consistent with linked-SNV (in-silico SNV-dominant)"
    elif np.isfinite(maxr2) and maxr2 >= 0.95:
        verdict = "unresolved (collinear: inversion near-perfectly tagged)"
    elif np.isfinite(fs):
        verdict = "mixed/partial (structure + linked-SNV both contribute)"
    else:
        verdict = "insufficient data"
    rows.append({
        "locus": loc, "recur_consensus": b.get("recur_consensus"),
        "measured_any": b.get("measured_any"), "inv_AF": b.get("inv_AF"),
        "breakpoint_in_gene": bp_in_gene, "bp_mediated_genes": a.get("bp_mediated_genes", []),
        "measured_qtl_gene": a.get("measured_qtl_gene", ""),
        "qtl_gene_breakpoint_mediated": bp_med,
        "fraction_structural": round(fs, 3) if np.isfinite(fs) else None,
        "between_bg_divergence": b.get("between_bg_divergence"),
        "n_diff_sites_050": b.get("n_diff_sites_050"),
        "tag_porubsky_concordance": round(conc, 3), "tag_reliable": bool(tag_reliable),
        "tag_max_r2_cis": maxr2, "n_cis_r2_ge095": b.get("n_cis_r2_ge095"),
        "cond_inv_survives": c.get("inv_effect_survives"),
        "cond_top_cis_r2_with_inversion": c.get("top_cis_r2_with_inversion"),
        "paml_genes_tested": sum(1 for g in paml.get(loc, []) if np.isfinite(g["omega_direct"]) and np.isfinite(g["omega_inverted"])),
        "paml_any_orientation_divergent": any(
            np.isfinite(g["omega_direct"]) and np.isfinite(g["omega_inverted"]) and g["q"] < 0.1
            for g in paml.get(loc, [])),
        "verdict": verdict,
    })

rows.sort(key=lambda r: (r["measured_any"] != 1, -(r["fraction_structural"] or 0)))
from collections import Counter
vc = Counter(r["verdict"] for r in rows)
json.dump({"verdict_counts": dict(vc), "rows": rows},
          open(os.path.join(_RESULTS, "locus_verdicts.json"), "w"), indent=1)
print("verdict counts:")
for v, n in vc.most_common():
    print(f"  {n:>3}  {v}")
print(f"\nmedian fraction_structural (all): "
      f"{np.nanmedian([r['fraction_structural'] for r in rows if r['fraction_structural'] is not None]):.3f}")
print("\nper-locus:")
for r in rows:
    print(f"  {r['locus']:<27} rec={r['recur_consensus']} meas={r['measured_any']} "
          f"fracS={r['fraction_structural']} maxr2={r['tag_max_r2_cis']} bp_gene={r['breakpoint_in_gene']} "
          f"-> {r['verdict']}")
print("wrote results/locus_verdicts.json")
