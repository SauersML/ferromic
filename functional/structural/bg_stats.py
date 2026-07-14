"""Per-locus background-genetics job (network): for every analysis locus, fetch phased
1000G SNVs in the breakpoint windows, classify backgrounds by the tag SNP, and compute:

Handle 2 (recurrence natural experiment):
  - between_bg_divergence: mean |af_inv - af_dir| over polymorphic sites (how diverged the
    inverted haplotype class is from direct — tight linkage = high).
  - pi_inv / pi_dir: nucleotide diversity within each background (2*af*(1-af)); high within-
    inverted diversity relative to divergence signals multiple independent origins.
  - n_diff_sites_050: SNVs with |af_inv-af_dir| >= 0.5 (near-fixed differences by orientation).

Handle 3 (collinearity / separability):
  - For the tag SNP, r^2 (LD) between the tag and every window SNV; report max r^2, and the
    count of SNVs in near-perfect LD (r^2 >= 0.8, 0.95) with the inversion tag. High max r^2
    with a functional cis SNV => statistically inseparable (unresolved-collinear).

Saves results/bg_stats.json and per-locus site tables to the artifacts dir.
"""
import json, os, sys, time
import numpy as np

try:
    from . import reconstruct as R
    from . import _data
except ImportError:
    import reconstruct as R  # type: ignore
    import _data  # type: ignore

L = 131072
_RESULTS = os.environ.get("STRUCTURAL_RESULTS_DIR", _data.RESULTS_DIR)
targets = {t["locus"]: t for t in _data.load_data("targets_master.json")}
analysis = _data.load_data("analysis_loci.json")
ARTDIR = os.environ.get("STRUCTURAL_BG_SITES_DIR", os.path.join(_RESULTS, "bg_sites"))
os.makedirs(ARTDIR, exist_ok=True)

def r2(a, b):
    """r^2 between two 0/1 haplotype allele vectors (drop missing)."""
    m = (a >= 0) & (b >= 0)
    if m.sum() < 10:
        return np.nan
    x = a[m].astype(float); y = b[m].astype(float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return float(r * r)

out = []
for loc in analysis:
  try:
    t = targets[loc]
    chrom = t["chrom"]
    tag = R.get_tag_record(chrom, int(t["tag_pos38"]))
    if tag is None:
        out.append({"locus": loc, "error": "tag_not_found"})
        json.dump(out, open(os.path.join(_RESULTS, "bg_stats.json"), "w"), indent=1); continue
    samples, tag_gt, tagra = tag
    cls = R.classify_backgrounds(samples, tag_gt, t["hom_inv_samples"], t["hom_dir_samples"])
    hap_inv, hap_dir = cls["hap_inv"], cls["hap_dir"]
    inv_allele = cls["inv_allele"]
    # inversion tag as 0/1 inverted indicator per haplotype
    tag_inv_ind = np.where(tag_gt < 0, -1, (tag_gt == inv_allele).astype(np.int8))
    wins = R.choose_windows(t["start38"], t["end38"], t["size"], L=L)
    t0 = time.time()
    afi_all, afd_all, diff_all = [], [], []
    max_r2 = 0.0; n_r2_080 = 0; n_r2_095 = 0; n_poly = 0
    site_rows = []
    for (ws, we, lab) in wins:
        _, recs = R.fetch_snvs(chrom, ws, we)
        _, _, tab = R.build_consensus(recs, hap_inv, hap_dir)
        for rec, row in zip(recs, tab):
            fi, fd = row["af_inv"], row["af_dir"]
            if 0 < (fi * row["n_inv"] + fd * row["n_dir"]):
                pass
            # overall polymorphic?
            if min(fi, 1 - fi) > 0.01 or min(fd, 1 - fd) > 0.01:
                n_poly += 1
                afi_all.append(fi); afd_all.append(fd); diff_all.append(abs(fi - fd))
                # LD with inversion tag
                rr = r2(tag_inv_ind, rec["gt"])
                if not np.isnan(rr):
                    if rr > max_r2:
                        max_r2 = rr
                    if rr >= 0.8: n_r2_080 += 1
                    if rr >= 0.95: n_r2_095 += 1
                    site_rows.append((rec["pos"], round(fi, 4), round(fd, 4),
                                      round(abs(fi - fd), 4), round(rr, 4)))
    afi = np.array(afi_all); afd = np.array(afd_all); dif = np.array(diff_all)
    rec = {
        "locus": loc, "chrom": chrom, "recur_consensus": t["recur_consensus"],
        "inv_AF": t["inv_AF"], "n_hom_inv": t["n_hom_inv"], "n_hom_dir": t["n_hom_dir"],
        "n_hap_inverted": cls["n_hap_inverted"], "n_hap_direct": cls["n_hap_direct"],
        "tag_porubsky_concordance": cls["tag_porubsky_concordance"],
        "measured_any": t["measured_any"], "geuv_n_sig": t["geuv_n_sig"],
        "gtex_has_sqtl": t["gtex_has_sqtl"],
        "n_poly_sites": int(n_poly),
        "between_bg_divergence": float(dif.mean()) if len(dif) else float("nan"),
        "n_diff_sites_050": int((dif >= 0.5).sum()),
        "n_diff_sites_020": int((dif >= 0.2).sum()),
        "pi_inv": float((2 * afi * (1 - afi)).mean()) if len(afi) else float("nan"),
        "pi_dir": float((2 * afd * (1 - afd)).mean()) if len(afd) else float("nan"),
        "tag_max_r2_cis": round(max_r2, 4),
        "n_cis_r2_ge080": int(n_r2_080), "n_cis_r2_ge095": int(n_r2_095),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out.append(rec)
    np.savez_compressed(os.path.join(ARTDIR, loc.replace(":", "_") + ".npz"),
                        sites=np.array(site_rows, dtype=np.float64))
    print(f"{loc:<28} rec={t['recur_consensus']} div={rec['between_bg_divergence']:.3f} "
          f"piI={rec['pi_inv']:.4f} piD={rec['pi_dir']:.4f} nDiff.5={rec['n_diff_sites_050']} "
          f"maxr2={rec['tag_max_r2_cis']:.3f} nR2>=.95={rec['n_cis_r2_ge095']} ({rec['elapsed_s']}s)", flush=True)
    json.dump(out, open(os.path.join(_RESULTS, "bg_stats.json"), "w"), indent=1)  # incremental
  except Exception as e:
    import traceback; traceback.print_exc()
    out.append({"locus": loc, "error": str(e)})
    json.dump(out, open(os.path.join(_RESULTS, "bg_stats.json"), "w"), indent=1)

json.dump(out, open(os.path.join(_RESULTS, "bg_stats.json"), "w"), indent=1)
print("wrote", os.path.join(_RESULTS, "bg_stats.json"))
