"""De-biased decomposition using TRUE per-individual phased haplotypes (full per-haplotype
SNV content, rare + common), instead of the major-allele consensus.

Orientation per haplotype: Porubsky/tag (concordance-gated loci only). Full SNV content per
haplotype: 1000G GRCh38 statistical phasing (each haplotype's actual phased alt alleles).
We sample K individual haplotypes per background, build each one's full sequence, score
SPLICE_SITE_USAGE, and decompose:

  a_i = SSU(inv_i)  - SSU(ref_inverted)          # per inverted haplotype i
  b_j = SSU(dir_j)  - SSU(ref_direct)            # per direct haplotype j
  d_struct = SSU(ref_inverted) - SSU(ref_direct) # orientation flip (haplotype-independent)
  point estimate: d_snv = mean_i a_i - mean_j b_j;  fraction_structural = ||d_struct||/(||d_struct||+||d_snv||)
  per-individual range: d_snv_i = a_i - mean_j b_j; fraction_i = ||d_struct||/(||d_struct||+||d_snv_i||)

Because individual haplotypes carry their full (incl. rare/intermediate) linked-SNV load, the
SNV component is larger than under consensus, so this de-biases the structural fraction DOWNWARD.
Consensus fraction is thus an upper bound; this reports the de-biased value + a per-individual range.
Deterministic (seed 42)."""
import json, os, sys, time
import numpy as np

try:  # package import
    from . import reconstruct as R
    from .score_consensus import _ssu
    from . import _data
except ImportError:  # loose script
    import reconstruct as R  # type: ignore
    from score_consensus import _ssu  # type: ignore
    import _data  # type: ignore

K = int(os.environ.get("KHAP", "16"))
SEED = 42
L = 131072

def flank_mask(qc):
    m = np.ones(qc["L"], dtype=bool); m[qc["seg_i1"]:qc["seg_i2"]] = False
    return m

def indiv_variants(recs, h):
    """Variants (pos,ref,alt) carried by haplotype index h (its phased allele == alt)."""
    return [(rec["pos"], rec["ref"], rec["alt"]) for rec in recs if rec["gt"][h] == 1]

def build_seq(chrom, ws, we, b1, b2, variants, invert):
    ref = R.load_ref(chrom, ws, we); seq = list(ref); Ln = len(seq)
    for pos, rf, alt in variants:
        idx = pos - ws
        if 0 <= idx < Ln and seq[idx] == rf:
            seq[idx] = alt
    if invert:
        i1 = max(0, b1 - ws); i2 = min(Ln, b2 - ws)
        seq[i1:i2] = list(R.revcomp("".join(seq[i1:i2])))
    return "".join(seq)

def score_window_perhap(chrom, ws, we, b1, b2, recs, hap_inv_idx, hap_dir_idx, rng):
    seqs, qc = R.construct_sequences(chrom, ws, we, b1, b2, [], [])  # for qc geometry
    m = flank_mask(qc)
    A = _ssu(seqs["ref_direct"])[0]      # ref_direct SSU
    B = _ssu(seqs["ref_inverted"])[0]    # ref_inverted SSU
    d_struct = (B - A)
    ds_norm = float(np.abs(d_struct)[m].sum())
    # sample haplotypes
    inv_sel = rng.choice(hap_inv_idx, size=min(K, len(hap_inv_idx)), replace=False)
    dir_sel = rng.choice(hap_dir_idx, size=min(K, len(hap_dir_idx)), replace=False)
    # direct: accumulate mean b_j
    bsum = np.zeros_like(A)
    for j in dir_sel:
        sj = build_seq(chrom, ws, we, b1, b2, indiv_variants(recs, int(j)), invert=False)
        bsum += (_ssu(sj)[0] - A)
    bmean = bsum / len(dir_sel)   # mean_j (SSU(dir_j)-SSU(ref_dir))
    # inverted: accumulate mean a_i and per-individual fraction
    asum = np.zeros_like(A); frac_i = []
    for i in inv_sel:
        si = build_seq(chrom, ws, we, b1, b2, indiv_variants(recs, int(i)), invert=True)
        ai = (_ssu(si)[0] - B)
        asum += ai
        dsnv_i = ai - bmean
        fi = ds_norm / (ds_norm + float(np.abs(dsnv_i)[m].sum())) if ds_norm > 0 else float("nan")
        frac_i.append(fi)
    amean = asum / len(inv_sel)
    d_snv = amean - bmean
    snv_norm = float(np.abs(d_snv)[m].sum())
    frac_point = ds_norm / (ds_norm + snv_norm) if ds_norm > 0 else float("nan")
    return {
        "window": [ws, we], "n_inv_hap": int(len(inv_sel)), "n_dir_hap": int(len(dir_sel)),
        "D_struct": ds_norm, "D_snv_debiased": snv_norm,
        "fraction_structural_debiased": frac_point,
        "fraction_individual_median": float(np.nanmedian(frac_i)),
        "fraction_individual_min": float(np.nanmin(frac_i)),
        "fraction_individual_max": float(np.nanmax(frac_i)),
    }, frac_i

def run(loci, targets_path=None, out_path=None):
    targets_path = targets_path or _data.data_path("targets_master.json")
    out_path = out_path or _data.results_path("perhap_debiased.json")
    targets = {t["locus"]: t for t in json.load(open(targets_path))}
    out = []
    for loc in loci:
        t = targets[loc]; chrom = t["chrom"]
        tag = R.get_tag_record(chrom, int(t["tag_pos38"]))
        if tag is None:
            out.append({"locus": loc, "error": "tag_not_found"}); continue
        samples, tag_gt, _ = tag
        cls = R.classify_backgrounds(samples, tag_gt, t["hom_inv_samples"], t["hom_dir_samples"])
        hap_inv_idx = np.where(cls["hap_inv"])[0]; hap_dir_idx = np.where(cls["hap_dir"])[0]
        rng = np.random.default_rng(SEED)
        wins = R.choose_windows(t["start38"], t["end38"], t["size"], L=L)
        wres = []
        for (ws, we, lab) in wins:
            t0 = time.time()
            _, recs = R.fetch_snvs(chrom, ws, we)
            r, frac_i = score_window_perhap(chrom, ws, we, t["start38"], t["end38"],
                                            recs, hap_inv_idx, hap_dir_idx, rng)
            r["window_label"] = lab; r["elapsed_s"] = round(time.time() - t0, 1)
            wres.append(r)
            print(f"{loc} [{lab}] fracS_debiased={r['fraction_structural_debiased']:.3f} "
                  f"(indiv median {r['fraction_individual_median']:.3f}, "
                  f"range {r['fraction_individual_min']:.2f}-{r['fraction_individual_max']:.2f}) "
                  f"nInv={r['n_inv_hap']} nDir={r['n_dir_hap']} ({r['elapsed_s']}s)", flush=True)
        # locus-level disruption-weighted debiased fraction
        num = den = 0.0
        for r in wres:
            num += r["D_struct"]; den += r["D_struct"] + r["D_snv_debiased"]
        out.append({"locus": loc, "recur_consensus": t["recur_consensus"],
                    "fraction_structural_debiased": num / den if den else float("nan"),
                    "windows": wres})
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(out, open(out_path, "w"), indent=1)
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="De-biased per-haplotype structure-vs-SNV decomposition (AlphaGenome API)")
    ap.add_argument("--loci", nargs="*", default=None, help="locus strings; default = committed highconf_loci.json")
    ap.add_argument("--targets", default=None, help="targets_master.json (default: committed package copy)")
    ap.add_argument("--out", default=None, help="output json (default: committed results/structural/perhap_debiased.json)")
    a = ap.parse_args(argv)
    loci = a.loci or _data.load_data("highconf_loci.json")
    run(loci, targets_path=a.targets, out_path=a.out)
    print("wrote", a.out or _data.results_path("perhap_debiased.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
