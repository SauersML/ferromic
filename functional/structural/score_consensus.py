"""AlphaGenome scoring + exact structure-vs-SNV decomposition of the splice-usage
difference field.

For a window we build 4 sequences (see reconstruct.py) and query SPLICE_SITE_USAGE for
each. The per-position/per-track difference field decomposes EXACTLY (linear identity):

  d_total(p,t)  = SSU(full_inverted) - SSU(full_direct)
  d_struct(p,t) = SSU(ref_inverted)  - SSU(ref_direct)          # pure orientation flip
  d_snv(p,t)    = [SSU(full_inverted)-SSU(ref_inverted)]
                - [SSU(full_direct) -SSU(ref_direct)]           # differential linked SNVs
  =>  d_total == d_struct + d_snv   (verified to float precision as a QC)

Disruption magnitudes are L1 norms of these fields restricted to positions OUTSIDE the
flipped segment (frame-consistent: same genomic base identity in every sequence, so a
change there is a genuine context-mediated splicing perturbation) and optionally to a gene's
exonic footprint. fraction_structural = ||d_struct|| / (||d_struct|| + ||d_snv||).
"""
import os, json, sys, time
import numpy as np

try:  # package import
    from . import reconstruct as R
except ImportError:  # loose script
    import reconstruct as R

_CLIENT = None
def client():
    global _CLIENT
    if _CLIENT is None:
        from alphagenome.models import dna_client
        _CLIENT = dna_client.create(os.environ["ALPHAGENOME_API_KEY"])
    return _CLIENT

def _ssu(seq):
    from alphagenome.models import dna_client
    out = client().predict_sequence(
        sequence=seq,
        requested_outputs=[dna_client.OutputType.SPLICE_SITE_USAGE],
        ontology_terms=None,
    )
    v = out.splice_site_usage
    return np.asarray(v.values, dtype=np.float32), v.metadata


def score_window(chrom, ws, we, b1, b2, inv_cons, dir_cons, gene_spans=None):
    """Construct 4 seqs, query SSU for each, decompose. gene_spans: optional list of
    (name, gstart, gend) genomic spans for per-gene localization. Returns (result_dict,
    per_position_arrays_dict)."""
    seqs, qc = R.construct_sequences(chrom, ws, we, b1, b2, inv_cons, dir_cons)
    L = qc["L"]; i1 = qc["seg_i1"]; i2 = qc["seg_i2"]
    A, md = _ssu(seqs["ref_direct"])
    B, _ = _ssu(seqs["ref_inverted"])
    C, _ = _ssu(seqs["full_inverted"])
    D, _ = _ssu(seqs["full_direct"])
    d_struct = B - A
    d_snv = (C - B) - (D - A)
    d_total = C - D
    # exact reconciliation QC
    recon_max_abs = float(np.max(np.abs(d_total - (d_struct + d_snv))))
    # per-position magnitude summed over tracks
    pp_struct = np.abs(d_struct).sum(axis=1)   # (L,)
    pp_snv = np.abs(d_snv).sum(axis=1)
    pp_total = np.abs(d_total).sum(axis=1)
    # flank mask: outside flipped segment
    flank = np.ones(L, dtype=bool); flank[i1:i2] = False
    def norms(mask):
        return dict(struct=float(pp_struct[mask].sum()),
                    snv=float(pp_snv[mask].sum()),
                    total=float(pp_total[mask].sum()))
    n_flank = norms(flank)
    n_all = norms(np.ones(L, dtype=bool))
    def frac(n):
        den = n["struct"] + n["snv"]
        return n["struct"] / den if den > 0 else float("nan")
    res = {
        "chrom": chrom, "win_start": ws, "win_end": we, "b1": b1, "b2": b2,
        "qc": qc, "reconciliation_max_abs_err": recon_max_abs,
        "ntracks": int(A.shape[1]),
        "disruption_flank": n_flank, "disruption_all": n_all,
        "fraction_structural_flank": frac(n_flank),
        "fraction_structural_all": frac(n_all),
    }
    if gene_spans:
        genes = {}
        for name, gs, ge in gene_spans:
            gi1 = max(0, gs - ws); gi2 = min(L, ge - ws)
            if gi1 >= gi2:
                continue
            m = np.zeros(L, dtype=bool); m[gi1:gi2] = True
            m &= flank  # frame-consistent positions only
            if m.sum() == 0:
                continue
            n = norms(m)
            genes[name] = {**n, "fraction_structural": frac(n),
                           "n_pos": int(m.sum()), "gstart": gs, "gend": ge}
        res["genes"] = genes
    arrays = {"pp_struct": pp_struct, "pp_snv": pp_snv, "pp_total": pp_total}
    return res, arrays


def run_locus(t, L=131072, gene_spans_by_win=None, verbose=True):
    """Full per-locus scoring: classify backgrounds, per window build consensus + score."""
    chrom = t["chrom"]; start = t["start38"]; end = t["end38"]; size = t["size"]
    tag = R.get_tag_record(chrom, int(t["tag_pos38"]))
    if tag is None:
        return {"locus": t["locus"], "error": "tag_not_found"}
    samples, tag_gt, tagra = tag
    cls = R.classify_backgrounds(samples, tag_gt, t["hom_inv_samples"], t["hom_dir_samples"])
    wins = R.choose_windows(start, end, size, L=L)
    win_results = []
    all_arrays = {}
    for (ws, we, lab) in wins:
        t0 = time.time()
        _, recs = R.fetch_snvs(chrom, ws, we)
        inv_cons, dir_cons, site_tab = R.build_consensus(recs, cls["hap_inv"], cls["hap_dir"])
        gs = None
        if gene_spans_by_win is not None:
            gs = gene_spans_by_win.get((t["locus"], lab))
        res, arrays = score_window(chrom, ws, we, start, end, inv_cons, dir_cons, gene_spans=gs)
        res["window_label"] = lab
        res["n_snvs_fetched"] = len(recs)
        res["n_inv_consensus"] = len(inv_cons)
        res["n_dir_consensus"] = len(dir_cons)
        res["n_sites_differential_ge02"] = int(sum(1 for r in site_tab if r["differential"] >= 0.2))
        res["elapsed_s"] = round(time.time() - t0, 1)
        win_results.append(res)
        all_arrays[f"{t['locus']}|{lab}"] = arrays
        if verbose:
            print(f"  [{lab}] snvs={len(recs)} invc={len(inv_cons)} dirc={len(dir_cons)} "
                  f"D_struct={res['disruption_flank']['struct']:.2f} D_snv={res['disruption_flank']['snv']:.2f} "
                  f"fracS={res['fraction_structural_flank']:.3f} recon_err={res['reconciliation_max_abs_err']:.2e} "
                  f"({res['elapsed_s']}s)", flush=True)
    out = {
        "locus": t["locus"], "chrom": chrom, "start38": start, "end38": end, "size": size,
        "tag_snp": t["tag_snp"], "tag_variantId": t["tag_variantId"],
        "recur_consensus": t["recur_consensus"], "inv_AF": t["inv_AF"],
        "n_hom_inv": t["n_hom_inv"], "n_hom_dir": t["n_hom_dir"],
        "background": {k: v for k, v in cls.items() if k not in ("hap_inv", "hap_dir")},
        "windows": win_results,
    }
    return out, all_arrays


def main(argv=None):
    import argparse
    try:
        from . import _data
    except ImportError:
        import _data  # type: ignore
    ap = argparse.ArgumentParser(description="Consensus structure-vs-SNV AlphaGenome decomposition")
    ap.add_argument("--loci", nargs="*", default=None, help="locus strings; default = smoke on 17q21")
    ap.add_argument("--loci-file", default=None, help="json list of locus strings")
    ap.add_argument("--targets", default=None, help="targets_master.json (default: committed package copy)")
    ap.add_argument("--L", type=int, default=131072)
    ap.add_argument("--out", default="results/ag_decomp.json")
    ap.add_argument("--arraydir", default=None)
    ap.add_argument("--gene-spans", default=None, help="json {locus|label: [[name,gs,ge],...]}")
    args = ap.parse_args(argv)
    targets_path = args.targets or _data.data_path("targets_master.json")
    targets = {t["locus"]: t for t in json.load(open(targets_path))}
    if args.loci_file:
        args.loci = json.load(open(args.loci_file))
    if not args.loci:
        args.loci = ["chr17:45585159-46292045"]
    gsb = None
    if args.gene_spans:
        raw = json.load(open(args.gene_spans))
        gsb = {}
        for k, v in raw.items():
            loc, lab = k.split("|")
            gsb[(loc, lab)] = v
    results = []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for loc in args.loci:
        if loc not in targets:
            print("SKIP unknown locus", loc); continue
        print(f"=== {loc} ===", flush=True)
        try:
            out, arrays = run_locus(targets[loc], L=args.L, gene_spans_by_win=gsb)
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"locus": loc, "error": str(e)})
            json.dump(results, open(args.out, "w"), indent=1)  # incremental save
            continue
        results.append(out)
        if args.arraydir:
            os.makedirs(args.arraydir, exist_ok=True)
            for key, arr in arrays.items():
                safe = key.replace(":", "_").replace("|", "__")
                np.savez_compressed(os.path.join(args.arraydir, safe + ".npz"), **arr)
        json.dump(results, open(args.out, "w"), indent=1)  # incremental save
    json.dump(results, open(args.out, "w"), indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
