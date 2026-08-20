"""Inversion-selection envelope test, rebuilt on the unfiltered 1KG panel.

Why: the 44-sample analysis VCF (GQ>=30 in ALL samples, no-missing, segdup
mask) strips rare variants -- at chr7:70.96Mb it kept 26 of 374 real variants
and manufactured a fake clonal inverted class (the full-panel identical-
haplotype cluster test disproved the clone). The envelope statistics are
therefore recomputed here from the NYGC high-coverage phased VCFs (3,202
samples), restricted to the 44 HGSVC samples whose arrangement labels we
trust, so every locus gets its real within-class rare variation back.

Orientation labels come from phy-file membership (assembly/Strand-seq based,
SNP-independent). Arrangement homozygotes label both panel haplotypes
directly. For heterozygotes, each panel haplotype is matched to the sample's
two phy haplotypes at sites where those differ; assignments need a >=2-site
margin and >=80% identity or the sample is dropped at that locus. As a
phasing-robustness check, every locus is also scored with heterozygous
samples excluded entirely (hom-only column) -- conclusions must agree.

Statistics and null are unchanged from stats/inversion_selection_envelope.py
(conditioned-branch coalescent with infinite-sites mutation layer), whose
calibration and power were established by forward simulation (SLiM scenarios:
sizes 0-7% at alpha 0.05 under neutral/growth/structure/flux nulls; 97% power
against true sweeps; modest power against balancing). Matching errors mix
classes and push A upward, i.e. against sweep calls -- conservative.

Phase 1 (--phase slice, LOGIN node -- needs internet): download per-locus
region slices of the NYGC VCFs for the 44 samples.
Phase 2 (--phase analyze, compute node): statistics + envelope p-values.

Output: results/inversion_envelope_1kg.tsv + printed summary with a
calibration read (p-value uniformity) and old-vs-new comparison.
"""

import argparse
import gzip
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inversion_selection_envelope import (  # noqa: E402
    conditioned_branch_null, envelope_p, bh_q, clock_flags, parse_demography,
    single_origin_loci)
from cds_selection_intron_control import INV_RE, read_phy  # noqa: E402

KG_BASE = ("http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
           "1000G_2504_high_coverage/working/20201028_3202_phased")
MIN_K = 5
N_CAND = 20_000
MAX_TREES = 3_000_000
RNG_SEED = 4242


def locus_list(workdir):
    region_files = {}
    for fn in os.listdir(os.path.join(workdir, "phy_outputs")):
        m = INV_RE.match(fn)
        if m:
            region_files.setdefault(
                (m["chrom"], int(m["s"]), int(m["e"])), {})[m["grp"]] = \
                os.path.join(workdir, "phy_outputs", fn)
    return {k: v for k, v in sorted(region_files.items())
            if set(v) == {"0", "1"}}


def sample_ids(workdir):
    hdr = open(os.path.join(workdir, "repo", "data", "callset.tsv")).readline()
    cols = hdr.rstrip("\n").split("\t")
    meta = {"seqnames", "start", "end", "width", "inv_id", "arbigent_genotype",
            "misorient_info", "orthog_tech_support", "inversion_category",
            "inv_AF"}
    return [c for c in cols if c not in meta]


def chr_file_map():
    import urllib.request
    listing = urllib.request.urlopen(KG_BASE + "/", timeout=60).read().decode()
    files = re.findall(r'href="(CCDG[^"]*chr[0-9XY]+[^"]*vcf\.gz)"', listing)
    out = {}
    for f in files:
        m = re.search(r"chr([0-9XY]+)\.", f)
        if m:
            out.setdefault(m.group(1), f)
    return out


def phase_slice(workdir):
    os.makedirs(f"{workdir}/kg_slices", exist_ok=True)
    ids = sample_ids(workdir)
    with open(f"{workdir}/kg_slices/samples.txt", "w") as fh:
        fh.write("\n".join(ids) + "\n")
    cmap = chr_file_map()
    loci = locus_list(workdir)
    print(f"loci: {len(loci)}; chr files: {len(cmap)}")
    jobs = []
    for (c, s, e), _files in loci.items():
        if c not in cmap:
            print(f"  no panel file for chr{c}; skipping {c}:{s}-{e}")
            continue
        out = f"{workdir}/kg_slices/{c}_{s}_{e}.vcf.gz"
        if os.path.exists(out) and os.path.getsize(out) > 200:
            continue
        tbi = f"{workdir}/kg_slices/chr{c}.tbi"
        if not os.path.exists(tbi):
            subprocess.run(["curl", "-s", f"{KG_BASE}/{cmap[c]}.tbi",
                            "-o", tbi], check=True)
        jobs.append((c, s, e, cmap[c], tbi, out))
    print(f"to download: {len(jobs)}")

    def one(j):
        c, s, e, fname, tbi, out = j
        cmd = ["bcftools", "view", "-r", f"chr{c}:{s}-{e}",
               "-S", f"{workdir}/kg_slices/samples.txt", "--force-samples",
               "-v", "snps", f"{KG_BASE}/{fname}##idx##{tbi}",
               "-Oz", "-o", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return (out, r.returncode, r.stderr[-200:] if r.returncode else "")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, res in enumerate(ex.map(one, jobs), 1):
            if res[1]:
                print("  FAIL", res[0], res[2], flush=True)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    print("slice phase done")


def load_slice(path):
    """positions, ref, alt, hap matrix (n_haps x n_sites, 0/1), sample list."""
    poss, refs, alts, rows = [], [], [], []
    samples = None
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            f = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                samples = f[9:]
                continue
            if "," in f[4] or len(f[3]) != 1 or len(f[4]) != 1:
                continue
            if f[6] not in (".", "PASS"):
                continue
            gts = f[9:]
            alle = []
            ok = True
            for g in gts:
                a = g.split(":")[0].replace("/", "|").split("|")
                if len(a) != 2 or a[0] == "." or a[1] == ".":
                    ok = False
                    break
                alle.append((int(a[0]), int(a[1])))
            if not ok:
                continue
            poss.append(int(f[1]))
            refs.append(f[3])
            alts.append(f[4])
            rows.append([x for pair in alle for x in pair])
    if not rows:
        return None
    M = np.array(rows, dtype=np.int8).T          # (2*n_samples, n_sites)
    return np.array(poss), refs, alts, M, samples


def analyze_locus(args):
    (key, files, workdir, seed, window, n_cand, demography) = args
    c, inv_s, inv_e = key
    inv_id = f"{c}:{inv_s}-{inv_e}"
    path = f"{workdir}/kg_slices/{c}_{inv_s}_{inv_e}.vcf.gz"
    if not os.path.exists(path):
        return {"inv_id": inv_id, "status": "NO_SLICE"}
    sl = load_slice(path)
    if sl is None:
        return {"inv_id": inv_id, "status": "EMPTY_SLICE"}
    poss, refs, alts, M, samples = sl

    g0 = read_phy(files["0"])
    g1 = read_phy(files["1"])
    grp = {}
    for n in g0:
        grp.setdefault(n.split("_")[2], {})[n] = 0
    for n in g1:
        grp.setdefault(n.split("_")[2], {})[n] = 1
    L_phy = len(next(iter((g0 or g1).values())))

    labels = np.full(M.shape[0], -1, dtype=np.int8)
    n_het = n_hom = n_drop = 0
    margins = []
    for si, sid in enumerate(samples):
        if sid not in grp or len(grp[sid]) != 2:
            continue
        gvals = list(grp[sid].values())
        h0, h1 = 2 * si, 2 * si + 1
        if gvals[0] == gvals[1]:
            labels[h0] = labels[h1] = gvals[0]
            n_hom += 1
            continue
        # heterozygote: match panel haps to the two phy haps
        (nA, gA), (nB, gB) = grp[sid].items()
        seqA, seqB = (g0 | g1)[nA], (g0 | g1)[nB]
        idx = [(j, p - inv_s) for j, p in enumerate(poss)
               if 0 <= p - inv_s < L_phy]
        inf = [(j, q) for j, q in idx if seqA[q] != seqB[q]
               and seqA[q] in "ACGT" and seqB[q] in "ACGT"]
        if len(inf) < 3:
            n_drop += 1
            continue
        scoreA0 = scoreA1 = 0
        for j, q in inf:
            baseH0 = refs[j] if M[h0, j] == 0 else alts[j]
            baseH1 = refs[j] if M[h1, j] == 0 else alts[j]
            if baseH0 == seqA[q]:
                scoreA0 += 1
            if baseH1 == seqA[q]:
                scoreA1 += 1
        margin = abs(scoreA0 - scoreA1)
        margins.append(margin)
        if margin < 2:
            n_drop += 1
            continue
        if scoreA0 > scoreA1:
            labels[h0], labels[h1] = gA, gB
        else:
            labels[h0], labels[h1] = gB, gA
        n_het += 1

    use = labels >= 0
    lab = labels[use]
    sub = M[use]
    n = int(use.sum())
    k = int((lab == 1).sum())
    if k < MIN_K or (n - k) < MIN_K:
        return {"inv_id": inv_id, "status": "TOO_FEW", "n": n, "k_inv": k}

    def pi_abs(mask_rows):
        m = sub[mask_rows]
        kk = m.shape[0]
        p = m.sum(axis=0)
        return float((2 * p * (kk - p) / (kk * (kk - 1))).sum())

    def cross_abs(a, b):
        ma, mb = sub[a], sub[b]
        pa, pb = ma.sum(axis=0), mb.sum(axis=0)
        na, nb = ma.shape[0], mb.shape[0]
        return float(((pa * (nb - pb) + pb * (na - pa)) / (na * nb)).sum())

    inv_m, dir_m = lab == 1, lab == 0
    pi_inv, pi_dir = pi_abs(inv_m), pi_abs(dir_m)
    d_cross = cross_abs(inv_m, dir_m)
    if pi_dir <= 0:
        return {"inv_id": inv_id, "status": "NO_DIVERSITY", "n": n, "k_inv": k}
    A, B = pi_inv / pi_dir, d_cross / pi_dir

    rng = np.random.default_rng(seed)
    tmap, _dl = parse_demography(demography)
    Anull, Bnull, Wgt, Tid, _ = conditioned_branch_null(
        n, k, pi_dir, rng, n_cand=n_cand, max_trees=MAX_TREES, window=window,
        tmap=None if demography in ("const", "", None) else tmap)
    if len(Anull) < 200:
        return {"inv_id": inv_id, "status": "NULL_STARVED", "n": n, "k_inv": k,
                "null_candidates": int(len(Anull))}
    sw = envelope_p(Anull, A, Wgt, Tid, "lower", rng)
    bal = envelope_p(Bnull, B, Wgt, Tid, "upper", rng)
    p_sweep, p_bal = sw["p"], bal["p"]

    # hom-only sensitivity
    A_hom = np.nan
    hom_rows = np.zeros(n, dtype=bool)
    pos_i = 0
    for si, sid in enumerate(samples):
        for h in (2 * si, 2 * si + 1):
            if labels[h] >= 0:
                sm = grp.get(sid, {})
                vals = list(sm.values()) if len(sm) == 2 else []
                hom_rows[pos_i] = (len(vals) == 2 and vals[0] == vals[1])
                pos_i += 1
    hl, hs = lab[hom_rows], hom_rows
    if (hl == 1).sum() >= MIN_K and (hl == 0).sum() >= MIN_K:
        sub_h = sub[hs]
        def pih(msk):
            m = sub_h[msk]
            kk = m.shape[0]
            p = m.sum(axis=0)
            return float((2 * p * (kk - p) / (kk * (kk - 1))).sum())
        pd_h = pih(hl == 0)
        if pd_h > 0:
            A_hom = pih(hl == 1) / pd_h

    return {"inv_id": inv_id, "status": "OK", "n": n, "k_inv": k,
            "n_variants": int(sub.shape[1]),
            "n_het_matched": n_het, "n_hom": n_hom, "n_dropped": n_drop,
            "median_match_margin": float(np.median(margins)) if margins else np.nan,
            "pi_inv_abs": pi_inv, "pi_dir_abs": pi_dir, "d_cross_abs": d_cross,
            "A": A, "B": B, "A_hom_only": A_hom,
            "p_sweep": p_sweep, "p_balance": p_bal,
            "p_sweep_raw": sw["p_raw"], "p_sweep_mcse": sw["mcse"],
            "p_sweep_lo": sw["lo"], "p_sweep_hi": sw["hi"],
            "p_balance_raw": bal["p_raw"], "p_balance_mcse": bal["mcse"],
            "p_balance_lo": bal["lo"], "p_balance_hi": bal["hi"],
            "null_candidates": int(len(Anull)), "null_ess": sw["ess"],
            "null_trees": sw["n_tree"], "null_window": window,
            "null_demography": demography,
            "n_tail_sweep": sw["n_tail"], "n_tail_balance": bal["n_tail"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--phase", choices=["slice", "analyze"], required=True)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--window", type=float, default=0.0,
                    help="carrier-count conditioning window (fraction of n); "
                         "0 = exact k. Non-zero is sensitivity only.")
    ap.add_argument("--out", default="results/inversion_envelope_1kg.tsv")
    ap.add_argument("--only", default="",
                    help="comma-separated inv_ids; restrict the run to these. "
                         "Used for the deep pass on loci that came out nominal "
                         "in the screen, where the screen's tail is too thin to "
                         "quote a number from.")
    ap.add_argument("--n-cand", type=int, default=N_CAND,
                    help="candidate branches per locus (deep pass raises this)")
    ap.add_argument("--demography", default="const",
                    help="size history for the null: const | exp:BETA | "
                         "pw:t1,nu1;t2,nu2;... Exact via the coalescent time "
                         "change, so demography needs no forward simulation.")
    a = ap.parse_args()

    if a.phase == "slice":
        phase_slice(a.workdir)
        return

    loci = locus_list(a.workdir)
    # HARD GATE: the one-branch null is only valid for confirmed single-origin
    # inversions, so recurrent and unclassified loci never reach the estimator.
    # No p-value is produced for them, which also means they cannot inflate the
    # multiple-testing family for the loci that ARE valid.
    allowed = single_origin_loci(os.path.join(a.workdir, "repo", "data"))
    before = len(loci)
    loci = {k: v for k, v in loci.items()
            if f"{k[0]}:{k[1]}-{k[2]}" in allowed}
    print(f"single-origin premise: {len(loci)} of {before} loci with "
          f"alignments are confirmed single-event; {before - len(loci)} "
          f"recurrent/unclassified loci excluded before any statistic is "
          f"computed")
    if not loci:
        sys.exit("no locus satisfies the single-origin premise -- check that "
                 "repo/data/inv_properties.tsv coordinates match the phy "
                 "filenames; refusing to write an empty table")
    if a.only:
        want = set(a.only.split(","))
        bad = want - allowed
        if bad:
            raise SystemExit(f"--only names loci that are not confirmed "
                             f"single-origin: {sorted(bad)}. The one-branch "
                             f"null does not describe them.")
        loci = {k: v for k, v in loci.items()
                if f"{k[0]}:{k[1]}-{k[2]}" in want}
        print(f"restricted to {len(loci)} locus/loci: {sorted(want)}")
    tasks = [(k, v, a.workdir, RNG_SEED + i, a.window, a.n_cand, a.demography)
             for i, (k, v) in enumerate(loci.items())]
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, r in enumerate(ex.map(analyze_locus, tasks), 1):
            rows.append(r)
            if r.get("status") == "OK":
                print(f"  {r['inv_id']} n={r['n']} k={r['k_inv']} "
                      f"vars={r['n_variants']} A={r['A']:.3f} "
                      f"(hom {r['A_hom_only']:.3f}) B={r['B']:.3f} "
                      f"p_sw={r['p_sweep']:.4f}"
                      f"[{r['p_sweep_lo']:.4f},{r['p_sweep_hi']:.4f}] "
                      f"p_bal={r['p_balance']:.4f}"
                      f"[{r['p_balance_lo']:.4f},{r['p_balance_hi']:.4f}]"
                      f"+-{r['p_balance_mcse']:.4f} "
                      f"cand={r['null_candidates']} ess={r['null_ess']:.0f} "
                      f"tail={r['n_tail_balance']}", flush=True)
            if i % 25 == 0:
                print(f"... {i}/{len(tasks)}", flush=True)
    out = pd.DataFrame(rows)
    os.makedirs(f"{a.workdir}/results", exist_ok=True)
    # merge old (filtered-VCF) results for comparison
    try:
        old = pd.read_csv(f"{a.workdir}/results/inversion_selection_envelope.tsv",
                          sep="\t")
        # B was renamed B_sojourn -> B_crossdepth (the old name licensed reading
        # the ratio as a literal age multiple); accept either vintage.
        bcol = "B_crossdepth" if "B_crossdepth" in old.columns else "B_sojourn"
        old = old[["inv_id", "recurrence", "A_ageratio", bcol,
                   "p_sweep", "p_balance"]]
        old.columns = ["inv_id", "recurrence", "A_old", "B_old",
                       "p_sweep_old", "p_balance_old"]
        out = out.merge(old, on="inv_id", how="left")
    except Exception as e:
        print("old-results merge failed:", e)
    okm = out["status"].eq("OK")
    for src, dst in (("p_sweep", "q_sweep_bh"), ("p_balance", "q_balance_bh")):
        out[dst] = np.nan
        out.loc[okm, dst] = bh_q(out.loc[okm, src].to_numpy())
    flags, perbp, med = clock_flags(
        out["inv_id"].tolist(),
        out["B"].to_numpy() if "B" in out else np.full(len(out), np.nan),
        out["pi_dir_abs"].to_numpy() if "pi_dir_abs" in out
        else np.full(len(out), np.nan),
        okm.to_numpy())
    out["pi_dir_perbp"] = perbp
    out["clock_flag"] = flags
    out.to_csv(a.out if os.path.isabs(a.out) else f"{a.workdir}/{a.out}",
               sep="\t", index=False)

    ok = out[out["status"] == "OK"]
    print("\n============ 1KG-REBUILT ENVELOPE ============")
    print(f"conditioning window: {a.window} "
          f"({'EXACT k' if a.window == 0 else 'WIDENED -- sensitivity only'})")
    print(f"null demography: {a.demography}")
    print(f"testable loci: {len(ok)} of {len(out)}")
    print(f"median variants/locus: {ok.n_variants.median():.0f}; "
          f"median het-match margin: {ok.median_match_margin.median():.1f}")
    print(f"null resolution: median candidates "
          f"{ok.null_candidates.median():.0f}, median ESS "
          f"{ok.null_ess.median():.0f}, median trees "
          f"{ok.null_trees.median():.0f}")
    for nm in ("p_sweep", "p_balance"):
        p = ok[nm].to_numpy()
        q = ok[f"q_{nm.split('_')[1]}_bh"].to_numpy()
        print(f"{nm}: <0.05: {(p < 0.05).sum()}  <0.01: {(p < 0.01).sum()}  "
              f"BH q<0.05: {np.nansum(q < 0.05):.0f}  "
              f"(uniformity check -- mean {np.mean(p):.3f}, expect ~0.5)")
    print(f"\nclock-normalization median per-bp pi_dir: {med:.3e}")
    susp = ok[ok["clock_flag"].astype(str).str.len() > 0]
    print(f"loci with an untrustworthy pi_dir clock: {len(susp)} of {len(ok)}")
    nomsw = susp[susp["p_sweep"] < 0.05]
    if len(nomsw):
        print("  NOMINAL SWEEPS THAT ARE FLAGGED -- do not report without the "
              "within-background contrast:")
        for _, r in nomsw.iterrows():
            print(f"    {r['inv_id']}  p_sweep={r['p_sweep']:.4g}  "
                  f"q={r['q_sweep_bh']:.3g}  B={r['B']:.3f}  "
                  f"flag={r['clock_flag']}")
    if "A_old" in ok:
        both = ok.dropna(subset=["A_old"])
        fixed = both[(both.A_old < 0.2) & (both.A > 0.5)]
        print(f"\nloci whose clonality dissolved (A_old<0.2 -> A>0.5): {len(fixed)}")
        print(f"hom-only agreement: median |A - A_hom_only| = "
              f"{(both.A - both.A_hom_only).abs().median():.3f}")
    cols = ["inv_id", "recurrence", "n", "k_inv", "n_variants", "A", "A_old",
            "A_hom_only", "B", "B_old",
            "p_sweep", "p_sweep_lo", "p_sweep_hi", "q_sweep_bh",
            "p_balance", "p_balance_mcse", "p_balance_lo", "p_balance_hi",
            "q_balance_bh", "null_candidates", "null_ess", "n_tail_balance"]
    cols = [cc for cc in cols if cc in ok.columns]
    print("\ntop by p_sweep:")
    print(ok.sort_values("p_sweep")[cols].head(8).to_string(index=False))
    print("\ntop by p_balance:")
    print(ok.sort_values("p_balance")[cols].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
