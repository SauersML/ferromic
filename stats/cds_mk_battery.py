"""Arrangement-conditional McDonald-Kreitman battery.

The information unit is the MUTATION, not the haplotype pair. Every variant in
a gene's in-frame CDS alignment is classified twice:

  ORIGIN -- which arrangement's genealogy it arose on. A variant whose minor
  allele occurs only among inverted haplotypes (at derived frequency < 1 in
  that group) arose on the inverted genealogy; the allele shared by both
  arrangements is ancestral. Private-to-direct likewise. Variants present in
  both arrangements are "shared" (gene flux / ancestral variation) and are
  excluded from origin-based counts. Columns where each arrangement is fixed
  for a different allele are arrangement FIXED differences (origin ambiguous,
  consequence well-defined) and feed the divergence row of the MK tables.

  CONSEQUENCE -- nonsynonymous vs synonymous, read from the codon: ancestral
  codon = shared/majority context, derived codon = ancestral with the derived
  base substituted. Intronic variants (from the whole-region alignments) form
  a third class used only as a negative control.

Under the null "coding consequence does not affect a mutation's survival
differently on the two backgrounds", each mutation's N/S label is independent
of the branch it fell on, so the 2x2 table

      {arose-on-inverted, arose-on-direct} x {N, S}

is testable by Fisher's exact test conditional on its margins -- exact under
any demography, inversion age, group size, or genealogy. Counts pool across
genes within a locus (they share haplotypes but each mutation is one event;
class labels stay independent), and loci combine by an exact Monte Carlo
stratified test (central hypergeometric draws per locus conditional on
margins). The battery:

  primary   global + per-locus N/S x arrangement-of-origin (protection: N
            depleted among inverted-origin; degeneration: N excess)
  control   S/intron x arrangement-of-origin -- must be null; deviation
            flags artifacts (paralogs, mapping), not biology
  SFS       within each arrangement, derived-frequency shift of N vs S
            (class-label permutation, locus-stratified) -- efficacy axis
  MK-div    fixed differences vs polymorphism (N/S), classic MK per locus

Outputs: results/cds_mk_genes.tsv, results/cds_mk_loci.tsv, printed summary.
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cds_selection_intron_control import (  # noqa: E402
    CDS_RE, INV_RE, encode, read_phy, load_gtf, intron_sites)

MC_DRAWS = 200_000
SFS_PERMS = 20_000
RNG_SEED = 2026
BASES = "ACGT"

CODON_TABLE = {}
_aa = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRR"
       "VVVVAAAADDEEGGGG")
for _i, _c in enumerate([a + b + c for a in "TCAG" for b in "TCAG" for c in "TCAG"]):
    CODON_TABLE[_c] = _aa[_i]


def classify_column(col, inv_mask):
    """Return (kind, derived, ancestral, origin, derived_freq) for one column.

    kind: 'mono' | 'multi' | 'shared' | 'private' | 'fixed_diff' | 'missing'
    origin: 'inv' | 'dir' | None
    """
    ok = col != 255
    if not (ok & inv_mask).any() or not (ok & ~inv_mask).any():
        return ("missing",) + (None,) * 4
    vals = np.unique(col[ok])
    if len(vals) == 1:
        return ("mono",) + (None,) * 4
    if len(vals) > 2:
        return ("multi",) + (None,) * 4
    a, b = vals
    in_inv = {v: bool(((col == v) & inv_mask).any()) for v in vals}
    in_dir = {v: bool(((col == v) & ~inv_mask).any()) for v in vals}
    both_a = in_inv[a] and in_dir[a]
    both_b = in_inv[b] and in_dir[b]
    if both_a and both_b:
        return ("shared",) + (None,) * 4
    if not both_a and not both_b:
        # each arrangement fixed for a different allele
        return ("fixed_diff", a, b, None, None)
    anc, der = (a, b) if both_a else (b, a)
    origin = "inv" if in_inv[der] else "dir"
    grp = inv_mask if origin == "inv" else ~inv_mask
    n_ok = int((ok & grp).sum())
    k_der = int(((col == der) & grp).sum())
    return ("private", der, anc, origin, k_der / n_ok)


def consequence(cmat, j, der, anc, inv_mask=None):
    """N/S for substituting anc->der at CDS position j (0-based, in frame)."""
    c0 = 3 * (j // 3)
    codon_anc = []
    for k in range(3):
        col = cmat[:, c0 + k]
        ok = col != 255
        if not ok.any():
            return None
        if c0 + k == j:
            codon_anc.append(anc)
        else:
            vals, cnt = np.unique(col[ok], return_counts=True)
            codon_anc.append(int(vals[np.argmax(cnt)]))
    if any(v > 3 for v in codon_anc):
        return None
    anc_str = "".join(BASES[v] for v in codon_anc)
    der_codon = list(codon_anc)
    der_codon[j - c0] = der
    der_str = "".join(BASES[v] for v in der_codon)
    aa_a, aa_d = CODON_TABLE[anc_str], CODON_TABLE[der_str]
    return "S" if aa_a == aa_d else "N"


def fixed_consequence(cmat, j, inv_mask):
    """N/S for an arrangement-fixed difference at CDS position j."""
    c0 = 3 * (j // 3)
    def codon(mask):
        out = []
        for k in range(3):
            col = cmat[mask, c0 + k]
            ok = col != 255
            if not ok.any():
                return None
            vals, cnt = np.unique(col[ok], return_counts=True)
            v = int(vals[np.argmax(cnt)])
            if v > 3:
                return None
            out.append(v)
        return "".join(BASES[v] for v in out)
    ci, cd = codon(inv_mask), codon(~inv_mask)
    if ci is None or cd is None:
        return None
    return "S" if CODON_TABLE[ci] == CODON_TABLE[cd] else "N"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    a = ap.parse_args()
    os.chdir(a.workdir)
    os.makedirs("results", exist_ok=True)

    tests = pd.read_csv("repo/data/gene_inversion_direct_inverted.tsv", sep="\t")
    cip = pd.read_csv("repo/data/cds_identical_proportions.tsv", sep="\t")
    cip["inv_id"] = (cip["chr"].astype(str).str.replace("chr", "", regex=False)
                     + ":" + cip["inv_start"].astype(int).astype(str)
                     + "-" + cip["inv_end"].astype(int).astype(str))
    recurrence = (cip.groupby("inv_id")["consensus"].first()
                  .map({0: "single-event", 1: "recurrent"}))

    phy_dir = "phy_outputs"
    cds_files, region_files = {}, {}
    for fn in os.listdir(phy_dir):
        m = CDS_RE.match(fn)
        if m:
            cds_files[(m["grp"], m["gene"], m["enst"], m["chrom"],
                       int(m["is"]), int(m["ie"]))] = os.path.join(phy_dir, fn)
            continue
        m = INV_RE.match(fn)
        if m:
            region_files.setdefault(
                (m["chrom"], int(m["s"]), int(m["e"])), {})[m["grp"]] = \
                os.path.join(phy_dir, fn)

    chroms = {f"chr{t.split(':')[0]}" for t in tests["inv_id"]}
    all_exons, tx_exons = load_gtf("gencode.v47.basic.annotation.gtf.gz", chroms)

    by_locus = defaultdict(list)
    for _, r in tests.iterrows():
        by_locus[r["inv_id"]].append(r)

    gene_rows, locus_rows = [], []
    sfs_records = []          # (inv_id, origin, class, derived_freq)
    locus_tables = {}         # inv_id -> [N_i, S_i, N_d, S_d]
    locus_ctrl = {}           # inv_id -> [S_i, I_i, S_d, I_d]

    for inv_id, locus_tests in sorted(by_locus.items()):
        chrom_num, coords = inv_id.split(":")
        inv_s, inv_e = (int(x) for x in coords.split("-"))
        chrom = f"chr{chrom_num}"
        reg = region_files.get((chrom_num, inv_s, inv_e)) or \
            region_files.get((chrom, inv_s, inv_e))
        if not reg or set(reg) != {"0", "1"}:
            continue
        seq_dir = read_phy(reg["0"])
        seq_inv = read_phy(reg["1"])
        names = sorted(seq_dir) + sorted(seq_inv)
        is_inv_full = np.array([n in seq_inv for n in names])
        L = len(next(iter(seq_dir.values())))
        region = np.vstack([encode((seq_dir | seq_inv)[n]) for n in names])
        print(f"[{inv_id}] {len(locus_tests)} genes", flush=True)

        T = np.zeros(4, dtype=int)      # N_i S_i N_d S_d
        C = np.zeros(4, dtype=int)      # S_i I_i S_d I_d
        FX = np.zeros(2, dtype=int)     # N_fix S_fix

        for r in locus_tests:
            key = (r["gene_name"], r["transcript_id"], chrom, inv_s, inv_e)
            f0, f1 = cds_files.get(("0", *key)), cds_files.get(("1", *key))
            if not (f0 and f1):
                continue
            cseq = read_phy(f0) | read_phy(f1)
            present = [n for n in names if n in cseq]
            idx = np.array([names.index(n) for n in present])
            inv_mask = is_inv_full[idx]
            if inv_mask.sum() == 0 or (~inv_mask).sum() == 0:
                continue
            cmat = np.vstack([encode(cseq[nm]) for nm in present])

            g = {"N_inv": 0, "S_inv": 0, "N_dir": 0, "S_dir": 0,
                 "N_fix": 0, "S_fix": 0, "shared_cds": 0, "multi_cds": 0}
            for j in range(cmat.shape[1]):
                kind, der, anc, origin, freq = classify_column(cmat[:, j], inv_mask)
                if kind == "shared":
                    g["shared_cds"] += 1
                elif kind == "multi":
                    g["multi_cds"] += 1
                elif kind == "fixed_diff":
                    cq = fixed_consequence(cmat, j, inv_mask)
                    if cq:
                        g[f"{cq}_fix"] += 1
                elif kind == "private":
                    cq = consequence(cmat, j, der, anc)
                    if cq:
                        tag = "inv" if origin == "inv" else "dir"
                        g[f"{cq}_{tag}"] += 1
                        sfs_records.append((inv_id, origin, cq, freq))

            # intron variants of this gene (origin only)
            isites = intron_sites(r["transcript_id"], chrom, all_exons, tx_exons)
            I_inv = I_dir = 0
            if isites is not None:
                cols = isites - (inv_s - 1)
                cols = cols[(cols >= 0) & (cols < L)]
                imat = region[np.ix_(idx, cols)]
                for j in range(imat.shape[1]):
                    kind, der, anc, origin, freq = classify_column(
                        imat[:, j], inv_mask)
                    if kind == "private":
                        if origin == "inv":
                            I_inv += 1
                        else:
                            I_dir += 1

            T += [g["N_inv"], g["S_inv"], g["N_dir"], g["S_dir"]]
            C += [g["S_inv"], I_inv, g["S_dir"], I_dir]
            FX += [g["N_fix"], g["S_fix"]]
            gene_rows.append({
                "gene_name": r["gene_name"], "inv_id": inv_id,
                "recurrence": recurrence.get(inv_id, "unknown"),
                "k_inv": int(inv_mask.sum()), "k_dir": int((~inv_mask).sum()),
                **g, "I_inv": I_inv, "I_dir": I_dir})

        if T.sum() == 0:
            continue
        locus_tables[inv_id] = T
        locus_ctrl[inv_id] = C
        tab = [[T[0], T[1]], [T[2], T[3]]]
        orr, p2 = fisher_exact(tab, alternative="two-sided")
        _, p_prot = fisher_exact(tab, alternative="less")     # N depleted on inv
        _, p_degen = fisher_exact(tab, alternative="greater")  # N excess on inv
        ctab = [[C[0], C[1]], [C[2], C[3]]]
        _, p_ctrl = fisher_exact(ctab, alternative="two-sided")
        # classic MK on the pooled locus: fixed vs polymorphic (all origins)
        poly_N, poly_S = T[0] + T[2], T[1] + T[3]
        mk_tab = [[FX[0], FX[1]], [poly_N, poly_S]]
        _, p_mk = fisher_exact(mk_tab, alternative="two-sided")
        locus_rows.append({
            "inv_id": inv_id, "recurrence": recurrence.get(inv_id, "unknown"),
            "N_inv": T[0], "S_inv": T[1], "N_dir": T[2], "S_dir": T[3],
            "OR": orr, "p_two": p2, "p_protection": p_prot,
            "p_degeneration": p_degen,
            "S_ctrl_inv": C[0], "I_ctrl_inv": C[1],
            "S_ctrl_dir": C[2], "I_ctrl_dir": C[3], "p_control": p_ctrl,
            "N_fix": FX[0], "S_fix": FX[1], "p_mk_divergence": p_mk})

    genes = pd.DataFrame(gene_rows)
    loci = pd.DataFrame(locus_rows)
    genes.to_csv("results/cds_mk_genes.tsv", sep="\t", index=False)
    loci.to_csv("results/cds_mk_loci.tsv", sep="\t", index=False)
    pd.DataFrame(sfs_records,
                 columns=["inv_id", "origin", "cls", "freq"]).to_csv(
        "results/cds_mk_sfs.tsv", sep="\t", index=False)

    # ---------------- global exact MC stratified test -----------------------
    rng = np.random.default_rng(RNG_SEED)
    obs_stat = 0.0
    draws = np.zeros(MC_DRAWS)
    for T in locus_tables.values():
        Ni, Si, Nd, Sd = (int(x) for x in T)
        M, row_i, K = Ni + Si + Nd + Sd, Ni + Si, Ni + Nd
        if M == 0 or row_i == 0 or row_i == M or K == 0 or K == M:
            continue
        E = row_i * K / M
        obs_stat += Ni - E
        draws += rng.hypergeometric(K, M - K, row_i, MC_DRAWS) - E
    p_glob_two = ((np.abs(draws) >= abs(obs_stat) - 1e-9).sum() + 1) / (MC_DRAWS + 1)
    p_glob_prot = ((draws <= obs_stat + 1e-9).sum() + 1) / (MC_DRAWS + 1)
    p_glob_degen = ((draws >= obs_stat - 1e-9).sum() + 1) / (MC_DRAWS + 1)

    # same global machinery for the S-vs-intron negative control
    obs_c = 0.0
    draws_c = np.zeros(MC_DRAWS)
    for C in locus_ctrl.values():
        Si, Ii, Sd, Id = (int(x) for x in C)
        M, row_i, K = Si + Ii + Sd + Id, Si + Ii, Si + Sd
        if M == 0 or row_i == 0 or row_i == M or K == 0 or K == M:
            continue
        E = row_i * K / M
        obs_c += Si - E
        draws_c += rng.hypergeometric(K, M - K, row_i, MC_DRAWS) - E
    p_ctrl_glob = ((np.abs(draws_c) >= abs(obs_c) - 1e-9).sum() + 1) / (MC_DRAWS + 1)

    # ---------------- SFS shift, locus-stratified permutation ---------------
    sfs = pd.DataFrame(sfs_records, columns=["inv_id", "origin", "cls", "freq"])
    sfs_out = {}
    for origin in ("inv", "dir"):
        sub = sfs[sfs.origin == origin]
        strata = [g for _, g in sub.groupby("inv_id")
                  if g.cls.nunique() == 2]
        if not strata:
            sfs_out[origin] = (np.nan, np.nan, 0, 0)
            continue
        def shift(gs):
            tot, w = 0.0, 0
            for g in gs:
                u = mannwhitneyu(g[g.cls == "N"].freq, g[g.cls == "S"].freq,
                                 alternative="two-sided").statistic
                nN, nS = (g.cls == "N").sum(), (g.cls == "S").sum()
                auc = u / (nN * nS)
                tot += (auc - 0.5) * (nN + nS)
                w += nN + nS
            return tot / w
        obs = shift(strata)
        null = np.empty(SFS_PERMS)
        for b in range(SFS_PERMS):
            perm = []
            for g in strata:
                gg = g.copy()
                gg["cls"] = rng.permutation(gg["cls"].to_numpy())
                if gg.cls.nunique() == 2:
                    perm.append(gg)
            null[b] = shift(perm)
        p = ((np.abs(null) >= abs(obs) - 1e-12).sum() + 1) / (SFS_PERMS + 1)
        nN = int((sub.cls == "N").sum()); nS = int((sub.cls == "S").sum())
        sfs_out[origin] = (obs, p, nN, nS)

    # ---------------- report ----------------------------------------------
    print("\n================= MK BATTERY =================")
    tot = np.zeros(4, dtype=int)
    for T in locus_tables.values():
        tot += T
    print(f"loci: {len(locus_tables)}; pooled counts "
          f"N_inv={tot[0]} S_inv={tot[1]} N_dir={tot[2]} S_dir={tot[3]}")
    ratio_i = tot[0] / tot[1] if tot[1] else np.nan
    ratio_d = tot[2] / tot[3] if tot[3] else np.nan
    print(f"N/S inverted-origin: {ratio_i:.3f}   direct-origin: {ratio_d:.3f}")
    print(f"GLOBAL exact stratified: two-sided p={p_glob_two:.4g}  "
          f"protection p={p_glob_prot:.4g}  degeneration p={p_glob_degen:.4g}")
    print(f"NEGATIVE CONTROL (S vs intron): global p={p_ctrl_glob:.4g}  "
          f"(should be null; small p = artifact alarm)")
    for origin, (obs, p, nN, nS) in sfs_out.items():
        print(f"SFS shift [{origin}]: weighted AUC-0.5 = {obs:+.4f} "
              f"(N={nN}, S={nS}) p={p:.4g}  "
              f"(negative = N rarer than S = purifying)")
    print("\nper-locus (loci with >=10 informative mutations):")
    big = loci[(loci[["N_inv", "S_inv", "N_dir", "S_dir"]].sum(axis=1) >= 10)]
    cols = ["inv_id", "recurrence", "N_inv", "S_inv", "N_dir", "S_dir", "OR",
            "p_two", "p_protection", "p_degeneration", "p_control",
            "N_fix", "S_fix", "p_mk_divergence"]
    print(big[cols].sort_values("p_two").to_string(index=False))
    print("\nWrote results/cds_mk_genes.tsv, results/cds_mk_loci.tsv")


if __name__ == "__main__":
    main()
