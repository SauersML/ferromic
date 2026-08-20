"""Permutation replacement for the per-gene CDS jackknife z-test.

The original test (stats/per_gene_cds_differences_jackknife.py) compares the
fraction of identical haplotype pairs between the inverted and direct CDS
alignments of a gene, using a leave-one-haplotype-out jackknife SE and a normal
approximation. That approximation fails exactly where the reported signal is
strongest: with tiny groups and identity proportions near 1.0 the jackknife
badly underestimates the SE, producing |z| up to 44.5, and 2*(1 - norm_cdf(z))
underflows to exactly 0 in double precision for |z| >~ 8.3. Nine of the twenty
genes reported at FDR 0.05 carry p = 0 by that route (the se = 0 boundary
branch in the original code never fired; all nine SEs are finite). A k = 3
gene with jackknife z = 8.6 has a true permutation p of 0.41.

Here the same statistic

    Delta = p_identical(inverted) - p_identical(direct)

is referred to its permutation distribution under the null that orientation
carries no information about sequence: orientation labels are shuffled across
the haplotypes observed for that gene, holding both group sizes fixed. The
statistic depends only on the multiset of sequence-identity classes in each
group, so every draw reduces to a bincount and the whole null is vectorised.

Enumeration is exact when C(n, min(k_direct, k_inverted)) is small enough --
which covers the small-group cases that broke the jackknife -- and Monte Carlo
otherwise, escalating the draw count when the tail is sparse so that small
p-values are resolved rather than floored.

Multiplicity: genes within one inversion share haplotypes and are therefore not
independent (71 of the 164 tests lie in the 8p23.1 locus alone). Benjamini-
Hochberg assumes positive dependence; Benjamini-Yekutieli is valid under
arbitrary dependence. Both are reported, and BY is the defensible default here.

Inputs:
  data/gene_inversion_direct_inverted.tsv   (gene x inversion tests to re-examine)
  data/cds_identical_proportions.tsv        (recurrence label per inversion)
  per-gene group0_/group1_ .phy.gz alignments, located via the CDS_PHY_DIR env
  var or recovered from data/phy_outputs.zip (see stats/four_fold_pi.py).

Output:
  data/per_gene_cds_permutation.tsv
(The companion figure is drawn by stats/cds_permutation_joint_control.py.)
"""

import gzip
import itertools
import math
import os
import re
import sys

import numpy as np
import pandas as pd

_STATS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_STATS_DIR)
_DATA_DIR = os.path.join(_REPO_DIR, "data")

TESTS_TSV = os.path.join(_DATA_DIR, "gene_inversion_direct_inverted.tsv")
CDS_SUMMARY_TSV = os.path.join(_DATA_DIR, "cds_identical_proportions.tsv")
OUT_TSV = os.path.join(_DATA_DIR, "per_gene_cds_permutation.tsv")

EXACT_LIMIT = 2_000_000     # enumerate all assignments when C(n, kmin) <= this
MC_BASE = 100_000           # first-pass Monte Carlo draws
MC_ESCALATED = 2_000_000    # redrawn at this size when the tail is sparse
ESCALATE_HITS = 50          # escalate when the first pass sees <= this many hits
CHUNK = 50_000              # draws per vectorised block
RNG_SEED = 2026

FNAME_RE = re.compile(
    r"^group(?P<grp>[01])_(?P<gene>.+?)_(?P<ensg>ENSG[0-9.]+)_(?P<enst>ENST[0-9.]+)_"
    r"(?P<chrom>chr[^_]+)_cds_start(?P<cs>\d+)_cds_end(?P<ce>\d+)_"
    r"inv_start(?P<is>\d+)_inv_end(?P<ie>\d+)\.phy\.gz$"
)


def resolve_phy_dir():
    d = os.environ.get("CDS_PHY_DIR")
    if d and os.path.isdir(d):
        return d
    sys.path.insert(0, _STATS_DIR)
    import four_fold_pi  # noqa: E402  (reuses its zip/LFS recovery)

    d, _tmp = four_fold_pi.resolve_phy_dir()
    return d


def read_phy(path):
    """Return {haplotype_name: sequence} from a sequential one-line-per-taxon PHYLIP."""
    seqs = {}
    with gzip.open(path, "rt") as fh:
        n_declared = int(fh.readline().split()[0])
        for line in fh:
            if not line.strip():
                continue
            name, seq = line.rstrip("\n").split(None, 1)
            seqs[name] = seq.replace(" ", "")
    if len(seqs) != n_declared:
        raise ValueError(f"{path}: header says {n_declared} taxa, parsed {len(seqs)}")
    return seqs


def _ident_pairs(counts):
    """Identical-pair count per row, given per-class counts (rows = draws)."""
    return (counts * (counts - 1) // 2).sum(axis=-1)


def _deltas_from_small_counts(counts_small, total_counts, kmin_is_inverted,
                              pairs_small, pairs_big):
    counts_big = total_counts - counts_small
    ident_small = _ident_pairs(counts_small)
    ident_big = _ident_pairs(counts_big)
    p_small = ident_small / pairs_small
    p_big = ident_big / pairs_big
    return (p_small - p_big) if kmin_is_inverted else (p_big - p_small)


def _counts_from_index_block(idx_block, classes, n_classes):
    """Per-draw class counts for a (B, kmin) block of haplotype indices."""
    b, k = idx_block.shape
    flat = (np.arange(b)[:, None] * n_classes + classes[idx_block]).ravel()
    return np.bincount(flat, minlength=b * n_classes).reshape(b, n_classes)


def perm_test(classes, k_inv, delta_obs, rng):
    """Two-sided permutation p for |Delta|, group sizes held fixed.

    Returns (p_value, method, n_assignments_used).
    """
    n = len(classes)
    n_classes = int(classes.max()) + 1
    k_dir = n - k_inv
    kmin = min(k_inv, k_dir)
    kmin_is_inverted = k_inv <= k_dir
    pairs_small = kmin * (kmin - 1) // 2
    pairs_big = (n - kmin) * (n - kmin - 1) // 2
    if pairs_small == 0 or pairs_big == 0:
        return math.nan, "undefined_group_size", 0
    total_counts = np.bincount(classes, minlength=n_classes)
    thr = abs(delta_obs) - 1e-12

    def count_hits(index_iter, n_draws):
        hits = 0
        for start in range(0, n_draws, CHUNK):
            block = next(index_iter)
            counts = _counts_from_index_block(block, classes, n_classes)
            d = _deltas_from_small_counts(counts, total_counts, kmin_is_inverted,
                                          pairs_small, pairs_big)
            hits += int((np.abs(d) >= thr).sum())
        return hits

    n_assign = math.comb(n, kmin)
    if n_assign <= EXACT_LIMIT:
        combos = itertools.combinations(range(n), kmin)

        def exact_blocks():
            while True:
                block = list(itertools.islice(combos, CHUNK))
                if not block:
                    return
                yield np.asarray(block, dtype=np.int64)

        hits = count_hits(exact_blocks(), n_assign)
        return hits / n_assign, "exact", n_assign

    def mc_blocks(total):
        for start in range(0, total, CHUNK):
            b = min(CHUNK, total - start)
            noise = rng.random((b, n))
            yield np.argpartition(noise, kmin - 1, axis=1)[:, :kmin]

    hits = count_hits(mc_blocks(MC_BASE), MC_BASE)
    if hits <= ESCALATE_HITS:
        hits = count_hits(mc_blocks(MC_ESCALATED), MC_ESCALATED)
        return (hits + 1) / (MC_ESCALATED + 1), "monte_carlo", MC_ESCALATED
    return (hits + 1) / (MC_BASE + 1), "monte_carlo", MC_BASE


def _step_up(p, factor):
    """Shared BH/BY step-up machinery; factor scales the raw BH threshold."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    running = 1.0
    for j, i in enumerate(order[::-1]):
        rank = n - j
        running = min(running, p[i] * n * factor / rank)
        q[i] = running
    return q


def bh(p):
    return _step_up(p, 1.0)


def by(p):
    n = len(p)
    return _step_up(p, sum(1.0 / k for k in range(1, n + 1)))


def recurrence_labels():
    cip = pd.read_csv(CDS_SUMMARY_TSV, sep="\t")
    cip["inv_id"] = (cip["chr"].astype(str).str.replace("chr", "", regex=False)
                     + ":" + cip["inv_start"].astype(int).astype(str)
                     + "-" + cip["inv_end"].astype(int).astype(str))
    return (cip.groupby("inv_id")["consensus"].first()
            .map({0: "single-event", 1: "recurrent"}))


def main():
    tests = pd.read_csv(TESTS_TSV, sep="\t")
    phy_dir = resolve_phy_dir()
    labels = recurrence_labels()
    print(f"Alignments dir: {phy_dir}")
    print(f"Gene x inversion tests to re-examine: {len(tests)}")

    files = {}
    for fn in os.listdir(phy_dir):
        m = FNAME_RE.match(fn)
        if m:
            files[(m["grp"], m["gene"], m["enst"], m["chrom"],
                   int(m["is"]), int(m["ie"]))] = os.path.join(phy_dir, fn)

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for n_done, (_, r) in enumerate(tests.iterrows(), 1):
        chrom_num, coords = r["inv_id"].split(":")
        inv_s, inv_e = (int(x) for x in coords.split("-"))
        base = (r["gene_name"], r["transcript_id"], f"chr{chrom_num}", inv_s, inv_e)
        f0, f1 = files.get(("0", *base)), files.get(("1", *base))
        if not (f0 and f1):
            rows.append({"gene_name": r["gene_name"], "inv_id": r["inv_id"],
                         "status": "MISSING_PHY"})
            continue
        seq_dir, seq_inv = read_phy(f0), read_phy(f1)
        overlap = set(seq_dir) & set(seq_inv)
        if overlap:
            raise ValueError(f"{r['gene_name']}: haplotype(s) in both orientations: "
                             f"{sorted(overlap)[:3]}")

        class_of = {}
        classes, is_inv = [], []
        for name, seq in itertools.chain(seq_dir.items(), seq_inv.items()):
            classes.append(class_of.setdefault(seq, len(class_of)))
            is_inv.append(name in seq_inv)
        classes = np.asarray(classes)
        inv_mask = np.asarray(is_inv)

        k_inv, k_dir = int(inv_mask.sum()), int((~inv_mask).sum())
        cnt_inv = np.bincount(classes[inv_mask], minlength=len(class_of))
        cnt_dir = np.bincount(classes[~inv_mask], minlength=len(class_of))
        p_inv = _ident_pairs(cnt_inv) / (k_inv * (k_inv - 1) // 2)
        p_dir = _ident_pairs(cnt_dir) / (k_dir * (k_dir - 1) // 2)
        delta_obs = p_inv - p_dir

        pval, method, n_used = perm_test(classes, k_inv, delta_obs, rng)
        rows.append({
            "gene_name": r["gene_name"],
            "transcript_id": r["transcript_id"],
            "inv_id": r["inv_id"],
            "recurrence": labels.get(r["inv_id"], "unknown"),
            "k_direct": k_dir,
            "k_inverted": k_inv,
            "n_seq_classes": len(class_of),
            "p_direct": p_dir,
            "p_inverted": p_inv,
            "delta": delta_obs,
            "perm_p": pval,
            "perm_method": method,
            "n_assignments": n_used,
            "jackknife_p": r["p_value"],
            "jackknife_q": r["q_value"],
            "jackknife_note": r.get("note", ""),
            "reproduces_published": (abs(p_dir - r["p_direct"]) < 1e-6
                                     and abs(p_inv - r["p_inverted"]) < 1e-6),
            "status": "OK",
        })
        if n_done % 25 == 0:
            print(f"  ... {n_done}/{len(tests)}", flush=True)

    out = pd.DataFrame(rows)
    ok = out["status"].eq("OK").to_numpy()
    out.loc[ok, "perm_q_bh"] = bh(out.loc[ok, "perm_p"].to_numpy())
    out.loc[ok, "perm_q_by"] = by(out.loc[ok, "perm_p"].to_numpy())
    out = out.sort_values(["perm_q_by", "perm_p", "gene_name"], na_position="last")
    out.to_csv(OUT_TSV, sep="\t", index=False)

    d = out[out["status"].eq("OK")]
    print("\n=== reproduction check ===")
    print(f"  identity proportions matching the published table: "
          f"{int(d['reproduces_published'].sum())}/{len(d)}")
    print(f"  exact enumeration: {int(d.perm_method.eq('exact').sum())}; "
          f"Monte Carlo: {int(d.perm_method.eq('monte_carlo').sum())}")

    print("\n=== how many genes were testable, by recurrence ===")
    print(d.groupby("recurrence")
          .agg(genes=("gene_name", "size"), loci=("inv_id", "nunique"))
          .to_string())

    was = set(map(tuple, d.loc[d.jackknife_q < 0.05, ["gene_name", "inv_id"]].to_numpy()))
    for lab, col in (("BH", "perm_q_bh"), ("BY", "perm_q_by")):
        sig = d[d[col] < 0.05]
        now = set(map(tuple, sig[["gene_name", "inv_id"]].to_numpy()))
        print(f"\n=== permutation-significant at {lab} q<0.05: {len(sig)} genes, "
              f"{sig.inv_id.nunique()} loci ===")
        print(sig.groupby("recurrence").size().to_string() or "  (none)")
        print(f"  of the {len(was)} original jackknife genes, {len(was & now)} survive")

    print("\n=== jackknife boundary artefacts ===")
    b = d[d.jackknife_p <= 1e-299]
    print(f"  genes with jackknife p at the boundary branch: {len(b)}; "
          f"of these, permutation q_BY<0.05: {int((b.perm_q_by < 0.05).sum())}")
    print(f"  their inverted-group haplotype counts: {sorted(b.k_inverted.unique())}")
    print(f"\nWrote {OUT_TSV}")


if __name__ == "__main__":
    main()
