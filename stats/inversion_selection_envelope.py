"""Selection on the INVERSION itself: a neutral-envelope test.

The selected unit is the arrangement, not a site. Under neutrality an allele's
age and its frequency are coupled: common alleles are old, young alleles are
rare. An inversion that is too YOUNG for its frequency rose faster than drift
allows (sweep-like positive selection); one whose total sojourn is too LONG
for a segregating polymorphism has been kept alive (balancing selection).
Both ages are readable from the whole-region alignments:

  pi_inv   diversity within the inverted class   ~ age of the inverted clade
  pi_dir   diversity within the direct class     ~ the local coalescent clock
  d_cross  divergence between the classes        ~ depth of the split between
                                                   the two arrangement classes

d_cross is NOT the age of the inversion mutation: the mutation can fall
anywhere along the stem subtending the carrier clade, while cross-class
coalescence happens above that stem. So B bounds the sojourn from above and
should be read as "how deep is the between-arrangement genealogy relative to
local within-arrangement diversity", not as "the arrangement is B times older
than the clock". The null is built from the same quantity, so the p-value is
unaffected by that distinction -- only the prose is.

Statistics are clock-normalized (A = pi_inv/pi_dir, B = d_cross/pi_dir) so
locus mutation rate cancels; genealogy variance, which dominates, is fully
modeled by the null.

Null: the inversion is a neutral mutation with k carriers among n sampled
haplotypes. Conditioned-branch coalescent: simulate Kingman genealogies of n
lineages; every branch with EXACTLY k descendants (window=0, the default) is a
candidate origin, weighted by its length (mutation opportunity). For each
candidate, mutations are dropped Poisson on every branch at a rate calibrated
so the direct class's expected diversity matches its observed value, and A and
B are recomputed from the mutated sample. One-sided envelope p-values:

  p_sweep     = P_null(A <= A_obs)   too little internal diversity
  p_balance   = P_null(B >= B_obs)   too much cross-arrangement divergence

Three inference properties this file is explicit about, because earlier
versions were not:

  * EXACT k. `window` widens the accepted descendant count to k +- round(wn);
    it defaults to 0, so the null conditions on the observed carrier count
    exactly. Widening is anti-conservative for p_balance -- lower-k branches
    are both shallower and (since E[branch length subtending k] ~ 1/k) more
    heavily weighted, so they drag the null's B distribution down. Use
    --window only as a sensitivity check, never for the headline number.
  * TAIL MONTE CARLO ERROR IS REPORTED. Candidates are neither independent
    (many come from one tree) nor equally weighted, so the raw candidate count
    overstates precision. Every p-value ships with a tree-level bootstrap 95%
    CI, the weighted effective sample size, and the number of candidates
    actually in the tail. A p-value whose CI spans an order of magnitude is a
    Monte Carlo statement, not an evidential one.
  * MULTIPLE TESTING. Loci are corrected across the whole screen (BH q-values
    per axis, in the output table). A nominal p is never the headline.

Primary scope: single-event inversions (one origin = one branch; the model is
exact for them). Recurrent loci are reported but flagged: multiple origins
violate the one-branch null and bias A upward (toward neutral/old), so their
p_sweep is conservative.

Second axis, local adaptation: sample names carry superpopulations. Hudson
Fst of the inversion across superpops, ranked against the Fst of
frequency-matched SNPs from the same locus alignment; high percentile =
geographic differentiation beyond locus-typical drift.

Caveats stated once: no gene flux in the null (irrelevant for single-event;
conservative for recurrent); population structure inflates all deep splits,
partially absorbed by the pi_dir normalization; 44 samples make per-pop
frequencies coarse.

Output: results/inversion_selection_envelope.tsv + printed summary.
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cds_selection_intron_control import INV_RE, encode, read_phy  # noqa: E402

N_CAND = 20_000        # weighted candidate branches per locus
MAX_TREES = 3_000_000  # exact-k branches are rare at intermediate k
N_BOOT = 1000          # tree-level bootstrap draws for the p-value CI
RNG_SEED = 2026
MIN_K = 3              # need >= 3 haplotypes per class for within-diversity


def single_origin_loci(repo_data="repo/data"):
    """The loci this test is ENTITLED to run on: confirmed single-origin.

    The null represents the inversion as one branch of one genealogy. That is
    exact for a single-origin arrangement and simply false for a recurrent one,
    whose carriers are a mixture of independently arisen lineages. It is equally
    unusable for a locus whose recurrence status is unknown, because then the
    premise is unverified rather than satisfied.

    Earlier versions computed p-values for every locus with enough haplotypes
    and left the caveat in prose. That was wrong twice over: it published
    numbers whose model does not describe the data, and it then charged the
    valid locus a multiple-testing penalty for the invalid ones. This function
    is the gate, so no p-value can be produced outside the premise.

    Source of truth: data/inv_properties.tsv, column
    `0_single_1_recur_consensus` (0 = single event, 1 = recurrent, NA =
    unclassified). Returns the set of "chr:start-end" ids with a 0."""
    path = os.path.join(repo_data, "inv_properties.tsv")
    props = pd.read_csv(path, sep="\t")
    need = {"Chromosome", "Start", "End", "0_single_1_recur_consensus"}
    missing = need - set(props.columns)
    if missing:
        raise SystemExit(f"{path} lacks {sorted(missing)}; cannot establish "
                         f"the single-origin premise, so refusing to run")
    # Compare NUMERICALLY. pandas reads this column as float because of the NA
    # rows, so a string comparison sees "0.0" and silently matches nothing --
    # which is exactly how an earlier version admitted zero loci and looked like
    # a coordinate-format problem.
    consensus = pd.to_numeric(props["0_single_1_recur_consensus"],
                              errors="coerce")
    keep = props[consensus == 0]
    out = set()
    for _, r in keep.iterrows():
        try:
            out.add(f"{str(r['Chromosome']).replace('chr', '')}:"
                    f"{int(float(r['Start']))}-{int(float(r['End']))}")
        except (ValueError, TypeError):
            continue
    if not out:
        raise SystemExit(
            f"{path}: no locus has 0_single_1_recur_consensus == 0. Refusing "
            f"to continue -- an empty premise set means the gate is broken, "
            f"not that no inversion qualifies.")
    return out


def parse_demography(spec):
    """Build an inverse time change t(tau) for a population size history.

    A variable-size coalescent is a TIME-CHANGED standard coalescent: topology
    is untouched and only the times move, with tau'(t) = 1/nu(t) for relative
    size nu (SauersML/Descent, Descent/Coalescent/VariableSize.lean --
    `deriv_timeChange`, tau' = 1/lambda, and `timeChange beta t = (e^{bt}-1)/b`
    for the exponential history). So the standard-coalescent sampler below stays
    exactly as it is, and demography enters ONLY by mapping simulated standard
    times back to real times. This makes arbitrary demography exact rather than
    something the forward simulations have to defend.

    Only the SHAPE of nu matters here: the mutation layer calibrates its rate to
    each locus's observed pi_dir, so any constant factor on nu cancels.

      "const"            nu == 1 (the standard coalescent; identity map)
      "exp:BETA"         nu(t) = exp(-BETA*t), growing forward in time.
                         tau(t) = (e^{Bt}-1)/B, so t(tau) = ln(1+B*tau)/B.
      "pw:t1,nu1;t2,nu2" piecewise-constant nu, breakpoints in REAL coalescent
                         time (units of 2N0 generations), first epoch starts at
                         0. Fully general in the limit.

    Returns (callable on an array of standard times, label)."""
    spec = (spec or "const").strip()
    if spec in ("", "const", "none"):
        return (lambda tau: tau), "const"
    if spec.startswith("exp:"):
        beta = float(spec.split(":", 1)[1])
        if beta == 0:
            return (lambda tau: tau), "const"
        return (lambda tau: np.log1p(beta * tau) / beta), spec
    if spec.startswith("pw:"):
        parts = [p for p in spec[3:].split(";") if p.strip()]
        ts, nus = [0.0], []
        for p in parts:
            a, b = p.split(",")
            ts.append(float(a))
            nus.append(float(b))
        # nus has one entry per epoch START; the list above gives epoch
        # boundaries t1..tm, so we need m+1 sizes: first from the first pair.
        if len(nus) != len(ts) - 1:
            raise ValueError(f"bad demography spec: {spec}")
        # epoch j covers [ts[j], ts[j+1]) with size nus[j]; last extends to inf
        t_start = np.array(ts[:-1] + [ts[-1]], dtype=float)
        nu = np.array(nus + [nus[-1]], dtype=float)
        if np.any(nu <= 0):
            raise ValueError("relative sizes must be positive")
        # tau at each epoch start
        tau_start = np.zeros_like(t_start)
        for j in range(1, len(t_start)):
            tau_start[j] = (tau_start[j - 1]
                            + (t_start[j] - t_start[j - 1]) / nu[j - 1])

        def inv(tau):
            tau = np.asarray(tau, dtype=float)
            j = np.clip(np.searchsorted(tau_start, tau, side="right") - 1,
                        0, len(t_start) - 1)
            return t_start[j] + (tau - tau_start[j]) * nu[j]
        return inv, spec
    raise ValueError(f"unknown demography spec: {spec}")


def expected_exact_k_length(k):
    """E[total branch length subtending exactly k of n leaves] = 2/k.

    Independent of n. This is Fu's spectrum branch length, proved in
    SauersML/Descent (Descent/Coalescent/SiteFrequencySpectrum.lean --
    `spectrumBranchLength i = 2/i`, cross-checked there against the total tree
    length by `sum_spectrumBranchLength`). Our importance weights ARE these
    branch lengths, so this is an exact analytic unit test of the sampler that
    generates the null: no simulation needed to know the right answer."""
    return 2.0 / float(k)


def group_stats(mat, mask_a, mask_b):
    """Per-bp AND absolute (mean diffs/pair) pi within a, b, and between."""
    def counts(mask):
        sub = mat[mask]
        ok = sub != 255
        c = np.stack([((sub == v) & ok).sum(axis=0) for v in range(4)])
        return c, ok.sum(axis=0)
    ca, na = counts(mask_a)
    cb, nb = counts(mask_b)

    def pi_within(c, nok):
        pairs = nok * (nok - 1) / 2.0
        same = (c * (c - 1) / 2.0).sum(axis=0)
        good = pairs > 0
        perbp = float((pairs[good] - same[good]).sum() / pairs[good].sum())
        # absolute: mean pairwise difference count, scaled to full-size pairs
        absd = float(((pairs[good] - same[good]) / pairs[good]).sum())
        return perbp, absd

    def pi_between(ca, na, cb, nb):
        pairs = na * nb
        same = (ca * cb).sum(axis=0)
        good = pairs > 0
        perbp = float((pairs[good] - same[good]).sum() / pairs[good].sum())
        absd = float(((pairs[good] - same[good]) / pairs[good]).sum())
        return perbp, absd

    return pi_within(ca, na), pi_within(cb, nb), pi_between(ca, na, cb, nb)


def conditioned_branch_null(n, k, obs_dir_abs, rng,
                            n_cand=N_CAND, max_trees=MAX_TREES,
                            window=0.0, tmap=None):
    """Sample (A, B) under the neutral conditioned-branch model WITH an
    infinite-sites mutation layer.

    For each candidate inversion branch, mutations are dropped Poisson on
    every branch of the tree at a rate calibrated so the DIRECT class's
    expected diversity equals its observed absolute diversity (mean pairwise
    difference count). A and B are then computed from the mutated sample, so
    small loci are honestly noisy in the null too.

    `window` widens the accepted descendant count to k +- round(window*n);
    window=0 (the default) conditions on EXACTLY k carriers.

    `tmap` is the inverse time change from parse_demography: node heights are
    simulated in standard-coalescent time and mapped through it before branch
    lengths are taken, which makes the null exact under that size history
    (topology is demography-invariant; only times change). None = constant size.

    Returns arrays A (pi_inv/pi_dir), B (d_cross/pi_dir), weights, tree_id,
    n_trees. tree_id is the index of the genealogy each candidate came from --
    candidates sharing a tree are not independent, so it is the resampling unit
    for the p-value's Monte Carlo CI."""
    win = int(round(window * n))
    lo, hi = max(2, k - win), min(n - 2, k + win)
    A, B, W, T = [], [], [], []
    trees = 0
    while len(A) < n_cand and trees < max_trees:
        trees += 1
        # one Kingman tree: record every branch (members, length). Members stay
        # plain lists here -- materializing bool masks for every branch of every
        # tree dominated the runtime, and most trees carry no exact-k candidate.
        alive = {i: ([i], 0.0) for i in range(n)}
        branches = []          # (members list, child height, parent height)
        candidates = []        # index into branches
        t = 0.0
        next_id = n
        ids = list(range(n))
        while len(ids) > 1:
            m = len(ids)
            t += rng.exponential(2.0 / (m * (m - 1)))
            i = int(rng.integers(m))
            j = int(rng.integers(m - 1))
            if j >= i:
                j += 1
            a, b = ids[i], ids[j]
            ma, ta = alive.pop(a)
            mb, tb = alive.pop(b)
            for mem, t0 in ((ma, ta), (mb, tb)):
                branches.append((mem, t0, t))
                if lo <= len(mem) <= hi:
                    candidates.append(len(branches) - 1)
            alive[next_id] = (ma + mb, t)
            ids = [x for x in ids if x not in (a, b)]
            ids.append(next_id)
            next_id += 1
        if not candidates:
            continue
        sels = np.zeros((len(branches), n), dtype=bool)     # (B, n)
        for bi, (mem, _, _) in enumerate(branches):
            sels[bi, mem] = True
        # heights in standard-coalescent time, then mapped to real time if a
        # demography was supplied. Lengths are always taken AFTER the map.
        h_child = np.array([c for _, c, _ in branches])
        h_par = np.array([p for _, _, p in branches])
        if tmap is not None:
            h_child, h_par = tmap(h_child), tmap(h_par)
        lens = h_par - h_child
        for ci in candidates:
            C = sels[ci]
            D = ~C
            kk = int(C.sum())
            nD = n - kk
            # per-branch pair weights
            inC = (sels & C).sum(axis=1)
            inD = (sels & D).sum(axis=1)
            w_inv = inC * (kk - inC)
            w_dir = inD * (nD - inD)
            w_cross = inC * (nD - inD) + inD * (kk - inC)
            pairs_inv = kk * (kk - 1) / 2.0
            pairs_dir = nD * (nD - 1) / 2.0
            pairs_cross = float(kk * nD)
            denom = (lens * w_dir).sum() / pairs_dir
            if denom <= 0:
                continue
            c_rate = obs_dir_abs / denom
            K = rng.poisson(c_rate * lens)
            pi_dir_sim = (K * w_dir).sum() / pairs_dir
            if pi_dir_sim <= 0:
                continue
            pi_inv_sim = (K * w_inv).sum() / pairs_inv
            d_cross_sim = (K * w_cross).sum() / pairs_cross
            A.append(pi_inv_sim / pi_dir_sim)
            B.append(d_cross_sim / pi_dir_sim)
            W.append(lens[ci])
            T.append(trees - 1)
    return (np.array(A), np.array(B), np.array(W),
            np.array(T, dtype=np.int64), trees)


def envelope_p(stat, obs, weights, tree_id, tail, rng, n_boot=N_BOOT):
    """Weighted one-sided envelope p-value with its Monte Carlo error.

    The candidates are branch-length-weighted and clustered by genealogy, so
    neither len(stat) nor the unweighted tail count describes the precision.
    Reported instead:

      p     weighted tail mass, add-one corrected (as before)
      lo/hi 95% bootstrap CI, resampling TREES (the independent unit) -- this
            captures both the weight imbalance and the within-tree correlation
      ess   weighted effective sample size, (sum w)^2 / sum w^2
      ntail number of candidate branches actually in the tail

    Bootstrapping is done on per-tree weight sums, so cost is O(n_boot * trees)
    scalar work rather than a resample of every candidate.

    On the add-one: the conventional (x+1)/(m+1) floor divides by the number of
    candidates, which silently claims m INDEPENDENT draws. With importance
    weights and several branches per tree the independent-draw count is the
    weighted ESS, not m -- and m overstates it by more than an order of
    magnitude here, manufacturing a floor far below the actual resolution. The
    floor is therefore applied on the ESS scale, and the unfloored weighted tail
    fraction is reported alongside it so nothing is hidden."""
    hit = (stat <= obs + 1e-12) if tail == "lower" else (stat >= obs - 1e-12)
    m = len(stat)
    tot = weights.sum()
    nan = float("nan")
    if m == 0 or tot <= 0:
        return {"p": nan, "p_raw": nan, "lo": nan, "hi": nan, "mcse": nan,
                "ess": 0.0, "n_tree": 0, "n_tail": 0}

    ess = float(tot ** 2 / (weights ** 2).sum())
    p_raw = float(weights[hit].sum() / tot)
    p = (p_raw * ess + 1.0) / (ess + 1.0)

    nt = int(tree_id.max()) + 1
    w_tree = np.bincount(tree_id, weights=weights, minlength=nt)
    s_tree = np.bincount(tree_id, weights=weights * hit, minlength=nt)
    keep = w_tree > 0
    w_tree, s_tree = w_tree[keep], s_tree[keep]
    n_tree = len(w_tree)

    # cluster-robust MC standard error of the ratio estimator, clustered by tree
    mcse = float(np.sqrt(((s_tree - p_raw * w_tree) ** 2).sum()) / tot)

    boots = np.empty(n_boot)
    for b in range(n_boot):
        c = rng.integers(0, n_tree, n_tree)
        wsum = w_tree[c].sum()
        boots[b] = s_tree[c].sum() / wsum if wsum > 0 else np.nan
    boots = boots[np.isfinite(boots)]
    boots = (boots * ess + 1.0) / (ess + 1.0)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) else (nan, nan))
    return {"p": p, "p_raw": p_raw, "lo": float(lo), "hi": float(hi),
            "mcse": mcse, "ess": ess, "n_tree": n_tree,
            "n_tail": int(hit.sum())}


def selftest_exact_k(n, ks, trees, rng):
    """Check the sampler's E[L_k] against the analytic 2/k.

    Runs the SAME tree code the null uses, accumulates total branch length
    subtending exactly k leaves, and compares the mean over ALL trees (including
    trees with no such branch) to expected_exact_k_length(k). A sampler that
    mis-weights the conditioning would fail here; nothing about this test is
    simulation-calibrated, so it is a real check and not a consistency loop."""
    print(f"analytic self-test: E[L_k] vs 2/k, n={n}, {trees} trees per k")
    print(f"{'k':>4} {'simulated':>12} {'analytic 2/k':>13} {'ratio':>8} "
          f"{'mc_se':>9}")
    rows = []
    for k in ks:
        tot = np.zeros(trees)
        for r in range(trees):
            alive = {i: ([i], 0.0) for i in range(n)}
            acc = 0.0
            t = 0.0
            next_id = n
            ids = list(range(n))
            while len(ids) > 1:
                m = len(ids)
                t += rng.exponential(2.0 / (m * (m - 1)))
                i = int(rng.integers(m))
                j = int(rng.integers(m - 1))
                if j >= i:
                    j += 1
                a, b = ids[i], ids[j]
                ma, ta = alive.pop(a)
                mb, tb = alive.pop(b)
                for mem, t0 in ((ma, ta), (mb, tb)):
                    if len(mem) == k:
                        acc += t - t0
                alive[next_id] = (ma + mb, t)
                ids = [x for x in ids if x not in (a, b)]
                ids.append(next_id)
                next_id += 1
            tot[r] = acc
        sim = float(tot.mean())
        se = float(tot.std(ddof=1) / np.sqrt(trees))
        ana = expected_exact_k_length(k)
        print(f"{k:>4} {sim:>12.5f} {ana:>13.5f} {sim / ana:>8.4f} "
              f"{se:>9.5f}")
        rows.append((k, sim, ana, se))
    worst = max(abs(s - a) / max(e, 1e-12) for _k, s, a, e in rows)
    print(f"\nworst deviation: {worst:.2f} Monte Carlo SEs "
          f"({'PASS' if worst < 4 else 'FAIL'})")
    return rows


def clock_flags(inv_ids, b_vals, pi_dir_vals, ok_mask, per_bp=False):
    """Flag loci where the pi_dir clock normalization is not trustworthy.

    Both statistics divide by pi_dir, so they assume the direct class is one
    ordinary-depth family at this locus. Two ways that fails, both learned the
    hard way at chr7:70.96Mb, where a hidden ancient split inside the DIRECT
    class inflated the clock and produced p_sweep = 1e-3 for a locus with
    nothing going on:

      B<1        cross-arrangement divergence BELOW within-direct diversity.
                 Structurally impossible for a single-origin arrangement, so the
                 direct class must contain lineages deeper than the split itself.
      pi_dir_hi  per-bp direct-class diversity far above the screen median --
                 same disease, milder presentation.

    A flagged sweep call means nothing until the inverted class is compared
    against direct haplotypes on its OWN background. Returned as a string column
    so the flag travels with the row instead of living in someone's memory.

    `per_bp` says whether pi_dir_vals is ALREADY per-base-pair. The 1KG path
    stores absolute mean pairwise differences and needs dividing by the region
    length; the legacy filtered path stores per-bp directly and must not be
    divided again."""
    ids = np.asarray(inv_ids, dtype=object)
    B = np.asarray(b_vals, dtype=float)
    perbp = np.full(len(ids), np.nan)
    for i, iid in enumerate(ids):
        try:
            if per_bp:
                perbp[i] = float(pi_dir_vals[i])
                continue
            _c, se = str(iid).split(":")
            s, e = se.split("-")
            L = int(e) - int(s)
            if L > 0:
                perbp[i] = float(pi_dir_vals[i]) / L
        except (ValueError, TypeError):
            continue
    med = np.nanmedian(perbp[ok_mask]) if ok_mask.any() else np.nan
    out = []
    for i in range(len(ids)):
        f = []
        if np.isfinite(B[i]) and B[i] < 1.0:
            f.append("B<1")
        if np.isfinite(perbp[i]) and np.isfinite(med) and perbp[i] > 2.0 * med:
            f.append("pi_dir_hi")
        out.append(";".join(f))
    return out, perbp, med


def bh_q(pvals):
    """Benjamini-Hochberg q-values, NaN-safe, order preserved."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    idx = np.nonzero(np.isfinite(p))[0]
    if not len(idx):
        return q
    ps = p[idx]
    order = np.argsort(ps)
    m = len(ps)
    ranked = ps[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[idx[order]] = np.minimum(ranked, 1.0)
    return q


def hudson_fst(c1, n1, c2, n2):
    """Hudson 1992 Fst for one biallelic site from derived counts/sizes."""
    if n1 < 2 or n2 < 2:
        return np.nan
    p1, p2 = c1 / n1, c2 / n2
    num = (p1 - p2) ** 2 - p1 * (1 - p1) / (n1 - 1) - p2 * (1 - p2) / (n2 - 1)
    den = p1 * (1 - p2) + p2 * (1 - p1)
    return num / den if den > 0 else np.nan


def locus_fst_axis(mat, inv_mask, pops):
    """Inversion Fst across superpops vs frequency-matched SNP Fst."""
    sp = np.array(pops)
    use_pops = [p for p in np.unique(sp) if (sp == p).sum() >= 8]
    if len(use_pops) < 2:
        return np.nan, np.nan, 0
    def multi_fst(carrier_mask):
        nums, dens = 0.0, 0.0
        for i in range(len(use_pops)):
            for j in range(i + 1, len(use_pops)):
                m1, m2 = sp == use_pops[i], sp == use_pops[j]
                f = hudson_fst(int(carrier_mask[m1].sum()), int(m1.sum()),
                               int(carrier_mask[m2].sum()), int(m2.sum()))
                if np.isfinite(f):
                    nums += f; dens += 1
        return nums / dens if dens else np.nan
    fst_inv = multi_fst(inv_mask)
    f_glob = inv_mask.mean()

    # frequency-matched SNPs from the same alignment
    ok = mat != 255
    fsts = []
    step = max(1, mat.shape[1] // 20000)
    for jcol in range(0, mat.shape[1], step):
        col = mat[:, jcol]
        okj = col != 255
        if okj.sum() < 0.8 * len(col):
            continue
        vals, cnts = np.unique(col[okj], return_counts=True)
        if len(vals) != 2:
            continue
        der = vals[np.argmin(cnts)]
        fr = (col == der).sum() / okj.sum()
        if abs(min(fr, 1 - fr) - min(f_glob, 1 - f_glob)) > 0.05:
            continue
        fsts.append(multi_fst(col == der))
    fsts = np.array([f for f in fsts if np.isfinite(f)])
    if len(fsts) < 20 or not np.isfinite(fst_inv):
        return fst_inv, np.nan, len(fsts)
    pct = (fsts < fst_inv).mean()
    return fst_inv, pct, len(fsts)


def process_locus(args):
    key, files, recurrence, seed, window, demography = args
    chrom, inv_s, inv_e = key
    rng = np.random.default_rng(seed)
    seq_dir = read_phy(files["0"])
    seq_inv = read_phy(files["1"])
    names = sorted(seq_dir) + sorted(seq_inv)
    inv_mask = np.array([n in seq_inv for n in names])
    n, k = len(names), int(inv_mask.sum())
    if k < MIN_K or (n - k) < MIN_K:
        return {"inv_id": f"{chrom}:{inv_s}-{inv_e}", "n": n, "k_inv": k,
                "recurrence": recurrence, "status": "TOO_FEW"}
    mat = np.vstack([encode((seq_dir | seq_inv)[nm]) for nm in names])
    (pi_inv, pi_inv_abs), (pi_dir, pi_dir_abs), (d_cross, d_cross_abs) = \
        group_stats(mat, inv_mask, ~inv_mask)
    if pi_dir <= 0:
        return {"inv_id": f"{chrom}:{inv_s}-{inv_e}", "n": n, "k_inv": k,
                "recurrence": recurrence, "status": "NO_DIVERSITY"}
    A_obs, B_obs = pi_inv_abs / pi_dir_abs, d_cross_abs / pi_dir_abs

    tmap, _demlab = parse_demography(demography)
    A, B, W, Tid, trees = conditioned_branch_null(
        n, k, pi_dir_abs, rng, window=window,
        tmap=None if demography in ("const", "", None) else tmap)
    mc = {}
    if len(A) < 200:
        status = "NULL_STARVED"
        p_sweep = p_bal = np.nan
    else:
        status = "OK"
        sw = envelope_p(A, A_obs, W, Tid, "lower", rng)
        bal = envelope_p(B, B_obs, W, Tid, "upper", rng)
        p_sweep, p_bal = sw["p"], bal["p"]
        mc = {"p_sweep_raw": sw["p_raw"], "p_sweep_mcse": sw["mcse"],
              "p_sweep_lo": sw["lo"], "p_sweep_hi": sw["hi"],
              "p_balance_raw": bal["p_raw"], "p_balance_mcse": bal["mcse"],
              "p_balance_lo": bal["lo"], "p_balance_hi": bal["hi"],
              "null_ess": sw["ess"], "null_trees": sw["n_tree"],
              "n_tail_sweep": sw["n_tail"], "n_tail_balance": bal["n_tail"]}

    pops = [nm.split("_")[0] for nm in names]
    fst_inv, fst_pct, n_matched = locus_fst_axis(mat, inv_mask, pops)

    return {"inv_id": f"{chrom}:{inv_s}-{inv_e}", "n": n, "k_inv": k,
            "recurrence": recurrence, "status": status,
            "pi_inv": pi_inv, "pi_dir": pi_dir, "d_cross": d_cross,
            "A_ageratio": A_obs, "B_crossdepth": B_obs,
            "null_candidates": int(len(A)), "null_window": window,
            "null_demography": demography,
            "p_sweep": p_sweep, "p_balance": p_bal, **mc,
            "fst_inv": fst_inv, "fst_percentile": fst_pct,
            "n_matched_snps": n_matched}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--window", type=float, default=0.0,
                    help="carrier-count conditioning window as a fraction of "
                         "n; 0 (default) = exact k. Non-zero is a sensitivity "
                         "check only -- it biases p_balance anti-conservative.")
    ap.add_argument("--out", default="results/inversion_selection_envelope.tsv")
    ap.add_argument("--demography", default="const",
                    help="size history for the null: const | exp:BETA | "
                         "pw:t1,nu1;t2,nu2;... (see parse_demography)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the analytic E[L_k]=2/k check and exit")
    a = ap.parse_args()
    if a.selftest:
        selftest_exact_k(80, [3, 5, 9, 20, 40], 4000,
                         np.random.default_rng(RNG_SEED))
        return
    if not a.workdir:
        ap.error("--workdir is required unless --selftest")
    os.chdir(a.workdir)
    os.makedirs("results", exist_ok=True)

    cip = pd.read_csv("repo/data/cds_identical_proportions.tsv", sep="\t")
    cip["inv_id"] = (cip["chr"].astype(str).str.replace("chr", "", regex=False)
                     + ":" + cip["inv_start"].astype(int).astype(str)
                     + "-" + cip["inv_end"].astype(int).astype(str))
    recurrence = (cip.groupby("inv_id")["consensus"].first()
                  .map({0: "single-event", 1: "recurrent"}).to_dict())
    # broader recurrence source if available
    try:
        props = pd.read_csv("repo/data/inv_properties.tsv", sep="\t")
        if {"chr", "region_start", "region_end",
                "0_single_1_recur_consensus"} <= set(props.columns):
            for _, r in props.iterrows():
                iid = (str(r["chr"]).replace("chr", "") + ":"
                       + str(int(r["region_start"])) + "-"
                       + str(int(r["region_end"])))
                recurrence.setdefault(
                    iid, {0: "single-event", 1: "recurrent"}.get(
                        r["0_single_1_recur_consensus"], "unknown"))
    except Exception:
        pass

    region_files = {}
    for fn in os.listdir("phy_outputs"):
        m = INV_RE.match(fn)
        if m:
            region_files.setdefault(
                (m["chrom"], int(m["s"]), int(m["e"])), {})[m["grp"]] = \
                os.path.join("phy_outputs", fn)

    # HARD GATE, same as the 1KG path: the one-branch null describes confirmed
    # single-origin inversions only. Recurrent and unclassified loci are dropped
    # before any statistic exists, so no invalid p-value can be reported or
    # enter a multiple-testing family.
    allowed = single_origin_loci("repo/data")
    tasks, skipped = [], 0
    for i, (key, files) in enumerate(sorted(region_files.items())):
        if set(files) != {"0", "1"}:
            continue
        iid = f"{key[0]}:{key[1]}-{key[2]}"
        if iid not in allowed:
            skipped += 1
            continue
        tasks.append((key, files, recurrence.get(iid, "single-event"),
                      RNG_SEED + i, a.window, a.demography))
    print(f"single-origin premise: {len(tasks)} loci admitted, {skipped} "
          f"recurrent/unclassified loci excluded before any statistic",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, res in enumerate(ex.map(process_locus, tasks), 1):
            rows.append(res)
            if res.get("status") == "OK":
                print(f"  {res['inv_id']} [{res['recurrence']}] "
                      f"k={res['k_inv']}/{res['n']} A={res['A_ageratio']:.3f} "
                      f"B={res['B_crossdepth']:.3f} "
                      f"p_sweep={res['p_sweep']:.4f} "
                      f"[{res['p_sweep_lo']:.4f},{res['p_sweep_hi']:.4f}] "
                      f"p_bal={res['p_balance']:.4f} "
                      f"[{res['p_balance_lo']:.4f},{res['p_balance_hi']:.4f}] "
                      f"ess={res['null_ess']:.0f} "
                      f"tail={res['n_tail_balance']} "
                      f"fst_pct={res.get('fst_percentile', float('nan')):.2f}",
                      flush=True)
            if i % 25 == 0:
                print(f"... {i}/{len(tasks)}", flush=True)

    out = pd.DataFrame(rows)
    # multiple testing across the screen, per axis, over testable loci only
    okm = out["status"].eq("OK")
    for src, dst in (("p_sweep", "q_sweep_bh"), ("p_balance", "q_balance_bh")):
        out[dst] = np.nan
        out.loc[okm, dst] = bh_q(out.loc[okm, src].to_numpy())
    # same clock guard the 1KG path carries; pi_dir here is already per-bp
    nanv = np.full(len(out), np.nan)
    flags, perbp, med = clock_flags(
        out["inv_id"].tolist(),
        out["B_crossdepth"].to_numpy() if "B_crossdepth" in out else nanv,
        out["pi_dir"].to_numpy() if "pi_dir" in out else nanv,
        okm.to_numpy(), per_bp=True)
    out["pi_dir_perbp"] = perbp
    out["clock_flag"] = flags
    out.to_csv(a.out, sep="\t", index=False)

    ok = out[out["status"] == "OK"]
    se = ok[ok["recurrence"] == "single-event"]
    print("\n=============== ENVELOPE SUMMARY ===============")
    print(f"conditioning window: {a.window} "
          f"({'EXACT k' if a.window == 0 else 'WIDENED -- sensitivity only'})")
    print(f"testable loci: {len(ok)} of {len(out)}; single-event: {len(se)}")
    print(f"null resolution: median candidates {ok.null_candidates.median():.0f}"
          f", median ESS {ok.null_ess.median():.0f}, median trees "
          f"{ok.null_trees.median():.0f}")
    for name, d in (("single-event", se), ("all testable", ok)):
        if not len(d):
            continue
        print(f"\n[{name}]")
        print(f"  sweep-like    p<0.05: {(d.p_sweep < 0.05).sum()}  "
              f"BH q<0.05: {(d.q_sweep_bh < 0.05).sum()}")
        print(f"  balancing     p<0.05: {(d.p_balance < 0.05).sum()}  "
              f"BH q<0.05: {(d.q_balance_bh < 0.05).sum()}")
        print(f"  fst_percentile>0.95: "
              f"{(d.fst_percentile > 0.95).sum()} of "
              f"{d.fst_percentile.notna().sum()} with Fst axis")
    cols = ["inv_id", "recurrence", "k_inv", "n", "A_ageratio", "B_crossdepth",
            "p_sweep", "p_sweep_lo", "p_sweep_hi", "q_sweep_bh",
            "p_balance", "p_balance_lo", "p_balance_hi", "q_balance_bh",
            "null_ess", "n_tail_balance", "fst_percentile"]
    cols = [c for c in cols if c in ok.columns]
    print("\nmost extreme (by p_sweep):")
    print(ok.sort_values("p_sweep")[cols].head(10).to_string(index=False))
    print("\nmost extreme (by p_balance):")
    print(ok.sort_values("p_balance")[cols].head(10).to_string(index=False))
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
