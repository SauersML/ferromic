#!/usr/bin/env python3
"""EXPERIMENT: flank-relative derived-orientation caller + benchmark.

Root cause (proven): the production caller reads the ABSOLUTE alignment strand of
the inversion interior, which equals each outgroup's local assembly/synteny
orientation -- not the inversion state. So outgroups assembled in opposite local
orientation give spurious "inverted" calls and disagree ~65% of the time.

Fix tested here: for each outgroup, compare the dominant orthologous chain strand of
the inversion INTERIOR to the dominant chain strand of the flanking syntenic
BACKBONE (left+right flanks, outside the breakpoints). The inversion is present
(reference derived w.r.t. that outgroup) iff interior strand != backbone strand.
This is invariant to assembly orientation, so outgroups should now agree.

Usage:
  FOURFOLD set not needed. Provide chain dir via --aln (dir with hg38.<sp>.all.chain.gz).
  python stats/polarize_v2_experiment.py --aln <dir> [--margin 10000 --flank 100000 --trim-frac 0.10 --trim-max 20000]
"""
from __future__ import annotations
import argparse, gzip, json, os, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data")

# Phylogenetic depth from human. Four catarrhine outgroups: deeper ones root the
# call under depth-weighted parsimony. (Marmoset/gibbon/mouse were tested but added
# noise at SD loci without resolving deep-toggling inversions like 17q21.31, whose
# entire catarrhine clade has re-inverted -- those need SNP-divergence, not synteny.)
OUTGROUPS = [
    {"name": "chimp",     "chain": "hg38.panTro6.all.chain.gz",  "depth": 1},
    {"name": "gorilla",   "chain": "hg38.gorGor6.all.chain.gz",  "depth": 2},
    {"name": "orangutan", "chain": "hg38.ponAbe3.all.chain.gz",  "depth": 3},
    {"name": "macaque",   "chain": "hg38.rheMac10.all.chain.gz", "depth": 4},
]
REAL_CHR_RE = re.compile(r"^chr([0-9]+|[XY])$", re.IGNORECASE)
MIN_BP = 200
DOM_RATIO = 1.5   # dominant chain must beat the runner-up by this factor to be "strong"


def norm_chrom(raw):
    c = str(raw).strip()
    while c.lower().startswith("chr"):
        c = c[3:]
    cl = c.lower()
    core = {"x": "X", "y": "Y"}.get(cl, str(int(cl)) if cl.isdigit() else c.upper())
    return "chr" + core


def chain_blocks(path):
    """Yield (chrom, tstart1, tend1, qStrand, chain_id) per ungapped block. Target
    is always + and 0-based half-open; query strand gives the outgroup orientation."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="latin-1") as fh:
        t_chrom = None; q_strand = "+"; chain_id = 0; t_pos = 0
        for line in fh:
            s = line.strip()
            if not s:
                t_chrom = None; continue
            if s.startswith("chain"):
                f = s.split()
                t_chrom, t_pos = norm_chrom(f[2]), int(f[5])
                q_strand = f[9]
                chain_id = f[12] if len(f) > 12 else f[1]
                continue
            if t_chrom is None:
                continue
            parts = line.split()
            size = int(parts[0])
            if size > 0:
                yield (t_chrom, t_pos + 1, t_pos + size, q_strand, chain_id)
            # advance target position by block + dt gap
            dt = int(parts[1]) if len(parts) >= 3 else 0
            t_pos += size + dt


def dominant_by_interval(chain_path, intervals, BIN=100000):
    """intervals: list of (slot, chrom, s, e). Return slot -> {dom_strand,dom_bp,second_bp}."""
    index = defaultdict(lambda: defaultdict(list))
    for (slot, chrom, s, e) in intervals:
        if s < 1: s = 1
        for b in range((s - 1) // BIN, (e - 1) // BIN + 1):
            index[chrom][b].append((slot, s, e))
    per = {slot: {} for (slot, _, _, _) in intervals}   # slot -> chain_id -> [strand, bp]
    for (chrom, ts, te, q_strand, chain_id) in chain_blocks(chain_path):
        bins = index.get(chrom)
        if not bins or q_strand not in ("+", "-") or not REAL_CHR_RE.match(chrom):
            continue
        seen = None
        for b in range((ts - 1) // BIN, (te - 1) // BIN + 1):
            recs = bins.get(b)
            if not recs:
                continue
            for (slot, s, e) in recs:
                ov = min(te, e) - max(ts, s) + 1
                if ov <= 0:
                    continue
                if seen is None: seen = set()
                tag = (slot, chain_id)
                if tag in seen: continue
                seen.add(tag)
                d = per[slot].setdefault(chain_id, [q_strand, 0])
                d[1] += ov
    # return the raw {chain_id -> [strand, bp]} per slot so the caller can anchor to
    # the backbone CHAIN identity across flank/interior (not just dominant strand).
    return per


def load_catalog():
    """Catalog from data/inversion_polarity.tsv (chrom,start,end,inv_af,recurrence,old call)."""
    cat = {}
    with open(os.path.join(DATA, "inversion_polarity.tsv")) as fh:
        import csv
        for r in csv.DictReader(fh, delimiter="\t"):
            try:
                s, e = int(r["start"]), int(r["end"])
            except (ValueError, KeyError):
                continue
            af = r.get("inv_af_ref", "")
            try: af = float(af)
            except: af = None
            cat[r["inv_id"]] = {
                "chrom": norm_chrom(r["chrom"]), "start": s, "end": e, "inv_af": af,
                "recurrence": r.get("recurrence_class", "unknown"),
                "old_confidence": r.get("confidence", ""),
                "old_anc": r.get("ancestral_orientation", ""),
                "old_flip": r.get("flip_ref_polarity", ""),
            }
    return cat


def build_intervals(cat, margin, flank, trim_frac, trim_max):
    out = []
    for key, rec in cat.items():
        s, e = rec["start"], rec["end"]
        length = e - s + 1
        trim = int(min(trim_max, trim_frac * length))
        if length - 2 * trim < MIN_BP:
            trim = 0
        out.append(((key, "interior"), rec["chrom"], s + trim, e - trim))
        out.append(((key, "lflank"), rec["chrom"], s - margin - flank, s - margin))
        out.append(((key, "rflank"), rec["chrom"], e + margin, e + margin + flank))
    return out


def call_relative(interior, lflank, rflank):
    """Backbone-chain-anchored synteny-break detection.

    interior/lflank/rflank are {chain_id -> [strand, bp]}. Find the BACKBONE chain
    (most flank bp); its strand is the local synteny orientation. Then in the
    interior compare how much of the SAME backbone chain continues (collinear) vs
    how much an OPPOSITE-strand chain takes over (inversion). Returns
    (orientation, confidence, detail)."""
    interior = interior or {}; lflank = lflank or {}; rflank = rflank or {}
    # Backbone = the SPANNING syntenic chain: present in BOTH flanks, max total flank
    # bp. Requiring both flanks rejects one-sided paralogs and anchors true synteny.
    common = set(lflank) & set(rflank)
    if common:
        bb_id = max(common, key=lambda c: lflank[c][1] + rflank[c][1])
        bb_strand = lflank[bb_id][0]
        bb_flank_bp = lflank[bb_id][1] + rflank[bb_id][1]
        spanning = True
    else:
        # no chain bridges both flanks; fall back to strongest single-flank chain
        fl = {}
        for d in (lflank, rflank):
            for cid, (st, bp) in d.items():
                e = fl.setdefault(cid, [st, 0]); e[1] += bp
        if not fl:
            return None, "no_flank", {}
        bb_id = max(fl, key=lambda c: fl[c][1]); bb_strand = fl[bb_id][0]
        bb_flank_bp = fl[bb_id][1]; spanning = False
    if bb_flank_bp < MIN_BP:
        return None, "weak_flank", {}

    # Does the backbone chain CONTINUE through the interior (collinear), or break and
    # get replaced by an opposite-strand chain (inverted)? Bias to collinear unless
    # the backbone is largely absent AND a reverse chain clearly fills the interior --
    # this rejects inverted-SD/paralog alignments that align both ways.
    bb_int = interior.get(bb_id, [None, 0])[1]
    same_other = max([bp for cid, (st, bp) in interior.items()
                      if st == bb_strand and cid != bb_id] + [0])
    opp_bp = max([bp for cid, (st, bp) in interior.items() if st != bb_strand] + [0])
    same = max(bb_int, same_other)
    if same == 0 and opp_bp == 0:
        return None, "no_interior", {"backbone": bb_strand, "spanning": spanning}

    lf_strand = max(lflank.items(), key=lambda kv: kv[1][1])[1][0] if lflank else None
    rf_strand = max(rflank.items(), key=lambda kv: kv[1][1])[1][0] if rflank else None
    flank_agree = (lf_strand is not None and lf_strand == rf_strand)

    # Decision: backbone bridges -> collinear; backbone broken + reverse fill -> inverted.
    if bb_int >= MIN_BP:                      # spanning chain continues through interior
        orient = "collinear"; margin = same / (same + opp_bp) if (same + opp_bp) else 1.0
    elif opp_bp >= MIN_BP and opp_bp > same:  # backbone broke, reverse chain fills it
        orient = "inverted"; margin = opp_bp / (same + opp_bp) if (same + opp_bp) else 1.0
    elif same >= MIN_BP:                      # sibling same-strand chain continues
        orient = "collinear"; margin = same / (same + opp_bp) if (same + opp_bp) else 1.0
    else:
        return None, "weak_interior", {"backbone": bb_strand, "spanning": spanning}

    strong = spanning and flank_agree and margin >= 0.80 and max(same, opp_bp) >= MIN_BP
    detail = {"backbone": bb_strand, "bb_id": bb_id, "spanning": spanning,
              "bb_int": bb_int, "same": same, "opp": opp_bp, "margin": round(margin, 2),
              "flank_agree": flank_agree}
    return orient, ("strong" if strong else "weak"), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aln", required=True, help="dir with hg38.<sp>.all.chain.gz")
    ap.add_argument("--margin", type=int, default=10000)
    ap.add_argument("--flank", type=int, default=100000)
    ap.add_argument("--trim-frac", type=float, default=0.10)
    ap.add_argument("--trim-max", type=int, default=20000)
    ap.add_argument("--out", default=os.path.join(DATA, "_polarity_v2.json"))
    ap.add_argument("--rescan", action="store_true", help="force re-scan of chain files")
    args = ap.parse_args()

    cat = load_catalog()
    print(f"catalog: {len(cat)} inversions")
    tag = os.path.basename(os.path.normpath(args.aln))
    raw_path = os.path.join(DATA, f"_polarity_raw_{tag}_m{args.margin}_f{args.flank}.json")
    if os.path.exists(raw_path) and not args.rescan:
        print(f"loading cached raw chains {raw_path}")
        raw = json.load(open(raw_path))
    else:
        intervals = build_intervals(cat, args.margin, args.flank, args.trim_frac, args.trim_max)
        raw = {og["name"]: {} for og in OUTGROUPS}
        for og in OUTGROUPS:
            path = os.path.join(args.aln, og["chain"])
            if not os.path.exists(path):
                print(f"[{og['name']}] MISSING {path}; skipping"); continue
            print(f"[{og['name']}] scanning {og['chain']} ...")
            dom = dominant_by_interval(path, intervals)
            for (slot, tag), chains in [((k, t), v) for (k, t), v in dom.items()]:
                raw[og["name"]].setdefault(slot, {})[tag] = chains
        json.dump(raw, open(raw_path, "w"))
        print(f"cached raw chains -> {raw_path}")

    # per outgroup, per locus relative call (cheap; iterate here)
    calls = defaultdict(dict)
    for og in OUTGROUPS:
        ogd = raw.get(og["name"], {})
        for key in cat:
            slots = ogd.get(key, {})
            orient, c, det = call_relative(slots.get("interior"),
                                           slots.get("lflank"),
                                           slots.get("rflank"))
            calls[key][og["name"]] = (orient, c, det)

    json.dump({k: {og: [v[0], v[1], v[2]] for og, v in d.items()} for k, d in calls.items()},
              open(args.out, "w"))
    print(f"wrote {args.out}")
    benchmark(cat, calls)


DEPTH = {o["name"]: o["depth"] for o in OUTGROUPS}


def consensus(calld):
    """Depth-weighted parsimony over per-outgroup bridging calls.
    Returns (flip_ref_derived: bool|None, confidence, n_inf, n_agree)."""
    votes = []  # (depth, supports_ref_ancestral, strong)
    for og, (orient, strength, _det) in calld.items():
        if orient not in ("collinear", "inverted"):
            continue
        votes.append((DEPTH[og], orient == "collinear", strength == "strong"))
    if not votes:
        return None, "assumed", 0, 0
    w = lambda strong: 1.0 if strong else 0.5
    w_anc = sum(d * w(s) for d, a, s in votes if a)
    w_der = sum(d * w(s) for d, a, s in votes if not a)
    if w_anc == w_der:
        ref_anc = max(votes, key=lambda v: v[0])[1]   # deepest informative roots it
    else:
        ref_anc = w_anc > w_der
    flip = not ref_anc                                 # reference derived?
    n_inf = len(votes)
    n_agree = sum(1 for d, a, s in votes if a == ref_anc)
    strong_agree = any(s for d, a, s in votes if a == ref_anc)
    deepest_agrees = (max(votes, key=lambda v: v[0])[1] == ref_anc)
    frac = n_agree / n_inf
    if frac == 1.0 and n_inf >= 2 and strong_agree:
        conf = "high"
    elif frac >= 0.75 and strong_agree and deepest_agrees:
        conf = "high"        # deepest (root) + 3/4 strong agreement is well-rooted
    elif frac >= 0.75 and deepest_agrees:
        conf = "moderate"
    elif strong_agree and deepest_agrees:
        conf = "moderate"
    else:
        conf = "low"
    return flip, conf, n_inf, n_agree


def benchmark(cat, calls):
    sp = [o["name"] for o in OUTGROUPS]
    # ---- consensus (final per-locus) evaluation ----
    from collections import Counter
    conf_ct = Counter(); der_minor = der_tot = 0
    for key, rec in cat.items():
        flip, conf, n_inf, n_agree = consensus(calls[key])
        conf_ct[conf] += 1
        if flip is not None and rec["inv_af"] is not None:
            der_af = (1 - rec["inv_af"]) if flip else rec["inv_af"]
            der_tot += 1; der_minor += (der_af < 0.5)
    print("\n---- CONSENSUS (depth-weighted parsimony, final per-locus call) ----")
    print("  confidence:", dict(conf_ct))
    hi = conf_ct["high"] + conf_ct["moderate"]
    print(f"  high+moderate: {hi}/{len(cat)} = {hi/len(cat)*100:.0f}%")
    if der_tot:
        print(f"  derived is MINOR allele: {der_minor}/{der_tot} = {der_minor/der_tot*100:.0f}%")
    # cross-outgroup agreement on the RELATIVE call
    allagree = discord = n2 = 0
    informative = 0
    for key in cat:
        cc = [calls[key][s][0] for s in sp if s in calls[key]]
        cc = [c for c in cc if c in ("collinear", "inverted")]
        if cc: informative += 1
        if len(cc) >= 2:
            n2 += 1
            if len(set(cc)) == 1: allagree += 1
            else: discord += 1
    print("\n==================== BENCHMARK (flank-relative) ====================")
    print(f"loci with >=1 informative outgroup: {informative}/{len(cat)}")
    if n2:
        print(f"loci with >=2 informative: {n2}")
        print(f"  ALL agree: {allagree} ({allagree/n2*100:.0f}%)   discordant: {discord} ({discord/n2*100:.0f}%)")
    # pairwise
    import itertools
    print("pairwise agreement (both informative):")
    for a, b in itertools.combinations(sp, 2):
        both = ag = 0
        for key in cat:
            x = calls[key].get(a, (None,))[0]; y = calls[key].get(b, (None,))[0]
            if x in ("collinear", "inverted") and y in ("collinear", "inverted"):
                both += 1; ag += (x == y)
        if both:
            print(f"  {a[:4]:4s} vs {b[:4]:4s}: {ag}/{both} = {ag/both*100:.0f}%")
    # compare to OLD calls + derived freq sanity
    print("vs OLD production calls:")
    # consensus relative call -> reference ancestral?  collinear majority => ref ancestral
    flips_new = same_as_old = 0; cmp = 0
    der_minor = der_tot = 0
    for key, rec in cat.items():
        cc = [calls[key][s][0] for s in sp if s in calls[key] and calls[key][s][0] in ("collinear","inverted")]
        if not cc: continue
        inv_votes = sum(1 for c in cc if c == "inverted")
        col_votes = len(cc) - inv_votes
        ref_derived = inv_votes > col_votes   # majority of outgroups see interior inverted vs backbone
        flip = ref_derived
        flips_new += flip
        # derived freq: derived = reference orientation if flip else inverted(non-ref)
        af = rec["inv_af"]  # inverted (non-ref) allele freq
        if af is not None:
            der_af = (1 - af) if flip else af
            der_tot += 1
            if der_af < 0.5: der_minor += 1
        if rec["old_flip"] in ("0", "1"):
            cmp += 1
            same_as_old += (str(int(flip)) == rec["old_flip"])
    print(f"  new flip_ref_polarity=1 (reference derived): {flips_new}")
    if cmp: print(f"  agreement with OLD flip on {cmp} comparable loci: {same_as_old}/{cmp} = {same_as_old/cmp*100:.0f}%")
    if der_tot: print(f"  derived is the MINOR allele: {der_minor}/{der_tot} = {der_minor/der_tot*100:.0f}%  (expect high if calls are biologically sane)")


if __name__ == "__main__":
    main()
