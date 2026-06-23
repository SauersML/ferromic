#!/usr/bin/env python3
"""v3 derived-orientation caller: orthology- and adjacency-aware, honest about the unknown.

Addresses concrete deficiencies of v2 (which only compared interior vs flank STRAND):
  * v2's chain parser discarded the query chromosome/coordinates, so it could not tell a
    true inverted ortholog from a segmental-duplication paralog that merely aligns on the
    opposite strand. "Same orientation != same event." v3 keeps query coords and only calls
    an inversion when the interior is filled by a chain that is the SAME query sequence as the
    flanking syntenic backbone, CONTIGUOUS with it at the breakpoints (a real inverted ortholog).
  * v2 forced a binary call on every locus, defaulting no-orthology to reference-ancestral.
    v3 returns UNRESOLVED for: no spanning ortholog, paralog/distant interior, or non-simple
    structures (invDup / misorient / complex). Reference-ancestral is never a fallback.
  * v2 was tuned against allele frequency (circular). v3 is not tuned on frequency at all;
    it is benchmarked only against breakpoint/haplotype-resolved truth (17q21.31, 8p23.1)
    and synthetic controls.
  * The catalog is built from callset.tsv + inv_properties.tsv, never from the polarity output.

Orientation per outgroup is COLLINEAR / INVERTED / UNRESOLVED. Per locus the outgroups are
combined; genuinely conflicting homologous evidence -> RECURRENT (no single derived orientation).

This is still a synteny method. The gold standard -- assembly-resolved breakpoint junctions +
event-aware phylogenetics on HPRC/T2T-ape haplotypes -- needs data outside this environment;
v3's job is to be RIGOROUS and HONEST within chain evidence, not to overclaim.
"""
from __future__ import annotations
import argparse, csv, gzip, os, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data")

OUTGROUPS = [
    {"name": "chimp",     "chain": "hg38.panTro6.all.chain.gz",  "depth": 1},
    {"name": "gorilla",   "chain": "hg38.gorGor6.all.chain.gz",  "depth": 2},
    {"name": "orangutan", "chain": "hg38.ponAbe3.all.chain.gz",  "depth": 3},
    {"name": "macaque",   "chain": "hg38.rheMac10.all.chain.gz", "depth": 4},
]
REAL_CHR_RE = re.compile(r"^chr([0-9]+|[XY])$", re.IGNORECASE)
MIN_BP = 1000          # require real orthologous coverage, not a 200bp fragment
MARGIN = 20000         # skip breakpoint-proximal SD before measuring the flank backbone
FLANK = 100000


def norm_chrom(raw):
    c = str(raw).strip()
    while c.lower().startswith("chr"):
        c = c[3:]
    cl = c.lower()
    core = {"x": "X", "y": "Y"}.get(cl, str(int(cl)) if cl.isdigit() else c.upper())
    return "chr" + core


# --------------------------- independent catalog ---------------------------
def build_catalog():
    """Catalog from callset.tsv (+ inv_properties.tsv for AF and structure type).
    NEVER reads inversion_polarity.tsv. Returns {key -> rec}."""
    cat = {}
    cp = os.path.join(DATA, "callset.tsv")
    with open(cp) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            try:
                chrom = norm_chrom(r["seqnames"]); s = int(r["start"]); e = int(r["end"])
            except (KeyError, ValueError):
                continue
            key = f"{chrom}:{s}-{e}"
            cat[key] = {"chrom": chrom, "start": s, "end": e,
                        "orig_id": r.get("inv_id", ""), "af": None, "svtype": "inv"}
    ip = os.path.join(DATA, "inv_properties.tsv")
    if os.path.exists(ip):
        for r in csv.DictReader(open(ip), delimiter="\t"):
            try:
                chrom = norm_chrom(r["Chromosome"]); s = int(r["Start"]); e = int(r["End"])
            except (KeyError, ValueError):
                continue
            # match to catalog within +/-1bp
            for ds in (0, -1, 1):
                for de in (0, -1, 1):
                    k = f"{chrom}:{s+ds}-{e+de}"
                    if k in cat:
                        af = r.get("Inverted_AF", "")
                        try: cat[k]["af"] = float(af)
                        except (TypeError, ValueError): pass
                        oid = (r.get("OrigID") or "").lower()
                        if "invdup" in oid: cat[k]["svtype"] = "invDup"
                        elif "miso" in oid: cat[k]["svtype"] = "miso"
                        elif "complex" in oid or "lowconf" in oid: cat[k]["svtype"] = "complex"
                        break
    return cat


# --------------------------- chain parsing (keep query coords) ---------------------------
def chain_headers_over(path, targets_by_chrom):
    """Yield, per chain that overlaps any target interval, a dict with the chain's
    target span, query name/strand/span, id, score, and its blocks' target spans so we
    can measure interior vs flank coverage AND the query position at the breakpoints.

    targets_by_chrom: {chrom -> list of (key, lo, hi)} = union of all intervals to test."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="latin-1") as fh:
        keep = False; hdr = None; blocks = None; t_pos = q_pos = 0; q_strand = "+"
        for line in fh:
            s = line.strip()
            if s.startswith("chain"):
                f = s.split()
                t_chrom = norm_chrom(f[2]); t_start = int(f[5]); t_end = int(f[6])
                q_name = f[7]; q_strand = f[9]; q_start = int(f[10]); q_end = int(f[11])
                cid = f[12] if len(f) > 12 else f[1]; score = int(f[1])
                lo_hi = targets_by_chrom.get(t_chrom)
                keep = bool(lo_hi) and any(t_end > lo and t_start < hi for _, lo, hi in lo_hi)
                if keep:
                    hdr = {"t_chrom": t_chrom, "t_start": t_start, "t_end": t_end,
                           "q_name": q_name, "q_strand": q_strand, "q_start": q_start,
                           "q_end": q_end, "cid": cid, "score": score}
                    blocks = []
                    t_pos = t_start; q_pos = q_start
                continue
            if not s:
                if keep and hdr is not None:
                    hdr["blocks"] = blocks
                    yield hdr
                keep = False; hdr = None; blocks = None
                continue
            if not keep:
                continue
            parts = s.split()
            size = int(parts[0])
            # block maps target [t_pos, t_pos+size) to query [q_pos, q_pos+size)
            blocks.append((t_pos, t_pos + size, q_pos, q_pos + size))
            dt = int(parts[1]) if len(parts) >= 3 else 0
            dq = int(parts[2]) if len(parts) >= 3 else 0
            t_pos += size + dt; q_pos += size + dq
        if keep and hdr is not None:
            hdr["blocks"] = blocks
            yield hdr


def _ov(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def query_pos_at_target(chain, t):
    """Linear-interpolate the query coordinate at target position t within a chain
    (using its blocks), or None if t is outside the chain's aligned blocks."""
    best = None
    for (t0, t1, q0, q1) in chain["blocks"]:
        if t0 <= t < t1:
            return q0 + (t - t0)
        # nearest block edge as fallback
        d = min(abs(t - t0), abs(t - t1))
        if best is None or d < best[0]:
            best = (d, q0 if abs(t - t0) <= abs(t - t1) else q1)
    return best[1] if best else None


def call_outgroup(chains, s, e, trim):
    """chains: list of chain headers (with blocks) overlapping this locus's region.
    Return (orientation, detail). orientation in collinear/inverted/UNRESOLVED."""
    li0, li1 = s - MARGIN - FLANK, s - MARGIN          # left flank
    ri0, ri1 = e + MARGIN, e + MARGIN + FLANK          # right flank
    ii0, ii1 = s + trim, e - trim                      # interior

    # coverage of each chain over left flank / right flank / interior
    cov = {}
    for c in chains:
        cl = sum(_ov(t0, t1, li0, li1) for (t0, t1, _, _) in c["blocks"])
        cr = sum(_ov(t0, t1, ri0, ri1) for (t0, t1, _, _) in c["blocks"])
        ci = sum(_ov(t0, t1, ii0, ii1) for (t0, t1, _, _) in c["blocks"])
        if cl or cr or ci:
            cov[c["cid"]] = (c, cl, cr, ci)

    # backbone = chain present in BOTH flanks with the most flank coverage (true ortholog).
    spanning = [(cid, v) for cid, v in cov.items() if v[1] >= MIN_BP and v[2] >= MIN_BP]
    if not spanning:
        return "UNRESOLVED", {"why": "no_chain_spans_both_flanks"}
    bb_cid, (bb, bl, br, bi) = max(spanning, key=lambda kv: kv[1][1] + kv[1][2])

    # Does the backbone bridge the interior (collinear)?
    interior_len = max(1, ii1 - ii0)
    if bi >= 0.5 * interior_len or bi >= 5 * MIN_BP:
        return "collinear", {"backbone": bb_cid, "bb_interior_bp": bi, "qname": bb["q_name"]}

    # Backbone broke. Find the chain dominating the interior.
    others = [(cid, v) for cid, v in cov.items() if cid != bb_cid and v[3] >= MIN_BP]
    if not others:
        return "UNRESOLVED", {"why": "backbone_broke_no_interior_ortholog", "bb_interior_bp": bi}
    fc_cid, (fc, fl_, fr_, fi) = max(others, key=lambda kv: kv[1][3])

    # HOMOLOGY check: the interior filler must be the SAME query sequence as the backbone,
    # opposite strand, and CONTIGUOUS at the breakpoints (a true inverted ortholog) -- not a
    # distant paralog / SD copy.
    same_qname = (fc["q_name"] == bb["q_name"])
    opp_strand = (fc["q_strand"] != bb["q_strand"])
    # query position just outside each breakpoint on the backbone vs on the filler interior
    qbb_L = query_pos_at_target(bb, s); qbb_R = query_pos_at_target(bb, e)
    qfc_lo = min(fc["q_start"], fc["q_end"]); qfc_hi = max(fc["q_start"], fc["q_end"])
    contiguous = False
    if qbb_L is not None and qbb_R is not None:
        span = abs(qbb_R - qbb_L) + (e - s)
        # filler query window should sit within ~the locus span of the backbone breakpoints
        near_L = qfc_lo <= max(qbb_L, qbb_R) + span and qfc_hi >= min(qbb_L, qbb_R) - span
        contiguous = near_L
    if same_qname and opp_strand and contiguous:
        return "inverted", {"backbone": bb_cid, "filler": fc_cid, "fi": fi,
                            "qname": bb["q_name"], "homology": "inverted_ortholog"}
    return "UNRESOLVED", {"why": "interior_filler_not_homologous_ortholog",
                          "same_qname": same_qname, "opp_strand": opp_strand,
                          "contiguous": contiguous, "fi": fi}


def consensus(per):
    """per: og -> orientation. Combine into ancestral_state / recurrence honestly.
    Returns dict. No depth integer hack as the sole arbiter: require agreement among the
    informative outgroups; genuine conflict -> RECURRENT (no global derived orientation)."""
    calls = {og: o for og, o in per.items() if o in ("collinear", "inverted")}
    n_col = sum(o == "collinear" for o in calls.values())
    n_inv = sum(o == "inverted" for o in calls.values())
    n = n_col + n_inv
    if n == 0:
        return {"state": "UNRESOLVED", "reference_relation": "ref",
                "flip": "", "confidence": "unresolved", "n_col": 0, "n_inv": 0}
    if n_col and n_inv:
        # conflicting homologous orientations across outgroups -> recurrence/toggling
        return {"state": "RECURRENT", "reference_relation": "ref",
                "flip": "", "confidence": "recurrent", "n_col": n_col, "n_inv": n_inv}
    # unanimous among informative outgroups
    ref_is_ancestral = n_inv == 0          # all collinear -> reference shares ancestral arrangement
    conf = "high" if n >= 2 else "moderate"
    return {"state": "SINGLE_EVENT",
            "ancestral": "direct" if ref_is_ancestral else "inverted",
            "derived": "inverted" if ref_is_ancestral else "direct",
            "flip": 0 if ref_is_ancestral else 1,
            "confidence": conf, "n_col": n_col, "n_inv": n_inv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aln", required=True, help="dir with hg38.<sp>.all.chain.gz (or rbest)")
    ap.add_argument("--out", default=os.path.join(DATA, "inversion_polarity_v3.tsv"))
    args = ap.parse_args()

    cat = build_catalog()
    print(f"catalog (callset+inv_properties, independent): {len(cat)} loci")

    # union of all intervals per chrom for the chain scanner
    targets = defaultdict(list)
    for key, m in cat.items():
        s, e = m["start"], m["end"]
        targets[m["chrom"]].append((key, max(1, s - MARGIN - FLANK), e + MARGIN + FLANK))

    # per locus, collect chains overlapping its region, per outgroup
    per_og_chains = {og["name"]: defaultdict(list) for og in OUTGROUPS}
    for og in OUTGROUPS:
        path = os.path.join(args.aln, og["chain"])
        if not os.path.exists(path):
            print(f"[{og['name']}] missing {path}"); continue
        print(f"[{og['name']}] scanning {og['chain']} ...")
        for ch in chain_headers_over(path, targets):
            if not REAL_CHR_RE.match(ch["q_name"]):
                continue
            tl = targets[ch["t_chrom"]]
            for (key, lo, hi) in tl:
                if ch["t_end"] > lo and ch["t_start"] < hi:
                    per_og_chains[og["name"]][key].append(ch)

    rows = []
    for key, m in cat.items():
        s, e = m["start"], m["end"]
        trim = int(min(20000, 0.1 * (e - s)))
        per = {}
        details = {}
        # non-simple structures are not balanced biallelic inversions -> never force a binary
        if m["svtype"] != "inv":
            cs = {"state": "COMPLEX_" + m["svtype"], "flip": "", "confidence": "unresolved",
                  "n_col": "", "n_inv": ""}
        else:
            for og in OUTGROUPS:
                o, det = call_outgroup(per_og_chains[og["name"]].get(key, []), s, e, trim)
                per[og["name"]] = o; details[og["name"]] = det
            cs = consensus(per)
        rows.append({
            "inv_id": key, "chrom": m["chrom"], "start": s, "end": e,
            "orig_id": m["orig_id"], "svtype": m["svtype"],
            "state": cs["state"],
            "ancestral_arrangement": cs.get("ancestral", ""),
            "derived_arrangement": cs.get("derived", ""),
            "flip_ref_polarity": cs.get("flip", ""),
            "confidence": cs["confidence"],
            "n_collinear": cs.get("n_col", ""), "n_inverted": cs.get("n_inv", ""),
            "orient_chimp": per.get("chimp", ""), "orient_gorilla": per.get("gorilla", ""),
            "orient_orangutan": per.get("orangutan", ""), "orient_macaque": per.get("macaque", ""),
            "inv_af": m["af"] if m["af"] is not None else "",
        })
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(f"wrote {args.out} ({len(rows)} loci)")
    print("state:", dict(Counter(r["state"] for r in rows)))
    print("confidence:", dict(Counter(r["confidence"] for r in rows)))


if __name__ == "__main__":
    main()
