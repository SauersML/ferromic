#!/usr/bin/env python3
"""No-drop, multi-outgroup polarization of inversion orientation.

Purpose
-------
Assign EVERY inversion in the catalog an ancestral-vs-derived orientation call
using the great-ape/macaque outgroups and maximum parsimony, with real
alignment evidence and the absolute minimum of assumed fallbacks. Project-wide,
"inverted" is redefined to mean the DERIVED arrangement w.r.t. the primate
ancestor (no longer "the non-hg38-reference arrangement").

Why multiple outgroups (the science)
------------------------------------
A pairwise alignment strand is only a RELATIVE signal: it says human and the
outgroup differ in orientation, not which arrangement is ancestral. A single
outgroup therefore mis-polarizes exactly the recurrent / toggling inversions
that flip independently across lineages (the textbook trap is 17q21.31/MAPT,
which is polymorphic in chimp too). The literature standard is to root the call
on several outgroups along the primate tree

    (((( human, chimp ), gorilla ), orangutan ), macaque )

and apply parsimony:

  * the inversion interval aligns to an outgroup on the '+' strand  -> that
    outgroup is COLLINEAR with the hg38 reference arrangement -> it supports
    "reference orientation = ancestral" (group0/direct ancestral).
  * it aligns on the '-' strand -> the outgroup is INVERTED relative to the
    reference -> it supports "reference orientation = derived" (group1/inverted
    ancestral).

If the informative outgroups AGREE, that arrangement is ancestral (high
confidence, and the agreement bounds the recurrence risk). If they DISAGREE,
that is itself a positive signal of recurrence / incomplete lineage sorting,
not a failure: we still emit a parsimony call (weighted toward the deeper
outgroups, which constrain the root) but flag it `outgroup_discordant`.

Evidence sources
----------------
Per outgroup we use the UCSC hg38-vs-<outgroup> alignment over the inversion
interior:
  * the all-chains (small, permissive: good COVERAGE of SD / divergent
    interiors), read as per-block strand;
  * for chimp additionally the reciprocal-best-style NET AXT (high stringency)
    as a cross-check when available locally.
We read the breakpoint-TRIMMED interior strand majority (to suppress
inverted-repeat noise at the junctions) and, as a tie-break, the single largest
collinear block (the true ortholog is usually the longest chain; paralogous
chains are shorter and, crucially, are not echoed by the other outgroups).

No-drop guarantee
-----------------
Every inversion in the catalog (union of inv_properties.tsv and callset.tsv)
appears in the output with a call. Low confidence, outgroup-discordant
(recurrent), and the irreducible "no ape orthology" residual are reported as
first-class tiers rather than dropped. `assumed` (reference-ancestral by
parsimony default) is used only where NO outgroup has any orthologous alignment.

Two phases (so decision rules can be retuned without re-streaming GBs):
  collect : stream each alignment once, cache per-interval per-outgroup strand
            evidence to data/_polarity_evidence.json
  decide  : apply the parsimony hierarchy, write data/inversion_polarity.tsv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# Outgroups ordered by increasing phylogenetic distance from human. `depth` is
# the parsimony weight used to break discordant votes (deeper constrains root).
OUTGROUPS = [
    {"name": "chimp",     "sp": "panTro6",  "depth": 1,
     "chain": "hg38.panTro6.all.chain.gz",  "net": "hg38.panTro6.net.axt.gz",
     "dir": "vsPanTro6"},
    {"name": "gorilla",   "sp": "gorGor6",  "depth": 2,
     "chain": "hg38.gorGor6.all.chain.gz",  "net": "hg38.gorGor6.net.axt.gz",
     "dir": "vsGorGor6"},
    {"name": "orangutan", "sp": "ponAbe3",  "depth": 3,
     "chain": "hg38.ponAbe3.all.chain.gz",  "net": "hg38.ponAbe3.net.axt.gz",
     "dir": "vsPonAbe3"},
    {"name": "macaque",   "sp": "rheMac10", "depth": 4,
     "chain": "hg38.rheMac10.all.chain.gz", "net": "hg38.rheMac10.net.axt.gz",
     "dir": "vsRheMac10"},
]
UCSC = "http://hgdownload.soe.ucsc.edu/goldenpath/hg38"

CACHE_PATH = os.path.join(DATA_DIR, "_polarity_evidence.json")
OUT_TABLE = os.path.join(DATA_DIR, "inversion_polarity.tsv")

REAL_CHR_RE = re.compile(r"^chr([0-9]+|[XY])$", re.IGNORECASE)
AXT_HEADER_RE = re.compile(
    r"^(-?\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+([+-])\s+(\d+)$"
)

# Breakpoint trim and calling thresholds.
TRIM_FRACTION = 0.10
TRIM_MAX = 20000
MIN_BP = 200
FRAC_HIGH = 0.90
FRAC_MOD = 0.70
LARGEST_RATIO = 1.5
LARGEST_MIN_BP = 500


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# Catalog (union -> no drops)
# ----------------------------------------------------------------------------

def norm_chrom(raw):
    if raw is None:
        return None
    c = str(raw).strip()
    while c.lower().startswith("chr"):
        c = c[3:]
    if not c:
        return None
    cl = c.lower()
    if cl == "x":
        core = "X"
    elif cl == "y":
        core = "Y"
    elif cl in {"m", "mt"}:
        core = "M"
    elif cl.isdigit():
        core = str(int(cl))
    else:
        core = c.upper()
    return "chr" + core


def _resolve(name):
    for base in (os.getcwd(), DATA_DIR, REPO_ROOT):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return os.path.join(DATA_DIR, name)


def load_catalog():
    cat = {}

    def add(chrom, start, end, src, recurrence=None, inv_af=None, orig=None):
        chrom = norm_chrom(chrom)
        try:
            start, end = int(start), int(end)
        except (ValueError, TypeError):
            return
        key = f"{chrom}:{start}-{end}"
        rec = cat.setdefault(key, {
            "inv_id": key, "chrom": chrom, "start": start, "end": end,
            "sources": set(), "recurrence": "unknown", "inv_af": None,
            "orig_ids": set(),
        })
        rec["sources"].add(src)
        if recurrence is not None and rec["recurrence"] == "unknown":
            rec["recurrence"] = recurrence
        if inv_af is not None and rec["inv_af"] is None:
            rec["inv_af"] = inv_af
        if orig:
            rec["orig_ids"].add(orig)

    p = _resolve("inv_properties.tsv")
    if os.path.exists(p):
        with open(p) as fh:
            rdr = csv.reader(fh, delimiter="\t")
            hdr = next(rdr)
            idx = {h.strip(): i for i, h in enumerate(hdr)}
            ci, si, ei = idx.get("Chromosome"), idx.get("Start"), idx.get("End")
            rci = idx.get("0_single_1_recur_consensus")
            afi, oii = idx.get("Inverted_AF"), idx.get("OrigID")
            for row in rdr:
                if None in (ci, si, ei) or max(ci, si, ei) >= len(row):
                    continue
                lbl = row[rci].strip() if rci is not None and rci < len(row) else ""
                recurrence = {"1": "recurrent", "0": "single-event"}.get(lbl, "unknown")
                af = None
                if afi is not None and afi < len(row):
                    try:
                        af = float(row[afi])
                    except (ValueError, TypeError):
                        af = None
                orig = row[oii].strip() if oii is not None and oii < len(row) else None
                add(row[ci], row[si], row[ei], "inv_properties", recurrence, af, orig)

    p = _resolve("callset.tsv")
    if os.path.exists(p):
        with open(p) as fh:
            rdr = csv.reader(fh, delimiter="\t")
            hdr = next(rdr)
            idx = {h.strip(): i for i, h in enumerate(hdr)}
            ci = idx.get("seqnames", idx.get("Chromosome"))
            si, ei = idx.get("start", idx.get("Start")), idx.get("end", idx.get("End"))
            oii = idx.get("inv_id")
            for row in rdr:
                if None in (ci, si, ei) or max(ci, si, ei) >= len(row):
                    continue
                orig = row[oii].strip() if oii is not None and oii < len(row) else None
                add(row[ci], row[si], row[ei], "callset", orig=orig)

    return cat


# ----------------------------------------------------------------------------
# Evidence accumulation (single pass per file; tags 'full' and 'trim')
# ----------------------------------------------------------------------------

def interior(rec):
    s, e = rec["start"], rec["end"]
    length = e - s + 1
    trim = int(min(TRIM_MAX, TRIM_FRACTION * length))
    if length - 2 * trim < MIN_BP:
        trim = 0
    return rec["chrom"], s + trim, e - trim


def build_intervals(cat):
    """Return list of (slot, chrom, s, e) where slot=(key, tag)."""
    out = []
    for key, rec in cat.items():
        out.append(((key, "full"), rec["chrom"], rec["start"], rec["end"]))
        c, s, e = interior(rec)
        out.append(((key, "trim"), c, s, e))
    return out


def accumulate(block_iter, intervals, BIN=100000):
    """block_iter -> (chrom_norm, t_start1based, t_end1based, q_name, strand).
    Returns slot -> {'+':[tot,max,n], '-':[...]} over REAL chimp chroms only."""
    index = defaultdict(lambda: defaultdict(list))
    for (slot, chrom, s, e) in intervals:
        for b in range((s - 1) // BIN, (e - 1) // BIN + 1):
            index[chrom][b].append((slot, s, e))
    acc = {slot: {"+": [0, 0, 0], "-": [0, 0, 0]} for (slot, _, _, _) in intervals}
    n = 0
    for (chrom, ts, te, q_name, strand) in block_iter:
        n += 1
        if n % 2000000 == 0:
            log(f"  ...{n} blocks")
        bins = index.get(chrom)
        if not bins or not REAL_CHR_RE.match(q_name):
            continue
        if strand not in ("+", "-"):
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
                if seen is None:
                    seen = set()
                if slot in seen:
                    continue
                seen.add(slot)
                cell = acc[slot][strand]
                cell[0] += ov
                if ov > cell[1]:
                    cell[1] = ov
                cell[2] += 1
    log(f"  parsed {n} blocks")
    return acc


def _opener(path):
    return gzip.open(path, "rt", encoding="latin-1") if path.endswith(".gz") \
        else open(path, encoding="latin-1")


def net_axt_blocks(path):
    with _opener(path) as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            m = AXT_HEADER_RE.match(header.strip())
            if not m:
                continue
            fh.readline(); fh.readline()
            yield (norm_chrom(m.group(2)), int(m.group(3)), int(m.group(4)),
                   m.group(5), m.group(8))


def chain_blocks(path):
    """Yield per ungapped block of a UCSC chain. Header:
    chain score tName tSize tStrand tStart tEnd qName qSize qStrand qStart qEnd id
    Target is always '+' and 0-based half-open; query strand qStrand gives the
    outgroup orientation. We yield (chrom, tstart1, tend1, qName, qStrand,
    chain_id, score) per ungapped block, 1-based inclusive target spans."""
    with _opener(path) as fh:
        t_chrom = q_name = None
        q_strand = "+"
        chain_id = score = 0
        t_pos = 0
        for line in fh:
            s = line.strip()
            if not s:
                t_chrom = None
                continue
            if s.startswith("chain"):
                f = s.split()
                t_chrom, t_pos = norm_chrom(f[2]), int(f[5])
                q_name, q_strand = f[7], f[9]
                score = int(f[1])
                chain_id = f[12] if len(f) > 12 else f[1]
                continue
            if t_chrom is None:
                continue
            parts = line.split()
            size = int(parts[0])
            if size > 0:
                yield (t_chrom, t_pos + 1, t_pos + size, q_name, q_strand,
                       chain_id, score)
            t_pos += size + (int(parts[1]) if len(parts) >= 3 else 0)


def accumulate_chain_dominant(block_iter, intervals, BIN=100000):
    """Like :func:`accumulate` but resolves each locus to its single DOMINANT
    orthologous chain (the chain with the most interior overlap bp), which
    suppresses paralogous/SD secondary chains that otherwise inject spurious
    inverted strand. Returns slot -> {'dom_strand','dom_bp','second_bp'} over
    REAL outgroup chromosomes only."""
    index = defaultdict(lambda: defaultdict(list))
    for (slot, chrom, s, e) in intervals:
        for b in range((s - 1) // BIN, (e - 1) // BIN + 1):
            index[chrom][b].append((slot, s, e))
    # slot -> chain_id -> [strand, bp]
    per = {slot: {} for (slot, _, _, _) in intervals}
    n = 0
    for (chrom, ts, te, q_name, strand, chain_id, score) in block_iter:
        n += 1
        if n % 2000000 == 0:
            log(f"  ...{n} chain blocks")
        bins = index.get(chrom)
        if not bins or not REAL_CHR_RE.match(q_name) or strand not in ("+", "-"):
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
                if seen is None:
                    seen = set()
                tag = (slot, chain_id)
                if tag in seen:
                    continue
                seen.add(tag)
                d = per[slot].setdefault(chain_id, [strand, 0])
                d[1] += ov
    log(f"  parsed {n} chain blocks")
    out = {}
    for slot, chains in per.items():
        if not chains:
            out[slot] = {"dom_strand": None, "dom_bp": 0, "second_bp": 0}
            continue
        ranked = sorted(chains.values(), key=lambda x: x[1], reverse=True)
        out[slot] = {
            "dom_strand": ranked[0][0], "dom_bp": ranked[0][1],
            "second_bp": ranked[1][1] if len(ranked) > 1 else 0,
        }
    return out


# ----------------------------------------------------------------------------
# COLLECT
# ----------------------------------------------------------------------------

def collect(args):
    cat = load_catalog()
    log(f"Catalog: {len(cat)} inversions (union inv_properties.tsv + callset.tsv).")
    intervals = build_intervals(cat)

    cache = {"meta": {"trim_fraction": TRIM_FRACTION, "trim_max": TRIM_MAX,
                      "outgroups": [o["name"] for o in OUTGROUPS]},
             "loci": {}}
    for key, rec in cat.items():
        cache["loci"][key] = {
            "inv_id": key, "chrom": rec["chrom"], "start": rec["start"],
            "end": rec["end"], "recurrence": rec["recurrence"],
            "inv_af": rec["inv_af"], "orig_ids": sorted(rec["orig_ids"]),
            "sources": sorted(rec["sources"]), "ev": {},
        }

    def fold(acc, og_name, kind):
        for (slot, cell) in acc.items():
            key, tag = slot
            self_ev = cache["loci"][key]["ev"].setdefault(og_name, {})
            self_ev[f"{kind}_{tag}"] = cell

    for og in OUTGROUPS:
        chain = _resolve(og["chain"])
        if os.path.exists(chain):
            log(f"[{og['name']}] CHAIN pass (dominant orthologous chain): {chain}")
            fold(accumulate_chain_dominant(chain_blocks(chain), intervals),
                 og["name"], "chain")
        else:
            log(f"[{og['name']}] chain missing ({og['chain']}); skipping.")
        if og.get("net"):
            net = _resolve(og["net"])
            if os.path.exists(net):
                log(f"[{og['name']}] NET pass: {net}")
                fold(accumulate(net_axt_blocks(net), intervals), og["name"], "net")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(args.cache, "w") as fh:
        json.dump(cache, fh)
    log(f"Wrote {args.cache} ({len(cache['loci'])} loci).")


# ----------------------------------------------------------------------------
# DECIDE (per-outgroup relative call -> parsimony)
# ----------------------------------------------------------------------------

def _strand_call(cell):
    """cell {'+':[tot,max,n],'-':[...]} -> ('collinear'|'inverted'|None, frac, tot)."""
    if not cell:
        return None, None, 0
    p, m = cell["+"][0], cell["-"][0]
    tot = p + m
    if tot <= 0:
        return None, None, 0
    frac = max(p, m) / tot
    return ("collinear" if p >= m else "inverted"), frac, tot


def _largest_call(cell):
    if not cell:
        return None
    mp, mm = cell["+"][1], cell["-"][1]
    if max(mp, mm) < LARGEST_MIN_BP:
        return None
    big, small = (mp, mm) if mp >= mm else (mm, mp)
    if small == 0 or big >= LARGEST_RATIO * small:
        return "collinear" if mp >= mm else "inverted"
    return None


def _dom_chain_call(cell):
    """cell {'dom_strand','dom_bp','second_bp'} -> (orientation, strength). The
    dominant orthologous chain's strand IS the orientation; strength is 'strong'
    when it clearly outweighs the next chain (true ortholog, not a paralog)."""
    if not cell or not cell.get("dom_strand"):
        return None, None
    dom_bp = cell.get("dom_bp", 0)
    if dom_bp < MIN_BP:
        return None, None
    call = "collinear" if cell["dom_strand"] == "+" else "inverted"
    second = cell.get("second_bp", 0)
    strong = (second == 0) or (dom_bp >= LARGEST_RATIO * second)
    return call, ("strong" if strong else "weak")


def outgroup_orientation(ev):
    """ev = {'net_trim':cell,'chain_trim':cell,...} for one outgroup. Return
    (orientation, basis, strength), orientation in {'collinear','inverted',None}.
    Prefers the high-stringency NET strand-majority, then the dominant
    orthologous CHAIN (trimmed interior before full)."""
    # NET (reciprocal-best) strand majority — only present for chimp.
    for kind in ("net_trim", "net_full"):
        call, frac, tot = _strand_call(ev.get(kind))
        if call and tot >= MIN_BP and frac >= FRAC_MOD:
            return call, kind, ("strong" if frac >= FRAC_HIGH else "weak")
    # Dominant orthologous chain.
    for kind in ("chain_trim", "chain_full"):
        call, strength = _dom_chain_call(ev.get(kind))
        if call:
            return call, kind, strength
    return None, None, None


def decide(args):
    with open(args.cache) as fh:
        cache = json.load(fh)
    loci = cache["loci"]
    og_names = [o["name"] for o in OUTGROUPS]
    depth = {o["name"]: o["depth"] for o in OUTGROUPS}

    tiers = defaultdict(int)
    rows = []
    for key in sorted(loci, key=lambda k: (loci[k]["chrom"], loci[k]["start"])):
        d = loci[key]
        ev = d.get("ev", {})
        # Per-outgroup relative orientation.
        per = {}
        votes = []  # (name, depth, supports_ref_ancestral(bool), strong(bool))
        for name in og_names:
            call, basis, strength = outgroup_orientation(ev.get(name, {}))
            per[name] = call or "NA"
            if call:
                supports_ref_anc = (call == "collinear")
                votes.append((name, depth[name], supports_ref_anc, strength == "strong"))

        if not votes:
            anc, conf, evidence = "direct", "assumed", "no_ape_orthology"
            discord = False
            n_anc = n_der = 0
        else:
            n_anc = sum(1 for v in votes if v[2])          # support ref ancestral
            n_der = sum(1 for v in votes if not v[2])       # support ref derived
            discord = (n_anc > 0 and n_der > 0)
            # Parsimony rooting: weight each outgroup vote by phylogenetic depth
            # (the deeper outgroups constrain the primate root, so a Homo-Pan
            # shared-derived inversion is correctly called derived even though
            # chimp matches the human reference -- the 17q21.31 lesson).
            w_anc = sum(v[1] for v in votes if v[2])
            w_der = sum(v[1] for v in votes if not v[2])
            if w_anc == w_der:
                ref_anc = max(votes, key=lambda v: v[1])[2]  # deepest informative
            else:
                ref_anc = (w_anc > w_der)
            anc = "direct" if ref_anc else "inverted"
            # Confidence from how strongly the outgroups support the chosen call.
            agree = [v for v in votes if v[2] == ref_anc]
            support = len(agree) / len(votes)
            strong_agree = any(v[3] for v in agree)
            deepest_agrees = (max(votes, key=lambda v: v[1])[2] == ref_anc)
            if support == 1.0 and len(votes) >= 2 and strong_agree:
                conf = "high"
            elif support >= 0.75 and strong_agree and deepest_agrees:
                conf = "moderate"
            elif support == 1.0 and strong_agree:        # single strong outgroup
                conf = "moderate"
            elif support > 0.5:
                conf = "low"
            else:
                conf = "low"                              # even split / weak
            tag = "congruent" if not discord else "discordant"
            evidence = f"{tag}[{support:.2f}]:" + ",".join(
                f"{v[0]}{'+' if v[2] else '-'}" for v in votes)

        derived = "inverted" if anc == "direct" else "direct"
        flip = (anc == "inverted")
        tiers[conf] += 1

        inv_af = d.get("inv_af")
        derived_af = "NA"
        minor_is_derived = "NA"
        if inv_af is not None:
            derived_af = round((1 - inv_af) if flip else inv_af, 6)
            minor = "inverted" if inv_af < 0.5 else ("direct" if inv_af > 0.5 else "tie")
            if minor in ("direct", "inverted"):
                minor_is_derived = (minor == derived)

        rows.append({
            "inv_id": key, "chrom": d["chrom"], "start": d["start"], "end": d["end"],
            "orig_id": ";".join(d.get("orig_ids") or []),
            "recurrence_class": d.get("recurrence", "unknown"),
            "sources": ";".join(d.get("sources") or []),
            "ancestral_orientation": anc,
            "derived_orientation": derived,
            "flip_ref_polarity": "1" if flip else "0",
            "confidence": conf,
            "n_outgroups_informative": len(votes),
            "n_support_ref_ancestral": n_anc,
            "n_support_ref_derived": n_der,
            "outgroup_discordant": "1" if discord else "0",
            "orient_chimp": per.get("chimp", "NA"),
            "orient_gorilla": per.get("gorilla", "NA"),
            "orient_orangutan": per.get("orangutan", "NA"),
            "orient_macaque": per.get("macaque", "NA"),
            "evidence": evidence,
            "inv_af_ref": "NA" if inv_af is None else inv_af,
            "derived_af": derived_af,
            "minor_is_derived": minor_is_derived,
        })

    cols = ["inv_id", "chrom", "start", "end", "orig_id", "recurrence_class",
            "sources", "ancestral_orientation", "derived_orientation",
            "flip_ref_polarity", "confidence", "n_outgroups_informative",
            "n_support_ref_ancestral", "n_support_ref_derived",
            "outgroup_discordant", "orient_chimp", "orient_gorilla",
            "orient_orangutan", "orient_macaque", "evidence", "inv_af_ref",
            "derived_af", "minor_is_derived"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n = len(rows)
    flips = sum(1 for r in rows if r["flip_ref_polarity"] == "1")
    disc = sum(1 for r in rows if r["outgroup_discordant"] == "1")
    real = sum(1 for r in rows if r["confidence"] != "assumed")
    log("")
    log("=== Multi-outgroup no-drop polarization summary ===")
    log(f"Inversions in catalog:          {n} (no drops)")
    log(f"Calls from real ape evidence:   {real}/{n}")
    for t in ("high", "moderate", "low", "assumed"):
        log(f"  {t:9s}: {tiers.get(t, 0)}")
    log(f"Outgroup-discordant (recurrence/ILS signal): {disc}")
    log(f"Loci where ref orientation is DERIVED (flip): {flips}")
    log(f"Wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description="Multi-outgroup inversion polarization.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--cache", default=CACHE_PATH)
    c.set_defaults(func=collect)
    de = sub.add_parser("decide")
    de.add_argument("--cache", default=CACHE_PATH)
    de.add_argument("--out", default=OUT_TABLE)
    de.set_defaults(func=decide)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
