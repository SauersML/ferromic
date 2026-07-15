#!/usr/bin/env python3
"""Per-locus quantitative diff: T2T chimpanzee vs panTro6 alignment to GRCh38.

For every inversion, read the per-locus PAFs (stageb/<inv_id>/{chimpT2T,panTro6}.paf)
and, over the inversion window, compute for each reference:
  - cov_win_pct   : % of the [win_start,win_end] window covered by chimp alignments
  - cov_inv_pct   : % of the inversion interval [start,end] covered
  - n_segments    : number of alignment records (>= MIN_ALN) in window
  - n_gaps        : number of uncovered gaps (> GAP_MIN bp) inside the window
  - largest_gap   : largest such gap (bp)
  - longest_block : longest contiguous covered stretch (bp)
  - identity      : coverage-weighted gap-excluded identity (matches / aln_block_len)
  - frac_rev_inv  : reverse-strand fraction over the inversion interval

Then emits, per locus, a diff (T2T - panTro6) and a short human-readable note of
what the T2T assembly resolved. Output: diff_chimp.tsv + diff_chimp.json.
"""
import sys, os, json, gzip
from collections import defaultdict

inv_tsv, stageb, out_tsv, out_json = sys.argv[1:5]
MIN_ALN = 2000
GAP_MIN = 1000

def load_inv(path):
    rows = []
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ix = {c: i for i, c in enumerate(hdr)}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            rows.append(dict(
                inv_id=p[ix["inv_id"]], chrom=p[ix["chrom"]],
                start=int(p[ix["start"]]), end=int(p[ix["end"]]),
                ws=int(p[ix["win_start"]]), we=int(p[ix["win_end"]]),
                af=p[ix["inverted_af"]], disease=p[ix["disease_genes"]],
                size_bp=int(p[ix["size_bp"]]),
            ))
    return rows

def read_paf_segments(path, chrom, ws, we):
    """Return list of (t_start, t_end, strand, matches, blocklen) clipped to window."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    segs = []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 12:
                continue
            if f[5] != chrom:
                continue
            ts, te = int(f[7]), int(f[8])
            if te - ts < MIN_ALN:
                continue
            cs, ce = max(ts, ws), min(te, we)
            if ce <= cs:
                continue
            strand = f[4]
            matches = int(f[9]); blocklen = int(f[10])
            segs.append((cs, ce, strand, matches, blocklen, ts, te))
    return segs

def merge_intervals(ivs):
    if not ivs:
        return []
    ivs = sorted(ivs)
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out

def clip_len(segs, a, b):
    tot = 0
    for s in segs:
        lo, hi = max(s[0], a), min(s[1], b)
        if hi > lo:
            tot += hi - lo
    return tot

def metrics(segs, inv):
    ws, we, a, b = inv["ws"], inv["we"], inv["start"], inv["end"]
    win_len = we - ws
    inv_len = b - a
    covered = merge_intervals([(s[0], s[1]) for s in segs])
    cov_win = sum(e - s for s, e in covered)
    cov_inv = clip_len(covered, a, b)
    # gaps inside window
    gaps = []
    prev = ws
    for s, e in covered:
        if s > prev:
            gaps.append(s - prev)
        prev = max(prev, e)
    if we > prev:
        gaps.append(we - prev)
    big_gaps = [g for g in gaps if g > GAP_MIN]
    longest_block = max((e - s for s, e in covered), default=0)
    # identity over window, coverage-weighted
    num = sum(s[3] for s in segs); den = sum(s[4] for s in segs)
    ident = (num / den) if den > 0 else float("nan")
    # reverse fraction over inversion interval
    rev = 0; tot = 0
    for s in segs:
        lo, hi = max(s[0], a), min(s[1], b)
        if hi > lo:
            tot += hi - lo
            if s[2] == "-":
                rev += hi - lo
    frac_rev = (rev / tot) if tot > 0 else float("nan")
    return dict(
        cov_win_pct=round(100 * cov_win / win_len, 1) if win_len else 0.0,
        cov_inv_pct=round(100 * cov_inv / inv_len, 1) if inv_len else 0.0,
        n_segments=len(segs),
        n_gaps=len(big_gaps),
        largest_gap=max(big_gaps) if big_gaps else 0,
        longest_block=longest_block,
        identity=round(ident, 4) if ident == ident else None,
        frac_rev_inv=round(frac_rev, 3) if frac_rev == frac_rev else None,
    )

def note(t2t, pan, inv):
    if pan["n_segments"] == 0 and t2t["n_segments"] == 0:
        return "no chimp alignment in either assembly at this locus"
    if pan["n_segments"] == 0:
        return f"panTro6 had NO alignment here; T2T covers {t2t['cov_win_pct']}% of the window"
    parts = []
    dcov = round(t2t["cov_win_pct"] - pan["cov_win_pct"], 1)
    if abs(dcov) >= 1:
        parts.append(f"window coverage {pan['cov_win_pct']}%\u2192{t2t['cov_win_pct']}% ({dcov:+.1f} pp)")
    dgap = t2t["n_gaps"] - pan["n_gaps"]
    if dgap != 0:
        parts.append(f"gaps {pan['n_gaps']}\u2192{t2t['n_gaps']}")
    dlg = t2t["largest_gap"] - pan["largest_gap"]
    if abs(dlg) >= GAP_MIN:
        parts.append(f"largest gap {pan['largest_gap']//1000}kb\u2192{t2t['largest_gap']//1000}kb")
    dblk = t2t["longest_block"] - pan["longest_block"]
    if abs(dblk) >= GAP_MIN:
        parts.append(f"longest contiguous block {pan['longest_block']//1000}kb\u2192{t2t['longest_block']//1000}kb")
    if t2t["identity"] and pan["identity"]:
        di = round(t2t["identity"] - pan["identity"], 4)
        if abs(di) >= 0.002:
            parts.append(f"aln identity {pan['identity']:.3f}\u2192{t2t['identity']:.3f} ({di:+.3f})")
    if (t2t["frac_rev_inv"] is not None and pan["frac_rev_inv"] is not None
            and abs(t2t["frac_rev_inv"] - pan["frac_rev_inv"]) >= 0.1):
        parts.append(f"chimp orientation frac_rev {pan['frac_rev_inv']}\u2192{t2t['frac_rev_inv']}")
    if not parts:
        return "T2T and panTro6 agree closely at this locus (no material change)"
    return "T2T vs panTro6: " + "; ".join(parts)

rows = load_inv(inv_tsv)
recs = []
for inv in rows:
    d = os.path.join(stageb, inv["inv_id"])
    t2t_segs = read_paf_segments(os.path.join(d, "chimpT2T.paf"), inv["chrom"], inv["ws"], inv["we"])
    pan_segs = read_paf_segments(os.path.join(d, "panTro6.paf"), inv["chrom"], inv["ws"], inv["we"])
    mt = metrics(t2t_segs, inv); mp = metrics(pan_segs, inv)
    rec = dict(inv_id=inv["inv_id"], chrom=inv["chrom"], start=inv["start"], end=inv["end"],
               size_bp=inv["size_bp"], af=inv["af"], disease=inv["disease"],
               t2t=mt, panTro6=mp, note=note(mt, mp, inv))
    recs.append(rec)

cols = ["cov_win_pct", "cov_inv_pct", "n_segments", "n_gaps", "largest_gap",
        "longest_block", "identity", "frac_rev_inv"]
with open(out_tsv, "w") as fh:
    hdr = ["inv_id", "chrom", "start", "end", "size_bp", "af", "disease_genes"]
    for c in cols:
        hdr += [f"t2t_{c}", f"panTro6_{c}"]
    hdr += ["note"]
    fh.write("\t".join(hdr) + "\n")
    for r in recs:
        line = [r["inv_id"], r["chrom"], str(r["start"]), str(r["end"]),
                str(r["size_bp"]), r["af"], r["disease"]]
        for c in cols:
            line += [str(r["t2t"][c]), str(r["panTro6"][c])]
        line += [r["note"]]
        fh.write("\t".join(line) + "\n")

with open(out_json, "w") as fh:
    json.dump(recs, fh, indent=1)

# summary to stdout
n = len(recs)
pan_empty = sum(1 for r in recs if r["panTro6"]["n_segments"] == 0)
t2t_empty = sum(1 for r in recs if r["t2t"]["n_segments"] == 0)
gap_improved = sum(1 for r in recs if r["t2t"]["n_gaps"] < r["panTro6"]["n_gaps"])
cov_improved = sum(1 for r in recs if r["t2t"]["cov_win_pct"] > r["panTro6"]["cov_win_pct"] + 1)
pan_only_empty = sum(1 for r in recs if r["panTro6"]["n_segments"] == 0 and r["t2t"]["n_segments"] > 0)
sys.stderr.write(
    f"loci={n} panTro6_empty={pan_empty} t2t_empty={t2t_empty} "
    f"panTro6_empty_but_t2t_covers={pan_only_empty} "
    f"cov_improved(>1pp)={cov_improved} fewer_gaps={gap_improved}\n")
print(json.dumps(dict(loci=n, panTro6_empty=pan_empty, t2t_empty=t2t_empty,
      panTro6_empty_but_t2t_covers=pan_only_empty,
      cov_improved_gt1pp=cov_improved, fewer_gaps=gap_improved)))
