#!/usr/bin/env python3
"""AA polarization with Fisher-exact tag SNPs (correct statistic for rare/frequency-
mismatched tags that r^2 misses). Tag = SNP whose alt-allele carriage is significantly
associated with the inversion (Fisher p<1e-3); per-tag ancestral direction (allele enriched
on inverted vs chimp-ancestral AA) voted, weighted by -log10(p)."""
import csv, os, re, subprocess, gzip, math
WD = "/projects/standard/hsiehph/sauer354/di/polarize"
VD = "/projects/standard/hsiehph/sauer354/di/vcfs"
FLANK = 100000; PMAX = 1e-3

def vp(ch): return "%s/%s.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz" % (VD, ch)
def ic(gt):
    g = gt.split('_')[0]
    return {'0|0': 0, '00': 0, '1|1': 2, '0|1': 1, '1|0': 1}.get(g)
def hg(s):
    m = re.search(r'(HG\d+|NA\d+)', s); return m.group(1) if m else s
def vsamp(ch):
    with gzip.open(vp(ch), 'rt') as fh:
        for ln in fh:
            if ln.startswith('#CHROM'): return [hg(x) for x in ln.rstrip().split('\t')[9:]]
    return []
_LF = {}
def lfac(n):
    if n < 2: return 0.0
    if n not in _LF: _LF[n] = sum(math.log(i) for i in range(2, n + 1))
    return _LF[n]
def fisher(a, b, c, d):
    n = a + b + c + d
    def lp(a, b, c, d):
        return (lfac(a + b) + lfac(c + d) + lfac(a + c) + lfac(b + d)
                - lfac(n) - lfac(a) - lfac(b) - lfac(c) - lfac(d))
    p0 = lp(a, b, c, d); tot = 0.0; r1 = a + b; c1 = a + c
    for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1):
        lpx = lp(x, r1 - x, c1 - x, n - r1 - c1 + x)
        if lpx <= p0 + 1e-9: tot += math.exp(lpx)
    return min(1.0, tot)

def main():
    rows = list(csv.DictReader(open(WD + '/ref/callset.tsv'), delimiter='\t'))
    meta = {'seqnames', 'start', 'end', 'width', 'inv_id', 'arbigent_genotype',
            'misorient_info', 'orthog_tech_support', 'inversion_category', 'inv_AF'}
    cs = [c for c in rows[0].keys() if c not in meta]
    out = []
    for r in rows:
        ch, s, e = r['seqnames'], int(r['start']), int(r['end'])
        if not os.path.exists(vp(ch)): out.append((r, None, 0, 'no_vcf')); continue
        inv = {hg(sm): ic(r[sm]) for sm in cs}
        carr = set(sm for sm, v in inv.items() if v and v > 0)
        non = set(sm for sm, v in inv.items() if v == 0)
        if len(carr) < 3 or len(non) < 5: out.append((r, None, 0, 'few_carr')); continue
        samp = vsamp(ch)
        try:
            lines = subprocess.check_output(['tabix', vp(ch), '%s:%d-%d' % (ch, max(1, s - FLANK), e + FLANK)],
                                            stderr=subprocess.DEVNULL).decode().splitlines()
        except Exception:
            lines = []
        votes = []
        for ln in lines:
            f = ln.split('\t')
            if len(f) < 10: continue
            ref, alt, info = f[3], f[4], f[7]
            if len(ref) != 1 or len(alt) != 1: continue
            m = re.search(r'AA=([ACGTacgt])', info)
            if not m: continue
            aa = m.group(1).upper()
            if aa not in (ref, alt): continue
            gt = {sm: g.split(':')[0] for sm, g in zip(samp, f[9:])}
            def has(sm):
                g = gt.get(sm, '.'); return None if '.' in g else ('1' in g)
            ca = sum(1 for sm in carr if has(sm) is True)
            cn = sum(1 for sm in carr if has(sm) is False)
            na = sum(1 for sm in non if has(sm) is True)
            nn = sum(1 for sm in non if has(sm) is False)
            if ca + cn < 2 or na + nn < 3: continue
            p = fisher(ca, cn, na, nn)
            if p > PMAX: continue
            cfrac = ca / (ca + cn) if (ca + cn) else 0
            nfrac = na / (na + nn) if (na + nn) else 0
            inv_allele = alt if cfrac > nfrac else ref
            votes.append((-math.log10(max(p, 1e-300)), inv_allele == aa))
        if not votes: out.append((r, None, 0, 'no_tag')); continue
        w_i = sum(w for w, a in votes if a); w_d = sum(w for w, a in votes if not a)
        flip = 1 if w_i > w_d else 0; n = len(votes)
        frac = max(w_i, w_d) / (w_i + w_d) if (w_i + w_d) else 0
        conf = 'high' if (n >= 4 and frac >= 0.85) else ('moderate' if (n >= 2 and frac >= 0.75) else 'low')
        out.append((r, flip, n, conf))
    with open(WD + '/aa_fisher_calls.tsv', 'w') as fh:
        fh.write('inv_id\tchrom\tstart\tend\taa_flip\tn_tag\tconfidence\n')
        for r, flip, n, conf in out:
            fh.write('%s\t%s\t%s\t%s\t%s\t%d\t%s\n' % (
                r['inv_id'], r['seqnames'], r['start'], r['end'],
                '' if flip is None else flip, n, conf))
    from collections import Counter
    c = [o for o in out if o[1] is not None]
    print('called', len(c), 'conf', dict(Counter(o[3] for o in c)))

main()
