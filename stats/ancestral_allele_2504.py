#!/usr/bin/env python3
"""2504-sample ancestral-allele polarization. Removes the 44-sample power blocker: loci
that are singletons in the 44-sample callset but present in the 1000G phase3 SV map
(2504 samples) get full-panel inversion genotypes; combined with the 2504-sample 1000G
phase3 SNPs+AA (streamed by remote tabix, hg19), the AA tag method regains power. Lifts
loci hg38->hg19 with a self-contained chain parser."""
import csv, gzip, re, subprocess, os, math
WD = "/projects/standard/hsiehph/sauer354/di/polarize"
os.environ['HTS_CACHE'] = '/tmp'
URLB = WD + '/kg3/ALL.chr%s.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz'


def load_chain(path):
    blocks = {}; tName = None; tpos = qpos = 0; qstrand = '+'; qsize = 0; qName = None
    for ln in gzip.open(path, 'rt'):
        s = ln.strip()
        if s.startswith('chain'):
            f = s.split(); tName = f[2]; tpos = int(f[5]); qName = f[7]; qsize = int(f[8]); qstrand = f[9]; qpos = int(f[10])
        elif s and tName:
            p = s.split(); size = int(p[0]); q0 = qpos if qstrand == '+' else qsize - (qpos + size)
            blocks.setdefault(tName, []).append((tpos, tpos + size, q0, qName, qstrand))
            if len(p) >= 3: tpos += size + int(p[1]); qpos += size + int(p[2])
            else: tName = None
    return blocks


def lift(b, ch, pos):
    for (t0, t1, q0, qn, qst) in b.get(ch, []):
        if t0 <= pos < t1:
            off = pos - t0
            return (qn, q0 + off if qst == '+' else q0 + (t1 - t0) - off)
    return None


def corr(xs, ys):
    n = len(xs)
    if n < 20: return None
    mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs); syy = sum((b - my) ** 2 for b in ys)
    if sxx < 1e-9 or syy < 1e-9: return None
    return sxy / math.sqrt(sxx * syy)


def main():
    blocks = load_chain(WD + '/hg38ToHg19.over.chain.gz')
    pol = list(csv.DictReader(open(WD + '/ref/inversion_polarity.tsv'), delimiter='\t'))
    svrec = {}
    for ln in gzip.open(WD + '/sv1kg.vcf.gz', 'rt'):
        if ln.startswith('#CHROM'): continue
        if ln.startswith('#'): continue
        f = ln.split('\t')
        if 'SVTYPE=INV' not in f[7]: continue
        m = re.search(r';END=(\d+)', f[7])
        if not m: continue
        svrec[(f[0], int(f[1]), int(m.group(1)))] = (f[2], f[9:])
    def ov(a0, a1, b0, b1): return max(0, min(a1, b1) - max(a0, b0))
    out = open(WD + '/aa_2504_calls.tsv', 'w')
    out.write("inv_id\tflip\tntag\tfrac\tcarr2504\tconfidence\tgold_tier\tgold_flip\n")
    for x in pol:
        ch = x['chrom'] if x['chrom'].startswith('chr') else 'chr' + x['chrom']
        a = lift(blocks, ch, int(x['start'])); b = lift(blocks, ch, int(x['end']))
        if not a or not b or a[0] != b[0]: continue
        h19 = a[0].replace('chr', ''); lo, hi = sorted((a[1], b[1]))
        hit = [k for k in svrec if k[0] == h19 and ov(lo, hi, k[1], k[2]) > 0.3 * min(hi - lo, k[2] - k[1])]
        if not hit: continue
        svid, svgt = svrec[hit[0]]
        invg = [None if '.' in g.split(':')[0] else g.split(':')[0].count('1') for g in svgt]
        carr = sum(1 for v in invg if v and v > 0)
        try:
            lines = subprocess.check_output(['tabix', URLB % h19, '%s:%d-%d' % (h19, max(1, lo - 50000), hi + 50000)],
                                            stderr=subprocess.DEVNULL, timeout=180).decode().splitlines()
        except Exception:
            continue
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
            sa = [None if '.' in g.split(':')[0] else g.split(':')[0].count('1') for g in f[9:]]
            xs = []; ys = []
            for iv, sv in zip(invg, sa):
                if iv is None or sv is None: continue
                xs.append(iv); ys.append(sv)
            r = corr(xs, ys)
            if r is None or abs(r) < 0.5: continue
            inv_allele = alt if r > 0 else ref
            votes.append((r * r, inv_allele == aa))
        if not votes: continue
        wi = sum(w for w, a in votes if a); wd = sum(w for w, a in votes if not a)
        flip = 1 if wi > wd else 0; n = len(votes); frac = max(wi, wd) / (wi + wd)
        conf = 'high' if (n >= 5 and frac >= 0.8 and carr >= 10) else ('moderate' if (n >= 3 and frac >= 0.7) else 'low')
        out.flush(); out.write("%s\t%d\t%d\t%.3f\t%d\t%s\t%s\t%s\n" % (x['inv_id'], flip, n, frac, carr, conf, x['evidence_tier'], x['flip_ref_polarity']))
    out.close()
    rows = list(csv.DictReader(open(WD + '/aa_2504_calls.tsv'), delimiter='\t'))
    g = [x for x in rows if x['gold_tier'].startswith('gold')]
    ag = sum(1 for x in g if x['flip'] == x['gold_flip'])
    gh = [x for x in g if x['confidence'] in ('high', 'moderate')]
    agh = sum(1 for x in gh if x['flip'] == x['gold_flip'])
    print("2504-AA: %d calls; vs gold all: %d/%d; vs gold high/mod: %d/%d = %d%%" % (
        len(rows), ag, len(g), agh, len(gh), (100 * agh // len(gh) if gh else 0)))


main()
