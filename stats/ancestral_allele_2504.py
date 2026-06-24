#!/usr/bin/env python3
"""AA polarization using the 1000G 30x ensemble SV callset (3202 samples, hg38, 719 INV) for
inversion genotypes -- catches 15/31 hard loci (vs 4 in the phase3 SV map) -- combined with
the phase3 SNPs+AA (2504 samples). sv30x is subset+reordered to the 2504 phase3 samples so
genotypes align. Loci lifted hg38->hg19 for the SNP region only."""
import csv, gzip, re, subprocess, os, math
WD = "/projects/standard/hsiehph/sauer354/di/polarize"
P3 = WD + '/kg3/ALL.chr%s.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz'


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


def gtn(g):
    a = g.split(':')[0].replace('|', '/')
    if '.' in a: return None
    return a.count('1')


def main():
    blocks = load_chain(WD + '/hg38ToHg19.over.chain.gz')
    # sample alignment: phase3 order defines the columns; map to sv30x column index
    p3 = [s.strip() for s in open(WD + '/p3_samples.txt')]
    sv_samples = [s.strip() for s in open(WD + '/sv30x_samples.txt')]
    svidx = {s: i for i, s in enumerate(sv_samples)}
    keep = [svidx[s] for s in p3 if s in svidx]   # sv30x columns in phase3 order
    # load sv30x INV records (genotype columns)
    def ov(a0, a1, b0, b1): return max(0, min(a1, b1) - max(a0, b0))
    def sv_hits(ch, s, e):
        try:
            ls = subprocess.check_output(['tabix', WD+'/sv30x.vcf.gz', '%s:%d-%d'%(ch, max(1,s-200), e+200)], stderr=subprocess.DEVNULL, timeout=60).decode().splitlines()
        except Exception:
            try:
                ls = subprocess.check_output(['tabix', WD+'/sv30x.vcf.gz', '%s:%d-%d'%(ch, max(1,s-200), e+200)], stderr=subprocess.DEVNULL, timeout=60).decode().splitlines()
            except Exception:
                return []
        out=[]
        for ln in ls:
            f=ln.rstrip(chr(10)).split(chr(9))
            if 'SVTYPE=INV' not in f[7]: continue
            m=re.search(r';END=(\d+)',f[7])
            if not m: continue
            cc=f[0] if f[0].startswith('chr') else 'chr'+f[0]
            gts=f[9:]
            out.append((cc,int(f[1]),int(m.group(1)),[gtn(gts[i]) for i in keep]))
        return out
    pol = list(csv.DictReader(open(WD + '/ref/inversion_polarity.tsv'), delimiter='\t'))
    out = open(WD + '/aa_sv30x_calls.tsv', 'w')
    out.write("inv_id\tflip\tntag\tfrac\tcarr\tconfidence\tgold_tier\tgold_flip\n")
    for x in pol:
        ch = x['chrom'] if x['chrom'].startswith('chr') else 'chr' + x['chrom']
        s, e = int(x['start']), int(x['end'])
        hit = [r for r in sv_hits(ch,s,e) if ov(s, e, r[1], r[2]) > 0.3 * min(e - s, r[2] - r[1])]
        if not hit: continue
        invg = max(hit, key=lambda r: ov(s, e, r[1], r[2]))[3]
        carr = sum(1 for v in invg if v and v > 0)
        if carr < 5: continue
        a = lift(blocks, ch, s); b = lift(blocks, ch, e)
        if not a or not b or a[0] != b[0]: continue
        h19 = a[0].replace('chr', ''); lo, hi = sorted((a[1], b[1]))
        try:
            lines = subprocess.check_output(['tabix', P3 % h19, '%s:%d-%d' % (h19, max(1, lo - 50000), hi + 50000)],
                                            stderr=subprocess.DEVNULL, timeout=120).decode().splitlines()
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
            sa = [gtn(g) for g in f[9:]]
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
        conf = 'high' if (n >= 8 and frac >= 0.85 and carr >= 20) else ('moderate' if (n >= 4 and frac >= 0.75 and carr >= 10) else 'low')
        out.write("%s\t%d\t%d\t%.3f\t%d\t%s\t%s\t%s\n" % (x['inv_id'], flip, n, frac, carr, conf, x['evidence_tier'], x['flip_ref_polarity']))
        out.flush()
    out.close()
    rows = list(csv.DictReader(open(WD + '/aa_sv30x_calls.tsv'), delimiter='\t'))
    g = [r for r in rows if r['gold_tier'].startswith('gold') and r['confidence'] in ('high', 'moderate')]
    ag = sum(1 for r in g if r['flip'] == r['gold_flip'])
    print("sv30x-AA: %d calls; gold high/mod agree %d/%d" % (len(rows), ag, len(g)))


main()
