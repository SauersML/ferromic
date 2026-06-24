#!/usr/bin/env python3
"""ANCHOR-FREE PanGenie reconstruction -- removes the 12-sample coverage limit.
The inversion's decomposed variants form the dominant interval-spanning LD block in [s,e],
and the decompose convention gives their direction (panel REF = DIRECT orientation, so the
inversion's ALT = INVERTED). So: find the largest mutual-LD block of interval variants that
SPANS the inversion (variants near both breakpoints), align all to the majority direction
(consensus 'inverted' dosage), build the proxy across all 718 samples (no anchor needed),
then run the validated AA method (flanking SNPs + chimp-ancestral allele from wAA)."""
import csv, gzip, re, subprocess, os, math
WD = "/projects/standard/hsiehph/sauer354/di/polarize"
V718 = "/projects/standard/hsiehph/hsiehph/DataFromUW_GS/PNG_forPaper/PanGenie/hgsvc_hprc_MELandPygmy/nrid_hg38_newVCFv7_1KG/allVCFmerge/temp/merge_decompose1KG_647samples_77PNG.vcf.gz"
WAA = "/projects/standard/hsiehph/hsiehph/DataFromUW_GS/wAA/%s.DNS_NDL_AFR_EAS_PNG_subset.qc.nomissing.phasedNoMissing.wAA.vcf.gz"
os.environ['HTS_CACHE'] = '/tmp'


def gtn(g):
    a = g.split(':')[0].replace('|', '/')
    return None if '.' in a else a.count('1')


def tab(v, region):
    try:
        return subprocess.check_output(['tabix', v, region], stderr=subprocess.DEVNULL, timeout=150).decode().splitlines()
    except Exception:
        return []


def corr(xs, ys):
    n = len(xs)
    if n < 30: return None
    mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs); syy = sum((b - my) ** 2 for b in ys)
    if sxx < 1e-9 or syy < 1e-9: return None
    return sxy / math.sqrt(sxx * syy)


def r2c(c): return c * c if c is not None else 0


def main():
    pol = list(csv.DictReader(open(WD + '/ref/inversion_polarity.tsv'), delimiter='\t'))
    out = open(WD + '/aa_anchorfree_calls.tsv', 'w')
    out.write("inv_id\tflip\tntag\tfrac\tinv_af\tblock\tspan\tconfidence\tgold_tier\tgold_flip\n")
    for x in pol:
        ch = x['chrom'] if x['chrom'].startswith('chr') else 'chr' + x['chrom']
        s, e = int(x['start']), int(x['end'])
        L = max(1, e - s)
        rows718 = tab(V718, "%s:%d-%d" % (ch, max(1, s - 120000), e + 120000))
        if len(rows718) < 10: continue
        var = []
        for ln in rows718:
            f = ln.split('\t')
            if len(f) < 10 or len(f[3]) != 1 or len(f[4]) != 1: continue
            g = [gtn(z) for z in f[9:]]
            var.append((int(f[1]), f[3], f[4], g))
        n718 = len(var[0][3])
        idx = [i for i in range(n718) if all(v[3][i] is not None for v in var)]
        if len(idx) < 200: continue
        def vec(v): return [v[3][i] for i in idx]
        # interior variants with intermediate AF
        interior = [v for v in var if s <= v[0] <= e and 0.02 < sum(vec(v)) / (2 * len(idx)) < 0.98]
        if len(interior) < 4: continue
        ivecs = [vec(v) for v in interior]
        # largest mutual-LD block (>0.8), then keep only if it SPANS the interval
        best = []
        for i in range(len(interior)):
            blk = [j for j in range(len(interior)) if r2c(corr(ivecs[i], ivecs[j])) > 0.8]
            if len(blk) > len(best): best = blk
        if len(best) < 3: continue
        bpos = [interior[j][0] for j in best]
        span = (max(bpos) - min(bpos)) / L
        if span < 0.3: continue  # must span the inversion, not a local cluster
        # align block variants to majority direction (seed = highest-AF block member)
        seed = max(best, key=lambda j: sum(ivecs[j]))
        proxy_raw = [0.0] * len(idx); cnt = 0
        for j in best:
            c = corr(ivecs[seed], ivecs[j])
            sgn = 1 if (c is None or c >= 0) else -1
            for k in range(len(idx)):
                proxy_raw[k] += ivecs[j][k] if sgn > 0 else (2 - ivecs[j][k])
            cnt += 1
        proxy = [round(p / cnt) for p in proxy_raw]
        inv_af = sum(proxy) / (2 * len(idx))
        if not (0.01 < inv_af < 0.99): continue
        if len(set(proxy)) < 2: continue
        # AA join
        waa = {}
        for ln in tab(WAA % ch, "%s:%d-%d" % (ch, max(1, s - 120000), e + 120000)):
            f = ln.split('\t')
            if len(f) < 8 or len(f[3]) != 1 or len(f[4]) != 1: continue
            m = re.search(r'AA=([ACGTacgt])', f[7])
            if m: waa[(int(f[1]), f[3], f[4])] = m.group(1).upper()
        votes = []
        for v in var:
            if s - 2000 <= v[0] <= e + 2000: continue
            aa = waa.get((v[0], v[1], v[2]))
            if aa is None or aa not in (v[1], v[2]): continue
            sv = vec(v)
            c = corr(proxy, sv)
            if c is None or abs(c) < 0.5: continue
            inv_allele = v[2] if c > 0 else v[1]
            votes.append((c * c, inv_allele == aa))
        if not votes: continue
        wi = sum(w for w, a in votes if a); wd = sum(w for w, a in votes if not a)
        flip = 1 if wi > wd else 0; n = len(votes); frac = max(wi, wd) / (wi + wd)
        conf = 'high' if (n >= 8 and frac >= 0.85 and len(best) >= 5 and span >= 0.5) else ('moderate' if (n >= 4 and frac >= 0.75) else 'low')
        out.write("%s\t%d\t%d\t%.3f\t%.3f\t%d\t%.2f\t%s\t%s\t%s\n" % (x['inv_id'], flip, n, frac, inv_af, len(best), span, conf, x['evidence_tier'], x['flip_ref_polarity']))
        out.flush()
    out.close()
    rows = list(csv.DictReader(open(WD + '/aa_anchorfree_calls.tsv'), delimiter='\t'))
    g = [r for r in rows if r['gold_tier'].startswith('gold') and r['confidence'] in ('high', 'moderate')]
    ag = sum(1 for r in g if r['flip'] == r['gold_flip'])
    print("anchorfree-AA: %d calls; gold high/mod agree %d/%d" % (len(rows), ag, len(g)))


main()
