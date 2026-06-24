#!/usr/bin/env python3
"""Resolve inversion polarity from the 718-sample PanGenie decompose VCF (ACCURATE genotypes,
unlike noisy SV callers). The inversion is decomposed into allele-IDs with no INV record, so
reconstruct it INTERNALLY: in the inversion interval, the decomposed inversion variants form a
large mutually-LD block -> take that block's consensus as the inversion genotype proxy (718
samples; no sample-ID mapping or external anchor needed). Then standard AA: flanking SNPs in LD
with the proxy, polarized by their chimp-ancestral allele (joined by coordinate from the
375-sample wAA VCF, since the decompose VCF has no AA). All hg38."""
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
        return subprocess.check_output(['tabix', v, region], stderr=subprocess.DEVNULL, timeout=120).decode().splitlines()
    except Exception:
        return []


def r2(xs, ys):
    n = len(xs)
    if n < 30: return 0
    mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs); syy = sum((b - my) ** 2 for b in ys)
    if sxx < 1e-9 or syy < 1e-9: return 0
    return (sxy * sxy) / (sxx * syy)


def corr(xs, ys):
    n = len(xs)
    mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs); syy = sum((b - my) ** 2 for b in ys)
    if sxx < 1e-9 or syy < 1e-9: return None
    return sxy / math.sqrt(sxx * syy)


def main():
    pol = list(csv.DictReader(open(WD + '/ref/inversion_polarity.tsv'), delimiter='\t'))
    out = open(WD + '/aa_pangenie_calls.tsv', 'w')
    out.write("inv_id\tflip\tntag\tfrac\tinv_af\tblock\tconfidence\tgold_tier\tgold_flip\n")
    for x in pol:
        ch = x['chrom'] if x['chrom'].startswith('chr') else 'chr' + x['chrom']
        s, e = int(x['start']), int(x['end'])
        L = e - s
        rows718 = tab(V718, "%s:%d-%d" % (ch, max(1, s - 120000), e + 120000))
        if not rows718:
            continue
        # parse biallelic variants
        var = []  # (pos, ref, alt, [dosages])
        for ln in rows718:
            f = ln.split('\t')
            if len(f) < 10: continue
            if len(f[3]) != 1 or len(f[4]) != 1: continue
            g = [gtn(x) for x in f[9:]]
            var.append((int(f[1]), f[3], f[4], g))
        if len(var) < 10: continue
        nsamp = len(var[0][3])
        idx = [i for i in range(nsamp) if all(v[3][i] is not None for v in var)]
        if len(idx) < 100: continue
        def vec(v): return [v[3][i] for i in idx]
        # interior variants (inside inversion) with intermediate AF
        interior = [v for v in var if s <= v[0] <= e]
        cand = [v for v in interior if 0.02 < (sum(vec(v)) / (2 * len(idx))) < 0.98]
        if len(cand) < 3:
            cand = [v for v in var if s - 5000 <= v[0] <= e + 5000 and 0.02 < (sum(vec(v)) / (2 * len(idx))) < 0.98]
        if len(cand) < 3: continue
        # inversion proxy = largest mutual-LD block among candidates
        best_block = []
        cvecs = [vec(v) for v in cand]
        for i in range(len(cand)):
            block = [j for j in range(len(cand)) if r2(cvecs[i], cvecs[j]) > 0.8]
            if len(block) > len(best_block):
                best_block = block
        if len(best_block) < 3: continue
        # proxy = mean dosage over the block (rounded per sample)
        proxy = [round(sum(cvecs[j][k] for j in best_block) / len(best_block)) for k in range(len(idx))]
        inv_af = sum(proxy) / (2 * len(idx))
        if inv_af < 0.01 or inv_af > 0.99: continue
        # AA for flanking SNPs from wAA
        waa = {}
        for ln in tab(WAA % ch, "%s:%d-%d" % (ch, max(1, s - 120000), e + 120000)):
            f = ln.split('\t')
            if len(f) < 8 or len(f[3]) != 1 or len(f[4]) != 1: continue
            m = re.search(r'AA=([ACGTacgt])', f[7])
            if m: waa[(int(f[1]), f[3], f[4])] = m.group(1).upper()
        votes = []
        for v in var:
            if s - 2000 <= v[0] <= e + 2000:  # flanking only
                continue
            aa = waa.get((v[0], v[1], v[2]))
            if aa is None or aa not in (v[1], v[2]): continue
            sv = vec(v)
            c = corr(proxy, sv)
            if c is None or abs(c) < 0.5: continue
            inv_allele = v[2] if c > 0 else v[1]
            votes.append((c * c, inv_allele == aa))
        if not votes:
            continue
        wi = sum(w for w, a in votes if a); wd = sum(w for w, a in votes if not a)
        flip = 1 if wi > wd else 0; n = len(votes); frac = max(wi, wd) / (wi + wd)
        conf = 'high' if (n >= 8 and frac >= 0.85 and len(best_block) >= 5) else ('moderate' if (n >= 4 and frac >= 0.75) else 'low')
        out.write("%s\t%d\t%d\t%.3f\t%.3f\t%d\t%s\t%s\t%s\n" % (
            x['inv_id'], flip, n, frac, inv_af, len(best_block), conf, x['evidence_tier'], x['flip_ref_polarity']))
        out.flush()
    out.close()
    rows = list(csv.DictReader(open(WD + '/aa_pangenie_calls.tsv'), delimiter='\t'))
    g = [r for r in rows if r['gold_tier'].startswith('gold') and r['confidence'] in ('high', 'moderate')]
    ag = sum(1 for r in g if r['flip'] == r['gold_flip'])
    print("PanGenie-AA: %d calls; gold high/mod agree %d/%d" % (len(rows), ag, len(g)))


main()
