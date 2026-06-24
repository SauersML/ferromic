#!/usr/bin/env python3
"""ANCHORED PanGenie reconstruction. The unanchored LD-block proxy failed (captured flanking
haplotype). Here we anchor with our ACCURATE 44-sample ArbiGent inversion genotypes: in the
sample overlap (HG/NA names shared with the 718-sample PanGenie decompose VCF), find the
decomposed variants whose genotype MATCHES the true inversion genotype (r^2>0.9) -> those ARE
the inversion's decomposed representation -> build the inversion proxy across all 718 samples.
Then AA: flanking SNPs in LD with the proxy, polarized by chimp-ancestral allele (wAA, by
coordinate). Validates on gold loci (common -> well-anchored in the 12-sample overlap)."""
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
    if n < 6: return 0
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


def inv_geno(gt):
    g = gt.split('_')[0]
    return {'0|0': 0, '00': 0, '1|1': 2, '0|1': 1, '1|0': 1}.get(g)


def main():
    # 718 sample names
    s718 = [s.strip() for s in subprocess.check_output(['bash', '-c', "zcat %s | grep -m1 '#CHROM' | cut -f10- | tr '\\t' '\\n'" % V718]).decode().splitlines()]
    pos718 = {s: i for i, s in enumerate(s718)}
    # callset (ArbiGent) inversion genotypes
    cs = list(csv.DictReader(open(WD + '/ref/callset.tsv'), delimiter='\t'))
    meta = {'seqnames', 'start', 'end', 'width', 'inv_id', 'arbigent_genotype', 'misorient_info', 'orthog_tech_support', 'inversion_category', 'inv_AF'}
    csamp = [c for c in cs[0].keys() if c not in meta]
    overlap = [c for c in csamp if c in pos718]   # HG/NA shared samples
    ov_idx718 = [pos718[c] for c in overlap]
    csrow = {}
    for r in cs:
        ch = r['seqnames'] if r['seqnames'].startswith('chr') else 'chr' + r['seqnames']
        csrow[(ch, int(r['start']))] = r
    pol = list(csv.DictReader(open(WD + '/ref/inversion_polarity.tsv'), delimiter='\t'))
    out = open(WD + '/aa_anchored_calls.tsv', 'w')
    out.write("inv_id\tflip\tntag\tfrac\tn_anchor_carr\tn_invtag\tconfidence\tgold_tier\tgold_flip\n")
    print("overlap samples (44-callset in 718):", len(overlap))
    for x in pol:
        ch = x['chrom'] if x['chrom'].startswith('chr') else 'chr' + x['chrom']
        s, e = int(x['start']), int(x['end'])
        # anchor: ArbiGent genotype in overlap samples
        crow = None
        for ds in range(-3000, 3001, 250):
            if (ch, s + ds) in csrow: crow = csrow[(ch, s + ds)]; break
        if not crow: continue
        anchor = [inv_geno(crow[c]) for c in overlap]
        valid_a = [i for i, a in enumerate(anchor) if a is not None]
        if len(valid_a) < 6: continue
        carr = sum(1 for i in valid_a if anchor[i] > 0)
        if carr < 1: continue
        a_vec = [anchor[i] for i in valid_a]
        if len(set(a_vec)) < 2: continue  # need variation to anchor
        rows718 = tab(V718, "%s:%d-%d" % (ch, max(1, s - 120000), e + 120000))
        if not rows718: continue
        var = []
        for ln in rows718:
            f = ln.split('\t')
            if len(f) < 10 or len(f[3]) != 1 or len(f[4]) != 1: continue
            g = f[9:]
            var.append((int(f[1]), f[3], f[4], g))
        # find inversion's decomposed variants = those matching the anchor in overlap.
        # CRITICAL: align each variant's direction to the anchor (some decomposed alleles
        # are anti-correlated, i.e. their ALT marks DIRECT) so the proxy is consistent.
        invtags = []  # (variant, sign) sign=+1 if ALT~inverted, -1 if ALT~direct
        for v in var:
            if not (s <= v[0] <= e): continue
            ov_g = [gtn(v[3][ov_idx718[i]]) for i in valid_a]
            if any(z is None for z in ov_g): continue
            if r2(a_vec, ov_g) > 0.85:
                cc = corr(a_vec, ov_g)
                invtags.append((v, 1 if (cc is None or cc >= 0) else -1))
        if len(invtags) < 2: continue
        # proxy across all 718 (aligned dosage: 2-d for anti-correlated variants)
        n718 = len(s718)
        proxy = []
        for k in range(n718):
            vals = []
            for v, sgn in invtags:
                z = gtn(v[3][k])
                if z is not None:
                    vals.append(z if sgn > 0 else 2 - z)
            if not vals: proxy.append(None)
            else: proxy.append(round(sum(vals) / len(vals)))
        keep = [k for k in range(n718) if proxy[k] is not None]
        if len(keep) < 200: continue
        pvec = [proxy[k] for k in keep]
        if len(set(pvec)) < 2: continue
        # AA join from wAA
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
            sv = [gtn(v[3][k]) for k in keep]
            if any(z is None for z in sv): continue
            c = corr(pvec, sv)
            if c is None or abs(c) < 0.5: continue
            inv_allele = v[2] if c > 0 else v[1]
            votes.append((c * c, inv_allele == aa))
        if not votes: continue
        wi = sum(w for w, a in votes if a); wd = sum(w for w, a in votes if not a)
        flip = 1 if wi > wd else 0; n = len(votes); frac = max(wi, wd) / (wi + wd)
        conf = 'high' if (n >= 8 and frac >= 0.85 and carr >= 3) else ('moderate' if (n >= 4 and frac >= 0.75) else 'low')
        out.write("%s\t%d\t%d\t%.3f\t%d\t%d\t%s\t%s\t%s\n" % (x['inv_id'], flip, n, frac, carr, len(invtags), conf, x['evidence_tier'], x['flip_ref_polarity']))
        out.flush()
    out.close()
    rows = list(csv.DictReader(open(WD + '/aa_anchored_calls.tsv'), delimiter='\t'))
    g = [r for r in rows if r['gold_tier'].startswith('gold') and r['confidence'] in ('high', 'moderate')]
    ag = sum(1 for r in g if r['flip'] == r['gold_flip'])
    print("anchored-AA: %d calls; gold high/mod agree %d/%d" % (len(rows), ag, len(g)))


main()
