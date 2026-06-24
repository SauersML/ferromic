#!/usr/bin/env python3
"""COMBINED-ANCHOR reconstruction. Validated anchored method (4/4 gold) used ArbiGent (12
samples in 718). Here we ALSO use the PANEL INV genotype for the 10 panel-samples that are in
the 718 -> union anchor (~22 samples), BOTH accurate inversion genotypes with consistent
direction (0=direct/ref, 2=inverted/alt). For each locus: build the anchor over the union;
find the 718 decomposed alleles matching it (r^2>0.85, sign-aligned to anchor direction);
proxy across all 718; AA with flanking SNVs polarized by chimp-ancestral allele (wAA)."""
import csv, re, subprocess, os, math
WD = "/projects/standard/hsiehph/sauer354/di/polarize"
PANEL = "/projects/standard/hsiehph/hsiehph/DataFromUW_GS/PNG_forPaper/PanGenie/hgsvc_hprc_MELandPygmy/nrid_hg38_newVCFv7_1KG/inputFiles/HPRCandMEL_PangenieNRID.v7.vcf.gz"
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
    if n < 6: return None
    mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs); syy = sum((b - my) ** 2 for b in ys)
    if sxx < 1e-9 or syy < 1e-9: return None
    return sxy / math.sqrt(sxx * syy)


def inv_geno(gt):
    g = gt.split('_')[0]
    return {'0|0': 0, '00': 0, '1|1': 2, '0|1': 1, '1|0': 1}.get(g)


def main():
    s718 = [s.strip() for s in subprocess.check_output(['bash', '-c', "zcat %s | grep -m1 '#CHROM' | cut -f10- | tr '\\t' '\\n'" % V718]).decode().splitlines()]
    pos718 = {s: i for i, s in enumerate(s718)}
    psamp = [s.strip() for s in subprocess.check_output(['bash', '-c', "tabix -H %s | grep -m1 CHROM | cut -f10- | tr '\\t' '\\n'" % PANEL]).decode().splitlines()]
    p_in_718 = [(i, pos718[s]) for i, s in enumerate(psamp) if s in pos718]  # (panel_col, 718_col)
    # ArbiGent callset
    cs = list(csv.DictReader(open(WD + '/ref/callset.tsv'), delimiter='\t'))
    meta = {'seqnames', 'start', 'end', 'width', 'inv_id', 'arbigent_genotype', 'misorient_info', 'orthog_tech_support', 'inversion_category', 'inv_AF'}
    csamp = [c for c in cs[0].keys() if c not in meta]
    arb_in_718 = [(c, pos718[c]) for c in csamp if c in pos718]  # (sample_name, 718_col)
    csrow = {}
    for r in cs:
        ch = r['seqnames'] if r['seqnames'].startswith('chr') else 'chr' + r['seqnames']
        csrow[(ch, int(r['start']))] = r
    pol = list(csv.DictReader(open(WD + '/ref/inversion_polarity.tsv'), delimiter='\t'))
    out = open(WD + '/aa_anchored2_calls.tsv', 'w')
    out.write("inv_id\tflip\tntag\tfrac\tn_anchor\tcarr\tn_invtag\tconfidence\tgold_tier\tgold_flip\n")
    print("ArbiGent-in-718:", len(arb_in_718), " panel-in-718:", len(p_in_718))
    for x in pol:
        ch = x['chrom'] if x['chrom'].startswith('chr') else 'chr' + x['chrom']
        s, e = int(x['start']), int(x['end'])
        L = max(1, e - s)
        # --- build combined anchor over 718 columns: {718col: dosage} ---
        anc = {}
        crow = None
        for ds in range(-3000, 3001, 250):
            if (ch, s + ds) in csrow: crow = csrow[(ch, s + ds)]; break
        if crow:
            for name, col in arb_in_718:
                d = inv_geno(crow[name])
                if d is not None: anc[col] = d
        # panel INV genotype (merge duplicate INV records) for panel-in-718 samples
        invrecs = []
        for ln in tab(PANEL, "%s:%d-%d" % (ch, max(1, s - 8000), e + 8000)):
            f = ln.split('\t')
            if 'SVTYPE=INV' not in f[7]: continue
            m = re.search(r'-INV-(\d+)', f[2])
            if not m: continue
            p = int(f[1]); pe = p + int(m.group(1)); ov = min(e, pe) - max(s, p)
            if ov > 0 and ov / max(L, pe - p) > 0.3: invrecs.append(f)
        for pcol, col in p_in_718:
            if invrecs:
                vals = [gtn(rec[9 + pcol]) for rec in invrecs]
                vals = [v for v in vals if v is not None]
                if vals and col not in anc: anc[col] = max(vals)  # ArbiGent takes precedence
        if len(anc) < 6: continue
        acols = sorted(anc)
        a_vec = [anc[c] for c in acols]
        carr = sum(1 for v in a_vec if v > 0)
        if carr < 1 or len(set(a_vec)) < 2: continue
        # --- find inversion's decomposed alleles in 718 (interior) matching the anchor ---
        rows718 = tab(V718, "%s:%d-%d" % (ch, max(1, s - 120000), e + 120000))
        if not rows718: continue
        var = []
        for ln in rows718:
            f = ln.split('\t')
            if len(f) < 10 or len(f[3]) != 1 or len(f[4]) != 1: continue
            var.append((int(f[1]), f[3], f[4], [gtn(z) for z in f[9:]]))
        invtags = []
        for (p, r, al, g) in var:
            if not (s <= p <= e): continue
            ov_g = [g[c] for c in acols]
            if any(z is None for z in ov_g): continue
            if r2(a_vec, ov_g) > 0.85:
                cc = corr(a_vec, ov_g)
                invtags.append((g, 1 if (cc is None or cc >= 0) else -1))
        if len(invtags) < 2: continue
        n718 = len(s718)
        proxy = []
        for k in range(n718):
            vals = [(gg[k] if sg > 0 else 2 - gg[k]) for gg, sg in invtags if gg[k] is not None]
            proxy.append(round(sum(vals) / len(vals)) if vals else None)
        keep = [k for k in range(n718) if proxy[k] is not None]
        if len(keep) < 200: continue
        pvec = [proxy[k] for k in keep]
        if len(set(pvec)) < 2: continue
        # --- AA ---
        waa = {}
        for ln in tab(WAA % ch, "%s:%d-%d" % (ch, max(1, s - 120000), e + 120000)):
            f = ln.split('\t')
            if len(f) < 8 or len(f[3]) != 1 or len(f[4]) != 1: continue
            m = re.search(r'AA=([ACGTacgt])', f[7])
            if m: waa[(int(f[1]), f[3], f[4])] = m.group(1).upper()
        votes = []
        for (p, r, al, g) in var:
            if s - 2000 <= p <= e + 2000: continue
            aa = waa.get((p, r, al))
            if aa is None or aa not in (r, al): continue
            sv = [g[k] for k in keep]
            if any(z is None for z in sv): continue
            c = corr(pvec, sv)
            if c is None or abs(c) < 0.5: continue
            inv_allele = al if c > 0 else r
            votes.append((c * c, inv_allele == aa))
        if not votes: continue
        wi = sum(w for w, a in votes if a); wd = sum(w for w, a in votes if not a)
        flip = 1 if wi > wd else 0; n = len(votes); frac = max(wi, wd) / (wi + wd)
        conf = 'high' if (n >= 8 and frac >= 0.85 and carr >= 3 and len(invtags) >= 3) else ('moderate' if (n >= 4 and frac >= 0.75 and carr >= 2) else 'low')
        out.write("%s\t%d\t%d\t%.3f\t%d\t%d\t%d\t%s\t%s\t%s\n" % (x['inv_id'], flip, n, frac, len(acols), carr, len(invtags), conf, x['evidence_tier'], x['flip_ref_polarity']))
        out.flush()
    out.close()
    rows = list(csv.DictReader(open(WD + '/aa_anchored2_calls.tsv'), delimiter='\t'))
    g = [r for r in rows if r['gold_tier'].startswith('gold') and r['confidence'] in ('high', 'moderate')]
    ag = sum(1 for r in g if r['flip'] == r['gold_flip'])
    print("anchored2-AA: %d calls; gold high/mod agree %d/%d" % (len(rows), ag, len(g)))


main()
