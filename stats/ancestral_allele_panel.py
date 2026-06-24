#!/usr/bin/env python3
"""PANEL-ANCHORED reconstruction -- removes the 12-sample overlap limit entirely.
The PANEL VCF (HPRCandMEL_PangenieNRID.v7, 49 assembly samples) has each inversion as an INV
record WITH genotypes (ALT = inverted orientation) AND ordinary SNV/indel records that join to
the 718-sample decompose VCF by (pos,ref,alt). So:
  1. panel INV genotype (49 samples) = accurate inversion genotype, DIRECTION KNOWN.
  2. panel SNV/indel tags in LD (r^2>0.8) with that INV genotype = inversion tags, sign known.
  3. match each tag (pos,ref,alt) -> 718 decompose record -> dosage across ALL 718 samples.
  4. proxy = sign-aligned mean tag dosage across 718 (inverted dosage, no sign ambiguity, no
     anchor-overlap needed -- the panel is self-contained and covers EVERY inversion).
  5. AA: flanking 718 SNVs polarized by chimp-ancestral allele (wAA, by coordinate)."""
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


def corr(xs, ys):
    n = len(xs)
    if n < 6: return None
    mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs); syy = sum((b - my) ** 2 for b in ys)
    if sxx < 1e-9 or syy < 1e-9: return None
    return sxy / math.sqrt(sxx * syy)


def main():
    pol = list(csv.DictReader(open(WD + '/ref/inversion_polarity.tsv'), delimiter='\t'))
    out = open(WD + '/aa_panel_calls.tsv', 'w')
    out.write("inv_id\tflip\tntag\tfrac\tinv_af718\tn_paneltag\tnref\tcarr98\tconfidence\tgold_tier\tgold_flip\n")
    for x in pol:
        ch = x['chrom'] if x['chrom'].startswith('chr') else 'chr' + x['chrom']
        s, e = int(x['start']), int(x['end'])
        L = max(1, e - s)
        # 1. MERGE all overlapping panel INV records (near-duplicate breakpoints split the
        # carriers across many records) -> a haplotype is inverted if ANY duplicate calls it.
        prows = tab(PANEL, "%s:%d-%d" % (ch, max(1, s - 8000), e + 8000))
        invrecs = []
        for ln in prows:
            f = ln.split('\t')
            if 'SVTYPE=INV' not in f[7]: continue
            m = re.search(r'-INV-(\d+)', f[2])
            if not m: continue
            p = int(f[1]); ln_ = int(m.group(1)); pe = p + ln_
            ov = min(e, pe) - max(s, p)
            ro = ov / max(L, ln_) if ov > 0 else 0
            if ro > 0.3: invrecs.append(f)
        if not invrecs: continue
        nps = len(invrecs[0][9:])
        inv_dos = []
        for i in range(nps):
            vals = [gtn(rec[9 + i]) for rec in invrecs]
            vals = [v for v in vals if v is not None]
            inv_dos.append(max(vals) if vals else None)  # union of carriers
        valid = [i for i, d in enumerate(inv_dos) if d is not None]
        if len(valid) < 20: continue
        a_vec = [inv_dos[i] for i in valid]
        carr = sum(a_vec)
        if carr < 2 or len(set(a_vec)) < 2: continue
        # 2. panel tags: SNV/indel in LD with INV genotype across 49 panel samples
        ptags = []  # (pos, ref, alt, sign)
        for ln in tab(PANEL, "%s:%d-%d" % (ch, max(1, s - 100000), e + 100000)):
            f = ln.split('\t')
            if 'SVTYPE=INV' in f[7]: continue
            if len(f) < 10: continue
            tv = [gtn(g) for g in f[9:]]
            tvv = [tv[i] for i in valid]
            if any(z is None for z in tvv) or len(set(tvv)) < 2: continue
            c = corr(a_vec, tvv)
            if c is None or c * c < 0.8: continue
            ptags.append((int(f[1]), f[3], f[4], 1 if c > 0 else -1))
        if len(ptags) < 2: continue
        # 3. read 718 window ONCE; build panel-tag proxy0 (DIRECTED but noisy: cross-cohort
        # LD decay), then REFINE to the inversion's own tight LD-block within the 718 (clean).
        tagset = {(p, r, al): sg for p, r, al, sg in ptags}
        allrows = tab(V718, "%s:%d-%d" % (ch, max(1, s - 120000), e + 120000))
        if not allrows: continue
        allvar = []  # (pos, ref, alt, dosages[718])
        n718 = None
        for ln in allrows:
            f = ln.split('\t')
            if len(f) < 10 or len(f[3]) != 1 or len(f[4]) != 1: continue
            g = [gtn(z) for z in f[9:]]
            if n718 is None: n718 = len(g)
            allvar.append((int(f[1]), f[3], f[4], g))
        if n718 is None: continue
        acc = [0.0] * n718; cnt = [0] * n718; matched = 0
        for (p, r, al, g) in allvar:
            sg = tagset.get((p, r, al))
            if sg is None: continue
            for k in range(n718):
                if g[k] is not None:
                    acc[k] += g[k] if sg > 0 else (2 - g[k]); cnt[k] += 1
            matched += 1
        if matched < 2: continue
        keep = [k for k in range(n718) if cnt[k] > 0]
        if len(keep) < 200: continue
        proxy0 = [acc[k] / cnt[k] for k in keep]  # DIRECTED (inverted dosage), continuous
        if max(proxy0) - min(proxy0) < 0.1: continue
        # REFINE: interior 718 variants concordant with proxy0 = the inversion's own decompose
        refined = []  # (sign, dosages_over_keep)
        for (p, r, al, g) in allvar:
            if not (s - 1000 <= p <= e + 1000): continue
            sv = [g[k] for k in keep]
            if any(z is None for z in sv) or len(set(sv)) < 2: continue
            c = corr(proxy0, sv)
            if c is None or abs(c) < 0.7: continue
            refined.append((1 if c > 0 else -1, sv))
        if len(refined) >= 3:
            proxy = [round(sum((sv[i] if sg > 0 else 2 - sv[i]) for sg, sv in refined) / len(refined)) for i in range(len(keep))]
        else:
            proxy = [round(v) for v in proxy0]  # fall back to panel-tag proxy
        inv_af = sum(proxy) / (2 * len(keep))
        if len(set(proxy)) < 2 or not (0.005 < inv_af < 0.995): continue
        nref = len(refined)
        # 4. AA with flanking 718 SNVs + chimp AA (wAA, by coord)
        waa = {}
        for ln in tab(WAA % ch, "%s:%d-%d" % (ch, max(1, s - 120000), e + 120000)):
            f = ln.split('\t')
            if len(f) < 8 or len(f[3]) != 1 or len(f[4]) != 1: continue
            m = re.search(r'AA=([ACGTacgt])', f[7])
            if m: waa[(int(f[1]), f[3], f[4])] = m.group(1).upper()
        votes = []
        for (p, r, al, g) in allvar:
            if s - 2000 <= p <= e + 2000: continue
            aa = waa.get((p, r, al))
            if aa is None or aa not in (r, al): continue
            sv = [g[k] for k in keep]
            if any(z is None for z in sv): continue
            c = corr(proxy, sv)
            if c is None or abs(c) < 0.5: continue
            inv_allele = al if c > 0 else r
            votes.append((c * c, inv_allele == aa))
        if not votes: continue
        wi = sum(w for w, a in votes if a); wd = sum(w for w, a in votes if not a)
        flip = 1 if wi > wd else 0; n = len(votes); frac = max(wi, wd) / (wi + wd)
        conf = 'high' if (n >= 8 and frac >= 0.85 and nref >= 4 and carr >= 5) else ('moderate' if (n >= 4 and frac >= 0.75 and nref >= 3) else 'low')
        out.write("%s\t%d\t%d\t%.3f\t%.3f\t%d\t%d\t%d\t%s\t%s\t%s\n" % (x['inv_id'], flip, n, frac, inv_af, matched, nref, carr, conf, x['evidence_tier'], x['flip_ref_polarity']))
        out.flush()
    out.close()
    rows = list(csv.DictReader(open(WD + '/aa_panel_calls.tsv'), delimiter='\t'))
    g = [r for r in rows if r['gold_tier'].startswith('gold') and r['confidence'] in ('high', 'moderate')]
    ag = sum(1 for r in g if r['flip'] == r['gold_flip'])
    print("PANEL-AA: %d calls; gold high/mod agree %d/%d" % (len(rows), ag, len(g)))


main()
