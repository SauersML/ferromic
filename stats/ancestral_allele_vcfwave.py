#!/usr/bin/env python3
"""DECOMPOSE-REPLICATION reconstruction -- the complete, anchor-free solution.
The 718 decompose VCF was built by decomposing the panel (each inversion = one INV record,
REF=direct seq, ALT=inverted seq, both EXPLICIT sequences). We replicate that decompose with
vcfwave on each panel INV record -> the exact primitive (pos,ref,alt) variants of the INVERTED
allele. Those tuples are present in the 718 (same decompose) -> match them -> proxy = mean ALT
dosage across ALL 718 samples, DIRECTION KNOWN (these variants ARE the inverted allele, no
anchor, no cross-cohort LD decay). Then AA: flanking 718 SNVs polarized by chimp-ancestral
allele (wAA, by coordinate)."""
import csv, re, subprocess, os, math, tempfile
WD = "/projects/standard/hsiehph/sauer354/di/polarize"
VCFWAVE = "/projects/standard/hsiehph/sauer354/di/mamba/envs/vcflib/bin/vcfwave"
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


def decompose_inv(ch, s, e, L):
    """Run vcfwave on the locus's panel INV records; return set of primitive (pos,ref,alt)."""
    recs = []
    seen = set()
    for ln in tab(PANEL, "%s:%d-%d" % (ch, max(1, s - 8000), e + 8000)):
        f = ln.split('\t')
        if 'SVTYPE=INV' not in f[7]: continue
        m = re.search(r'-INV-(\d+)', f[2])
        if not m: continue
        p = int(f[1]); pe = p + int(m.group(1)); ov = min(e, pe) - max(s, p)
        if ov <= 0 or ov / max(L, pe - p) < 0.3: continue
        key = (f[1], f[3], f[4])
        if key in seen: continue
        seen.add(key)
        recs.append('\t'.join(f[:8]))
    if not recs: return set()
    with tempfile.NamedTemporaryFile('w', suffix='.vcf', delete=False, dir='/tmp') as t:
        t.write("##fileformat=VCFv4.2\n##contig=<ID=%s>\n" % ch)
        t.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for r in recs: t.write(r + "\n")
        tmp = t.name
    prims = set()
    try:
        o = subprocess.check_output([VCFWAVE, '-L', '500000', tmp], stderr=subprocess.DEVNULL, timeout=300).decode()
        for ln in o.splitlines():
            if ln.startswith('#'): continue
            f = ln.split('\t')
            if len(f) < 5: continue
            r, a = f[3], f[4]
            if r == a or r.startswith('<') or a.startswith('<'): continue
            # SNVs + short indels (left as emitted by vcfwave; 718 used the same realigner)
            if len(r) <= 50 and len(a) <= 50:
                prims.add((int(f[1]), r, a))
    except Exception:
        pass
    finally:
        os.unlink(tmp)
    return prims


def main():
    pol = list(csv.DictReader(open(WD + '/ref/inversion_polarity.tsv'), delimiter='\t'))
    out = open(WD + '/aa_vcfwave_calls.tsv', 'w')
    out.write("inv_id\tflip\tntag\tfrac\tinv_af\tn_prim\tn_match\tconfidence\tgold_tier\tgold_flip\n")
    for x in pol:
        ch = x['chrom'] if x['chrom'].startswith('chr') else 'chr' + x['chrom']
        s, e = int(x['start']), int(x['end'])
        L = max(1, e - s)
        prims = decompose_inv(ch, s, e, L)
        if len(prims) < 3: continue
        allrows = tab(V718, "%s:%d-%d" % (ch, max(1, s - 120000), e + 120000))
        if not allrows: continue
        allvar = []
        n718 = None
        for ln in allrows:
            f = ln.split('\t')
            if len(f) < 10 or len(f[3]) != 1 or len(f[4]) != 1: continue
            g = [gtn(z) for z in f[9:]]
            if n718 is None: n718 = len(g)
            allvar.append((int(f[1]), f[3], f[4], g))
        if n718 is None: continue
        # proxy: mean ALT dosage of matched inverted-allele primitives (all same direction)
        acc = [0.0] * n718; cnt = [0] * n718; nmatch = 0
        for (p, r, a, g) in allvar:
            if (p, r, a) not in prims: continue
            for k in range(n718):
                if g[k] is not None:
                    acc[k] += g[k]; cnt[k] += 1
            nmatch += 1
        if nmatch < 3: continue
        keep = [k for k in range(n718) if cnt[k] > 0]
        if len(keep) < 200: continue
        proxy = [round(acc[k] / cnt[k]) for k in keep]
        inv_af = sum(proxy) / (2 * len(keep))
        if len(set(proxy)) < 2 or not (0.003 < inv_af < 0.997): continue
        waa = {}
        for ln in tab(WAA % ch, "%s:%d-%d" % (ch, max(1, s - 120000), e + 120000)):
            f = ln.split('\t')
            if len(f) < 8 or len(f[3]) != 1 or len(f[4]) != 1: continue
            m = re.search(r'AA=([ACGTacgt])', f[7])
            if m: waa[(int(f[1]), f[3], f[4])] = m.group(1).upper()
        votes = []
        for (p, r, a, g) in allvar:
            if s - 2000 <= p <= e + 2000: continue
            aa = waa.get((p, r, a))
            if aa is None or aa not in (r, a): continue
            sv = [g[k] for k in keep]
            if any(z is None for z in sv): continue
            c = corr(proxy, sv)
            if c is None or abs(c) < 0.5: continue
            inv_allele = a if c > 0 else r
            votes.append((c * c, inv_allele == aa))
        if not votes: continue
        wi = sum(w for w, a in votes if a); wd = sum(w for w, a in votes if not a)
        flip = 1 if wi > wd else 0; n = len(votes); frac = max(wi, wd) / (wi + wd)
        conf = 'high' if (n >= 8 and frac >= 0.85 and nmatch >= 5) else ('moderate' if (n >= 4 and frac >= 0.75 and nmatch >= 3) else 'low')
        out.write("%s\t%d\t%d\t%.3f\t%.3f\t%d\t%d\t%s\t%s\t%s\n" % (x['inv_id'], flip, n, frac, inv_af, len(prims), nmatch, conf, x['evidence_tier'], x['flip_ref_polarity']))
        out.flush()
    out.close()
    rows = list(csv.DictReader(open(WD + '/aa_vcfwave_calls.tsv'), delimiter='\t'))
    g = [r for r in rows if r['gold_tier'].startswith('gold') and r['confidence'] in ('high', 'moderate')]
    ag = sum(1 for r in g if r['flip'] == r['gold_flip'])
    print("vcfwave-AA: %d calls; gold high/mod agree %d/%d" % (len(rows), ag, len(g)))


main()
