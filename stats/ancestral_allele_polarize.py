#!/usr/bin/env python3
"""Polarize inversions by the multi-SNP ancestral-allele method (independent of alignment).
For each SNP in tight LD with the inversion, the orientation linked to the ANCESTRAL allele
(AA field = chimp ancestral state) is the ancestral orientation. Vote across all tag SNPs."""
import csv, os, re, subprocess, sys, math
WD="/projects/standard/hsiehph/sauer354/di/polarize"
VD="/projects/standard/hsiehph/sauer354/di/vcfs"
FLANK=100000; MIN_TAG_R=0.6; MIN_SAMPLES=12

def vcf_path(ch): return f"{VD}/{ch}.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz"

def load_callset():
    rows=list(csv.DictReader(open(WD+"/ref/callset.tsv"),delimiter="\t"))
    hdr=rows[0].keys()
    meta={'seqnames','start','end','width','inv_id','arbigent_genotype','misorient_info',
          'orthog_tech_support','inversion_category','inv_AF'}
    samples=[c for c in hdr if c not in meta]
    return rows, samples

def inv_count(gt):
    g=gt.split("_")[0]
    if g in ("0|0","00"): return 0
    if g in ("1|1",): return 2
    if g in ("0|1","1|0"): return 1
    return None  # noreads/./././idup/0000

def vcf_samples(ch):
    import gzip
    with gzip.open(vcf_path(ch),'rt') as fh:
        for ln in fh:
            if ln.startswith("#CHROM"):
                return ln.rstrip().split("\t")[9:]
    return []

def hgid(s):
    m=re.search(r'(HG\d+|NA\d+)',s); return m.group(1) if m else s

def corr(x,y):
    n=len(x)
    if n<MIN_SAMPLES: return None
    mx=sum(x)/n; my=sum(y)/n
    sxy=sum((a-mx)*(b-my) for a,b in zip(x,y))
    sxx=sum((a-mx)**2 for a in x); syy=sum((b-my)**2 for b in y)
    if sxx<1e-9 or syy<1e-9: return None
    return sxy/math.sqrt(sxx*syy)

def main():
    rows,csamples=load_callset()
    out=[]
    for r in rows:
        ch=r['seqnames']; s,e=int(r['start']),int(r['end']); iid=r['inv_id2'] if 'inv_id2' in r else r['inv_id']
        if not os.path.exists(vcf_path(ch)): out.append((r,None,0,'no_vcf')); continue
        invc={hgid(sm): inv_count(r[sm]) for sm in csamples}
        carriers=sum(1 for v in invc.values() if v and v>0)
        if carriers<2: out.append((r,None,0,'no_carriers')); continue
        vsamp=[hgid(x) for x in vcf_samples(ch)]
        # tabix the locus +/- FLANK
        try:
            lines=subprocess.check_output(['tabix',vcf_path(ch),f'{ch}:{max(1,s-FLANK)}-{e+FLANK}'],
                                          stderr=subprocess.DEVNULL).decode().splitlines()
        except: lines=[]
        votes=[]  # (weight, inverted_is_ancestral)
        for ln in lines:
            f=ln.split("\t")
            if len(f)<10: continue
            ref,alt,info=f[3],f[4],f[7]
            if len(ref)!=1 or len(alt)!=1: continue
            m=re.search(r'AA=([ACGTacgt])',info)
            if not m: continue
            aa=m.group(1).upper()
            if aa not in (ref,alt): continue
            gts=f[9:]
            xs=[];ys=[]
            for vs,g in zip(vsamp,gts):
                ic=invc.get(vs)
                if ic is None: continue
                gg=g.split(':')[0]
                if '.' in gg: continue
                ac=gg.count('1')
                xs.append(ic); ys.append(ac)
            r2=corr(xs,ys)
            if r2 is None or abs(r2)<MIN_TAG_R: continue
            # allele linked to inverted: r>0 -> ALT, r<0 -> REF
            inv_allele = alt if r2>0 else ref
            inverted_is_anc = (inv_allele==aa)
            votes.append((r2*r2, inverted_is_anc))
        if not votes: out.append((r,None,0,'no_tag_snps')); continue
        w_inv=sum(w for w,a in votes if a); w_dir=sum(w for w,a in votes if not a)
        flip=1 if w_inv>w_dir else 0
        ntag=len(votes); frac=max(w_inv,w_dir)/(w_inv+w_dir)
        conf='high' if (ntag>=5 and frac>=0.8) else ('moderate' if (ntag>=3 and frac>=0.7) else 'low')
        out.append((r,flip,ntag,conf))
    # write
    with open(WD+'/ancestral_allele_calls.tsv','w') as fh:
        fh.write("inv_id\tchrom\tstart\tend\taa_flip\tn_tag\tconfidence\n")
        for r,flip,ntag,conf in out:
            fh.write(f"{r['inv_id']}\t{r['seqnames']}\t{r['start']}\t{r['end']}\t{'' if flip is None else flip}\t{ntag}\t{conf}\n")
    called=[o for o in out if o[1] is not None]
    from collections import Counter
    print(f"called {len(called)}/{len(out)} loci")
    print("confidence:",dict(Counter(o[3] for o in called)))
if __name__=="__main__": main()
