#!/usr/bin/env python3
"""ANCHOR-FREE block + PANEL orientation. Decouples coverage from orientation:
 - COVERAGE: discover the inversion's tight LD-block in [s,e] using ALL 718 samples (no
   per-sample anchor needed to FIND the block).
 - ORIENTATION (the part anchor-free failed at): break the sign with the PANEL -- (a) the 10
   panel-in-718 samples' INV genotype (corr with block proxy), else (b) match block AF to the
   panel inversion AF. Then AA with flanking 718 SNVs + chimp AA (wAA)."""
import csv, re, subprocess, os, math
WD = "/projects/standard/hsiehph/sauer354/di/polarize"
PANEL = "/projects/standard/hsiehph/hsiehph/DataFromUW_GS/PNG_forPaper/PanGenie/hgsvc_hprc_MELandPygmy/nrid_hg38_newVCFv7_1KG/inputFiles/HPRCandMEL_PangenieNRID.v7.vcf.gz"
V718 = "/projects/standard/hsiehph/hsiehph/DataFromUW_GS/PNG_forPaper/PanGenie/hgsvc_hprc_MELandPygmy/nrid_hg38_newVCFv7_1KG/allVCFmerge/temp/merge_decompose1KG_647samples_77PNG.vcf.gz"
WAA = "/projects/standard/hsiehph/hsiehph/DataFromUW_GS/wAA/%s.DNS_NDL_AFR_EAS_PNG_subset.qc.nomissing.phasedNoMissing.wAA.vcf.gz"
os.environ['HTS_CACHE'] = '/tmp'

def gtn(g):
    a = g.split(':')[0].replace('|', '/'); return None if '.' in a else a.count('1')
def tab(v, region):
    try: return subprocess.check_output(['tabix', v, region], stderr=subprocess.DEVNULL, timeout=150).decode().splitlines()
    except Exception: return []
def corr(xs, ys):
    n = len(xs)
    if n < 6: return None
    mx = sum(xs)/n; my = sum(ys)/n
    sxy = sum((a-mx)*(b-my) for a,b in zip(xs,ys)); sxx = sum((a-mx)**2 for a in xs); syy = sum((b-my)**2 for b in ys)
    if sxx < 1e-9 or syy < 1e-9: return None
    return sxy/math.sqrt(sxx*syy)
def r2c(c): return c*c if c is not None else 0

def main():
    s718 = subprocess.check_output(['bash','-c',"zcat %s | grep -m1 '#CHROM' | cut -f10- | tr '\\t' '\\n'" % V718]).decode().split()
    pos718 = {s:i for i,s in enumerate(s718)}
    psamp = subprocess.check_output(['bash','-c',"tabix -H %s | grep -m1 CHROM | cut -f10- | tr '\\t' '\\n'" % PANEL]).decode().split()
    p_in = [(i, pos718[s]) for i,s in enumerate(psamp) if s in pos718]  # (panelcol,718col)
    pol = list(csv.DictReader(open(WD+'/ref/inversion_polarity.tsv'), delimiter='\t'))
    out = open(WD+'/aa_afree_panel_calls.tsv','w')
    out.write("inv_id\tflip\tntag\tfrac\tinv_af\tpanel_af\tblock\torient\tconfidence\tgold_tier\tgold_flip\n")
    for x in pol:
        ch = x['chrom'] if x['chrom'].startswith('chr') else 'chr'+x['chrom']
        s,e = int(x['start']), int(x['end']); L = max(1,e-s)
        # panel inversion AF + 10-sample genotype (merged INV records)
        invrecs = []
        for ln in tab(PANEL, "%s:%d-%d"%(ch,max(1,s-8000),e+8000)):
            f = ln.split('\t')
            if 'SVTYPE=INV' not in f[7]: continue
            m = re.search(r'-INV-(\d+)', f[2])
            if not m: continue
            p=int(f[1]); pe=p+int(m.group(1)); ov=min(e,pe)-max(s,p)
            if ov>0 and ov/max(L,pe-p)>0.3: invrecs.append(f)
        if not invrecs: continue
        nps = len(invrecs[0][9:])
        pdos = []
        for i in range(nps):
            vv=[gtn(rec[9+i]) for rec in invrecs]; vv=[v for v in vv if v is not None]
            pdos.append(max(vv) if vv else None)
        pv=[d for d in pdos if d is not None]
        if not pv: continue
        panel_af = sum(pv)/(2*len(pv))
        if not (0.003 < panel_af < 0.997): continue
        pgeno = {col:pdos[pc] for pc,col in p_in if pdos[pc] is not None}
        # 718 interior block discovery
        rows = tab(V718, "%s:%d-%d"%(ch,max(1,s-120000),e+120000))
        allvar=[]; n718=None
        for ln in rows:
            f=ln.split('\t')
            if len(f)<10 or len(f[3])!=1 or len(f[4])!=1: continue
            g=[gtn(z) for z in f[9:]]
            if n718 is None: n718=len(g)
            allvar.append((int(f[1]),g))
        if n718 is None: continue
        idx=[i for i in range(n718) if all(g[i] is not None for _,g in allvar)] if False else list(range(n718))
        # interior intermediate-AF variants
        interior=[]
        for p,g in allvar:
            if not (s<=p<=e): continue
            gg=[z for z in g if z is not None]
            if len(gg)<300: continue
            af=sum(gg)/(2*len(gg))
            if 0.005<af<0.995: interior.append((p,g))
        if len(interior)<4: continue
        keepall=[k for k in range(n718) if all(g[k] is not None for _,g in interior)]
        if len(keepall)<200: continue
        ivecs=[[g[k] for k in keepall] for _,g in interior]
        # largest mutual-LD block spanning the interval
        best=[]
        for i in range(len(interior)):
            blk=[j for j in range(len(interior)) if r2c(corr(ivecs[i],ivecs[j]))>0.8]
            if len(blk)>len(best): best=blk
        if len(best)<3: continue
        bpos=[interior[j][0] for j in best]
        span=(max(bpos)-min(bpos))/L
        if span<0.25: continue
        # raw proxy = mean ALT dosage over block (unoriented), aligned internally to seed
        seed=best[0]
        proxy_raw=[0.0]*len(keepall)
        for j in best:
            c=corr(ivecs[seed],ivecs[j]); sg=1 if (c is None or c>=0) else -1
            for k in range(len(keepall)): proxy_raw[k]+= ivecs[j][k] if sg>0 else 2-ivecs[j][k]
        proxy_raw=[v/len(best) for v in proxy_raw]
        # ORIENT with panel: (a) corr with 10-sample panel geno, else (b) AF match
        orient=None; how=''
        pg=[]; pp=[]
        kmap={k:i for i,k in enumerate(keepall)}
        for col,d in pgeno.items():
            if col in kmap: pg.append(proxy_raw[kmap[col]]); pp.append(d)
        if len(pp)>=4 and len(set(pp))>=2:
            c=corr(pp,pg)
            if c is not None and abs(c)>0.45: orient = 1 if c>0 else -1; how='geno'
        if orient is None:
            af_raw=sum(proxy_raw)/(2*len(keepall))
            orient = 1 if abs(af_raw-panel_af)<abs((1-af_raw)-panel_af) else -1; how='af'
        proxy=[round(v if orient>0 else 2-v) for v in proxy_raw]
        inv_af=sum(proxy)/(2*len(keepall))
        if len(set(proxy))<2: continue
        # AA
        waa={}
        for ln in tab(WAA%ch, "%s:%d-%d"%(ch,max(1,s-120000),e+120000)):
            f=ln.split('\t')
            if len(f)<8 or len(f[3])!=1 or len(f[4])!=1: continue
            m=re.search(r'AA=([ACGTacgt])',f[7])
            if m: waa[(int(f[1]),f[3],f[4])]=m.group(1).upper()
        # re-read flanking with ref/alt
        votes=[]
        for ln in rows:
            f=ln.split('\t')
            if len(f)<10 or len(f[3])!=1 or len(f[4])!=1: continue
            p=int(f[1])
            if s-2000<=p<=e+2000: continue
            aa=waa.get((p,f[3],f[4]))
            if aa is None or aa not in (f[3],f[4]): continue
            g=[gtn(z) for z in f[9:]]
            sv=[g[k] for k in keepall]
            if any(z is None for z in sv): continue
            c=corr(proxy,sv)
            if c is None or abs(c)<0.5: continue
            inv_allele=f[4] if c>0 else f[3]
            votes.append((c*c, inv_allele==aa))
        if not votes: continue
        wi=sum(w for w,a in votes if a); wd=sum(w for w,a in votes if not a)
        flip=1 if wi>wd else 0; n=len(votes); frac=max(wi,wd)/(wi+wd)
        conf='high' if (n>=8 and frac>=0.85 and how=='geno' and len(best)>=5) else ('moderate' if (n>=4 and frac>=0.75) else 'low')
        out.write("%s\t%d\t%d\t%.3f\t%.3f\t%.3f\t%d\t%s\t%s\t%s\t%s\n"%(x['inv_id'],flip,n,frac,inv_af,panel_af,len(best),how,conf,x['evidence_tier'],x['flip_ref_polarity']))
        out.flush()
    out.close()
    rows=list(csv.DictReader(open(WD+'/aa_afree_panel_calls.tsv'),delimiter='\t'))
    g=[r for r in rows if r['gold_tier'].startswith('gold') and r['confidence'] in ('high','moderate')]
    ag=sum(1 for r in g if r['flip']==r['gold_flip'])
    print("afree-panel: %d calls; gold high/mod agree %d/%d"%(len(rows),ag,len(g)))
main()
