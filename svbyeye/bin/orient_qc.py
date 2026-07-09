#!/usr/bin/env python3
"""Orientation QC: for each inversion, classify each assembly haplotype (query
contig) as inverted or reference by the dominant alignment strand across the
inversion interval, then compare observed inverted fraction to Inverted_AF.

Reads stageb/<inv_id>/*.paf (per-sample records) produced by distribute_paf.py.
Output: qc/orientation_qc.tsv
"""
import sys, os, glob
from collections import defaultdict

inv_tsv, stageb_root, out_tsv = sys.argv[1], sys.argv[2], sys.argv[3]
MIN_COV_FRAC = 0.30   # a contig must cover >=30% of the inversion interval to be assessed

# load inversions
invs = []
with open(inv_tsv) as fh:
    h = fh.readline().rstrip("\n").split("\t"); ix={c:i for i,c in enumerate(h)}
    for line in fh:
        p=line.rstrip("\n").split("\t")
        invs.append((p[ix["inv_id"]], p[ix["chrom"]], int(p[ix["start"]]), int(p[ix["end"]]),
                     p[ix["inverted_af"]], p[ix["size_bp"]], p[ix["recurrent"]]))

def clip_overlap(a0,a1,b0,b1):
    lo=max(a0,b0); hi=min(a1,b1); return max(0,hi-lo)

rows=[]
for inv_id, chrom, istart, iend, af, size_bp, recur in invs:
    d=os.path.join(stageb_root, inv_id)
    L=iend-istart
    per_contig=defaultdict(list)  # q.name -> [(ts,te,strand)]
    for paf in glob.glob(os.path.join(d,"*.paf")):
        with open(paf) as fh:
            for line in fh:
                f=line.split("\t")
                if len(f)<12: continue
                if f[5]!=chrom: continue
                try: ts=int(f[7]); te=int(f[8])
                except ValueError: continue
                strand=f[4]
                if clip_overlap(ts,te,istart,iend)>0:
                    per_contig[f[0]].append((ts,te,strand))
    n_hap=0; n_inv=0; n_ref=0; n_ambig=0
    for q,recs in per_contig.items():
        rev=0; fwd=0
        for ts,te,strand in recs:
            ov=clip_overlap(ts,te,istart,iend)
            if strand=="-": rev+=ov
            else: fwd+=ov
        cov=rev+fwd
        if L>0 and cov < MIN_COV_FRAC*L:  # contig barely covers interval -> skip
            continue
        n_hap+=1
        frac_rev = rev/cov if cov>0 else 0
        if frac_rev>0.6: n_inv+=1
        elif frac_rev<0.4: n_ref+=1
        else: n_ambig+=1
    frac_inverted = (n_inv/n_hap) if n_hap>0 else float("nan")
    rows.append([inv_id, chrom, istart, iend, size_bp, af, recur,
                 n_hap, n_ref, n_inv, n_ambig,
                 ("%.4f"%frac_inverted) if n_hap>0 else "NA"])

with open(out_tsv,"w") as out:
    out.write("inv_id\tchrom\tstart\tend\tsize_bp\tinverted_af_expected\trecurrent\t"
              "n_hap_assessed\tn_reference\tn_inverted\tn_ambiguous\tfrac_inverted_observed\t"
              "af_concordance\tpolarity\n")
    for r in rows:
        exp=None; obs=None
        try: exp=float(r[5])
        except (ValueError,TypeError): pass
        try: obs=float(r[11])
        except (ValueError,TypeError): pass
        if exp is not None and obs is not None:
            d_same=abs(obs-exp); d_flip=abs(obs-(1-exp))
            conc="%.4f"%min(d_same,d_flip)
            pol="hg38-ref" if d_same<=d_flip else "hg38-inv"
        else:
            conc="NA"; pol="NA"
        out.write("\t".join(map(str,r))+"\t"+conc+"\t"+pol+"\n")

# quick concordance summary to stderr
import statistics
pairs=[]
for r in rows:
    try:
        exp=float(r[5]); obs=float(r[11])
        if exp==exp and obs==obs: pairs.append((exp,obs))
    except (ValueError, TypeError): pass
sys.stderr.write(f"inversions={len(rows)} with_AF_and_obs={len(pairs)}\n")
if len(pairs)>=3:
    xs=[p[0] for p in pairs]; ys=[p[1] for p in pairs]
    mx=statistics.mean(xs); my=statistics.mean(ys)
    num=sum((x-mx)*(y-my) for x,y in pairs)
    den=(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))**0.5
    r=num/den if den>0 else float("nan")
    mad=statistics.median(abs(x-y) for x,y in pairs)
    sys.stderr.write(f"Pearson(exp_AF, obs_frac)={r:.3f}  median|exp-obs|={mad:.3f}\n")

# polarity-aware concordance
conc=[]
for r in rows:
    try:
        exp=float(r[5]); obs=float(r[11])
        conc.append(min(abs(obs-exp),abs(obs-(1-exp))))
    except (ValueError,TypeError): pass
if conc:
    conc.sort(); med=conc[len(conc)//2]
    within10=sum(1 for c in conc if c<0.10)/len(conc)
    within20=sum(1 for c in conc if c<0.20)/len(conc)
    sys.stderr.write(f"polarity-aware: n={len(conc)} median|Δ|={med:.3f}  within0.10={within10:.2f}  within0.20={within20:.2f}\n")
