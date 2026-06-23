import csv, subprocess, os
WD="/projects/standard/hsiehph/sauer354/di/polarize"
HG38="/projects/standard/hsiehph/sauer354/di/hg38.no_alt.fa"
FLANK=150000
faidx_names=set(l.split('\t')[0] for l in open(HG38+".fai"))
chr_prefixed = any(n.startswith("chr") for n in faidx_names)
def cn(c):
    c=str(c)
    if chr_prefixed: return c if c.startswith("chr") else "chr"+c
    return c[3:] if c.startswith("chr") else c
pol=list(csv.DictReader(open(WD+"/ref/inversion_polarity.tsv"),delimiter="\t"))
ss={r["inv_id"]:r for r in csv.DictReader(open(WD+"/ref/strandseq_polarity.tsv"),delimiter="\t")}
hard=[r for r in pol if r["resolution_status"] in ("provisional","unresolved")]
valid=[r for r in pol if r["inv_id"] in ss and ss[r["inv_id"]]["confidence"] in ("high","moderate") and ss[r["inv_id"]]["strandseq_flip"] in ("0","1")]
seen=set(); want=[]
for r in hard+valid:
    if r["inv_id"] in seen: continue
    seen.add(r["inv_id"]); want.append(r)
print(f"hard={len(hard)} valid={len(valid)} total={len(want)}")
fa=open(WD+"/ref/hg38_regions.fa","w"); man=open(WD+"/ref/regions_manifest.tsv","w")
man.write("inv_id\tregion_name\tchrom\treg_start\treg_end\tloc_start\tloc_end\tis_valid\tss_flip\n")
ok=0
flen={l.split('\t')[0]:int(l.split('\t')[1]) for l in open(HG38+".fai")}
for r in want:
    chrom=cn(r["chrom"]); s,e=int(r["start"]),int(r["end"])
    if chrom not in flen: continue
    rs,re=max(1,s-FLANK),min(flen[chrom],e+FLANK)
    seq=subprocess.check_output(["samtools","faidx",HG38,f"{chrom}:{rs}-{re}"]).decode()
    body="".join(seq.splitlines()[1:])
    if len(body) < (re-rs)*0.8: continue
    name=r["inv_id"].replace(":","_").replace("-","_")
    fa.write(f">{name}\n")
    for i in range(0,len(body),80): fa.write(body[i:i+80]+"\n")
    isv=int(r["inv_id"] in ss and ss[r["inv_id"]]["confidence"] in ("high","moderate"))
    man.write(f"{r['inv_id']}\t{name}\t{chrom}\t{rs}\t{re}\t{s}\t{e}\t{isv}\t{ss.get(r['inv_id'],{}).get('strandseq_flip','')}\n")
    ok+=1
fa.close(); man.close()
print(f"wrote {ok} regions")
