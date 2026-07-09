#!/usr/bin/env python3
"""Build an HTML gallery of all inversion SVbyEye plots + orientation-QC table.
Sorted by genomic position; flags AF-discordant loci and empty plots.
"""
import sys, os, glob, csv, html

qc_tsv, plots_dir, out_html = sys.argv[1], sys.argv[2], sys.argv[3]

def chrom_key(c):
    c=c.replace("chr","")
    return (0,int(c)) if c.isdigit() else (1, {"X":23,"Y":24,"M":25}.get(c,99))

rows=[]
with open(qc_tsv) as fh:
    r=csv.DictReader(fh, delimiter="\t")
    for row in r: rows.append(row)
rows.sort(key=lambda x:(chrom_key(x["chrom"]), int(x["start"])))

def fnum(x):
    try: return float(x)
    except (ValueError, TypeError): return float("nan")

n_total=len(rows); n_with_plot=0; n_placeholder=0; discord=[]
cards=[]
for row in rows:
    iid=row["inv_id"]
    png=os.path.join(plots_dir, iid+".png")
    has=os.path.exists(png)
    if has: n_with_plot+=1
    exp=fnum(row.get("inverted_af_expected")); obs=fnum(row.get("frac_inverted_observed"))
    d = fnum(row.get("af_concordance"))  # polarity-aware
    if d==d and d>0.25: discord.append((iid,exp,obs,d))
    flag=""
    if row.get("n_hap_assessed","0")=="0": flag="⚠ no haplotypes span locus (below detection size)"; n_placeholder+=1
    elif d==d and d>0.25: flag=f"⚠ low AF concordance (Δ={d:.2f})"
    rel=os.path.basename(png); trel=iid+".track.png"; grel=iid+".grad.png"
    has_track=os.path.exists(os.path.join(plots_dir,trel)); has_grad=os.path.exists(os.path.join(plots_dir,grel))
    pol=row.get('polarity','NA'); conc=row.get('af_concordance','NA')
    imgs=""
    if has_track: imgs+='<a href="'+trel+'"><img loading="lazy" src="'+trel+'" title="population orientation tracks (binary)"></a>'
    if has_grad: imgs+='<a href="'+grel+'"><img loading="lazy" src="'+grel+'" title="directional gradient"></a>'
    if has: imgs+='<a href="'+rel+'"><img loading="lazy" src="'+rel+'" title="SVbyEye miropeat"></a>'
    if not imgs: imgs='<div class="noimg">no plot</div>'
    cards.append(f"""
    <div class="card">
      <div class="hdr">{html.escape(iid)} <span class="flag">{html.escape(flag)}</span></div>
      <div class="meta">{html.escape(row['chrom'])}:{int(row['start']):,}-{int(row['end']):,}
        &nbsp;|&nbsp; {int(row['size_bp']):,} bp
        &nbsp;|&nbsp; AF<sub>exp</sub>={row.get('inverted_af_expected')}
        &nbsp;|&nbsp; frac<sub>obs</sub>={row.get('frac_inverted_observed')}
        &nbsp;|&nbsp; concordance={conc} ({pol})
        &nbsp;|&nbsp; haps={row.get('n_hap_assessed')} (ref={row.get('n_reference')}, inv={row.get('n_inverted')}, amb={row.get('n_ambiguous')})
        &nbsp;|&nbsp; recur={row.get('recurrent')}</div>
      <div class="imgs">{imgs}</div>
    </div>""")

summary=f"""<div class="summary">
  <b>{n_total}</b> inversions &nbsp;|&nbsp; <b>{n_with_plot}</b> with plots &nbsp;|&nbsp;
  <b>{n_placeholder}</b> with no spanning haplotype &nbsp;|&nbsp;
  <b>{len(discord)}</b> low polarity-aware concordance (Δ&gt;0.25)
</div>"""

doc=f"""<!doctype html><html><head><meta charset="utf-8"><title>Inversions — SVbyEye</title>
<style>
body{{font-family:-apple-system,Arial,sans-serif;margin:16px;background:#fafafa;color:#222}}
h1{{font-size:20px}} .summary{{padding:8px;background:#eef;border-radius:6px;margin:8px 0}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;margin:14px 0;box-shadow:0 1px 3px #0001}}
.hdr{{font-weight:700;font-size:15px}} .flag{{color:#b00;font-weight:600;font-size:12px}}
.meta{{color:#555;font-size:12px;margin:4px 0}}
img{{max-width:100%;height:auto;border:1px solid #eee;border-radius:4px}}
.noimg{{color:#999;padding:20px}}
</style></head><body>
<h1>Inversions visualized with SVbyEye (assembly haplotypes vs hg38)</h1>
{summary}
{''.join(cards)}
</body></html>"""
with open(out_html,"w") as f: f.write(doc)
print(f"gallery: {out_html}  ({n_with_plot}/{n_total} plots, {n_placeholder} empty, {len(discord)} discordant)")
