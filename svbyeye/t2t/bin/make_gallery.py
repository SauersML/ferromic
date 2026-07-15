#!/usr/bin/env python3
"""Build a browsable HTML gallery: per inversion, the T2T-chimp miropeat next to
the panTro6 miropeat, with the T2T-vs-panTro6 diff note. Sorted so the loci where
T2T changes the picture most come first.

Usage: make_gallery.py diff_chimp.json plots_dir out_html [rel_plots_prefix]
"""
import sys, json, os, html

diff_json, plots_dir, out_html = sys.argv[1:4]
rel = sys.argv[4] if len(sys.argv) > 4 else "plots"

recs = json.load(open(diff_json))

def improvement_score(r):
    t, p = r["t2t"], r["panTro6"]
    if p["n_segments"] == 0 and t["n_segments"] > 0:
        return 10_000 + t["cov_win_pct"]
    s = 0.0
    s += (t["cov_win_pct"] - p["cov_win_pct"])
    s += 3 * (p["n_gaps"] - t["n_gaps"])
    s += (t["longest_block"] - p["longest_block"]) / 1000.0
    s += (p["largest_gap"] - t["largest_gap"]) / 1000.0
    return s

recs.sort(key=improvement_score, reverse=True)

def cell(inv_id, ref):
    fn = f"{inv_id}.{ref}.png"
    path = os.path.join(plots_dir, fn)
    if os.path.exists(path):
        return f'<img loading="lazy" src="{rel}/{fn}" alt="{ref}">'
    return f'<div class="missing">{ref}: no plot</div>'

rows = []
for r in recs:
    iid = html.escape(r["inv_id"])
    loc = f'{r["chrom"]}:{r["start"]:,}-{r["end"]:,}'
    dis = "" if r["disease"] in ("NA", "") else f' &middot; genes: {html.escape(r["disease"])}'
    note = html.escape(r["note"])
    t, p = r["t2t"], r["panTro6"]
    stat = (f'T2T cov {t["cov_win_pct"]}% / gaps {t["n_gaps"]} / '
            f'block {t["longest_block"]//1000}kb &nbsp;|&nbsp; '
            f'panTro6 cov {p["cov_win_pct"]}% / gaps {p["n_gaps"]} / '
            f'block {p["longest_block"]//1000}kb')
    rows.append(f'''
    <div class="locus">
      <h3>{iid}</h3>
      <div class="meta">{loc} &middot; {r["size_bp"]:,} bp{dis}</div>
      <div class="note">{note}</div>
      <div class="stat">{stat}</div>
      <div class="pair">
        <figure><figcaption>GRCh38 &times; T2T chimpanzee (mPanTro3 v2.0)</figcaption>{cell(r["inv_id"],"chimpT2T")}</figure>
        <figure><figcaption>GRCh38 &times; panTro6 (previous)</figcaption>{cell(r["inv_id"],"panTro6")}</figure>
      </div>
    </div>''')

doc = f'''<!doctype html><html><head><meta charset="utf-8">
<title>SVbyEye inversion synteny: T2T chimpanzee vs panTro6</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#faf9f7;color:#1a1a1a}}
 header{{padding:18px 24px;background:#fff;border-bottom:1px solid #e5e2dc}}
 header h1{{margin:0 0 4px;font-size:20px}} header p{{margin:0;color:#555;font-size:13px}}
 .wrap{{max-width:1400px;margin:0 auto;padding:16px 24px}}
 .locus{{background:#fff;border:1px solid #e5e2dc;border-radius:8px;padding:14px 16px;margin:14px 0}}
 .locus h3{{margin:0;font-size:15px;font-family:ui-monospace,Menlo,monospace}}
 .meta{{color:#666;font-size:12px;margin:2px 0 6px}}
 .note{{font-size:13px;background:#f3f6f3;border-left:3px solid #2e8b3d;padding:6px 10px;border-radius:3px;margin-bottom:6px}}
 .stat{{font-size:11px;color:#777;font-family:ui-monospace,monospace;margin-bottom:8px}}
 .pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
 figure{{margin:0}} figcaption{{font-size:12px;color:#444;margin-bottom:4px;font-weight:600}}
 img{{width:100%;border:1px solid #eee;border-radius:4px}}
 .missing{{padding:30px;text-align:center;color:#aaa;background:#f6f6f6;border-radius:4px}}
</style></head><body>
<header>
 <h1>SVbyEye inversion synteny plots &mdash; T2T chimpanzee vs panTro6</h1>
 <p>{len(recs)} inversion loci. Left = GRCh38 &times; T2T chimpanzee (mPanTro3 v2.0, GCA_028858775.2, the drop-in replacement). Right = GRCh38 &times; panTro6 (previous). Sorted by how much T2T changes the picture.</p>
</header>
<div class="wrap">{''.join(rows)}</div>
</body></html>'''

open(out_html, "w").write(doc)
print(f"wrote {out_html} with {len(recs)} loci")
