"""Assemble the per-inversion SVbyEye figures into a supplementary PDF.

The revision states that chimpanzee-versus-GRCh38 orientation was determined by
manual review of SVbyEye alignment plots. The review used one figure per locus;
this script collects every one of those figures into a single supplementary
document so a reader can check each call, and pulls two of them out as worked
examples for the response letter.

One page per inversion, in genomic order, carrying:
  * chimp    -- chimpanzee (panTro6) aligned across the locus. This is the panel
                the orientation call was read from.
  * miro_id  -- SVbyEye miropeat on a curated set of human haplotypes.
  * track    -- one row per assembly haplotype, coloured by strand vs GRCh38.
  * invhap   -- inverted-haplotype view.
  * grad     -- directional colour gradient across the locus.

Each page header records the inversion ID, coordinates, size, recurrence class,
inverted allele frequency, and the recorded chimp orientation call, so the
figure and the call cannot drift apart.

How to read the chimp panel: orientation is judged RELATIVE TO THE FLANKS
inside the inversion boundaries. A uniformly reverse-strand ribbon across the
locus AND its flanks means the chimpanzee contig itself is in reverse
orientation, not that the locus is inverted in chimpanzee.

Inputs:  web/figures-site/public/inversions/img/*.webp
         data/chimp_alignment_responses.json
         data/inv_properties.tsv
Outputs: data/svbyeye_all_inversions.pdf
         data/svbyeye_examples.pdf / .png
"""

import io
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

_STATS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_STATS)
_DATA = os.path.join(_REPO, "data")
IMG_DIR = os.path.join(_REPO, "web", "figures-site", "public", "inversions", "img")

RESPONSES = os.path.join(_DATA, "chimp_alignment_responses.json")
INVPROPS = os.path.join(_DATA, "inv_properties.tsv")
OUT_PDF = os.path.join(_DATA, "svbyeye_all_inversions.pdf")
OUT_CHIMP_PDF = os.path.join(_DATA, "svbyeye_chimp_orientation.pdf")
OUT_EX_PDF = os.path.join(_DATA, "svbyeye_examples.pdf")
OUT_EX_PNG = os.path.join(_DATA, "svbyeye_examples.png")

# (view suffix, caption, max rendered height). The track and gradient panels
# carry one row per assembly haplotype and are 2048 px tall at source; capping
# them keeps the whole-gallery PDF near 50 MB instead of 180 MB while staying
# legible on screen.
VIEWS = [
    ("chimp", "Chimpanzee (panTro6) vs GRCh38 - the panel the call is read from",
     None),
    ("miro_id", "SVbyEye miropeat, curated human haplotypes", 700),
    ("track", "Per-haplotype orientation tracks (green = same strand as GRCh38)",
     900),
    ("invhap", "Inverted-haplotype view", None),
    ("grad", "Directional gradient across the locus", 900),
]
CHIMP_ONLY = [VIEWS[0]]
MAX_W = 850           # downsample wide panels; keeps the PDF a sane size
JPEG_QUALITY = 68     # pages are embedded as JPEG streams, not raw bitmaps
CALL_LABEL = {"direct": "GRCh38 = ancestral (direct vs chimp)",
              "inverted": "GRCh38 = derived (inverted vs chimp)",
              "na": "not callable"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_image(inv_id, view):
    path = os.path.join(IMG_DIR, f"{inv_id}.{view}.webp")
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert("RGB")
    if im.width > MAX_W:
        h = int(im.height * MAX_W / im.width)
        im = im.resize((MAX_W, h), Image.LANCZOS)
    return np.asarray(im)


def sort_key(rec):
    c = rec["chrom"].replace("chr", "")
    order = {"X": 23, "Y": 24}
    return (order.get(c, int(c) if c.isdigit() else 99), rec["start"])


def page_image(rec, meta, views=VIEWS):
    """Compose one supplementary page as a single RGB image.

    Built with PIL rather than matplotlib: 292 pages of raster panels through a
    figure canvas exhausts memory, while stacking the panels directly keeps the
    footprint to one page at a time.
    """
    inv_id = rec["inv_id"]
    panels = []
    for v, lbl, max_h in views:
        path = os.path.join(IMG_DIR, f"{inv_id}.{v}.webp")
        if not os.path.exists(path):
            continue
        im = Image.open(path).convert("RGB")
        if im.width != MAX_W:
            h = max(1, int(im.height * MAX_W / im.width))
            im = im.resize((MAX_W, h), Image.LANCZOS)
        if max_h and im.height > max_h:
            w = max(1, int(im.width * max_h / im.height))
            im = im.resize((w, max_h), Image.LANCZOS)
        panels.append((lbl, im))
    if not panels:
        return None

    call = rec.get("classification", "na")
    header = [
        f"{inv_id}    {rec['region']}    {rec['size_bp']:,} bp",
        f"recurrence: {meta.get('recurrence', 'unclassified')}    "
        f"inverted AF: {meta.get('af', 'NA')}    "
        f"chimp call: {CALL_LABEL.get(call, call)}"
        + ("    [one of the 93 analysed loci]" if meta.get("in93") else ""),
    ]
    pad, head_h, lbl_h = 12, 46, 16
    total_h = head_h + sum(im.height + lbl_h + pad for _, im in panels) + pad
    page = Image.new("RGB", (MAX_W + 2 * pad, total_h), "white")
    draw = ImageDraw.Draw(page)
    font = _font(15)
    small = _font(11)
    draw.text((pad, 6), header[0], fill="black", font=font)
    draw.text((pad, 26), header[1], fill="#444444", font=small)
    y = head_h
    for lbl, im in panels:
        draw.text((pad, y), lbl, fill="#666666", font=small)
        y += lbl_h
        page.paste(im, (pad, y))
        y += im.height + pad
        im.close()
    # Round-trip through JPEG so Pillow embeds a DCTDecode stream in the PDF.
    # Pasting the raw bitmaps instead produced a 247 MB file for 292 pages.
    buf = io.BytesIO()
    page.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
    page.close()
    buf.seek(0)
    return Image.open(buf)


_FONT_CACHE = {}


def _font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for cand in ("/System/Library/Fonts/Helvetica.ttc",
                 "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(cand):
            try:
                _FONT_CACHE[size] = ImageFont.truetype(cand, size)
                return _FONT_CACHE[size]
            except OSError:
                pass
    _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def main():
    if not os.path.isdir(IMG_DIR):
        sys.exit(f"figure directory not found: {IMG_DIR}")
    recs = json.load(open(RESPONSES))["responses"]
    print(f"{len(recs)} reviewed loci")

    props = pd.read_csv(INVPROPS, sep="\t")
    meta = {}
    for _, r in props.iterrows():
        cons = r.get("0_single_1_recur_consensus")
        lab = {0: "single-event", 1: "recurrent"}.get(
            int(cons) if pd.notna(cons) and str(cons).strip() != "" else -1,
            "no consensus call")
        meta[str(r.get("OrigID"))] = dict(
            recurrence=lab,
            af=r.get("Inverted_AF"),
            in93=lab in ("single-event", "recurrent"),
        )

    recs = sorted(recs, key=sort_key)
    counts = {}
    for r in recs:
        counts[r.get("classification", "na")] = \
            counts.get(r.get("classification", "na"), 0) + 1
    print("orientation calls:", counts)

    def build(path, views, title):
        state = {"n": 0}

        def pages():
            for rec in recs:
                im = page_image(rec, meta.get(rec["inv_id"], {}), views)
                if im is None:
                    continue
                state["n"] += 1
                if state["n"] % 50 == 0:
                    print(f"  {state['n']} pages", flush=True)
                yield im

        it = pages()
        first = next(it, None)
        if first is None:
            sys.exit("no figures found")
        first.save(path, "PDF", save_all=True, append_images=it,
                   resolution=150.0, title=title)
        print(f"Wrote {path}: {state['n']} pages, "
              f"{os.path.getsize(path) / 1e6:.1f} MB")

    # Only the chimpanzee-vs-GRCh38 panel is built. The other views (miropeat
    # and per-haplotype strand tracks against GRCh38) are a different analysis
    # and are not what the orientation calls were read from, so shipping them
    # in this supplement would misrepresent what was reviewed.
    build(OUT_CHIMP_PDF, CHIMP_ONLY,
          "Chimpanzee-vs-GRCh38 alignment for every inversion locus")

    # ---- two worked examples for the response letter ---------------------
    by_call = {}
    for r in recs:
        by_call.setdefault(r.get("classification", "na"), []).append(r)
    ex = []
    for call in ("direct", "inverted"):
        pool = [r for r in by_call.get(call, [])
                if meta.get(r["inv_id"], {}).get("in93")]
        pool = sorted(pool, key=lambda r: -r["size_bp"])
        if pool:
            ex.append(pool[0])
    if len(ex) == 2:
        fig, axes = plt.subplots(2, 1, figsize=(9, 6.4))
        for ax, rec in zip(axes, ex):
            arr = load_image(rec["inv_id"], "chimp")
            ax.imshow(arr)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            m = meta.get(rec["inv_id"], {})
            ax.set_title(
                f"{rec['inv_id']}  {rec['region']}  ({m.get('recurrence', '')})"
                f"\n{CALL_LABEL[rec['classification']]}", fontsize=9)
        fig.suptitle("Chimpanzee alignment examples used for orientation "
                     "polarization", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, .96])
        fig.savefig(OUT_EX_PDF)
        fig.savefig(OUT_EX_PNG, dpi=200)
        plt.close(fig)
        print(f"Wrote {OUT_EX_PDF} and {OUT_EX_PNG}: "
              f"{ex[0]['inv_id']} (direct), {ex[1]['inv_id']} (inverted)")


if __name__ == "__main__":
    main()
