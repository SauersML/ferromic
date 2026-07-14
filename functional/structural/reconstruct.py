"""Haplotype reconstruction for the structure-vs-SNV in-silico decomposition.

Builds four constructed sequences per breakpoint window from REAL phased data:
  ref-direct    : GRCh38 reference (direct orientation)
  ref-inverted  : reference with the in-window inversion segment reverse-complemented
  full-inverted : ref-inverted + real phased INVERTED-background consensus SNVs
  full-direct   : ref-direct   + real phased DIRECT-background consensus SNVs

Phased SNVs come from the 1000G high-coverage GRCh38 3202-sample panel (region-streamed).
Backgrounds are defined by the experiment-#8 tag SNP, with the inverted allele determined
by concordance against the Porubsky phased orientation calls (tag-validation QC).

Deterministic; no randomness. All coordinates 0-based half-open internally.
"""
import functools
import os

import numpy as np
import py2bit
import pysam

try:  # package import (functional.structural.reconstruct)
    from .. import paths as _paths
except ImportError:  # run as a loose script
    import paths as _paths  # type: ignore

_DEFAULT_PANEL_DIR = ("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
                      "1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/")


def ref_2bit_path():
    """Resolve the GRCh38 2bit via functional.paths (flag/env/data-root); no hard-coded path."""
    return _paths.resolve("reference_2bit")


def panel_dir():
    """1000G panel directory/URL prefix. FUNCTIONAL_THOUSAND_GENOMES_PANEL_DIR overrides the
    default EBI FTP URL with a local mirror; region-streamed either way."""
    return os.environ.get("FUNCTIONAL_THOUSAND_GENOMES_PANEL_DIR", _DEFAULT_PANEL_DIR)


def panel_url(chrom):
    # chrX uses the .v2 panel filename; autosomes do not
    base = panel_dir()
    if not base.endswith("/"):
        base += "/"
    suffix = "filtered.SNV_INDEL_SV_phased_panel.v2.vcf.gz" if chrom == "chrX" \
        else "filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
    return f"{base}1kGP_high_coverage_Illumina.{chrom}.{suffix}"
AG_LENGTHS = [16384, 131072, 524288, 1048576]

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def load_ref(chrom, start, end):
    """Return uppercase reference sequence for [start,end) (0-based half-open)."""
    tb = py2bit.open(ref_2bit_path())
    try:
        seq = tb.sequence(chrom, start, end).upper()
    finally:
        tb.close()
    return seq


@functools.lru_cache(maxsize=None)
def _open_panel(chrom):
    return pysam.VariantFile(panel_url(chrom))


def fetch_snvs(chrom, start, end):
    """Fetch biallelic SNV records in [start,end). Returns (samples, list of dicts).
    Each dict: pos(0-based), ref, alt, gt (np.int8 array of length 2*nsamples, hap-major:
    sample i occupies [2i, 2i+1]; -1 for missing)."""
    vf = _open_panel(chrom)
    samples = list(vf.header.samples)
    recs = []
    for r in vf.fetch(chrom, start, end):
        if r.alts is None or len(r.alts) != 1:
            continue
        alt = r.alts[0]
        if len(r.ref) != 1 or len(alt) != 1:
            continue  # SNVs only
        if r.ref.upper() not in "ACGT" or alt.upper() not in "ACGT":
            continue
        gt = np.full(2 * len(samples), -1, dtype=np.int8)
        for i, s in enumerate(samples):
            a = r.samples[s]["GT"]
            if a[0] is not None:
                gt[2 * i] = a[0]
            if len(a) > 1 and a[1] is not None:
                gt[2 * i + 1] = a[1]
        recs.append({"pos": r.pos - 1, "ref": r.ref.upper(), "alt": alt.upper(), "gt": gt})
    return samples, recs


def get_tag_record(chrom, pos1):
    """Fetch the tag SNP record (pos1 = 1-based). Returns (samples, gt array) or None."""
    vf = _open_panel(chrom)
    samples = list(vf.header.samples)
    for r in vf.fetch(chrom, pos1 - 1, pos1 + 1):
        if r.pos == pos1:
            gt = np.full(2 * len(samples), -1, dtype=np.int8)
            for i, s in enumerate(samples):
                a = r.samples[s]["GT"]
                if a[0] is not None:
                    gt[2 * i] = a[0]
                if len(a) > 1 and a[1] is not None:
                    gt[2 * i + 1] = a[1]
            return samples, gt, (r.ref.upper(), r.alts[0].upper() if r.alts else None)
    return None


def classify_backgrounds(samples, tag_gt, porubsky_hom_inv, porubsky_hom_dir):
    """Determine which tag allele marks the inverted background by concordance with
    Porubsky homozygous calls, then label every haplotype.

    Returns dict with:
      inv_allele (0 or 1), concordance stats, hap_inv/hap_dir boolean arrays (len 2*nsamples).
    """
    sidx = {s: i for i, s in enumerate(samples)}
    # mean alt dosage among Porubsky homs present in panel
    def mean_alt(sample_list):
        vals = []
        for s in sample_list:
            if s in sidx:
                i = sidx[s]
                g = tag_gt[[2 * i, 2 * i + 1]]
                g = g[g >= 0]
                if len(g):
                    vals.append(g.mean())
        return (np.mean(vals) if vals else np.nan), len(vals)
    inv_alt, n_inv = mean_alt(porubsky_hom_inv)
    dir_alt, n_dir = mean_alt(porubsky_hom_dir)
    # inverted allele = the tag allele more frequent on the inverted background
    inv_allele = 1 if (np.nan_to_num(inv_alt, nan=0) >= np.nan_to_num(dir_alt, nan=1)) else 0
    # per-haplotype concordance: inverted homs' haplotypes should carry inv_allele
    def hap_concord(sample_list, expect_inv):
        ok = tot = 0
        for s in sample_list:
            if s in sidx:
                i = sidx[s]
                for h in (tag_gt[2 * i], tag_gt[2 * i + 1]):
                    if h < 0:
                        continue
                    tot += 1
                    is_inv = (h == inv_allele)
                    ok += int(is_inv == expect_inv)
        return ok, tot
    ok_i, tot_i = hap_concord(porubsky_hom_inv, True)
    ok_d, tot_d = hap_concord(porubsky_hom_dir, False)
    conc = (ok_i + ok_d) / (tot_i + tot_d) if (tot_i + tot_d) else np.nan
    hap_inv = (tag_gt == inv_allele)
    hap_dir = (tag_gt == (1 - inv_allele))
    return {
        "inv_allele": int(inv_allele),
        "inv_bg_mean_alt_dosage": float(np.nan_to_num(inv_alt)),
        "dir_bg_mean_alt_dosage": float(np.nan_to_num(dir_alt)),
        "n_porubsky_hom_inv_in_panel": int(n_inv),
        "n_porubsky_hom_dir_in_panel": int(n_dir),
        "tag_porubsky_concordance": float(conc),
        "n_hap_inverted": int(hap_inv.sum()),
        "n_hap_direct": int(hap_dir.sum()),
        "hap_inv": hap_inv,
        "hap_dir": hap_dir,
    }


def build_consensus(recs, hap_inv, hap_dir, thresh=0.5):
    """Per-site alt frequency on each background; consensus variant when alt-freq >= thresh.
    Returns (inv_consensus, dir_consensus, site_table).
    *_consensus: list of (pos, ref, consensus_base). site_table: list of dicts for QC."""
    inv_cons, dir_cons, table = [], [], []
    for rec in recs:
        gt = rec["gt"]
        gi = gt[hap_inv]; gi = gi[gi >= 0]
        gd = gt[hap_dir]; gd = gd[gd >= 0]
        fi = gi.mean() if len(gi) else 0.0
        fd = gd.mean() if len(gd) else 0.0
        inv_base = rec["alt"] if fi >= thresh else rec["ref"]
        dir_base = rec["alt"] if fd >= thresh else rec["ref"]
        if inv_base != rec["ref"]:
            inv_cons.append((rec["pos"], rec["ref"], inv_base))
        if dir_base != rec["ref"]:
            dir_cons.append((rec["pos"], rec["ref"], dir_base))
        table.append({"pos": rec["pos"], "ref": rec["ref"], "alt": rec["alt"],
                      "af_inv": float(fi), "af_dir": float(fd),
                      "n_inv": int(len(gi)), "n_dir": int(len(gd)),
                      "differential": abs(fi - fd)})
    return inv_cons, dir_cons, table


def _apply_variants(seq_list, win_start, variants, ref_check):
    """Apply (pos,ref,base) variants to seq_list in reference coords. Returns n_applied, n_refmismatch."""
    applied = mism = 0
    L = len(seq_list)
    for pos, ref, base in variants:
        idx = pos - win_start
        if 0 <= idx < L:
            if seq_list[idx] != ref:
                mism += 1
                continue
            seq_list[idx] = base
            applied += 1
    return applied, mism


def construct_sequences(chrom, win_start, win_end, b1, b2, inv_consensus, dir_consensus):
    """Build the four sequences for window [win_start,win_end). The inverted segment within
    the window is [max(b1,win_start), min(b2,win_end)). Returns dict of sequences + QC."""
    ref = load_ref(chrom, win_start, win_end)
    L = len(ref)
    i1 = max(0, b1 - win_start)
    i2 = min(L, b2 - win_start)
    assert i1 < i2, f"inversion segment not in window: i1={i1} i2={i2}"

    def flip(seq_list):
        seg = "".join(seq_list[i1:i2])
        seq_list[i1:i2] = list(revcomp(seg))
        return seq_list

    ref_direct = list(ref)
    ref_inverted = flip(list(ref))

    full_dir_list = list(ref)
    n_app_d, n_mis_d = _apply_variants(full_dir_list, win_start, dir_consensus, ref)
    full_direct = full_dir_list  # direct orientation

    full_inv_list = list(ref)
    n_app_i, n_mis_i = _apply_variants(full_inv_list, win_start, inv_consensus, ref)
    full_inv_list = flip(full_inv_list)
    full_inverted = full_inv_list

    seqs = {
        "ref_direct": "".join(ref_direct),
        "ref_inverted": "".join(ref_inverted),
        "full_direct": "".join(full_direct),
        "full_inverted": "".join(full_inverted),
    }
    # QC
    rc_roundtrip = (revcomp(revcomp(ref)) == ref)
    qc = {
        "win_start": win_start, "win_end": win_end, "L": L,
        "seg_i1": int(i1), "seg_i2": int(i2), "seg_len": int(i2 - i1),
        "n_inv_variants_applied": int(n_app_i), "n_inv_variants_refmismatch": int(n_mis_i),
        "n_dir_variants_applied": int(n_app_d), "n_dir_variants_refmismatch": int(n_mis_d),
        "rc_roundtrip_ok": bool(rc_roundtrip),
        "len_ok": bool(L in AG_LENGTHS),
        # structural change only affects the flipped segment: ref_direct vs ref_inverted differ
        # only within [i1,i2)
        "structural_diff_confined": bool(
            ref_direct[:i1] == ref_inverted[:i1] and ref_direct[i2:] == ref_inverted[i2:]),
    }
    return seqs, qc


def choose_windows(start, end, size, L=131072):
    """Return list of (win_start, win_end, label). Small inversions -> single centered window;
    large -> per-breakpoint windows. L must be in AG_LENGTHS."""
    assert L in AG_LENGTHS
    inv_span = end - start
    if inv_span + 20000 <= L:  # fits with >=10kb flank each side
        mid = (start + end) // 2
        ws = mid - L // 2
        return [(ws, ws + L, "whole")]
    else:
        wsL = start - L // 2
        wsR = end - L // 2
        return [(wsL, wsL + L, "bpL"), (wsR, wsR + L, "bpR")]
