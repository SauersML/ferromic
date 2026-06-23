"""Single source of truth for codon-aware coding-sequence diversity.

Both ``stats/four_fold_pi.py`` (4-fold synonymous π) and ``stats/pin_pis.py``
(πN at 0-fold / πS at 4-fold sites) previously duplicated the PHYLIP reader, the
per-site π estimator, the standard genetic code, the degeneracy classification and
the per-site π loop. The duplication let the two drift apart: four_fold_pi gained a
codon-aware π fix (a haplotype only contributes its third base at a 4-fold site when
its OWN first two codon positions establish a 4-fold family) while pin_pis kept a
plain per-column π that counted bases from haplotypes whose codon family was unknown
(N/gap at codon positions 1-2). This module centralises the shared logic so the two
analyses cannot diverge again, and exposes one codon-aware π routine used by both.

Conventions:
- Sequences are equal-length uppercase strings over A/C/G/T/N/-, in reading frame
  (codon i spans columns 3i, 3i+1, 3i+2).
- A "site" for codon-aware π is a triple (codon_start, pos, cls) where pos in {0,1,2}
  and cls in {"0","4"} (0-fold / 4-fold). The measured column is codon_start+pos.
"""
from __future__ import annotations

import gzip
import itertools
import re
from collections import Counter

import numpy as np

VALID = frozenset("ACGT")

# ------------------------- genetic code / degeneracy -------------------------
# Standard genetic code (NCBI translation table 1).
_BASES = "TCAG"
_AA = (
    "FFLLSSSSYY**CC*W"
    "LLLLPPPPHHQQRRRR"
    "IIIMTTTTNNKKSSRR"
    "VVVVAAAADDEEGGGG"
)
CODON_TABLE = {
    a + b + c: _AA[i]
    for i, (a, b, c) in enumerate(itertools.product(_BASES, repeat=3))
}


def degeneracy(codon, pos):
    """N-fold degeneracy of ``pos`` (0,1,2) in ``codon`` under the standard code.

    Counts how many of the four nucleotides at ``pos`` yield the same amino acid as
    ``codon`` (the codon itself counts). Returns ``None`` for a stop / undefined
    codon, so such codons belong to neither the 0-fold nor the 4-fold set."""
    aa = CODON_TABLE.get(codon)
    if aa is None or aa == "*":
        return None
    n_same = 0
    for b in _BASES:
        if CODON_TABLE.get(codon[:pos] + b + codon[pos + 1:]) == aa:
            n_same += 1
    return n_same


# Precomputed (codon, pos) -> bool for sense codons.
ZEROFOLD = {}
FOURFOLD = {}
for _cod in CODON_TABLE:
    if CODON_TABLE[_cod] == "*":
        continue
    for _p in range(3):
        _d = degeneracy(_cod, _p)
        ZEROFOLD[(_cod, _p)] = (_d == 1)
        FOURFOLD[(_cod, _p)] = (_d == 4)

# The eight 4-fold-degenerate codon prefixes (positions 1-2). A codon's third
# position is 4-fold iff its first two bases form one of these families; this is
# exactly {prefix : FOURFOLD[(prefix+X, 2)] for any X}.
FOURFOLD_PREFIXES = frozenset(
    cod[:2] for cod in CODON_TABLE if FOURFOLD.get((cod, 2))
)

# Codon sets by position/class for O(1) membership in the hot per-site pi loop.
_ZERO_CODONS_BY_POS = (set(), set(), set())
_FOUR_CODONS_BY_POS = (set(), set(), set())
for (_c, _p), _v in ZEROFOLD.items():
    if _v:
        _ZERO_CODONS_BY_POS[_p].add(_c)
for (_c, _p), _v in FOURFOLD.items():
    if _v:
        _FOUR_CODONS_BY_POS[_p].add(_c)


# ------------------------------ PHYLIP I/O ------------------------------
def read_phy(path):
    """Return ``(sequences, length)`` from a (optionally gzipped) PHYLIP alignment.

    Mirrors the sequence extraction in cds/axt_to_phy.py. Returns ``([], length)``
    if the file is malformed or rows are ragged."""
    opener = gzip.open if str(path).endswith(".gz") else open
    seqs = []
    with opener(path, "rt") as fh:
        first = fh.readline().split()
        if len(first) != 2:
            return [], 0
        try:
            exp_len = int(first[1])
        except ValueError:
            return [], 0
        for line in fh:
            if not line.strip():
                continue
            m = re.search(r"([ACGTNacgtn-]+)\s*$", line)
            if m:
                seqs.append(m.group(1).upper())
    if not seqs:
        return [], exp_len
    L = len(seqs[0])
    if any(len(s) != L for s in seqs) or L != exp_len:
        return [], exp_len
    return seqs, L


# ------------------------------ π estimator ------------------------------
def site_pi(column):
    """Per-site π with the Rust pipeline estimator (src/stats.rs).

    ``column`` is an iterable of single-character bases. Returns ``None`` for an
    uncallable site (< 2 called A/C/G/T bases)."""
    counts = Counter(b for b in column if b in VALID)
    n = sum(counts.values())
    if n < 2:
        return None
    sum_sq = sum(c * c for c in counts.values())
    return n / (n - 1.0) * (1.0 - sum_sq / (n * n))


def locus_pi(seqs, columns):
    """Mean per-site π over ``columns`` (callable sites only). For whole-CDS /
    whole-locus π where every called base at a column is counted."""
    vals = []
    for col in columns:
        p = site_pi(s[col] for s in seqs)
        if p is not None:
            vals.append(p)
    if not vals:
        return np.nan, 0
    return float(np.mean(vals)), len(vals)


# ----------------------- codon-site classification -----------------------
def fourfold_codon_starts(seqs, L):
    """Yield the start index of each codon that is 4-fold-degenerate at position 3.

    A codon is included only when every *called* haplotype (ACGT at positions 1-2)
    carries a 4-fold prefix; haplotypes with a gap/N at positions 1-2 are ignored
    here (family unknown) and are also excluded from this codon's π by
    :func:`class_aware_locus_pi`."""
    for cs in range(0, L - 2, 3):
        ok = False
        for s in seqs:
            p1, p2 = s[cs], s[cs + 1]
            if p1 in VALID and p2 in VALID:
                if (p1 + p2) not in FOURFOLD_PREFIXES:
                    ok = False
                    break
                ok = True
        if ok:
            yield cs


def classify_zero_four(seqs, L):
    """Classify each codon position as 0-fold / 4-fold across the sample.

    Yields ``(column_index, "0" | "4")``. A position is reported only when every
    fully-called (ACGT) codon agrees on the class; codons with gap/N/stop in a
    haplotype don't vote, and a disagreement skips the position. Classification is
    done once on the combined sample so it is orientation-independent."""
    for cs in range(0, L - 2, 3):
        for pos in range(3):
            col = cs + pos
            is_zero = is_four = True
            seen = False
            for s in seqs:
                codon = s[cs:cs + 3]
                if any(b not in VALID for b in codon):
                    continue
                key = (codon, pos)
                if key not in ZEROFOLD:  # stop codon
                    is_zero = is_four = False
                    break
                seen = True
                if not ZEROFOLD[key]:
                    is_zero = False
                if not FOURFOLD[key]:
                    is_four = False
                if not is_zero and not is_four:
                    break
            if not seen:
                continue
            if is_zero:
                yield col, "0"
            elif is_four:
                yield col, "4"


def class_aware_locus_pi(seqs, sites):
    """Mean per-site π at degenerate sites, counting each haplotype's base ONLY when
    its own codon establishes the required degeneracy class.

    ``sites`` is an iterable of ``(codon_start, pos, cls)`` with ``cls`` in
    ``{"0","4"}``. For each site the measured column is ``codon_start + pos``; a
    haplotype contributes its base there only if its full codon (the three columns
    from ``codon_start``) is ACGT and is ``cls``-fold at ``pos``. This is the
    codon-aware rule from four_fold_pi, generalised to 0-fold sites for pin_pis, so
    a haplotype with N/gap elsewhere in the codon (unknown family) never contributes.

    Returns ``(mean_pi, n_callable_sites)``."""
    # Group sites by codon_start so each haplotype's codon is sliced once per codon
    # (not once per position), and use precomputed codon-class sets for membership.
    by_codon = {}
    for cs, pos, cls in sites:
        by_codon.setdefault(cs, []).append((pos, cls))

    vals = []
    for cs, posclasses in by_codon.items():
        codons = [s[cs:cs + 3] for s in seqs]
        for pos, cls in posclasses:
            want = _FOUR_CODONS_BY_POS[pos] if cls == "4" else _ZERO_CODONS_BY_POS[pos]
            col = cs + pos
            bases = [s[col] for s, codon in zip(seqs, codons) if codon in want]
            p = site_pi(bases)
            if p is not None:
                vals.append(p)
    if not vals:
        return np.nan, 0
    return float(np.mean(vals)), len(vals)
