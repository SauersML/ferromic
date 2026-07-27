"""Orientation origin count, scored by the reference recurrence classifier.

This is the classifier of ``hsiehphLab/inversionSimulation`` -- the pipeline
behind the manuscript's Fig. 1G -- applied to a haplotype x biallelic-site
matrix: build a nucleotide alignment on the upstream reference backbone, infer
an IQ-TREE maximum-likelihood tree rooted on the ancestral outgroup, collapse
the outgroup, and count the minimum number of orientation state changes with
Biopython's Fitch ``ParsimonyScorer``. ``classify`` returns that count;
``count >= 2`` is the recurrent call.

The pipeline steps themselves live in ``simulations/refsim/refsim.py``, which
documents the file-by-file correspondence with the upstream repository. This
module is the entry point for callers that hold genotypes rather than a
simulated tree sequence (``features.extract_features``, the reproduction tests).

Requirements: Biopython and an ``iqtree2`` binary on ``PATH`` (or ``IQTREE_BIN``).
Simulated loci are normally scored inside ``simulations/refsim/run_grid.py``,
which goes through the full VCF-equivalent path with the real nucleotide states;
``classify`` reconstructs the equivalent two-state alignment from ``G``.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile

import numpy as np

from . import paths

# Two-state encoding of the genotype matrix: column-wise ancestral base taken
# from the upstream backbone, derived base a fixed transversion partner. The
# parsimony score depends on the tree topology and the orientation trait only,
# so the choice of derived base does not enter the returned count; it enters
# only through the ML tree, where a consistent 2-state coding is what the
# genotype matrix actually carries.
_DERIVED = {ord("A"): ord("T"), ord("T"): ord("A"),
            ord("C"): ord("G"), ord("G"): ord("C"),
            ord("N"): ord("T")}

_refsim = None


def _load_refsim():
    """Import ``simulations/refsim/refsim.py`` by path (simulations/ is not a package)."""
    global _refsim
    if _refsim is None:
        fp = os.path.join(paths.REPO_ROOT, "simulations", "refsim", "refsim.py")
        spec = importlib.util.spec_from_file_location("ferromic_refsim", fp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _refsim = mod
    return _refsim


def build_alignment(path, G, labels, backbone, refsim):
    """Write the reference-style alignment for a 0/1 genotype matrix.

    One sequence per haplotype on the upstream backbone, with the derived allele
    substituted where ``G == 1``, preceded by the all-ancestral outgroup
    sequence -- the same alignment shape ``phasedVCF2Fasta.py`` produces.
    """
    G = np.asarray(G, dtype=np.uint8)
    n_hap, n_site = G.shape
    if n_site > len(backbone):
        raise ValueError(f"{n_site} sites exceed the {len(backbone)} bp backbone")
    # Spread the sites evenly over the backbone so no two share a column.
    cols = np.linspace(0, len(backbone) - 1, num=n_site, dtype=int) if n_site else []

    anc = bytearray(backbone)
    seqs = [bytearray(backbone) for _ in range(n_hap)]
    for j, col in enumerate(cols):
        derived = _DERIVED.get(anc[col], ord("T"))
        for h in range(n_hap):
            if G[h, j]:
                seqs[h][col] = derived

    names = ["hap%04d" % h for h in range(n_hap)]
    with open(path, "wb") as fh:
        fh.write(b">" + refsim.OUTGROUP.encode() + b"\n" + bytes(anc) + b"\n")
        for name, seq in zip(names, seqs):
            fh.write(b">" + name.encode() + b"\n" + bytes(seq) + b"\n")
    mapping = [(n, "TRUE" if int(l) == 1 else "FALSE", n)
               for n, l in zip(names, labels)]
    return mapping


def classify(G, labels, seed=1, workdir=None):
    """Inferred number of orientation origin events (``minMutHomoplasy``).

    ``G``: (n_hap, n_site) 0/1 haplotype x biallelic-site matrix.
    ``labels``: (n_hap,) orientation in {0 = direct, 1 = inverted}.
    Returns the Fitch minimum number of orientation state changes on the
    IQ-TREE ML tree; ``>= 2`` is the recurrent call.

    Order-invariant: IQ-TREE's search depends on the order of the sequences in
    the alignment, so rows are first put in a canonical order (lexicographic by
    genotype, then by orientation). Permuting the input therefore cannot change
    the returned count. Upstream has a fixed sample order and needs no such step.
    """
    G = np.asarray(G, dtype=np.uint8)
    if G.shape[1] == 0:
        # No segregating sites: no tree is identifiable and the orientation
        # trait can always be explained by a single change.
        return 1

    labels = np.asarray(labels)
    order = sorted(range(G.shape[0]),
                   key=lambda i: (tuple(int(x) for x in G[i]), int(labels[i])))
    G = G[order]
    labels = labels[order]

    refsim = _load_refsim()
    backbone = refsim.load_reference()
    tmp = workdir or tempfile.mkdtemp(prefix="parsimony_")
    try:
        os.makedirs(tmp, exist_ok=True)
        aln = os.path.join(tmp, "locus.fa")
        mapping = build_alignment(aln, G, labels, backbone, refsim)
        treefile = refsim.run_iqtree(aln, os.path.join(tmp, "locus"), seed=seed)
        return int(refsim.min_mutations(treefile, mapping))
    finally:
        if workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)
