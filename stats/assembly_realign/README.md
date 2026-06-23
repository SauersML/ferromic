# Assembly-realignment polarization experiment (MSI) — NEGATIVE RESULT

A full assembly-resolved attempt to polarize the ~212 `provisional`/`unresolved`
inversion loci by directly realigning all **12 T2T ape haplotype assemblies** (2025
Complete Ape Genomes: chimp, bonobo, gorilla, both orangutans, siamang) to the human
reference, run on MSI compute (`acn*` nodes, Slurm).

## Pipeline
1. `build_regions.py` — extract each inversion locus ±150 kb from hg38 as a query region.
2. `align.sbatch` — `minimap2 -x asm20 -c --secondary=yes -N 20` of each ape assembly
   (reference) vs the hg38 regions (query), one Slurm array task per haplotype.
3. `paf_polarize.py` — per haplotype, detect the inversion signature (locus interior
   aligns on the opposite strand vs the flanking backbone, same contig); diploid HOM/HET
   per species; depth-weighted parsimony (siamang deepest) → flip. `realign_calls.tsv`.

## Why it is NOT integrated
Validated against the trusted references, the calls are no better than chance:

| reference (trusted) | agreement |
|---|---|
| published consortium SYRI/PAV (`gold_t2t_apes`) | **44%** (7/16) |
| Strand-seq gold | **55%** (26/47) |
| even single-event + unanimous-ape + high-conf | **50%** |

It gets **17q21.31 wrong**. Root cause (confirmed by inspecting the raw PAF): the hard
loci sit in **segmental duplications** where each ape contig aligns to 5+ paralogous
positions in *both* orientations, so a strand-orientation heuristic cannot recover the
true ancestral orientation — the same paralog confound that defeats chain synteny.

## Conclusion
This is the definitive, maximal-effort test: even with the complete T2T ape assemblies and
real compute, custom orientation calling cannot polarize these SD-rich loci. The published
consortium SYRI/PAV pipeline (synteny-block-aware, 88% vs Strand-seq) is the reliable
assembly tier — and it produced **no inversion call** at these loci precisely because its
own sophisticated SV calling could not resolve them. These loci are genuinely unresolvable
by ape-assembly comparison; resolving them requires direct orientation measurement
(targeted Strand-seq) of the specific loci. The source of truth therefore stays the
validated `gold_t2t_apes`/`gold_strandseq`/`gold_deepOG` tiers (144 resolved) + honest
quarantine; `realign_calls.tsv` is retained only as the experiment's record.
