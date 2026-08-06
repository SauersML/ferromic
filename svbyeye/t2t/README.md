# svbyeye/t2t — inversion synteny plots against a T2T ape genome

Regenerates the SVbyEye inversion synteny plots against the **T2T chimpanzee**
assembly (NHGRI_mPanTro3-v2.0_pri, GenBank `GCA_028858775.2`), replacing the
fragmented **panTro6** chimp reference used by the parent `svbyeye/` pipeline.

## Why

panTro6 is gappy and fragmented in segmental-duplication (SD)-rich regions —
exactly where inversions and their breakpoints sit. The complete, contiguous T2T
chimpanzee assembly gives accurate ancestral-outgroup synteny at those features.
This is an accuracy upgrade, not a cosmetic reference swap: see the diff below.

## What it produces

For each inversion in `inv_properties.tsv`, an SVbyEye `plotMiro` miropeat of
**GRCh38 (target) × T2T chimpanzee (query)** across the inversion window, with the
inversion interval annotated. The chimp is aligned to GRCh38 with the *same*
minimap2 parameters as the parent pipeline (`-x asm20 -c --eqx --secondary=no
-s 25000 -K 1G`). panTro6 is aligned identically to produce a per-locus
**T2T-vs-panTro6 diff** (`results/diff_chimp.tsv`).

## Diff summary (292 loci, this callset)

| metric | result |
|---|---|
| loci where panTro6 had NO alignment but T2T covers | **9** |
| loci with >1 pp better window coverage under T2T | **83** |
| loci with >1 pp **worse** window coverage under T2T (SD multi-mapping) | **26** |
| loci with fewer alignment gaps under T2T | **57** |
| empty-alignment loci | 27 (panTro6) → 21 (T2T) |
| loci inverted in the chimp outgroup (rev frac >0.6) | 79 |

The clearest cases are the SD-rich chr16 large inversions. `chr16-15384483-INV-891665`
(891 kb) fragments across **7 panTro6 sequences** — `panTro6_chr16` plus six
unplaced `chrUn_NW_*` scaffolds — but maps to just **two contiguous T2T
chromosomes**. Neighboring chr16 inversions gain +9.6 to +16.3 pp window coverage,
halve their gaps, and shrink their largest gap (e.g. 278 kb → 108 kb). See
`examples/` for representative side-by-side comparison pairs.

## Run order

```bash
sbatch bin/build_env.sbatch        # micromamba: minimap2 2.30, samtools, R + SVbyEye
sbatch bin/download_genomes.sbatch # pinned T2T chimp + panTro6 + GRCh38 (md5-checked)
sbatch bin/prep.sbatch             # hg38 index + inversion table + relabel chimp FASTAs
sbatch bin/smoke.sbatch            # exact-entrypoint smoke on chr21+chr22
sbatch --array=1-2 bin/align_chimp.sbatch   # T2T + panTro6 -> hg38 (matched params)
sbatch --array=1-N bin/plot_all.sbatch      # per-locus miropeats (N = #inversions)
sbatch bin/finalize.sbatch                  # diff table + HTML gallery
```

Paths in the `.sbatch` files use an artifacts root env var; adjust `ART` for your
host. All assemblies + parameters are pinned in `config.yaml`.

## Swapping the ape

Edit the `ape_reference` block in `config.yaml` to any other primate-T2T assembly
(bonobo mPanPan1, gorilla mGorGor1, orangutan) — the rest of the pipeline is
reference-agnostic. A multi-ape panel is `align_chimp.sbatch` array tasks per ape.

## Notes / limits

- Alignment in near-identical SD sequence can still multi-map; `--secondary=no`
  keeps one alignment per query region. Orientation is read from the primary
  alignments and reported per plot (`results/diff_chimp.tsv`).
- The parent `svbyeye/` pipeline's multi-haplotype human population tracks
  (`visualize_inv.R` views 2/3) are independent of the chimp reference and are
  unchanged by this swap; this module covers the chimp/ancestral synteny view.
- Full regenerated plot set (292 T2T + 292 panTro6) + browsable gallery are
  produced by `finalize.sbatch`; only representative examples are committed here
  (the parent module keeps figures on the external figures site).
