# `functional/splice/` — gene-localised AlphaGenome splice disruption

Predict which gene's splicing an inversion disrupts, and validate the prediction against
measured GTEx sQTLs. This is the **gene-localised** splice formulation (per-gene splice-site
disruption, top gene per inversion) — the formulation that beat the breakpoint baseline and
validated against measured splicing — not a coarse window-aggregate composite.

## Modules

- **`score_alphagenome.py`** — the API driver. Each inversion breakpoint is modelled as a
  reverse-complement substitution in a window centred on the breakpoint, scored with
  AlphaGenome's `RNA_SEQ` and `SPLICE_SITE_USAGE` recommended variant scorers. Produces, per
  inversion, a gene×track signed expression LFC matrix and a per-gene splice-site-usage
  disruption (`splice_abs`), saved as one `.npz` per region. Requires `alphagenome` +
  `ALPHAGENOME_API_KEY` and network access.
- **`formulations.py`** — the validated formulation, recomputed from the cached `.npz` (no API):
  per gene, splice disruption = `splice_abs`; the inversion's **top-splice gene** = the gene with
  the largest disruption, `ag_max_splice` = that value.
- **`validate_gtex.py`** — is the predicted top-splice gene actually a *measured* GTEx sGene? The
  sGene hit-rate over checkable loci is the validation readout.

## Run

```bash
# Recompute the validated formulation from cached AlphaGenome scores (no API)
python -m functional.splice.cli formulate \
    --npz-dir $FUNCTIONAL_ALPHAGENOME_SCORES \
    --out functional/results/splice/per_inversion_splice.tsv

# (upstream) score inversions via the AlphaGenome API -> per-inversion .npz
ALPHAGENOME_API_KEY=... python -m functional.splice.cli score \
    --inversions functional/data/inversions.tsv --out-dir <agscore_dir>
```

## Outputs

- `../results/splice/per_inversion_splice.tsv` — per inversion: number of AlphaGenome-scored
  genes, top-splice gene, max splice disruption.

The `.npz` scores are large and not committed; regenerate them with `... cli score` or point
`FUNCTIONAL_ALPHAGENOME_SCORES` at an existing set. `../tests/test_splice_reproduction.py`
checks the formulation reproduces the recorded top-splice gene + max splice per inversion.

## Provenance

AlphaGenome recommended `RNA_SEQ` + `SPLICE_SITE_USAGE` scorers; GTEx tissue tracks used for the
per-gene expression direction. sGene validation uses the GTEx v10 sQTL lookup
(`functional.regulatory.sqtl_gtex`).
