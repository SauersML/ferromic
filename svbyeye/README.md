# SVbyEye inversion visualization pipeline

## Publication panTro6 plots for the 93 consensus loci

The manuscript supplement uses only the publication renderer
`bin/plot_chimp_hires.R`.  Every page has two tracks—human GRCh38 above and
chimpanzee panTro6 below—and a shallow dashed red rectangle on the human track
marks the inversion.  Full-height breakpoint lines, population-orientation
tracks, and the older manual-review images are not supplement inputs.

The complete source set is generated with one combined alignment so panTro6 is
scanned once:

```bash
python3 bin/prepare_chimp_consensus_targets.py \
  --properties data/inv_properties.tsv \
  --reference /path/to/hg38.no_alt.fa \
  --cytobands /path/to/cytoBand.txt.gz \
  --fasta work/targets.consensus93.fa \
  --manifest work/manifest.tsv

samtools faidx work/targets.consensus93.fa
minimap2 -x asm20 -c --eqx --secondary=yes -N 50 -t 8 \
  work/targets.consensus93.fa panTro6.fa.gz > work/all.consensus93.paf

python3 bin/render_chimp_consensus.py \
  --manifest work/manifest.tsv \
  --paf work/all.consensus93.paf \
  --plot-script bin/plot_chimp_hires.R \
  --rscript /path/to/SVbyEye/environment/bin/Rscript \
  --output-dir work/rendered
```

The immutable rendered bundle is checksum-pinned in
`reproducibility/manuscript_sources.json`.  GitHub Actions downloads that bundle
and `stats/svbyeye_supplement.py` verifies the generator hash, locus set, genomic
order, and identical PDF media boxes before assembling the 93-page source and
the two-locus numbered example figure.

Visualize every inversion in the callset by aligning phased human assemblies
(HGSVC3) plus chimpanzee to GRCh38 and reading alignment orientation across each
locus. Built on the [SVbyEye](https://github.com/daewoooo/SVbyEye) R package.
Runs in parallel on MSI (Slurm).

For each inversion it produces three views:

1. **SVbyEye miropeat** (`<inv_id>.png`) — canonical miropeat on a small curated
   set (chimp + representative direct/inverted haplotypes); inverted haplotypes
   show a reverse-strand ribbon.
2. **Population orientation tracks** (`<inv_id>.track.png`) — every assembly
   haplotype as a row, colored by strand vs GRCh38 (green = same, blue =
   inverted). The inverted-allele fraction is readable by eye.
3. **Directional gradient** (`<inv_id>.grad.png`) — alignments binned and colored
   by query position; direct haplotypes show an ascending color ramp, inverted
   ones a reversed ramp.

An orientation QC step compares the observed inverted-haplotype fraction to the
reported `Inverted_AF`, accounting for polarity (GRCh38 carries the inverted
allele at some loci).

## Inputs (on MSI)

- Reference: `di/hg38.no_alt.fa` (+ `.fai`)
- Inversions: `inv_properties.tsv` (repo root) → `make_inv_table.py` produces a
  QC'd `inversions.tsv` with per-locus plotting windows
- Query assemblies: HGSVC3 phased assemblies
  `shared/HGSVC3/centromereFixed/HGSVC3_repaired_assemblies/*-asm-renamed-reort.fa.gz`
- Chimp: `panTro6.fa.gz` (UCSC)
- Tools: modules `minimap2/2.30`, `samtools/1.21`; SVbyEye R env at
  `shared/conda_shared/envs/svbyeye_pipeline`

Paths in the `.sbatch` files are MSI-specific
(`/projects/standard/hsiehph/sauer354/di/svbyeye`); adjust for another host.

## Pipeline (`bin/`)

| Stage | Script | What it does |
|-------|--------|--------------|
| Prep  | `make_inv_table.py` | Parse `inv_properties.tsv`, validate vs `.fai`, emit `inversions.tsv` + windows |
| A     | `build_index.sbatch` | Build `hg38.asm20.mmi` (minimap2) |
| A     | `align_asm.sbatch` | Array: each assembly → hg38 PAF (`-x asm20 -c --eqx --secondary=no -s 25000 -K 1G`, 120 GB) |
| B1    | `distribute_paf.py` / `distribute.sbatch` | One pass per sample; write records overlapping each inversion window to `stageb/<inv_id>/<sample>.paf` (race-free) |
| B2    | `visualize_inv.R` / `plot_inv.sbatch` | Per inversion: miropeat + track plot; also calls `track_gradient.R` |
| B2    | `track_gradient.R` | Directional-gradient population plot |
| QC    | `orient_qc.py` | Classify each haplotype's orientation; polarity-aware AF concordance |
| Report| `make_gallery.py` / `finalize.sbatch` | HTML gallery + QC table |
| Chimp | `align_hg38_to_chimp.sbatch` | Single whole-genome GRCh38 → panTro6 alignment (chimp as reference) |

## PAV (`pav/`) — assembly-based inversion *calling*

The visualization above uses an orientation heuristic for QC, not a validated
caller. `pav/` sets up [PAV](https://github.com/EichlerLab/pav) (Phased Assembly
Variant caller) for real per-haplotype inversion genotypes:

- `split_one.sh` / `split_hap.sbatch` — split each HGSVC3 combined assembly into
  per-haplotype FASTAs (PAV input).
- `pav_config.json`, `pav_assemblies.tsv` — PAV run config (reference + assembly
  table).
- `pav_dryrun.sbatch` — validate the PAV DAG before committing compute.

## Run order

```bash
# on MSI, from di/svbyeye
python3 bin/make_inv_table.py inputs/inv_properties.tsv <ref>.fai inputs/inversions.tsv
sbatch bin/build_index.sbatch
sbatch --array=1-66 bin/align_asm.sbatch          # after index
sbatch --array=1-66 bin/distribute.sbatch         # after alignments
sbatch --array=1-292%60 bin/plot_inv.sbatch       # after distribute
sbatch bin/finalize.sbatch                        # after plots -> gallery + QC
```

## Manual chimp-alignment review

From the repository root, run:

```bash
python3 svbyeye/review_chimp_alignments.py
```

The local review app opens every pre-made `chimp vs GRCh38` figure in genomic
order. Choose **Direct**, **Inverted**, or **N/A**; each click is immediately
saved to `data/chimp_alignment_responses.json` with the exact inversion ID,
coordinates, and source image filename. Existing responses are loaded when the
app is restarted, and the app can export the current records as CSV or JSON.

Keyboard shortcuts are `D`, `I`, and `N`; use the left and right arrow keys to
move without changing a response.
