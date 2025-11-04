# Scripts

This directory contains automation scripts for the Ferromic project.

## `generate_phewas_figures_json.py`

Automatically generates the list of PheWAS Manhattan plots to display on the figures site.

### Purpose

Uses the same statistical filtering logic as `stats/forest.py` to determine which inversions should have Manhattan plots displayed on the website.

### Logic

The script:

1. Reads `data/phewas_results.tsv` (PheWAS association results)
2. Filters for significant associations: **Q_GLOBAL ≤ 0.05** (FDR-corrected)
3. Identifies all inversions with at least one significant phenotype association
4. Maps inversion IDs to genomic coordinates using `data/inv_properties.tsv`
5. Generates figure entries with filenames matching the Manhattan plot naming convention
6. Writes output to `web/figures-site/data/phewas-figures.json`

### Usage

**Automatic (during build):**
```bash
cd web/figures-site
npm run build  # Automatically runs this script via prebuild hook
```

**Manual:**
```bash
python3 scripts/generate_phewas_figures_json.py
```

### Dependencies

- Python 3.x (standard library only - no external packages required)
- Input files:
  - `data/phewas_results.tsv` - PheWAS association results
  - `data/inv_properties.tsv` - Inversion metadata (coordinates, IDs, etc.)

### Output

Generates `web/figures-site/data/phewas-figures.json`:

```json
[
  {
    "title": "PheWAS Manhattan — chr10:79,542,901-80,217,413",
    "filename": "phewas_plots/phewas_chr10_79_542_901-80_217_413.pdf",
    "description": "Phenome-wide association plot for the 674 kbp inversion at chr10:79,542,901-80,217,413."
  },
  ...
]
```

### Naming Convention

Manhattan plot PDFs must follow this naming pattern:
```
phewas_plots/phewas_chr{CHR}_{START}-{END}.pdf
```

Where coordinates use underscores instead of commas (e.g., `76_109_081` instead of `76,109,081`).

This matches the output of `stats/manhattan_phe.py`.
