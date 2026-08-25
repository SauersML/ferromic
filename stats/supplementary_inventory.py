"""Canonical final ordering of supplementary figures and tables.

The response letter contains several figures shown only to reviewers.  They are
not automatically supplementary figures.  In particular, the 17q21.31
tagging-SNP-versus-imputed-dosage panel is response-only and is deliberately
absent from ``FINAL_SUPPLEMENTARY_FIGURES``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


GENE_FLUX_CAPTION = (
    Path(__file__).resolve().parents[1]
    / "simulations/refsim/gene_flux_caption.txt"
).read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class SupplementaryFigure:
    number: int
    key: str
    title: str
    source: str
    original_number: int | None
    asset: str | None
    caption: str | None


FINAL_SUPPLEMENTARY_FIGURES = (
    SupplementaryFigure(1, "old_recurrent", "Power analysis under scenarios of old recurrent inversion events", "original Figure S1", 1, None, None),
    SupplementaryFigure(2, "younger_recurrent", "Power analysis under scenarios of younger recurrent inversion events", "original Figure S2", 2, None, None),
    SupplementaryFigure(3, "recent_recurrent", "Power analysis under scenarios of recent recurrent inversion events", "original Figure S3", 3, None, None),
    SupplementaryFigure(4, "very_recent_recurrent", "Power analysis under scenarios of very recent recurrent inversion events", "original Figure S4", 4, None, None),
    SupplementaryFigure(
        5,
        "gene_flux_model",
        "Models for partial isolation between orientations with gene flux",
        "revision figure",
        None,
        "simulations/refsim/gene_flux_model.png",
        "Orange and blue indicate inverted and direct orientation populations, respectively. Gray indicates the ancestral population. The vertical axis is time before present in thousands of years (25 years per generation). Gene flow within an orientation occurs at m = 10-8 per lineage per generation, and gene flux between orientations at m = 10-7 per lineage per generation is shown with arrows. Left: recurrent inversion simulation method. The inversion event happens three times. Right: single-event inversion simulation method. A single event leads to divergence between orientations.",
    ),
    SupplementaryFigure(
        6,
        "gene_flux_performance",
        "Simulation results with gene flux: performance of the evolutionary approach for detecting recurrent inversions",
        "revision figure",
        None,
        "simulations/refsim/gene_flux_fpr_power.png",
        GENE_FLUX_CAPTION,
    ),
    SupplementaryFigure(
        7,
        "four_fold_diversity",
        "Nucleotide diversity at whole inversion loci and at 4-fold-degenerate sites",
        "revision figure",
        None,
        "data/four_fold_pi.png",
        "(A, B) Nucleotide diversity (π) of direct (blue) and inverted (purple) haplotypes, by recurrence class. Diversity is measured across the inversion locus (A) and at 4-fold-degenerate sites within coding sequences (B). Violins show the distribution and boxes indicate the median and interquartile range. Lines connect two orientations at the same locus. P-values are from two-sided paired Wilcoxon signed-rank tests. The 26 loci with 4-fold-degenerate sites in both orientations and a consensus recurrence classification (single-event n = 7, recurrent n = 19) are shown. Concordance of the per-locus orientation difference (Δπ = π_inverted − π_direct) is measured between 4-fold sites and the whole locus (C), and between 4-fold sites and the whole coding sequence (D). Spearman correlations are indicated for all loci and for the recurrence classification consensus set.",
    ),
    SupplementaryFigure(
        8,
        "sd_recurrence",
        "Comparison of nucleotide diversity between Porubsky et al. (2022) consensus- and SD-based recurrence classifications",
        "revision figure",
        None,
        "data/recurrence_sd_figure.png",
        "Left: Recurrent (orange) and single-event (blue-green) loci (n = 93) plotted by flanking SD size (x-axis) and sequence identity (y-axis). Shaded regions indicate the heuristic cutoff for recurrence: flanking SD ≥ 10 kbp and ≥ 95% identity. Right: The orientation-recurrence interaction effect on nucleotide diversity for the Porubsky et al. (2022) consensus calls (blue) and the SD heuristic classifier (purple). Under both classifications, diversity differences between arrangements are substantially larger in single-event inversions.",
    ),
    SupplementaryFigure(
        9,
        "ancestral_derived_diversity",
        "Comparison of nucleotide diversity by ancestral and derived orientation",
        "revision figure",
        None,
        "results/figure2a_repolarized/figure2a_repolarized.png",
        "The ancestral and derived orientations are determined with respect to the chimpanzee (PanTro6) for 57 inversion loci (Table S9). The 33 loci with at least two haplotypes in each orientation are shown: single-event (left, n = 19) or recurrent (right, n = 14). The distribution of nucleotide diversity (π) across loci is shown for each of the groups. The boxplot shows the median and interquartile range. The lines connect the ancestral and derived orientations at the same locus, which are colored by the log2 of (π_ancestral / π_derived), a measure of fold change in diversity.",
    ),
    SupplementaryFigure(
        10,
        "divergence",
        "Differentiation between orientations considering within-orientation diversity",
        "revision figure",
        None,
        "data/divergence_fst_dxy_da_by_type.png",
        "Hudson's FST (left), absolute divergence dxy (center) and net divergence da (right) between direct and inverted haplotypes, for single-event (blue) and recurrent (purple) inversions. Points correspond to loci. Horizontal bars indicate medians. The p-values are from two-sided Mann–Whitney U tests between recurrence categories.",
    ),
    SupplementaryFigure(11, "breakpoint_fst", "FST differences between breakpoint-proximal and middle regions of inversion loci", "original Figure S5", 5, None, None),
    SupplementaryFigure(12, "mapt_polymorphisms", "MAPT coding sequence polymorphisms across the 17q21.31 inversion locus’s inverted and direct haplotypes", "original Figure S6", 6, None, None),
    SupplementaryFigure(13, "ages_trajectories", "Allele frequency changes over time of the inversion tagging SNPs", "original Figure S7", 7, None, None),
    SupplementaryFigure(14, "imputation_accuracy", "Cross-validated imputation accuracy", "original Figure S8", 8, None, None),
    SupplementaryFigure(
        15,
        "hsinv0284_validation",
        "Validation of imputed inversion dosage against experimental genotypes at 6q24.1 (HsInv0284)",
        "revision figure",
        None,
        "data/imputation_benchmark_HsInv0284.png",
        "Left: Comparison of our imputed dosage (y-axis) against the experimentally genotyped dosage (x-axis) from Giner-Delgado et al. (2019) for 517 samples from the 1000 Genomes Project. The dotted line represents exact agreement. Right: Inverted allele frequencies stratified by superpopulation, comparing the external experimental dosages to our imputed dosage model.",
    ),
    SupplementaryFigure(16, "population_frequencies", "Imputed allele frequency by population in the NIH All of Us cohort", "original Figure S9", 9, None, None),
    SupplementaryFigure(
        17,
        "within_ancestry_pc",
        "Sensitivity of ancestry-specific PheWAS associations to within-ancestry principal-component adjustment",
        "revision figure",
        None,
        "results/phewas_within_ancestry/effect_pvalue_correspondence.png",
        "(A) Comparison of inversion effect estimates obtained using global and within-ancestry principal components. (B) Comparison of inversion p-values obtained using global and within-ancestry principal components.",
    ),
    SupplementaryFigure(18, "phenotype_categories", "Category-level meta-analysis of disease associations", "original Figure S10", 10, None, None),
    SupplementaryFigure(19, "family_history", "Concordance between family and personal history for 17q21.31 allele associations", "original Figure S12", 12, None, None),
    SupplementaryFigure(20, "heritability", "Correlations between heritability and inversion effects", "original Figure S13", 13, None, None),
)


SVBYEYE_APPENDIX_TITLE = (
    "SVbyEye alignments for the 93 consensus-classified inversions"
)

SVBYEYE_CONSENSUS_PDF = (
    "output/pdf/svbyeye/Supplemental_File_SVbyEye_consensus_93_loci.pdf"
)

SUPPLEMENT_TEMPLATE = (
    "reproducibility/templates/INV_MS_SupplementaryMaterials_v3.8.final.docx"
)

SUPPLEMENT_TEMPLATE_SHA256 = (
    "970176dd02306734a14f5cbdf40fa8f2d813ae9a92f3b374abae5b228fd59fb4"
)


FINAL_SUPPLEMENTARY_TABLE_ORDER = (
    "Old recurrent events",
    "Young recurrent events",
    "Recent recurrent events",
    "Very recent recurrent events",
    "Gene-flux simulation sweep",
    "Inversion catalog",
    "Coding-site diversity",
    "4-fold diversity concordance",
    "Chimpanzee polarity per locus",
    "Genomic-architecture controls",
    "Divergence between orientations",
    "CDS conservation genes",
    "dN/dS (ω) results",
    "Ancient DNA best tagging SNPs",
    "Ancient DNA, all tagging SNPs",
    "Imputation results",
    "Imputation external benchmarks",
    "PheWAS results",
    "Within-ancestry PC PheWAS",
    "Phenotype categories",
    "17q21 tagging PheWAS",
)


RESPONSE_ONLY_FIGURE_TITLES = (
    "Examples of inversion alignments to chimpanzee used for polarization",
    "Comparison of 17q21.31 PheWAS results, showing imputed inversion dosage "
    "versus tagging SNP dosage",
)


ORIGINAL_FIGURE_TO_FINAL = {
    figure.original_number: figure.number
    for figure in FINAL_SUPPLEMENTARY_FIGURES
    if figure.original_number is not None
}
