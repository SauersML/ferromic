"""Canonical final ordering of supplementary figures and tables.

The response letter contains several figures shown only to reviewers.  They are
not automatically supplementary figures.  The 17q21.31
tagging-SNP-versus-imputed-dosage comparison was Figure S11 of the submitted
supplement; the revision re-ran it against the single tagging SNP used in
Table S21 (``stats/tag_vs_imputed_concordance.py``) and that regenerated
panel is Figure S19.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_GENE_FLUX_CAPTION_PATH = (
    Path(__file__).resolve().parents[1]
    / "simulations/refsim/gene_flux_caption.txt"
)

# The caption is regenerated from the simulation grid by
# ``simulations/refsim/make_report.py``. Importing this module must not fail
# when the grid has not been regenerated in the current checkout (for example
# on a runner that only assembles figures), so fall back to a placeholder that
# is obvious in any assembled document rather than raising at import time.
try:
    GENE_FLUX_CAPTION = _GENE_FLUX_CAPTION_PATH.read_text(
        encoding="utf-8"
    ).strip()
except FileNotFoundError:  # pragma: no cover - depends on checkout contents
    GENE_FLUX_CAPTION = (
        "[gene-flux caption unavailable: run simulations/refsim/make_report.py "
        "to regenerate simulations/refsim/gene_flux_caption.txt]"
    )


def _matching_brace(text: str, start: int) -> int:
    """Index of the ``}`` closing the ``{`` at ``start``, or -1."""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _parse_caption(text: str, italic: bool, vert_align, out: list) -> None:
    buffer = ""
    index = 0
    while index < len(text):
        character = text[index]
        if character == "*":
            close = text.find("*", index + 1)
            if close == -1:
                buffer += character
                index += 1
                continue
            if buffer:
                out.append((buffer, italic, vert_align))
                buffer = ""
            _parse_caption(text[index + 1 : close], True, vert_align, out)
            index = close + 1
        elif character in "^_" and text[index + 1 : index + 2] == "{":
            close = _matching_brace(text, index + 1)
            if close == -1:
                buffer += character
                index += 1
                continue
            if buffer:
                out.append((buffer, italic, vert_align))
                buffer = ""
            nested = "superscript" if character == "^" else "subscript"
            _parse_caption(text[index + 2 : close], italic, nested, out)
            index = close + 1
        else:
            buffer += character
            index += 1
    if buffer:
        out.append((buffer, italic, vert_align))


def caption_segments(text: str) -> list:
    """Split caption markup into ``(text, italic, vertical alignment)`` runs.

    ``^{...}`` marks a superscript, ``_{...}`` a subscript and ``*...*``
    italics.  Markers nest, so ``*d_{xy}*`` is an italic d carrying an italic
    subscript.  Captions without markup yield a single run, so the assembled
    document is unchanged for them.
    """
    segments: list = []
    _parse_caption(text, False, None, segments)
    return segments


def caption_plain_text(text: str) -> str:
    """The caption with its markup removed."""
    return "".join(content for content, _, _ in caption_segments(text))


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
        "Orange and blue indicate inverted and direct orientation populations, respectively. Gray indicates the ancestral population. The vertical axis is time before present in thousands of years (25 years per generation). Gene flow within an orientation occurs at m = 10^{-8} per lineage per generation, and gene flux between orientations at m = 10^{-7} per lineage per generation is shown with arrows. Left: recurrent inversion simulation method. The inversion event happens three times. Right: single-event inversion simulation method. A single event leads to divergence between orientations.",
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
        "(A, B) Nucleotide diversity (π) of direct (blue) and inverted (purple) haplotypes, by recurrence class. Diversity is measured across the inversion locus (A) and at 4-fold-degenerate sites within coding sequences (B). Violins show the distribution and boxes indicate the median and interquartile range. Lines connect two orientations at the same locus. P-values are from two-sided paired Wilcoxon signed-rank tests. The 26 loci with 4-fold-degenerate sites in both orientations and a consensus recurrence classification (single-event n = 7, recurrent n = 19) are shown. Concordance of the per-locus orientation difference (Δπ = π_{inverted} − π_{direct}) is measured between 4-fold sites and the whole locus (C), and between 4-fold sites and the whole coding sequence (D). Spearman correlations are indicated for all loci and for the recurrence classification consensus set.",
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
        "The ancestral and derived orientations are determined with respect to the chimpanzee (PanTro6) for 57 inversion loci (Table S9). The 33 loci with at least two haplotypes in each orientation are shown: single-event (left, n = 19) or recurrent (right, n = 14). The distribution of nucleotide diversity (π) across loci is shown for each of the groups. The boxplot shows the median and interquartile range. The lines connect the ancestral and derived orientations at the same locus, which are colored by the log_{2} of (π_{ancestral} / π_{derived}), a measure of fold change in diversity.",
    ),
    SupplementaryFigure(
        10,
        "divergence",
        "Differentiation between orientations considering within-orientation diversity",
        "revision figure",
        None,
        "data/divergence_fst_dxy_da_by_type.png",
        "Hudson's *F_{ST}* (left), absolute divergence *d_{xy}* (center) and net divergence *d_{a}* (right) between direct and inverted haplotypes, for single-event (blue) and recurrent (purple) inversions. Points correspond to loci. Horizontal bars indicate medians. The p-values are from two-sided Mann–Whitney U tests between recurrence categories.",
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
    SupplementaryFigure(
        19,
        "tag_snp_concordance",
        "Comparison of 17q21.31 PheWAS results using imputed inversion dosage versus tagging-SNP dosage",
        "revision figure",
        None,
        "data/tag_vs_imputed_concordance.png",
        "(A) Association p-values from the imputed inversion dosage analysis and the tagging-SNP analysis, colored by significance. (B) Effect estimates for the two dosage measures. Gray points had BH q ≥ 0.05 in both analyses, blue points were significant using imputed dosage only, orange points were significant using tagging-SNP dosage only, and green points were significant in both analyses. This figure replaces Figure S11 of the original submission, which compared p-values against a three-SNP unanimity hard call; the tagging-SNP association results are given in Table S21.",
    ),
    SupplementaryFigure(20, "family_history", "Concordance between family and personal history for 17q21.31 allele associations", "original Figure S12", 12, None, None),
    SupplementaryFigure(21, "heritability", "Correlations between heritability and inversion effects", "original Figure S13", 13, None, None),
    SupplementaryFigure(
        22,
        "cds_test_calibration_power",
        "Calibration and power of the CDS permutation test",
        "revision figure",
        None,
        "data/cds_permutation_calibration_power.png",
        "Left: realized per-test rejection rate versus the nominal p-value threshold, computed from 100,000 null datasets generated by shuffling orientation labels once per inversion, in which any discovery is false by construction (Methods); the dashed line marks equality between realized and nominal rates. Right: power to detect a simulated difference in CDS pair identity between orientations at nominal p \u2264 0.05, averaged over the 66 tested genes; each gene\u2019s inverted haplotype class was regenerated at the indicated simulated difference, in both directions, from the gene\u2019s observed sequence pool (Methods). The dashed line marks 80% power.",
    ),
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
)


ORIGINAL_FIGURE_TO_FINAL = {
    figure.original_number: figure.number
    for figure in FINAL_SUPPLEMENTARY_FIGURES
    if figure.original_number is not None
}
