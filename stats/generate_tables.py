#!/usr/bin/env python3
"""Generate the supplementary tables Excel workbook.

This utility orchestrates the steps required to build the manuscript
supplementary tables. It performs the following operations:

1. Curates the inversion catalog from ``data/inv_properties.tsv``.
2. Loads the inversion-level permutation analysis of CDS conservation used in
   the revision.
3. Aggregates the published TSV artefacts into a single Excel workbook with an
   "Information" worksheet that explains each tab.

The resulting ``supplementary_tables.xlsx`` file is saved under the Next.js
public directory so the web site can link to it directly.
"""

from __future__ import annotations

import argparse
import math
import sys
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
NEXT_PUBLIC_DIR = REPO_ROOT / "web" / "figures-site" / "public"
DEFAULT_OUTPUT = NEXT_PUBLIC_DIR / "downloads" / "supplementary_tables.xlsx"

BEST_TAGGING_FILENAME = "best_tagging_snps_qvalues.tsv"

INV_COLUMNS_KEEP: List[str] = [
    "Chromosome",
    "Start",
    "End",
    "Number_recurrent_events",
    "OrigID",
    "Size_.kbp.",
    "Inverted_AF",
    "verdictRecurrence_hufsah",
    "verdictRecurrence_benson",
    "0_single_1_recur_consensus",
]

INV_RENAME_MAP: Dict[str, str] = {
    "Number_recurrent_events": "number recurrent events",
    "OrigID": "Inversion ID",
    "Size_.kbp.": "Size (kbp)",
    "Inverted_AF": "Inversion allele frequency",
    "hudson_fst_hap_group_0v1": "Hudson's FST",
    "0_pi_filtered": "Direct haplotypes pi",
    "1_pi_filtered": "Inverted haplotypes pi",
}


INVERSION_COLUMN_DEFS: Dict[str, str] = OrderedDict(
    [
        ("Chromosome", "The chromosome number (GRCh38 reference)."),
        ("Start", "The 1-based start coordinate of the inversion (GRCh38)."),
        ("End", "The 1-based end coordinate of the inversion (GRCh38)."),
        (
            "number recurrent events",
            "The minimum number of orientation changes inferred on the haplotype tree by Porubsky et al. (2022).",
        ),
        ("Inversion ID", "The unique identifier assigned to the inversion (format: chr-start-inv-id)."),
        ("Size (kbp)", "The length of the inverted segment in kilobase pairs."),
        (
            "Inversion allele frequency",
            "The frequency of the inverted allele observed in the phased reference panel (n=82 haplotypes).",
        ),
        ("verdictRecurrence_hufsah", "Recurrence classification based on the Hufsah algorithm."),
        ("verdictRecurrence_benson", "Recurrence classification based on the Benson algorithm."),
        (
            "0_single_1_recur_consensus",
            "Consensus recurrence status used throughout this study: 0 indicates a Single-event inversion (evolved via a single historical mutational event), 1 indicates a Recurrent inversion (evolved via multiple independent events).",
        ),
        (
            "Hudson's FST",
            "Hudson's fixation index (FST) comparing inverted (haplotype group 1) and direct (haplotype group 0) chromosomes across informative sites.",
        ),
        (
            "Direct haplotypes pi",
            "Nucleotide diversity (π) among direct haplotypes after site filtering.",
        ),
        (
            "Inverted haplotypes pi",
            "Nucleotide diversity (π) among inverted haplotypes after site filtering.",
        ),
    ]
)

GENE_CONSERVATION_COLUMN_DEFS: Dict[str, str] = OrderedDict(
    [
        ("Gene", "HGNC gene symbol."),
        ("Transcript", "Ensembl transcript ID used for the CDS analysis."),
        ("Inversion ID", "The identifier of the inversion overlapping this gene."),
        ("Recurrence class", "Consensus recurrence class of the inversion locus."),
        ("Direct haplotypes", "Number of direct-orientation haplotypes contributing the CDS."),
        ("Inverted haplotypes", "Number of inverted-orientation haplotypes contributing the CDS."),
        ("Sequence classes", "Number of distinct CDS sequence-identity classes across both orientations."),
        (
            "Orientation more conserved",
            "Indicates which haplotype orientation (Inverted or Direct) has a higher proportion of identical CDS pairs based on the sign of Δ.",
        ),
        (
            "Fixed CDS differences",
            "Count of CDS sites where direct and inverted haplotype groups are each fixed to different alleles (strict fixed-difference criterion).",
        ),
        (
            "Direct identical pair proportion",
            "The fraction of pairwise comparisons among direct haplotypes that resulted in 100% identical amino acid sequences.",
        ),
        (
            "Inverted identical pair proportion",
            "The fraction of pairwise comparisons among inverted haplotypes that resulted in 100% identical amino acid sequences.",
        ),
        (
            "Δ (inverted − direct)",
            "The difference in identical pair proportions (Inverted minus Direct). Positive values indicate higher conservation in the inverted orientation.",
        ),
        (
            "Permutation p-value",
            "Two-sided p-value from the joint inversion-level permutation null, which shuffles orientation labels once per inversion and applies the same assignment to every gene in that inversion.",
        ),
        (
            "Westfall-Young FWER p-value",
            "Westfall-Young adjusted p-value controlling the family-wise error rate under the joint permutation null.",
        ),
        (
            "Direct FDR q-value",
            "False-discovery-rate q-value estimated directly from the joint permutation null across the 130 tested genes.",
        ),
    ]
)

PHEWAS_COLUMN_DEFS: Dict[str, str] = OrderedDict(
    [
        (
            "Phenotype",
            "The unique phecode string representing the disease phenotype (derived from ICD billing codes).",
        ),
        ("Inversion", "The unique identifier of the chromosomal inversion locus being tested."),
        (
            "BH_P_GLOBAL",
            "Benjamini-Hochberg adjusted p-value (global FDR) corrected across all phenotypes and inversions tested in the study.",
        ),
        (
            "N_Controls",
            "The number of control participants (individuals without the phenotype) included in the analysis.",
        ),
        (
            "OR",
            "The Odds Ratio (OR) representing the change in disease risk per copy of the inversion allele. Derived from the exponential of the logistic regression beta coefficient.",
        ),
        (
            "CI_LO_OR",
            "The lower bound of the 95% confidence interval for the Odds Ratio. Calculated via Profile Likelihood for Firth/Penalized models, or Wald/Score methods for standard MLE.",
        ),
        ("CI_HI_OR", "The upper bound of the 95% confidence interval for the Odds Ratio."),
        (
            "N_Total",
            "The total number of participants (Cases + Controls) included in the logistic regression model after quality control and exclusion of related individuals.",
        ),
        ("N_Cases", "The number of case participants (individuals with the phenotype) included in the analysis."),
        (
            "P_Value_unadjusted",
            "The nominal p-value for the association. Derived from a Likelihood Ratio Test (LRT) for stable fits, or a Score Test/Firth Penalized Likelihood if the standard model failed to converge or exhibited separation.",
        ),
        (
            "P_Source_x",
            "The specific statistical test used to generate the p-value (e.g., 'lrt_mle', 'score_chi2', 'score_boot_mle'). Identifies if fallback methods were required.",
        ),
        (
            "CI_Method",
            "The statistical method used to calculate the confidence intervals (e.g., 'profile' for robust likelihood-based intervals, or 'wald_mle').",
        ),
        (
            "Inference_Type",
            "The statistical framework selected by the pipeline (e.g., 'mle', 'firth', 'score'). 'Firth' indicates penalized regression was used to handle rare case counts or separation.",
        ),
        (
            "Model_Notes",
            "Diagnostic flags generated during model fitting (e.g., 'sex_restricted' if analysis was limited to one sex, 'ridge_seeded' if regularization was needed for convergence).",
        ),
        (
            "Sig_Global",
            "Boolean indicator (TRUE/FALSE) denoting if the association is statistically significant at the global FDR threshold (q < 0.05).",
        ),
        (
            "Beta",
            "Logistic regression beta coefficient (log odds) for the inversion dosage term.",
        ),
        (
            "P_LRT_AncestryxDosage",
            "P-value from a Stage-2 Likelihood Ratio or Rao Score test comparing a model with 'Ancestry x Inversion' interaction terms against a base model. Tests if the inversion's effect size differs significantly by genetic ancestry.",
        ),
        (
            "P_Stage2_Valid",
            "Boolean indicating if the Stage-2 ancestry interaction model converged successfully and produced a valid p-value.",
        ),
        (
            "Stage2_P_Source",
            "The method used to calculate the interaction p-value (e.g., 'rao_score' is used for robust multi-degree-of-freedom tests when multiple ancestry groups are present).",
        ),
        (
            "Stage2_Inference_Type",
            "The statistical framework used for the Stage-2 interaction test.",
        ),
        ("Stage2_Model_Notes", "Diagnostic notes specific to the Stage-2 interaction model fit."),
        (
            "EUR_N",
            "Total participants included in the European ancestry stratum analysis.",
        ),
        ("EUR_N_Cases", "Number of cases in the European ancestry stratum."),
        ("EUR_N_Controls", "Number of controls in the European ancestry stratum."),
        (
            "EUR_OR",
            "Odds Ratio estimated specifically within the European ancestry stratum.",
        ),
        ("EUR_P", "Nominal p-value for the association within the European ancestry stratum."),
        (
            "EUR_P_Source",
            "Source of the p-value for the European ancestry stratum (e.g., 'score_chi2' if case counts were low).",
        ),
        (
            "EUR_Inference_Type",
            "Statistical framework used for the European ancestry stratum (e.g., 'firth' if the stratum had low case counts).",
        ),
        ("EUR_CI_Method", "Method used for confidence intervals in the European ancestry stratum."),
        ("EUR_CI_LO_OR", "Lower 95% CI bound for the European ancestry stratum."),
        ("EUR_CI_HI_OR", "Upper 95% CI bound for the European ancestry stratum."),
        (
            "AFR_N",
            "Total participants included in the African ancestry stratum analysis.",
        ),
        ("AFR_N_Cases", "Number of cases in the African ancestry stratum."),
        ("AFR_N_Controls", "Number of controls in the African ancestry stratum."),
        (
            "AFR_OR",
            "Odds Ratio estimated specifically within the African ancestry stratum.",
        ),
        ("AFR_P", "Nominal p-value for the association within the African ancestry stratum."),
        (
            "AFR_P_Source",
            "Source of the p-value for the African ancestry stratum (e.g., 'score_chi2' if case counts were low).",
        ),
        (
            "AFR_Inference_Type",
            "Statistical framework used for the African ancestry stratum (e.g., 'firth' if the stratum had low case counts).",
        ),
        ("AFR_CI_Method", "Method used for confidence intervals in the African ancestry stratum."),
        ("AFR_CI_LO_OR", "Lower 95% CI bound for the African ancestry stratum."),
        ("AFR_CI_HI_OR", "Upper 95% CI bound for the African ancestry stratum."),
        (
            "AMR_N",
            "Total participants included in the Admixed American ancestry stratum analysis.",
        ),
        ("AMR_N_Cases", "Number of cases in the Admixed American ancestry stratum."),
        ("AMR_N_Controls", "Number of controls in the Admixed American ancestry stratum."),
        (
            "AMR_OR",
            "Odds Ratio estimated specifically within the Admixed American ancestry stratum.",
        ),
        ("AMR_P", "Nominal p-value for the association within the Admixed American ancestry stratum."),
        (
            "AMR_P_Source",
            "Source of the p-value for the Admixed American ancestry stratum (e.g., 'score_chi2' if case counts were low).",
        ),
        (
            "AMR_Inference_Type",
            "Statistical framework used for the Admixed American ancestry stratum (e.g., 'firth' if the stratum had low case counts).",
        ),
        ("AMR_CI_Method", "Method used for confidence intervals in the Admixed American ancestry stratum."),
        ("AMR_CI_LO_OR", "Lower 95% CI bound for the Admixed American ancestry stratum."),
        ("AMR_CI_HI_OR", "Upper 95% CI bound for the Admixed American ancestry stratum."),
        (
            "SAS_N",
            "Total participants included in the South Asian ancestry stratum analysis.",
        ),
        ("SAS_N_Cases", "Number of cases in the South Asian ancestry stratum."),
        ("SAS_N_Controls", "Number of controls in the South Asian ancestry stratum."),
        (
            "SAS_OR",
            "Odds Ratio estimated specifically within the South Asian ancestry stratum.",
        ),
        ("SAS_P", "Nominal p-value for the association within the South Asian ancestry stratum."),
        (
            "SAS_P_Source",
            "Source of the p-value for the South Asian ancestry stratum (e.g., 'score_chi2' if case counts were low).",
        ),
        (
            "SAS_Inference_Type",
            "Statistical framework used for the South Asian ancestry stratum (e.g., 'firth' if the stratum had low case counts).",
        ),
        ("SAS_CI_Method", "Method used for confidence intervals in the South Asian ancestry stratum."),
        ("SAS_CI_LO_OR", "Lower 95% CI bound for the South Asian ancestry stratum."),
        ("SAS_CI_HI_OR", "Upper 95% CI bound for the South Asian ancestry stratum."),
        (
            "EAS_N",
            "Total participants included in the East Asian ancestry stratum analysis.",
        ),
        ("EAS_N_Cases", "Number of cases in the East Asian ancestry stratum."),
        ("EAS_N_Controls", "Number of controls in the East Asian ancestry stratum."),
        (
            "EAS_OR",
            "Odds Ratio estimated specifically within the East Asian ancestry stratum.",
        ),
        ("EAS_P", "Nominal p-value for the association within the East Asian ancestry stratum."),
        (
            "EAS_P_Source",
            "Source of the p-value for the East Asian ancestry stratum (e.g., 'score_chi2' if case counts were low).",
        ),
        (
            "EAS_Inference_Type",
            "Statistical framework used for the East Asian ancestry stratum (e.g., 'firth' if the stratum had low case counts).",
        ),
        ("EAS_CI_Method", "Method used for confidence intervals in the East Asian ancestry stratum."),
        ("EAS_CI_LO_OR", "Lower 95% CI bound for the East Asian ancestry stratum."),
        ("EAS_CI_HI_OR", "Upper 95% CI bound for the East Asian ancestry stratum."),
        (
            "MID_N",
            "Total participants included in the Middle Eastern ancestry stratum analysis.",
        ),
        ("MID_N_Cases", "Number of cases in the Middle Eastern ancestry stratum."),
        ("MID_N_Controls", "Number of controls in the Middle Eastern ancestry stratum."),
        (
            "MID_OR",
            "Odds Ratio estimated specifically within the Middle Eastern ancestry stratum.",
        ),
        ("MID_P", "Nominal p-value for the association within the Middle Eastern ancestry stratum."),
        (
            "MID_P_Source",
            "Source of the p-value for the Middle Eastern ancestry stratum (e.g., 'score_chi2' if case counts were low).",
        ),
        (
            "MID_Inference_Type",
            "Statistical framework used for the Middle Eastern ancestry stratum (e.g., 'firth' if the stratum had low case counts).",
        ),
        ("MID_CI_Method", "Method used for confidence intervals in the Middle Eastern ancestry stratum."),
        ("MID_CI_LO_OR", "Lower 95% CI bound for the Middle Eastern ancestry stratum."),
        ("MID_CI_HI_OR", "Upper 95% CI bound for the Middle Eastern ancestry stratum."),
    ]
)

WITHIN_ANCESTRY_PHEWAS_COLUMN_DEFS: Dict[str, str] = OrderedDict(
    [
        ("population", "All of Us genetic-ancestry group used for the stratified model."),
        ("population_label", "Expanded label for the genetic-ancestry group."),
        ("Inversion", "Inversion identifier."),
        ("locus", "Cytogenetic locus label, where available."),
        ("Phenotype", "Phecode-derived phenotype label."),
        ("pooled_or", "Odds ratio from the original pooled multi-ancestry PheWAS."),
        ("pooled_q", "Global BH-adjusted p-value from the original pooled PheWAS."),
        (
            "existing_or",
            "Odds ratio from the existing ancestry-stratified model using 16 global principal components.",
        ),
        ("existing_p", "Nominal p-value from the existing ancestry-stratified model."),
        ("within_or", "Odds ratio after replacing the global components with 16 components fitted within the ancestry group."),
        ("within_p", "Nominal p-value from the within-ancestry-PC sensitivity model."),
        (
            "within_q_selected_set",
            "BH-adjusted p-value within the preselected sensitivity set; this is not an independent replication statistic.",
        ),
        ("within_n_total", "Total participants in the within-ancestry-PC model."),
        ("within_n_cases", "Cases in the within-ancestry-PC model."),
        ("within_n_controls", "Controls in the within-ancestry-PC model."),
        ("within_ci_lo_or", "Lower reported confidence bound for the within-ancestry-PC odds ratio."),
        ("within_ci_hi_or", "Upper reported confidence bound for the within-ancestry-PC odds ratio."),
        ("evaluable", "TRUE when both PC strategies produced valid estimates for direct comparison."),
        (
            "direction_concordant",
            "TRUE when the existing and within-ancestry-PC log odds ratios have the same sign.",
        ),
        (
            "beta_shift_within_minus_existing",
            "Within-ancestry-PC log odds ratio minus the existing stratified log odds ratio.",
        ),
        ("absolute_beta_shift", "Absolute value of the change in log odds ratio."),
        ("not_evaluable_reason", "Reason a comparison could not be evaluated."),
    ]
)

def _phewas_desc(column: str, fallback: str) -> str:
    return PHEWAS_COLUMN_DEFS.get(column, fallback)

TAG_PHEWAS_COLUMN_DEFS: Dict[str, str] = OrderedDict(
    [
        ("Phenotype", _phewas_desc("Phenotype", "Phenotype identifier.")),
        ("BH_P_GLOBAL", _phewas_desc("BH_P_GLOBAL", "Global Benjamini-Hochberg adjusted p-value.")),
        ("P_Value_unadjusted", "Nominal p-value for the association using the tagging SNP model."),
        ("N_Total", _phewas_desc("N_Total", "Total participants analyzed.")),
        ("N_Cases", _phewas_desc("N_Cases", "Number of cases.")),
        ("N_Controls", _phewas_desc("N_Controls", "Number of controls.")),
        ("Beta", _phewas_desc("Beta", "Logistic regression beta coefficient.")),
        (
            "OR",
            "Odds Ratio representing the change in disease risk per copy of the inversion haplotype (defined by tagging SNPs).",
        ),
        ("P_Valid", _phewas_desc("P_Valid", "Whether the p-value is valid.")),
        ("P_Source_x", _phewas_desc("P_Source", "Statistic used for the p-value.")),
        ("OR_CI95", _phewas_desc("OR_CI95", "95% confidence interval for the odds ratio.")),
        ("CI_Method", _phewas_desc("CI_Method", "Method used to compute the confidence interval.")),
        ("CI_Sided", _phewas_desc("CI_Sided", "Indicates if CI is one- or two-sided.")),
        ("CI_Valid", _phewas_desc("CI_Valid", "Whether the confidence interval is valid.")),
        ("CI_LO_OR", _phewas_desc("CI_LO_OR", "Lower CI bound for odds ratio.")),
        ("CI_HI_OR", _phewas_desc("CI_HI_OR", "Upper CI bound for odds ratio.")),
        ("Used_Ridge", _phewas_desc("Used_Ridge", "TRUE if ridge regularization was used.")),
        ("Final_Is_MLE", _phewas_desc("Final_Is_MLE", "TRUE if final fit uses MLE.")),
        ("Used_Firth", _phewas_desc("Used_Firth", "TRUE if Firth penalization was required.")),
        ("Inference_Type", _phewas_desc("Inference_Type", "Inference framework used.")),
        ("N_Total_Used", _phewas_desc("N_Total_Used", "Participants contributing to final model.")),
        ("N_Cases_Used", _phewas_desc("N_Cases_Used", "Case count contributing to final model.")),
        ("N_Controls_Used", _phewas_desc("N_Controls_Used", "Control count contributing to final model.")),
        ("Model_Notes", _phewas_desc("Model_Notes", "Diagnostic notes for this association.")),
        ("Inversion", _phewas_desc("Inversion", "Inversion identifier.")),
        ("P_LRT_Overall", _phewas_desc("P_LRT_Overall", "Overall LRT p-value.")),
        ("P_Overall_Valid", _phewas_desc("P_Overall_Valid", "Validity flag for overall LRT.")),
        ("P_Source_y", _phewas_desc("P_Source", "Statistic used for overall p-value.")),
        ("P_Method", _phewas_desc("P_Method", "Computation method for overall p-value.")),
        ("Sig_Global", _phewas_desc("Sig_Global", "TRUE if globally significant (q < 0.05).")),
        ("CI_Valid_DISPLAY", _phewas_desc("CI_Valid_DISPLAY", "Display flag for CI.")),
        ("CI_Method_DISPLAY", _phewas_desc("CI_Method_DISPLAY", "Display text for CI method.")),
        ("OR_CI95_DISPLAY", _phewas_desc("OR_CI95_DISPLAY", "Formatted CI for display.")),
        ("CI_LO_OR_DISPLAY", _phewas_desc("CI_LO_OR_DISPLAY", "Formatted lower CI bound.")),
        ("CI_HI_OR_DISPLAY", _phewas_desc("CI_HI_OR_DISPLAY", "Formatted upper CI bound.")),
    ]
)

CATEGORY_COLUMN_DEFS: Dict[str, str] = OrderedDict(
    [
        ("Inversion", "The Inversion ID."),
        ("Category", "The phecode category being tested."),
        ("Phenotypes in category", "Total number of phenotypes in this category."),
        ("Phenotypes included in GBJ", "Number of phenotypes passing QC that were included in the omnibus test."),
        ("Phenotypes included in GLS", "Number of phenotypes included in the GLS directional meta-analysis."),
        ("P_GBJ", "P-value for the GBJ omnibus test (testing if any signal exists in the category)."),
        ("GLS test statistic", "Test statistic for the Generalized Least Squares directional meta-analysis."),
        ("P_GLS", "P-value for the GLS directional test."),
        (
            "Direction",
            "The aggregate direction of effect (Increased Risk or Decreased Risk) if the GLS test is significant.",
        ),
        ("N_Individuals", "Number of individuals contributing to the category-level analysis."),
        ("GBJ_Draws", "Number of Monte Carlo draws used to approximate the GBJ p-value."),
        ("Phenotypes", "List or count of phenotypes in the category considered for GBJ."),
        ("Phenotypes_GLS", "List or count of phenotypes in the category considered for GLS."),
        ("BH_P_GBJ", "Benjamini-Hochberg adjusted p-value for the GBJ test."),
        ("BH_P_GLS", "Benjamini-Hochberg adjusted p-value for the GLS test."),
    ]
)

IMPUTATION_COLUMN_DEFS: Dict[str, str] = OrderedDict(
    [
        (
            "Inversion",
            "Inversion coordinates (chr:start-end, GRCh38) corresponding to the OrigID used for model training.",
        ),
        ("n_components", "Number of PLS components selected via cross-validation."),
        (
            "unbiased_pearson_r2",
            "Pearson r² correlation between imputed and true dosage in held-out cross-validation folds.",
        ),
        ("p_value", "P-value comparing the trained model against a null intercept-only model."),
        ("p_fdr_bh", "FDR adjusted p-value."),
        (
            "overall_allele_frequency_AoU",
            "Allele frequency of the inverted allele across all (overall) populations in the All of Us dataset when imputation performance meets the unbiased Pearson r² > 0.5 threshold.",
        ),
        (
            "afr_allele_frequency_AoU",
            "Allele frequency of the inverted allele in African (afr) samples in the All of Us dataset when unbiased Pearson r² > 0.5.",
        ),
        (
            "amr_allele_frequency_AoU",
            "Allele frequency of the inverted allele in American (amr) samples in the All of Us dataset when unbiased Pearson r² > 0.5.",
        ),
        (
            "eas_allele_frequency_AoU",
            "Allele frequency of the inverted allele in East Asian (eas) samples in the All of Us dataset when unbiased Pearson r² > 0.5.",
        ),
        (
            "eur_allele_frequency_AoU",
            "Allele frequency of the inverted allele in European (eur) samples in the All of Us dataset when unbiased Pearson r² > 0.5.",
        ),
        (
            "mid_allele_frequency_AoU",
            "Allele frequency of the inverted allele in Middle Eastern (mid) samples in the All of Us dataset when unbiased Pearson r² > 0.5.",
        ),
        (
            "sas_allele_frequency_AoU",
            "Allele frequency of the inverted allele in South Asian (sas) samples in the All of Us dataset when unbiased Pearson r² > 0.5.",
        ),
        (
            "Use",
            "Boolean flag indicating if the inversion met the quality threshold (r² > 0.5 and q < 0.05) for inclusion in the PheWAS.",
        ),
    ]
)

BEST_TAGGING_COLUMN_DEFS: Dict[str, str] = OrderedDict(
    [
        (
            "inversion_region",
            "Inversion interval (GRCh38/hg38 coordinates) reported by the tagging SNP pipeline (chr:start-end).",
        ),
        (
            "p_x",
            "P-value (P_X) from the ancient selection summary statistics corresponding to the tagging SNP (hg19/GRCh37).",
        ),
        ("s", "Selection coefficient estimate from the selection summary statistics (hg19/GRCh37)."),
        ("REF", "Reference allele for the tagging SNP in the selection dataset."),
        ("ALT", "Alternate allele for the tagging SNP in the selection dataset."),
        ("AF", "Alternate allele frequency reported in the selection summary statistics."),
        (
            "REF_freq_direct",
            "Frequency of the REF allele among direct (haplotype group 0) chromosomes in the tagging SNP analysis.",
        ),
        (
            "REF_freq_inverted",
            "Frequency of the REF allele among inverted (haplotype group 1) chromosomes in the tagging SNP analysis.",
        ),
        (
            "ALT_freq_direct",
            "Frequency of the ALT allele among direct (haplotype group 0) chromosomes in the tagging SNP analysis.",
        ),
        (
            "ALT_freq_inverted",
            "Frequency of the ALT allele among inverted (haplotype group 1) chromosomes in the tagging SNP analysis.",
        ),
        (
            "exclusion_reasons",
            "Semicolon-delimited reasons why the tagging SNP did not pass quality filters (e.g., low r², low haplotype count, missing selection stats).",
        ),
        (
            "correlation_r",
            "Pearson correlation (r) between the tagging SNP allele and inversion orientation (direct vs. inverted haplotypes).",
        ),
        ("abs_r", "Absolute correlation |r| for the tagging SNP within the inversion region."),
        ("hg37_coordinate", "Tagging SNP coordinate on GRCh37/hg19 in chr:pos format (e.g., chr1:10583)."),
        ("hg38_coordinate", "Tagging SNP coordinate on GRCh38/hg38 in chr:pos format (e.g., chr1:10583)."),
        (
            "bh_p_value",
            "Benjamini–Hochberg adjusted p-value across inversions that passed tagging SNP quality filters (computed from P_X).",
        ),
    ]
)

SIMULATION_COLUMN_DEFS: Dict[str, str] = OrderedDict(
    [
        ("First inversion event (years ago)", "Time of the first inversion event."),
        ("Second inversion event (years ago)", "Time of the second inversion event."),
        ("Third inversion event (years ago)", "Time of the third inversion event."),
        ("Sample size (haplotypes)", "Number of haplotypes simulated."),
        ("Inversion frequency", "Frequency of the inversion."),
        ("Recombination rate (per generation per base pair)", "Recombination rate used in simulation."),
        ("Gene flow (per generation per chromosome)", "Gene flow rate used in simulation."),
    ]
)

PAML_COLUMN_DEFS: Dict[str, str] = OrderedDict(
    [
        ("region", "The identifier of the genomic inversion region (e.g., chr17:42000-45000)."),
        ("gene", "The gene symbol or identifier being analyzed."),
        (
            "status",
            "The final result of the pipeline for this gene (success or partial_success rows are retained).",
        ),
        ("cmc_p_value", "P-value for the Clade Model C test."),
        ("cmc_bh_p_value", "Benjamini-Hochberg adjusted p-value for the Clade Model C test."),
        ("cmc_lrt_stat", "Likelihood ratio test statistic for the Clade Model C comparison."),
        ("cmc_lnl_h1", "Log-likelihood of the alternative hypothesis (different ω for divergent sites)."),
        ("cmc_lnl_h0", "Log-likelihood of the null hypothesis (shared ω for divergent sites)."),
        ("cmc_p0", "Proportion of sites in site class 0 (strictly conserved)."),
        ("cmc_p1", "Proportion of sites in site class 1 (neutral evolution)."),
        ("cmc_p2", "Proportion of sites in site class 2 (divergent selection class of interest)."),
        ("cmc_omega0", "dN/dS (ω) estimate for conserved site class 0."),
        ("cmc_omega2_direct", "dN/dS (ω) estimate for divergent sites in the Direct clade."),
        ("cmc_omega2_inverted", "dN/dS (ω) estimate for divergent sites in the Inverted clade."),
        ("cmc_kappa", "Estimated transition/transversion ratio (κ)."),
        (
            "n_leaves_pruned",
            "Number of sequences retained after intersecting the region tree and gene alignment.",
        ),
        (
            "taxa_used",
            "Semicolon-separated list of the exact samples included in the PAML analysis (reproducibility).",
        ),
    ]
)

GENE_PERMUTATION_TSV = DATA_DIR / "per_gene_cds_permutation.tsv"
GENE_JOINT_CONTROL_TSV = DATA_DIR / "cds_permutation_joint_control.tsv"
FIXED_DIFF_SUMMARY_TSV = DATA_DIR / "fixed_diff_summary.tsv"

PHEWAS_RESULTS = DATA_DIR / "phewas_results.tsv"
WITHIN_ANCESTRY_PHEWAS_RESULTS = DATA_DIR / "phewas_within_ancestry_correspondence.tsv"
PHEWAS_TAGGING_RESULTS = DATA_DIR / "all_pop_phewas_tag.tsv"
CATEGORIES_RESULTS = DATA_DIR / "phewas v2 - categories.tsv"
IMPUTATION_RESULTS = DATA_DIR / "imputation_results.tsv"
INV_PROPERTIES = DATA_DIR / "inv_properties.tsv"
POPULATION_METRICS = DATA_DIR / "output.csv"
POPULATION_FREQUENCIES = DATA_DIR / "inversion_population_frequencies.tsv"
BEST_TAGGING_RESULTS = DATA_DIR / BEST_TAGGING_FILENAME
PAML_RESULTS = DATA_DIR / "GRAND_PAML_RESULTS.tsv"
IMPUTATION_RESULTS_MERGED = DATA_DIR / "imputation_results_merged.tsv"

TABLE_S1 = DATA_DIR / "tables.xlsx - Table S1.tsv"
TABLE_S2 = DATA_DIR / "tables.xlsx - Table S2.tsv"
TABLE_S3 = DATA_DIR / "tables.xlsx - Table S3.tsv"
TABLE_S4 = DATA_DIR / "tables.xlsx - Table S4.tsv"


# --------------------------------------------------------------------------- #
# Revision tables: analysis-set provenance and per-locus power/identifiability
# --------------------------------------------------------------------------- #

def _load_tsv(path: Path, what: str) -> pd.DataFrame:
    """Read a committed TSV, failing loudly if the generating script never ran."""
    if not path.exists():
        raise SupplementaryTablesError(
            f"{what} table missing: {path}. Run the script that generates it."
        )
    return pd.read_csv(path, sep="\t")


def _load_exclusion_reasons() -> pd.DataFrame:
    return _load_tsv(DATA_DIR / "table_s5_exclusion_reasons.tsv",
                     "Inversion exclusion-reason")


def _load_fourfold_correlations() -> pd.DataFrame:
    df = _load_tsv(
        DATA_DIR / "four_fold_pi_correlations.tsv",
        "4-fold diversity concordance",
    )
    expected = {
        ("wholeLocus", "fourfold"): (0.501267, 0.00908858),
        ("fourfold", "wholeCDS"): (0.628613, 0.000583291),
    }
    for measures, (expected_rho, expected_p) in expected.items():
        row = df.loc[
            df["subset"].eq("recurrence_classified")
            & df["measure_x"].eq(measures[0])
            & df["measure_y"].eq(measures[1])
            & df["statistic"].eq("spearman_rho")
        ]
        if len(row) != 1:
            raise SupplementaryTablesError(
                f"4-fold concordance lacks the unique recurrence-classified {measures} correlation."
            )
        rho = float(row.iloc[0]["value"])
        p_value = float(row.iloc[0]["p_value"])
        if not (math.isclose(rho, expected_rho, abs_tol=5e-7)
                and math.isclose(p_value, expected_p, abs_tol=5e-10)):
            raise SupplementaryTablesError(
                f"4-fold concordance {measures} is stale: observed rho={rho}, p={p_value}."
            )
    return df


# --- revision tables -------------------------------------------------------
# Added for the reviewer response. Each reads a committed artefact so the
# workbook stays reproducible from the repository alone.

REFSIM_DIR = REPO_ROOT / "simulations" / "refsim"


def _load_coding_site_diversity() -> pd.DataFrame:
    """4-fold pi and piN/piS per locus, from the two scripts that compute them."""
    ff = _load_tsv(DATA_DIR / "four_fold_pi_by_inversion.tsv", "4-fold diversity")
    pn = _load_tsv(DATA_DIR / "pin_pis_by_inversion.tsv", "piN/piS")
    ff = ff[pd.to_numeric(ff["recurrence"], errors="coerce").isin([0, 1])].copy()
    pn = pn[pd.to_numeric(pn["recurrence"], errors="coerce").isin([0, 1])].copy()
    key = ["chr", "region_start", "region_end"]
    dup = [c for c in pn.columns if c in ff.columns and c not in key]
    return ff.merge(pn.drop(columns=dup), on=key, how="outer")


def _load_divergence() -> pd.DataFrame:
    return _load_tsv(DATA_DIR / "divergence_da_dxy_by_type.tsv",
                     "Dxy / da divergence")


def _load_ages_all_tags() -> pd.DataFrame:
    return _load_tsv(DATA_DIR / "ages_multi_tag_snps.tsv",
                     "AGES all tagging SNPs")


def _load_architecture_controls() -> pd.DataFrame:
    df = _load_tsv(
        DATA_DIR / "recurrence_controls_summary.tsv",
        "Genomic-architecture controls",
    )
    required = set(ARCHITECTURE_CONTROLS_COLUMN_DEFS)
    missing = required - set(df.columns)
    if missing:
        raise SupplementaryTablesError(
            "Genomic-architecture controls are missing columns: "
            + ", ".join(sorted(missing))
        )
    for col in ("n", "n_recur", "n_single"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    if len(df) != 9 or not (df["n"] == df["n_recur"] + df["n_single"]).all():
        raise SupplementaryTablesError(
            "Genomic-architecture controls must contain nine rows with n = n_recur + n_single in every row."
        )
    return df


def _load_chimp_polarity() -> pd.DataFrame:
    return _load_tsv(
        REPO_ROOT / "results" / "figure2a_repolarized" / "figure2a_locus_audit.tsv",
        "Chimpanzee polarity audit")


def _load_imputation_benchmarks() -> pd.DataFrame:
    """External checks on imputed dosage: ScoreInvHap, plus experimental 6q24.1."""
    sih = _load_tsv(DATA_DIR / "scoreinvhap_concordance.tsv",
                    "ScoreInvHap concordance")
    sih = sih.rename(columns={"n_samples": "n", "r2": "agreement_r2",
                              "hardcall_concordance": "hard_call_concordance",
                              "our_inv_allele_freq": "inverted_allele_freq_imputed",
                              "sih_inv_allele_freq": "inverted_allele_freq_external"})
    sih["comparison"] = "ScoreInvHap (Ruiz-Arenas et al. 2019)"

    per_sample = _load_tsv(DATA_DIR / "imputation_benchmark_HsInv0284.tsv",
                           "HsInv0284 benchmark")
    summary = _load_tsv(DATA_DIR / "imputation_benchmark_HsInv0284_summary.tsv",
                        "HsInv0284 benchmark summary")
    overall = summary[summary["group"] == "ALL"].iloc[0]
    hsi = pd.DataFrame([{
        "inversion": "6q24.1 (HsInv0284)",
        "comparison": "Experimental genotypes (Giner-Delgado et al. 2019)",
        "n": int(overall["n"]),
        "agreement_r2": float(overall["r2"]),
        "hard_call_concordance": float(overall["concordance"]),
        "inverted_allele_freq_imputed": per_sample["imputed_dosage"].mean() / 2,
        "inverted_allele_freq_external": per_sample["experimental_dosage"].mean() / 2,
    }])
    cols = ["inversion", "comparison", "n", "agreement_r2", "hard_call_concordance",
            "inverted_allele_freq_imputed", "inverted_allele_freq_external"]
    out = pd.concat([hsi[cols], sih[cols]], ignore_index=True)
    order = ["6q24.1 (HsInv0284)", "17q21.31", "8p23.1"]
    out["_o"] = out["inversion"].apply(
        lambda x: order.index(x) if x in order else len(order))
    out = out.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    expected = {
        "6q24.1 (HsInv0284)": (517, 0.9428699595637554, 0.9961315280464217),
        "17q21.31": (500, 0.9443, 0.976),
        "8p23.1": (500, 0.7575, 0.758),
    }
    if len(out) != 3 or set(out["inversion"]) != set(expected):
        raise SupplementaryTablesError("External imputation benchmark must contain exactly the three validated loci.")
    for _, row in out.iterrows():
        n, r2, concordance = expected[row["inversion"]]
        if (int(row["n"]) != n
                or not math.isclose(float(row["agreement_r2"]), r2, abs_tol=5e-7)
                or not math.isclose(float(row["hard_call_concordance"]), concordance, abs_tol=5e-7)):
            raise SupplementaryTablesError(
                f"External imputation benchmark is stale for {row['inversion']}."
            )
    return out


def _load_flux_sweep() -> pd.DataFrame:
    """Load the complete sweep's pooled counts and derive display statistics.

    The full grid contains 12 depth-by-recombination cells per scenario and
    flux value, with 120 deterministic replicate loci per cell.  Keeping the
    eight pooled rows is sufficient for the reviewer-response claims while
    avoiding an otherwise redundant 96-row expansion of the same counts.
    """
    df = _load_tsv(REFSIM_DIR / "gene_flux_summary.tsv", "Gene-flux sweep")
    required = {
        "scenario", "m_flux", "n_cells", "replicates_per_cell", "reps",
        "n_called", "trend_p",
    }
    missing = required - set(df.columns)
    if missing:
        raise SupplementaryTablesError(
            "Gene-flux summary is missing columns: " + ", ".join(sorted(missing))
        )
    numeric = ["m_flux", "n_cells", "replicates_per_cell", "reps", "n_called", "trend_p"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if len(df) != 8 or int(df["reps"].sum()) != 11_520:
        raise SupplementaryTablesError(
            f"Gene-flux summary must contain 8 pooled rows and 11,520 loci; observed {len(df)} rows and {int(df['reps'].sum())} loci."
        )
    if not (df["reps"] == df["n_cells"] * df["replicates_per_cell"]).all():
        raise SupplementaryTablesError("Gene-flux replicate totals do not equal cells x replicates per cell.")
    expected_counts = {
        ("single-event", 0.0): (37, 0.00712342373249036),
        ("single-event", 1e-8): (49, 0.00712342373249036),
        ("single-event", 1e-7): (45, 0.00712342373249036),
        ("single-event", 1e-6): (66, 0.00712342373249036),
        ("recurrent", 0.0): (1099, 0.305858351751984),
        ("recurrent", 1e-8): (1133, 0.305858351751984),
        ("recurrent", 1e-7): (1139, 0.305858351751984),
        ("recurrent", 1e-6): (1121, 0.305858351751984),
    }
    observed_keys = set(zip(df["scenario"], df["m_flux"]))
    if observed_keys != set(expected_counts):
        raise SupplementaryTablesError(
            "Gene-flux summary does not contain the complete two-scenario by four-flux grid."
        )
    for row in df.itertuples(index=False):
        expected_called, expected_trend = expected_counts[(row.scenario, row.m_flux)]
        if (int(row.n_called) != expected_called
                or not math.isclose(float(row.trend_p), expected_trend, abs_tol=5e-15)):
            raise SupplementaryTablesError(
                f"Gene-flux summary is stale for {row.scenario} at m={row.m_flux}."
            )
    df["recurrent_call_rate"] = df["n_called"] / df["reps"]

    def wilson(row: pd.Series) -> tuple[float, float]:
        z = 1.96
        n = float(row["reps"])
        p = float(row["recurrent_call_rate"])
        denominator = 1.0 + z * z / n
        centre = (p + z * z / (2.0 * n)) / denominator
        half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
        return max(0.0, centre - half), min(1.0, centre + half)

    intervals = df.apply(wilson, axis=1)
    df["ci_low"] = [value[0] for value in intervals]
    df["ci_high"] = [value[1] for value in intervals]
    df["metric"] = df["scenario"].map({
        "single-event": "False-positive rate",
        "recurrent": "Power",
    })
    if df["metric"].isna().any():
        raise SupplementaryTablesError("Gene-flux summary contains an unknown scenario label.")
    return df.sort_values(["scenario", "m_flux"], kind="mergesort").reset_index(drop=True)


CODING_DIVERSITY_COLUMN_DEFS = {
    "chr": "Chromosome.",
    "region_start": "Inversion start coordinate (GRCh38).",
    "region_end": "Inversion end coordinate (GRCh38).",
    "recurrence": "Consensus recurrence label (0 = single-event, 1 = recurrent).",
    "n_cds": "Coding sequences overlapping the locus.",
    "n_cds_with_fourfold": "Coding sequences contributing 4-fold-degenerate sites.",
    "fourfold_sites_direct": "4-fold-degenerate sites compared, direct haplotypes.",
    "fourfold_sites_inverted": "4-fold-degenerate sites compared, inverted haplotypes.",
    "pi_fourfold_direct": "Nucleotide diversity at 4-fold sites, direct haplotypes.",
    "pi_fourfold_inverted": "Nucleotide diversity at 4-fold sites, inverted haplotypes.",
    "pi_wholeCDS_direct": "Nucleotide diversity across whole coding sequence, direct.",
    "pi_wholeCDS_inverted": "Nucleotide diversity across whole coding sequence, inverted.",
    "pi_wholeLocus_direct": "Nucleotide diversity across the whole locus, direct.",
    "pi_wholeLocus_inverted": "Nucleotide diversity across the whole locus, inverted.",
    "zerofold_sites_direct": "0-fold-degenerate sites compared, direct haplotypes.",
    "zerofold_sites_inverted": "0-fold-degenerate sites compared, inverted haplotypes.",
    "piN_direct": "Nonsynonymous diversity (0-fold sites), direct haplotypes.",
    "piN_inverted": "Nonsynonymous diversity (0-fold sites), inverted haplotypes.",
    "piS_direct": "Synonymous diversity (4-fold sites), direct haplotypes.",
    "piS_inverted": "Synonymous diversity (4-fold sites), inverted haplotypes.",
    "piN_piS_direct": "Ratio of nonsynonymous to synonymous diversity, direct.",
    "piN_piS_inverted": "Ratio of nonsynonymous to synonymous diversity, inverted.",
}

DIVERGENCE_COLUMN_DEFS = {
    "chr": "Chromosome.",
    "region_start": "Inversion start coordinate (GRCh38).",
    "region_end": "Inversion end coordinate (GRCh38).",
    "category": "Recurrence category of the locus.",
    "hudson_pi_hap_group_0": "Nucleotide diversity within direct haplotypes.",
    "hudson_pi_hap_group_1": "Nucleotide diversity within inverted haplotypes.",
    "hudson_fst_hap_group_0v1": "Hudson's FST between orientations.",
    "dxy": "Absolute divergence between orientations.",
    "da": "Net divergence between orientations (Dxy minus mean within-orientation diversity).",
}

AGES_ALL_TAGS_COLUMN_DEFS = {
    "region": "Inversion locus.",
    "rsid": "Tagging SNP identifier.",
    "chrom_hg38": "Chromosome (GRCh38).",
    "pos_hg38": "Position (GRCh38).",
    "chrom_hg19": "Chromosome (GRCh37), as queried in AGES.",
    "pos_hg19": "Position (GRCh37), as queried in AGES.",
    "r_with_inversion": "Correlation between the SNP and inversion orientation.",
    "abs_r": "Absolute correlation with orientation.",
    "alt_enriched_on": "Orientation on which the alternate allele is enriched.",
    "ages_S": "Selection coefficient reported by AGES for the tested allele.",
    "ages_S_inverted_allele": "Selection coefficient signed to the inverted allele.",
    "ages_S_ci_lo": "Lower bound of the selection coefficient interval.",
    "ages_S_ci_hi": "Upper bound of the selection coefficient interval.",
    "ages_SE": "Standard error of the selection coefficient.",
    "ages_P_X": "AGES selection p-value.",
    "ages_FDR": "Benjamini-Hochberg adjusted AGES p-value.",
    "ages_FILTER": "AGES quality filter status.",
}

ARCHITECTURE_CONTROLS_COLUMN_DEFS = {
    "outcome": "Quantity being compared between recurrence classes.",
    "control": "Conditioning set used in the conditional-randomization test.",
    "effect": "Estimated effect on the stated scale.",
    "ci_lo": "Lower bound of the 95% confidence interval.",
    "ci_hi": "Upper bound of the 95% confidence interval.",
    "p": "Two-sided p-value.",
    "n": "Loci contributing to the estimate.",
    "n_recur": "Recurrent loci contributing.",
    "n_single": "Single-event loci contributing.",
    "scale": "Scale on which the effect is expressed.",
}

CHIMP_POLARITY_COLUMN_DEFS = {
    "inv_id": "Inversion locus.",
    "chrom": "Chromosome.",
    "start": "Inversion start coordinate (GRCh38).",
    "end": "Inversion end coordinate (GRCh38).",
    "recurrence": "Consensus recurrence label.",
    "chimp_call": "Which human arrangement is shared with the chimpanzee, from manual review.",
    "flip_ref_polarity": "Whether the reference orientation had to be flipped to become ancestral.",
    "included_in_plot": "Whether the locus enters the diversity figure.",
    "included_in_model": "Whether the locus enters the statistical model.",
    "plot_exclusion_reason": "Why the locus was excluded from the figure, if it was.",
    "model_exclusion_reason": "Why the locus was excluded from the model, if it was.",
    "pi_ancestral": "Nucleotide diversity among ancestral-orientation haplotypes.",
    "pi_derived": "Nucleotide diversity among derived-orientation haplotypes.",
}

IMPUTATION_BENCHMARK_COLUMN_DEFS = {
    "inversion": "Inversion locus.",
    "comparison": "External genotype source compared against.",
    "n": "Samples compared.",
    "agreement_r2": "Squared Pearson correlation between imputed and external dosage.",
    "hard_call_concordance": "Fraction of samples agreeing after rounding to 0/1/2.",
    "inverted_allele_freq_imputed": "Inverted allele frequency from the imputed dosage.",
    "inverted_allele_freq_external": "Inverted allele frequency from the external genotypes.",
}

FLUX_SWEEP_COLUMN_DEFS = {
    "scenario": "Simulated locus class: single-event or recurrent.",
    "metric": "The recurrent-call rate is a false-positive rate for single-event loci and power for recurrent loci.",
    "m_flux": "Between-orientation gene flux, per lineage per generation.",
    "n_cells": "Number of inversion-age by recombination-rate grid cells pooled in this row.",
    "replicates_per_cell": "Simulated loci per grid cell.",
    "reps": "Total simulated loci pooled in this row.",
    "n_called": "Loci called recurrent by the reference classifier.",
    "recurrent_call_rate": "Proportion called recurrent: the false-positive rate for single-event loci, the power for recurrent loci.",
    "ci_low": "Lower bound of the Wilson 95% interval.",
    "ci_high": "Upper bound of the Wilson 95% interval.",
    "trend_p": "Two-sided Cochran-Armitage trend-test p-value across the four gene-flux levels for this scenario.",
}

FOURFOLD_CORR_COLUMN_DEFS = {
    "subset": "Consensus-classified loci with 4-fold sites in both orientations.",
    "measure_x": "First diversity measure in the comparison.",
    "measure_y": "Second diversity measure in the comparison.",
    "comparison": "Human-readable description of the comparison, including any interval.",
    "n_loci": "Loci (or locus-orientation observations) contributing.",
    "statistic": (
        "Which concordance statistic the row reports: a Spearman correlation of "
        "orientation differences or of diversity levels, the fraction of loci "
        "agreeing in sign, or the median simulated correlation attainable if "
        "agreement were perfect apart from 4-fold site-sampling noise."
    ),
    "value": "Value of the statistic.",
    "p_value": (
        "Two-sided p-value for correlations and sign agreement; for noise "
        "ceilings, the fraction of simulations at or below the observed "
        "correlation."
    ),
}

# --------------------------------------------------------------------------- #
# Canonical inversion naming (Reviewer 2 #11)
# --------------------------------------------------------------------------- #

_CANON_COL = "inversion"


def _canonical_locus_map() -> Dict[str, str]:
    """Map every identifier style in the catalog to canonical ``chr:start-end``.

    Sheets identify loci four different ways -- explicit chrom/start/end columns,
    the Porubsky ``chrN-start-INV-length`` OrigID, an underscore-joined
    ``chrN_start_end`` region key, and ``chr:start-end`` itself. The OrigID's
    coordinates are *not* the analysed locus coordinates, so it has to be mapped
    through the catalog rather than reformatted.
    """
    mapping: Dict[str, str] = {}
    path = DATA_DIR / "inv_properties.tsv"
    if not path.exists():
        return mapping
    inv = pd.read_csv(path, sep="\t", dtype=str).rename(columns=lambda c: c.strip())
    for _, r in inv.iterrows():
        chrom = str(r.get("Chromosome", "")).strip()
        start, end = str(r.get("Start", "")).strip(), str(r.get("End", "")).strip()
        if not chrom or not start or not end:
            continue
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"
        try:
            canon = f"{chrom}:{int(float(start))}-{int(float(end))}"
        except ValueError:
            continue
        for key in (canon, f"{chrom}_{int(float(start))}_{int(float(end))}",
                    str(r.get("OrigID", "")).strip()):
            if key:
                mapping[key] = canon
    return mapping


def _normalise_locus_token(token: str, locus_map: Dict[str, str]):
    """Coerce any locus identifier style to canonical ``chr:start-end``.

    Handles ``chrN:start-end``, bare ``N:start-end`` (Tables S10/S11/S17),
    ``chrN_start_end`` (the PAML region key) and the Porubsky
    ``chrN-start-INV-length`` OrigID, whose coordinates differ from the analysed
    locus and so must go through the catalog rather than be reformatted.
    """
    if token is None:
        return pd.NA
    t = str(token).strip()
    if not t or t.upper() == "NA":
        return pd.NA
    if t in locus_map:
        return locus_map[t]
    if not t.startswith("chr"):
        prefixed = f"chr{t}"
        if prefixed in locus_map:
            return locus_map[prefixed]
        t = prefixed
    m = re.match(r"^(chr[\w]+)[:_](\d+)[-_](\d+)$", t)
    if m:
        canon = f"{m.group(1)}:{int(m.group(2))}-{int(m.group(3))}"
        return locus_map.get(canon, canon)
    return locus_map.get(t, pd.NA)


def _add_canonical_inversion_column(df: pd.DataFrame,
                                    locus_map: Dict[str, str]) -> pd.DataFrame:
    """Prepend (or normalise) a canonical ``inversion`` column, Reviewer 2 #11.

    Sheets identify loci four different ways. Original identifier columns are kept
    so nothing becomes untraceable; this only guarantees that one consistently
    formatted column exists and reads the same in every table.
    """
    if df.empty:
        return df
    cols = {str(c).strip().lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    chrom_c = pick("chromosome", "chr", "chrom", "chr_std")
    start_c = pick("start", "region_start", "inv_start")
    end_c = pick("end", "region_end", "inv_end")

    canon = None
    if chrom_c and start_c and end_c:
        def build(row):
            try:
                c = str(row[chrom_c]).strip()
                if not c or c.upper() == "NA":
                    return pd.NA
                if not c.startswith("chr"):
                    c = f"chr{c}"
                return f"{c}:{int(float(row[start_c]))}-{int(float(row[end_c]))}"
            except (TypeError, ValueError):
                return pd.NA
        canon = df.apply(build, axis=1)
    else:
        for cand in ("inversion", "inversion id", "inv_id", "origid", "region",
                     "inversion_region", "locus", "inversion_id"):
            col = pick(cand)
            if col is None:
                continue
            mapped = df[col].map(lambda v: _normalise_locus_token(v, locus_map))
            if mapped.notna().sum() >= max(1, int(0.5 * len(df))):
                canon = mapped
                break

    if canon is None or canon.notna().sum() == 0:
        return df
    out = df.copy()
    if _CANON_COL in out.columns:
        out[_CANON_COL] = canon                 # normalise in place
    else:
        out.insert(0, _CANON_COL, canon)
    # Always first, so the same identifier sits in the same place in every table.
    ordered = [_CANON_COL] + [c for c in out.columns if c != _CANON_COL]
    return out[ordered]


@dataclass
class SheetInfo:
    name: str
    description: str
    column_defs: Dict[str, str]
    loader: Callable[[], pd.DataFrame]
    # Optional raw-column -> printed-header overrides. Anything not listed here
    # is prettified by ``_pretty_label``.
    column_labels: Dict[str, str] = field(default_factory=dict)


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

# Fixed scientific inventory for the revision. These counts make deletion of an
# obsolete table, accidental filtering, or a partial upstream export a hard
# failure instead of a silently different workbook.
EXPECTED_SUPPLEMENTARY_DATA_ROWS = (
    18,    # S1  Old recurrent events
    18,    # S2  Young recurrent events
    18,    # S3  Recent recurrent events
    18,    # S4  Very recent recurrent events
    8,     # S5  Gene-flux simulation sweep
    93,    # S6  Inversion catalog
    35,    # S7  Coding-site diversity
    26,    # S8  4-fold diversity concordance
    93,    # S9  Chimpanzee polarity per locus
    9,     # S10 Genomic-architecture controls
    93,    # S11 Divergence between orientations
    130,   # S12 CDS conservation genes
    206,   # S13 dN/dS results
    93,    # S14 Ancient DNA best tagging SNPs
    45,    # S15 Ancient DNA, all tagging SNPs
    75,    # S16 Imputation results
    3,     # S17 Imputation external benchmarks
    7_630, # S18 PheWAS results: seven inversions by 1,090 phenotypes
    234,   # S19 Within-ancestry PC PheWAS
    112,   # S20 Phenotype categories
    1_097, # S21 17q21 tagging-SNP PheWAS
)


class SupplementaryTablesError(RuntimeError):
    """Raised for unrecoverable supplementary table failures."""


# Printed headers. Supplementary tables are read by people, so no column may
# reach the page as a variable name. Most of the raw names are systematic -- the
# PheWAS sheets repeat one suffix set across six ancestry prefixes -- so the
# labelling is rules plus a short override list, not 200 hand-written strings.
_ANCESTRY = {
    "AFR": "African", "AMR": "Admixed American", "EAS": "East Asian",
    "EUR": "European", "MID": "Middle Eastern", "SAS": "South Asian",
}

# Suffixes shared by the per-ancestry PheWAS blocks.
_SUFFIX_LABELS = {
    "OR": "odds ratio",
    "CI_HI_OR": "odds ratio upper 95% CI",
    "CI_LO_OR": "odds ratio lower 95% CI",
    "CI_Method": "confidence interval method",
    "Inference_Type": "inference type",
    "N": "samples",
    "N_Cases": "cases",
    "N_Controls": "controls",
    "P": "p-value",
    "P_Source": "p-value source",
}

_EXPLICIT_LABELS = {
    "0_single_1_recur_consensus": "Consensus recurrence (0 = single-event, 1 = recurrent)",
    "AF": "Allele frequency",
    "ALT": "Alternate allele",
    "ALT_freq_direct": "Alternate allele frequency, direct haplotypes",
    "ALT_freq_inverted": "Alternate allele frequency, inverted haplotypes",
    "REF": "Reference allele",
    "REF_freq_direct": "Reference allele frequency, direct haplotypes",
    "REF_freq_inverted": "Reference allele frequency, inverted haplotypes",
    "abs_r": "Absolute correlation with orientation",
    "ancestral_allele": "Ancestral allele",
    "ancestral_allele_confidence": "Confidence in the ancestral allele call",
    "ancestral_allele_n_tag": "Tagging SNPs supporting the ancestral allele",
    "bh_p_value": "Benjamini-Hochberg adjusted p-value",
    "both_orientations": "Both orientations observed",
    "chr_std": "Chromosome",
    "clade_with_higher_omega2": "Orientation with the higher omega",
    "consensus": "Consensus recurrence label",
    "consensus_recurrence": "Consensus recurrence label",
    "correlation_r": "Correlation with orientation",
    "exclusion_reason": "Reason for exclusion",
    "exclusion_reasons": "Reasons for exclusion",
    "gene": "Gene",
    "gene_name": "Gene",
    "hg37_coordinate": "Coordinate (GRCh37)",
    "hg38_coordinate": "Coordinate (GRCh38)",
    "inv_id": "Inversion",
    "inv_underpowered_lt4": "Fewer than four inverted haplotypes",
    "inversion": "Inversion",
    "inversion_region": "Inversion region",
    "k_dir": "Direct haplotypes compared",
    "k_inv": "Inverted haplotypes compared",
    "measure_x": "First diversity measure",
    "measure_y": "Second diversity measure",
    "n_call_ancestral": "Loci called ancestral",
    "n_call_derived": "Loci called derived",
    "n_components": "Partial least squares components",
    "n_leaves_pruned": "Tree tips pruned",
    "n_loci": "Loci",
    "n_methods_informative": "Informative methods",
    "not_identifiable_flags": "Identifiability warnings",
    "omega2_direct": "Omega, direct haplotypes",
    "omega2_inverted": "Omega, inverted haplotypes",
    "orig_id": "Original identifier",
    "overall_p_value": "Overall p-value",
    "overall_q_value": "Overall q-value",
    "p-value": "p-value",
    "p_value": "p-value",
    "p_x": "p-value",
    "p2_divergent_class": "Proportion of codons in the divergent site class",
    "p_fdr_bh": "Benjamini-Hochberg adjusted p-value",
    "p_recurrent_insample": "Probability of recurrence (in-sample)",
    "p_recurrent_loo": "Probability of recurrence (leave-one-out)",
    "recurrence": "Recurrence",
    "recurrence_class": "Recurrence class",
    "reference_inverted_AF": "Inverted allele frequency (reference panel)",
    "region": "Region",
    "s": "Selection coefficient",
    "sd_call_insample": "Architecture-based recurrence call (in-sample)",
    "sd_call_loo": "Architecture-based recurrence call (leave-one-out)",
    "sd_identity_pct": "Flanking repeat identity (%)",
    "sd_size_kbp": "Flanking repeat size (kbp)",
    "spearman_rho": "Spearman correlation",
    "status": "Status",
    "strandseq": "Strand-seq orientation",
    "strandseq_confidence": "Confidence in the Strand-seq orientation",
    "strandseq_species": "Species used for the Strand-seq orientation",
    "subset": "Locus subset",
    "synteny": "Synteny orientation",
    "t2t_apes": "Great-ape T2T orientation",
    "t2t_apes_confidence": "Confidence in the great-ape T2T orientation",
    "t2t_apes_species": "Species used for the great-ape T2T orientation",
    "taxa_used": "Taxa used",
    "transcript": "Transcript",
    "transcript_id": "Transcript",
    "unbiased_pearson_r2": "Cross-validated imputation r2",
    "comparison": "Comparison",
    "analysed": "Analysed",
    "Use": "Used in the analysis set",
}
for _sp in ("chimp", "gorilla", "macaque", "orangutan"):
    _EXPLICIT_LABELS[f"synteny_{_sp}"] = f"Synteny orientation vs {_sp}"
for _pop in ("overall", "afr", "amr", "eas", "eur", "mid", "sas"):
    _EXPLICIT_LABELS[f"{_pop}_allele_frequency_AoU"] = (
        "Imputed inverted allele frequency, All of Us"
        + ("" if _pop == "overall" else f" ({_ANCESTRY[_pop.upper()]})"))
for _who in ("benson", "hufsah"):
    _EXPLICIT_LABELS[f"verdictRecurrence_{_who}"] = f"Recurrence verdict, reviewer {_who.title()}"
for _k in ("kappa", "lnl_h0", "lnl_h1", "lrt_stat", "omega0", "omega2_direct",
           "omega2_inverted", "p0", "p1", "p2", "p_value", "bh_p_value"):
    _EXPLICIT_LABELS.setdefault(f"cmc_{_k}", "Clade model C " + _k.replace("_", " "))


_EXPLICIT_LABELS.update({
    # revision tables
    "inversion": "Inversion locus",
    "Inversion": "Original inversion ID",
    "region_start": "Start (GRCh38)", "region_end": "End (GRCh38)",
    "n_cds": "Coding sequences", "n_cds_used": "Coding sequences used",
    "n_cds_with_fourfold": "Coding sequences with 4-fold sites",
    "fourfold_sites_direct": "4-fold sites, direct",
    "fourfold_sites_inverted": "4-fold sites, inverted",
    "zerofold_sites_direct": "0-fold sites, direct",
    "zerofold_sites_inverted": "0-fold sites, inverted",
    "pi_fourfold_direct": "pi at 4-fold sites, direct",
    "pi_fourfold_inverted": "pi at 4-fold sites, inverted",
    "pi_wholeCDS_direct": "pi across coding sequence, direct",
    "pi_wholeCDS_inverted": "pi across coding sequence, inverted",
    "pi_wholeLocus_direct": "pi across locus, direct",
    "pi_wholeLocus_inverted": "pi across locus, inverted",
    "piN_direct": "piN, direct", "piN_inverted": "piN, inverted",
    "piS_direct": "piS, direct", "piS_inverted": "piS, inverted",
    "piN_piS_direct": "piN/piS, direct", "piN_piS_inverted": "piN/piS, inverted",
    "category": "Recurrence category",
    "hudson_pi_hap_group_0": "pi within direct haplotypes",
    "hudson_pi_hap_group_1": "pi within inverted haplotypes",
    "hudson_fst_hap_group_0v1": "Hudson's FST between orientations",
    "hudson_dxy_hap_group_0v1": "Dxy between orientations",
    "dxy": "Absolute divergence (Dxy)", "da": "Net divergence (da)",
    "outcome": "Outcome", "control": "Control strategy", "effect": "Effect",
    "ci_lo": "Lower 95% CI", "ci_hi": "Upper 95% CI",
    "ci_low": "Lower 95% CI", "ci_high": "Upper 95% CI",
    "p": "p-value", "n_recur": "Recurrent loci", "n_single": "Single-event loci",
    "scale": "Effect scale",
    "chrom": "Chromosome", "start": "Start (GRCh38)", "end": "End (GRCh38)",
    "chimp_call": "Arrangement shared with chimpanzee",
    "flip_ref_polarity": "Reference orientation flipped to ancestral",
    "included_in_plot": "Included in the figure",
    "included_in_model": "Included in the model",
    "plot_exclusion_reason": "Reason excluded from the figure",
    "model_exclusion_reason": "Reason excluded from the model",
    "pi_ancestral": "pi, ancestral orientation",
    "pi_derived": "pi, derived orientation",
    "selection_kind": "Selection inference type", "context": "Context",
    "chrom_hg19": "Chromosome (GRCh37)", "pos_hg19": "Position (GRCh37)",
    "chrom_hg38": "Chromosome (GRCh38)", "pos_hg38": "Position (GRCh38)",
    "rsid": "Tagging SNP", "r_with_inversion": "Correlation with orientation",
    "alt_enriched_on": "Alternate allele enriched on",
    "ages_ref": "Reference allele (AGES)", "ages_alt": "Alternate allele (AGES)",
    "ages_S": "Selection coefficient",
    "ages_S_inverted_allele": "Selection coefficient, inverted allele",
    "ages_S_ci_lo": "Selection coefficient lower CI",
    "ages_S_ci_hi": "Selection coefficient upper CI",
    "ages_SE": "Selection coefficient standard error",
    "ages_P_X": "Selection p-value",
    "ages_FDR": "Benjamini-Hochberg adjusted selection p-value",
    "ages_FILTER": "AGES quality filter", "in_ages": "Present in AGES",
    "n": "Samples", "agreement_r2": "Imputed vs external r2",
    "hard_call_concordance": "Hard-call concordance",
    "inverted_allele_freq_imputed": "Inverted allele frequency, imputed",
    "inverted_allele_freq_external": "Inverted allele frequency, external",
    "scenario": "Simulated locus class",
    "metric": "Performance metric",
    "m_flux": "Gene flux (per lineage per generation)",
    "n_cells": "Grid cells pooled",
    "replicates_per_cell": "Loci per grid cell",
    "reps": "Simulated loci", "n_called": "Loci called recurrent",
    "recurrent_call_rate": "Proportion called recurrent",
    "trend_p": "Trend-test p-value",
})


def _pretty_label(col: str) -> str:
    """A printed header, not a variable name."""
    raw = str(col)
    if raw in _EXPLICIT_LABELS:
        return _EXPLICIT_LABELS[raw]
    head, _, rest = raw.partition("_")
    if head in _ANCESTRY and rest in _SUFFIX_LABELS:
        return f"{_ANCESTRY[head]} {_SUFFIX_LABELS[rest]}"
    if raw in _SUFFIX_LABELS:
        return _SUFFIX_LABELS[raw][0].upper() + _SUFFIX_LABELS[raw][1:]
    if "_" not in raw and not raw.islower():
        return raw                       # already a printed header
    words = raw.replace("_", " ").strip()
    return words[0].upper() + words[1:] if words else raw


def _prune_columns(
    df: pd.DataFrame, column_defs: Dict[str, str], sheet_name: str
) -> pd.DataFrame:
    expected_cols = list(column_defs.keys())
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise SupplementaryTablesError(
            f"Sheet '{sheet_name}' is missing required columns: {', '.join(missing)}."
        )
    return df.loc[:, expected_cols].copy()


def _format_chr_pos(chrom: str | float | int | None, pos: str | float | int | None) -> str | pd._libs.missing.NAType:
    if chrom is None or pos is None:
        return pd.NA

    chrom_text = str(chrom).removeprefix("chr").removesuffix(".0")
    try:
        chrom_text = str(int(float(chrom_text)))
    except (ValueError, TypeError):
        chrom_text = chrom_text

    pos_val = pd.to_numeric(pos, errors="coerce")
    if pd.isna(pos_val):
        return pd.NA

    return f"chr{chrom_text}:{int(pos_val)}"


def _format_chr_pos_from_text(value: str | float | int | None) -> str | pd._libs.missing.NAType:
    if value is None or pd.isna(value):
        return pd.NA

    text = str(value)
    if ":" not in text:
        return pd.NA

    chrom, pos = text.split(":", 1)
    return _format_chr_pos(chrom, pos)


def _coalesce_coordinate(
    df: pd.DataFrame,
    *,
    existing_col: str,
    chrom_col: str,
    pos_col: str,
) -> pd.Series:
    """Return a chr:pos coordinate preferring the explicit column when present.

    The best-tagging SNP artefact may already include a fully formatted coordinate
    column (e.g., ``hg38``). If that column is missing or empty, fall back to
    formatting chromosome/position pairs or a single ``chr:pos`` text column.
    """

    result = pd.Series(pd.NA, index=df.index)

    if existing_col in df.columns:
        result = result.combine_first(df[existing_col])

    if {chrom_col, pos_col}.issubset(df.columns):
        formatted = pd.Series(
            [_format_chr_pos(chrom, pos) for chrom, pos in zip(df[chrom_col], df[pos_col])],
            index=df.index,
        )
        result = result.combine_first(formatted)
    elif pos_col in df.columns:
        formatted = df[pos_col].apply(_format_chr_pos_from_text)
        result = result.combine_first(formatted)

    return result


def _prepare_merge_columns(df: pd.DataFrame, chrom_col: str, start_col: str, end_col: str) -> pd.DataFrame:
    def _normalize_chr(series: pd.Series) -> pd.Series:
        return series.astype(str).str.replace(r"^chr", "", regex=True).str.strip()

    result = df.copy()
    result["_merge_chr"] = _normalize_chr(result[chrom_col])
    result["_merge_start"] = pd.to_numeric(result[start_col], errors="coerce").astype("Int64")
    result["_merge_end"] = pd.to_numeric(result[end_col], errors="coerce").astype("Int64")
    return result


def _merge_population_metrics(inv_df: pd.DataFrame) -> pd.DataFrame:
    if not POPULATION_METRICS.exists():
        raise SupplementaryTablesError(f"Population metrics CSV not found: {POPULATION_METRICS}")

    metrics_df = pd.read_csv(POPULATION_METRICS, dtype=str, low_memory=False)
    required_cols = [
        "chr",
        "region_start",
        "region_end",
        "hudson_fst_hap_group_0v1",
        "0_pi_filtered",
        "1_pi_filtered",
    ]

    missing_metrics = [col for col in required_cols if col not in metrics_df.columns]
    if missing_metrics:
        raise SupplementaryTablesError(
            "Population metrics CSV is missing required columns: " + ", ".join(missing_metrics)
        )

    inv_with_keys = _prepare_merge_columns(inv_df, "Chromosome", "Start", "End")
    metrics_with_keys = _prepare_merge_columns(metrics_df, "chr", "region_start", "region_end")

    metrics_trimmed = metrics_with_keys[
        ["_merge_chr", "_merge_start", "_merge_end", "hudson_fst_hap_group_0v1", "0_pi_filtered", "1_pi_filtered"]
    ]

    # The upstream VCF pipeline can occasionally emit the same locus twice (a tiny
    # zero-diversity region), which would break the one_to_one validation below. Collapse
    # duplicate metric rows on the merge key (keep the most-informative copy) so a benign
    # upstream duplicate cannot fail the whole supplementary-table generation.
    metrics_trimmed = (
        metrics_trimmed.sort_values(["0_pi_filtered", "1_pi_filtered"], na_position="first")
        .drop_duplicates(subset=["_merge_chr", "_merge_start", "_merge_end"], keep="last")
    )

    merged = inv_with_keys.merge(
        metrics_trimmed,
        how="left",
        on=["_merge_chr", "_merge_start", "_merge_end"],
        validate="one_to_one",
    )

    helper_cols = [col for col in merged.columns if col.startswith("_merge_")]
    return merged.drop(columns=helper_cols)


def _load_imputation_performance_ids(min_r2: float = 0.5) -> set[str]:
    df = _load_tsv(IMPUTATION_RESULTS_MERGED, "Merged imputation performance")

    required_cols = {"id", "unbiased_pearson_r2"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise SupplementaryTablesError(
            "Imputation performance results are missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    df = df[list(required_cols)].copy()
    df["unbiased_pearson_r2"] = pd.to_numeric(df["unbiased_pearson_r2"], errors="coerce")
    return set(df.loc[df["unbiased_pearson_r2"] > min_r2, "id"].dropna().astype(str).str.strip())


def _load_population_frequency_table() -> tuple[pd.DataFrame, List[str]]:
    if not POPULATION_FREQUENCIES.exists():
        raise SupplementaryTablesError(
            f"Inversion population frequency TSV not found: {POPULATION_FREQUENCIES}"
        )

    freq_df = pd.read_csv(POPULATION_FREQUENCIES, sep="\t", dtype=str, low_memory=False)
    required_cols = {"Inversion", "Population", "Allele_Freq"}
    missing_cols = required_cols - set(freq_df.columns)
    if missing_cols:
        raise SupplementaryTablesError(
            "Inversion population frequency TSV is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    freq_df = freq_df[list(required_cols)].copy()
    freq_df["Population"] = freq_df["Population"].str.strip().str.lower()
    freq_df["Allele_Freq"] = pd.to_numeric(freq_df["Allele_Freq"], errors="coerce")
    freq_df["column_name"] = (
        freq_df["Population"].replace({"all": "overall"}) + "_allele_frequency_AoU"
    )

    duplicate_mask = freq_df.duplicated(subset=["Inversion", "Population"], keep=False)
    if duplicate_mask.any():
        dup_rows = freq_df.loc[duplicate_mask, ["Inversion", "Population"]].drop_duplicates()
        raise SupplementaryTablesError(
            "Inversion population frequency TSV contains duplicate inversion/population pairs:\n"
            + dup_rows.to_csv(index=False)
        )

    pivot = freq_df.pivot(index="Inversion", columns="column_name", values="Allele_Freq").reset_index()
    column_names = sorted(freq_df["column_name"].unique())
    return pivot, column_names


def _add_population_allele_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    freq_pivot, freq_columns = _load_population_frequency_table()
    imputation_ok_ids = _load_imputation_performance_ids()

    freq_pivot = freq_pivot.rename(columns={"Inversion": "OrigID"})
    merged = df.merge(freq_pivot, how="left", on="OrigID")

    for col in freq_columns:
        if col not in merged.columns:
            merged[col] = pd.NA

    merged = merged.copy()
    freq_cols_existing = [c for c in freq_columns if c in merged.columns]

    if not imputation_ok_ids:
        for col in freq_cols_existing:
            merged[col] = pd.NA
        return merged

    valid_mask = merged["OrigID"].isin(imputation_ok_ids)
    for col in freq_cols_existing:
        merged.loc[~valid_mask, col] = pd.NA

    return merged


def _load_inversion_catalog() -> pd.DataFrame:
    if not INV_PROPERTIES.exists():
        raise SupplementaryTablesError(f"Inversion properties TSV not found: {INV_PROPERTIES}")

    df = pd.read_csv(INV_PROPERTIES, sep="\t", dtype=str, low_memory=False)
    keepable = [c for c in df.columns if str(c).strip()]
    df = df.loc[:, keepable]

    missing = [col for col in INV_COLUMNS_KEEP if col not in df.columns]
    if missing:
        raise SupplementaryTablesError(
            "Inversion properties TSV is missing required columns: " + ", ".join(missing)
        )

    analysed = _load_exclusion_reasons()
    required_selection = {"OrigID", "analysed", "consensus_recurrence"}
    missing_selection = required_selection - set(analysed.columns)
    if missing_selection:
        raise SupplementaryTablesError(
            "Inversion selection table is missing columns: "
            + ", ".join(sorted(missing_selection))
        )
    analysed = analysed.loc[
        analysed["analysed"].astype(str).str.lower().eq("yes"),
        ["OrigID", "consensus_recurrence"],
    ].copy()
    if len(analysed) != 93 or analysed["OrigID"].duplicated().any():
        raise SupplementaryTablesError(
            f"Expected 93 unique consensus-classified inversion IDs; observed {len(analysed)} rows."
        )

    df = df[INV_COLUMNS_KEEP].copy().merge(
        analysed, on="OrigID", how="inner", validate="one_to_one"
    )
    counts = df["consensus_recurrence"].value_counts().to_dict()
    if counts != {"single": 61, "recurrent": 32}:
        raise SupplementaryTablesError(
            f"Expected 61 single-event and 32 recurrent inversions; observed {counts}."
        )
    df = df.drop(columns="consensus_recurrence")
    df = _merge_population_metrics(df)
    df = df.rename(columns=INV_RENAME_MAP)
    return _prune_columns(df, INVERSION_COLUMN_DEFS, "Inversion catalog")


def _load_gene_conservation() -> pd.DataFrame:
    per_gene = _load_tsv(GENE_PERMUTATION_TSV, "Per-gene CDS permutation")
    joint = _load_tsv(GENE_JOINT_CONTROL_TSV, "Joint CDS permutation control")
    required_per_gene = {
        "gene_name", "transcript_id", "inv_id", "recurrence", "k_direct",
        "k_inverted", "n_seq_classes", "p_direct", "p_inverted", "delta",
        "status",
    }
    required_joint = {
        "gene_name", "inv_id", "recurrence", "k_inverted", "delta",
        "joint_p", "wy_fwer_p", "direct_fdr_q",
    }
    for label, frame, required in (
        ("per-gene CDS permutation", per_gene, required_per_gene),
        ("joint CDS permutation", joint, required_joint),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise SupplementaryTablesError(
                f"{label} table is missing columns: {', '.join(sorted(missing))}"
            )

    per_gene = per_gene.loc[per_gene["status"].eq("OK"), list(required_per_gene)].copy()
    keys = ["gene_name", "inv_id"]
    if per_gene.duplicated(keys).any() or joint.duplicated(keys).any():
        raise SupplementaryTablesError("CDS permutation inputs contain duplicate gene/inversion keys.")
    df = joint.merge(
        per_gene.drop(columns=["recurrence", "k_inverted", "delta"]),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if len(df) != 130 or df["transcript_id"].isna().any():
        raise SupplementaryTablesError(
            f"Expected 130 fully matched CDS permutation tests; observed {len(df)} rows."
        )
    numeric_cols = [
        "k_direct", "k_inverted", "n_seq_classes", "p_direct", "p_inverted",
        "delta", "joint_p", "wy_fwer_p", "direct_fdr_q",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not ((df["k_direct"] >= 4) & (df["k_inverted"] >= 4)).all():
        raise SupplementaryTablesError("The CDS permutation table contains a gene with fewer than four haplotypes in an orientation.")
    n_significant = int((df["direct_fdr_q"] < 0.05).sum())
    if n_significant != 13:
        raise SupplementaryTablesError(
            f"Expected 13 genes at direct FDR q < 0.05; observed {n_significant}."
        )

    if not FIXED_DIFF_SUMMARY_TSV.exists():
        raise SupplementaryTablesError(
            f"Fixed differences summary TSV is missing: {FIXED_DIFF_SUMMARY_TSV}"
        )

    fixed_df = pd.read_csv(
        FIXED_DIFF_SUMMARY_TSV, sep="\t", dtype=str, low_memory=False
    )

    key_cols = ["gene_name", "transcript_id", "inv_id"]
    required_fixed_cols = key_cols + ["n_fixed_differences"]
    missing_fixed_cols = [c for c in required_fixed_cols if c not in fixed_df.columns]
    if missing_fixed_cols:
        raise SupplementaryTablesError(
            "Fixed differences summary TSV is missing required columns: "
            + ", ".join(missing_fixed_cols)
        )

    duplicate_keys = fixed_df.duplicated(subset=key_cols, keep=False)
    if duplicate_keys.any():
        dup_rows = (
            fixed_df.loc[duplicate_keys, key_cols]
            .drop_duplicates()
            .sort_values(key_cols, kind="mergesort")
        )
        raise SupplementaryTablesError(
            "Fixed differences summary contains duplicate gene/transcript/inversion combinations:\n"
            + dup_rows.to_csv(index=False)
        )

    fixed_df = fixed_df[required_fixed_cols].copy()
    fixed_df["n_fixed_differences"] = pd.to_numeric(
        fixed_df["n_fixed_differences"], errors="coerce"
    ).astype("Int64")

    df = df.merge(fixed_df, how="left", on=key_cols)

    def orientation(row: pd.Series) -> str:
        delta = row.get("delta")
        if pd.isna(delta):
            return "Unknown"
        if delta > 0:
            return "Inverted"
        if delta < 0:
            return "Direct"
        return "Tie"

    df["Orientation more conserved"] = df.apply(orientation, axis=1)

    rename_map = {
        "gene_name": "Gene",
        "transcript_id": "Transcript",
        "inv_id": "Inversion ID",
        "recurrence": "Recurrence class",
        "k_direct": "Direct haplotypes",
        "k_inverted": "Inverted haplotypes",
        "n_seq_classes": "Sequence classes",
        "p_direct": "Direct identical pair proportion",
        "p_inverted": "Inverted identical pair proportion",
        "delta": "Δ (inverted − direct)",
        "joint_p": "Permutation p-value",
        "wy_fwer_p": "Westfall-Young FWER p-value",
        "direct_fdr_q": "Direct FDR q-value",
        "n_fixed_differences": "Fixed CDS differences",
    }

    df = df.rename(columns=rename_map)
    df = _prune_columns(df, GENE_CONSERVATION_COLUMN_DEFS, "CDS conservation genes")
    df = df.sort_values("Direct FDR q-value", kind="mergesort").reset_index(drop=True)
    return df


def _load_simple_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SupplementaryTablesError(f"Required TSV not found: {path}")
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def _clean_phewas_df(
    df: pd.DataFrame, sheet_name: str, column_defs: Dict[str, str]
) -> pd.DataFrame:
    if "P_LRT_Overall" not in df.columns:
        raise SupplementaryTablesError(
            f"{sheet_name} lacks the canonical primary association p-value P_LRT_Overall."
        )
    canonical = pd.to_numeric(df["P_LRT_Overall"], errors="coerce")
    for collision_col in ("P_Value_x", "P_Value_y", "P_Value"):
        if collision_col not in df.columns:
            continue
        candidate = pd.to_numeric(df[collision_col], errors="coerce")
        comparable = canonical.notna() & candidate.notna()
        mismatch = comparable & ~canonical.eq(candidate)
        if mismatch.any():
            first = mismatch[mismatch].index[0]
            raise SupplementaryTablesError(
                f"{sheet_name} has conflicting primary p-values at row {first}: "
                f"P_LRT_Overall={canonical.loc[first]}, {collision_col}={candidate.loc[first]}."
            )
    df = df.drop(
        columns=[c for c in ("P_Value_x", "P_Value_y", "P_Value") if c in df.columns]
    ).copy()
    # The main PheWAS table presents this canonical statistic under a concise
    # human-facing label, while the tagging-SNP table also retains the original
    # pipeline field and its associated validity metadata.
    df["P_Value_unadjusted"] = df["P_LRT_Overall"]

    if "Q_GLOBAL" in df.columns and "BH_P_GLOBAL" not in df.columns:
        df = df.rename(columns={"Q_GLOBAL": "BH_P_GLOBAL"})

    if "P_Source" in df.columns and "P_Source_x" not in df.columns:
        df = df.rename(columns={"P_Source": "P_Source_x"})

    return _prune_columns(df, column_defs, sheet_name)


def _load_phewas_results() -> pd.DataFrame:
    df = _load_simple_tsv(PHEWAS_RESULTS)
    cleaned = _clean_phewas_df(df, "PheWAS results", PHEWAS_COLUMN_DEFS)
    n_inversions = cleaned["Inversion"].nunique(dropna=True)
    n_phenotypes = cleaned["Phenotype"].nunique(dropna=True)
    if len(cleaned) != 7_630 or n_inversions != 7 or n_phenotypes != 1_090:
        raise SupplementaryTablesError(
            "PheWAS results must contain the complete 7 x 1,090 test grid "
            f"(7,630 rows); observed rows={len(cleaned)}, inversions={n_inversions}, phenotypes={n_phenotypes}."
        )
    q_values = pd.to_numeric(cleaned["BH_P_GLOBAL"], errors="coerce")
    if int((q_values < 0.05).sum()) != 39:
        raise SupplementaryTablesError(
            f"Expected 39 PheWAS tests at global FDR q < 0.05; observed {int((q_values < 0.05).sum())}."
        )
    if cleaned.duplicated(["Phenotype", "Inversion"]).any():
        raise SupplementaryTablesError("PheWAS results contain duplicate phenotype/inversion tests.")
    return cleaned


def _load_within_ancestry_phewas() -> pd.DataFrame:
    df = _load_simple_tsv(WITHIN_ANCESTRY_PHEWAS_RESULTS)
    df = _prune_columns(
        df,
        WITHIN_ANCESTRY_PHEWAS_COLUMN_DEFS,
        "Within-ancestry PC PheWAS",
    )
    evaluable = df["evaluable"].astype(str).str.lower().eq("true")
    if len(df) != 234 or int(evaluable.sum()) != 187:
        raise SupplementaryTablesError(
            f"Within-ancestry sensitivity table must contain 234 rows and 187 evaluable comparisons; observed {len(df)} and {int(evaluable.sum())}."
        )
    return df


def _load_categories() -> pd.DataFrame:
    df = _load_simple_tsv(CATEGORIES_RESULTS)
    columns_to_drop = ["Z_Cap", "Dropped", "Method", "Shrinkage", "Lambda"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    rename_map = {
        "K_Total": "Phenotypes in category",
        "K_GBJ": "Phenotypes included in GBJ",
        "T_GLS": "GLS test statistic",
        "K_GLS": "Phenotypes included in GLS",
        "P_GLS": "P_GLS",
        "Q_GLS": "BH_P_GLS",
        "Q_GBJ": "BH_P_GBJ",
    }
    df = df.rename(columns=rename_map)
    return _prune_columns(df, CATEGORY_COLUMN_DEFS, "Phenotype categories")


def _load_phewas_tagging() -> pd.DataFrame:
    if PHEWAS_TAGGING_RESULTS.exists():
        df = _load_simple_tsv(PHEWAS_TAGGING_RESULTS)
        return _clean_phewas_df(df, "17q21 tagging PheWAS", TAG_PHEWAS_COLUMN_DEFS)

    raise SupplementaryTablesError(
        "PheWAS tagging results were not found in the data directory. Please add all_pop_phewas_tag.tsv."
    )


def _load_imputation_results() -> pd.DataFrame:
    df = _load_simple_tsv(IMPUTATION_RESULTS)
    # Ensure we have a single OrigID column for merging
    if "OrigID" in df.columns:
        if "id" in df.columns:
            df = df.drop(columns=["id"])
    elif "id" in df.columns:
        df = df.rename(columns={"id": "OrigID"})
    else:
        raise SupplementaryTablesError("Imputation results are missing 'OrigID' or 'id' column.")

    # Rename remaining columns to match definitions
    df = df.rename(columns={"best_n_components": "n_components", "model_p_value": "p_value"})

    # Remove unnamed columns (Column 6 and Column 9)
    columns_to_drop = ["Column 6", "Column 9"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    inv_properties = _load_simple_tsv(INV_PROPERTIES)
    required_cols = {"OrigID", "Chromosome", "Start", "End", "0_single_1_recur_consensus"}
    missing_cols = required_cols - set(inv_properties.columns)
    if missing_cols:
        raise SupplementaryTablesError(
            f"Missing required columns in inv_properties.tsv: {', '.join(sorted(missing_cols))}"
        )

    inv_properties = inv_properties[list(required_cols)].copy()
    inv_properties["0_single_1_recur_consensus"] = inv_properties["0_single_1_recur_consensus"].str.strip()
    inv_properties = inv_properties[inv_properties["0_single_1_recur_consensus"].isin(["0", "1"])]

    inv_properties["Start"] = pd.to_numeric(inv_properties["Start"], errors="coerce")
    inv_properties["End"] = pd.to_numeric(inv_properties["End"], errors="coerce")
    inv_properties = inv_properties.dropna(subset=["Start", "End"])
    inv_properties["Start"] = inv_properties["Start"].astype(int)
    inv_properties["End"] = inv_properties["End"].astype(int)

    inv_properties["Inversion"] = inv_properties.apply(
        lambda row: f"{row['Chromosome']}:{row['Start']}-{row['End']}", axis=1
    )

    df = df.merge(inv_properties[["OrigID", "Inversion"]], on="OrigID", how="inner")

    if "Use" not in df.columns:
        r2 = pd.to_numeric(df.get("unbiased_pearson_r2"), errors="coerce")
        q_values = pd.to_numeric(df.get("p_fdr_bh"), errors="coerce")
        use_flag = (r2 > 0.5) & (q_values < 0.05)
        use_flag = use_flag.astype("boolean")
        df["Use"] = use_flag.mask(r2.isna() | q_values.isna(), pd.NA)

    df = _add_population_allele_frequencies(df)
    df = _prune_columns(df, IMPUTATION_COLUMN_DEFS, "Imputation results")
    r2 = pd.to_numeric(df["unbiased_pearson_r2"], errors="coerce")
    use = df["Use"].astype("boolean")
    if len(df) != 75 or int((r2 > 0.5).sum()) != 12 or int(use.fillna(False).sum()) != 11:
        raise SupplementaryTablesError(
            "Imputation table must contain 75 fitted models, 12 with cross-validated r2 > 0.5, and 11 passing both quality criteria."
        )
    return df


def _load_best_tagging_snps() -> pd.DataFrame:
    df = _load_tsv(BEST_TAGGING_RESULTS, "Best tagging SNPs")
    # Rename uppercase 'S' to lowercase 's' to match the column definition schema
    if "S" in df.columns and "s" not in df.columns:
        df = df.rename(columns={"S": "s"})

    if "q_value" in df.columns and "bh_p_value" not in df.columns:
        df = df.rename(columns={"q_value": "bh_p_value"})

    df = df.copy()
    df["hg37_coordinate"] = _coalesce_coordinate(
        df,
        existing_col="hg37",
        chrom_col="chromosome_hg37",
        pos_col="position_hg37",
    )
    df["hg38_coordinate"] = _coalesce_coordinate(
        df,
        existing_col="hg38",
        chrom_col="chromosome_hg38",
        pos_col="position_hg38",
    )
    return _prune_columns(df, BEST_TAGGING_COLUMN_DEFS, "Best tagging SNPs")

def _load_paml_results() -> pd.DataFrame:
    """Load and harmonize PAML output for the dN/dS summary table, using winner-level summaries."""
    df = _load_simple_tsv(PAML_RESULTS)

    rename_map: Dict[str, str] = {}

    if "overall_p_value" in df.columns and "cmc_p_value" not in df.columns:
        rename_map["overall_p_value"] = "cmc_p_value"
    if "cmc_q_value" in df.columns and "cmc_bh_p_value" not in df.columns:
        rename_map["cmc_q_value"] = "cmc_bh_p_value"
    if "overall_q_value" in df.columns and "cmc_bh_p_value" not in df.columns:
        rename_map["overall_q_value"] = "cmc_bh_p_value"
    if "overall_lrt_stat" in df.columns and "cmc_lrt_stat" not in df.columns:
        rename_map["overall_lrt_stat"] = "cmc_lrt_stat"
    if "overall_h1_lnl" in df.columns and "cmc_lnl_h1" not in df.columns:
        rename_map["overall_h1_lnl"] = "cmc_lnl_h1"
    if "overall_h0_lnl" in df.columns and "cmc_lnl_h0" not in df.columns:
        rename_map["overall_h0_lnl"] = "cmc_lnl_h0"

    if "winner_p0" in df.columns and "cmc_p0" not in df.columns:
        rename_map["winner_p0"] = "cmc_p0"
    if "winner_p1" in df.columns and "cmc_p1" not in df.columns:
        rename_map["winner_p1"] = "cmc_p1"
    if "winner_p2" in df.columns and "cmc_p2" not in df.columns:
        rename_map["winner_p2"] = "cmc_p2"
    if "winner_omega0" in df.columns and "cmc_omega0" not in df.columns:
        rename_map["winner_omega0"] = "cmc_omega0"
    if "winner_omega2_direct" in df.columns and "cmc_omega2_direct" not in df.columns:
        rename_map["winner_omega2_direct"] = "cmc_omega2_direct"
    if "winner_omega2_inverted" in df.columns and "cmc_omega2_inverted" not in df.columns:
        rename_map["winner_omega2_inverted"] = "cmc_omega2_inverted"
    if "winner_kappa" in df.columns and "cmc_kappa" not in df.columns:
        rename_map["winner_kappa"] = "cmc_kappa"

    if rename_map:
        df = df.rename(columns=rename_map)

    def _status_priority(value: Optional[str]) -> int:
        """Map textual run status to a numeric priority used to select the winning run."""
        if value == "success":
            return 2
        if value == "partial_success":
            return 1
        return 0

    def _choose_winner_run(row: pd.Series) -> int:
        """Choose the winning run index based on the winner seed suffix."""
        # Prioritize the run that actually generated the winning model parameters.
        # This fixes issues where 'status' is success for both runs, but one run
        # failed to produce metadata (taxa_used) or had convergence warnings hidden
        # in the reason field.
        h1_seed = str(row.get("h1_winner_seed", ""))
        if h1_seed.endswith("run_2"):
            return 2
        if h1_seed.endswith("run_1"):
            return 1

        h0_seed = str(row.get("h0_winner_seed", ""))
        if h0_seed.endswith("run_2"):
            return 2
        if h0_seed.endswith("run_1"):
            return 1

        # Fallback to status priority if seed information is missing
        status_run_1 = row.get("status_run_1")
        status_run_2 = row.get("status_run_2")
        priority_run_1 = _status_priority(status_run_1)
        priority_run_2 = _status_priority(status_run_2)
        if priority_run_2 > priority_run_1:
            return 2
        return 1

    winner_run: Optional[pd.Series] = None

    if "status_run_1" in df.columns and "status_run_2" in df.columns:
        winner_run = df.apply(_choose_winner_run, axis=1)

        if "status" not in df.columns:
            df["status"] = df["status_run_1"]
            df.loc[winner_run == 2, "status"] = df.loc[winner_run == 2, "status_run_2"]

    if winner_run is None and {
        "n_leaves_pruned_run_1",
        "n_leaves_pruned_run_2",
        "taxa_used_run_1",
        "taxa_used_run_2",
    }.issubset(df.columns):
        # Compute the winning run even if status is already present so we can
        # propagate metadata columns consistently.
        winner_run = df.apply(_choose_winner_run, axis=1)

    if (
        winner_run is not None
        and "n_leaves_pruned" not in df.columns
        and "n_leaves_pruned_run_1" in df.columns
        and "n_leaves_pruned_run_2" in df.columns
    ):
        df["n_leaves_pruned"] = df["n_leaves_pruned_run_1"]
        df.loc[winner_run == 2, "n_leaves_pruned"] = df.loc[winner_run == 2, "n_leaves_pruned_run_2"]

    if (
        winner_run is not None
        and "taxa_used" not in df.columns
        and "taxa_used_run_1" in df.columns
        and "taxa_used_run_2" in df.columns
    ):
        df["taxa_used"] = df["taxa_used_run_1"]
        df.loc[winner_run == 2, "taxa_used"] = df.loc[winner_run == 2, "taxa_used_run_2"]

    if "status" not in df.columns:
        raise SupplementaryTablesError("PAML results file is missing status information required for the summary table.")

    if "region" in df.columns:
        df["region"] = df["region"].str.replace(
            r"^([^_]+)_([^_]+)_([^_]+)$",
            r"\1:\2-\3",
            regex=True,
        )

    df = _prune_columns(df, PAML_COLUMN_DEFS, "dN/dS (ω) results")
    p_values = pd.to_numeric(df["cmc_p_value"], errors="coerce")
    q_values = pd.to_numeric(df["cmc_bh_p_value"], errors="coerce")
    if len(df) != 206 or int((p_values < 0.05).sum()) != 5 or int((q_values < 0.05).sum()) != 0:
        raise SupplementaryTablesError(
            "PAML table must contain 206 genes, five nominal p < 0.05 results, and no BH-significant results."
        )
    if {"region", "gene"}.issubset(df.columns):
        df = df.sort_values(["region", "gene"], kind="mergesort")
    return df.reset_index(drop=True)


def _load_simulation_table(path: Path) -> pd.DataFrame:
    df = _load_simple_tsv(path)
    return _prune_columns(df, SIMULATION_COLUMN_DEFS, path.name)

def build_workbook(output_path: Path) -> None:
    sheet_infos: List[SheetInfo] = []
    sheet_frames: List[pd.DataFrame] = []
    _LOCUS_MAP = _canonical_locus_map()

    def _finalize_frame_for_output(df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with missing values filled for display.

        Supplementary tables should not contain empty cells when the source
        data are missing. Replacing blank entries with the string ``"NA"``
        makes the absence of a value explicit in the exported workbook.
        """

        finalized = df.copy()
        finalized.replace(to_replace=r"^\s*$", value=pd.NA, regex=True, inplace=True)
        # Modern pandas raises (instead of silently upcasting) when filling a numeric column
        # with a non-numeric sentinel, so cast to object first; then every column accepts "NA".
        finalized = finalized.astype(object)
        finalized.fillna("NA", inplace=True)
        return finalized

    def register(sheet: SheetInfo) -> None:
        sheet_infos.append(sheet)
        print(f"Preparing sheet: {sheet.name}")
        df = sheet.loader()
        df = _add_canonical_inversion_column(df, _LOCUS_MAP)
        sheet_frames.append(_finalize_frame_for_output(df))

    register(
        SheetInfo(
            name="Old recurrent events",
            description="Parameters used in simulations under different scenarios of old recurrent inversion events. Simulations were generated using a structured coalescent framework (Methods). The three inversion events are set to emerge at 500, 250, 100 thousand years ago. Six inversion frequencies (1%, 2%, 5%, 10%, 25%, and 50%) are considered.  Three recombination rates, including zero, 1e-8, and 1e-6 per generation per base pair are simulated. Gene flow is set as 1e-8 per generation per chromosome only between groups of haplotypes in the same orientations.",
            column_defs=SIMULATION_COLUMN_DEFS,
            loader=lambda: _load_simulation_table(TABLE_S1),
        )
    )

    register(
        SheetInfo(
            name="Young recurrent events",
            description="Parameters used in simulations under different scenarios of young recurrent inversion events. Simulations were generated using a structured coalescent framework (Methods). The three inversion events are set to emerge at 250, 100, 50 thousand years ago. Six inversion frequencies (1%, 2%, 5%, 10%, 25%, and 50%) are considered.  Three recombination rates, including zero, 1e-8, and 1e-6 per generation per base pair are simulated. Gene flow is set as 1e-8 per generation per chromosome only between groups of haplotypes in the same orientations.",
            column_defs=SIMULATION_COLUMN_DEFS,
            loader=lambda: _load_simulation_table(TABLE_S2),
        )
    )

    register(
        SheetInfo(
            name="Recent recurrent events",
            description="Parameters used in simulations under different scenarios of recent recurrent inversion events. Simulations were generated using a structured coalescent framework (Methods). The three inversion events are set to emerge at 100, 50, 25 thousand years ago. Six inversion frequencies (1%, 2%, 5%, 10%, 25%, and 50%) are considered.  Three recombination rates, including zero, 1e-8, and 1e-6 per generation per base pair are simulated. Gene flow is set as 1e-8 per generation per chromosome only between groups of haplotypes in the same orientations.",
            column_defs=SIMULATION_COLUMN_DEFS,
            loader=lambda: _load_simulation_table(TABLE_S3),
        )
    )

    register(
        SheetInfo(
            name="Very recent recurrent events",
            description="Parameters used in simulations under different scenarios of very recent recurrent inversion events. Simulations were generated using a structured coalescent framework (Methods). The three inversion events are set to emerge at 50, 25, 10 thousand years ago. Six inversion frequencies (1%, 2%, 5%, 10%, 25%, and 50%) are considered.  Three recombination rates, including zero, 1e-8, and 1e-6 per generation per base pair are simulated. Gene flow is set as 1e-8 per generation per chromosome only between groups of haplotypes in the same orientations.",
            column_defs=SIMULATION_COLUMN_DEFS,
            loader=lambda: _load_simulation_table(TABLE_S4),
        )
    )

    register(
        SheetInfo(
            name="Inversion catalog",
            description=(
                "The 93 balanced human chromosomal inversions analyzed in this study: 61 single-event and 32 recurrent loci. "
                "Inversion calls, coordinates, and recurrence classifications are derived from Porubsky et al. (2022) "
                "using Strand-seq and long-read sequencing on the 1000 Genomes Project panel (GRCh38 coordinates). "
                "Chromosome, Start, End, number recurrent events, Inversion ID, Size (kbp), Inversion allele frequency, "
                "verdictRecurrence_hufsah, and verdictRecurrence_benson columns are sourced directly from Porubsky et al. "
                "(2022). Only loci for which the two recurrence methods agree are included. NA in Hudson's FST, Direct "
                "haplotypes pi, and Inverted haplotypes pi "
                "reflects that these metrics could not be calculated because the region lacked polymorphisms or had too few "
                "haplotypes."
            ),
            column_defs=INVERSION_COLUMN_DEFS,
            loader=_load_inversion_catalog,
        )
    )

    register(
        SheetInfo(
            name="CDS conservation genes",
            description=(
                "Inversion-level permutation analysis of protein-coding gene conservation within inversion loci. The 130 genes "
                "have at least four haplotypes in each orientation. Orientation labels were shuffled once per inversion and the "
                "same assignment was applied to every gene at that locus, preserving within-inversion dependence. Tests quantify "
                "differences in the proportion of identical coding-sequence pairs between inverted and direct haplotypes."
            ),
            column_defs=GENE_CONSERVATION_COLUMN_DEFS,
            loader=_load_gene_conservation,
        )
    )

    register(
        SheetInfo(
            name="dN/dS (ω) results",
            description=(
                "Results of the dN/dS (ω) analysis testing for genes with significantly different selective regimes between "
                "direct and inverted orientations. Across all columns, NA indicates that the inversion–CDS pair was excluded, "
                "for example due to an uninformative tree topology, insufficient haplotype counts, or PAML run failures."
            ),
            column_defs=PAML_COLUMN_DEFS,
            loader=_load_paml_results,
        )
    )

    register(
        SheetInfo(
            name="Imputation results",
            description=(
                "Performance metrics for the machine learning models (Partial Least Squares regression) used to impute inversion "
                "dosage from flanking SNP genotypes. Models were trained on the 82 phased haplotypes from the reference panel. "
                "For allele frequency columns, values with an imputation accuracy below r^2 0.5 were omitted, so NA marks "
                "instances where the frequency was not reported."
            ),
            column_defs=IMPUTATION_COLUMN_DEFS,
            loader=_load_imputation_results,
        )
    )

    register(
        SheetInfo(
            name="PheWAS results",
            description=(
                "Phenome-wide association study (PheWAS) results linking imputed inversion dosages to electronic health record "
                "(EHR) phenotypes in the NIH All of Us cohort (v8). Association tests were performed using logistic regression "
                "adjusted for age, age squared, genetically inferred sex, and 16 global genetic principal components. For the "
                "main PheWAS analysis, "
                "NA values denote models that failed to converge or produced unstable fits. Interaction tests were only run when "
                "the main result met the FDR threshold, so NA in interaction columns indicates the follow-up test was not "
                "performed. Ancestry-specific analyses were likewise conditioned on main FDR significance; NA in those columns "
                "means the test was skipped or the ancestry stratum had insufficient cases."
            ),
            column_defs=PHEWAS_COLUMN_DEFS,
            loader=_load_phewas_results,
        )
    )

    register(
        SheetInfo(
            name="Within-ancestry PC PheWAS",
            description=(
                "Sensitivity analysis for residual fine-scale population structure. The 37 phenotypes implicated by the pooled "
                "PheWAS were retested against all seven inversions separately within the AFR, AMR, EAS, EUR, MID, and SAS All of "
                "Us genetic-ancestry groups. Each model adjusted for age, age squared, genetically inferred sex, and 16 principal "
                "components fitted de novo within that group. The table compares these estimates with the existing ancestry-"
                "stratified estimates that used the 16 global components. Because the phenotype set was selected from the pooled "
                "findings, selected-set q-values are descriptive sensitivity statistics rather than independent replication tests."
            ),
            column_defs=WITHIN_ANCESTRY_PHEWAS_COLUMN_DEFS,
            loader=_load_within_ancestry_phewas,
        )
    )

    register(
        SheetInfo(
            name="Phenotype categories",
            description=(
                "Aggregate statistical tests assessing whether specific inversions are associated with entire categories of "
                "phenotypes (e.g., 'Dermatologic'). Uses the Generalized Berk-Jones (GBJ) test for set-based significance and "
                "Generalized Least Squares (GLS) for directional effects."
            ),
            column_defs=CATEGORY_COLUMN_DEFS,
            loader=_load_categories,
        )
    )

    register(
        SheetInfo(
            name="Ancient DNA best tagging SNPs",
            description=(
                "Top tagging SNP for each inversion locus, derived from the latest ancient DNA selection analysis of "
                "West Eurasian genomes in the AGES database. Selection statistics (S and P_X) originate from that "
                "ancient DNA summary table, allele frequencies are stratified by direct vs. inverted haplotypes, and "
                "BH-adjusted p-values reflect Benjamini–Hochberg correction across inversions passing quality filters. NA values "
                "appear when a locus was excluded for a reason documented in the exclusion_reasons column."
            ),
            column_defs=BEST_TAGGING_COLUMN_DEFS,
            loader=_load_best_tagging_snps,
        )
    )

    register(
        SheetInfo(
            name="17q21 tagging PheWAS",
            description=(
                "Validation PheWAS for the 17q21 inversion locus using a tagging SNP (rs105255341) instead of imputed dosage. "
                "This ensures that the pleiotropic effects observed (e.g., obesity vs. breast cancer protection) are robust to "
                "the method of genotype determination. NA values indicate models that failed to converge or produced unstable "
                "fits."
            ),
            column_defs=TAG_PHEWAS_COLUMN_DEFS,
            loader=_load_phewas_tagging,
        )
    )

    register(
        SheetInfo(
            name="4-fold diversity concordance",
            description=(
                "Spearman correlations between per-locus orientation differences (inverted minus direct) in "
                "nucleotide diversity measured three ways: across the whole locus, across whole coding sequence, "
                "and restricted to 4-fold degenerate sites. A locus contributes only when both orientations "
                "have 4-fold sites and the locus has a consensus recurrence classification."
            ),
            column_defs=FOURFOLD_CORR_COLUMN_DEFS,
            loader=_load_fourfold_correlations,
        )
    )

    register(
        SheetInfo(
            name="Gene-flux simulation sweep",
            description=(
                "False-positive rate and power of the recurrence classifier across between-orientation gene flux under the "
                "structured-coalescent model. Each row pools 12 inversion-age by recombination-rate cells with 120 simulated "
                "loci per cell, for 11,520 loci overall. Rates carry Wilson 95% intervals. Trend p-values are from two-sided "
                "Cochran-Armitage tests across the four gene-flux levels."
            ),
            column_defs=FLUX_SWEEP_COLUMN_DEFS,
            loader=_load_flux_sweep,
        )
    )

    register(
        SheetInfo(
            name="Coding-site diversity",
            description=(
                "Nucleotide diversity restricted to 4-fold-degenerate sites, and piN/piS at 0-fold and 4-fold sites, "
                "per orientation, for inversion loci containing coding sequence. NA marks a quantity that is not defined "
                "for that locus, usually because it has no coding sequence or no variation in one orientation."
            ),
            column_defs=CODING_DIVERSITY_COLUMN_DEFS,
            loader=_load_coding_site_diversity,
        )
    )

    register(
        SheetInfo(
            name="Divergence between orientations",
            description=(
                "Absolute (Dxy) and net (da) divergence between orientations at each locus, alongside Hudson's FST and the "
                "within-orientation diversities it is built from. FST depends on within-group diversity, so a difference in "
                "FST between recurrence classes need not reflect a difference in divergence."
            ),
            column_defs=DIVERGENCE_COLUMN_DEFS,
            loader=_load_divergence,
        )
    )

    register(
        SheetInfo(
            name="Genomic-architecture controls",
            description=(
                "Conditional-randomization tests of the orientation-by-recurrence diversity interaction and recurrence-class "
                "differences in FST and da: without conditioning; conditioned on inversion length, inverted allele frequency, "
                "local SNP density and CDS density; and additionally conditioned on recombination rate and chromosome-arm position."
            ),
            column_defs=ARCHITECTURE_CONTROLS_COLUMN_DEFS,
            loader=_load_architecture_controls,
        )
    )

    register(
        SheetInfo(
            name="Chimpanzee polarity per locus",
            description=(
                "Manual review of panTro6-GRCh38 alignments deciding, for each locus, which human arrangement is ancestral, "
                "and the diversity recomputed with haplotypes grouped as ancestral or derived. Loci excluded from the figure "
                "or the model carry the reason for exclusion."
            ),
            column_defs=CHIMP_POLARITY_COLUMN_DEFS,
            column_labels={"inv_id": "Original inversion ID"},
            loader=_load_chimp_polarity,
        )
    )

    register(
        SheetInfo(
            name="Ancient DNA, all tagging SNPs",
            description=(
                "Every tagging SNP at the four loci with an AGES selection result, not only the best one, so that the "
                "selection signal can be judged against the whole set rather than a single marker. Coefficients are also "
                "given signed to the inverted allele."
            ),
            column_defs=AGES_ALL_TAGS_COLUMN_DEFS,
            loader=_load_ages_all_tags,
        )
    )

    register(
        SheetInfo(
            name="Imputation external benchmarks",
            description=(
                "Agreement between imputed inversion dosage and genotypes that were never used in training: experimental "
                "calls at 6q24.1 (HsInv0284) and ScoreInvHap at 17q21.31 and 8p23.1."
            ),
            column_defs=IMPUTATION_BENCHMARK_COLUMN_DEFS,
            loader=_load_imputation_benchmarks,
        )
    )

    registered = {sheet.name: (sheet, frame)
                  for sheet, frame in zip(sheet_infos, sheet_frames)}
    expected = set(FINAL_SUPPLEMENTARY_TABLE_ORDER)
    actual = set(registered)
    if actual != expected or len(registered) != len(sheet_infos):
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise SupplementaryTablesError(
            "Supplementary-table registry does not match the manuscript order: "
            f"missing={missing}, unexpected={unexpected}, "
            f"registered={len(sheet_infos)}, unique={len(registered)}."
        )
    ordered = [registered[name] for name in FINAL_SUPPLEMENTARY_TABLE_ORDER]
    sheet_infos = [sheet for sheet, _ in ordered]
    sheet_frames = [frame for _, frame in ordered]

    if len(sheet_infos) != 21:
        raise SupplementaryTablesError(
            f"The revision defines exactly 21 supplementary tables; observed {len(sheet_infos)}."
        )
    for index, (sheet, frame) in enumerate(zip(sheet_infos, sheet_frames), start=1):
        expected_rows = EXPECTED_SUPPLEMENTARY_DATA_ROWS[index - 1]
        if len(frame) != expected_rows:
            raise SupplementaryTablesError(
                f"Table S{index} ({sheet.name}) must contain {expected_rows:,} data rows; "
                f"observed {len(frame):,}."
            )
        if frame.columns.duplicated().any():
            duplicates = sorted(set(frame.columns[frame.columns.duplicated()].astype(str)))
            raise SupplementaryTablesError(
                f"Table S{index} ({sheet.name}) has duplicate source columns: {duplicates}."
            )
        printed = [
            sheet.column_labels.get(column, _pretty_label(column))
            for column in frame.columns
        ]
        duplicate_labels = sorted({label for label in printed if printed.count(label) > 1})
        if duplicate_labels:
            raise SupplementaryTablesError(
                f"Table S{index} ({sheet.name}) has duplicate printed headers: {duplicate_labels}."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        readme_ws = workbook.add_worksheet("Information")

        def base_format(**kwargs: object):
            return workbook.add_format({"bg_color": "#FFFFFF", **kwargs})

        header_fmt = base_format(bold=True, font_size=14, bottom=1)
        desc_fmt = base_format(italic=True, text_wrap=True)
        col_name_fmt = base_format(bold=True, text_wrap=True, bg_color="#EEEEEE")
        col_def_fmt = base_format(text_wrap=True)

        title_rich_fmt = base_format(bold=True)
        title_cell_fmt = base_format(text_wrap=True, valign="top", align="left")
        table_header_fmt = base_format(bold=True)
        default_cell_fmt = base_format()

        readme_ws.set_column(0, 0, 32, default_cell_fmt)
        readme_ws.set_column(1, 1, 120, default_cell_fmt)

        row = 0
        for i, sheet_info in enumerate(sheet_infos, start=1):
            readme_ws.write(row, 0, f"Table S{i}: {sheet_info.name}", header_fmt)
            row += 1

            readme_ws.merge_range(row, 0, row, 1, sheet_info.description, desc_fmt)
            row += 1

            readme_ws.write(row, 0, "Column", col_name_fmt)
            readme_ws.write(row, 1, "Definition", col_name_fmt)
            row += 1

            for col_name, definition in sheet_info.column_defs.items():
                readme_ws.write(
                    row, 0,
                    sheet_info.column_labels.get(col_name, _pretty_label(col_name)),
                    col_name_fmt,
                )
                readme_ws.write(row, 1, definition, col_def_fmt)
                row += 1

            row += 2

        for i, (sheet_info, df) in enumerate(zip(sheet_infos, sheet_frames), start=1):
            sheet_name = f"Table S{i}"
            df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=2, header=False)

            worksheet = writer.sheets[sheet_name]
            num_cols = max(len(df.columns), 1)
            worksheet.set_column(0, num_cols - 1, None, default_cell_fmt)

            if num_cols > 1:
                worksheet.merge_range(0, 0, 0, num_cols - 1, "", title_cell_fmt)

            worksheet.write_rich_string(
                0,
                0,
                title_rich_fmt,
                f"Table S{i}. {sheet_info.name}.",
                f" {sheet_info.description}",
                title_cell_fmt,
            )

            for col_idx, col_name in enumerate(df.columns):
                worksheet.write(
                    1, col_idx,
                    sheet_info.column_labels.get(col_name, _pretty_label(col_name)),
                    table_header_fmt,
                )

    print(f"Supplementary tables written to {output_path}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate supplementary tables workbook.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination path for the Excel workbook (default: web/figures-site/public/downloads).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    try:
        build_workbook(args.output.resolve())
    except SupplementaryTablesError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive guardrail
        print(f"ERROR: Unexpected failure while generating tables: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
