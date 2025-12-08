#!/usr/bin/env python3
"""Generate the supplementary tables Excel workbook.

This utility orchestrates the steps required to build the manuscript
supplementary tables. It performs the following operations:

1. Curates the inversion catalog from ``data/inv_properties.tsv``.
2. Ensures the CDS conservation test results are produced by running the
   ``stats/per_gene_cds_differences_jackknife.py`` pipeline and filters the
   BH FDR results (q < 0.05).
3. Aggregates the published TSV artefacts into a single Excel workbook with a
   "Read me" worksheet that explains each tab.

The resulting ``supplementary_tables.xlsx`` file is saved under the Next.js
public directory so the web site can link to it directly.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import warnings
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
NEXT_PUBLIC_DIR = REPO_ROOT / "web" / "figures-site" / "public"
DEFAULT_OUTPUT = NEXT_PUBLIC_DIR / "downloads" / "supplementary_tables.xlsx"

GITHUB_TOKEN_ENVS = ("GITHUB_TOKEN", "GH_TOKEN")
GITHUB_REPO_ENV = "GITHUB_REPOSITORY"
DEFAULT_REPO_SLUG = "SauersML/ferromic"

BEST_TAGGING_WORKFLOW = "batch_best_tagging_snps.yml"
BEST_TAGGING_ARTIFACT = "best-tagging-snps-results"
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
            "The estimated number of independent inversion recurrence events based on coalescent simulations.",
        ),
        ("Inversion ID", "The unique identifier assigned to the inversion (format: chr-start-inv-id)."),
        ("Size (kbp)", "The length of the inverted segment in kilobase pairs."),
        (
            "Inversion allele frequency",
            "The frequency of the inverted allele observed in the phased reference panel (n=88 haplotypes).",
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
        ("SE(Δ)", "Standard error of the difference (Δ), calculated via leave-one-haplotype-out jackknife."),
        ("p-value", "Nominal p-value testing the null hypothesis that conservation is equal between orientations."),
        ("BH p-value", "Benjamini-Hochberg adjusted p-value controlling the false discovery rate (FDR)."),
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
        ("Inversion", "The Inversion ID."),
        ("n_components", "Number of PLS components selected via cross-validation."),
        (
            "unbiased_pearson_r2",
            "Pearson r² correlation between imputed and true dosage in held-out cross-validation folds.",
        ),
        ("p_value", "P-value comparing the trained model against a null intercept-only model."),
        ("p_fdr_bh", "FDR adjusted p-value."),
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

GENE_RESULTS_SCRIPT = REPO_ROOT / "stats" / "per_gene_cds_differences_jackknife.py"
GENE_RESULTS_TSV = DATA_DIR / "gene_inversion_direct_inverted.tsv"
CDS_SUMMARY_TSV = DATA_DIR / "cds_identical_proportions.tsv"
FIXED_DIFF_SUMMARY_TSV = DATA_DIR / "fixed_diff_summary.tsv"

PHEWAS_RESULTS = DATA_DIR / "phewas_results.tsv"
PHEWAS_TAGGING_RESULTS = DATA_DIR / "all_pop_phewas_tag.tsv"
CATEGORIES_RESULTS_CANDIDATES = (
    DATA_DIR / "categories.tsv",
    DATA_DIR / "phewas v2 - categories.tsv",
)
IMPUTATION_RESULTS = DATA_DIR / "imputation_results.tsv"
INV_PROPERTIES = DATA_DIR / "inv_properties.tsv"
POPULATION_METRICS = DATA_DIR / "output.csv"
BEST_TAGGING_RESULTS = DATA_DIR / BEST_TAGGING_FILENAME
PAML_RESULTS = DATA_DIR / "GRAND_PAML_RESULTS.tsv"

TABLE_S1 = DATA_DIR / "tables.xlsx - Table S1.tsv"
TABLE_S2 = DATA_DIR / "tables.xlsx - Table S2.tsv"
TABLE_S3 = DATA_DIR / "tables.xlsx - Table S3.tsv"
TABLE_S4 = DATA_DIR / "tables.xlsx - Table S4.tsv"


@dataclass
class SheetInfo:
    name: str
    description: str
    column_defs: Dict[str, str]
    loader: Callable[[], pd.DataFrame]


class SupplementaryTablesError(RuntimeError):
    """Raised for unrecoverable supplementary table failures."""


def _github_headers(token: Optional[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_json(url: str, token: Optional[str], params: Optional[Dict[str, str]] = None) -> dict:
    if params:
        url = f"{url}?{urlencode(params)}"

    req = Request(url, headers=_github_headers(token))
    try:
        with urlopen(req) as response:
            return json.load(response)
    except HTTPError as exc:  # pragma: no cover - network failure edge case
        raise SupplementaryTablesError(
            f"GitHub API request failed for {url} (HTTP {exc.code})."
        ) from exc
    except URLError as exc:  # pragma: no cover - network failure edge case
        raise SupplementaryTablesError(f"Unable to reach GitHub API at {url}: {exc.reason}.") from exc


def _download_github_artifact(
    *,
    workflow_file: str,
    artifact_name: str,
    expected_member: str,
    destination: Path,
) -> Path:
    token = next((os.environ.get(env) for env in GITHUB_TOKEN_ENVS if os.environ.get(env)), None)
    repo = os.environ.get(GITHUB_REPO_ENV) or DEFAULT_REPO_SLUG

    runs_url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/runs"
    runs_json = _github_json(
        runs_url,
        token,
        params={"status": "success", "per_page": 1, "exclude_pull_requests": "true"},
    )
    runs = runs_json.get("workflow_runs", [])
    if not runs:
        raise SupplementaryTablesError(f"No successful runs found for workflow {workflow_file} in {repo}.")

    run_id = runs[0].get("id")
    artifacts_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts"
    artifacts = _github_json(artifacts_url, token, params={"per_page": 100}).get("artifacts", [])
    artifact = next((a for a in artifacts if a.get("name") == artifact_name), None)
    if artifact is None:
        raise SupplementaryTablesError(
            f"Artifact '{artifact_name}' not found in workflow run {run_id} for {repo}."
        )

    if token:
        download_url = artifact.get("archive_download_url")
        req = Request(download_url, headers=_github_headers(token))
    else:
        # Public unauthenticated fallback via nightly.link
        download_url = f"https://nightly.link/{repo}/actions/runs/{run_id}/{artifact_name}.zip"
        req = Request(download_url)

    try:
        with urlopen(req) as response:
            archive_bytes = response.read()
    except HTTPError as exc:  # pragma: no cover - network failure edge case
        raise SupplementaryTablesError(
            f"Failed to download artifact {artifact_name} (HTTP {exc.code})."
        ) from exc
    except URLError as exc:  # pragma: no cover - network failure edge case
        raise SupplementaryTablesError(
            f"Unable to download artifact {artifact_name} from GitHub: {exc.reason}."
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        member = next((name for name in zf.namelist() if name.endswith(expected_member)), None)
        if member is None:
            raise SupplementaryTablesError(
                f"Expected file {expected_member} not found inside artifact {artifact_name}."
            )

        with zf.open(member) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    return destination


def _prune_columns(df: pd.DataFrame, column_defs: Dict[str, str], sheet_name: str) -> pd.DataFrame:
    expected_cols = list(column_defs.keys())
    available_cols = [col for col in expected_cols if col in df.columns]
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        warnings.warn(
            f"Sheet '{sheet_name}' is missing columns: {', '.join(missing)}. "
            "Proceeding with available columns only.",
            RuntimeWarning,
        )

    return df.loc[:, available_cols].copy()


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

    merged = inv_with_keys.merge(
        metrics_trimmed,
        how="left",
        on=["_merge_chr", "_merge_start", "_merge_end"],
        validate="one_to_one",
    )

    helper_cols = [col for col in merged.columns if col.startswith("_merge_")]
    return merged.drop(columns=helper_cols)


def ensure_cds_summary() -> Path:
    """Ensure cds_identical_proportions.tsv exists, generating it if .phy files are available."""
    if CDS_SUMMARY_TSV.exists():
        return CDS_SUMMARY_TSV

    # Check if we have .phy files to run the pipeline
    phy_files = list(REPO_ROOT.glob("*.phy"))
    if len(phy_files) >= 100:  # Arbitrary threshold indicating we have the dataset
        print(f"Found {len(phy_files)} .phy files. Running cds_differences.py to generate summary...")
        try:
            cds_diff_script = REPO_ROOT / "stats" / "cds_differences.py"
            if not cds_diff_script.exists():
                raise SupplementaryTablesError(f"CDS differences script not found: {cds_diff_script}")
            
            # Run cds_differences.py from repo root
            result = subprocess.run(
                [sys.executable, str(cds_diff_script)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            if result.returncode != 0:
                print(f"cds_differences.py stderr:\n{result.stderr}", file=sys.stderr)
                raise SupplementaryTablesError(
                    f"cds_differences.py failed with exit code {result.returncode}"
                )
            
            if not CDS_SUMMARY_TSV.exists():
                raise SupplementaryTablesError(
                    "cds_differences.py completed but did not produce cds_identical_proportions.tsv"
                )
            
            print(f"✅ Generated {CDS_SUMMARY_TSV.name}")
            return CDS_SUMMARY_TSV
            
        except subprocess.TimeoutExpired:
            raise SupplementaryTablesError("cds_differences.py timed out after 1 hour")
        except Exception as e:
            print(f"Failed to run cds_differences.py: {e}", file=sys.stderr)
            raise SupplementaryTablesError(
                "cds_identical_proportions.tsv is missing and could not be generated from local inputs."
            )

    raise SupplementaryTablesError(
        "cds_identical_proportions.tsv is missing. Please add it to the data directory or provide the required inputs "
        "to generate it locally."
    )


def ensure_gene_results() -> Path:
    """Ensure gene_inversion_direct_inverted.tsv exists, generating it if CDS summary is available."""
    if GENE_RESULTS_TSV.exists():
        return GENE_RESULTS_TSV

    # First ensure we have the CDS summary
    cds_summary = ensure_cds_summary()
    
    # Check if we have pairs files to run the per-gene analysis
    pairs_files = list(REPO_ROOT.glob("pairs_CDS__*.tsv"))
    if len(pairs_files) >= 100:  # Threshold indicating we have the dataset
        print(f"Found {len(pairs_files)} pairs files. Running per_gene_cds_differences_jackknife.py...")
        try:
            gene_script = REPO_ROOT / "stats" / "per_gene_cds_differences_jackknife.py"
            if not gene_script.exists():
                raise SupplementaryTablesError(f"Per-gene script not found: {gene_script}")
            
            # Run per_gene_cds_differences_jackknife.py from repo root
            result = subprocess.run(
                [sys.executable, str(gene_script)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            if result.returncode != 0:
                print(f"per_gene_cds_differences_jackknife.py stderr:\n{result.stderr}", file=sys.stderr)
                raise SupplementaryTablesError(
                    f"per_gene_cds_differences_jackknife.py failed with exit code {result.returncode}"
                )
            
            if not GENE_RESULTS_TSV.exists():
                raise SupplementaryTablesError(
                    "per_gene_cds_differences_jackknife.py completed but did not produce gene_inversion_direct_inverted.tsv"
                )
            
            print(f"✅ Generated {GENE_RESULTS_TSV.name}")
            return GENE_RESULTS_TSV
            
        except subprocess.TimeoutExpired:
            raise SupplementaryTablesError("per_gene_cds_differences_jackknife.py timed out after 1 hour")
        except Exception as e:
            print(f"Failed to run per_gene_cds_differences_jackknife.py: {e}", file=sys.stderr)
            raise SupplementaryTablesError(
                "gene_inversion_direct_inverted.tsv is missing and could not be generated from local inputs."
            )

    raise SupplementaryTablesError(
        "gene_inversion_direct_inverted.tsv is missing. Please add it to the data directory or provide the required "
        "inputs to generate it locally."
    )


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

    df = df[INV_COLUMNS_KEEP].copy()
    df = _merge_population_metrics(df)
    df = df.rename(columns=INV_RENAME_MAP)
    return _prune_columns(df, INVERSION_COLUMN_DEFS, "Inversion catalog")


def _load_gene_conservation() -> pd.DataFrame:
    tsv_path = ensure_gene_results()
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, low_memory=False)

    numeric_cols = ["p_direct", "p_inverted", "delta", "se_delta", "p_value", "q_value"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

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
        "p_direct": "Direct identical pair proportion",
        "p_inverted": "Inverted identical pair proportion",
        "delta": "Δ (inverted − direct)",
        "se_delta": "SE(Δ)",
        "p_value": "p-value",
        "q_value": "BH p-value",
        "n_fixed_differences": "Fixed CDS differences",
    }

    df = df.rename(columns=rename_map)
    df = _prune_columns(df, GENE_CONSERVATION_COLUMN_DEFS, "CDS conservation genes")
    df = df.sort_values("BH p-value", kind="mergesort").reset_index(drop=True)
    return df


def _load_simple_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SupplementaryTablesError(f"Required TSV not found: {path}")
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def _clean_phewas_df(
    df: pd.DataFrame, sheet_name: str, column_defs: Dict[str, str]
) -> pd.DataFrame:
    # Check for P_Value_x and P_Value_y columns
    if "P_Value_x" in df.columns and "P_Value_y" in df.columns:
        # Convert to numeric for comparison
        p_x = pd.to_numeric(df["P_Value_x"], errors="coerce")
        p_y = pd.to_numeric(df["P_Value_y"], errors="coerce")

        both_nan = p_x.isna() & p_y.isna()
        both_equal = p_x == p_y
        all_match = (both_nan | both_equal).all()

        if not all_match:
            diff_mask = ~(both_nan | both_equal)
            first_diff_idx = diff_mask.idxmax() if diff_mask.any() else None
            warnings.warn(
                "P_Value_x and P_Value_y columns have different values. "
                f"Using P_Value_x where available. First difference at row {first_diff_idx}: "
                f"P_Value_x={df.loc[first_diff_idx, 'P_Value_x']}, "
                f"P_Value_y={df.loc[first_diff_idx, 'P_Value_y']}",
                RuntimeWarning,
            )
            fill_mask = df["P_Value_x"].isna() & df["P_Value_y"].notna()
            if fill_mask.any():
                df.loc[fill_mask, "P_Value_x"] = df.loc[fill_mask, "P_Value_y"]

        df = df.drop(columns=["P_Value_y"])
        df = df.rename(columns={"P_Value_x": "P_Value_unadjusted"})

    if "P_Value_unadjusted" not in df.columns and "P_Value" in df.columns:
        df = df.rename(columns={"P_Value": "P_Value_unadjusted"})

    if "Q_GLOBAL" in df.columns and "BH_P_GLOBAL" not in df.columns:
        df = df.rename(columns={"Q_GLOBAL": "BH_P_GLOBAL"})

    if "P_Source" in df.columns and "P_Source_x" not in df.columns:
        df = df.rename(columns={"P_Source": "P_Source_x"})

    empty_cols = [
        col for col in df.columns if df[col].isna().all() or (df[col].astype(str).str.strip() == "").all()
    ]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    return _prune_columns(df, column_defs, sheet_name)


def _load_phewas_results() -> pd.DataFrame:
    df = _load_simple_tsv(PHEWAS_RESULTS)
    return _clean_phewas_df(df, "PheWAS results", PHEWAS_COLUMN_DEFS)


def _load_categories() -> pd.DataFrame:
    for candidate in CATEGORIES_RESULTS_CANDIDATES:
        if candidate.exists():
            df = _load_simple_tsv(candidate)
            # Remove Z_Cap and Dropped columns if present
            columns_to_drop = ["Z_Cap", "Dropped", "Method", "Shrinkage", "Lambda"]
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

            # Rename columns for clarity
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
    raise SupplementaryTablesError("Unable to locate categories TSV in the data directory.")


def _load_phewas_tagging() -> pd.DataFrame:
    if PHEWAS_TAGGING_RESULTS.exists():
        df = _load_simple_tsv(PHEWAS_TAGGING_RESULTS)
        return _clean_phewas_df(df, "17q21 tagging PheWAS", TAG_PHEWAS_COLUMN_DEFS)

    raise SupplementaryTablesError(
        "PheWAS tagging results were not found in the data directory. Please add all_pop_phewas_tag.tsv."
    )


def _load_imputation_results() -> pd.DataFrame:
    df = _load_simple_tsv(IMPUTATION_RESULTS)
    # Rename columns to match definitions
    df = df.rename(
        columns={
            "id": "Inversion",
            "best_n_components": "n_components",
            "model_p_value": "p_value",
        }
    )

    # Remove unnamed columns (Column 6 and Column 9)
    columns_to_drop = ["Column 6", "Column 9"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    return _prune_columns(df, IMPUTATION_COLUMN_DEFS, "Imputation results")


def _load_paml_results() -> pd.DataFrame:
    df = _load_simple_tsv(PAML_RESULTS)
    if "status" not in df.columns:
        raise SupplementaryTablesError("PAML results file is missing the 'status' column.")

    df = df[df["status"].isin(["success", "partial_success"])]
    if "region" in df.columns:
        df["region"] = df["region"].str.replace(
            r"^([^_]+)_([^_]+)_([^_]+)$", r"\1:\2-\3", regex=True
        )
    df = _prune_columns(df, PAML_COLUMN_DEFS, "dN/dS (ω) results")
    if {"region", "gene"}.issubset(df.columns):
        df = df.sort_values(["region", "gene"], kind="mergesort")
    return df.reset_index(drop=True)


def _ensure_best_tagging_results() -> Path:
    if BEST_TAGGING_RESULTS.exists():
        return BEST_TAGGING_RESULTS

    print("Best tagging SNP results missing; attempting to download latest artifact ...")
    return _download_github_artifact(
        workflow_file=BEST_TAGGING_WORKFLOW,
        artifact_name=BEST_TAGGING_ARTIFACT,
        expected_member=BEST_TAGGING_FILENAME,
        destination=BEST_TAGGING_RESULTS,
    )


def _load_best_tagging_snps() -> pd.DataFrame:
    path = _ensure_best_tagging_results()
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
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
    if {"region", "gene"}.issubset(df.columns):
        df = df.sort_values(["region", "gene"], kind="mergesort")
    return df.reset_index(drop=True)


def _load_simulation_table(path: Path) -> pd.DataFrame:
    df = _load_simple_tsv(path)
    return _prune_columns(df, SIMULATION_COLUMN_DEFS, path.name)

def build_workbook(output_path: Path) -> None:
    sheet_infos: List[SheetInfo] = []
    sheet_frames: List[pd.DataFrame] = []

    def _finalize_frame_for_output(df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with missing values filled for display.

        Supplementary tables should not contain empty cells when the source
        data are missing. Replacing blank entries with the string ``"NA"``
        makes the absence of a value explicit in the exported workbook.
        """

        finalized = df.copy()
        finalized.replace(to_replace=r"^\s*$", value=pd.NA, regex=True, inplace=True)
        finalized.fillna("NA", inplace=True)
        return finalized

    def register(sheet: SheetInfo) -> None:
        sheet_infos.append(sheet)
        print(f"Preparing sheet: {sheet.name}")
        df = sheet.loader()
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
                "A comprehensive catalog of the 93 balanced human chromosomal inversions analyzed in this study. "
                "Inversion calls, coordinates, and recurrence classifications are derived from Porubsky et al. (2022) "
                "using Strand-seq and long-read sequencing on the 1000 Genomes Project panel (GRCh38 coordinates)."
            ),
            column_defs=INVERSION_COLUMN_DEFS,
            loader=_load_inversion_catalog,
        )
    )

    register(
        SheetInfo(
            name="CDS conservation genes",
            description=(
                "Analysis of protein-coding gene conservation within inversion loci. Tests quantify differences in the "
                "proportion of identical Coding Sequence (CDS) pairs between inverted and direct haplotypes, identifying genes "
                "where the inverted orientation maintains significantly higher (or lower) sequence conservation."
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
                "direct and inverted orientations."
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
                "dosage from flanking SNP genotypes. Models were trained on the 88 phased haplotypes from the reference panel."
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
                "adjusted for age, sex, 16 genetic principal components, and ancestry categories."
            ),
            column_defs=PHEWAS_COLUMN_DEFS,
            loader=_load_phewas_results,
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
                "BH-adjusted p-values reflect Benjamini–Hochberg correction across inversions passing quality filters."
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
                "the method of genotype determination."
            ),
            column_defs=TAG_PHEWAS_COLUMN_DEFS,
            loader=_load_phewas_tagging,
        )
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
                readme_ws.write(row, 0, col_name, col_name_fmt)
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
                worksheet.write(1, col_idx, col_name, table_header_fmt)

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
