"""Build the supplementary tables the revision promises, with plain column names.

The response letter cites a number of "Table S[X]" placeholders and an editorial
note asks whether the column names were ever made readable. This script writes
one clearly-named TSV per promised table, straight from the analysis outputs, so
the tables and the numbers in the text come from the same place.

Tables produced (data/table_response_*.tsv):
  four_fold_pi          pi at 4-fold-degenerate sites vs whole-CDS and
                        whole-locus pi, per inversion            (Reviewer 1)
  pin_pis               piN and piS at 0-fold and 4-fold sites, per inversion,
                        and the paired tests                     (Reviewer 1)
  divergence            Hudson FST, Dxy, within-class pi and net divergence da
                        per inversion, by recurrence class       (Reviewer 1)
  chimp_polarization    per-locus chimpanzee orientation call, whether the locus
                        entered the derived/ancestral diversity comparison, and
                        why not when it did not                  (Reviewer 2.7)
  architecture_controls recurrence effects under covariate adjustment and
                        matching, including recombination rate and genomic
                        compartment                              (Reviewer 3.3)
  phewas_lambda         genomic-control factor per inversion, its bootstrap CI,
                        and how ancestry-informative each inversion is
                                                                 (Reviewer 3.4)
  within_ancestry_meta  every reported association re-estimated inside ancestry
                        groups and meta-analysed                 (Reviewer 3.4)
  gene_span             whole-gene versus CDS haplotype identity, per gene
                                                                 (CDS section)
  population_frequency  imputed inverted-allele frequency by genetic-ancestry
                        group, per inversion                     (Reviewer 2.10)
  imputation_benchmark  imputed dosage against external genotypes
                                                                 (Reviewer 2.3)
  ages_tag_snps         every tagging SNP tested in AGES, per locus
                                                                 (Reviewer 2.9)

Missing inputs are skipped with a note rather than aborting the run.
"""

import os

import numpy as np
import pandas as pd

_STATS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_STATS)
_DATA = os.path.join(_REPO, "data")
_RESULTS = os.path.join(_REPO, "results")

OUT_PREFIX = os.path.join(_DATA, "table_response_")


def _read(name, subdir=_DATA, **kw):
    path = os.path.join(subdir, name)
    if not os.path.exists(path):
        print(f"  [skip] missing {path}")
        return None
    return pd.read_csv(path, sep="\t", **kw)


def _write(df, stem, note):
    if df is None or not len(df):
        return
    path = f"{OUT_PREFIX}{stem}.tsv"
    df.to_csv(path, sep="\t", index=False, na_rep="NA")
    print(f"  wrote {os.path.basename(path)}  ({len(df)} rows, "
          f"{df.shape[1]} columns) -- {note}")


def _inv_label(chrom, start, end):
    c = str(chrom)
    c = c if c.startswith("chr") else f"chr{c}"
    return f"{c}:{int(start):,}-{int(end):,}"


def four_fold():
    df = _read("four_fold_pi_by_inversion.tsv")
    if df is None:
        return
    out = pd.DataFrame({
        "Inversion (GRCh38)": [_inv_label(r.chr, r.region_start, r.region_end)
                               for r in df.itertuples()],
        "Recurrence class": df["recurrence"].map(
            {0: "single-event", 1: "recurrent"}).fillna("not classified"),
        "Coding sequences in locus": df["n_cds"],
        "Coding sequences with 4-fold sites": df["n_cds_with_fourfold"],
        "4-fold sites, direct haplotypes": df["fourfold_sites_direct"],
        "4-fold sites, inverted haplotypes": df["fourfold_sites_inverted"],
        "pi at 4-fold sites, direct": df["pi_fourfold_direct"],
        "pi at 4-fold sites, inverted": df["pi_fourfold_inverted"],
        "pi across whole CDS, direct": df["pi_wholeCDS_direct"],
        "pi across whole CDS, inverted": df["pi_wholeCDS_inverted"],
        "pi across whole inversion locus, direct": df["pi_wholeLocus_direct"],
        "pi across whole inversion locus, inverted": df["pi_wholeLocus_inverted"],
    })
    _write(out, "four_fold_pi",
           "pi at 4-fold-degenerate sites vs whole-CDS and whole-locus pi")


def pin_pis():
    df = _read("pin_pis_by_inversion.tsv")
    if df is not None:
        out = pd.DataFrame({
            "Inversion (GRCh38)": [_inv_label(r.chr, r.region_start, r.region_end)
                                   for r in df.itertuples()],
            "Recurrence class": df["recurrence"].map(
                {0: "single-event", 1: "recurrent"}).fillna("not classified"),
            "Coding sequences used": df["n_cds_used"],
            "0-fold (nonsynonymous) sites, direct": df["zerofold_sites_direct"],
            "0-fold (nonsynonymous) sites, inverted": df["zerofold_sites_inverted"],
            "4-fold (synonymous) sites, direct": df["fourfold_sites_direct"],
            "4-fold (synonymous) sites, inverted": df["fourfold_sites_inverted"],
            "piN (0-fold), direct": df["piN_direct"],
            "piN (0-fold), inverted": df["piN_inverted"],
            "piS (4-fold), direct": df["piS_direct"],
            "piS (4-fold), inverted": df["piS_inverted"],
            "piN/piS, direct": df["piN_piS_direct"],
            "piN/piS, inverted": df["piN_piS_inverted"],
        })
        _write(out, "pin_pis",
               "piN and piS at 0-fold and 4-fold sites, per inversion")

    tests = _read("pin_pis_tests.tsv")
    if tests is not None:
        out = tests.rename(columns={
            "metric": "Quantity", "test": "Test",
            "category": "Recurrence class", "n": "Inversions",
            "median_direct": "Median, direct haplotypes",
            "median_inverted": "Median, inverted haplotypes",
            "statistic": "Test statistic", "p_value": "p-value"})
        _write(out, "pin_pis_tests",
               "paired and between-class tests on piN, piS and piN/piS")


def divergence():
    df = _read("divergence_da_dxy_by_type.tsv")
    if df is None:
        return
    out = pd.DataFrame({
        "Inversion (GRCh38)": [_inv_label(r.chr, r.region_start, r.region_end)
                               for r in df.itertuples()],
        "Recurrence class": df["category"],
        "Hudson FST between orientations": df["hudson_fst_hap_group_0v1"],
        "Absolute divergence Dxy": df["dxy"],
        "pi within direct haplotypes": df.get("hudson_pi_hap_group_0"),
        "pi within inverted haplotypes": df.get("hudson_pi_hap_group_1"),
        "Net divergence da (Dxy - mean within-class pi)": df["da"],
    })
    _write(out, "divergence",
           "FST, Dxy and net divergence da per inversion")


def chimp_polarization():
    df = _read("figure2a_locus_audit.tsv",
               subdir=os.path.join(_RESULTS, "figure2a_repolarized"))
    if df is None:
        return
    reason = {
        "unresolved_chimp_orientation":
            "chimpanzee alignment did not support a confident orientation call",
        "one_orientation_missing_pi":
            "fewer than two haplotypes in one orientation, so pi is undefined "
            "there (this is a property of the sample, not of the polarization)",
        "": "",
    }
    out = pd.DataFrame({
        "Inversion (GRCh38)": [_inv_label(r.chrom, r.start, r.end)
                               for r in df.itertuples()],
        "Inversion ID": df["inv_id"],
        "Recurrence class": df["recurrence"],
        "Chimpanzee orientation call": df["chimp_call"].map(
            {"direct": "GRCh38 allele is ancestral",
             "inverted": "GRCh38 allele is derived",
             "na": "not callable"}).fillna("not callable"),
        "GRCh38 orientation flipped for the analysis": df["flip_ref_polarity"],
        "Shown in the polarized figure": df["included_in_plot"],
        "Used in the ancestral-vs-derived model": df["included_in_model"],
        "Reason for exclusion from the model":
            df["model_exclusion_reason"].fillna("").map(
                lambda v: reason.get(v, v)),
        "pi, ancestral orientation": df["pi_ancestral"],
        "pi, derived orientation": df["pi_derived"],
    })
    _write(out, "chimp_polarization",
           "per-locus chimpanzee orientation call and its use downstream")


def architecture_controls():
    df = _read("recurrence_controls_summary.tsv")
    if df is not None:
        out = df.rename(columns={
            "outcome": "Quantity compared",
            "control": "Control strategy",
            "effect": "Estimate", "ci_lo": "95% CI lower",
            "ci_hi": "95% CI upper", "p": "p-value",
            "n": "Loci used", "n_recur": "Recurrent loci",
            "n_single": "Single-event loci",
            "scale": "Estimate is a ratio or a difference"})
        _write(out, "architecture_controls",
               "recurrence effects under adjustment and matching")

    cov = _read("recurrence_controls_covariates.tsv")
    if cov is not None:
        out = cov.rename(columns={
            "region_id": "Inversion (GRCh38)", "chr_std": "Chromosome",
            "region_start": "Start", "region_end": "End",
            "Recurrence": "Recurrence class", "size_kbp": "Inversion size (kbp)",
            "inv_af": "Inverted allele frequency",
            "snp_density": "Segregating sites per kbp",
            "cds_density": "CDS segments per kbp",
            "recomb_cM_per_Mb": "Recombination rate across locus (cM/Mb)",
            "recomb_cM_per_Mb_flank":
                "Recombination rate in 1 Mb flanks (cM/Mb)",
            "rel_arm_position":
                "Position along chromosome arm (0 = centromere, 1 = telomere)",
            "dist_to_centromere": "Distance to centromere (bp)",
            "pi_direct": "pi, direct haplotypes",
            "pi_inverted": "pi, inverted haplotypes",
            "fst": "Hudson FST", "dxy": "Dxy", "pi_avg": "Mean within-class pi",
            "da": "Net divergence da"}).drop(columns=["recur"], errors="ignore")
        _write(out, "architecture_covariates",
               "per-locus architecture covariates used by those controls")


def phewas_lambda():
    df = _read("phewas_lambda_gc.tsv")
    if df is not None:
        out = df.rename(columns={
            "inversion": "Inversion ID", "label": "Locus",
            "n_tests": "Phecodes tested",
            "lambda_gc": "Genomic control factor (lambda)",
            "lambda_lo": "lambda, 95% CI lower",
            "lambda_hi": "lambda, 95% CI upper",
            "lambda_excl_significant":
                "lambda excluding family-significant associations",
            "lambda_ncases_ge1000":
                "lambda restricted to phecodes with >= 1000 cases",
            "n_tests_ncases_ge1000": "Phecodes with >= 1000 cases",
            "dosage_fst_between_ancestry":
                "Between-ancestry FST of imputed dosage",
            "n_family_significant": "Associations at BH q < 0.05"})
        _write(out, "phewas_lambda",
               "genomic-control calibration per inversion")

    hits = _read("phewas_lambda_significant_hits.tsv")
    if hits is not None:
        out = hits.rename(columns={
            "Phenotype": "Phecode", "inv_label": "Locus",
            "Inversion": "Inversion ID", "OR": "Odds ratio",
            "n_cases": "Cases", "p": "p-value",
            "q_global": "BH q-value", "lambda_used": "lambda applied",
            "p_gc": "p-value after genomic control",
            "q_gc": "BH q-value after genomic control",
            "category": "Phenotype category"})
        _write(out, "phewas_hits_genomic_control",
               "reported associations before and after genomic control")


def within_ancestry():
    df = _read("phewas_within_ancestry_meta.tsv")
    if df is None:
        return
    out = df.rename(columns={
        "Phenotype": "Phecode", "inv_label": "Locus",
        "Inversion": "Inversion ID",
        "p_multiancestry": "p-value, pooled multi-ancestry model",
        "q_global": "BH q-value, pooled model",
        "or_multiancestry": "Odds ratio, pooled model",
        "beta_multiancestry": "log odds ratio, pooled model",
        "beta_within_ancestry_meta":
            "log odds ratio, meta-analysis of within-ancestry estimates",
        "se_within_ancestry_meta": "Standard error of that meta-analysis",
        "p_within_ancestry_meta": "p-value, within-ancestry meta-analysis",
        "shrinkage_ratio":
            "Within-ancestry effect / pooled effect (1 = no attenuation)",
        "cochran_Q": "Cochran Q across ancestry groups",
        "p_heterogeneity": "p-value for between-ancestry heterogeneity",
        "I2": "I-squared (%)", "n_ancestries": "Ancestry groups contributing",
        "ancestries": "Ancestry groups"})
    _write(out, "within_ancestry_meta",
           "reported associations re-estimated within ancestry groups")


def gene_span():
    df = _read("gene_span_conservation.tsv")
    if df is not None:
        df = df[df["status"] == "OK"]
        out = pd.DataFrame({
            "Gene": df["gene_name"],
            "Transcript": df["transcript_id"],
            "Inversion": df["inv_id"],
            "Recurrence class": df["recurrence"],
            "Gene span (bp)": df["span_bp"],
            "Coding sequence length (bp)": df["cds_bp"],
            "Callable fraction of gene span": df.get("span_callable_frac"),
            "Direct haplotypes": df["k_direct"],
            "Inverted haplotypes": df["k_inverted"],
            "Identical pairs / pairs, CDS, direct":
                df["cds_prop_identical_direct"],
            "Identical pairs / pairs, CDS, inverted":
                df["cds_prop_identical_inverted"],
            "Identical pairs / pairs, whole gene, direct":
                df["gene_prop_identical_direct"],
            "Identical pairs / pairs, whole gene, inverted":
                df["gene_prop_identical_inverted"],
            "pi across CDS, direct": df["cds_pi_direct"],
            "pi across CDS, inverted": df["cds_pi_inverted"],
            "pi across whole gene, direct": df["gene_pi_direct"],
            "pi across whole gene, inverted": df["gene_pi_inverted"],
        })
        _write(out, "gene_span_conservation",
               "whole-gene versus CDS haplotype identity, per gene")

    mod = _read("gene_span_conservation_model.tsv")
    if mod is not None:
        out = mod.rename(columns={
            "statistic": "Sequence window", "analysis": "Analysis",
            "estimate": "Estimate", "p_value": "p-value",
            "n_units": "Independent units (inversions)",
            "sd_units": "Between-inversion SD or effect needed for 80% power"})
        _write(out, "gene_span_model",
               "refit of the conservation model on entire gene sequences")


def population_frequency():
    df = _read("inversion_population_frequencies.tsv")
    if df is None:
        return
    out = df.rename(columns={
        "Inversion": "Inversion ID", "Population": "Genetic ancestry group",
        "N": "Participants", "Mean_Dosage": "Mean imputed dosage",
        "Allele_Freq": "Imputed inverted allele frequency",
        "CI95_Lower": "95% CI lower", "CI95_Upper": "95% CI upper"})
    keep = ["Inversion ID", "Genetic ancestry group", "Participants",
            "Mean imputed dosage", "Imputed inverted allele frequency",
            "95% CI lower", "95% CI upper"]
    _write(out[[c for c in keep if c in out.columns]], "population_frequency",
           "imputed inverted-allele frequency by ancestry group")


def imputation_benchmark():
    df = _read("scoreinvhap_concordance.tsv")
    if df is None:
        return
    # The 6q24.1 comparison against experimental genotypes lives in its own
    # file; the response quotes all three benchmarks together, so put them in
    # one table rather than leaving the reader to join them.
    ext = _read("imputation_benchmark_HsInv0284_summary.tsv")
    if ext is not None and "group" in ext.columns:
        allrow = ext[ext["group"].astype(str).str.upper() == "ALL"]
        if len(allrow):
            r = allrow.iloc[0]
            df = pd.concat([pd.DataFrame([{
                "inversion": "6q24.1 (HsInv0284)",
                "model_id": "chr6-141867315-INV-29159",
                "n_samples": r["n"],
                "pearson_r": r["pearson_r"],
                "r2": r["r2"],
                "spearman_rho": np.nan,
                "hardcall_concordance": r["concordance"],
                "our_inv_allele_freq": np.nan,
                "sih_inv_allele_freq": np.nan,
            }]), df], ignore_index=True)
        df["comparison"] = np.where(
            df["inversion"].astype(str).str.startswith("6q24.1"),
            "experimental genotypes, Giner-Delgado et al. 2019 (1000 Genomes)",
            "ScoreInvHap calls")
    out = df.rename(columns={
        "comparison": "External call set",
        "inversion": "Inversion", "model_id": "Imputation model",
        "n_samples": "Samples compared",
        "pearson_r": "Pearson r, imputed vs external dosage",
        "r2": "r-squared, imputed vs external dosage",
        "spearman_rho": "Spearman rho",
        "hardcall_concordance": "Hard-call genotype concordance",
        "our_inv_allele_freq": "Inverted allele frequency, this study",
        "sih_inv_allele_freq": "Inverted allele frequency, external call"})
    _write(out, "imputation_benchmark",
           "imputed dosage against external genotypes")


def ages_tag_snps():
    df = _read("ages_multi_tag_snps.tsv")
    if df is None:
        return
    out = df.rename(columns={
        "region": "Locus", "selection_kind": "Selection inference",
        "context": "Context", "chrom_hg19": "Chromosome (GRCh37)",
        "pos_hg19": "Position (GRCh37)", "chrom_hg38": "Chromosome (GRCh38)",
        "pos_hg38": "Position (GRCh38)", "rsid": "Tagging SNP",
        "r_with_inversion": "r with inversion genotype",
        "abs_r": "|r| with inversion genotype",
        "ages_ref": "AGES reference allele", "ages_alt": "AGES alternate allele",
        "alt_enriched_on": "Alternate allele enriched on which orientation",
        "ages_S": "Selection coefficient (AGES, alternate allele)",
        "ages_S_inverted_allele":
            "Selection coefficient oriented to the inverted allele",
        "ages_S_ci_lo": "95% CI lower", "ages_S_ci_hi": "95% CI upper",
        "ages_SE": "Standard error", "ages_P_X": "p-value",
        "ages_FDR": "Benjamini-Hochberg q-value",
        "ages_FILTER": "AGES quality filter", "in_ages": "Present in AGES"})
    _write(out, "ages_tag_snps",
           "every tagging SNP tested in AGES, per locus")


def main():
    print("Building response supplementary tables into data/")
    for fn in (four_fold, pin_pis, divergence, chimp_polarization,
               architecture_controls, phewas_lambda, within_ancestry,
               gene_span, population_frequency, imputation_benchmark,
               ages_tag_snps):
        print(f"\n{fn.__name__}:")
        fn()
    print("\nDone.")


if __name__ == "__main__":
    main()
