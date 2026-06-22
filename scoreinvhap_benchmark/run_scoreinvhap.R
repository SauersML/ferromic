#!/usr/bin/env Rscript
# Genotype the 8p23.1 and 17q21.31 inversions with the published tool
# scoreInvHap (Ruiz-Arenas et al., PLOS Genetics 2019) on 1000 Genomes
# phase-3 (GRCh37) samples.
#
# scoreInvHap ships reference objects built from 1000 Genomes for exactly the
# two inversions the reviewer asked about: inv8_001 (8p23.1) and inv17_007
# (17q21.31). Passing inv=<id> makes scoreInvHap() auto-load the matching
# SNPsR2 / hetRefs / Refs reference objects internally.
#
# We emit, per sample, the maximum-posterior inversion genotype and its
# numeric inverted-allele dosage (NN=0, NI=1, II=2).
#
# Usage:
#   Rscript run_scoreinvhap.R <vcf> <inversion_key> <out_tsv>
# where <inversion_key> is one of: inv8p23.1 inv17q21.31

suppressMessages({
  library(scoreInvHap)
  library(VariantAnnotation)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: run_scoreinvhap.R <vcf> <inversion_key> <out_tsv>")
}
vcf_path <- args[[1]]
inv_key  <- args[[2]]
out_tsv  <- args[[3]]

# Map the human-facing key to the scoreInvHap reference identifier.
key_map <- list(
  "inv8p23.1"   = "inv8_001",
  "inv17q21.31" = "inv17_007"
)
if (!inv_key %in% names(key_map)) {
  stop(sprintf("Unknown inversion key '%s'", inv_key))
}
inv_id <- key_map[[inv_key]]

cat(sprintf("Reading VCF %s for %s (scoreInvHap id %s)\n",
            vcf_path, inv_key, inv_id))

# Read genotypes. scoreInvHap reference SNPs are on GRCh37, matching the
# 1000 Genomes phase-3 VCFs this workflow supplies.
vcf <- readVcf(vcf_path, genome = "hg19")
cat(sprintf("VCF variants read: %d ; samples: %d\n",
            length(rowRanges(vcf)), ncol(vcf)))

# scoreInvHap matches SNPs to its reference BY NAME. 1000G phase-3 rows are named by rsID,
# but the reference may use chr:position; the rsID dbSNP builds can differ -> "no common
# SNPs". Auto-detect which naming overlaps the reference and rename the VCF accordingly.
data(SNPsR2, package = "scoreInvHap")
ref_names <- names(SNPsR2[[inv_id]])
rr        <- rowRanges(vcf)
orig      <- rownames(vcf)
pos_names      <- paste(as.character(seqnames(rr)), start(rr), sep = ":")
pos_names_chr  <- paste0("chr", pos_names)
cands <- list(rsID = orig, `chr:pos` = pos_names, `chrN:pos` = pos_names_chr)
cat("Reference SNPs:", length(ref_names), "| head:", paste(head(ref_names), collapse = ","), "\n")
ov <- sapply(cands, function(nm) length(intersect(nm, ref_names)))
cat("VCF name overlap with reference -> ",
    paste(sprintf("%s=%d", names(ov), ov), collapse = "  "), "\n")
best <- names(which.max(ov))
if (ov[[best]] > 0 && best != "rsID") {
  rownames(vcf) <- cands[[best]]
  cat(sprintf("Renamed VCF SNPs to '%s' (%d common with reference).\n", best, ov[[best]]))
} else {
  cat(sprintf("Using original rsID names (%d common with reference).\n", ov[["rsID"]]))
}

# inv=<id> triggers internal loading of the bundled reference objects.
res <- scoreInvHap(SNPlist = vcf, inv = inv_id)

# inversion=TRUE collapses haplotype labels (Na/Ia/...) into N/I, giving
# inversion genotypes NN / NI / II.
geno <- classification(res, inversion = TRUE)
lab  <- as.character(geno)
samples <- names(geno)

dosage <- rep(NA_real_, length(lab))
dosage[lab == "NN"] <- 0
dosage[lab == "NI" | lab == "IN"] <- 1
dosage[lab == "II"] <- 2
# Generic fallback: count "I" characters in any unexpected 2-char label.
unresolved <- which(is.na(dosage))
for (i in unresolved) {
  dosage[i] <- lengths(regmatches(lab[i], gregexpr("I", lab[i])))
}

best_score <- apply(scores(res), 1, max)
# Align certainty/scores to the (possibly filtered) classification samples.
best_score <- best_score[samples]

out <- data.frame(
  SampleID = samples,
  scoreInvHap_class = lab,
  scoreInvHap_dosage = dosage,
  scoreInvHap_maxscore = round(as.numeric(best_score), 4),
  stringsAsFactors = FALSE
)
write.table(out, out_tsv, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("Wrote %d genotypes to %s\n", nrow(out), out_tsv))
cat("Class distribution:\n")
print(table(out$scoreInvHap_class))
