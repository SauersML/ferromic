#!/usr/bin/env Rscript
# SVbyEye miropeat of GRCh38 (target) vs a chimpanzee assembly (query) at one
# inversion locus. This is the ancestral-outgroup synteny view -- the panel the
# panTro6 -> T2T reference swap actually changes. Layout mirrors the original
# ferromic/svbyeye/bin/visualize_inv.R miropeat: plotMiro colored by alignment
# direction, with the inversion interval annotated as a rectangle on the target.
#
# Args: paf_file out_png inv_id chrom istart iend wstart wend af disease refname
suppressPackageStartupMessages({library(SVbyEye); library(ggplot2); library(GenomicRanges)})
a <- commandArgs(trailingOnly = TRUE)
paf_file <- a[1]; out_png <- a[2]; inv_id <- a[3]; chrom <- a[4]
istart <- as.integer(a[5]); iend <- as.integer(a[6])
wstart <- as.integer(a[7]); wend <- as.integer(a[8])
af <- a[9]; disease <- a[10]; refname <- a[11]
MIN_ALN <- 2000

placeholder <- function(msg) {
  p <- ggplot() + annotate("text", x = .5, y = .5,
        label = paste0(inv_id, "\n[", refname, "]\n", msg), size = 6) + theme_void()
  ggsave(out_png, p, width = 11, height = 4, dpi = 150)
  cat("PLACEHOLDER", refname, inv_id, msg, "\n"); quit(save = "no", status = 0)
}

if (!file.exists(paf_file) || file.info(paf_file)$size == 0)
  placeholder("no overlapping alignments")
pt <- tryCatch(readPaf(paf_file, include.paf.tags = TRUE, restrict.paf.tags = "cg"),
               error = function(e) data.frame())
if (nrow(pt) == 0) placeholder("empty PAF")
pt <- pt[pt$t.name == chrom, ]; if (nrow(pt) == 0) placeholder("no records on target chrom")
gr_win <- GRanges(chrom, IRanges(wstart, wend))
gr_inv <- GRanges(chrom, IRanges(istart, iend))
pt <- tryCatch(subsetPafAlignments(pt, target.region = gr_win), error = function(e) pt)
pt <- tryCatch(filterPaf(pt, min.align.len = MIN_ALN), error = function(e) pt)
if (nrow(pt) == 0) placeholder("nothing in window after filter")

L <- iend - istart
clip <- function(a0, a1, b0, b1) { lo <- pmax(a0, b0); hi <- pmin(a1, b1); pmax(0, hi - lo) }
ov <- clip(pt$t.start, pt$t.end, istart, iend)
cov_inv <- sum(ov); rev_inv <- sum(ov[pt$strand == "-"])
frac_rev <- if (cov_inv > 0) rev_inv / cov_inv else NA
orient <- if (is.na(frac_rev)) "NA" else if (frac_rev > 0.6) "inverted vs hg38" else
          if (frac_rev < 0.4) "collinear vs hg38" else "mixed"
kb <- round(L / 1000, 1)
covpct <- round(100 * min(1, cov_inv / L), 1)
ttl <- sprintf("%s  |  hg38 vs %s  |  %s:%s-%s  %skb  AF=%s",
        inv_id, refname, chrom, format(istart, big.mark = ","),
        format(iend, big.mark = ","), kb, af)
subt <- sprintf("chimp orientation across INV: %s (rev frac=%s, INV covered=%s%%)%s",
        orient, ifelse(is.na(frac_rev), "NA", sprintf("%.2f", frac_rev)), covpct,
        if (disease != "NA" && nzchar(disease)) paste0("  |  genes: ", disease) else "")

plt <- tryCatch(plotMiro(paf.table = pt, color.by = "direction"), error = function(e) NULL)
if (is.null(plt)) placeholder("plotMiro failed")
plt <- tryCatch(addAnnotation(plt, annot.gr = gr_inv, coordinate.space = "target",
        shape = "rectangle", annotation.label = "INV"), error = function(e) plt)
plt <- plt + labs(title = ttl, subtitle = subt) +
  theme(plot.title = element_text(size = 8), plot.subtitle = element_text(size = 7),
        legend.position = "bottom")
nn <- length(unique(pt$q.name))
ggsave(out_png, plt, width = 12, height = max(3.5, min(12, nn * 0.8 + 2.8)),
       dpi = 150, limitsize = FALSE)
cat(sprintf("QC\t%s\t%s\t%s\t%d\t%d\t%s\t%.1f\t%s\n",
    refname, inv_id, chrom, istart, iend,
    ifelse(is.na(frac_rev), "NA", sprintf("%.3f", frac_rev)), covpct, orient))
cat("WROTE", out_png, "\n")
