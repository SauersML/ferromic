#!/usr/bin/env Rscript
suppressPackageStartupMessages({ library(SVbyEye); library(ggplot2) })
args <- commandArgs(trailingOnly=TRUE)
paf <- args[1]; out <- args[2]
cat("SVbyEye version:", as.character(packageVersion("SVbyEye")), "\n")
pt <- readPaf(paf.file=paf, include.paf.tags=TRUE, restrict.paf.tags="cg")
cat("rows:", nrow(pt), "\n")
p <- plotMiro(paf.table=pt, color.by="direction")
ggsave(out, plot=p, width=10, height=6, dpi=150)
cat("WROTE", out, "\n")
