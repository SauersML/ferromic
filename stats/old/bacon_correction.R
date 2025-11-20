#!/usr/bin/env Rscript

# BACON correction for PheWAS results
# Applies per-inversion bias and inflation correction using empirical null distribution
#
# Method: Per-inversion BACON (van Iterson et al. 2017)
# - Each inversion = one "BACON set" with its own empirical null
# - Fits 3-component mixture: N(μ₀, σ₀²) [null] + N(μ₊, σ₊²) + N(μ₋, σ₋²) [alternatives]
# - Corrects z-scores: z_corr = (z_raw - μ₀) / σ₀
# - Recomputes p-values from corrected z-scores
#
# Why per-inversion?
# - Each inversion may have different:
#   * Pleiotropy (fraction of truly associated phenotypes)
#   * Confounding structure (ancestry, batch effects)
#   * Inflation/deflation patterns
# - Forcing shared correction would misestimate individual empirical nulls
#
# Assumptions:
# - P_Value_x is two-sided p-value from normal/large-df t-test
# - Majority of phenotypes are null for each inversion (BACON robust to ~50% alternatives)
# - Test statistics within inversion share same confounding structure
# - Phenotype correlations are acceptable (absorbed into σ₀)
#
# Implementation details:
# - Clamps p=0 to machine epsilon (keeps strongest hits, avoids Inf z-scores)
# - Beta coefficients remain uncorrected (only z/p calibrated for inference)
# - For corrected effect sizes: would need bacon(effectsizes=..., standarderrors=...)
#
# Multiple testing:
# - Applies BH-FDR globally across all inversions × phenotypes
# - Controls overall false discovery rate at specified level

library(bacon)
library(ggplot2)

# Read PheWAS results
phewas <- read.delim("data/phewas_results.tsv", stringsAsFactors = FALSE)

# Get unique inversions
inversions <- unique(phewas$Inversion)
cat("Found", length(inversions), "inversions\n")
cat("Inversions:", paste(inversions, collapse=", "), "\n\n")

# Storage for results
all_results <- list()
bacon_params <- data.frame(
  Inversion = character(),
  mu_0 = numeric(),
  sigma_0 = numeric(),
  p_0 = numeric(),
  n_tests = integer(),
  stringsAsFactors = FALSE
)

# Process each inversion separately
for (inv in inversions) {
  cat("Processing:", inv, "\n")
  
  # Subset data for this inversion
  inv_data <- phewas[phewas$Inversion == inv, ]
  
  # Extract test statistics
  # Compute z-scores from Beta and P_Value_x
  # z = sign(Beta) * qnorm(1 - P/2)
  valid_idx <- !is.na(inv_data$Beta) & !is.na(inv_data$P_Value_x) & inv_data$P_Value_x > 0
  
  if (sum(valid_idx) < 100) {
    cat("  WARNING: Only", sum(valid_idx), "valid tests for", inv, "- skipping\n\n")
    next
  }
  
  beta <- inv_data$Beta[valid_idx]
  pval <- inv_data$P_Value_x[valid_idx]
  
  # Clamp extremely small p-values to avoid Inf z-scores
  # Machine underflow (p=0) → clamp to smallest representable positive value
  # This keeps strongest associations in the data rather than dropping them
  min_pval <- .Machine$double.xmin
  pval_clamped <- pmax(pval, min_pval)
  
  # Compute z-scores (assumes P_Value_x is two-sided)
  z_raw <- sign(beta) * qnorm(1 - pval_clamped / 2)
  
  # Verify all finite (should be true after clamping)
  if (any(!is.finite(z_raw))) {
    cat("  WARNING: Non-finite z-scores detected after clamping\n")
    finite_idx <- is.finite(z_raw)
    z_raw <- z_raw[finite_idx]
    beta <- beta[finite_idx]
    pval <- pval[finite_idx]
  }
  
  cat("  Valid tests:", length(z_raw), "\n")
  
  # Run BACON
  bc <- bacon(teststatistics = z_raw, 
              niter = 5000L, 
              nburnin = 2000L,
              verbose = FALSE)
  
  # Extract parameters
  # estimates(bc) returns matrix: [p.0, p.1, p.2, mu.0, mu.1, mu.2, sigma.0, sigma.1, sigma.2]
  # p.0 = proportion of null component (π₀)
  params <- estimates(bc)
  mu_0 <- bias(bc)        # Mean of empirical null (component 0)
  sigma_0 <- inflation(bc) # SD of empirical null (component 0)
  p_0 <- params[1]        # Proportion null (π₀)
  
  cat("  Bias (mu_0):", round(mu_0, 4), "\n")
  cat("  Inflation (sigma_0):", round(sigma_0, 4), "\n")
  cat("  Proportion null (p_0):", round(p_0, 4), "\n")
  
  # Warn if extreme pleiotropy detected
  if (p_0 < 0.5) {
    cat("  ⚠️  WARNING: Very high pleiotropy (π₀ < 0.5)\n")
    cat("      Empirical null may be unreliable - consider negative control phenotypes\n")
  }
  
  # Get corrected statistics
  z_corr <- tstat(bc, corrected = TRUE)
  p_corr <- pval(bc, corrected = TRUE)
  
  # Store results (align with potentially filtered indices)
  result_df <- data.frame(
    Phenotype = inv_data$Phenotype[valid_idx],
    Inversion = inv,
    Beta = beta,
    P_raw = pval,
    Z_raw = z_raw,
    Z_corrected = z_corr,
    P_corrected = p_corr,
    stringsAsFactors = FALSE
  )
  
  all_results[[inv]] <- list(
    data = result_df,
    bacon_obj = bc,
    params = params
  )
  
  # Store parameters
  bacon_params <- rbind(bacon_params, data.frame(
    Inversion = inv,
    mu_0 = mu_0,
    sigma_0 = sigma_0,
    p_0 = p_0,
    n_tests = length(z_raw),
    stringsAsFactors = FALSE
  ))
  
  cat("\n")
}

# Save parameter summary
write.table(bacon_params, 
            "stats/bacon_parameters.tsv", 
            sep = "\t", 
            row.names = FALSE, 
            quote = FALSE)

cat("Saved BACON parameters to stats/bacon_parameters.tsv\n")

# Combine all corrected results
all_corrected <- do.call(rbind, lapply(all_results, function(x) x$data))

# Apply FDR correction (Benjamini-Hochberg) across ALL tests (all inversions)
all_corrected$P_adjusted_BH <- p.adjust(all_corrected$P_corrected, method = "BH")

write.table(all_corrected, 
            "stats/bacon_corrected_results.tsv", 
            sep = "\t", 
            row.names = FALSE, 
            quote = FALSE)

cat("Saved corrected results to stats/bacon_corrected_results.tsv\n")
cat("  - P_corrected: BACON-corrected p-values\n")
cat("  - P_adjusted_BH: BH-FDR adjusted across all inversions and phenotypes\n")

# Create QQ plots comparing unadjusted vs BACON-corrected
pdf("stats/bacon_qq_plots.pdf", width = 14, height = 10)

par(mfrow = c(2, 3))
for (inv in names(all_results)) {
  result <- all_results[[inv]]$data
  
  # Sort p-values
  p_raw_sorted <- sort(result$P_raw)
  p_corr_sorted <- sort(result$P_corrected)
  n <- length(p_raw_sorted)
  
  # Expected uniform distribution
  expected <- (1:n) / (n + 1)
  
  # Convert to -log10
  obs_raw <- -log10(p_raw_sorted)
  obs_corr <- -log10(p_corr_sorted)
  exp_vals <- -log10(expected)
  
  # Plot
  plot(exp_vals, obs_raw, 
       pch = 20, col = rgb(0.8, 0.4, 0, 0.4), cex = 0.8,
       xlab = "Expected -log10(P)",
       ylab = "Observed -log10(P)",
       main = inv,
       xlim = c(0, max(exp_vals)),
       ylim = c(0, max(c(obs_raw, obs_corr))))
  
  # Add BACON-corrected points
  points(exp_vals, obs_corr, pch = 20, col = rgb(0, 0.4, 0.8, 0.4), cex = 0.8)
  
  # Identity line
  abline(0, 1, col = "black", lty = 2, lwd = 1.5)
  
  # Add parameter text
  params <- bacon_params[bacon_params$Inversion == inv, ]
  text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]),
       y = par("usr")[4] - 0.05 * diff(par("usr")[3:4]),
       labels = sprintf("μ₀=%.3f\nσ₀=%.3f\nλ=%.3f", 
                       params$mu_0, params$sigma_0, params$sigma_0^2),
       adj = c(0, 1), cex = 0.7, font = 2)
  
  # Add legend
  legend("topleft", 
         legend = c("Unadjusted", "BACON"),
         col = c(rgb(0.8, 0.4, 0, 0.6), rgb(0, 0.4, 0.8, 0.6)),
         pch = 20, pt.cex = 1.5, cex = 0.7, bg = "white")
}

dev.off()
cat("Saved QQ plots to stats/bacon_qq_plots.pdf\n")

# Also save as PNG
png("stats/bacon_qq_plots.png", width = 1400, height = 1000, res = 100)

par(mfrow = c(2, 3))
for (inv in names(all_results)) {
  result <- all_results[[inv]]$data
  
  # Sort p-values
  p_raw_sorted <- sort(result$P_raw)
  p_corr_sorted <- sort(result$P_corrected)
  n <- length(p_raw_sorted)
  
  # Expected uniform distribution
  expected <- (1:n) / (n + 1)
  
  # Convert to -log10
  obs_raw <- -log10(p_raw_sorted)
  obs_corr <- -log10(p_corr_sorted)
  exp_vals <- -log10(expected)
  
  # Plot
  plot(exp_vals, obs_raw, 
       pch = 20, col = rgb(0.8, 0.4, 0, 0.4), cex = 0.8,
       xlab = "Expected -log10(P)",
       ylab = "Observed -log10(P)",
       main = inv,
       xlim = c(0, max(exp_vals)),
       ylim = c(0, max(c(obs_raw, obs_corr))))
  
  # Add BACON-corrected points
  points(exp_vals, obs_corr, pch = 20, col = rgb(0, 0.4, 0.8, 0.4), cex = 0.8)
  
  # Identity line
  abline(0, 1, col = "black", lty = 2, lwd = 1.5)
  
  # Add parameter text
  params <- bacon_params[bacon_params$Inversion == inv, ]
  text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]),
       y = par("usr")[4] - 0.05 * diff(par("usr")[3:4]),
       labels = sprintf("μ₀=%.3f\nσ₀=%.3f\nλ=%.3f", 
                       params$mu_0, params$sigma_0, params$sigma_0^2),
       adj = c(0, 1), cex = 0.7, font = 2)
  
  # Add legend
  legend("topleft", 
         legend = c("Unadjusted", "BACON"),
         col = c(rgb(0.8, 0.4, 0, 0.6), rgb(0, 0.4, 0.8, 0.6)),
         pch = 20, pt.cex = 1.5, cex = 0.7, bg = "white")
}

dev.off()
cat("Saved QQ plots to stats/bacon_qq_plots.png\n")

cat("\n=== BACON Correction Complete ===\n\n")

# Print formatted summary
cat(rep("=", 100), "\n", sep="")
cat("BACON Correction Summary - Per-Inversion Empirical Null Parameters\n")
cat(rep("=", 100), "\n\n", sep="")

# Calculate lambda_GC
bacon_params$lambda_GC <- bacon_params$sigma_0 ^ 2

# Shorten names for display
bacon_params$Inv_Short <- gsub("chr", "", bacon_params$Inversion)
bacon_params$Inv_Short <- gsub("-INV-", ":", bacon_params$Inv_Short)

# Main table
cat(sprintf("%-25s %10s %10s %10s %10s %10s\n", 
            "Inversion", "μ₀ (bias)", "σ₀ (infl)", "λ_GC", "π₀ (null)", "N tests"))
cat(rep("-", 100), "\n", sep="")

for (i in 1:nrow(bacon_params)) {
  cat(sprintf("%-25s %10.4f %10.4f %10.4f %10.4f %10d\n",
              bacon_params$Inv_Short[i],
              bacon_params$mu_0[i],
              bacon_params$sigma_0[i],
              bacon_params$lambda_GC[i],
              bacon_params$p_0[i],
              bacon_params$n_tests[i]))
}

cat("\n")
cat(rep("=", 100), "\n", sep="")
cat("Top Significant Associations (Global FDR < 0.05)\n")
cat(rep("=", 100), "\n\n", sep="")

# Get top associations by global FDR
top_assoc <- all_corrected[order(all_corrected$P_adjusted_BH), ]
top_assoc <- top_assoc[top_assoc$P_adjusted_BH < 0.05, ]

if (nrow(top_assoc) > 0) {
  # Show top 20
  n_show <- min(20, nrow(top_assoc))
  for (i in 1:n_show) {
    row <- top_assoc[i, ]
    inv_short <- gsub("chr", "", row$Inversion)
    inv_short <- gsub("-INV-", ":", inv_short)
    
    cat(sprintf("%2d. %s\n", i, row$Phenotype))
    cat(sprintf("    Inversion: %s\n", inv_short))
    cat(sprintf("    P_raw: %.2e → P_corrected: %.2e → P_adj_BH: %.2e\n", 
                row$P_raw, row$P_corrected, row$P_adjusted_BH))
    cat(sprintf("    Beta: %.4f, Z_raw: %.2f → Z_corr: %.2f\n\n", 
                row$Beta, row$Z_raw, row$Z_corrected))
  }
  
  if (nrow(top_assoc) > n_show) {
    cat(sprintf("... and %d more associations with global FDR < 0.05\n\n", 
                nrow(top_assoc) - n_show))
  }
} else {
  cat("No associations with global FDR < 0.05\n\n")
}

cat(rep("=", 100), "\n", sep="")
cat("Interpretation\n")
cat(rep("=", 100), "\n\n", sep="")

for (i in 1:nrow(bacon_params)) {
  inv <- bacon_params$Inv_Short[i]
  mu <- bacon_params$mu_0[i]
  sigma <- bacon_params$sigma_0[i]
  lambda <- bacon_params$lambda_GC[i]
  p0 <- bacon_params$p_0[i]
  
  cat(inv, ":\n", sep="")
  
  # Bias interpretation
  abs_mu <- abs(mu)
  if (abs_mu < 0.05) {
    bias_interp <- "negligible"
  } else if (abs_mu < 0.1) {
    bias_interp <- "small"
  } else if (abs_mu < 0.2) {
    bias_interp <- "moderate"
  } else {
    bias_interp <- "substantial"
  }
  
  direction <- ifelse(mu > 0, "positive", "negative")
  cat(sprintf("  Bias:      %s (%s shift, μ₀=%.3f)\n", bias_interp, direction, mu))
  
  # Inflation interpretation
  if (sigma < 1.05) {
    infl_interp <- "minimal"
  } else if (sigma < 1.1) {
    infl_interp <- "mild"
  } else if (sigma < 1.2) {
    infl_interp <- "moderate"
  } else {
    infl_interp <- "substantial"
  }
  
  if (sigma > 1) {
    cat(sprintf("  Inflation: %s (σ₀=%.3f, λ=%.3f)\n", infl_interp, sigma, lambda))
  } else {
    cat(sprintf("  Deflation: σ₀=%.3f, λ=%.3f\n", sigma, lambda))
  }
  
  # Pleiotropy (π₀ = proportion of truly null tests)
  if (p0 > 0.95) {
    pleio <- "low pleiotropy"
  } else if (p0 > 0.85) {
    pleio <- "moderate pleiotropy"
  } else {
    pleio <- "high pleiotropy"
  }
  non_null_pct <- (1 - p0) * 100
  cat(sprintf("  Null prop: π₀=%.1f%% (%s)\n", p0 * 100, pleio))
  cat(sprintf("             → ~%.0f%% estimated non-null (may include weak/undetectable effects)\n", non_null_pct))
  
  # Recommendation
  if (sigma > 1.1 || abs_mu > 0.1) {
    cat("  → Correction RECOMMENDED: substantial confounding detected\n")
  } else {
    cat("  → Minimal correction needed: well-calibrated tests\n")
  }
  
  cat("\n")
}

cat(rep("=", 100), "\n", sep="")
cat("Key Findings\n")
cat(rep("=", 100), "\n\n", sep="")

# Find most problematic
max_infl_idx <- which.max(bacon_params$sigma_0)
max_bias_idx <- which.max(abs(bacon_params$mu_0))
most_pleio_idx <- which.min(bacon_params$p_0)

cat("1. Highest inflation:", bacon_params$Inv_Short[max_infl_idx], "\n")
cat(sprintf("   σ₀=%.3f, λ=%.3f\n", 
            bacon_params$sigma_0[max_infl_idx], 
            bacon_params$lambda_GC[max_infl_idx]))
cat(sprintf("   → Test statistics are inflated by %.1f%%\n\n", 
            (bacon_params$lambda_GC[max_infl_idx] - 1) * 100))

cat("2. Largest bias:", bacon_params$Inv_Short[max_bias_idx], "\n")
cat(sprintf("   μ₀=%.3f\n", bacon_params$mu_0[max_bias_idx]))
direction <- ifelse(bacon_params$mu_0[max_bias_idx] > 0, "positive", "negative")
cat(sprintf("   → Systematic %s shift in test statistics\n\n", direction))

cat("3. Most pleiotropic:", bacon_params$Inv_Short[most_pleio_idx], "\n")
cat(sprintf("   π₀=%.1f%% (null proportion)\n", bacon_params$p_0[most_pleio_idx] * 100))
cat(sprintf("   → ~%.0f%% estimated non-null (includes weak/undetectable effects)\n", 
            (1 - bacon_params$p_0[most_pleio_idx]) * 100))
cat(sprintf("   → Only %.1f%% actually significant at FDR < 0.05\n\n",
            100 * sum(all_corrected[all_corrected$Inversion == bacon_params$Inversion[most_pleio_idx], ]$P_adjusted_BH < 0.05) / 
            bacon_params$n_tests[most_pleio_idx]))

cat(rep("=", 100), "\n", sep="")
cat("Significant Associations (FDR < 0.05)\n")
cat(rep("=", 100), "\n\n", sep="")

# Count significant associations
sig_total <- sum(all_corrected$P_adjusted_BH < 0.05, na.rm = TRUE)
cat("Total significant at FDR < 0.05:", sig_total, "\n\n")

for (inv in unique(all_corrected$Inversion)) {
  inv_data <- all_corrected[all_corrected$Inversion == inv, ]
  sig_count <- sum(inv_data$P_adjusted_BH < 0.05, na.rm = TRUE)
  
  inv_short <- gsub("chr", "", inv)
  inv_short <- gsub("-INV-", ":", inv_short)
  
  cat(sprintf("%s: %d significant\n", inv_short, sig_count))
}

cat("\n")
cat(rep("=", 100), "\n", sep="")
cat("Top Significant Associations (Global FDR < 0.05)\n")
cat(rep("=", 100), "\n\n", sep="")

# Get top associations by global FDR
top_assoc <- all_corrected[order(all_corrected$P_adjusted_BH), ]
top_assoc <- top_assoc[top_assoc$P_adjusted_BH < 0.05, ]

if (nrow(top_assoc) > 0) {
  # Show top 20
  n_show <- min(20, nrow(top_assoc))
  for (i in 1:n_show) {
    row <- top_assoc[i, ]
    inv_short <- gsub("chr", "", row$Inversion)
    inv_short <- gsub("-INV-", ":", inv_short)
    
    cat(sprintf("%2d. %s\n", i, row$Phenotype))
    cat(sprintf("    Inversion: %s\n", inv_short))
    cat(sprintf("    P_raw: %.2e → P_corrected: %.2e → P_adj_BH: %.2e\n", 
                row$P_raw, row$P_corrected, row$P_adjusted_BH))
    cat(sprintf("    Beta: %.4f, Z_raw: %.2f → Z_corr: %.2f\n\n", 
                row$Beta, row$Z_raw, row$Z_corrected))
  }
  
  if (nrow(top_assoc) > n_show) {
    cat(sprintf("... and %d more associations with global FDR < 0.05\n\n", 
                nrow(top_assoc) - n_show))
  }
} else {
  cat("No associations with global FDR < 0.05\n\n")
}

cat(rep("=", 100), "\n", sep="")
