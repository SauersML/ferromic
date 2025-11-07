import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

N_INDIVIDUALS = 10_000
N_PHENOTYPES = 5_000
MAF_LIST = [0.01, 0.10, 0.30]
PHENOTYPE_LIST = [10, 100, 1_000]
CORRELATION_LIST = [0.0, 0.2, 0.8]
N_REPLICATES = 5

# Median of chi-square with 1 degree of freedom
CHI2_MEDIAN_1DF = 0.454936423119572


def compute_chi2_stats(x, y):
    """
    Compute chi-square statistics for simple linear regression.

    x: (n,) genotype vector
    y: (n, m) phenotype matrix

    Returns: (m,) array of chi-square statistics

    Uses closed-form formula avoiding residual matrix:
    r^2 = (sxy^2) / (sxx * syy)
    chi2 = (n - 2) * r^2 / (1 - r^2)
    """
    n = x.shape[0]

    # Center x
    x_centered = x - x.mean()
    sxx = np.sum(x_centered ** 2)

    # Center y
    y_centered = y - y.mean(axis=0, keepdims=True)

    # Sufficient statistics
    sxy = x_centered @ y_centered  # (m,)
    syy = np.sum(y_centered ** 2, axis=0)  # (m,)

    # Correlation-based chi-square
    r_squared = (sxy ** 2) / (sxx * syy)
    chi2 = (n - 2) * r_squared / (1.0 - r_squared)

    return chi2


def simulate_lambda_for_maf(maf, n_replicates=N_REPLICATES):
    """Vectorized over replicates."""
    n = N_INDIVIDUALS
    m = N_PHENOTYPES

    # Generate all replicates at once
    genotypes = np.random.binomial(2, maf, size=(n_replicates, n)).astype(np.float32)
    phenotypes = np.random.normal(0.0, 1.0, size=(n_replicates, n, m)).astype(np.float32)

    lambda_values = []
    for i in range(n_replicates):
        chi2_stats = compute_chi2_stats(genotypes[i], phenotypes[i])
        lambda_gc = np.median(chi2_stats) / CHI2_MEDIAN_1DF
        lambda_values.append(lambda_gc)

    return lambda_values


def simulate_lambda_for_n_phenotypes(n_phenotypes, maf=0.10, n_replicates=N_REPLICATES):
    """Vectorized over replicates."""
    n = N_INDIVIDUALS
    m = n_phenotypes

    # Generate all replicates at once
    genotypes = np.random.binomial(2, maf, size=(n_replicates, n)).astype(np.float32)
    phenotypes = np.random.normal(0.0, 1.0, size=(n_replicates, n, m)).astype(np.float32)

    lambda_values = []
    for i in range(n_replicates):
        chi2_stats = compute_chi2_stats(genotypes[i], phenotypes[i])
        lambda_gc = np.median(chi2_stats) / CHI2_MEDIAN_1DF
        lambda_values.append(lambda_gc)

    return lambda_values


def simulate_lambda_with_correlated_phenotypes(correlation, maf=0.10, n_individuals=None, n_phenotypes=None, n_replicates=N_REPLICATES):
    """Vectorized over replicates."""
    n = n_individuals if n_individuals is not None else N_INDIVIDUALS
    m = n_phenotypes if n_phenotypes is not None else N_PHENOTYPES
    rho = correlation

    # Generate all replicates at once
    genotypes = np.random.binomial(2, maf, size=(n_replicates, n)).astype(np.float32)

    # Generate correlated phenotypes using a common factor model
    # For correlation rho, Y_i = sqrt(rho) * Z + sqrt(1-rho) * epsilon_i
    common_factor = np.random.normal(0.0, 1.0, size=(n_replicates, n, 1)).astype(np.float32)
    independent_noise = np.random.normal(0.0, 1.0, size=(n_replicates, n, m)).astype(np.float32)

    if rho == 0.0:
        phenotypes = independent_noise
    else:
        phenotypes = np.sqrt(rho) * common_factor + np.sqrt(1.0 - rho) * independent_noise

    lambda_values = []
    for i in range(n_replicates):
        chi2_stats = compute_chi2_stats(genotypes[i], phenotypes[i])
        lambda_gc = np.median(chi2_stats) / CHI2_MEDIAN_1DF
        lambda_values.append(lambda_gc)

    return lambda_values


def main():
    np.random.seed(20251107)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs("lambda", exist_ok=True)
    
    # Test 1: Varying MAF
    lambdas = {}

    for maf in MAF_LIST:
        lambdas[maf] = simulate_lambda_for_maf(maf)

    print("Genomic control lambda values")
    print("Each row: one MAF; entries: five replicate null PheWAS runs.")
    for maf in MAF_LIST:
        vals = lambdas[maf]
        formatted = "  ".join(f"{v:.6f}" for v in vals)
        print(f"MAF {maf:.2f}:  {formatted}")

    x_vals = []
    y_vals = []
    for maf in MAF_LIST:
        for lam in lambdas[maf]:
            x_vals.append(maf)
            y_vals.append(lam)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_vals, y_vals)
    plt.xlabel("Minor allele frequency")
    plt.ylabel("Genomic control lambda")
    plt.title("Null PheWAS lambdas across MAF and replicates")
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Expected λ=1')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lambda/lambda_vs_maf.png", dpi=150)
    print("Saved figure: lambda/lambda_vs_maf.png")
    plt.close()

    # Test 2: Varying number of phenotypes
    print("\n" + "="*60)
    print("Varying number of phenotypes (fixed MAF=0.10)")
    print("="*60)

    lambdas_phenotypes = {}

    for n_pheno in PHENOTYPE_LIST:
        lambdas_phenotypes[n_pheno] = simulate_lambda_for_n_phenotypes(n_pheno)

    print("\nGenomic control lambda values")
    print("Each row: one phenotype count; entries: five replicate null PheWAS runs.")
    for n_pheno in PHENOTYPE_LIST:
        vals = lambdas_phenotypes[n_pheno]
        formatted = "  ".join(f"{v:.6f}" for v in vals)
        print(f"N_phenotypes {n_pheno:5d}:  {formatted}")

    x_vals_pheno = []
    y_vals_pheno = []
    for n_pheno in PHENOTYPE_LIST:
        for lam in lambdas_phenotypes[n_pheno]:
            x_vals_pheno.append(n_pheno)
            y_vals_pheno.append(lam)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_vals_pheno, y_vals_pheno)
    plt.xlabel("Number of phenotypes")
    plt.ylabel("Genomic control lambda")
    plt.title("Null PheWAS lambdas across phenotype count and replicates")
    plt.xscale("log")
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Expected λ=1')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lambda/lambda_vs_n_phenotypes.png", dpi=150)
    print("Saved figure: lambda/lambda_vs_n_phenotypes.png")
    plt.close()

    # Test 3: Varying phenotype correlation
    print("\n" + "="*60)
    print("Varying phenotype correlation (fixed MAF=0.10)")
    print("100,000 individuals, 1,000 phenotypes")
    print("="*60)

    lambdas_correlation = {}

    for corr in CORRELATION_LIST:
        lambdas_correlation[corr] = simulate_lambda_with_correlated_phenotypes(
            corr, n_individuals=100_000, n_phenotypes=1_000
        )

    print("\nGenomic control lambda values")
    print("Each row: one correlation coefficient; entries: five replicate null PheWAS runs.")
    for corr in CORRELATION_LIST:
        vals = lambdas_correlation[corr]
        formatted = "  ".join(f"{v:.6f}" for v in vals)
        print(f"Correlation {corr:.1f}:  {formatted}")

    x_vals_corr = []
    y_vals_corr = []
    for corr in CORRELATION_LIST:
        for lam in lambdas_correlation[corr]:
            x_vals_corr.append(corr)
            y_vals_corr.append(lam)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_vals_corr, y_vals_corr)
    plt.xlabel("Phenotype correlation coefficient")
    plt.ylabel("Genomic control lambda")
    plt.title("Null PheWAS lambdas across phenotype correlation and replicates")
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Expected λ=1')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lambda/lambda_vs_correlation.png", dpi=150)
    print("Saved figure: lambda/lambda_vs_correlation.png")
    plt.close()


if __name__ == "__main__":
    main()
