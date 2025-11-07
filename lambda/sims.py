import numpy as np
import matplotlib.pyplot as plt

N_INDIVIDUALS = 10_000
N_PHENOTYPES = 5_000
MAF_LIST = [0.01, 0.10, 0.30]
PHENOTYPE_LIST = [10, 100, 1_000]
CORRELATION_LIST = [0.0, 0.2, 0.8]
N_REPLICATES = 5

# Median of chi-square with 1 degree of freedom
CHI2_MEDIAN_1DF = 0.454936423119572


def simulate_lambda_for_maf(maf):
    n = N_INDIVIDUALS
    m = N_PHENOTYPES

    p = maf
    probs = [(1.0 - p) ** 2, 2.0 * p * (1.0 - p), p ** 2]

    genotypes = np.random.choice([0.0, 1.0, 2.0], size=n, p=probs)

    phenotypes = np.random.normal(loc=0.0, scale=1.0, size=(n, m))

    x = genotypes
    x_mean = x.mean()
    x_centered = x - x_mean
    sxx = np.sum(x_centered ** 2)

    y = phenotypes
    y_mean = y.mean(axis=0, keepdims=True)
    y_centered = y - y_mean

    sxy = x_centered @ y_centered

    beta1 = sxy / sxx

    residuals = y_centered - np.outer(x_centered, beta1)

    sse = np.sum(residuals ** 2, axis=0)

    sigma2_hat = sse / (n - 2)
    var_beta1 = sigma2_hat / sxx

    t_stats = beta1 / np.sqrt(var_beta1)

    chi2_stats = t_stats ** 2

    lambda_gc = np.median(chi2_stats) / CHI2_MEDIAN_1DF
    return lambda_gc


def simulate_lambda_for_n_phenotypes(n_phenotypes, maf=0.10):
    n = N_INDIVIDUALS
    m = n_phenotypes

    p = maf
    probs = [(1.0 - p) ** 2, 2.0 * p * (1.0 - p), p ** 2]

    genotypes = np.random.choice([0.0, 1.0, 2.0], size=n, p=probs)

    phenotypes = np.random.normal(loc=0.0, scale=1.0, size=(n, m))

    x = genotypes
    x_mean = x.mean()
    x_centered = x - x_mean
    sxx = np.sum(x_centered ** 2)

    y = phenotypes
    y_mean = y.mean(axis=0, keepdims=True)
    y_centered = y - y_mean

    sxy = x_centered @ y_centered

    beta1 = sxy / sxx

    residuals = y_centered - np.outer(x_centered, beta1)

    sse = np.sum(residuals ** 2, axis=0)

    sigma2_hat = sse / (n - 2)
    var_beta1 = sigma2_hat / sxx

    t_stats = beta1 / np.sqrt(var_beta1)

    chi2_stats = t_stats ** 2

    lambda_gc = np.median(chi2_stats) / CHI2_MEDIAN_1DF
    return lambda_gc


def simulate_lambda_with_correlated_phenotypes(correlation, maf=0.10, n_individuals=None, n_phenotypes=None):
    n = n_individuals if n_individuals is not None else N_INDIVIDUALS
    m = n_phenotypes if n_phenotypes is not None else N_PHENOTYPES

    p = maf
    probs = [(1.0 - p) ** 2, 2.0 * p * (1.0 - p), p ** 2]

    genotypes = np.random.choice([0.0, 1.0, 2.0], size=n, p=probs)

    # Generate correlated phenotypes using a common factor model
    # For correlation rho, Y_i = sqrt(rho) * Z + sqrt(1-rho) * epsilon_i
    # where Z is a common factor and epsilon_i are independent noise terms
    rho = correlation
    common_factor = np.random.normal(loc=0.0, scale=1.0, size=(n, 1))
    independent_noise = np.random.normal(loc=0.0, scale=1.0, size=(n, m))

    if rho == 0.0:
        phenotypes = independent_noise
    else:
        phenotypes = np.sqrt(rho) * common_factor + np.sqrt(1.0 - rho) * independent_noise

    x = genotypes
    x_mean = x.mean()
    x_centered = x - x_mean
    sxx = np.sum(x_centered ** 2)

    y = phenotypes
    y_mean = y.mean(axis=0, keepdims=True)
    y_centered = y - y_mean

    sxy = x_centered @ y_centered

    beta1 = sxy / sxx

    residuals = y_centered - np.outer(x_centered, beta1)

    sse = np.sum(residuals ** 2, axis=0)

    sigma2_hat = sse / (n - 2)
    var_beta1 = sigma2_hat / sxx

    t_stats = beta1 / np.sqrt(var_beta1)

    chi2_stats = t_stats ** 2

    lambda_gc = np.median(chi2_stats) / CHI2_MEDIAN_1DF
    return lambda_gc


def main():
    np.random.seed(20251107)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs("lambda", exist_ok=True)
    
    # Test 1: Varying MAF
    lambdas = {}

    for maf in MAF_LIST:
        maf_values = []
        for _ in range(N_REPLICATES):
            lam = simulate_lambda_for_maf(maf)
            maf_values.append(lam)
        lambdas[maf] = maf_values

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
    plt.ylim(0.9, 1.1)
    plt.grid(True, alpha=0.3)
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
        pheno_values = []
        for _ in range(N_REPLICATES):
            lam = simulate_lambda_for_n_phenotypes(n_pheno)
            pheno_values.append(lam)
        lambdas_phenotypes[n_pheno] = pheno_values

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
    plt.ylim(0.9, 1.1)
    plt.grid(True, alpha=0.3)
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
        corr_values = []
        for _ in range(N_REPLICATES):
            lam = simulate_lambda_with_correlated_phenotypes(corr, n_individuals=100_000, n_phenotypes=1_000)
            corr_values.append(lam)
        lambdas_correlation[corr] = corr_values

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
    plt.ylim(0.9, 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lambda/lambda_vs_correlation.png", dpi=150)
    print("Saved figure: lambda/lambda_vs_correlation.png")
    plt.close()


if __name__ == "__main__":
    main()
