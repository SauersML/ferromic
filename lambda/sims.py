import numpy as np
import matplotlib.pyplot as plt

N_INDIVIDUALS = 10_000
N_PHENOTYPES = 5_000
MAF_LIST = [0.01, 0.10, 0.30]
PHENOTYPE_LIST = [10, 100, 1_000]
N_REPLICATES = 3

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


def main():
    np.random.seed(20251107)

    # Test 1: Varying MAF
    lambdas = {}

    for maf in MAF_LIST:
        maf_values = []
        for _ in range(N_REPLICATES):
            lam = simulate_lambda_for_maf(maf)
            maf_values.append(lam)
        lambdas[maf] = maf_values

    print("Genomic control lambda values")
    print("Each row: one MAF; entries: three replicate null PheWAS runs.")
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
    plt.show()

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
    print("Each row: one phenotype count; entries: three replicate null PheWAS runs.")
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
    plt.show()


if __name__ == "__main__":
    main()
