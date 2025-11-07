import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

N_INDIVIDUALS = 10_000
N_PHENOTYPES = 5_000
MAF_LIST = [0.01, 0.10, 0.30]
PHENOTYPE_LIST = [10, 100, 1_000]
CORRELATION_LIST = [0.0, 0.2, 0.8]
CLUSTER_LIST = [1, 2, 4, 8, 20, 80, 200, 500]
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
    """Process one replicate at a time to minimize memory usage."""
    n = N_INDIVIDUALS
    m = N_PHENOTYPES

    lambda_values = []
    for i in range(n_replicates):
        # Generate one replicate at a time
        genotype = np.random.binomial(2, maf, size=n).astype(np.float32)
        phenotypes = np.random.normal(0.0, 1.0, size=(n, m)).astype(np.float32)

        chi2_stats = compute_chi2_stats(genotype, phenotypes)
        lambda_gc = np.median(chi2_stats) / CHI2_MEDIAN_1DF
        lambda_values.append(lambda_gc)

    return lambda_values


def simulate_lambda_for_n_phenotypes(n_phenotypes, maf=0.10, n_replicates=N_REPLICATES):
    """Process one replicate at a time to minimize memory usage."""
    n = N_INDIVIDUALS
    m = n_phenotypes

    lambda_values = []
    for i in range(n_replicates):
        # Generate one replicate at a time
        genotype = np.random.binomial(2, maf, size=n).astype(np.float32)
        phenotypes = np.random.normal(0.0, 1.0, size=(n, m)).astype(np.float32)

        chi2_stats = compute_chi2_stats(genotype, phenotypes)
        lambda_gc = np.median(chi2_stats) / CHI2_MEDIAN_1DF
        lambda_values.append(lambda_gc)

    return lambda_values


def simulate_lambda_with_correlated_phenotypes(correlation, maf=0.10, n_individuals=None, n_phenotypes=None, n_replicates=N_REPLICATES, block_size=100):
    """
    Process one replicate at a time, streaming phenotypes in blocks.

    This dramatically reduces memory usage by:
    1. Not stacking across replicates (saves factor of n_replicates)
    2. Processing phenotypes in blocks (peak memory ~n_individuals × block_size instead of n_individuals × n_phenotypes)

    For n_individuals=100k, n_phenotypes=1000, block_size=100:
    Peak memory per replicate: ~120MB instead of ~2GB
    """
    n = n_individuals if n_individuals is not None else N_INDIVIDUALS
    m = n_phenotypes if n_phenotypes is not None else N_PHENOTYPES
    rho = correlation

    lambda_values = []

    for rep in range(n_replicates):
        # Generate genotype for this replicate
        genotype = np.random.binomial(2, maf, size=n).astype(np.float32)

        # Center genotype once
        x_centered = genotype - genotype.mean()
        sxx = np.sum(x_centered ** 2)

        # Generate common factor once per replicate (if rho > 0)
        if rho > 0.0:
            common_factor = np.random.normal(0.0, 1.0, size=(n, 1)).astype(np.float32)

        # Pre-allocate array for chi-square statistics
        chi2_stats = np.empty(m, dtype=np.float32)

        # Process phenotypes in blocks
        idx = 0
        for start_idx in range(0, m, block_size):
            end_idx = min(start_idx + block_size, m)
            block_m = end_idx - start_idx

            # Generate correlated phenotypes for this block
            # Model: Y_i = sqrt(rho) * Z + sqrt(1-rho) * epsilon_i
            if rho == 0.0:
                phenotype_block = np.random.normal(0.0, 1.0, size=(n, block_m)).astype(np.float32)
            else:
                independent_noise = np.random.normal(0.0, 1.0, size=(n, block_m)).astype(np.float32)
                phenotype_block = np.sqrt(rho) * common_factor + np.sqrt(1.0 - rho) * independent_noise

            # Center phenotypes
            y_centered = phenotype_block - phenotype_block.mean(axis=0, keepdims=True)

            # Compute sufficient statistics for this block
            sxy = x_centered @ y_centered  # (block_m,)
            syy = np.sum(y_centered ** 2, axis=0)  # (block_m,)

            # Compute chi-square for this block
            r_squared = (sxy ** 2) / (sxx * syy)
            block_chi2 = (n - 2) * r_squared / (1.0 - r_squared)

            # Store results
            chi2_stats[idx:idx+block_m] = block_chi2
            idx += block_m

        # Compute lambda_GC from all chi-square statistics
        lambda_gc = np.median(chi2_stats) / CHI2_MEDIAN_1DF
        lambda_values.append(lambda_gc)

    return lambda_values


def simulate_lambda_with_clustered_phenotypes(n_clusters, within_cluster_correlation=0.5, maf=0.10, n_individuals=None, n_phenotypes=100, n_replicates=N_REPLICATES):
    """
    Simulate lambda_GC with phenotypes organized into clusters.

    Within each cluster, phenotypes have correlation = within_cluster_correlation.
    Between clusters, phenotypes are independent (correlation = 0).

    Args:
        n_clusters: Number of phenotype clusters
        within_cluster_correlation: Correlation within each cluster (default 0.5)
        maf: Minor allele frequency
        n_individuals: Number of individuals
        n_phenotypes: Total number of phenotypes (default 100)
        n_replicates: Number of replicates

    Returns:
        List of lambda_GC values, one per replicate
    """
    n = n_individuals if n_individuals is not None else N_INDIVIDUALS
    m = n_phenotypes
    rho = within_cluster_correlation

    # Compute phenotypes per cluster
    phenotypes_per_cluster = m // n_clusters
    remainder = m % n_clusters

    lambda_values = []

    for rep in range(n_replicates):
        # Generate genotype for this replicate
        genotype = np.random.binomial(2, maf, size=n).astype(np.float32)

        # Center genotype once
        x_centered = genotype - genotype.mean()
        sxx = np.sum(x_centered ** 2)

        # Pre-allocate array for chi-square statistics
        chi2_stats = np.empty(m, dtype=np.float32)

        # Generate phenotypes cluster by cluster
        idx = 0
        for cluster_idx in range(n_clusters):
            # Determine cluster size (distribute remainder across first few clusters)
            cluster_size = phenotypes_per_cluster + (1 if cluster_idx < remainder else 0)

            # Generate cluster-specific common factor
            if rho > 0.0:
                cluster_common_factor = np.random.normal(0.0, 1.0, size=(n, 1)).astype(np.float32)

            # Generate phenotypes for this cluster
            # Model: Y_i = sqrt(rho) * Z_cluster + sqrt(1-rho) * epsilon_i
            if rho == 0.0:
                phenotype_cluster = np.random.normal(0.0, 1.0, size=(n, cluster_size)).astype(np.float32)
            else:
                independent_noise = np.random.normal(0.0, 1.0, size=(n, cluster_size)).astype(np.float32)
                phenotype_cluster = np.sqrt(rho) * cluster_common_factor + np.sqrt(1.0 - rho) * independent_noise

            # Center phenotypes
            y_centered = phenotype_cluster - phenotype_cluster.mean(axis=0, keepdims=True)

            # Compute sufficient statistics for this cluster
            sxy = x_centered @ y_centered  # (cluster_size,)
            syy = np.sum(y_centered ** 2, axis=0)  # (cluster_size,)

            # Compute chi-square for this cluster
            r_squared = (sxy ** 2) / (sxx * syy)
            cluster_chi2 = (n - 2) * r_squared / (1.0 - r_squared)

            # Store results
            chi2_stats[idx:idx+cluster_size] = cluster_chi2
            idx += cluster_size

        # Compute lambda_GC from all chi-square statistics
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

    # Test 4: Varying number of phenotype clusters with within-cluster correlation
    print("\n" + "="*60)
    print("Varying number of phenotype clusters (fixed MAF=0.10)")
    print("100 phenotypes total, 0.5 correlation within each cluster")
    print("="*60)

    lambdas_clusters = {}

    for n_clusters in CLUSTER_LIST:
        lambdas_clusters[n_clusters] = simulate_lambda_with_clustered_phenotypes(
            n_clusters=n_clusters,
            within_cluster_correlation=0.5,
            n_phenotypes=100,
            n_individuals=N_INDIVIDUALS
        )

    print("\nGenomic control lambda values")
    print("Each row: one cluster count; entries: five replicate null PheWAS runs.")
    for n_clusters in CLUSTER_LIST:
        vals = lambdas_clusters[n_clusters]
        formatted = "  ".join(f"{v:.6f}" for v in vals)
        pheno_per_cluster = 100 // n_clusters
        print(f"Clusters {n_clusters:3d} ({pheno_per_cluster:3d} pheno/cluster):  {formatted}")

    x_vals_clusters = []
    y_vals_clusters = []
    for n_clusters in CLUSTER_LIST:
        for lam in lambdas_clusters[n_clusters]:
            x_vals_clusters.append(n_clusters)
            y_vals_clusters.append(lam)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_vals_clusters, y_vals_clusters)
    plt.xlabel("Number of phenotype clusters")
    plt.ylabel("Genomic control lambda")
    plt.title("Null PheWAS lambdas: 100 phenotypes with 0.5 within-cluster correlation")
    plt.xscale("log")
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Expected λ=1')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lambda/lambda_vs_clusters.png", dpi=150)
    print("Saved figure: lambda/lambda_vs_clusters.png")
    plt.close()


if __name__ == "__main__":
    main()
