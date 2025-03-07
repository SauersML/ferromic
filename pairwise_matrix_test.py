import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import time
from scipy.stats import mannwhitneyu, gaussian_kde

# Constants
MIN_LENGTH = 1_000_000  # Minimum sequence length (1M bp)
EDGE_SIZE = 250_000     # Number of positions from each edge
MAX_PLOT_POINTS = 100_000  # Downsample limit for plotting
BW_METHOD = 0.5  # Bandwidth method for KDE
LOGSPACE_POINTS = 2000  # Number of points for logspace in KDE plots

def parse_header(line):
    """
    Parse a header line starting with '>' for filtered theta/pi sequences.

    Args:
        line (str): Header line.

    Returns:
        tuple: (metric, length, full_header) or None if not filtered theta/pi ≥ MIN_LENGTH.
    """
    if not line.startswith(">") or "filtered" not in line.lower():
        return None
    header = line[1:].strip()
    parts = header.split('_')
    metric = None
    if "theta" in header.lower():
        metric = "theta"
    elif "pi" in header.lower():
        metric = "pi"
    else:
        return None

    try:
        start_idx = parts.index("start")
        end_idx = parts.index("end")
        start = int(parts[start_idx + 1])
        end = int(parts[end_idx + 1])
        length = end - start + 1  # Inclusive range
        if length >= MIN_LENGTH:
            return metric, length, header
        return None
    except (ValueError, IndexError):
        return None

def count_sequences(file_path):
    """
    First pass: Count filtered sequences and total sizes for pre-allocation.

    Args:
        file_path (Path): Input file path.

    Returns:
        tuple: Counts and total sizes for theta/pi edge/middle.
    """
    num_theta = 0
    num_pi = 0
    theta_edge_total = 0
    theta_middle_total = 0
    pi_edge_total = 0
    pi_middle_total = 0

    with open(file_path, 'r') as f:
        while True:
            try:
                header = next(f).strip()
                result = parse_header(header)
                if result:
                    _, length, _ = result
                    if length >= 2 * EDGE_SIZE:
                        if "theta" in result[0]:
                            num_theta += 1
                            theta_edge_total += 2 * EDGE_SIZE
                            theta_middle_total += length - 2 * EDGE_SIZE
                        elif "pi" in result[0]:
                            num_pi += 1
                            pi_edge_total += 2 * EDGE_SIZE
                            pi_middle_total += length - 2 * EDGE_SIZE
                    next(f)  # Skip data line
            except StopIteration:
                break

    print(f"Found {num_theta} filtered theta and {num_pi} filtered pi sequences ≥ {MIN_LENGTH:,} bp")
    return num_theta, num_pi, theta_edge_total, theta_middle_total, pi_edge_total, pi_middle_total

def load_data(file_path, num_theta, num_pi, theta_edge_total, theta_middle_total, pi_edge_total, pi_middle_total):
    """
    Second pass: Load data into pre-allocated arrays efficiently.

    Args:
        file_path (Path): Input file path.
        num_theta, num_pi (int): Number of theta/pi sequences.
        theta_edge_total, theta_middle_total, pi_edge_total, pi_middle_total (int): Total sizes.

    Returns:
        tuple: Cleaned theta/pi edge/middle arrays.
    """
    theta_edge = np.empty(theta_edge_total, dtype=np.float32)
    theta_middle = np.empty(theta_middle_total, dtype=np.float32)
    pi_edge = np.empty(pi_edge_total, dtype=np.float32)
    pi_middle = np.empty(pi_middle_total, dtype=np.float32)

    theta_edge_idx = 0
    theta_middle_idx = 0
    pi_edge_idx = 0
    pi_middle_idx = 0
    line_num = 0

    with open(file_path, 'r') as f:
        while True:
            try:
                header = next(f).strip()
                line_num += 1
                result = parse_header(header)
                if result:
                    metric, length, full_header = result
                    data_line = next(f).strip()
                    line_num += 1
                    print(f"DATA LINE (Line {line_num}): {data_line[:100]}{'...' if len(data_line) > 100 else ''}")
                    data = np.array([float(x) if x.upper() != 'NA' else np.nan for x in data_line.split(',')],
                                  dtype=np.float32)

                    if len(data) != length:
                        print(f"WARNING: Line {line_num-1} - Length mismatch in {full_header[:50]}... "
                              f"({len(data):,} vs {length:,})")
                        continue

                    if length < 2 * EDGE_SIZE:
                        print(f"WARNING: Line {line_num-1} - Sequence too short: {length:,} bp")
                        continue

                    edge_left = data[:EDGE_SIZE]
                    edge_right = data[-EDGE_SIZE:]
                    middle = data[EDGE_SIZE:-EDGE_SIZE]

                    if metric == "theta":
                        theta_edge[theta_edge_idx:theta_edge_idx + EDGE_SIZE] = edge_left
                        theta_edge_idx += EDGE_SIZE
                        theta_edge[theta_edge_idx:theta_edge_idx + EDGE_SIZE] = edge_right
                        theta_edge_idx += EDGE_SIZE
                        theta_middle[theta_middle_idx:theta_middle_idx + len(middle)] = middle
                        theta_middle_idx += len(middle)
                        print(f"DEBUG: Theta {full_header[:50]}...: {2*EDGE_SIZE:,} edge, {len(middle):,} middle")
                    elif metric == "pi":
                        pi_edge[pi_edge_idx:pi_edge_idx + EDGE_SIZE] = edge_left
                        pi_edge_idx += EDGE_SIZE
                        pi_edge[pi_edge_idx:pi_edge_idx + EDGE_SIZE] = edge_right
                        pi_edge_idx += EDGE_SIZE
                        pi_middle[pi_middle_idx:pi_middle_idx + len(middle)] = middle
                        pi_middle_idx += len(middle)
                        print(f"DEBUG: Pi {full_header[:50]}...: {2*EDGE_SIZE:,} edge, {len(middle):,} middle")
            except StopIteration:
                break
            except ValueError as e:
                print(f"ERROR: Line {line_num} - Data parsing failed: {e}")
                continue

    # Clean NaNs
    theta_edge_clean = theta_edge[~np.isnan(theta_edge)]
    theta_middle_clean = theta_middle[~np.isnan(theta_middle)]
    pi_edge_clean = pi_edge[~np.isnan(pi_edge)]
    pi_middle_clean = pi_middle[~np.isnan(pi_middle)]

    # Print means and medians of full cleaned data
    print(f"Full Theta Edge: mean={np.mean(theta_edge_clean):.6f}, median={np.median(theta_edge_clean):.6f}")
    print(f"Full Theta Middle: mean={np.mean(theta_middle_clean):.6f}, median={np.median(theta_middle_clean):.6f}")
    print(f"Full Pi Edge: mean={np.mean(pi_edge_clean):.6f}, median={np.median(pi_edge_clean):.6f}")
    print(f"Full Pi Middle: mean={np.mean(pi_middle_clean):.6f}, median={np.median(pi_middle_clean):.6f}")

    print(f"Loaded: Theta edge={len(theta_edge_clean):,}, middle={len(theta_middle_clean):,}, "
          f"Pi edge={len(pi_edge_clean):,}, middle={len(pi_middle_clean):,}")
    return theta_edge_clean, theta_middle_clean, pi_edge_clean, pi_middle_clean

def significance_test(edge, middle, metric):
    """
    Perform one-sided Mann-Whitney U test and compute mean/median differences.

    Args:
        edge (np.ndarray): Edge data.
        middle (np.ndarray): Middle data.
        metric (str): 'Theta' or 'Pi'.
    """
    if len(edge) < 10 or len(middle) < 10:
        print(f"WARNING: Too few {metric} values (edge={len(edge):,}, middle={len(middle):,})")
        return

    mean_diff = np.nanmean(middle) - np.nanmean(edge)
    median_diff = np.nanmedian(middle) - np.nanmedian(edge)
    stat, p = mannwhitneyu(middle, edge, alternative='greater')
    print(f"{metric} (middle > edge): U={stat:,}, p={p}, "
          f"Mean Diff (middle - edge)={mean_diff}, Median Diff (middle - edge)={median_diff}")

def downsample(data, max_points=MAX_PLOT_POINTS):
    """
    Downsample data for plotting.

    Args:
        data (np.ndarray): Data to downsample.
        max_points (int): Max number of points.

    Returns:
        np.ndarray: Downsampled data.
    """
    if len(data) > max_points:
        return np.random.choice(data, size=max_points, replace=False)
    return data

def create_smooth_log_plots(theta_edge, theta_middle, pi_edge, pi_middle, output_dir):
    """
    Generate smooth plots with log-scaled axes using KDE, with added diagnostics.

    Args:
        theta_edge, theta_middle, pi_edge, pi_middle (np.ndarray): Data arrays.
        output_dir (Path): Directory to save the plot.

    Returns:
        Path: Path to saved plot or None if failed.
    """
    print("Generating smooth log-scaled plots...")
    if (len(theta_edge) < 10 or len(theta_middle) < 10 or
        len(pi_edge) < 10 or len(pi_middle) < 10):
        print("ERROR: Insufficient data for plotting (need at least 10 points per category)")
        return None

    # Downsample for efficiency
    theta_edge_plot = downsample(theta_edge)
    theta_middle_plot = downsample(theta_middle)
    pi_edge_plot = downsample(pi_edge)
    pi_middle_plot = downsample(pi_middle)

    # Filter positive values for log scale
    theta_edge_pos = theta_edge_plot[theta_edge_plot > 0]
    theta_middle_pos = theta_middle_plot[theta_middle_plot > 0]
    pi_edge_pos = pi_edge_plot[pi_edge_plot > 0]
    pi_middle_pos = pi_middle_plot[pi_middle_plot > 0]

    if (len(theta_edge_pos) < 10 or len(theta_middle_pos) < 10 or
        len(pi_edge_pos) < 10 or len(pi_middle_pos) < 10):
        print("ERROR: Too few positive values for log-scale plotting")
        return None

    # Print means and medians of downsampled positive data
    print(f"Downsampled Theta Edge (positive): mean={np.mean(theta_edge_pos):.6f}, "
          f"median={np.median(theta_edge_pos):.6f}")
    print(f"Downsampled Theta Middle (positive): mean={np.mean(theta_middle_pos):.6f}, "
          f"median={np.median(theta_middle_pos):.6f}")
    print(f"Downsampled Pi Edge (positive): mean={np.mean(pi_edge_pos):.6f}, "
          f"median={np.median(pi_edge_pos):.6f}")
    print(f"Downsampled Pi Middle (positive): mean={np.mean(pi_middle_pos):.6f}, "
          f"median={np.median(pi_middle_pos):.6f}")

    # Compute KDE with adjusted bandwidth
    theta_edge_kde = gaussian_kde(theta_edge_pos, bw_method=BW_METHOD)
    theta_middle_kde = gaussian_kde(theta_middle_pos, bw_method=BW_METHOD)
    pi_edge_kde = gaussian_kde(pi_edge_pos, bw_method=BW_METHOD)
    pi_middle_kde = gaussian_kde(pi_middle_pos, bw_method=BW_METHOD)

    # Define x-range for plotting (log scale) with more points
    theta_x = np.logspace(np.log10(min(theta_edge_pos.min(), theta_middle_pos.min())),
                          np.log10(max(theta_edge_pos.max(), theta_middle_pos.max())), LOGSPACE_POINTS)
    pi_x = np.logspace(np.log10(min(pi_edge_pos.min(), pi_middle_pos.min())),
                       np.log10(max(pi_edge_pos.max(), pi_middle_pos.max())), LOGSPACE_POINTS)

    theta_edge_y = theta_edge_kde(theta_x)
    theta_middle_y = theta_middle_kde(theta_x)
    pi_edge_y = pi_edge_kde(pi_x)
    pi_middle_y = pi_middle_kde(pi_x)

    # Create figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Theta plot
    ax1.plot(theta_x, theta_edge_y, label='Edge', color='#4C78A8')
    ax1.plot(theta_x, theta_middle_y, label='Middle', color='#F28E2B')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('Filtered Theta (Log Scale)', fontsize=14)
    ax1.set_xlabel('Theta', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.legend()

    # Pi plot
    ax2.plot(pi_x, pi_edge_y, label='Edge', color='#4C78A8')
    ax2.plot(pi_x, pi_middle_y, label='Middle', color='#F28E2B')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('Filtered Pi (Log Scale)', fontsize=14)
    ax2.set_xlabel('Pi', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend()

    plt.tight_layout()
    plot_path = output_dir / 'filtered_theta_pi_smooth_log_1M.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"KDE plots saved to {plot_path}")
    return plot_path

def create_cdf_plots(theta_edge, theta_middle, pi_edge, pi_middle, output_dir):
    """
    Generate CDF plots for theta and pi data.

    Args:
        theta_edge, theta_middle, pi_edge, pi_middle (np.ndarray): Data arrays.
        output_dir (Path): Directory to save the plot.

    Returns:
        Path: Path to saved CDF plot or None if failed.
    """
    print("Generating CDF plots...")
    # Filter positive values for log scale
    theta_edge_pos = theta_edge[theta_edge > 0]
    theta_middle_pos = theta_middle[theta_middle > 0]
    pi_edge_pos = pi_edge[pi_edge > 0]
    pi_middle_pos = pi_middle[pi_middle > 0]

    if (len(theta_edge_pos) < 10 or len(theta_middle_pos) < 10 or
        len(pi_edge_pos) < 10 or len(pi_middle_pos) < 10):
        print("ERROR: Too few positive values for CDF plotting")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Theta CDF
    theta_edge_sorted = np.sort(theta_edge_pos)
    theta_middle_sorted = np.sort(theta_middle_pos)
    ax1.plot(theta_edge_sorted, np.linspace(0, 1, len(theta_edge_sorted)), label='Edge', color='#4C78A8')
    ax1.plot(theta_middle_sorted, np.linspace(0, 1, len(theta_middle_sorted)), label='Middle', color='#F28E2B')
    ax1.set_xscale('log')
    ax1.set_title('Theta CDF', fontsize=14)
    ax1.set_xlabel('Theta', fontsize=12)
    ax1.set_ylabel('Cumulative Probability', fontsize=12)
    ax1.legend()

    # Pi CDF
    pi_edge_sorted = np.sort(pi_edge_pos)
    pi_middle_sorted = np.sort(pi_middle_pos)
    ax2.plot(pi_edge_sorted, np.linspace(0, 1, len(pi_edge_sorted)), label='Edge', color='#4C78A8')
    ax2.plot(pi_middle_sorted, np.linspace(0, 1, len(pi_middle_sorted)), label='Middle', color='#F28E2B')
    ax2.set_xscale('log')
    ax2.set_title('Pi CDF', fontsize=14)
    ax2.set_xlabel('Pi', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.legend()

    plt.tight_layout()
    cdf_path = output_dir / 'filtered_theta_pi_cdf_log_1M.png'
    plt.savefig(cdf_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"CDF plot saved to {cdf_path}")
    return cdf_path

def main():
    """Main function to run the analysis."""
    start_time = time.time()

    file_path = Path('per_site_output.falsta')
    if not file_path.exists():
        print(f"ERROR: {file_path} not found!")
        return

    output_dir = file_path.parent

    # First pass: Count sequences and sizes
    num_theta, num_pi, theta_edge_total, theta_middle_total, pi_edge_total, pi_middle_total = count_sequences(file_path)
    if num_theta == 0 and num_pi == 0:
        print("No filtered sequences ≥ 1M bp found. Exiting.")
        return

    # Second pass: Load data efficiently
    theta_edge, theta_middle, pi_edge, pi_middle = load_data(file_path, num_theta, num_pi,
                                                            theta_edge_total, theta_middle_total,
                                                            pi_edge_total, pi_middle_total)

    # Significance tests
    print("\nSignificance Tests (One-Sided: Middle > Edge):")
    significance_test(theta_edge, theta_middle, "Theta")
    significance_test(pi_edge, pi_middle, "Pi")

    # Generate and open KDE plots
    kde_plot_path = create_smooth_log_plots(theta_edge, theta_middle, pi_edge, pi_middle, output_dir)
    if kde_plot_path and kde_plot_path.exists():
        try:
            if os.name == 'nt':  # Windows
                os.startfile(str(kde_plot_path))
            elif os.name == 'posix':  # MacOS/Linux
                cmd = 'open' if 'darwin' in os.sys.platform else 'xdg-open'
                os.system(f'{cmd} "{kde_plot_path}"')
        except Exception as e:
            print(f"WARNING: Failed to open KDE plot: {e}")

    # Generate and open CDF plots
    cdf_plot_path = create_cdf_plots(theta_edge, theta_middle, pi_edge, pi_middle, output_dir)
    if cdf_plot_path and cdf_plot_path.exists():
        try:
            if os.name == 'nt':  # Windows
                os.startfile(str(cdf_plot_path))
            elif os.name == 'posix':  # MacOS/Linux
                cmd = 'open' if 'darwin' in os.sys.platform else 'xdg-open'
                os.system(f'{cmd} "{cdf_plot_path}"')
        except Exception as e:
            print(f"WARNING: Failed to open CDF plot: {e}")

    print(f"Total execution time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
