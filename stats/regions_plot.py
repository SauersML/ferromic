import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import time

# Constants
MIN_LENGTH = 500_000  # Minimum sequence length (1M bp)
EDGE_SIZE = 100_000     # Number of positions from each edge

def parse_header(line):
    """Parse header line for filtered pi sequences only."""
    if not line.startswith(">") or "filtered_pi" not in line.lower():
        return None
    header = line[1:].strip()
    parts = header.split('_')
    try:
        start_idx = parts.index("start")
        end_idx = parts.index("end")
        start = int(parts[start_idx + 1])
        end = int(parts[end_idx + 1])
        length = end - start + 1
        if length >= MIN_LENGTH:
            return length, header
        return None
    except (ValueError, IndexError):
        return None

def load_and_average_data(file_path):
    """Load filtered Pi data and compute averages for beginning, middle, and end."""
    pi_beginning = []
    pi_middle = []
    pi_end = []
    
    with open(file_path, 'r') as f:
        while True:
            try:
                header = next(f).strip()
                result = parse_header(header)
                if result:
                    length, full_header = result
                    data_line = next(f).strip()
                    data = np.array([float(x) if x.upper() != 'NA' else np.nan for x in data_line.split(',')],
                                  dtype=np.float32)
                    
                    if len(data) != length:
                        print(f"WARNING: Length mismatch in {full_header[:50]}... ({len(data):,} vs {length:,})")
                        continue
                    
                    if length < 2 * EDGE_SIZE:
                        continue
                    
                    # Extract beginning (left edge), middle, and end (right edge)
                    beginning = data[:EDGE_SIZE]
                    middle = data[EDGE_SIZE:-EDGE_SIZE]
                    end = data[-EDGE_SIZE:]
                    
                    # Compute means, excluding NaN
                    pi_beginning.append(np.nanmean(beginning))
                    pi_middle.append(np.nanmean(middle))
                    pi_end.append(np.nanmean(end))
                    
                    print(f"Processed Filtered Pi {full_header[:50]}...: "
                          f"beginning={np.nanmean(beginning):.6f}, "
                          f"middle={np.nanmean(middle):.6f}, "
                          f"end={np.nanmean(end):.6f}")
            except StopIteration:
                break
            except ValueError as e:
                print(f"ERROR: Data parsing failed: {e}")
                continue
    
    return np.array(pi_beginning), np.array(pi_middle), np.array(pi_end)

def create_plot(pi_beginning, pi_middle, pi_end, output_dir):
    """Create a plot with three points per filtered Pi sequence connected by lines."""
    if len(pi_beginning) == 0:
        print("ERROR: No valid filtered Pi sequences found for plotting")
        return None
    
    # Compute average values across all sequences
    avg_beginning = np.nanmean(pi_beginning)
    avg_middle = np.nanmean(pi_middle)
    avg_end = np.nanmean(pi_end)
    
    print(f"Average Filtered Pi: Beginning={avg_beginning:.6f}, Middle={avg_middle:.6f}, End={avg_end:.6f}")
    
    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot individual sequences
    for i in range(len(pi_beginning)):
        if not (np.isnan(pi_beginning[i]) or np.isnan(pi_middle[i]) or np.isnan(pi_end[i])):
            ax.plot([0, 1, 2], [pi_beginning[i], pi_middle[i], pi_end[i]], 
                   color='gray', alpha=0.3, linewidth=1)
    
    # Plot average line
    ax.plot([0, 1, 2], [avg_beginning, avg_middle, avg_end], 
            color='blue', linewidth=2, label='Average Filtered Pi', marker='o')
    
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Beginning', 'Middle', 'End'], fontsize=14)
    ax.set_xlabel('Position in Sequence', fontsize=15)
    ax.set_ylabel('Pi Value', fontsize=15)
    ax.set_title(f'Filtered Pi Values Across Sequences (n={len(pi_beginning)})', fontsize=17)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(fontsize=13)
    
    plt.tight_layout()
    plot_path = output_dir / 'filtered_pi_beginning_middle_end.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")
    return plot_path

def main():
    """Main function to run the analysis."""
    start_time = time.time()
    
    file_path = Path('per_site_output.falsta')
    if not file_path.exists():
        print(f"ERROR: {file_path} not found!")
        return
    
    output_dir = file_path.parent
    
    # Load and average data
    pi_beginning, pi_middle, pi_end = load_and_average_data(file_path)
    
    if len(pi_beginning) == 0:
        print("No filtered Pi sequences ≥ 1M bp found. Exiting.")
        return
    
    # Generate and open plot
    plot_path = create_plot(pi_beginning, pi_middle, pi_end, output_dir)
    if plot_path and plot_path.exists():
        try:
            if os.name == 'nt':  # Windows
                os.startfile(str(plot_path))
            elif os.name == 'posix':  # MacOS/Linux
                cmd = 'open' if 'darwin' in os.sys.platform else 'xdg-open'
                os.system(f'{cmd} "{plot_path}"')
        except Exception as e:
            print(f"WARNING: Failed to open plot: {e}")
    
    print(f"Total execution time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
