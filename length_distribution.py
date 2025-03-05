import matplotlib.pyplot as plt
from pathlib import Path

def get_lengths(file_path):
    lengths = []
    with open(file_path, 'r') as f:
        for line in f:
            if 'start_' in line and 'end_' in line:
                parts = line.split('_')
                start = int(parts[5])  # After 'start'
                end = int(parts[7])    # After 'end'
                length = end - start
                if length > 0:
                    lengths.append(length)
                else:
                    print(f"Skipping invalid length: {line.strip()}")
    return lengths

def make_histograms(lengths, output_path):
    if not lengths:
        print("No lengths found.")
        return
    
    # Filter lengths between 0 and 10,000
    short_lengths = [x for x in lengths if 0 <= x <= 10000]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1])
    
    # Style settings
    plt.style.use('seaborn-v0_8-whitegrid')  # Clean, modern look
    
    # Subplot 1: All lengths
    ax1.hist(lengths, bins=30, color='#4C78A8', edgecolor='white', linewidth=1.5, alpha=0.9)
    ax1.set_title('All Sequence Lengths', fontsize=18, fontweight='bold', pad=15)
    ax1.set_xlabel('Length (bp)', fontsize=14, labelpad=10)
    ax1.set_ylabel('Frequency', fontsize=14, labelpad=10)
    ax1.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_facecolor('#F5F6F5')  # Light background
    
    # Subplot 2: Lengths 0 to 10,000
    if short_lengths:
        ax2.hist(short_lengths, bins=30, color='#F28E2B', edgecolor='white', linewidth=1.5, alpha=0.9)
    ax2.set_title('Sequence Lengths (0 - 10,000 bp)', fontsize=18, fontweight='bold', pad=15)
    ax2.set_xlabel('Length (bp)', fontsize=14, labelpad=10)
    ax2.set_ylabel('Frequency', fontsize=14, labelpad=10)
    ax2.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_facecolor('#F5F6F5')  # Light background
    
    # Ensure numeric ticks without scientific notation
    ax1.ticklabel_format(style='plain', axis='x')
    ax2.ticklabel_format(style='plain', axis='x')
    
    # Adjust layout and save
    plt.tight_layout(pad=3.0)
    fig.suptitle('Distribution of Sequence Lengths', fontsize=22, fontweight='bold', y=1.05)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved histograms to {output_path}")

def main():
    file_path = 'per_site_output.falsta'
    output_path = Path.home() / 'lengths_histograms.png'
    
    lengths = get_lengths(file_path)
    if lengths:
        make_histograms(lengths, output_path)
        
        # Open the plot
        import os
        if os.name == 'nt':
            os.startfile(output_path)
        elif os.name == 'posix':
            os.system(f'open "{output_path}"' if 'darwin' in os.sys.platform else f'xdg-open "{output_path}"')
    else:
        print("No valid data to plot.")

if __name__ == "__main__":
    main()
