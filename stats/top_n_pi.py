import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import heapq
import os
from pathlib import Path
import time

def parse_header(line):
    # Parse only lines explicitly labeled as filtered_pi
    if not line.strip().startswith(">filtered_pi"):
        print(f"DEBUG: Skipping non-filtered_pi line: {line.strip()[:50]}...")
        return None, None
    parts = line.strip().split('_')
    try:
        start_idx = parts.index('start')
        end_idx = parts.index('end')
        start = int(parts[start_idx + 1])
        end = int(parts[end_idx + 1])
        length = end - start
        if length <= 0:
            print(f"WARNING: Invalid length (<= 0) for header: {line.strip()[:50]}...")
            return None, None
        return length, line.strip()
    except (ValueError, IndexError):
        print(f"WARNING: Malformed header: {line.strip()[:50]}...")
        return None, None

def get_top_n_sequences(file_path, n=6):
    # Find top N filtered_pi sequences by length
    start_time = time.time()
    print(f"INFO: Parsing {file_path} for top {n} filtered_pi sequences at {time.ctime()}")
    
    top_n = []  # Min-heap: (length, header, file_offset)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        file_offset = 0
        line_count = 0
        header_count = 0
        for line in f:
            length, header = parse_header(line)
            if length is not None:
                header_count += 1
                data_offset = file_offset + len(line.encode('utf-8'))
                entry = (length, header, data_offset)
                if len(top_n) < n:
                    heapq.heappush(top_n, entry)
                    print(f"DEBUG: Added filtered_pi sequence {header_count}, length={length}, heap size={len(top_n)}")
                elif length > top_n[0][0]:
                    old = heapq.heapreplace(top_n, entry)
                    print(f"DEBUG: Replaced length={old[0]} with length={length}, header={header[:50]}...")
                file_offset += len(line.encode('utf-8'))
            line_count += 1
            if line_count % 10000 == 0:
                print(f"INFO: Scanned {line_count} lines, found {header_count} filtered_pi headers")
    
    print(f"INFO: Parsed {line_count} lines, found {header_count} filtered_pi headers in {time.time() - start_time:.2f}s")
    if not top_n:
        print("ERROR: No valid filtered_pi sequences found!")
        return []
    
    top_n = heapq.nlargest(n, top_n)  # Get top N in descending order
    print(f"INFO: Top {n} filtered_pi lengths: {[x[0] for x in top_n]}")
    return top_n

def load_data_for_headers(file_path, top_n):
    # Load data only for top N filtered_pi sequences, handling empty strings
    start_time = time.time()
    print(f"INFO: Loading data for {len(top_n)} filtered_pi sequences at {time.ctime()}")
    
    data_dict = {}
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for length, header, offset in top_n:
            print(f"DEBUG: Seeking to offset {offset} for header: {header[:50]}...")
            f.seek(offset)
            data_line = f.readline().strip()
            print(f"DEBUG: Read data line with {len(data_line)} chars")
            # Split and inspect data
            data_split = data_line.split(',')
            print(f"DEBUG: Split into {len(data_split)} values, first few: {data_split[:5]}")
            # Handle empty strings and 'NA'
            try:
                data = [float(x) if x != 'NA' and x != '' else np.nan for x in data_split]
                data = np.array(data, dtype=np.float32)
                data_dict[header] = (length, data)
                print(f"DEBUG: Loaded {len(data)} values for length={length}, header={header[:50]}...")
            except ValueError as e:
                print(f"ERROR: Failed to parse data for {header[:50]}...: {e}")
                print(f"DEBUG: Problematic data sample: {data_split[:10]}")
                data_dict[header] = (length, np.array([], dtype=np.float32))  # Empty array as fallback
    
    print(f"INFO: Loaded data for {len(data_dict)} filtered_pi sequences in {time.time() - start_time:.2f}s")
    return data_dict

def plot_top_n_sequences(data_dict):
    # Plot top N filtered_pi sequences with 1000 bp moving average
    start_time = time.time()
    print(f"INFO: Generating plot at {time.ctime()}")
    
    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, (header, (length, data)) in enumerate(data_dict.items()):
        print(f"DEBUG: Processing filtered_pi sequence {i+1}, length={length}, points={len(data)}")
        if len(data) == 0:
            print(f"WARNING: No data to plot for {header[:50]}...")
            continue
        series = pd.Series(data)
        smoothed = series.rolling(window=1000, min_periods=1, center=True).mean().to_numpy()
        positions = np.arange(len(data))
        color = plt.cm.Set1(i % 9)
        ax.plot(positions, smoothed, color=color, label=header, lw=2)
        print(f"DEBUG: Plotted smoothed data for {header[:50]}... with {len(smoothed)} points")
    
    ax.set_xlabel("Position (bp)", fontsize=16)
    ax.set_ylabel("Smoothed Pi Value (1000 bp window)", fontsize=16)
    ax.set_title(f"Top {len(data_dict)} Longest Filtered Pi Sequences", fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(title="Filtered Pi Headers", loc='upper right', fontsize=12, title_fontsize=14)
    
    plot_path = Path.home() / "top_filtered_pi_smoothed.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"INFO: Plot saved to {plot_path} in {time.time() - start_time:.2f}s")
    return plot_path

def main():
    start_time = time.time()
    print(f"INFO: Script started at {time.ctime()}")
    
    file_path = 'per_site_diversity_output.falsta'
    if not os.path.exists(file_path):
        print(f"ERROR: File {file_path} not found!")
        return
    
    top_n = get_top_n_sequences(file_path, n=6)
    if not top_n:
        print("ERROR: No filtered_pi sequences to process. Exiting.")
        return
    
    data_dict = load_data_for_headers(file_path, top_n)
    if not data_dict:
        print("ERROR: Failed to load filtered_pi data. Exiting.")
        return
    
    plot_path = plot_top_n_sequences(data_dict)
    print(f"INFO: Plot path: {plot_path}")
    
    if os.name == 'nt':
        print("INFO: Opening plot on Windows")
        os.startfile(plot_path)
    elif os.name == 'posix':
        cmd = 'open' if 'darwin' in os.sys.platform else 'xdg-open'
        print(f"INFO: Opening plot with {cmd}")
        os.system(f'{cmd} "{plot_path}"')
    
    print(f"INFO: Total execution time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
