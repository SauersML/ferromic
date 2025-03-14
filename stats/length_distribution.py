import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load the CSV data and compute the length of each region.
    df = pd.read_csv("inv_info.csv")
    df['length'] = df['region_end'] - df['region_start']
    
    # Sort by region length and print the chromosome and coordinates of the 10 smallest regions.
    smallest_regions = df.sort_values(by='length', ascending=True).head(10)
    print("Top 10 smallest ranges (chr, start, end):")
    for _, row in smallest_regions.iterrows():
        print(f"{row['chr']}: {row['region_start']} - {row['region_end']} (length: {row['length']})")
    
    # Apply a style for the plots.
    plt.style.use('ggplot')
    
    # Create two subplots: one for the full distribution and one zoomed into the 0-10,000 bp range.
    fig, (ax_full, ax_zoom) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    
    # Full distribution histogram.
    ax_full.hist(df['length'], bins=50, color='mediumseagreen', edgecolor='black')
    ax_full.set_title("Full Distribution of Region Lengths")
    ax_full.set_xlabel("Length (bp)")
    ax_full.set_ylabel("Frequency")
    ax_full.grid(True)
    
    # Zoomed in histogram for lengths between 0 and 10,000 bp.
    ax_zoom.hist(df['length'], bins=50, range=(0, 10000), color='cornflowerblue', edgecolor='black')
    ax_zoom.set_title("Zoomed Distribution (0 - 10,000 bp)")
    ax_zoom.set_xlabel("Length (bp)")
    ax_zoom.set_ylabel("Frequency")
    ax_zoom.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
