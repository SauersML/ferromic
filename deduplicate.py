import pandas as pd
import numpy as np
import re

# Load original data
original_df = pd.read_csv('all_pairwise_results.csv')
original_rows = len(original_df)

# Remove duplicates
df = original_df.drop_duplicates()
deleted_rows = original_rows - len(df)
print(f"Removed {deleted_rows} duplicate rows.")

# Extract transcript ID and genomic coordinates
df['transcript_id'] = df['CDS'].str.extract(r'(ENST\d+\.\d+)')[0]
coords = df['CDS'].str.extract(r'chr_(\w+)_start_(\d+)_end_(\d+)')
df['chrom'] = 'chr' + coords[0]
df['start'] = coords[1].astype(int)
df['end'] = coords[2].astype(int)

# Count unique transcripts and coordinates
num_transcripts = df['transcript_id'].nunique()
num_unique_coords = df.groupby(['chrom', 'start', 'end']).ngroups

print(f"Unique transcripts found: {num_unique_transcripts}")
print(f"Unique coordinates found: {num_unique_coords}")

# Check for transcripts mapping to multiple coordinates
transcript_coord_counts = df.groupby('transcript_id')[['chrom', 'start', 'end']].nunique()
multi_coord_transcripts = transcript_coord_counts[
    (transcript_coord_counts > 1).any(axis=1)
]

if not multi_coord_transcripts.empty:
    print(f"\nTranscripts mapping to multiple coordinates ({len(multi_coord_transcripts)}):")
    print(multi_coord_transcripts.reset_index())
else:
    print("\nNo transcripts mapping to multiple coordinates.")

# Check for coordinates mapping to multiple transcripts
coord_transcript_counts = df.groupby(['chrom', 'start', 'end'])['transcript_id'].nunique()
multi_transcript_coords = coord_transcript_counts[coord_transcript_counts > 1]

if not multi_transcript_coords.empty:
    print(f"\nFound {len(multi_transcript_coords)} coordinates containing multiple transcripts.")
    print(multi_transcript_coords.reset_index().rename(columns={'transcript_id':'num_transcripts'}))
else:
    print("\nNo coordinates contain multiple transcripts.")

# Final reporting
removed_rows = original_rows - len(df)
print(f"\nRows removed (duplicates): {original_rows - len(df)}")
print(f"Rows remaining after deduplication: {len(df)}")

# Save deduplicated file for future analysis
df.to_csv('all_pairwise_results_dedup.csv', index=False)
print("\nDeduplicated data saved as 'all_pairwise_results.csv'.")
