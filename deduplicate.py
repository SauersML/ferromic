import pandas as pd
import numpy as np
import re

# Load data and remove duplicates
df = pd.read_csv('all_pairwise_results.csv').drop_duplicates()

# Extract transcript ID and genomic coordinates
df['transcript_id'] = df['CDS'].str.extract(r'(ENST\d+\.\d+)')[0]
coords = df['CDS'].str.extract(r'chr_(\w+)_start_(\d+)_end_(\d+)')
df['chrom'] = 'chr' + coords[0]
df['start'] = coords[1].astype(int)
df['end'] = coords[2].astype(int)

# Analysis summary
unique_transcripts = df['transcript_id'].nunique()
unique_coords = df[['chrom', 'start', 'end']].drop_duplicates().shape[0]

print(f"Unique transcripts: {unique_transcripts}")
print(f"Unique coordinates: {unique_coords}")

# Check for discrepancies
transcript_coord_counts = df.groupby('transcript_id')[['chrom', 'start', 'end']].nunique()
discrepant_transcripts = transcript_coord_counts[
    (transcript_coord_counts['chrom'] > 1) | 
    (transcript_coord_counts['start'] > 1) | 
    (transcript_coord_counts['end'] > 1)
]

if not discrepant_transcripts.empty:
    print("\nTranscripts with multiple coordinate mappings:")
    print(discrepant_transcripts.reset_index())
else:
    print("\nNo transcripts map to multiple coordinates.")

# Save deduplicated data
df.to_csv('all_pairwise_results_dedup.csv', index=False)
print("\nDeduplicated data saved to 'all_pairwise_results.csv'")
