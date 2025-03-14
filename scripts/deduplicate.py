import pandas as pd

# Load original data
df = pd.read_csv('all_pairwise_results.csv')
original_rows = len(df)

# Remove duplicates (keeping the first occurrence)
df = df.drop_duplicates()
removed_rows = original_rows - len(df)

# Overwrite existing file with deduplicated data
df.to_csv('all_pairwise_results.csv', index=False)

# Reporting
print(f"Removed {removed_rows} duplicate rows.")
print(f"Rows remaining after deduplication: {len(df)}")
