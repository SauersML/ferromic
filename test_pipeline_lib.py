import sys
import os
import pandas as pd
import logging

# Setup logging to verify output
logging.basicConfig(level=logging.INFO)

# Append current dir to sys.path so we can import cds.pipeline_lib
sys.path.append(os.getcwd())

try:
    import cds.pipeline_lib as lib
except ImportError:
    print("Could not import cds.pipeline_lib via 'cds.pipeline_lib'. Trying direct import by path.")
    sys.path.append(os.path.join(os.getcwd(), 'cds'))
    import pipeline_lib as lib

print("Imported pipeline_lib successfully.")

# Test log_runtime_environment
lib.log_runtime_environment("TEST")

# Create a dummy TSV
tsv_path = "test_data.tsv"
with open(tsv_path, "w") as f:
    f.write("col1\tcol2\n1\tA\n2\tB\n")

print(f"Created {tsv_path}")

# Test safe_read_tsv_via_subprocess
print("Testing safe_read_tsv_via_subprocess...")
df = lib.safe_read_tsv_via_subprocess(tsv_path, log_prefix="[TEST]")

if df is not None:
    print("Read DataFrame successfully:")
    print(df)
    if df.shape == (2, 2) and df.iloc[0]['col1'] == 1:
        print("PASS: Content is correct.")
    else:
        print("FAIL: Content is incorrect.")
else:
    print("FAIL: safe_read_tsv_via_subprocess returned None.")

# Clean up
if os.path.exists(tsv_path):
    os.remove(tsv_path)
