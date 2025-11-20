import os
import sys
import glob
import logging
import pandas as pd
from datetime import datetime
import numpy as np

# Ensure pipeline_lib is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import pipeline_lib as lib
except ImportError:
    print("Error: Could not import pipeline_lib.", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 1. Find all partial result files
    files = glob.glob("partial_results_*.tsv")
    if not files:
        logging.warning("No partial_results_*.tsv files found. Nothing to aggregate.")
        # We create an empty result file to satisfy artifacts if needed, or just exit
        return

    logging.info(f"Found {len(files)} partial result files. Aggregating...")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, sep='\t')
            dfs.append(df)
        except Exception as e:
            logging.error(f"Failed to read {f}: {e}")

    if not dfs:
        logging.error("No valid dataframes loaded.")
        sys.exit(1)

    results_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Aggregated {len(results_df)} total rows.")

    # 2. Ensure columns and formatting (copied from omega_test.py)
    ordered_columns = [
        'region', 'gene', 'status',
        'bm_p_value', 'bm_q_value', 'bm_lrt_stat',
        'bm_omega_inverted', 'bm_omega_direct', 'bm_omega_background', 'bm_kappa',
        'bm_lnl_h1', 'bm_lnl_h0',
        'cmc_p_value', 'cmc_q_value', 'cmc_lrt_stat',
        'cmc_p0', 'cmc_p1', 'cmc_p2', 'cmc_omega0', 'cmc_omega2_direct', 'cmc_omega2_inverted', 'cmc_kappa',
        'cmc_lnl_h1', 'cmc_lnl_h0',
        'n_leaves_region', 'n_leaves_gene', 'n_leaves_pruned',
        'chimp_in_region', 'chimp_in_pruned',
        'taxa_used', 'reason'
    ]

    for col in ordered_columns:
        if col not in results_df.columns:
            results_df[col] = np.nan

    # 3. Apply FDR Correction
    if not results_df.empty:
        results_df = lib.compute_fdr(results_df)

    # 4. Write Final Output
    results_df = results_df[ordered_columns]

    output_filename = f"full_paml_results_{datetime.now().strftime('%Y-%m-%d')}.tsv"
    results_df.to_csv(output_filename, sep='\t', index=False, float_format='%.6g')

    logging.info(f"Final results written to {output_filename}")

    # Log summary stats
    counts = results_df['status'].value_counts().to_dict()
    logging.info("Status counts: " + str(counts))

    sig = results_df[(results_df['status'] == 'success') &
                     ((results_df['bm_q_value'] < lib.FDR_ALPHA) | (results_df['cmc_q_value'] < lib.FDR_ALPHA))]
    if not sig.empty:
         logging.info(f"Significant tests: {len(sig)}")
    else:
         logging.info("No significant tests.")

if __name__ == "__main__":
    main()
