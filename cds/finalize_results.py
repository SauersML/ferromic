import os
import sys
import io
import tracemalloc
import time

import faulthandler
faulthandler.enable(all_threads=True)

# Set environment variables to limit thread usage and prevent OOM/segfaults in constrained environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# Ensure pipeline_lib is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import pipeline_lib as lib
except ImportError:
    print("Error: Could not import pipeline_lib.", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


ORDERED_COLUMNS = [
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


def _default_record(region, gene, seed_columns):
    rec = {c: np.nan for c in ORDERED_COLUMNS + list(seed_columns)}
    rec.update({
        'region': region,
        'gene': gene,
        'status': 'PAML optimization failed',
        'reason': '',
        'models_present': set(),
        'reasons': [],
        'source_statuses': set()
    })
    return rec


def _merge_branch_info(record, row):
    if not pd.isna(row.get('bm_p_value')) and pd.isna(record.get('bm_p_value')):
        for col in ['bm_p_value', 'bm_lrt_stat', 'bm_omega_inverted', 'bm_omega_direct',
                    'bm_omega_background', 'bm_kappa', 'bm_lnl_h1', 'bm_lnl_h0']:
            if col in row:
                record[col] = row.get(col)


def _merge_clade_info(record, row, model):
    if model == 'h0' and pd.isna(record.get('cmc_lnl_h0')) and not pd.isna(row.get('cmc_lnl_h0')):
        record['cmc_lnl_h0'] = row.get('cmc_lnl_h0')
        record['cmc_h0_key'] = row.get('cmc_h0_key')
        record['h0_winner_seed'] = row.get('h0_winner_seed')
    if model == 'h1' and pd.isna(record.get('cmc_lnl_h1')) and not pd.isna(row.get('cmc_lnl_h1')):
        record['cmc_lnl_h1'] = row.get('cmc_lnl_h1')
        record['cmc_h1_key'] = row.get('cmc_h1_key')
        record['h1_winner_seed'] = row.get('h1_winner_seed')
        for param in ['cmc_p0', 'cmc_p1', 'cmc_p2', 'cmc_omega0', 'cmc_omega2_direct', 'cmc_omega2_inverted', 'cmc_kappa']:
            if param in row and not pd.isna(row.get(param)):
                record[param] = row.get(param)


def _merge_metadata(record, row):
    for col in ['n_leaves_region', 'n_leaves_gene', 'n_leaves_pruned', 'chimp_in_region', 'chimp_in_pruned', 'taxa_used']:
        if pd.isna(record.get(col)) and col in row:
            record[col] = row.get(col)


def _merge_seed_data(record, row, seed_columns):
    for col in seed_columns:
        if col in row and not pd.isna(row.get(col)):
            record[col] = row.get(col)


def _finalize_record(record):
    h0_present = np.isfinite(record.get('cmc_lnl_h0', np.nan))
    h1_present = np.isfinite(record.get('cmc_lnl_h1', np.nan))

    # Compute clade statistics when both lnL values are present
    if h0_present and h1_present:
        diff = record['cmc_lnl_h1'] - record['cmc_lnl_h0']
        record['cmc_lrt_stat'] = 2 * diff

        if diff < -1e-6:
            # Preserve the statistic but do not produce a p-value
            record['cmc_p_value'] = np.nan
            record['reasons'].append(f"Optimization failure: H1 < H0 (diff={diff:.4g})")
        else:
            lrt_for_p = max(record['cmc_lrt_stat'], 0.0)
            record['cmc_p_value'] = float(lib.chi2.sf(lrt_for_p, df=1))

    branch_data_present = not pd.isna(record.get('bm_p_value')) or not all(pd.isna(record.get(col)) for col in ['bm_p_value', 'bm_lrt_stat'])

    runtime_statuses = {s for s in record.get('source_statuses', set()) if str(s).startswith('runtime')}

    if runtime_statuses:
        record['reasons'].append('Upstream runtime error observed')

    if not h0_present and not h1_present:
        status_tag = 'Missing clade model likelihoods for H0 and H1'
        record['reasons'].append('Missing clade model likelihoods for H0 and H1')
    elif h0_present and not h1_present:
        status_tag = 'H1 clade likelihood missing'
        record['reasons'].append('Missing H1 clade likelihood')
    elif h1_present and not h0_present:
        status_tag = 'H0 clade likelihood missing'
        record['reasons'].append('Missing H0 clade likelihood')
    else:
        diff = record['cmc_lnl_h1'] - record['cmc_lnl_h0']
        if diff >= -1e-6:
            status_tag = 'Complete data with H1 at least as good as H0'
        else:
            status_tag = 'Complete data but H1 worse than H0'

    status_tags = []
    if runtime_statuses:
        status_tags.append('Runtime error observed in upstream run')

    status_tags.append(status_tag)

    if not branch_data_present:
        status_tags.append('Branch model results missing')
    elif pd.isna(record.get('bm_p_value')) and pd.isna(record.get('bm_lrt_stat')):
        status_tags.append('Branch statistics missing')

    record['status'] = '; '.join(status_tags)

    if record['reasons']:
        record['reason'] = ' | '.join(sorted(set(filter(None, record['reasons']))))
    record.pop('reasons', None)
    record.pop('models_present', None)
    record.pop('source_statuses', None)
    return record


def main():
    files = sorted(glob.glob("partial_results_*.tsv"))
    if not files:
        logging.warning("No partial_results_*.tsv files found. Nothing to aggregate.")
        return

    lib.log_runtime_environment("finalize_results")

    logging.info("Found %d partial result files. Aggregating...", len(files))
    dfs = []

    for i, f in enumerate(files, start=1):
        # Cheap file info before we touch pandas at all
        try:
            st = os.stat(f)
            size = st.st_size
        except OSError as e:
            logging.error("Skipping %s: os.stat failed: %r", f, e)
            continue

        logging.info("=" * 80)
        logging.info("FILE %d/%d: %s", i, len(files), f)
        logging.info("  File size: %d bytes (%.1f KB)", size, size / 1024.0)
        logging.info("  DataFrames in list so far: %d", len(dfs))

        if size == 0:
            logging.warning("  Skipping empty file.")
            continue

        # Peek at first couple of KB directly (helpful even if child segfaults)
        try:
            with open(f, "rb") as bin_f:
                raw_head = bin_f.read(2048)
            logging.info("  Raw head bytes: %r", raw_head[:200])
            try:
                first_line = raw_head.splitlines()[0].decode("utf-8", "replace")
            except Exception as e:
                first_line = f"<decode error: {e!r}>"
            logging.info("  First line (decoded): %r", first_line)
        except Exception as e:
            logging.error("  Failed to read head of %s: %r", f, e)
            continue

        # Make sure previous logs are flushed before the child is spawned
        for handler in logging.root.handlers:
            try:
                handler.flush()
            except Exception:
                pass

        logging.info("  → Parsing TSV via lib.safe_read_tsv_via_subprocess")

        df = lib.safe_read_tsv_via_subprocess(f, log_prefix=f"[file {i}/{len(files)}] ")

        if df is None:
            logging.error("  ✗ Failed to parse %s; moving on to next file.", f)
            continue

        logging.info("  ✓ Parsed successfully.")
        logging.info("    Shape: %s (rows × cols)", df.shape)
        try:
            mem_kb = df.memory_usage(deep=True).sum() / 1024.0
            logging.info("    Memory: %.1f KB", mem_kb)
        except Exception as e:
            logging.warning("    Failed to compute memory usage: %r", e)

        # Summarise dtypes
        dtype_counts = {}
        for dt in set(df.dtypes):
            dtype_counts[str(dt)] = int((df.dtypes == dt).sum())
        logging.info("    Dtypes: %s", dtype_counts)

        dfs.append(df)

    if not dfs:
        logging.error("No valid dataframes loaded.")
        sys.exit(1)

    logging.info("Concatenating %d dataframes...", len(dfs))
    raw = pd.concat(dfs, ignore_index=True)
    raw['paml_model'] = raw.get('paml_model', pd.Series(['both'] * len(raw))).fillna('both').str.lower()
    logging.info(f"Aggregated {len(raw)} total rows across models.")

    seed_columns = [c for c in raw.columns if c.startswith(('h0_s', 'h1_s'))]

    combined = {}
    for _, row in raw.iterrows():
        region = row.get('region')
        gene = row.get('gene')
        key = (region, gene)
        model = row.get('paml_model', 'both')
        rec = combined.setdefault(key, _default_record(region, gene, seed_columns))
        rec['models_present'].add(model)

        if 'status' in row and not pd.isna(row.get('status')):
            rec['source_statuses'].add(str(row.get('status')))

        if str(row.get('status')).startswith('runtime') or str(row.get('status')).startswith('paml_optim'):
            if row.get('reason'):
                rec['reasons'].append(str(row.get('reason')))

        _merge_branch_info(rec, row)
        _merge_metadata(rec, row)
        _merge_seed_data(rec, row, seed_columns)

        if model in ('h0', 'h1', 'both'):
            if model == 'both':
                _merge_clade_info(rec, row, 'h0')
                _merge_clade_info(rec, row, 'h1')
            else:
                _merge_clade_info(rec, row, model)

    finalized_rows = [_finalize_record(rec) for rec in combined.values()]
    results_df = pd.DataFrame(finalized_rows)

    for col in ORDERED_COLUMNS:
        if col not in results_df.columns:
            results_df[col] = np.nan

    results_df = lib.compute_fdr(results_df)

    remaining_cols = [c for c in results_df.columns if c not in ORDERED_COLUMNS]

    seed_sort_order = {'h0': 0, 'h1': 1}

    def _seed_key(col):
        parts = col.split('_')
        hypothesis = parts[0] if parts else ''
        seed = parts[1] if len(parts) > 1 else ''
        metric = '_'.join(parts[2:]) if len(parts) > 2 else ''

        seed_num = 0
        if seed.startswith('s') and seed[1:].isdigit():
            seed_num = int(seed[1:])

        return (seed_sort_order.get(hypothesis, 99), seed_num, metric, col)

    seed_cols_sorted = sorted((c for c in seed_columns if c in results_df.columns), key=_seed_key)
    other_cols = sorted(c for c in remaining_cols if c not in seed_columns)

    ordered_with_dynamic = ORDERED_COLUMNS + seed_cols_sorted + other_cols
    results_df = results_df[ordered_with_dynamic]

    output_filename = f"full_paml_results_{datetime.now().strftime('%Y-%m-%d')}.tsv"
    results_df.to_csv(output_filename, sep='\t', index=False, float_format='%.6g')
    logging.info(f"Final results written to {output_filename}")

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
