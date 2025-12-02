import re
from typing import List

import numpy as np
import pandas as pd

STATUS_EPSILON = 1e-6
STATUS_RUN_PATTERN = re.compile(r'^status_run_\d+$')
DEFAULT_SOURCE_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/GRAND_PAML_RESULTS.tsv"


def _runtime_issue_present(row: pd.Series, status_run_cols: List[str]) -> bool:
    return any(
        str(row[c]).startswith('runtime') for c in status_run_cols if c in row and pd.notna(row[c])
    )


def compute_status_series(df: pd.DataFrame, epsilon: float = STATUS_EPSILON) -> pd.Series:
    status_run_cols = [c for c in df.columns if STATUS_RUN_PATTERN.match(c)]

    def build_status(row: pd.Series) -> str:
        tags = []

        h0_present = np.isfinite(row.get('overall_h0_lnl', np.nan))
        h1_present = np.isfinite(row.get('overall_h1_lnl', np.nan))
        diff = row.get('overall_h1_lnl', np.nan) - row.get('overall_h0_lnl', np.nan)

        if _runtime_issue_present(row, status_run_cols):
            tags.append('Runtime error observed in at least one run')

        if not h0_present and not h1_present:
            tags.append('Missing H0 and H1 likelihoods')
        elif h0_present and not h1_present:
            tags.append('H1 likelihood missing')
        elif h1_present and not h0_present:
            tags.append('H0 likelihood missing')
        else:
            if diff >= -epsilon:
                tags.append('Complete data with H1 at least as good as H0')
            else:
                tags.append('Complete data but H1 worse than H0')

        if pd.isna(row.get('overall_p_value')):
            if h0_present and h1_present and diff < -epsilon:
                tags.append('P-value not computed because H1 is below H0')
            elif not (h0_present and h1_present):
                tags.append('P-value not computed because model results are incomplete')

        return '; '.join(tags) if tags else 'Status unavailable'

    return df.apply(build_status, axis=1)


def add_status_column(df: pd.DataFrame, epsilon: float = STATUS_EPSILON) -> pd.DataFrame:
    df = df.copy()
    if 'status' in df.columns:
        df = df.drop(columns=['status'])
    df['status'] = compute_status_series(df, epsilon=epsilon)
    return df


def _insert_status_after_gene(df: pd.DataFrame) -> pd.DataFrame:
    if 'status' not in df.columns:
        return df

    status_col = df.pop('status')
    if 'region' in df.columns and 'gene' in df.columns:
        insert_at = min(len(df.columns), df.columns.get_loc('gene') + 1)
        df.insert(insert_at, 'status', status_col)
    else:
        df['status'] = status_col
    return df


def retrofit_grand_results(
    source_url: str = DEFAULT_SOURCE_URL, output_filename: str = 'GRAND_PAML_RESULTS.tsv'
) -> None:
    print(f"Downloading GRAND PAML results from {source_url}...")
    df = pd.read_csv(source_url, sep='\t')

    print("Computing descriptive status tags...")
    df = add_status_column(df)
    df = _insert_status_after_gene(df)

    print(f"Writing updated results with status column to {output_filename}...")
    df.to_csv(output_filename, sep='\t', index=False, float_format='%.6g')
    print("Done.")


if __name__ == '__main__':
    retrofit_grand_results()
