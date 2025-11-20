"""
====================================================================================================
Permutation-Based Median dN/dS Analysis
====================================================================================================

This script implements a transcript-by-transcript label-permutation test to compare Group-0 vs.
Group-1 dN/dS (omega) values using median-based statistics.
"""

import os
import re
import sys
import math
import random
import pickle
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from urllib.parse import urlencode
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------
#                          CONFIGURATION SECTION
# -----------------------------------------------------------------------------------

INPUT_CSV = "all_pairwise_results.csv"
OUTPUT_DIR = "results"
CACHE_DIR = "cache"
MIN_SEQUENCES_PER_GROUP = 10
NUM_PERMUTATIONS = 10000
FDR_THRESHOLD = 0.05
USE_GENE_ANNOTATION = True


# -----------------------------------------------------------------------------------
#                           DATA READING & PREPROCESS
# -----------------------------------------------------------------------------------

def read_pairwise_data(csv_path: str) -> pd.DataFrame:
    """
    Read the CSV with columns:
      Seq1, Seq2, Group1, Group2, dN, dS, omega, CDS
    and parse transcript_id, chromosome coords (if present).
    We drop rows with missing omega.
    """
    print(f"Reading CSV from {csv_path}")
    df = pd.read_csv(csv_path)

    # Convert omega to float
    df['omega'] = pd.to_numeric(df['omega'], errors='coerce')
    df.dropna(subset=['omega'], inplace=True)

    # Extract transcript ID
    # (We do not put the regex in a comment to comply with your request.)
    pattern_tx = r'(ENST\d+\.\d+)'
    df['transcript_id'] = df['CDS'].str.extract(pattern_tx, expand=False)

    # Chrom coords
    pattern_coords = r'_chr(\w+)_start(\d+)_end(\d+)'
    coords = df['CDS'].str.extract(pattern_coords)
    df['chrom'] = coords[0].apply(lambda x: f"chr{x}" if pd.notna(x) else None)
    df['start'] = pd.to_numeric(coords[1], errors='coerce')
    df['end']   = pd.to_numeric(coords[2], errors='coerce')

    return df


# -----------------------------------------------------------------------------------
#                          GENE ANNOTATION (Optional)
# -----------------------------------------------------------------------------------

def get_gene_info(symbol: str) -> str:
    """
    Query MyGene.info to get an official name for a gene symbol, if possible.
    Returns 'Unknown' on error or if symbol is blank.
    """
    if not symbol or pd.isna(symbol):
        return "Unknown"
    try:
        url = f"http://mygene.info/v3/query?q=symbol:{symbol}&species=human&fields=name"
        r = requests.get(url, timeout=6)
        if r.ok:
            data = r.json()
            hits = data.get('hits', [])
            if hits:
                return hits[0].get('name', 'Unknown')
    except:
        pass
    return "Unknown"


def get_ucsc_annotation(chrom: str, start: float, end: float) -> (str, str):
    """
    Query UCSC knownGene track for chrom:start-end to find an overlapping gene. Then get a
    human-readable name from MyGene. Return (symbol, name). If none, (None, None).
    """
    if not chrom or pd.isna(start) or pd.isna(end):
        return (None, None)
    base_url = "https://api.genome.ucsc.edu/getData/track"
    params = {'genome':'hg38','track':'knownGene','chrom':chrom,'start':int(start),'end':int(end)}
    try:
        r = requests.get(f"{base_url}?{urlencode(params)}", timeout=6)
        if not r.ok:
            return (None,None)
        data = r.json()
        track = data.get('knownGene', [])
        if not isinstance(track, list):
            return (None,None)
        best, best_ov = None, 0
        for g in track:
            gstart = g.get('chromStart', 0)
            gend   = g.get('chromEnd', 0)
            overlap = max(0, min(gend, end) - max(gstart, start))
            if overlap>best_ov:
                best_ov = overlap
                best = g
        if best:
            symbol = best.get('geneName','Unknown')
            if symbol in ['none', None] or symbol.startswith('ENSG'):
                symbol = "Unknown"
            name = get_gene_info(symbol)
            return (symbol, name)
    except:
        pass
    return (None,None)


# -----------------------------------------------------------------------------------
#              PER-TRANSCRIPT: BUILD LOCAL SEQ->GROUP ASSIGNMENT
# -----------------------------------------------------------------------------------

def build_local_group_map_for_transcript(df_sub: pd.DataFrame) -> dict:
    """
    For the lines in this transcript's df_sub, assign each sequence to group=0 or group=1
    exactly as indicated by the row's Group1, Group2 columns. If there's a conflict
    (the same sequence is assigned to different groups in different rows),
    or if a sequence never appears in any row with a valid group in [0,1],
    we raise an error.

    Returns: dict { sequence_id: 0 or 1 }
    """
    seq_map = {}
    all_seqs = pd.unique(df_sub[['Seq1','Seq2']].values.ravel())

    # We'll track which sequences are assigned and to which group. If conflict => raise.
    for row in df_sub.itertuples():
        s1, g1 = row.Seq1, row.Group1
        s2, g2 = row.Seq2, row.Group2

        # Both must be in [0,1]
        if g1 not in [0,1]:
            raise ValueError(f"Transcript {row.transcript_id}: Seq1={s1} invalid group={g1}")
        if g2 not in [0,1]:
            raise ValueError(f"Transcript {row.transcript_id}: Seq2={s2} invalid group={g2}")

        # Seq1
        if s1 not in seq_map:
            seq_map[s1] = g1
        else:
            if seq_map[s1] != g1:
                raise ValueError(f"Conflict in transcript {row.transcript_id}, sequence {s1} assigned both {seq_map[s1]} and {g1}")

        # Seq2
        if s2 not in seq_map:
            seq_map[s2] = g2
        else:
            if seq_map[s2] != g2:
                raise ValueError(f"Conflict in transcript {row.transcript_id}, sequence {s2} assigned both {seq_map[s2]} and {g2}")

    # Now verify that every sequence in this transcript is indeed assigned:
    for s in all_seqs:
        if s not in seq_map:
            raise ValueError(f"Transcript {df_sub['transcript_id'].iloc[0]}: sequence {s} never assigned 0 or 1 in these rows")

    return seq_map


# -----------------------------------------------------------------------------------
#          OBSERVED STATISTIC & PERMUTATION TEST (median(0-0) - median(1-1))
# -----------------------------------------------------------------------------------

def compute_median_diff(df_sub: pd.DataFrame, seq_map: dict) -> (float, list, list):
    """
    Gather all 0-0 distances vs 1-1 distances, compute T_obs = median(0-0) - median(1-1).
    Return (T_obs, list_of_0_0_values, list_of_1_1_values).
    If either set is empty => (nan, [...], [...]).

    We assume every row has a valid omega, and seq_map has no conflicts.
    """
    pairwise = {}
    for row in df_sub.itertuples():
        pairwise[(row.Seq1, row.Seq2)] = row.omega
        pairwise[(row.Seq2, row.Seq1)] = row.omega

    all_seqs = pd.unique(df_sub[['Seq1','Seq2']].values.ravel())
    g0_vals, g1_vals = [], []

    for i in range(len(all_seqs)):
        for j in range(i+1, len(all_seqs)):
            si = all_seqs[i]
            sj = all_seqs[j]
            gi = seq_map[si]
            gj = seq_map[sj]
            if gi==0 and gj==0:
                v = pairwise.get((si, sj))
                if v is not None:
                    g0_vals.append(v)
            elif gi==1 and gj==1:
                v = pairwise.get((si, sj))
                if v is not None:
                    g1_vals.append(v)

    if len(g0_vals)==0 or len(g1_vals)==0:
        return (math.nan, g0_vals, g1_vals)
    return (np.median(g0_vals) - np.median(g1_vals), g0_vals, g1_vals)


def permutation_test(df_sub: pd.DataFrame,
                     seq_map: dict,
                     n0: int,
                     n1: int,
                     B: int) -> dict:
    """
    Label-permutation test. Observed T = median(0-0) - median(1-1).
    Then do B permutations, picking n0 seqs for group0, n1 for group1,
    compute T_perm, etc.

    Return a dict with effect_size, p_value, group0_count, group1_count, failure_reason.
    """
    T_obs, g0_vals, g1_vals = compute_median_diff(df_sub, seq_map)
    if math.isnan(T_obs):
        return dict(effect_size=math.nan, p_value=math.nan,
                    group0_count=len(g0_vals), group1_count=len(g1_vals),
                    failure_reason="No 0-0 or 1-1 pairs")

    # Pairwise for quick lookup
    pairwise = {}
    for row in df_sub.itertuples():
        pairwise[(row.Seq1, row.Seq2)] = row.omega
        pairwise[(row.Seq2, row.Seq1)] = row.omega

    all_seqs = pd.unique(df_sub[['Seq1','Seq2']].values.ravel())
    seq_list = list(all_seqs)
    count_extreme, total_valid = 0, 0

    for _ in range(B):
        random.shuffle(seq_list)
        perm0 = set(seq_list[:n0])
        perm1 = set(seq_list[n0:])
        tmp0, tmp1 = [], []

        for i in range(len(seq_list)):
            for j in range(i+1, len(seq_list)):
                si, sj = seq_list[i], seq_list[j]
                if si in perm0 and sj in perm0:
                    v = pairwise.get((si, sj))
                    if v is not None:
                        tmp0.append(v)
                elif si in perm1 and sj in perm1:
                    v = pairwise.get((si, sj))
                    if v is not None:
                        tmp1.append(v)

        if len(tmp0)==0 or len(tmp1)==0:
            continue
        Tb = np.median(tmp0) - np.median(tmp1)
        total_valid += 1
        if abs(Tb) >= abs(T_obs):
            count_extreme += 1

    if total_valid==0:
        return dict(effect_size=T_obs, p_value=math.nan,
                    group0_count=len(g0_vals), group1_count=len(g1_vals),
                    failure_reason="No valid permutations")
    p_val = max(count_extreme / total_valid, 1/(total_valid + 1))
    return dict(effect_size=T_obs, p_value=p_val,
                group0_count=len(g0_vals), group1_count=len(g1_vals),
                failure_reason=None)


# -----------------------------------------------------------------------------------
#                 BUILD GROUP0/GROUP1 MATRICES FOR PLOTTING
# -----------------------------------------------------------------------------------

def build_matrices_for_plotting(df_sub: pd.DataFrame, seqs0: list, seqs1: list):
    """
    For optional visualization: build NxN matrices of distances among group0 or group1.
    Return (matrix_0, matrix_1). If group0 is empty => matrix_0=None, etc.
    """
    pairwise = {}
    for row in df_sub.itertuples():
        pairwise[(row.Seq1, row.Seq2)] = row.omega
        pairwise[(row.Seq2, row.Seq1)] = row.omega

    mat0, mat1 = None, None
    if len(seqs0)>0:
        s0sorted = sorted(seqs0)
        mat0 = np.full((len(s0sorted),len(s0sorted)), np.nan)
        for i in range(len(s0sorted)):
            for j in range(i+1, len(s0sorted)):
                v = pairwise.get((s0sorted[i], s0sorted[j]))
                if v is not None:
                    mat0[i,j] = v
                    mat0[j,i] = v

    if len(seqs1)>0:
        s1sorted = sorted(seqs1)
        mat1 = np.full((len(s1sorted),len(s1sorted)), np.nan)
        for i in range(len(s1sorted)):
            for j in range(i+1, len(s1sorted)):
                v = pairwise.get((s1sorted[i], s1sorted[j]))
                if v is not None:
                    mat1[i,j] = v
                    mat1[j,i] = v
    return mat0, mat1


# -----------------------------------------------------------------------------------
#                    PARALLEL WORKER: analyze_transcript
# -----------------------------------------------------------------------------------

def analyze_transcript(args):
    """
    Worker for parallel usage. Steps:
      1) Build local group map (crash if conflict/unassigned)
      2) Count how many sequences in group0 vs group1 => if < MIN_SEQUENCES_PER_GROUP => skip
      3) If annotation is on, get coords from first row => fetch gene symbol
      4) Permutation test => effect_size, p_value
      5) Build group0, group1 matrices => store
      6) Return final dict for that transcript
    """
    tx_id, df_t, B = args

    # 1) Local assignment. If conflict => raises ValueError => program stops
    seq_map = build_local_group_map_for_transcript(df_t)

    allseqs = pd.unique(df_t[['Seq1','Seq2']].values.ravel())
    group0 = [s for s in allseqs if seq_map[s]==0]
    group1 = [s for s in allseqs if seq_map[s]==1]
    n0, n1 = len(group0), len(group1)
    if n0<MIN_SEQUENCES_PER_GROUP or n1<MIN_SEQUENCES_PER_GROUP:
        return dict(
            transcript_id=tx_id,
            coordinates=None,
            gene_symbol=None,
            gene_name=None,
            n0=n0,
            n1=n1,
            num_comp_group_0=0,
            num_comp_group_1=0,
            effect_size=math.nan,
            p_value=math.nan,
            corrected_p_value=math.nan,
            failure_reason=f"Insufficient sequences in group0 or group1 (n0={n0},n1={n1})",
            matrix_0=None,
            matrix_1=None,
            pairwise_comparisons=None
        )

    # 2) Possibly gene annotation
    coords_str, gsym, gname = None,None,None
    if USE_GENE_ANNOTATION:
        row0 = df_t.iloc[0]
        c = row0.chrom
        st= row0.start
        en= row0.end
        if isinstance(c,str) and not pd.isna(st) and not pd.isna(en):
            coords_str = f"{c}:{st}-{en}"
            symbol, name = get_ucsc_annotation(c, st, en)
            gsym, gname = symbol, name
    # 3) Permutation test
    perm_res = permutation_test(df_t, seq_map, n0, n1, B)

    # 4) Build plotting matrices
    m0, m1 = build_matrices_for_plotting(df_t, group0, group1)

    # Return
    return dict(
        transcript_id=tx_id,
        coordinates=coords_str,
        gene_symbol=gsym,
        gene_name=gname,
        n0=n0,
        n1=n1,
        num_comp_group_0=perm_res['group0_count'],
        num_comp_group_1=perm_res['group1_count'],
        effect_size=perm_res['effect_size'],
        p_value=perm_res['p_value'],
        corrected_p_value=math.nan,
        failure_reason=perm_res['failure_reason'],
        matrix_0=m0,
        matrix_1=m1,
        pairwise_comparisons=None
    )


# -----------------------------------------------------------------------------------
#                                MAIN
# -----------------------------------------------------------------------------------

def main():
    start_time = datetime.now()
    print(f"\n=== Permutation Analysis STARTED: {start_time} ===\n")
    print(f"Configuration:\n  INPUT_CSV={INPUT_CSV}\n  OUTPUT_DIR={OUTPUT_DIR}\n"
          f"  CACHE_DIR={CACHE_DIR}\n  MIN_SEQUENCES_PER_GROUP={MIN_SEQUENCES_PER_GROUP}\n"
          f"  NUM_PERMUTATIONS={NUM_PERMUTATIONS}\n  FDR_THRESHOLD={FDR_THRESHOLD}\n"
          f"  USE_GENE_ANNOTATION={USE_GENE_ANNOTATION}\n")

    # 1) Read data
    df = read_pairwise_data(INPUT_CSV)
    n_tx = df['transcript_id'].nunique()
    print(f"Found {n_tx} transcripts.\n")

    # Group by transcript_id
    grouped = df.groupby('transcript_id')
    
    tasks = []
    for tx_id, df_tx in grouped:
        tasks.append((tx_id, df_tx, NUM_PERMUTATIONS))

    print(f"Preparing to analyze {len(tasks)} transcripts in parallel.\n")
    results = []
    # 3) Parallel
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as exe:
        # If there's a conflict in any transcript, it will raise ValueError,
        # crashing the entire program as requested.
        for r in exe.map(analyze_transcript, tasks):
            results.append(r)

    # Build DataFrame
    res_df = pd.DataFrame(results)

    # 4) BH-FDR
    val_mask = res_df['p_value'].notna() & (res_df['p_value']>0)
    valid_df = res_df[val_mask].copy()
    valid_df = valid_df.sort_values('p_value')
    if len(valid_df)>0:
        m = len(valid_df)
        valid_df['rank'] = np.arange(1,m+1)
        valid_df['bh'] = valid_df['p_value']*m/valid_df['rank']
        valid_df['bh'] = valid_df['bh'].iloc[::-1].cummin().iloc[::-1]
        valid_df['bh'] = valid_df['bh'].clip(upper=1.0)
        bh_map = dict(zip(valid_df['transcript_id'], valid_df['bh']))
        res_df['corrected_p_value'] = res_df['transcript_id'].map(bh_map)
    else:
        res_df['corrected_p_value'] = np.nan

    # 5) Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Final results CSV
    drop_cols = ['matrix_0','matrix_1','pairwise_comparisons']
    out_csv = os.path.join(OUTPUT_DIR, "final_results_permutation.csv")
    res_df.drop(columns=drop_cols, errors='ignore').to_csv(out_csv, index=False)
    print(f"Saved final results to: {out_csv}")

    # Significant
    sig_mask = res_df['corrected_p_value'] < FDR_THRESHOLD
    num_sig = sig_mask.sum()
    print(f"Significant transcripts (FDR<{FDR_THRESHOLD}): {num_sig}")
    if num_sig>0:
        sig_df = res_df[sig_mask].sort_values('p_value')
        sig_csv = os.path.join(OUTPUT_DIR,"significant_transcripts.csv")
        sig_df.drop(columns=drop_cols, errors='ignore').to_csv(sig_csv, index=False)
        print(f"Significant transcripts saved to: {sig_csv}")

    # Pickle big data
    big_dict = {}
    for _, row in res_df.iterrows():
        tid = row['transcript_id']
        big_dict[tid] = {
            'matrix_0': row['matrix_0'],
            'matrix_1': row['matrix_1'],
            'p_value': row['p_value'],
            'corrected_p_value': row['corrected_p_value'],
            'effect_size': row['effect_size'],
            'n0': row['n0'],
            'n1': row['n1']
        }
    pkl_path = os.path.join(CACHE_DIR, "all_cds_results_permutation.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(big_dict,f)
    print(f"Stored matrix data to: {pkl_path}")

    # 6) Print summary
    print("\n=== Summary by Transcript ===")
    print(f"{'Transcript':<25} {'n0':>4} {'n1':>4} {'Effect':>10} {'p-val':>12} {'FDR':>12} {'FailureReason'}")
    sorted_df = res_df.sort_values('p_value', na_position='last')
    for _, row in sorted_df.iterrows():
        tid   = str(row['transcript_id'])
        eff   = row['effect_size']
        if pd.isna(eff):
            eff_str = "NA"
        else:
            eff_str = f"{eff:.4f}"
        pv = row['p_value']
        if pd.isna(pv):
            pv_str = "NA"
        else:
            pv_str = f"{pv:.3e}"
        cp = row['corrected_p_value']
        if pd.isna(cp):
            cp_str = "NA"
        else:
            cp_str = f"{cp:.3e}"
        fail = row['failure_reason'] if pd.notna(row['failure_reason']) else ""
        print(f"{tid:<25} {row['n0']:>4} {row['n1']:>4} {eff_str:>10} {pv_str:>12} {cp_str:>12} {fail}")

    end_time = datetime.now()
    print(f"\n=== Done. Finished at {end_time}, total runtime = {end_time - start_time} ===")


if __name__=="__main__":
    main()
