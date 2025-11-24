import os
import re
import sys
import glob
import subprocess
import tempfile
import getpass
import logging
import traceback
from datetime import datetime
import shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import urllib.request
import tarfile
import stat
import json
import hashlib
import random
import shlex

import numpy as np
from scipy.stats import chi2
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd

# --- ETE3 Configuration ---
os.environ["QT_QPA_PLATFORM"] = "offscreen"
user = getpass.getuser()
runtime_dir = f"/tmp/runtime-{user}"
os.makedirs(runtime_dir, exist_ok=True, mode=0o700)
os.environ['XDG_RUNTIME_DIR'] = runtime_dir

from ete3 import Tree
try:
    from ete3.treeview import TreeStyle, NodeStyle, TextFace, CircleFace, RectFace
    TREEVIEW_IMPORT_ERROR = None
except Exception as exc:
    TREEVIEW_IMPORT_ERROR = exc
    TreeStyle = NodeStyle = TextFace = CircleFace = RectFace = None


# ==============================================================================
# 1. Constants & Configuration
# ==============================================================================

def _load_inversion_whitelist():
    """
    Parses data/inv_properties.tsv to build the ALLOWED_REGIONS whitelist.
    Filters by '0_single_1_recur_consensus' column.
    """
    whitelist = []
    # Look for file in likely locations
    candidates = [
        "data/inv_properties.tsv",
        os.path.join(os.path.dirname(__file__), "../data/inv_properties.tsv")
    ]

    tsv_path = None
    for c in candidates:
        if os.path.exists(c):
            tsv_path = c
            break

    if not tsv_path:
        # Fallback or empty if file not found (prevents import crash, but will log warning)
        print("WARNING: data/inv_properties.tsv not found. ALLOWED_REGIONS will be empty.", file=sys.stderr)
        return []

    try:
        with open(tsv_path, 'r') as f:
            # Skip Header
            header = f.readline()

            # Dynamically find the index of the consensus column for robustness
            try:
                consensus_idx = header.strip().split('\t').index('0_single_1_recur_consensus')
            except ValueError:
                print("ERROR: Could not find '0_single_1_recur_consensus' in inv_properties.tsv header.", file=sys.stderr)
                return []


            for line in f:
                if not line.strip(): continue
                parts = line.strip().split('\t')

                if len(parts) <= consensus_idx: continue

                # New criteria: only include if 0_single_1_recur_consensus is 0 or 1
                consensus_val = parts[consensus_idx].strip()
                if consensus_val not in ["0", "1"]:
                    continue

                if len(parts) < 3: continue

                # Parse Columns
                chrom_raw = parts[0].strip()
                try:
                    start = int(parts[1].strip())
                    end = int(parts[2].strip())
                except (ValueError, IndexError):
                    continue # Skip malformed lines

                # Normalize Chromosome (ensure chr prefix)
                # The file has 'chr1', but just in case
                if not chrom_raw.lower().startswith('chr'):
                    chrom = f"chr{chrom_raw}"
                else:
                    chrom = chrom_raw

                # Ensure standard tuple format
                whitelist.append((chrom, start, end))

    except Exception as e:
        print(f"Error parsing inversion whitelist: {e}", file=sys.stderr)
        return []

    return whitelist

# Dynamically populate the constant
ALLOWED_REGIONS = _load_inversion_whitelist()

POP_COLORS = {
    'AFR': '#F05031', 'EUR': '#3173F0', 'EAS': '#35A83A',
    'SAS': '#F031D3', 'AMR': '#B345F0', 'CHIMP': '#808080'
}

DIVERGENCE_THRESHOLD = 0.10
FDR_ALPHA = 0.05
FLOAT_REGEX = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'

CACHE_SCHEMA_VERSION = "paml_cache.v1"
CACHE_FANOUT = 2
CACHE_LOCK_TIMEOUT_S = int(os.environ.get("PAML_CACHE_LOCK_TIMEOUT_S", "600"))
CACHE_LOCK_POLL_MS = (50, 250)

FIGURE_DIR = "tree_figures"
ANNOTATED_FIGURE_DIR = "annotated_tree_figures"
REGION_TREE_DIR = "region_trees"

# --- Analysis Configuration ---
CHECKPOINT_FILE = "paml_results.checkpoint.tsv"
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "100"))
KEEP_PAML_OUT = bool(int(os.environ.get("KEEP_PAML_OUT", "0")))
PAML_OUT_DIR  = os.environ.get("PAML_OUT_DIR", "paml_runs")
PAML_CACHE_DIR = os.environ.get("PAML_CACHE_DIR", "paml_cache")

IQTREE_TIMEOUT = int(os.environ.get("IQTREE_TIMEOUT", "7200"))
PAML_TIMEOUT   = int(os.environ.get("PAML_TIMEOUT", "20880"))

RUN_BRANCH_MODEL_TEST = False
RUN_CLADE_MODEL_TEST = True
PROCEED_ON_TERMINAL_ONLY = False

# --- Tournament seeds for PAML restarts ---
SEED_BANK = {
    "s1_def": {"kappa": 2.0, "omega": 0.4},
    "s2_pur": {"kappa": 1.0, "omega": 0.05},
    "s3_pos": {"kappa": 3.0, "omega": 2.5},
    "s4_mix": {"kappa": 5.0, "omega": 0.8},
}


# ==============================================================================
# 2. System & Tool Setup
# ==============================================================================

def setup_external_tools(base_dir):
    """
    Checks for PAML and IQ-TREE dependencies in the given base directory.
    Downloads and extracts them if missing.
    Returns (iqtree_bin, paml_bin).
    """
    paml_dir = os.path.join(base_dir, 'paml')
    paml_bin = os.path.join(paml_dir, 'bin', 'codeml')

    iqtree_dir = os.path.join(base_dir, 'iqtree-3.0.1-Linux')
    iqtree_bin = os.path.join(iqtree_dir, 'bin', 'iqtree3')

    # Setup PAML
    if not os.path.exists(paml_bin):
        logging.info("PAML not found. Downloading...")
        url = "https://github.com/abacus-gene/paml/releases/download/v4.10.9/paml-4.10.9-linux-x86_64.tar.gz"
        tar_path = os.path.join(base_dir, "paml.tar.gz")
        try:
            urllib.request.urlretrieve(url, tar_path)
            logging.info("Extracting PAML...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=base_dir)

            # Rename extracted folder to 'paml'
            root_folder = None
            with tarfile.open(tar_path, "r:gz") as tar:
                 for member in tar.getmembers():
                     if member.isdir() and '/' not in member.name.strip('/'):
                         root_folder = member.name
                         break

            if root_folder:
                extracted_folder = os.path.join(base_dir, root_folder)
                if os.path.exists(extracted_folder):
                    if os.path.exists(paml_dir):
                        shutil.rmtree(paml_dir)
                    os.rename(extracted_folder, paml_dir)
                else:
                    logging.warning(f"Expected extracted folder '{extracted_folder}' not found.")
            else:
                 logging.warning("Could not determine root folder from PAML tarball.")

            if os.path.exists(tar_path):
                os.remove(tar_path)

            if os.path.exists(paml_bin):
                st = os.stat(paml_bin)
                os.chmod(paml_bin, st.st_mode | stat.S_IEXEC)
                logging.info(f"PAML setup complete at {paml_dir}")
            else:
                logging.warning(f"PAML extracted but binary not found at {paml_bin}")

        except Exception as e:
            logging.error(f"Error setting up PAML: {e}")
    else:
        logging.info(f"PAML found at {paml_dir}")
        try:
            st = os.stat(paml_bin)
            os.chmod(paml_bin, st.st_mode | stat.S_IEXEC)
        except Exception as e:
            logging.warning(f"Could not set execute permissions on PAML: {e}")

    # Setup IQ-TREE
    if not os.path.exists(iqtree_bin):
        logging.info("IQ-TREE not found. Downloading...")
        url = "https://github.com/iqtree/iqtree3/releases/download/v3.0.1/iqtree-3.0.1-Linux.tar.gz"
        tar_path = os.path.join(base_dir, "iqtree.tar.gz")
        try:
            urllib.request.urlretrieve(url, tar_path)
            logging.info("Extracting IQ-TREE...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=base_dir)

            if os.path.exists(tar_path):
                os.remove(tar_path)

            if os.path.exists(iqtree_bin):
                st = os.stat(iqtree_bin)
                os.chmod(iqtree_bin, st.st_mode | stat.S_IEXEC)
                logging.info(f"IQ-TREE setup complete at {iqtree_dir}")
            else:
                 logging.warning(f"IQ-TREE extracted but binary not found at {iqtree_bin}")

        except Exception as e:
             logging.error(f"Error setting up IQ-TREE: {e}")
    else:
        logging.info(f"IQ-TREE found at {iqtree_dir}")
        try:
            st = os.stat(iqtree_bin)
            os.chmod(iqtree_bin, st.st_mode | stat.S_IEXEC)
        except Exception as e:
            logging.warning(f"Could not set execute permissions on IQ-TREE: {e}")

    return iqtree_bin, paml_bin

def run_command(command_list, work_dir, timeout=None, env=None, input_data=None):
    try:
        subprocess.run(
            command_list, cwd=work_dir, check=True,
            capture_output=True, text=True, shell=False,
            timeout=timeout, env=env, input=input_data
        )
    except subprocess.TimeoutExpired as e:
        cmd_str = ' '.join(command_list)
        raise RuntimeError(
            f"\n--- COMMAND TIMEOUT ---\nCOMMAND: '{cmd_str}'\nTIMEOUT: {timeout}s\nDIR: {work_dir}\n"
            f"--- PARTIAL STDOUT ---\n{e.stdout}\n--- PARTIAL STDERR ---\n{e.stderr}\n--- END ---"
        ) from e
    except subprocess.CalledProcessError as e:
        cmd_str = ' '.join(e.cmd)
        error_message = (
            f"\n--- COMMAND FAILED ---\n"
            f"COMMAND: '{cmd_str}'\nEXIT CODE: {e.returncode}\nWORKING DIR: {work_dir}\n"
            f"--- STDOUT ---\n{e.stdout}\n--- STDERR ---\n{e.stderr}\n--- END ---"
        )
        raise RuntimeError(error_message) from e

# ==============================================================================
# 3. Input Parsing (Metadata & Files)
# ==============================================================================

def load_gene_metadata(tsv_path='phy_metadata.tsv'):
    """Load gene coordinate metadata from a TSV file robustly."""
    if not os.path.exists(tsv_path):
        # Fallback check for data/ directory
        alt_path = os.path.join('data', tsv_path)
        if os.path.exists(alt_path):
            tsv_path = alt_path
        else:
            raise FileNotFoundError(
                f"Metadata file '{tsv_path}' not found (checked local and data/); cannot map genes to regions.")

    df = pd.read_csv(tsv_path, sep='\t', dtype=str)

    aliases = {
        'gene': ['gene', 'gene_name', 'GENE'],
        'enst': ['enst', 't_id', 'transcript', 'transcript_id'],
        'chr': ['chr', 'chrom', 'chromosome'],
        'start': ['start', 'tx_start', 'cds_start', 'overall_cds_start_1based'],
        'end': ['end', 'tx_end', 'cds_end', 'overall_cds_end_1based'],
    }
    col_map = {}
    for canon, names in aliases.items():
        for name in names:
            if name in df.columns:
                col_map[canon] = name
                break
    missing = [c for c in aliases if c not in col_map]
    if missing:
        raise KeyError(
            f"Metadata file missing columns {missing}. Available: {list(df.columns)}")

    def _norm_chr(x):
        if x is None or pd.isna(x):
            return None
        s = str(x).strip()
        s = s.replace('Chr', 'chr').replace('CHR', 'chr')
        if s in {'M', 'MT', 'Mt', 'chrMT', 'chrMt', 'MT_chr'}:
            return 'chrM'
        if not s.startswith('chr'):
            s = 'chr' + s.lstrip('chr')
        return s

    df['_gene'] = df[col_map['gene']].astype(str)
    df['_enst'] = df[col_map['enst']].astype(str)
    df['_chr'] = df[col_map['chr']].apply(_norm_chr)
    df['_start'] = pd.to_numeric(df[col_map['start']], errors='coerce')
    df['_end'] = pd.to_numeric(df[col_map['end']], errors='coerce')

    df = df.dropna(subset=['_gene', '_enst', '_chr', '_start', '_end'])

    flipped_mask = df['_start'] > df['_end']
    if flipped_mask.any():
        original_starts = df.loc[flipped_mask, '_start'].copy()
        df.loc[flipped_mask, '_start'] = df.loc[flipped_mask, '_end']
        df.loc[flipped_mask, '_end'] = original_starts

    df['_width'] = (df['_end'] - df['_start']).abs()
    df = df.sort_values(['_gene', '_enst', '_width'], ascending=[True, True, False])
    df = df.drop_duplicates(subset=['_gene', '_enst'], keep='first')

    df['_start'] = df['_start'].round().astype(int)
    df['_end'] = df['_end'].round().astype(int)

    meta = {}
    for _, row in df.iterrows():
        meta[(row['_gene'], row['_enst'])] = {
            'chrom': row['_chr'],
            'start': int(row['_start']),
            'end': int(row['_end']),
        }
    return meta

def parse_region_filename(path):
    """Extract chromosome and coordinates from a region filename."""
    name = os.path.basename(path)
    m = re.match(r"^combined_inversion_(?:chr)?([0-9]+|X|Y|M|MT)_start(\d+)_end(\d+)\.phy$", name, re.I)
    if not m:
        m = re.match(r"^combined_inversion_(?:chr)?([0-9]+|X|Y|M|MT)_(\d+)_(\d+)\.phy$", name, re.I)
    if not m:
        raise ValueError(f"Unrecognized region filename format: {name}")

    chrom_token, start_str, end_str = m.groups()
    chrom_token = chrom_token.upper()
    chrom = "chrM" if chrom_token in ("M", "MT") else f"chr{chrom_token}"
    start = int(start_str)
    end = int(end_str)
    if start > end:
        start, end = end, start

    return {
        'path': path,
        'chrom': chrom,
        'start': start,
        'end': end,
        'label': f"{chrom}_{start}_{end}"
    }

def parse_gene_filename(path, metadata):
    """Extract gene and transcript from a gene filename and augment with metadata."""
    name = os.path.basename(path)
    m = re.match(r"combined_([\w\.\-]+)_(ENST[^_]+)\.phy", name)
    if not m:
        m = re.match(r"combined_([\w\.\-]+)_(ENST[^_]+)_(chr[^_]+)_start(\d+)_end(\d+)\.phy", name)
    if not m:
        m = re.match(r"combined_([\w\.\-]+)_(ENST[^_]+)_(chr[^_]+)_(\d+)_(\d+)\.phy", name)
    if not m:
        raise ValueError(f"Unrecognized gene filename format: {name}")

    gene, enst = m.group(1), m.group(2)
    key = (gene, enst)
    if len(m.groups()) > 2:
        chrom = m.group(3)
        start = int(m.group(4))
        end = int(m.group(5))
    elif key in metadata:
        info = metadata[key]
        chrom, start, end = info['chrom'], info['start'], info['end']
    else:
        raise ValueError(f"Coordinates for {gene} {enst} not found in metadata or filename")

    return {
        'path': path,
        'gene': gene,
        'enst': enst,
        'chrom': chrom,
        'start': start,
        'end': end,
        'label': f"{gene}_{enst}"
    }

def build_region_gene_map(region_infos, gene_infos):
    """Map each region to the list of genes overlapping it."""
    region_map = {r['label']: [] for r in region_infos}
    for g in gene_infos:
        for r in region_infos:
            if g['chrom'] == r['chrom'] and not (g['end'] < r['start'] or g['start'] > r['end']):
                region_map[r['label']].append(g)
    return region_map

def read_taxa_from_phy(phy_path):
    """Return a list of taxa names from a PHYLIP alignment."""
    taxa = []
    with open(phy_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if parts:
                taxa.append(parts[0])
    return taxa

def _summarize_taxa_diagnostics(taxa):
    """Helper to generate a diagnostic string about taxa composition."""
    n = len(taxa)
    direct = [t for t in taxa if t.startswith('0')]
    inverted = [t for t in taxa if t.startswith('1')]
    chimp = next((t for t in taxa if 'pantro' in t.lower() or 'pan_troglodytes' in t.lower()), None)

    # Attempt to summarize populations if they follow the convention 0_POP_ID
    # e.g. 0_AFR_HG00000
    pop_counts = {}
    for t in taxa:
        m = re.search(r'[01]_([A-Z]{3})_', t)
        if m:
            pop = m.group(1)
            pop_counts[pop] = pop_counts.get(pop, 0) + 1

    pop_str = ", ".join([f"{k}:{v}" for k,v in pop_counts.items()]) if pop_counts else "N/A"

    return (f"Total={n}. "
            f"Direct={len(direct)} (needs >=1), "
            f"Inverted={len(inverted)} (needs >=1), "
            f"Chimp={chimp if chimp else 'MISSING'}. "
            f"Populations: [{pop_str}]. "
            f"Taxa List: {taxa[:10]}{'...' if n > 10 else ''}")

# ==============================================================================
# 4. Quality Control & Tree Operations
# ==============================================================================

def perform_qc(phy_file_path):
    """
    Performs quality control checks on a given phylip file.
    """
    with open(phy_file_path, 'r') as f:
        lines = f.readlines()

    if not lines or len(lines[0].strip().split()) < 2:
        return False, "File is empty or header is missing/malformed."

    header = lines[0].strip().split()
    seq_length = int(header[1])

    if seq_length % 3 != 0:
        return False, f"Sequence length {seq_length} not divisible by 3."

    sequences = {parts[0]: parts[1] for parts in (line.strip().split(maxsplit=1) for line in lines[1:]) if parts}

    human_seqs = [seq for name, seq in sequences.items() if name.startswith(('0', '1'))]
    chimp_name = next((name for name in sequences if 'pantro' in name.lower() or 'pan_troglodytes' in name.lower()), None)

    if not human_seqs or not chimp_name:
        missing = []
        if not human_seqs: missing.append("human sequences (starting with 0 or 1)")
        if not chimp_name: missing.append("chimp sequence (pantro/pan_troglodytes)")
        msg = f"Missing required sequences: {', '.join(missing)}."
        if not human_seqs and not chimp_name:
            msg += f" Found taxa: {list(sequences.keys())[:10]}"
        return False, msg

    chimp_seq = sequences[chimp_name]

    divergences = []
    for human_seq in human_seqs:
        diffs, comparable_sites = 0, 0
        for h_base, c_base in zip(human_seq, chimp_seq):
            if h_base != '-' and c_base != '-':
                comparable_sites += 1
                if h_base != c_base:
                    diffs += 1
        divergence = (diffs / comparable_sites) if comparable_sites > 0 else 0
        divergences.append(divergence)

    if not divergences:
        return False, "No comparable sites found to calculate divergence."

    median_divergence = np.median(divergences)
    if median_divergence > DIVERGENCE_THRESHOLD:
        return False, f"Median divergence {median_divergence:.2%} > {DIVERGENCE_THRESHOLD:.0%}."

    return True, "QC Passed"

def prune_region_tree(region_tree_path, taxa_to_keep, out_path):
    """Prune the region tree to the intersection of taxa."""
    tree = Tree(region_tree_path, format=1)
    leaf_names = set(tree.get_leaf_names())
    keep = [taxon for taxon in taxa_to_keep if taxon in leaf_names]
    tree.prune(keep, preserve_branch_length=True)
    tree.write(outfile=out_path, format=1)
    return out_path

def count_variable_codon_sites(phy_path, taxa_subset=None, max_sites_check=50000):
    with open(phy_path) as f:
        header = f.readline().strip().split()
        nseq, seqlen = int(header[0]), int(header[1])
        seqs = []
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            name, seq = parts[0], parts[1]
            if taxa_subset is None or name in taxa_subset:
                seqs.append(seq)
            if len(seqs) >= (len(taxa_subset) if taxa_subset else nseq): break
    if not seqs: return 0
    seqlen = min(seqlen, len(seqs[0]))
    var_codons = 0
    for i in range(0, min(seqlen, max_sites_check), 3):
        col = {s[i:i+3] for s in seqs if len(s) >= i+3}
        col = {c for c in col if '-' not in c and 'N' not in c and 'n' not in c}
        if len(col) > 1:
            var_codons += 1
    return var_codons

def _validate_internal_branch_labels(paml_tree_str: str, tree_obj: Tree, expected_marks: list):
    expected_counts = {mark: 0 for mark in expected_marks}
    for node in tree_obj.traverse():
        if not node.is_leaf() and hasattr(node, "paml_mark"):
            if node.paml_mark in expected_counts:
                expected_counts[node.paml_mark] += 1

    actual_counts = {mark: 0 for mark in expected_marks}
    for mark in expected_marks:
        pattern = re.compile(r"\)\s*(?::\s*" + FLOAT_REGEX + r")?\s*" + re.escape(mark))
        actual_counts[mark] = len(pattern.findall(paml_tree_str))

    for mark in expected_marks:
        assert actual_counts[mark] == expected_counts[mark], \
            f"Internal branch label validation failed for mark '{mark}'. Expected {expected_counts[mark]}, found {actual_counts[mark]}. Tree string: {paml_tree_str}"

def create_paml_tree_files(tree_path, work_dir, gene_name):
    logging.info(f"[{gene_name}] Labeling internal branches conservatively...")
    t = Tree(tree_path, format=1)

    direct_leaves = 0
    inverted_leaves = 0
    for node in t.traverse("postorder"):
        if node.is_leaf():
            if node.name.startswith('0'):
                node.add_feature("group_status", "direct")
                direct_leaves += 1
            elif node.name.startswith('1'):
                node.add_feature("group_status", "inverted")
                inverted_leaves += 1
            else:
                node.add_feature("group_status", "outgroup")
        else:
            child_statuses = {child.group_status for child in node.children}
            if len(child_statuses) == 1:
                node.add_feature("group_status", child_statuses.pop())
            else:
                node.add_feature("group_status", "both")

    if direct_leaves < 3 or inverted_leaves < 3:
        msg = f"Insufficient samples in a group (direct: {direct_leaves}, inverted: {inverted_leaves})"
        logging.warning(f"[{gene_name}] Skipping: {msg}.")
        return None, None, False, t, msg

    internal_direct_count = 0
    internal_inverted_count = 0
    for node in t.traverse():
        if not node.is_leaf():
            status = getattr(node, "group_status", "both")
            if status == "direct":
                internal_direct_count += 1
            elif status == "inverted":
                internal_inverted_count += 1

    analysis_is_informative = (internal_direct_count > 0 and internal_inverted_count > 0)
    reason = None
    if not analysis_is_informative:
        logging.warning(f"[{gene_name}] Topology is uninformative for internal branch analysis.")
        reason = "No pure internal branches found for both direct and inverted groups."

    t_h1 = t.copy()
    for node in t_h1.traverse():
        status = getattr(node, "group_status", "both")
        if status == "direct":
            node.add_feature("paml_mark", "#1")
        elif status == "inverted":
            node.add_feature("paml_mark", "#2")

    h1_newick = t_h1.write(format=1, features=["paml_mark"])
    h1_paml_str = re.sub(r"\[&&NHX:paml_mark=(#\d+)\]", r" \1", h1_newick)
    if (" #1" not in h1_paml_str) and (" #2" not in h1_paml_str):
        msg = "H1 tree has no labeled branches"
        logging.warning(f"[{gene_name}] {msg}; treating as uninformative.")
        return None, None, False, t, msg

    _validate_internal_branch_labels(h1_paml_str, t_h1, ['#1', '#2'])
    h1_tree_path = os.path.join(work_dir, f"{gene_name}_H1.tree")
    with open(h1_tree_path, 'w') as f:
        f.write("1\n" + h1_paml_str + "\n")

    t_h0 = t.copy()
    for node in t_h0.traverse():
        status = getattr(node, "group_status", "both")
        if status in ["direct", "inverted"]:
            node.add_feature("paml_mark", "#1")

    h0_newick = t_h0.write(format=1, features=["paml_mark"])
    h0_paml_str = re.sub(r"\[&&NHX:paml_mark=(#1)\]", r" \1", h0_newick)
    _validate_internal_branch_labels(h0_paml_str, t_h0, ['#1'])
    h0_tree_path = os.path.join(work_dir, f"{gene_name}_H0.tree")
    with open(h0_tree_path, 'w') as f:
        f.write("1\n" + h0_paml_str + "\n")

    return h1_tree_path, h0_tree_path, analysis_is_informative, t, reason

def _tree_layout(node):
    if node.is_leaf():
        name = node.name
        pop_match = re.search(r'_(AFR|EUR|EAS|SAS|AMR)_', name)
        pop = pop_match.group(1) if pop_match else 'CHIMP'
        color = POP_COLORS.get(pop, "#C0C0C0")
        nstyle = NodeStyle(fgcolor=color, hz_line_width=1, vt_line_width=1)
        if name.startswith('1'): nstyle["shape"], nstyle["size"] = "sphere", 10
        elif 'pantro' in name.lower() or 'pan_troglodytes' in name.lower(): nstyle["shape"], nstyle["size"] = "square", 10
        else: nstyle["shape"], nstyle["size"] = "circle", 10
        node.set_style(nstyle)
    elif node.support > 50:
        nstyle = NodeStyle(shape="circle", size=5, fgcolor="#444444")
        node.set_style(nstyle)
        support_face = TextFace(f"{node.support:.0f}", fsize=7, fgcolor="grey")
        support_face.margin_left = 2
        node.add_face(support_face, column=0, position="branch-top")
    else:
        nstyle = NodeStyle(shape="circle", size=3, fgcolor="#CCCCCC")
        node.set_style(nstyle)

def generate_tree_figure(tree_file, label, output_dir=FIGURE_DIR, make_figures=True):
    if not make_figures or TREEVIEW_IMPORT_ERROR is not None:
        return
    t = Tree(tree_file, format=1)
    ts = TreeStyle()
    ts.layout_fn = _tree_layout
    ts.show_leaf_name = False
    ts.show_scale = False
    ts.branch_vertical_margin = 8
    ts.title.add_face(TextFace(f"Phylogeny of Region {label}", fsize=16, ftype="Arial"), column=0)

    ts.legend.add_face(TextFace("Haplotype Status", fsize=10, ftype="Arial", fstyle="Bold"), column=0)
    ts.legend.add_face(CircleFace(5, "black", style="circle"), column=0); ts.legend.add_face(TextFace(" Direct", fsize=9), column=1)
    ts.legend.add_face(CircleFace(5, "black", style="sphere"), column=0); ts.legend.add_face(TextFace(" Inverted", fsize=9), column=1)
    ts.legend.add_face(RectFace(10, 10, "black", "black"), column=0); ts.legend.add_face(TextFace(" Chimpanzee (Outgroup)", fsize=9), column=1)
    ts.legend.add_face(TextFace(" "), column=2)
    ts.legend.add_face(TextFace("Super-population", fsize=10, ftype="Arial", fstyle="Bold"), column=3)
    for pop, color in POP_COLORS.items():
        ts.legend.add_face(CircleFace(10, color), column=3); ts.legend.add_face(TextFace(f" {pop}", fsize=9), column=4)
    ts.legend_position = 1

    os.makedirs(output_dir, exist_ok=True)
    figure_path = os.path.join(output_dir, f"{label}.png")
    t.render(figure_path, w=200, units="mm", dpi=300, tree_style=ts)

def generate_omega_result_figure(gene_name, region_label, status_annotated_tree, paml_params, output_dir=ANNOTATED_FIGURE_DIR, make_figures=True):
    if not make_figures or TREEVIEW_IMPORT_ERROR is not None:
        return
    PURIFYING_COLOR = "#0072B2"
    POSITIVE_COLOR = "#D55E00"
    NEUTRAL_COLOR = "#000000"

    def _normalize_omega(value):
        try:
            if value is None: return None
            coerced = float(value)
        except (TypeError, ValueError): return None
        if not np.isfinite(coerced): return None
        return coerced

    def _omega_to_color(omega_value):
        omega = _normalize_omega(omega_value)
        if omega is None: return NEUTRAL_COLOR, None
        if omega > 1.0: return POSITIVE_COLOR, omega
        if omega < 1.0: return PURIFYING_COLOR, omega
        return NEUTRAL_COLOR, omega

    def _omega_color_layout(node):
        nstyle = NodeStyle()
        nstyle["hz_line_width"] = 2
        nstyle["vt_line_width"] = 2
        status = getattr(node, "group_status", "both")
        omega_source = {
            'direct': paml_params.get('omega_direct'),
            'inverted': paml_params.get('omega_inverted'),
        }.get(status, paml_params.get('omega_background'))

        color, _ = _omega_to_color(omega_source)
        nstyle["hz_line_color"] = color
        nstyle["vt_line_color"] = color

        if node.is_leaf():
            name = node.name
            pop_match = re.search(r'_(AFR|EUR|EAS|SAS|AMR)_', name)
            pop = pop_match.group(1) if pop_match else 'CHIMP'
            leaf_color = POP_COLORS.get(pop, "#C0C0C0")
            nstyle["fgcolor"] = leaf_color
            nstyle["size"] = 5
        else:
            nstyle["size"] = 0
        node.set_style(nstyle)

    ts = TreeStyle()
    ts.layout_fn = _omega_color_layout
    ts.show_leaf_name = False
    ts.show_scale = False
    ts.branch_vertical_margin = 8
    ts.title.add_face(TextFace(f"dN/dS for {gene_name} under {region_label}", fsize=16, ftype="Arial"), column=0)

    ts.legend.add_face(TextFace("Selection Regime (ω = dN/dS)", fsize=10, ftype="Arial", fstyle="Bold"), column=0)
    legend_map = {
        'Direct Group': paml_params.get('omega_direct'),
        'Inverted Group': paml_params.get('omega_inverted'),
        'Background': paml_params.get('omega_background'),
    }
    for name, omega_raw in legend_map.items():
        color, normalized = _omega_to_color(omega_raw)
        if normalized is not None:
            legend_text = f" {name} (ω = {normalized:.3f})"
            ts.legend.add_face(RectFace(10, 10, fgcolor=color, bgcolor=color), column=0)
            ts.legend.add_face(TextFace(legend_text, fsize=9), column=1)
    ts.legend_position = 4

    os.makedirs(output_dir, exist_ok=True)
    figure_path = os.path.join(output_dir, f"{gene_name}__{region_label}_omega_results.png")
    status_annotated_tree.render(figure_path, w=200, units="mm", dpi=300, tree_style=ts)

# ==============================================================================
# 5. PAML Caching System
# ==============================================================================

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _canonical_phy_sha(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return _sha256_bytes(raw)

def _exe_fingerprint(path: str) -> dict:
    st = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "sha256": _sha256_file(path)
    }

def _fanout_dir(root: str, key_hex: str) -> str:
    return os.path.join(root, key_hex[:CACHE_FANOUT], key_hex[CACHE_FANOUT:2*CACHE_FANOUT], key_hex)

def _atomic_write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def _try_lock(cache_dir: str) -> bool:
    os.makedirs(cache_dir, exist_ok=True)
    lockdir = os.path.join(cache_dir, "LOCK")
    try:
        os.mkdir(lockdir)
        return True
    except FileExistsError:
        return False

def _unlock(cache_dir: str):
    lockdir = os.path.join(cache_dir, "LOCK")
    try:
        os.rmdir(lockdir)
    except FileNotFoundError:
        pass

def _with_lock(cache_dir: str):
    class _LockCtx:
        def __init__(self, d): self.d = d; self.locked = False
        def __enter__(self):
            start = time.time()
            while time.time() - start < CACHE_LOCK_TIMEOUT_S:
                if _try_lock(self.d):
                    self.locked = True
                    return self
                time.sleep(random.uniform(*[x/1000 for x in CACHE_LOCK_POLL_MS]))
            return self
        def __exit__(self, *a):
            if self.locked: _unlock(self.d)
    return _LockCtx(cache_dir)

def _hash_key_attempt(gene_phy_sha, tree_str, taxa_used_list, ctl_str, exe_fp, init_kappa=None, init_omega=None):
    key_dict = {
        "schema": CACHE_SCHEMA_VERSION,
        "gene_phy_sha": gene_phy_sha,
        "tree_sha": _sha256_bytes(tree_str.encode("utf-8")),
        "taxa_used": sorted(taxa_used_list),
        "ctl_sha": _sha256_bytes(ctl_str.encode("utf-8")),
        "codeml": exe_fp["sha256"],
        "init_kappa": init_kappa,
        "init_omega": init_omega,
    }
    return _sha256_bytes(json.dumps(key_dict, sort_keys=True).encode("utf-8")), key_dict

def _hash_key_pair(h0_key_hex: str, h1_key_hex: str, test_label: str, df: int, exe_fp: dict):
    key_dict = {
        "schema": CACHE_SCHEMA_VERSION,
        "pair_version": 1,
        "test": test_label,
        "df": df,
        "h0_attempt_key": h0_key_hex,
        "h1_attempt_key": h1_key_hex,
        "codeml": exe_fp["sha256"],
    }
    return _sha256_bytes(json.dumps(key_dict, sort_keys=True).encode("utf-8")), key_dict

def _paml_attempt_worker(task: dict):
    """Execute a single codeml attempt with specific seeds and caching."""
    gene_name = task["gene_name"]
    region_label = task["region_label"]
    cache_dir = task["cache_dir"]
    key_hex = task["key_hex"]
    timeout = task["timeout"]
    parser_func = globals().get(task.get("parser_func_name")) if task.get("parser_func_name") else None
    run_dir = task["run_dir"]
    out_name = task["out_name"]

    def _copy_if_exists(src, dst_dir, dst_name=None):
        try:
            if src and os.path.exists(src):
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, (dst_name if dst_name else os.path.basename(src)))
                shutil.copy(src, dst)
                return dst
        except Exception:
            pass
        return None

    try:
        payload = cache_read_json(cache_dir, key_hex, "attempt.json")
        if payload:
            return {
                "seed": task["seed"],
                "lnl": payload.get("lnl", np.nan),
                "params": payload.get("params", {}),
                "status": "success" if np.isfinite(payload.get("lnl", np.nan)) else "fail",
                "key": key_hex,
                "init_kappa": task["init_kappa"],
                "init_omega": task["init_omega"],
            }

        os.makedirs(run_dir, exist_ok=True)
        ctl_file = os.path.join(run_dir, f"{gene_name}_{out_name}.ctl")
        out_file = os.path.join(run_dir, f"{gene_name}_{out_name}")

        generate_paml_ctl(
            ctl_file,
            task["phy_abs"],
            task["tree_path"],
            out_file,
            **task["model_params"],
            init_kappa=task["init_kappa"],
            init_omega=task["init_omega"],
        )
        run_codeml_in(run_dir, ctl_file, task["paml_bin"], timeout)
        lnl = parse_paml_lnl(out_file)
        parsed = parser_func(out_file) if parser_func else {}

        payload = {
            "lnl": float(lnl),
            "params": parsed,
            "init_kappa": task["init_kappa"],
            "init_omega": task["init_omega"],
        }

        target_dir = _fanout_dir(cache_dir, key_hex)
        with _with_lock(target_dir):
            cache_write_json(cache_dir, key_hex, "attempt.json", payload)
            artifact_dir = os.path.join(target_dir, "artifacts")
            os.makedirs(artifact_dir, exist_ok=True)
            _copy_if_exists(out_file, artifact_dir, out_name)
            _copy_if_exists(ctl_file, artifact_dir, f"{out_name}.ctl")
            _copy_if_exists(os.path.join(run_dir, "mlc"), artifact_dir, "mlc")
            _copy_if_exists(task["tree_path"], artifact_dir, f"{out_name}.tree")

        return {
            "seed": task["seed"],
            "lnl": float(lnl),
            "params": parsed,
            "status": "success",
            "key": key_hex,
            "init_kappa": task["init_kappa"],
            "init_omega": task["init_omega"],
        }
    except Exception as exc:
        return {
            "seed": task["seed"],
            "lnl": np.nan,
            "params": {},
            "status": "fail",
            "reason": str(exc),
            "key": key_hex,
            "init_kappa": task["init_kappa"],
            "init_omega": task["init_omega"],
        }

def cache_read_json(root: str, key_hex: str, name: str):
    path = os.path.join(_fanout_dir(root, key_hex), name)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def cache_write_json(root: str, key_hex: str, name: str, payload: dict):
    dest_dir = _fanout_dir(root, key_hex)
    os.makedirs(dest_dir, exist_ok=True)
    _atomic_write_json(os.path.join(dest_dir, name), payload)


# ==============================================================================
# 6. Core Analysis Wrappers ("The Business Logic")
# ==============================================================================

def run_iqtree_task(region_info, iqtree_bin, threads, output_dir, timeout=7200, make_figures=True, figure_dir=FIGURE_DIR):
    """Run IQ-TREE for a region after basic QC and cache its tree."""
    label = region_info['label']
    path = region_info['path']
    start_time = datetime.now()
    logging.info(f"[{label}] START IQ-TREE with {threads} threads")
    try:
        os.makedirs(output_dir, exist_ok=True)
        cached_tree = os.path.join(output_dir, f"{label}.treefile")
        if os.path.exists(cached_tree):
            logging.info(f"[{label}] Using cached tree")
            return (label, cached_tree, None)

        taxa = read_taxa_from_phy(path)
        chimp = next((t for t in taxa if 'pantro' in t.lower() or 'pan_troglodytes' in t.lower()), None)

        # Check individual requirements to build a specific error message
        reasons = []
        if not chimp:
            reasons.append("missing chimp outgroup")
        if len(taxa) < 6:
            reasons.append(f"insufficient taxa (found {len(taxa)}, need >= 6)")

        direct_count = sum(1 for t in taxa if t.startswith('0'))
        inverted_count = sum(1 for t in taxa if t.startswith('1'))

        if direct_count < 3:
            reasons.append(f"insufficient direct orientation samples (found {direct_count}, need >= 3)")
        if inverted_count < 3:
            reasons.append(f"insufficient inverted orientation samples (found {inverted_count}, need >= 3)")

        if reasons:
            diag_summary = _summarize_taxa_diagnostics(taxa)
            reason = "; ".join(reasons) + f". [Diagnostics: {diag_summary}]"
            logging.warning(f"[{label}] Skipping region: {reason}")
            return (label, None, reason)

        temp_dir_base = '/dev/shm' if os.path.exists('/dev/shm') and os.access('/dev/shm', os.W_OK) else None
        temp_dir = tempfile.mkdtemp(prefix=f"iqtree_{label}_", dir=temp_dir_base)
        prefix = os.path.join(temp_dir, label)
        cmd = [iqtree_bin, '-s', os.path.abspath(path), '-m', 'MFP', '-T', str(threads), '--prefix', prefix, '-quiet', '-o', chimp]
        run_command(cmd, temp_dir, timeout=timeout)
        tree_src = f"{prefix}.treefile"
        if not os.path.exists(tree_src):
            raise FileNotFoundError('treefile missing')

        tmp_copy = cached_tree + f".tmp.{os.getpid()}"
        shutil.copy(tree_src, tmp_copy)
        os.replace(tmp_copy, cached_tree)

        try:
            if make_figures:
                generate_tree_figure(cached_tree, label, output_dir=figure_dir, make_figures=True)
        except Exception as e:
            logging.error(f"[{label}] Failed to generate region tree figure: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        elapsed = (datetime.now() - start_time).total_seconds()
        logging.info(f"[{label}] END IQ-TREE ({elapsed:.1f}s)")
        return (label, cached_tree, None)
    except Exception as e:
        logging.error(f"[{label}] IQ-TREE failed: {e}")
        return (label, None, str(e))


def _log_tail(fp, n=35, prefix=""):
    try:
        with open(fp, 'r') as f:
            lines = f.readlines()[-n:]
        for ln in lines:
            logging.info("%s%s", f"[{prefix}] " if prefix else "", ln.rstrip())
    except Exception as e:
        logging.debug("Could not read tail of %s: %s", fp, e)

def run_codeml_in(run_dir, ctl_path, paml_bin, timeout):
    """Creates a directory for a single codeml run and executes it there."""
    os.makedirs(run_dir, exist_ok=True)
    for pat in ('rst*', 'rub*', '2NG*', '2ML*', 'lnf', 'mlc'):
        for f in glob.glob(os.path.join(run_dir, pat)):
            try:
                os.remove(f)
            except OSError:
                pass

    cmd = [paml_bin, ctl_path]
    repro_cmd = f"{shlex.quote(os.path.abspath(paml_bin))} {shlex.quote(os.path.abspath(ctl_path))}"
    logging.info(f"REPRODUCE PAML: {repro_cmd}")
    run_command(cmd, run_dir, timeout=timeout, input_data="\n")

# ==============================================================================
# 7. Parsing & Stats (Helpers)
# ==============================================================================

def parse_simple_paml_output(outfile_path):
    """
    Parse kappa and the background omega from a one-ratio or H0 run.
    Returns dict with keys: {'kappa': float, 'omega_background': float}
    """
    params = {'kappa': np.nan, 'omega_background': np.nan}
    if not os.path.exists(outfile_path):
        raise FileNotFoundError(f"PAML output missing: {outfile_path}")

    with open(outfile_path, 'r') as f:
        text = f.read()

    if not text.strip():
        raise ValueError(f"PAML output file is empty: {outfile_path}")

    if "Time used" not in text and "lnL" not in text:
        raise RuntimeError(f"PAML output appears incomplete/crashed: {outfile_path}")

    for line in text.splitlines():
        if line.startswith('kappa'):
            m = re.search(r'kappa \(ts/tv\) = \s*(' + FLOAT_REGEX + ')', line)
            if m: params['kappa'] = float(m.group(1))
        elif re.search(r'\bw\b.*\(dN/dS\)', line) or re.search(r'\bw\b for branch', line):
            m = re.search(r'=\s*(' + FLOAT_REGEX + r')|type 0:\s*(' + FLOAT_REGEX + ')', line)
            if m:
                params['omega_background'] = float(m.group(1) or m.group(2))
    return params

def parse_h1_paml_output(outfile_path):
    if not os.path.exists(outfile_path):
        raise FileNotFoundError(f"BM output missing: {outfile_path}")

    with open(outfile_path, 'r') as f:
        text = f.read()

    if not text.strip():
        raise ValueError(f"BM output file is empty: {outfile_path}")

    if "Time used" not in text and "lnL" not in text:
        raise RuntimeError(f"BM output appears incomplete/crashed: {outfile_path}")

    params = {'kappa': np.nan, 'omega_background': np.nan, 'omega_direct': np.nan, 'omega_inverted': np.nan}

    if not os.path.exists(outfile_path):
        raise RuntimeError(f"PAML output file missing: {outfile_path}")

    omega_lines = []
    with open(outfile_path, 'r') as f:
        content = f.read()
        if not content.strip():
            raise RuntimeError(f"PAML output file is empty: {outfile_path}")

        f.seek(0)
        for line in f:
            if line.lstrip().startswith('kappa'):
                m = re.search(r'kappa \(ts/tv\)\s*=\s*(' + FLOAT_REGEX + ')', line)
                if m: params['kappa'] = float(m.group(1))
            if re.search(r'\bw\s*\(dN/dS\)', line) or re.search(r'w\s*for\s*branch\s*type', line) or re.search(r'w\s*ratios?\s*for\s*branches?', line):
                omega_lines.append(line)


    for line in omega_lines:
        if re.search(r'branch type\s*0', line):
            m = re.search(r'type\s*0:\s*(' + FLOAT_REGEX + ')', line)
            if m: params['omega_background'] = float(m.group(1))
        elif re.search(r'branch type\s*1', line):
            m = re.search(r'type\s*1:\s*(' + FLOAT_REGEX + ')', line)
            if m: params['omega_direct'] = float(m.group(1))
        elif re.search(r'branch type\s*2', line):
            m = re.search(r'type\s*2:\s*(' + FLOAT_REGEX + ')', line)
            if m: params['omega_inverted'] = float(m.group(1))
        else:
            m = re.search(r'=\s*(' + FLOAT_REGEX + r')|branches:\s*(' + FLOAT_REGEX + ')', line)
            if m:
                v = m.group(1) or m.group(2)
                if v: params['omega_background'] = float(v)
    return params

def parse_h1_cmc_paml_output(outfile_path):
    F = FLOAT_REGEX
    params = {
        'cmc_kappa': np.nan,
        'cmc_p0': np.nan, 'cmc_p1': np.nan, 'cmc_p2': np.nan,
        'cmc_omega0': np.nan,
        'cmc_omega2_direct': np.nan,
        'cmc_omega2_inverted': np.nan,
    }

    if not os.path.exists(outfile_path):
        raise FileNotFoundError(f"PAML output file missing: {outfile_path}")

    with open(outfile_path, 'r', errors='ignore') as f:
        text = f.read()

    if not text.strip():

        raise ValueError(f"PAML output file is empty: {outfile_path}")

    if "Time used" not in text and "lnL" not in text:
        raise RuntimeError(f"PAML output appears incomplete/crashed: {outfile_path}")

    m = re.search(r'\bkappa\s*\(ts/tv\)\s*[=:]\s*(' + F + r')', text, re.I)
    if m: params['cmc_kappa'] = float(m.group(1))

    beb = re.search(r'Bayes\s+Empirical\s+Bayes', text, re.I)
    scan_text = text[:beb.start()] if beb else text

    block = scan_text
    mblk = re.search(r'MLEs\s+of\s+dN/dS\s*\(w\)\s*for\s*site\s*classes.*?(?:\n|$)', scan_text, re.I)
    if mblk:
        start = mblk.start()
        block = scan_text[start:start+1200]

    m = re.search(r'(?m)^\s*proportion\s+(' + F + r')\s+(' + F + r')\s+(' + F + r')\s*$', block, re.I)
    if m:
        params['cmc_p0'], params['cmc_p1'], params['cmc_p2'] = map(float, m.groups())

    def _grab_bt(n):
        m = re.search(r'(?m)^\s*branch\s*type\s*' + str(n) + r'\s*:\s*(' + F + r')\s+(' + F + r')\s+(' + F + r')\s*$', block, re.I)
        return tuple(map(float, m.groups())) if m else None

    bt0 = _grab_bt(0)
    bt1 = _grab_bt(1)
    bt2 = _grab_bt(2)

    if bt0: params['cmc_omega0'] = bt0[0]
    if bt1: params['cmc_omega2_direct'] = bt1[2]
    if bt2: params['cmc_omega2_inverted'] = bt2[2]

    if np.isnan(params['cmc_p2']) and not np.isnan(params['cmc_p0']) and not np.isnan(params['cmc_p1']):
        params['cmc_p2'] = max(0.0, 1.0 - params['cmc_p0'] - params['cmc_p1'])

    return params

def compute_fdr(df):
    """
    Wrapper around statsmodels.stats.multitest.fdrcorrection.
    Applies to 'bm_p_value' and 'cmc_p_value' columns if they exist and are successful.
    Returns the modified DataFrame.
    """
    successful = df[df['status'] == 'success'].copy()
    if successful.empty:
        return df

    # FDR for branch-model test
    if 'bm_p_value' in successful.columns:
        bm_pvals = successful['bm_p_value'].dropna()
        if not bm_pvals.empty:
            _, qvals = fdrcorrection(bm_pvals, alpha=FDR_ALPHA, method='indep')
            df.loc[bm_pvals.index, 'bm_q_value'] = qvals
            logging.info(f"Applied FDR correction to {len(bm_pvals)} branch-model tests.")

    # FDR for clade-model test
    if 'cmc_p_value' in successful.columns:
        cmc_pvals = successful['cmc_p_value'].dropna()
        if not cmc_pvals.empty:
            _, qvals = fdrcorrection(cmc_pvals, alpha=FDR_ALPHA, method='indep')
            df.loc[cmc_pvals.index, 'cmc_q_value'] = qvals
            logging.info(f"Applied FDR correction to {len(cmc_pvals)} clade-model tests.")

    return df

def _ctl_string(seqfile, treefile, outfile, *, model, NSsites, ncatG=None,
                init_kappa=None, init_omega=None, fix_blength=0, base_opts: dict = None):
    base_opts = base_opts or {}
    kappa = init_kappa if init_kappa is not None else 2.0
    omega = init_omega if init_omega is not None else 0.5
    codonfreq = base_opts.get('CodonFreq', 2)
    method = base_opts.get('method', 0)
    seqtype = base_opts.get('seqtype', 1)
    icode = base_opts.get('icode', 0)
    cleandata = base_opts.get('cleandata', 0)

    lines = [
        f"seqfile = {seqfile}",
        f"treefile = {treefile}",
        f"outfile = {outfile}",
        "noisy = 0",
        "verbose = 0",
        "runmode = 0",
        f"seqtype = {seqtype}",
        f"CodonFreq = {codonfreq}",
        f"model = {model}",
        f"NSsites = {NSsites}",
        f"icode = {icode}",
        f"cleandata = {cleandata}",
        "fix_kappa = 0",
        f"kappa = {kappa}",
        "fix_omega = 0",
        f"omega = {omega}",
        f"fix_blength = {fix_blength}",
        f"method = {method}",
        "getSE = 0",
        "RateAncestor = 0",
    ]
    if ncatG is not None:
        lines.insert(11, f"ncatG = {ncatG}")
    return "\n".join(lines).strip()

def generate_paml_ctl(ctl_path, phy_file, tree_file, out_file, *,
                      model, NSsites, ncatG=None,
                      init_kappa=None, init_omega=None, fix_blength=0):
    os.makedirs(os.path.dirname(ctl_path), exist_ok=True)
    kappa = 2.0 if init_kappa is None else init_kappa
    omega = 0.4 if init_omega is None else init_omega

    ncat_line = f"ncatG = {ncatG}" if ncatG is not None else ""

    content = f"""
seqfile = {phy_file}
treefile = {tree_file}
outfile = {out_file}
noisy = 0
verbose = 0
runmode = 0
seqtype = 1
CodonFreq = 2
model = {model}
NSsites = {NSsites}
{ncat_line}
icode = 0
cleandata = 0
fix_kappa = 0
kappa = {kappa}
fix_omega = 0
omega = {omega}
fix_blength = {fix_blength}
method = 0
getSE = 0
RateAncestor = 0
""".strip() + "\n"
    with open(ctl_path, "w") as f:
        f.write(content)

def parse_paml_lnl(outfile_path):
    """Extracts the log-likelihood (lnL) value from a PAML output file."""
    with open(outfile_path, 'r') as f:
        for line in f:
            if 'lnL' in line:
                match = re.search(r'lnL\(.*\):\s*(' + FLOAT_REGEX + ')', line)
                if match:
                    return float(match.group(1))
    raise ValueError(f"Could not parse lnL from {outfile_path}")
def analyze_single_gene(gene_info, region_tree_path, region_label, paml_bin, cache_dir,
                        timeout=PAML_TIMEOUT,
                        run_branch_model=False,
                        run_clade_model=True,
                        proceed_on_terminal_only=False,
                        keep_paml_out=False,
                        paml_out_dir="paml_runs",
                        make_figures=True,
                        annotated_figure_dir=ANNOTATED_FIGURE_DIR,
                        target_clade_model=None):
    """Run codeml for a gene using the provided region tree."""
    gene_name = gene_info['label']
    normalized_target = (target_clade_model or "both").lower()
    final_result = {
        'gene': gene_name,
        'region': region_label,
        'status': 'runtime_error',
        'reason': 'Unknown failure',
        'paml_model': normalized_target,
    }
    temp_dir = None
    start_time = datetime.now()
    logging.info(f"[{gene_name}|{region_label}] START codeml")

    try:
        qc_passed, qc_message = perform_qc(gene_info['path'])
        if not qc_passed:
            final_result.update({'status': 'qc_fail', 'reason': qc_message})
            logging.warning(f"[{gene_name}|{region_label}] QC failed: {qc_message}")
            return final_result

        temp_dir_base = '/dev/shm' if os.path.exists('/dev/shm') and os.access('/dev/shm', os.W_OK) else os.getenv("PAML_TMPDIR")
        temp_dir = tempfile.mkdtemp(prefix=f"paml_{gene_name}_{region_label}_", dir=temp_dir_base)

        region_taxa = Tree(region_tree_path, format=1).get_leaf_names()
        gene_taxa = read_taxa_from_phy(gene_info['path'])
        keep = [taxon for taxon in gene_taxa if taxon in set(region_taxa)]
        has_chimp = any('pantro' in t.lower() or 'pan_troglodytes' in t.lower() for t in keep)
        if not has_chimp:
            final_result.update({'status': 'uninformative_topology', 'reason': 'Chimp outgroup missing in gene alignment'})
            return final_result
        if len(keep) < 4:
            final_result.update({'status': 'uninformative_topology', 'reason': f'Fewer than four shared taxa (n={len(keep)})'})
            return final_result

        pruned_tree = os.path.join(temp_dir, f"{gene_name}_pruned.tree")
        prune_region_tree(region_tree_path, keep, pruned_tree)
        t = Tree(pruned_tree, format=1)

        var_codons = count_variable_codon_sites(gene_info['path'], set(keep))
        if var_codons < 2:
            final_result.update({'status': 'uninformative_topology', 'reason': f'Fewer than 2 variable codon sites ({var_codons})'})
            return final_result

        if len(t.get_leaf_names()) < 4:
            final_result.update({'status': 'uninformative_topology', 'reason': 'Fewer than four taxa after pruning'})
            return final_result

        h1_tree, h0_tree, informative, status_tree, reason = create_paml_tree_files(pruned_tree, temp_dir, gene_name)
        if not informative:
            if proceed_on_terminal_only:
                logging.warning(f"[{gene_name}] No pure internal branches in both clades; proceeding as PROCEED_ON_TERMINAL_ONLY is True (lower power).")
            else:
                final_result.update({'status': 'uninformative_topology', 'reason': reason or 'No pure internal branches found for both direct and inverted groups.'})
                return final_result

        phy_abs = os.path.abspath(gene_info['path'])

        os.makedirs(cache_dir, exist_ok=True)
        exe_fp = _exe_fingerprint(paml_bin)
        gene_phy_sha = _canonical_phy_sha(phy_abs)
        h0_tree_str = _read_text(h0_tree)
        h1_tree_str = _read_text(h1_tree)
        taxa_used = t.get_leaf_names()

        ctl_bm_h0 = _ctl_string(phy_abs, h0_tree, "H0_bm.out", model=2, NSsites=0)
        ctl_bm_h1 = _ctl_string(phy_abs, h1_tree, "H1_bm.out", model=2, NSsites=0)
        h0_bm_key, _ = _hash_key_attempt(gene_phy_sha, h0_tree_str, taxa_used, ctl_bm_h0, exe_fp)
        h1_bm_key, _ = _hash_key_attempt(gene_phy_sha, h1_tree_str, taxa_used, ctl_bm_h1, exe_fp)

        def get_attempt_result(key_hex, tree_path, out_name, model_params, parser_func):
            def _safe_json_load(p):
                try:
                    with open(p, "r", encoding="utf-8") as f: return json.load(f)
                except Exception: return None

            def _copy_if_exists(src, dst_dir, dst_name=None):
                try:
                    if src and os.path.exists(src):
                        os.makedirs(dst_dir, exist_ok=True)
                        dst = os.path.join(dst_dir, (dst_name if dst_name else os.path.basename(src)))
                        shutil.copy(src, dst)
                        return dst
                except Exception: pass
                return None

            def _sha_file_safe(p):
                try: return _sha256_file(p)
                except Exception: return None

            payload = cache_read_json(cache_dir, key_hex, "attempt.json")
            if payload:
                logging.info(f"[{gene_name}|{region_label}] Using cached ATTEMPT (current key): {out_name}")
                art_dir = os.path.join(_fanout_dir(cache_dir, key_hex), "artifacts")
                tree_copy = os.path.join(art_dir, f"{out_name}.tree")
                if not os.path.exists(tree_copy):
                    _copy_if_exists(tree_path, art_dir, f"{out_name}.tree")

                try:
                    if parser_func:
                        need_keys = ("cmc_p0","cmc_p1","cmc_p2","cmc_omega0","cmc_omega2_direct","cmc_omega2_inverted")
                        params = payload.get("params", {}) or {}
                        def _bad(x): return x is None or (isinstance(x, float) and np.isnan(x))
                        if any(_bad(params.get(k)) for k in need_keys):
                            candidates = [os.path.join(art_dir, out_name), os.path.join(art_dir, "mlc")]
                            healed = {}
                            for raw_path in candidates:
                                if os.path.exists(raw_path):
                                    try: healed = parser_func(raw_path) or {}
                                    except Exception: pass
                                    if healed: break
                            if healed:
                                for k, v in healed.items():
                                    if _bad(params.get(k)) and v is not None and not (isinstance(v, float) and np.isnan(v)):
                                        params[k] = v
                                payload["params"] = params
                                cache_write_json(cache_dir, key_hex, "attempt.json", payload)
                                logging.info(f"[{gene_name}|{region_label}] Healed attempt.json params from artifacts for {out_name}")
                except Exception: pass
                return payload

            run_dir = os.path.join(temp_dir, out_name.replace(".out", ""))
            ctl_file = os.path.join(run_dir, f"{gene_name}_{out_name}.ctl")
            out_file = os.path.join(run_dir, f"{gene_name}_{out_name}")

            params = {**model_params, 'init_kappa': 2.0, 'init_omega': 0.5, 'fix_blength': model_params.get('fix_blength', 0)}
            generate_paml_ctl(ctl_file, phy_abs, tree_path, out_file, **params)
            run_codeml_in(run_dir, ctl_file, paml_bin, timeout)
            _log_tail(out_file, 25, prefix=f"[{gene_name}|{region_label}] {out_name} out (computed)")

            lnl = parse_paml_lnl(out_file)
            parsed = parser_func(out_file) if parser_func else {}

            payload = {"lnl": float(lnl), "params": parsed}

            target_dir = _fanout_dir(cache_dir, key_hex)
            with _with_lock(target_dir):
                cache_write_json(cache_dir, key_hex, "attempt.json", payload)
                artifact_dir = os.path.join(target_dir, "artifacts")
                os.makedirs(artifact_dir, exist_ok=True)
                _copy_if_exists(out_file, artifact_dir, out_name)
                _copy_if_exists(ctl_file, artifact_dir, f"{out_name}.ctl")
                mlc_path = os.path.join(run_dir, "mlc")
                _copy_if_exists(mlc_path, artifact_dir, "mlc")
                _copy_if_exists(tree_path, artifact_dir, f"{out_name}.tree")

            logging.info(f"[{gene_name}|{region_label}] Cached attempt {out_name} to {target_dir}")
            return payload

        bm_result = {}
        if run_branch_model:
            pair_key_bm, pair_key_dict_bm = _hash_key_pair(h0_bm_key, h1_bm_key, "branch_model", 1, exe_fp)
            pair_payload_bm = cache_read_json(cache_dir, pair_key_bm, "pair.json")
            if pair_payload_bm:
                logging.info(f"[{gene_name}|{region_label}] Using cached PAIR result for branch_model")
                bm_result = pair_payload_bm["result"]
            else:
                with ThreadPoolExecutor(max_workers=2) as ex:
                    fut_h0 = ex.submit(get_attempt_result, h0_bm_key, h0_tree, "H0_bm.out", {"model": 2, "NSsites": 0}, None)
                    fut_h1 = ex.submit(get_attempt_result, h1_bm_key, h1_tree, "H1_bm.out", {"model": 2, "NSsites": 0}, parse_h1_paml_output)
                    h0_payload = fut_h0.result()
                    h1_payload = fut_h1.result()
                lnl0, lnl1 = h0_payload.get("lnl", -np.inf), h1_payload.get("lnl", -np.inf)

                if np.isfinite(lnl0) and np.isfinite(lnl1) and lnl1 >= lnl0:
                    lrt = 2 * (lnl1 - lnl0)
                    p = chi2.sf(lrt, df=1)
                    bm_result = {
                        "bm_lnl_h0": lnl0, "bm_lnl_h1": lnl1, "bm_lrt_stat": float(lrt), "bm_p_value": float(p),
                        **{f"bm_{k}": v for k, v in h1_payload.get("params", {}).items()},
                        "bm_h0_key": h0_bm_key, "bm_h1_key": h1_bm_key,
                    }
                    with _with_lock(_fanout_dir(cache_dir, pair_key_bm)):
                        cache_write_json(cache_dir, pair_key_bm, "pair.json", {"key": pair_key_dict_bm, "result": bm_result})
                    logging.info(f"[{gene_name}|{region_label}] Cached LRT pair 'branch_model' (df=1)")
                else:
                    bm_result = {
                        "bm_p_value": np.nan,
                        "bm_lrt_stat": np.nan,
                        "bm_lnl_h0": lnl0,
                        "bm_lnl_h1": lnl1,
                        "bm_h0_key": h0_bm_key,
                        "bm_h1_key": h1_bm_key
                    }
                    with _with_lock(_fanout_dir(cache_dir, pair_key_bm)):
                        cache_write_json(cache_dir, pair_key_bm, "pair.json", {"key": pair_key_dict_bm, "result": bm_result})
                    logging.info(f"[{gene_name}|{region_label}] Cached LRT pair 'branch_model' (invalid or non-improvement)")
        else:
            logging.info(f"[{gene_name}|{region_label}] Skipping branch-model test as per configuration.")
            bm_result = {"bm_p_value": np.nan, "bm_lrt_stat": np.nan}

        cmc_result = {}
        if run_clade_model:
            target = normalized_target
            run_h0 = target in ("both", "h0")
            run_h1 = target in ("both", "h1")

            def _build_attempts(tree_path, tree_str, out_name, parser_func_name=None):
                attempts = []
                for seed_name, cfg in SEED_BANK.items():
                    ctl_str = _ctl_string(
                        phy_abs,
                        tree_path,
                        out_name,
                        model=3,
                        NSsites=2,
                        ncatG=3,
                        init_kappa=cfg["kappa"],
                        init_omega=cfg["omega"],
                    )
                    key_hex, _ = _hash_key_attempt(
                        gene_phy_sha,
                        tree_str,
                        taxa_used,
                        ctl_str,
                        exe_fp,
                        init_kappa=cfg["kappa"],
                        init_omega=cfg["omega"],
                    )
                    attempts.append({
                        "seed": seed_name,
                        "gene_name": gene_name,
                        "region_label": region_label,
                        "cache_dir": cache_dir,
                        "key_hex": key_hex,
                        "phy_abs": phy_abs,
                        "tree_path": tree_path,
                        "out_name": out_name,
                        "model_params": {"model": 3, "NSsites": 2, "ncatG": 3},
                        "paml_bin": paml_bin,
                        "timeout": timeout,
                        "init_kappa": cfg["kappa"],
                        "init_omega": cfg["omega"],
                        "parser_func_name": parser_func_name,
                        "run_dir": os.path.join(temp_dir, f"{out_name.replace('.out', '')}_{seed_name}"),
                    })
                return attempts

            def _run_tournament(attempts):
                if not attempts:
                    return []
                workers = min(len(attempts), max(1, os.cpu_count() or len(attempts)), 4)
                results = []
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futures = {ex.submit(_paml_attempt_worker, att): att for att in attempts}
                    for fut in as_completed(futures):
                        try:
                            results.append(fut.result())
                        except Exception as exc:
                            att = futures[fut]
                            results.append({
                                "seed": att.get("seed"),
                                "lnl": np.nan,
                                "params": {},
                                "status": "fail",
                                "reason": str(exc),
                                "key": att.get("key_hex"),
                                "init_kappa": att.get("init_kappa"),
                                "init_omega": att.get("init_omega"),
                            })
                return results

            h0_attempts = _build_attempts(h0_tree, h0_tree_str, "H0_cmc.out") if run_h0 else []
            h1_attempts = _build_attempts(h1_tree, h1_tree_str, "H1_cmc.out", parser_func_name="parse_h1_cmc_paml_output") if run_h1 else []

            h0_results = _run_tournament(h0_attempts)
            h1_results = _run_tournament(h1_attempts)

            def _record_attempts(results, prefix):
                for res in results:
                    base = f"{prefix}_{res['seed']}_"
                    final_result[f"{base}status"] = res.get("status")
                    final_result[f"{base}lnl"] = res.get("lnl", np.nan)
                    final_result[f"{base}init_kappa"] = res.get("init_kappa")
                    final_result[f"{base}init_omega"] = res.get("init_omega")
                    final_result[f"{base}key"] = res.get("key")
                    if res.get("reason"):
                        final_result[f"{base}reason"] = res.get("reason")
                    for k, v in (res.get("params") or {}).items():
                        final_result[f"{base}{k}"] = v

            _record_attempts(h0_results, "h0")
            _record_attempts(h1_results, "h1")

            def _best_result(results):
                successes = [r for r in results if r.get("status") == "success" and np.isfinite(r.get("lnl", np.nan))]
                if not successes:
                    return None
                return max(successes, key=lambda r: r.get("lnl", -np.inf))

            h0_best = _best_result(h0_results)
            h1_best = _best_result(h1_results)

            final_result["h0_winner_seed"] = h0_best.get("seed") if h0_best else None
            final_result["h1_winner_seed"] = h1_best.get("seed") if h1_best else None

            if run_h0 and run_h1 and h0_best and h1_best:
                pair_key_cmc, pair_key_dict_cmc = _hash_key_pair(h0_best["key"], h1_best["key"], "clade_model_c", 1, exe_fp)
                pair_payload_cmc = cache_read_json(cache_dir, pair_key_cmc, "pair.json")
                if pair_payload_cmc:
                    logging.info(f"[{gene_name}|{region_label}] Using cached PAIR result for clade_model_c")
                    cmc_result = dict(pair_payload_cmc["result"])
                else:
                    lnl0, lnl1 = h0_best.get("lnl", -np.inf), h1_best.get("lnl", -np.inf)
                    if np.isfinite(lnl0) and np.isfinite(lnl1) and lnl1 >= lnl0:
                        lrt = 2 * (lnl1 - lnl0)
                        p = chi2.sf(lrt, df=1)
                        cmc_result = {
                            "cmc_lnl_h0": lnl0,
                            "cmc_lnl_h1": lnl1,
                            "cmc_lrt_stat": float(lrt),
                            "cmc_p_value": float(p),
                            **(h1_best.get("params") or {}),
                            "cmc_h0_key": h0_best.get("key"),
                            "cmc_h1_key": h1_best.get("key"),
                        }
                        with _with_lock(_fanout_dir(cache_dir, pair_key_cmc)):
                            cache_write_json(cache_dir, pair_key_cmc, "pair.json", {"key": pair_key_dict_cmc, "result": cmc_result})
                        logging.info(f"[{gene_name}|{region_label}] Cached LRT pair 'clade_model_c' (df=1)")
                    else:
                        cmc_result = {
                            "cmc_p_value": np.nan,
                            "cmc_lrt_stat": np.nan,
                            "cmc_lnl_h0": lnl0,
                            "cmc_lnl_h1": lnl1,
                            "cmc_h0_key": h0_best.get("key"),
                            "cmc_h1_key": h1_best.get("key"),
                        }
                        with _with_lock(_fanout_dir(cache_dir, pair_key_cmc)):
                            cache_write_json(cache_dir, pair_key_cmc, "pair.json", {"key": pair_key_dict_cmc, "result": cmc_result})
                        logging.info(f"[{gene_name}|{region_label}] Cached LRT pair 'clade_model_c' (invalid or non-improvement)")
                cmc_result["h0_winner_seed"] = h0_best.get("seed")
                cmc_result["h1_winner_seed"] = h1_best.get("seed")
            elif run_h0 and h0_best:
                cmc_result = {
                    "cmc_lnl_h0": h0_best.get("lnl", np.nan),
                    "cmc_lnl_h1": np.nan,
                    "cmc_p_value": np.nan,
                    "cmc_lrt_stat": np.nan,
                    "cmc_h0_key": h0_best.get("key"),
                    "cmc_h1_key": None,
                    "h0_winner_seed": h0_best.get("seed"),
                    "h1_winner_seed": None,
                }
            elif run_h1 and h1_best:
                cmc_result = {
                    "cmc_lnl_h0": np.nan,
                    "cmc_lnl_h1": h1_best.get("lnl", np.nan),
                    "cmc_p_value": np.nan,
                    "cmc_lrt_stat": np.nan,
                    **(h1_best.get("params") or {}),
                    "cmc_h0_key": None,
                    "cmc_h1_key": h1_best.get("key"),
                    "h0_winner_seed": None,
                    "h1_winner_seed": h1_best.get("seed"),
                }
            else:
                cmc_result = {
                    "cmc_p_value": np.nan,
                    "cmc_lrt_stat": np.nan,
                    "cmc_lnl_h0": np.nan,
                    "cmc_lnl_h1": np.nan,
                    "cmc_h0_key": h0_best.get("key") if h0_best else None,
                    "cmc_h1_key": h1_best.get("key") if h1_best else None,
                    "h0_winner_seed": h0_best.get("seed") if h0_best else None,
                    "h1_winner_seed": h1_best.get("seed") if h1_best else None,
                }
        else:
            logging.info(f"[{gene_name}|{region_label}] Skipping clade-model test as per configuration.")
            cmc_result = {"cmc_p_value": np.nan, "cmc_lrt_stat": np.nan}

        bm_ok = not run_branch_model or not np.isnan(bm_result.get("bm_p_value", np.nan))
        clade_has_partial = run_clade_model and (
            (cmc_result.get("cmc_h0_key") is not None) or (cmc_result.get("cmc_h1_key") is not None)
        )
        cmc_ok = not run_clade_model or not np.isnan(cmc_result.get("cmc_p_value", np.nan))

        if bm_ok and cmc_ok:
            final_result.update({
                "status": "success", **bm_result, **cmc_result,
                "n_leaves_region": len(region_taxa), "n_leaves_gene": len(gene_taxa), "n_leaves_pruned": len(taxa_used),
                "chimp_in_region": any('pantro' in n.lower() for n in region_taxa),
                "chimp_in_pruned": any('pantro' in n.lower() for n in t.get_leaf_names()),
                "taxa_used": ';'.join(taxa_used)
            })
        elif bm_ok or clade_has_partial:
            final_result.update({
                "status": "partial_success",
                "reason": "Only one clade model hypothesis executed; deferring LRT to final aggregation.",
                **bm_result, **cmc_result,
                "n_leaves_region": len(region_taxa), "n_leaves_gene": len(gene_taxa), "n_leaves_pruned": len(taxa_used),
                "chimp_in_region": any('pantro' in n.lower() for n in region_taxa),
                "chimp_in_pruned": any('pantro' in n.lower() for n in t.get_leaf_names()),
                "taxa_used": ';'.join(taxa_used)
            })
        else:
            final_result.update({
                "status": "paml_optim_fail",
                "reason": "One or more requested LRTs failed to produce a valid result.",
                **bm_result, **cmc_result,
            })

        if keep_paml_out and final_result.get('status') == 'success':
            try:
                safe_region = re.sub(r'[^A-Za-z0-9_.-]+', '_', region_label)
                safe_gene   = re.sub(r'[^A-Za-z0-9_.-]+', '_', gene_name)
                dest_dir = os.path.join(paml_out_dir, f"{safe_gene}__{safe_region}")
                os.makedirs(dest_dir, exist_ok=True)

                for key in ["bm_h0_key", "bm_h1_key", "cmc_h0_key", "cmc_h1_key"]:
                    if final_result.get(key):
                        artifact_dir = os.path.join(_fanout_dir(cache_dir, final_result[key]), "artifacts")
                        if os.path.isdir(artifact_dir):
                            for f in os.listdir(artifact_dir):
                                shutil.copy(os.path.join(artifact_dir, f), dest_dir)

                if os.path.exists(h1_tree): shutil.copy(h1_tree, dest_dir)
                if os.path.exists(h0_tree): shutil.copy(h0_tree, dest_dir)
                if os.path.exists(pruned_tree): shutil.copy(pruned_tree, dest_dir)

            except Exception as e:
                logging.error(f"[{gene_name}|{region_label}] Failed to copy artifacts for keep_paml_out: {e}")

        if final_result['status'] == 'success':
            try:
                bm_params = {
                    'omega_direct': final_result.get('bm_omega_direct'),
                    'omega_inverted': final_result.get('bm_omega_inverted'),
                    'omega_background': final_result.get('bm_omega_background'),
                }
                generate_omega_result_figure(gene_name, region_label, status_tree, bm_params, output_dir=annotated_figure_dir, make_figures=make_figures)
            except Exception as fig_exc:
                logging.error(f"[{gene_name}] Failed to generate PAML results figure: {fig_exc}")

        elapsed = (datetime.now() - start_time).total_seconds()
        if final_result.get('status') == 'success':
            bm_stat = final_result.get('bm_lrt_stat'); bm_p = final_result.get('bm_p_value')
            cmc_stat = final_result.get('cmc_lrt_stat'); cmc_p = final_result.get('cmc_p_value')
            logging.info(f"[{gene_name}|{region_label}] "
                         f"BM LRT={bm_stat if pd.notna(bm_stat) else 'NA'} p={bm_p if pd.notna(bm_p) else 'NA'} | "
                         f"CMC LRT={cmc_stat if pd.notna(cmc_stat) else 'NA'} p={cmc_p if pd.notna(cmc_p) else 'NA'}")
        logging.info(f"[{gene_name}|{region_label}] END codeml ({elapsed:.1f}s) status={final_result['status']}")

        return final_result

    except Exception as e:
        logging.error(f"FATAL ERROR for gene '{gene_name}' under region '{region_label}'.\n{traceback.format_exc()}")
        final_result.update({'status': 'runtime_error', 'reason': str(e)})
        return final_result
    finally:
        if temp_dir:
            logging.info(f"[{gene_name}|{region_label}] PAML run directory available at: {temp_dir}")
