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
from concurrent.futures import ThreadPoolExecutor
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

ALLOWED_REGIONS = [
    ("chr1", 13104252, 13122521),
    ("chr10", 46983451, 47468232),
    ("chr11", 50154999, 50324102),
    ("chr15", 30618103, 32153204),
    ("chr15", 84373375, 84416696),
    ("chr16", 16721273, 18073542),
    ("chr16", 28471892, 28637651),
    ("chr2", 91832040, 92012663),
    ("chr2", 95800191, 96024403),
    ("chr3", 195749463, 195980207),
    ("chr7", 5989046, 6735643),
    ("chr7", 60911891, 61578023),
    ("chr7", 65219157, 65531823),
    ("chr7", 73113989, 74799029),
    ("chr7", 74869950, 75058098),
    ("chr8", 2343351, 2378385),
    ("chr8", 7301024, 12598379),
    ("chrX", 103989434, 104049428),
    ("chrX", 149599490, 149655967),
    ("chrX", 149681035, 149722249),
    ("chrX", 153149748, 153250226),
    ("chrX", 154347246, 154384867),
    ("chrX", 154591327, 154613096),
    ("chrX", 155386727, 155453982),
    ("chrX", 52077120, 52176974),
    ("chrX", 72997772, 73077479),
    ("chr1", 108310642, 108383736),
    ("chr11", 89920623, 89923848),
    ("chr16", 14954790, 15100859),
    ("chr7", 62290674, 62363143),
    ("chr7", 62408486, 62456444),
    ("chrX", 152729753, 152738707),
]

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

ENABLE_PAML_TIMEOUT = False
IQTREE_TIMEOUT = int(os.environ.get("IQTREE_TIMEOUT", "7200"))
PAML_TIMEOUT   = int(os.environ.get("PAML_TIMEOUT", "3600")) if ENABLE_PAML_TIMEOUT else None

RUN_BRANCH_MODEL_TEST = False
RUN_CLADE_MODEL_TEST = True
PROCEED_ON_TERMINAL_ONLY = False


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
        raise FileNotFoundError(
            f"Metadata file '{tsv_path}' not found; cannot map genes to regions.")

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
        return False, "Could not find both human and chimp sequences."
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
        logging.warning(f"[{gene_name}] Skipping due to insufficient samples in a group (direct: {direct_leaves}, inverted: {inverted_leaves}).")
        return None, None, False, t

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
    if not analysis_is_informative:
        logging.warning(f"[{gene_name}] Topology is uninformative for internal branch analysis.")

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
        logging.warning(f"[{gene_name}] H1 tree has no labeled branches; treating as uninformative.")
        return None, None, False, t
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

    return h1_tree_path, h0_tree_path, analysis_is_informative, t

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

def _hash_key_attempt(gene_phy_sha, tree_str, taxa_used_list, ctl_str, exe_fp):
    key_dict = {
        "schema": CACHE_SCHEMA_VERSION,
        "gene_phy_sha": gene_phy_sha,
        "tree_sha": _sha256_bytes(tree_str.encode("utf-8")),
        "taxa_used": sorted(taxa_used_list),
        "ctl_sha": _sha256_bytes(ctl_str.encode("utf-8")),
        "codeml": exe_fp["sha256"],
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
        taxa = read_taxa_from_phy(path)
        chimp = next((t for t in taxa if 'pantro' in t.lower() or 'pan_troglodytes' in t.lower()), None)
        if not chimp or len(taxa) < 6 or not any(t.startswith('0') for t in taxa) or not any(t.startswith('1') for t in taxa):
            reason = 'missing chimp or insufficient taxa/diversity'
            logging.warning(f"[{label}] Skipping region: {reason}")
            return (label, None, reason)

        os.makedirs(output_dir, exist_ok=True)
        cached_tree = os.path.join(output_dir, f"{label}.treefile")
        if os.path.exists(cached_tree):
            logging.info(f"[{label}] Using cached tree")
            return (label, cached_tree, None)

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
    with open(outfile_path, 'r') as f:
        for line in f:
            if line.startswith('kappa'):
                m = re.search(r'kappa \(ts/tv\) = \s*(' + FLOAT_REGEX + ')', line)
                if m: params['kappa'] = float(m.group(1))
            elif re.search(r'\bw\b.*\(dN/dS\)', line) or re.search(r'\bw\b for branch', line):
                m = re.search(r'=\s*(' + FLOAT_REGEX + r')|type 0:\s*(' + FLOAT_REGEX + ')', line)
                if m:
                    params['omega_background'] = float(m.group(1) or m.group(2))
    return params

def parse_h1_paml_output(outfile_path):
    params = {'kappa': np.nan, 'omega_background': np.nan, 'omega_direct': np.nan, 'omega_inverted': np.nan}
    omega_lines = []
    with open(outfile_path, 'r') as f:
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

    try:
        with open(outfile_path, 'r', errors='ignore') as f:
            text = f.read()
    except Exception:
        return params

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
    omega = 0.5 if init_omega is None else init_omega
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
{('ncatG = ' + str(ncatG)) if ncatG is not None else ''}
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
                        timeout=3600,
                        run_branch_model=False,
                        run_clade_model=True,
                        proceed_on_terminal_only=False,
                        keep_paml_out=False,
                        paml_out_dir="paml_runs",
                        make_figures=True,
                        annotated_figure_dir=ANNOTATED_FIGURE_DIR):
    """Run codeml for a gene using the provided region tree."""
    gene_name = gene_info['label']
    final_result = {'gene': gene_name, 'region': region_label, 'status': 'runtime_error', 'reason': 'Unknown failure'}
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

        h1_tree, h0_tree, informative, status_tree = create_paml_tree_files(pruned_tree, temp_dir, gene_name)
        if not informative:
            if proceed_on_terminal_only:
                logging.warning(f"[{gene_name}] No pure internal branches in both clades; proceeding as PROCEED_ON_TERMINAL_ONLY is True (lower power).")
            else:
                final_result.update({'status': 'uninformative_topology', 'reason': 'No pure internal branches found for both direct and inverted groups.'})
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

        ctl_cmc_h0 = _ctl_string(phy_abs, h0_tree, "H0_cmc.out", model=3, NSsites=2, ncatG=3)
        ctl_cmc_h1 = _ctl_string(phy_abs, h1_tree, "H1_cmc.out", model=3, NSsites=2, ncatG=3)
        h0_cmc_key, _ = _hash_key_attempt(gene_phy_sha, h0_tree_str, taxa_used, ctl_cmc_h0, exe_fp)
        h1_cmc_key, _ = _hash_key_attempt(gene_phy_sha, h1_tree_str, taxa_used, ctl_cmc_h1, exe_fp)

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

            def _parse_ctl_fields(ctl_path):
                FINT = r'[-+]?\d+'
                rx = {
                    "seqfile": re.compile(r'^\s*seqfile\s*=\s*(.+?)\s*$', re.I|re.M),
                    "treefile": re.compile(r'^\s*treefile\s*=\s*(.+?)\s*$', re.I|re.M),
                    "model": re.compile(r'^\s*model\s*=\s*(' + FINT + r')\s*$', re.I|re.M),
                    "NSsites": re.compile(r'^\s*NSsites\s*=\s*(' + FINT + r'(?:\s+' + FINT + r')*)\s*$', re.I|re.M),
                    "ncatG": re.compile(r'^\s*ncatG\s*=\s*(' + FINT + r')\s*$', re.I|re.M),
                    "fix_blength": re.compile(r'^\s*fix_blength\s*=\s*(' + FINT + r')\s*$', re.I|re.M),
                }
                try: s = _read_text(ctl_path)
                except Exception: return None
                def _pick_int(key, default=None):
                    m = rx[key].search(s)
                    return int(m.group(1)) if m else default
                ns_m = rx["NSsites"].search(s)
                ns_val = None
                if ns_m:
                    toks = ns_m.group(1).strip().split()
                    if toks:
                        try: ns_val = int(toks[0])
                        except ValueError: ns_val = None
                seqfile = rx["seqfile"].search(s)
                treefile = rx["treefile"].search(s)
                return {
                    "seqfile": seqfile.group(1).strip() if seqfile else None,
                    "treefile": treefile.group(1).strip() if treefile else None,
                    "model": _pick_int("model"),
                    "NSsites": ns_val,
                    "ncatG": _pick_int("ncatG", None),
                    "fix_blength": _pick_int("fix_blength", 0),
                }

            def _sha_file_safe(p):
                try: return _sha256_file(p)
                except Exception: return None

            def _legacy_find_equivalent(out_name, expect_params, expect_gene_phy_sha, expect_tree_sha):
                cur_dir = _fanout_dir(cache_dir, key_hex)
                candidate = os.path.join(cur_dir, "artifacts", f"{out_name}.ctl")
                if os.path.exists(candidate):
                    fields = _parse_ctl_fields(candidate)
                    if fields and fields["model"] == expect_params.get("model") \
                       and fields["NSsites"] == expect_params.get("NSsites") \
                       and (fields["ncatG"] or None) == expect_params.get("ncatG") \
                       and (fields["fix_blength"] or 0) == expect_params.get("fix_blength", 0):
                        seq_sha = _sha_file_safe(fields["seqfile"]) if fields.get("seqfile") else None
                        tree_sha = _sha_file_safe(fields["treefile"]) if fields.get("treefile") else None
                        if seq_sha == expect_gene_phy_sha and tree_sha == expect_tree_sha:
                            payload = cache_read_json(cache_dir, key_hex, "attempt.json")
                            if payload: return payload, cur_dir, candidate, fields.get("treefile")

                for lvl1 in (os.listdir(cache_dir) if os.path.isdir(cache_dir) else []):
                    p1 = os.path.join(cache_dir, lvl1)
                    if not os.path.isdir(p1) or len(lvl1) != 2: continue
                    for lvl2 in (os.listdir(p1) if os.path.isdir(p1) else []):
                        p2 = os.path.join(p1, lvl2)
                        if not os.path.isdir(p2) or len(lvl2) != 2: continue
                        for keydir in (os.listdir(p2) if os.path.isdir(p2) else []):
                            kd = os.path.join(p2, keydir)
                            if not os.path.isdir(kd): continue
                            att_json = os.path.join(kd, "attempt.json")
                            if not os.path.exists(att_json): continue
                            ctl_candidate = os.path.join(kd, "artifacts", f"{out_name}.ctl")
                            out_candidate = os.path.join(kd, "artifacts", out_name)
                            if not os.path.exists(ctl_candidate) or not os.path.exists(out_candidate): continue
                            fields = _parse_ctl_fields(ctl_candidate)
                            if not fields: continue
                            if fields["model"] != expect_params.get("model"): continue
                            if fields["NSsites"] != expect_params.get("NSsites"): continue
                            if (fields["ncatG"] or None) != expect_params.get("ncatG"): continue
                            if (fields["fix_blength"] or 0) != expect_params.get("fix_blength", 0): continue
                            seq_sha = _sha_file_safe(fields["seqfile"]) if fields.get("seqfile") else None
                            if seq_sha != expect_gene_phy_sha: continue
                            tree_sha = _sha_file_safe(fields["treefile"]) if fields.get("treefile") else None
                            if tree_sha != expect_tree_sha: continue
                            payload = _safe_json_load(att_json)
                            if payload and isinstance(payload, dict) and "lnl" in payload:
                                return payload, kd, ctl_candidate, fields.get("treefile")
                return None, None, None, None

            def _rehydrate_under_new_key(new_key_hex, payload, legacy_dir, legacy_ctl, legacy_tree):
                target_dir = _fanout_dir(cache_dir, new_key_hex)
                with _with_lock(target_dir):
                    cache_write_json(cache_dir, new_key_hex, "attempt.json", payload)
                    art_dst = os.path.join(target_dir, "artifacts")
                    os.makedirs(art_dst, exist_ok=True)
                    _copy_if_exists(os.path.join(legacy_dir, "artifacts", out_name), art_dst, out_name)
                    _copy_if_exists(os.path.join(legacy_dir, "artifacts", f"{out_name}.ctl"), art_dst, f"{out_name}.ctl")
                    _copy_if_exists(os.path.join(legacy_dir, "artifacts", "mlc"), art_dst, "mlc")
                    if legacy_tree and os.path.exists(legacy_tree):
                        _copy_if_exists(legacy_tree, art_dst, f"{out_name}.tree")

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

            expect_gene_phy_sha = gene_phy_sha
            expect_tree_sha = _sha_file_safe(tree_path)
            expect_params = {
                "model": model_params.get("model"),
                "NSsites": model_params.get("NSsites"),
                "ncatG": model_params.get("ncatG"),
                "fix_blength": model_params.get("fix_blength", 0),
            }
            legacy_payload, legacy_dir, legacy_ctl, legacy_tree = _legacy_find_equivalent(
                out_name, expect_params, expect_gene_phy_sha, expect_tree_sha
            )
            if legacy_payload:
                logging.info(f"[{gene_name}|{region_label}] Using cached ATTEMPT (legacy rehydrated): {out_name}")
                _rehydrate_under_new_key(key_hex, legacy_payload, legacy_dir, legacy_ctl, legacy_tree)
                return legacy_payload

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
            pair_key_cmc, pair_key_dict_cmc = _hash_key_pair(h0_cmc_key, h1_cmc_key, "clade_model_c", 1, exe_fp)
            pair_payload_cmc = cache_read_json(cache_dir, pair_key_cmc, "pair.json")
            if pair_payload_cmc:
                logging.info(f"[{gene_name}|{region_label}] Using cached PAIR result for clade_model_c")
                cmc_result = dict(pair_payload_cmc["result"])
                h1_key = cmc_result.get("cmc_h1_key") or pair_payload_cmc.get("key", {}).get("h1_attempt_key")
                def _bad(x): return (x is None) or (isinstance(x, float) and np.isnan(x))
                healed = {}
                if h1_key:
                    art_dir_h1 = os.path.join(_fanout_dir(cache_dir, h1_key), "artifacts")
                    candidates = [os.path.join(art_dir_h1, "H1_cmc.out"), os.path.join(art_dir_h1, "mlc")]
                    for raw_h1 in candidates:
                        if os.path.exists(raw_h1):
                            try: healed = parse_h1_cmc_paml_output(raw_h1) or {}
                            except Exception: pass
                            if healed: break

                if healed:
                    changed = False
                    for k, v in healed.items():
                        if k.startswith("cmc_") and (_bad(cmc_result.get(k)) or (k not in cmc_result)):
                            cmc_result[k] = v
                            changed = True
                    if changed:
                        with _with_lock(_fanout_dir(cache_dir, pair_key_cmc)):
                            cache_write_json(cache_dir, pair_key_cmc, "pair.json",
                                             {"key": pair_payload_cmc["key"], "result": cmc_result})
                        logging.info(f"[{gene_name}|{region_label}] Back-filled cmc_* in cached pair.json")
                    try:
                        if h1_key:
                            att = cache_read_json(cache_dir, h1_key, "attempt.json")
                            if isinstance(att, dict):
                                aparams = att.get("params", {}) or {}
                                a_changed = False
                                for k, v in healed.items():
                                    if k.startswith("cmc_") and (_bad(aparams.get(k)) or (k not in aparams)):
                                        aparams[k] = v
                                        a_changed = True
                                if a_changed:
                                    att["params"] = aparams
                                    cache_write_json(cache_dir, h1_key, "attempt.json", att)
                                    logging.info(f"[{gene_name}|{region_label}] Back-filled cmc_* in H1 attempt.json")
                    except Exception: pass
            else:
                with ThreadPoolExecutor(max_workers=2) as ex:
                    fut_h0 = ex.submit(get_attempt_result, h0_cmc_key, h0_tree, "H0_cmc.out",
                                       {"model": 3, "NSsites": 2, "ncatG": 3}, None)
                    fut_h1 = ex.submit(get_attempt_result, h1_cmc_key, h1_tree, "H1_cmc.out",
                                       {"model": 3, "NSsites": 2, "ncatG": 3}, parse_h1_cmc_paml_output)
                    h0_payload = fut_h0.result()
                    h1_payload = fut_h1.result()

                lnl0, lnl1 = h0_payload.get("lnl", -np.inf), h1_payload.get("lnl", -np.inf)
                if np.isfinite(lnl0) and np.isfinite(lnl1) and lnl1 >= lnl0:
                    lrt = 2 * (lnl1 - lnl0)
                    p = chi2.sf(lrt, df=1)
                    cmc_result = {
                        "cmc_lnl_h0": lnl0,
                        "cmc_lnl_h1": lnl1,
                        "cmc_lrt_stat": float(lrt),
                        "cmc_p_value": float(p),
                        **h1_payload.get("params", {}),
                        "cmc_h0_key": h0_cmc_key,
                        "cmc_h1_key": h1_cmc_key,
                    }
                    with _with_lock(_fanout_dir(cache_dir, pair_key_cmc)):
                        cache_write_json(cache_dir, pair_key_cmc, "pair.json",
                                         {"key": pair_key_dict_cmc, "result": cmc_result})
                    logging.info(f"[{gene_name}|{region_label}] Cached LRT pair 'clade_model_c' (df=1)")
                else:
                    cmc_result = {
                        "cmc_p_value": np.nan,
                        "cmc_lrt_stat": np.nan,
                        "cmc_lnl_h0": lnl0,
                        "cmc_lnl_h1": lnl1,
                        "cmc_h0_key": h0_cmc_key,
                        "cmc_h1_key": h1_cmc_key
                    }
                    with _with_lock(_fanout_dir(cache_dir, pair_key_cmc)):
                        cache_write_json(cache_dir, pair_key_cmc, "pair.json", {"key": pair_key_dict_cmc, "result": cmc_result})
                    logging.info(f"[{gene_name}|{region_label}] Cached LRT pair 'clade_model_c' (invalid or non-improvement)")
        else:
            logging.info(f"[{gene_name}|{region_label}] Skipping clade-model test as per configuration.")
            cmc_result = {"cmc_p_value": np.nan, "cmc_lrt_stat": np.nan}

        bm_ok = not run_branch_model or not np.isnan(bm_result.get("bm_p_value", np.nan))
        cmc_ok = not run_clade_model or not np.isnan(cmc_result.get("cmc_p_value", np.nan))

        if bm_ok and cmc_ok:
            final_result.update({
                "status": "success", **bm_result, **cmc_result,
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
