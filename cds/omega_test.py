import os
import re
import sys
import glob
import subprocess
import multiprocessing
import tempfile
import getpass
import logging
from logging.handlers import QueueHandler, QueueListener
import traceback
from datetime import datetime
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import deque
import threading
import time

# --- Scientific Computing Imports ---
import numpy as np
from scipy.stats import chi2
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
import pandas as pd

# --- ETE3 and QT Configuration for Headless Environments ---
# This is a necessary workaround to run ete3 in environments without a display server.
os.environ["QT_QPA_PLATFORM"] = "offscreen"
user = getpass.getuser()
runtime_dir = f"/tmp/runtime-{user}"
os.makedirs(runtime_dir, exist_ok=True, mode=0o700)
os.environ['XDG_RUNTIME_DIR'] = runtime_dir
from ete3 import Tree
from ete3.treeview import TreeStyle, NodeStyle, TextFace, CircleFace, RectFace

# --- Configuration Toggles ---
# Set to False to disable the PAML timeout for debugging or long runs.
ENABLE_PAML_TIMEOUT = False
# Toggle these flags to run only specific PAML models.
RUN_BRANCH_MODEL_TEST = False
RUN_CLADE_MODEL_TEST = True

# === REGION WHITELIST =========================================================
# Only regions whose (chromosome, start, end) triple appears in this list will run.
# Coordinates are inclusive and must exactly match the parsed filename values.
ALLOWED_REGIONS = [
    ("chr12", 46896694, 46915975),
    ("chr17", 45585159, 46292045),
    ("chr6", 76109081, 76158474),
    ("chr7", 57835188, 58119653),
]

# === PAML CACHE CONFIG =======================================================
import hashlib, json, time, random, shlex

PAML_CACHE_DIR = os.environ.get("PAML_CACHE_DIR", "paml_cache")
CACHE_SCHEMA_VERSION = "paml_cache.v1"
CACHE_FANOUT = 2  # two levels of 2 hex chars -> 256*256 buckets
CACHE_LOCK_TIMEOUT_S = int(os.environ.get("PAML_CACHE_LOCK_TIMEOUT_S", "600"))  # 10 min
CACHE_LOCK_POLL_MS = (50, 250)  # jittered backoff range

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
    # Minimal canonicalization: strip trailing spaces; keep original header
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
    # Ensure the target fanout directory exists before attempting to create the lock directory.
    # This prevents race conditions and avoids FileNotFoundError when creating the LOCK directory.
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

# ==============================================================================
# === CONFIGURATION & SETUP ====================================================
# ==============================================================================

# --- Centralized Logging ---
# A unique log file is created for each pipeline run.
LOG_FILE = f"pipeline_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

def start_logging():
    """Initializes queue-based logging for multiprocessing."""
    log_q = multiprocessing.Queue(-1)
    
    # The listener pulls from the queue and sends to the handlers.
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    listener = QueueListener(log_q, file_handler, stream_handler)
    listener.start()
    return log_q, listener

def worker_logging_init(log_q):
    """Configures logging for a worker process to use the shared queue."""
    root = logging.getLogger()
    root.handlers[:] = [QueueHandler(log_q)]
    root.setLevel(logging.INFO)

# --- Paths to Executables ---
# Assumes executables are in specific locations relative to the script's runtime directory.
IQTREE_PATH = os.path.abspath('../iqtree-3.0.1-Linux/bin/iqtree3')
PAML_PATH = os.path.abspath('../paml/bin/codeml')

# --- Analysis Parameters ---
DIVERGENCE_THRESHOLD = 0.10  # Max median human-chimp divergence to pass QC.
FDR_ALPHA = 0.05             # False Discovery Rate for significance.
PROCEED_ON_TERMINAL_ONLY = False # If True, proceed with analysis even if no pure internal branches are found.
FLOAT_REGEX = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'

# --- Visualization Configuration ---
POP_COLORS = {
    'AFR': '#F05031', 'EUR': '#3173F0', 'EAS': '#35A83A',
    'SAS': '#F031D3', 'AMR': '#B345F0', 'CHIMP': '#808080'
}

# --- Output Directories and Files ---
FIGURE_DIR = "tree_figures"
ANNOTATED_FIGURE_DIR = "annotated_tree_figures"
RESULTS_TSV = f"full_paml_results_{datetime.now().strftime('%Y-%m-%d')}.tsv"
REGION_TREE_DIR = "region_trees"

# --- Checkpointing and Output Retention ---
CHECKPOINT_FILE = "paml_results.checkpoint.tsv"
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "100"))
KEEP_PAML_OUT = bool(int(os.environ.get("KEEP_PAML_OUT", "0")))
PAML_OUT_DIR  = os.environ.get("PAML_OUT_DIR", "paml_runs")

# --- Concurrency & runtime knobs ---
def _detect_cpus():
    # Prefer cgroup/affinity-aware counts if available
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # Fallback for systems without sched_getaffinity (e.g., Windows)
        return os.cpu_count() or 1

CPU_COUNT = _detect_cpus()
REGION_WORKERS = int(os.environ.get("REGION_WORKERS", max(1, min(CPU_COUNT // 3, 4))))
# By default, give most CPUs to PAML, but let user override.
default_paml = max(1, CPU_COUNT - REGION_WORKERS)
if CPU_COUNT >= 4:
    default_paml = max(2, default_paml)
PAML_WORKERS = int(os.environ.get("PAML_WORKERS", default_paml))

# Optional: gate figure generation (tree render can be surprisingly expensive)
MAKE_FIGURES = bool(int(os.environ.get("MAKE_FIGURES", "1")))

# Subprocess timeouts (seconds). Tweak as appropriate for your datasets/cluster.
IQTREE_TIMEOUT = int(os.environ.get("IQTREE_TIMEOUT", "7200"))   # 2h default
PAML_TIMEOUT   = int(os.environ.get("PAML_TIMEOUT", "3600")) if ENABLE_PAML_TIMEOUT else None     # 1h default

# Prevent hidden multi-threading from MKL/OpenBLAS in child processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ==============================================================================
# === GENERIC HELPER FUNCTIONS =================================================
# ==============================================================================

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

def perform_qc(phy_file_path):
    """
    Performs quality control checks on a given phylip file.
    Checks for non-zero length, valid codon alignment, and human-chimp divergence.
    
    Returns:
        tuple: (bool, str) indicating if QC passed and a message.
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

def _tree_layout(node):
    """A layout function to dynamically style nodes for ete3 tree figures."""
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

def generate_tree_figure(tree_file, label):
    """Creates a publication-quality phylogenetic tree figure using ete3."""
    if not MAKE_FIGURES:
        return
    t = Tree(tree_file, format=1)
    ts = TreeStyle()
    ts.layout_fn = _tree_layout
    ts.show_leaf_name = False
    ts.show_scale = False
    ts.branch_vertical_margin = 8
    ts.title.add_face(TextFace(f"Phylogeny of Region {label}", fsize=16, ftype="Arial"), column=0)
    
    # Legend
    ts.legend.add_face(TextFace("Haplotype Status", fsize=10, ftype="Arial", fstyle="Bold"), column=0)
    ts.legend.add_face(CircleFace(5, "black", style="circle"), column=0); ts.legend.add_face(TextFace(" Direct", fsize=9), column=1)
    ts.legend.add_face(CircleFace(5, "black", style="sphere"), column=0); ts.legend.add_face(TextFace(" Inverted", fsize=9), column=1)
    ts.legend.add_face(RectFace(10, 10, "black", "black"), column=0); ts.legend.add_face(TextFace(" Chimpanzee (Outgroup)", fsize=9), column=1)
    ts.legend.add_face(TextFace(" "), column=2) # Spacer
    ts.legend.add_face(TextFace("Super-population", fsize=10, ftype="Arial", fstyle="Bold"), column=3)
    for pop, color in POP_COLORS.items():
        ts.legend.add_face(CircleFace(10, color), column=3); ts.legend.add_face(TextFace(f" {pop}", fsize=9), column=4)
    ts.legend_position = 1
    
    figure_path = os.path.join(FIGURE_DIR, f"{label}.png")
    t.render(figure_path, w=200, units="mm", dpi=300, tree_style=ts)

def generate_omega_result_figure(gene_name, region_label, status_annotated_tree, paml_params):
    """
    Creates a tree figure with branches colored by their estimated omega (dN/dS) value.
    This function visualizes the final results from the PAML model=2 analysis.

    Args:
        gene_name (str): The name of the gene, used for the figure title.
        region_label (str): Identifier for the region providing the topology.
        status_annotated_tree (ete3.Tree): The tree object with 'group_status' on each node.
        paml_params (dict): A dictionary of parsed omega values from the PAML H1 run.
    """
    if not MAKE_FIGURES:
        return
    # Define colors for selection regimes based on omega values
    PURIFYING_COLOR = "#0072B2" # Blue
    POSITIVE_COLOR = "#D55E00"  # Vermillion
    NEUTRAL_COLOR = "#000000"   # Black

    # This layout function determines the color of each branch based on its
    # group's estimated omega value.
    def _omega_color_layout(node):
        nstyle = NodeStyle()
        nstyle["hz_line_width"] = 2
        nstyle["vt_line_width"] = 2

        # Determine which omega value applies to this branch
        status = getattr(node, "group_status", "both")
        omega_val = 1.0 # Default to neutral
        if status == 'direct':
            omega_val = paml_params.get('omega_direct', 1.0)
        elif status == 'inverted':
            omega_val = paml_params.get('omega_inverted', 1.0)
        else: # 'both' and 'outgroup' fall into the background category
            omega_val = paml_params.get('omega_background', 1.0)

        # Assign color based on the omega value
        if omega_val > 1.0:
            color = POSITIVE_COLOR
        elif omega_val < 1.0:
            color = PURIFYING_COLOR
        else:
            color = NEUTRAL_COLOR
        
        nstyle["hz_line_color"] = color
        nstyle["vt_line_color"] = color

        # Style leaves to show their population identity, as before
        if node.is_leaf():
            name = node.name
            pop_match = re.search(r'_(AFR|EUR|EAS|SAS|AMR)_', name)
            pop = pop_match.group(1) if pop_match else 'CHIMP'
            leaf_color = POP_COLORS.get(pop, "#C0C0C0")
            nstyle["fgcolor"] = leaf_color
            nstyle["size"] = 5
        else:
            nstyle["size"] = 0 # Keep internal nodes invisible for a clean look

        node.set_style(nstyle)

    ts = TreeStyle()
    ts.layout_fn = _omega_color_layout
    ts.show_leaf_name = False
    ts.show_scale = False
    ts.branch_vertical_margin = 8
    ts.title.add_face(TextFace(f"dN/dS for {gene_name} under {region_label}", fsize=16, ftype="Arial"), column=0)
    
    # --- Create a dynamic legend based on the actual PAML results ---
    ts.legend.add_face(TextFace("Selection Regime (ω = dN/dS)", fsize=10, ftype="Arial", fstyle="Bold"), column=0)
    
    legend_map = {
        'Direct Group': paml_params.get('omega_direct'),
        'Inverted Group': paml_params.get('omega_inverted'),
        'Background': paml_params.get('omega_background'),
    }

    for name, omega in legend_map.items():
        if omega is not None and not np.isnan(omega):
            if omega > 1.0: color = POSITIVE_COLOR
            elif omega < 1.0: color = PURIFYING_COLOR
            else: color = NEUTRAL_COLOR
            legend_text = f" {name} (ω = {omega:.3f})"
            ts.legend.add_face(RectFace(10, 10, fgcolor=color, bgcolor=color), column=0)
            ts.legend.add_face(TextFace(legend_text, fsize=9), column=1)

    ts.legend_position = 4 # Position the legend in the top-right

    figure_path = os.path.join(ANNOTATED_FIGURE_DIR, f"{gene_name}__{region_label}_omega_results.png")
    status_annotated_tree.render(figure_path, w=200, units="mm", dpi=300, tree_style=ts)

# ==============================================================================
# === CORE ANALYSIS FUNCTIONS  ======================================
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
        lines.insert(11, f"ncatG = {ncatG}")  # after NSsites for stability
    return "\n".join(lines).strip()


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
        "test": test_label,           # "branch_model" or "clade_model_c"
        "df": df,                     # 1 here
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

def _with_lock(cache_dir: str):
    # contextmanager inline (py<=3.7 friendly)
    class _LockCtx:
        def __init__(self, d): self.d = d; self.locked = False
        def __enter__(self):
            start = time.time()
            while time.time() - start < CACHE_LOCK_TIMEOUT_S:
                if _try_lock(self.d):
                    self.locked = True
                    return self
                time.sleep(random.uniform(*[x/1000 for x in CACHE_LOCK_POLL_MS]))
            return self  # timeout → proceed without lock (best-effort)
        def __exit__(self, *a):
            if self.locked: _unlock(self.d)
    return _LockCtx(cache_dir)


def _validate_internal_branch_labels(paml_tree_str: str, tree_obj: Tree, expected_marks: list):
    expected_counts = {mark: 0 for mark in expected_marks}
    for node in tree_obj.traverse():
        if not node.is_leaf() and hasattr(node, "paml_mark"):
            if node.paml_mark in expected_counts:
                expected_counts[node.paml_mark] += 1

    actual_counts = {mark: 0 for mark in expected_marks}
    for mark in expected_marks:
        # allow optional ": <float>" between the closing paren and the label
        pattern = re.compile(r"\)\s*(?::\s*" + FLOAT_REGEX + r")?\s*" + re.escape(mark))
        actual_counts[mark] = len(pattern.findall(paml_tree_str))

    for mark in expected_marks:
        assert actual_counts[mark] == expected_counts[mark], \
            f"Internal branch label validation failed for mark '{mark}'. Expected {expected_counts[mark]}, found {actual_counts[mark]}. Tree string: {paml_tree_str}"


def create_paml_tree_files(iqtree_file, work_dir, gene_name):
    logging.info(f"[{gene_name}] Labeling internal branches conservatively...")
    t = Tree(iqtree_file, format=1)

    # Step 1: Propagate status up from the leaves ("post-order" traversal).
    # A temporary attribute 'group_status' is added to each node.
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
        else:  # This is an internal node.
            # Collect the statuses of all its immediate children.
            child_statuses = {child.group_status for child in node.children}

            if len(child_statuses) == 1:
                # If all children have the exact same status (e.g., all are 'inverted'),
                # this internal node inherits that "pure" status.
                node.add_feature("group_status", child_statuses.pop())
            else:
                # If children have different statuses (e.g., one is 'inverted' and one is 'direct',
                # or one is 'direct' and one is 'both'), this is a shared/ambiguous ancestor.
                node.add_feature("group_status", "both")

    # Group Size Gate: Ensure there are enough samples in each group to proceed.
    if direct_leaves < 3 or inverted_leaves < 3:
        logging.warning(f"[{gene_name}] Skipping due to insufficient samples in a group (direct: {direct_leaves}, inverted: {inverted_leaves}).")
        return None, None, False, t

    # Step 2: Check if the analysis will be informative by counting pure internal branches.
    internal_direct_count = 0
    internal_inverted_count = 0
    for node in t.traverse():
        # We only care about internal nodes for this count.
        if not node.is_leaf():
            status = getattr(node, "group_status", "both")
            if status == "direct":
                internal_direct_count += 1
            elif status == "inverted":
                internal_inverted_count += 1

    logging.info(f"[{gene_name}] Found {internal_direct_count} pure 'direct' internal branches.")
    logging.info(f"[{gene_name}] Found {internal_inverted_count} pure 'inverted' internal branches.")

    # The analysis is only considered informative if BOTH groups have at least one pure internal branch.
    analysis_is_informative = (internal_direct_count > 0 and internal_inverted_count > 0)
    if not analysis_is_informative:
        logging.warning(f"[{gene_name}] Topology is uninformative for internal branch analysis.")

    # Step 3: Create H1 (Alternative Model) Tree.
    # This traversal applies the PAML labels based on the determined 'group_status'.
    # This labels both pure internal nodes AND the terminal leaf branches.
    t_h1 = t.copy()
    for node in t_h1.traverse():
        status = getattr(node, "group_status", "both")
        if status == "direct":
            node.add_feature("paml_mark", "#1")
        elif status == "inverted":
            node.add_feature("paml_mark", "#2")
        # Any node with 'both' or 'outgroup' status remains unlabeled, defaulting to background.

    h1_newick = t_h1.write(format=1, features=["paml_mark"])
    # The regex cleans up the ete3 output to be PAML-compatible.
    h1_paml_str = re.sub(r"\[&&NHX:paml_mark=(#\d+)\]", r" \1", h1_newick)
    if (" #1" not in h1_paml_str) and (" #2" not in h1_paml_str):
        logging.warning(f"[{gene_name}] H1 tree has no labeled branches; treating as uninformative.")
        return None, None, False, t
    _validate_internal_branch_labels(h1_paml_str, t_h1, ['#1', '#2'])
    h1_tree_path = os.path.join(work_dir, f"{gene_name}_H1.tree")
    with open(h1_tree_path, 'w') as f:
        f.write("1\n" + h1_paml_str + "\n")


    # Step 4: Create H0 (Null Model) Tree.
    # Same logic: lump all pure human branches (internal and terminal) into one foreground group.
    t_h0 = t.copy()
    for node in t_h0.traverse():
        status = getattr(node, "group_status", "both")
        if status in ["direct", "inverted"]:
            node.add_feature("paml_mark", "#1") # Foreground group

    h0_newick = t_h0.write(format=1, features=["paml_mark"])
    h0_paml_str = re.sub(r"\[&&NHX:paml_mark=(#1)\]", r" \1", h0_newick)
    _validate_internal_branch_labels(h0_paml_str, t_h0, ['#1'])
    h0_tree_path = os.path.join(work_dir, f"{gene_name}_H0.tree")
    with open(h0_tree_path, 'w') as f:
        f.write("1\n" + h0_paml_str + "\n")

    # Return the tree object 't' which now has the 'group_status' features attached.
    return h1_tree_path, h0_tree_path, analysis_is_informative, t

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

def parse_h1_paml_output(outfile_path):
    params = {'kappa': np.nan, 'omega_background': np.nan, 'omega_direct': np.nan, 'omega_inverted': np.nan}
    omega_lines = []
    with open(outfile_path, 'r') as f:
        for line in f:
            if line.lstrip().startswith('kappa'):
                m = re.search(r'kappa \(ts/tv\)\s*=\s*(' + FLOAT_REGEX + ')', line)
                if m: params['kappa'] = float(m.group(1))
            # be permissive about indentation and wording
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
    """
    Parse PAML clade model C (model=3, NSsites=2, ncatG=3) from mlc/out.
    Extracts:
      cmc_kappa, cmc_p0, cmc_p1, cmc_p2, cmc_omega0,
      cmc_omega2_direct (branch type 1, col 3),
      cmc_omega2_inverted (branch type 2, col 3).
    Robust to spacing and ignores BEB 'w0:' grid.
    """
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

    # kappa
    m = re.search(r'\bkappa\s*\(ts/tv\)\s*[=:]\s*(' + F + r')', text, re.I)
    if m:
        params['cmc_kappa'] = float(m.group(1))

    # Ignore everything after BEB so we don't hit the 'w0:' grid
    beb = re.search(r'Bayes\s+Empirical\s+Bayes', text, re.I)
    scan_text = text[:beb.start()] if beb else text

    # Narrow to the 'MLEs of dN/dS (w) for site classes' block if present
    block = scan_text
    mblk = re.search(r'MLEs\s+of\s+dN/dS\s*\(w\)\s*for\s*site\s*classes.*?(?:\n|$)', scan_text, re.I)
    if mblk:
        start = mblk.start()
        # read a small window; the table is right below
        block = scan_text[start:start+1200]

    # proportions: "proportion  p0  p1  p2"
    m = re.search(r'(?m)^\s*proportion\s+(' + F + r')\s+(' + F + r')\s+(' + F + r')\s*$', block, re.I)
    if m:
        params['cmc_p0'], params['cmc_p1'], params['cmc_p2'] = map(float, m.groups())

    # branch type rows: "... type N:  <w0>  1.00000  <w2_for_that_branch>"
    def _grab_bt(n):
        m = re.search(r'(?m)^\s*branch\s*type\s*' + str(n) + r'\s*:\s*(' + F + r')\s+(' + F + r')\s+(' + F + r')\s*$', block, re.I)
        return tuple(map(float, m.groups())) if m else None

    bt0 = _grab_bt(0)
    bt1 = _grab_bt(1)
    bt2 = _grab_bt(2)

    if bt0: params['cmc_omega0'] = bt0[0]
    if bt1: params['cmc_omega2_direct'] = bt1[2]
    if bt2: params['cmc_omega2_inverted'] = bt2[2]

    # If p2 missing but p0/p1 found, compute it
    if np.isnan(params['cmc_p2']) and not np.isnan(params['cmc_p0']) and not np.isnan(params['cmc_p1']):
        params['cmc_p2'] = max(0.0, 1.0 - params['cmc_p0'] - params['cmc_p1'])

    return params

# ============================================================================
# === REGION/GENE HELPER FUNCTIONS ===========================================
# ============================================================================

def parse_region_filename(path):
    """Extract chromosome and coordinates from a region filename (accepts with/without 'chr')."""
    name = os.path.basename(path)
    # Accept: combined_inversion_14_start123_end456.phy and combined_inversion_chr14_start123_end456.phy
    m = re.match(r"^combined_inversion_(?:chr)?([0-9]+|X|Y|M|MT)_start(\d+)_end(\d+)\.phy$", name, re.I)
    if not m:
        # Also accept: combined_inversion_14_123_456.phy and combined_inversion_chr14_123_456.phy
        m = re.match(r"^combined_inversion_(?:chr)?([0-9]+|X|Y|M|MT)_(\d+)_(\d+)\.phy$", name, re.I)
    if not m:
        raise ValueError(f"Unrecognized region filename format: {name}")

    chrom_token, start_str, end_str = m.groups()
    chrom_token = chrom_token.upper()
    chrom = "chrM" if chrom_token in ("M", "MT") else f"chr{chrom_token}"
    start = int(start_str)
    end = int(end_str)
    if start > end:
        logging.warning(f"Region {name}: start({start}) > end({end}); swapping.")
        start, end = end, start

    return {
        'path': path,
        'chrom': chrom,
        'start': start,
        'end': end,
        'label': f"{chrom}_{start}_{end}"
    }



def load_gene_metadata(tsv_path='phy_metadata.tsv'):
    """Load gene coordinate metadata from a TSV file robustly."""
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(
            "Metadata file 'phy_metadata.tsv' not found; cannot map genes to regions.")

    # Read as strings so we can normalise and coerce ourselves
    df = pd.read_csv(tsv_path, sep='\t', dtype=str)

    # Map possible column aliases to canonical names
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

    # Normalise chromosome strings
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

    # Drop rows with missing critical values
    before = len(df)
    df = df.dropna(subset=['_gene', '_enst', '_chr', '_start', '_end'])
    dropped_missing = before - len(df)
    if dropped_missing:
        logging.warning(
            f"Metadata: dropped {dropped_missing} rows with missing gene/enst/chr/start/end.")

    # Swap start/end if reversed
    flipped = (df['_start'] > df['_end']).sum()
    if flipped:
        logging.warning(
            f"Metadata: found {flipped} rows with start > end; swapping.")
        s = df['_start'].copy()
        df.loc[df['_start'] > df['_end'], '_start'] = df.loc[df['_start'] > df['_end'], '_end']
        df.loc[df['_start'] > df['_end'], '_end'] = s[df['_start'] > df['_end']]

    # Collapse duplicates keeping widest span
    df['_width'] = (df['_end'] - df['_start']).abs()
    df = df.sort_values(['_gene', '_enst', '_width'], ascending=[True, True, False])
    dupes = df.duplicated(subset=['_gene', '_enst']).sum()
    if dupes:
        logging.info(
            f"Metadata: collapsing {dupes} duplicate (gene,enst) rows; keeping widest span.")
    df = df.drop_duplicates(subset=['_gene', '_enst'], keep='first')

    # Final cast to ints
    df['_start'] = df['_start'].round().astype(int)
    df['_end'] = df['_end'].round().astype(int)

    meta = {}
    for _, row in df.iterrows():
        meta[(row['_gene'], row['_enst'])] = {
            'chrom': row['_chr'],
            'start': int(row['_start']),
            'end': int(row['_end']),
        }

    logging.info(
        f"Loaded metadata for {len(meta)} (gene,enst) pairs after cleaning.")
    return meta


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
        # Coordinates were encoded in the filename
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


def prune_region_tree(region_tree_path, taxa_to_keep, out_path):
    """Prune the region tree to the intersection of taxa."""
    tree = Tree(region_tree_path, format=1)
    leaf_names = set(tree.get_leaf_names())
    keep = [taxon for taxon in taxa_to_keep if taxon in leaf_names]
    tree.prune(keep, preserve_branch_length=True)
    tree.write(outfile=out_path, format=1)
    return out_path


def count_variable_codon_sites(phy_path, taxa_subset=None, max_sites_check=50000):
    # Lightweight, column-wise variability check
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
    # Cap work on huge alignments
    for i in range(0, min(seqlen, max_sites_check), 3):
        col = {s[i:i+3] for s in seqs if len(s) >= i+3}
        col = {c for c in col if '-' not in c and 'N' not in c and 'n' not in c}
        if len(col) > 1:
            var_codons += 1
    return var_codons


def region_worker(region, iqtree_threads):
    """Run IQ-TREE for a region after basic QC and cache its tree."""
    label = region['label']
    path = region['path']
    start_time = datetime.now()
    logging.info(f"[{label}] START IQ-TREE with {iqtree_threads} threads")
    try:
        taxa = read_taxa_from_phy(path)
        chimp = next((t for t in taxa if 'pantro' in t.lower() or 'pan_troglodytes' in t.lower()), None)
        if not chimp or len(taxa) < 6 or not any(t.startswith('0') for t in taxa) or not any(t.startswith('1') for t in taxa):
            reason = 'missing chimp or insufficient taxa/diversity'
            logging.warning(f"[{label}] Skipping region: {reason}")
            return (label, None, reason)

        os.makedirs(REGION_TREE_DIR, exist_ok=True)
        cached_tree = os.path.join(REGION_TREE_DIR, f"{label}.treefile")
        if os.path.exists(cached_tree):
            logging.info(f"[{label}] Using cached tree")
            return (label, cached_tree, None)

        temp_dir_base = '/dev/shm' if os.path.exists('/dev/shm') and os.access('/dev/shm', os.W_OK) else None
        temp_dir = tempfile.mkdtemp(prefix=f"iqtree_{label}_", dir=temp_dir_base)
        prefix = os.path.join(temp_dir, label)
        cmd = [IQTREE_PATH, '-s', os.path.abspath(path), '-m', 'MFP', '-T', str(iqtree_threads), '--prefix', prefix, '-quiet', '-o', chimp]
        run_command(cmd, temp_dir, timeout=IQTREE_TIMEOUT)
        tree_src = f"{prefix}.treefile"
        if not os.path.exists(tree_src):
            raise FileNotFoundError('treefile missing')
        
        # Atomic copy to prevent corrupted cache files
        tmp_copy = cached_tree + f".tmp.{os.getpid()}"
        shutil.copy(tree_src, tmp_copy)
        os.replace(tmp_copy, cached_tree)

        try:
            generate_tree_figure(cached_tree, label)
        except Exception as e:
            logging.error(f"[{label}] Failed to generate region tree figure: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        elapsed = (datetime.now() - start_time).total_seconds()
        logging.info(f"[{label}] END IQ-TREE ({elapsed:.1f}s)")
        return (label, cached_tree, None)
    except Exception as e:
        logging.error(f"[{label}] IQ-TREE failed: {e}")
        return (label, None, str(e))


# ============================================================================
# === GENE WORKER USING REGION TOPOLOGY ======================================
# ============================================================================

def _log_tail(fp, n=35, prefix=""):
    try:
        with open(fp, 'r') as f:
            lines = f.readlines()[-n:]
        for ln in lines:
            logging.info("%s%s", f"[{prefix}] " if prefix else "", ln.rstrip())
    except Exception as e:
        logging.debug("Could not read tail of %s: %s", fp, e)

def run_codeml_in(run_dir, ctl_path, timeout):
    """Creates a directory for a single codeml run and executes it there."""
    os.makedirs(run_dir, exist_ok=True)
    # Proactively clean up any leftover files from a previous failed run.
    for pat in ('rst*', 'rub*', '2NG*', '2ML*', 'lnf', 'mlc'):
        for f in glob.glob(os.path.join(run_dir, pat)):
            try:
                os.remove(f)
            except OSError:
                pass

    cmd = [PAML_PATH, ctl_path]
    # For logging, create a fully reproducible, shell-quoted command string
    repro_cmd = f"{shlex.quote(os.path.abspath(PAML_PATH))} {shlex.quote(os.path.abspath(ctl_path))}"
    logging.info(f"REPRODUCE PAML: {repro_cmd}")
    run_command(cmd, run_dir, timeout=timeout, input_data="\n")

def codeml_worker(gene_info, region_tree_file, region_label):
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
        # Make run dirs unique and don't clean them up, per user request.
        temp_dir = tempfile.mkdtemp(prefix=f"paml_{gene_name}_{region_label}_", dir=temp_dir_base)

        region_taxa = Tree(region_tree_file, format=1).get_leaf_names()
        gene_taxa = read_taxa_from_phy(gene_info['path'])
        keep = [taxon for taxon in gene_taxa if taxon in set(region_taxa)]
        if len(keep) < 4:
            final_result.update({'status': 'uninformative_topology', 'reason': f'Fewer than four shared taxa (n={len(keep)})'})
            return final_result
        
        pruned_tree = os.path.join(temp_dir, f"{gene_name}_pruned.tree")
        prune_region_tree(region_tree_file, keep, pruned_tree)
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
            if PROCEED_ON_TERMINAL_ONLY:
                logging.warning(f"[{gene_name}] No pure internal branches in both clades; proceeding as PROCEED_ON_TERMINAL_ONLY is True (lower power).")
            else:
                final_result.update({'status': 'uninformative_topology', 'reason': 'No pure internal branches found for both direct and inverted groups.'})
                return final_result

        phy_abs = os.path.abspath(gene_info['path'])
        
        # --- PAML CACHE LOGIC ---
        os.makedirs(PAML_CACHE_DIR, exist_ok=True)
        exe_fp = _exe_fingerprint(PAML_PATH)
        gene_phy_sha = _canonical_phy_sha(phy_abs)
        h0_tree_str = _read_text(h0_tree)
        h1_tree_str = _read_text(h1_tree)
        taxa_used = t.get_leaf_names()

        # --- 1. Define all four attempts and their cache keys ---

        # Branch-model (bm) attempts
        ctl_bm_h0 = _ctl_string(phy_abs, h0_tree, "H0_bm.out", model=2, NSsites=0)
        ctl_bm_h1 = _ctl_string(phy_abs, h1_tree, "H1_bm.out", model=2, NSsites=0)
        h0_bm_key, _ = _hash_key_attempt(gene_phy_sha, h0_tree_str, taxa_used, ctl_bm_h0, exe_fp)
        h1_bm_key, _ = _hash_key_attempt(gene_phy_sha, h1_tree_str, taxa_used, ctl_bm_h1, exe_fp)

        # Clade-model (cmc) attempts
        ctl_cmc_h0 = _ctl_string(phy_abs, h0_tree, "H0_cmc.out", model=0, NSsites=22, ncatG=3)
        ctl_cmc_h1 = _ctl_string(phy_abs, h1_tree, "H1_cmc.out", model=3, NSsites=2, ncatG=3)
        h0_cmc_key, _ = _hash_key_attempt(gene_phy_sha, h0_tree_str, taxa_used, ctl_cmc_h0, exe_fp)
        h1_cmc_key, _ = _hash_key_attempt(gene_phy_sha, h1_tree_str, taxa_used, ctl_cmc_h1, exe_fp)

        # --- 2. Helper function to run a single codeml attempt if not cached ---
        def get_attempt_result(key_hex, tree_path, out_name, model_params, parser_func):
            """
            Returns a payload dict for a single codeml attempt, preferring the current cache key.
            On a miss, performs a legacy scan of the cache to find an equivalent run (by invariant inputs),
            then rehydrates it under the current key and copies ephemeral artifacts (notably the treefile)
            into the permanent cache.
        
            Payload shape: {"lnl": float, "params": {...}}; params may be {} if parser_func is None.
            """
            # --- quick helpers (local to keep this change self-contained) ---
            def _safe_json_load(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    return None
        
            def _copy_if_exists(src, dst_dir, dst_name=None):
                try:
                    if src and os.path.exists(src):
                        os.makedirs(dst_dir, exist_ok=True)
                        dst = os.path.join(dst_dir, (dst_name if dst_name else os.path.basename(src)))
                        shutil.copy(src, dst)
                        return dst
                except Exception as e:
                    logging.debug(f"Copy failed from {src} to {dst_dir}: {e}")
                return None
        
            def _parse_ctl_fields(ctl_path):
                """Extract minimal fields from a codeml .ctl file needed for legacy matching."""
                FINT = r'[-+]?\d+'
                FFLT = FLOAT_REGEX  # already defined globally
                rx = {
                    "seqfile": re.compile(r'^\s*seqfile\s*=\s*(.+?)\s*$', re.I|re.M),
                    "treefile": re.compile(r'^\s*treefile\s*=\s*(.+?)\s*$', re.I|re.M),
                    "model": re.compile(r'^\s*model\s*=\s*(' + FINT + r')\s*$', re.I|re.M),
                    "NSsites": re.compile(r'^\s*NSsites\s*=\s*(' + FINT + r'(?:\s+' + FINT + r')*)\s*$', re.I|re.M),
                    "ncatG": re.compile(r'^\s*ncatG\s*=\s*(' + FINT + r')\s*$', re.I|re.M),
                    "fix_blength": re.compile(r'^\s*fix_blength\s*=\s*(' + FINT + r')\s*$', re.I|re.M),
                }
                try:
                    s = _read_text(ctl_path)
                except Exception:
                    return None
        
                def _pick_int(key, default=None):
                    m = rx[key].search(s)
                    return int(m.group(1)) if m else default
        
                # NSsites may be a list, but for these models we pass a single value; normalize to int
                ns_m = rx["NSsites"].search(s)
                ns_val = None
                if ns_m:
                    toks = ns_m.group(1).strip().split()
                    if toks:
                        try:
                            ns_val = int(toks[0])
                        except ValueError:
                            ns_val = None
        
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
                try:
                    return _sha256_file(p)
                except Exception:
                    return None
        
            def _legacy_find_equivalent(out_name, expect_params, expect_gene_phy_sha, expect_tree_sha):
                """
                Scan existing cache for an attempt with the same out_name whose ctl
                refers to the same gene (by PHY sha), same tree (by treefile sha), and same model knobs.
                Returns (payload, legacy_key_dir, ctl_path, treefile_path) or (None, None, None, None)
                """
                # Fast path: if the current cache dir already has artifacts/ctl for out_name, just verify those.
                cur_dir = _fanout_dir(PAML_CACHE_DIR, key_hex)
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
                            payload = cache_read_json(PAML_CACHE_DIR, key_hex, "attempt.json")
                            if payload:
                                return payload, cur_dir, candidate, fields.get("treefile")
        
                # Full scan (bounded to cache structure)
                for lvl1 in (os.listdir(PAML_CACHE_DIR) if os.path.isdir(PAML_CACHE_DIR) else []):
                    p1 = os.path.join(PAML_CACHE_DIR, lvl1)
                    if not os.path.isdir(p1) or len(lvl1) != 2:  # fanout sanity
                        continue
                    for lvl2 in (os.listdir(p1) if os.path.isdir(p1) else []):
                        p2 = os.path.join(p1, lvl2)
                        if not os.path.isdir(p2) or len(lvl2) != 2:
                            continue
                        # Iterate keys under this fanout bucket
                        for keydir in (os.listdir(p2) if os.path.isdir(p2) else []):
                            kd = os.path.join(p2, keydir)
                            if not os.path.isdir(kd):
                                continue
                            att_json = os.path.join(kd, "attempt.json")
                            if not os.path.exists(att_json):
                                continue
                            ctl_candidate = os.path.join(kd, "artifacts", f"{out_name}.ctl")
                            out_candidate = os.path.join(kd, "artifacts", out_name)
                            if not os.path.exists(ctl_candidate) or not os.path.exists(out_candidate):
                                continue
                            fields = _parse_ctl_fields(ctl_candidate)
                            if not fields:
                                continue
                            # Match model knobs
                            if fields["model"] != expect_params.get("model"):
                                continue
                            if fields["NSsites"] != expect_params.get("NSsites"):
                                continue
                            if (fields["ncatG"] or None) != expect_params.get("ncatG"):
                                continue
                            if (fields["fix_blength"] or 0) != expect_params.get("fix_blength", 0):
                                continue
                            # Match input content by sha
                            seq_sha = _sha_file_safe(fields["seqfile"]) if fields.get("seqfile") else None
                            if seq_sha != expect_gene_phy_sha:
                                continue
                            tree_sha = _sha_file_safe(fields["treefile"]) if fields.get("treefile") else None
                            if tree_sha != expect_tree_sha:
                                continue
                            payload = _safe_json_load(att_json)
                            if payload and isinstance(payload, dict) and "lnl" in payload:
                                return payload, kd, ctl_candidate, fields.get("treefile")
                return None, None, None, None
        
            def _rehydrate_under_new_key(new_key_hex, payload, legacy_dir, legacy_ctl, legacy_tree):
                """
                Writes attempt.json into the new key directory and copies artifacts.
                Also copies the treefile used (which may be in a temp dir) into artifacts → permanent.
                """
                target_dir = _fanout_dir(PAML_CACHE_DIR, new_key_hex)
                with _with_lock(target_dir):
                    cache_write_json(PAML_CACHE_DIR, new_key_hex, "attempt.json", payload)
                    art_dst = os.path.join(target_dir, "artifacts")
                    os.makedirs(art_dst, exist_ok=True)
        
                    # Copy known artifacts from legacy (if present)
                    _copy_if_exists(os.path.join(legacy_dir, "artifacts", out_name), art_dst, out_name)
                    _copy_if_exists(os.path.join(legacy_dir, "artifacts", f"{out_name}.ctl"), art_dst, f"{out_name}.ctl")
                    _copy_if_exists(os.path.join(legacy_dir, "artifacts", "mlc"), art_dst, "mlc")
        
                    # Copy the legacy treefile into artifacts for permanence
                    if legacy_tree and os.path.exists(legacy_tree):
                        # Name it predictably so we can find it later regardless of temp paths
                        _copy_if_exists(legacy_tree, art_dst, f"{out_name}.tree")
        
            # --- 1) Try the current key directly ---
            payload = cache_read_json(PAML_CACHE_DIR, key_hex, "attempt.json")
            if payload:
                logging.info(f"[{gene_name}|{region_label}] Using cached ATTEMPT (current key): {out_name}")
                # Ensure the tree used is persisted (copy if missing)
                art_dir = os.path.join(_fanout_dir(PAML_CACHE_DIR, key_hex), "artifacts")
                tree_copy = os.path.join(art_dir, f"{out_name}.tree")
                if not os.path.exists(tree_copy):
                    _copy_if_exists(tree_path, art_dir, f"{out_name}.tree")

                # Heal params when possible by parsing a persisted raw artifact.
                # Prefer the run-specific out file; fall back to 'mlc' if needed.
                try:
                    if parser_func:
                        need_keys = ("cmc_p0","cmc_p1","cmc_p2","cmc_omega0","cmc_omega2_direct","cmc_omega2_inverted")
                        params = payload.get("params", {}) or {}
                        def _bad(x):
                            return x is None or (isinstance(x, float) and np.isnan(x))
                        if any(_bad(params.get(k)) for k in need_keys):
                            candidates = [os.path.join(art_dir, out_name), os.path.join(art_dir, "mlc")]
                            healed = {}
                            for raw_path in candidates:
                                if os.path.exists(raw_path):
                                    try:
                                        healed = parser_func(raw_path) or {}
                                    except Exception as _e:
                                        logging.debug(f"Heal-attempt parse failed for {raw_path}: {_e}")
                                    if healed:
                                        break
                            if healed:
                                for k, v in healed.items():
                                    if _bad(params.get(k)) and v is not None and not (isinstance(v, float) and np.isnan(v)):
                                        params[k] = v
                                payload["params"] = params
                                cache_write_json(PAML_CACHE_DIR, key_hex, "attempt.json", payload)
                                logging.info(f"[{gene_name}|{region_label}] Healed attempt.json params from artifacts for {out_name}")
                except Exception as _e:
                    logging.debug(f"Heal-attempt skipped for {out_name}: {_e}")

                return payload

            # --- 2) Legacy fallback: find an equivalent attempt and rehydrate it ---
            # Prepare invariants for matching
            expect_gene_phy_sha = gene_phy_sha  # from closure
            expect_tree_sha = _sha_file_safe(tree_path)
            # Normalize model knobs we care about for matching
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
        
            # --- 3) No cache found → run codeml and cache fresh (and persist the tree) ---
            run_dir = os.path.join(temp_dir, out_name.replace(".out", ""))
            ctl_file = os.path.join(run_dir, f"{gene_name}_{out_name}.ctl")
            out_file = os.path.join(run_dir, f"{gene_name}_{out_name}")
        
            # Fix initial params and generate ctl
            params = {**model_params, 'init_kappa': 2.0, 'init_omega': 0.5, 'fix_blength': model_params.get('fix_blength', 0)}
            generate_paml_ctl(ctl_file, phy_abs, tree_path, out_file, **params)
            run_codeml_in(run_dir, ctl_file, PAML_TIMEOUT)
            _log_tail(out_file, 25, prefix=f"[{gene_name}|{region_label}] {out_name} out (computed)")
        
            lnl = parse_paml_lnl(out_file)
            parsed = parser_func(out_file) if parser_func else {}
        
            payload = {"lnl": float(lnl), "params": parsed}
        
            cache_dir = _fanout_dir(PAML_CACHE_DIR, key_hex)
            with _with_lock(cache_dir):
                cache_write_json(PAML_CACHE_DIR, key_hex, "attempt.json", payload)
                artifact_dir = os.path.join(cache_dir, "artifacts")
                os.makedirs(artifact_dir, exist_ok=True)
                # Core artifacts
                _copy_if_exists(out_file, artifact_dir, out_name)
                _copy_if_exists(ctl_file, artifact_dir, f"{out_name}.ctl")
                mlc_path = os.path.join(run_dir, "mlc")
                _copy_if_exists(mlc_path, artifact_dir, "mlc")
                # Persist the exact tree used (temp paths can vanish)
                _copy_if_exists(tree_path, artifact_dir, f"{out_name}.tree")
        
            logging.info(f"[{gene_name}|{region_label}] Cached attempt {out_name} to {cache_dir}")
            return payload


        # --- 3. Process Branch-Model LRT ---
        bm_result = {}
        if RUN_BRANCH_MODEL_TEST:
            pair_key_bm, pair_key_dict_bm = _hash_key_pair(h0_bm_key, h1_bm_key, "branch_model", 1, exe_fp)
            pair_payload_bm = cache_read_json(PAML_CACHE_DIR, pair_key_bm, "pair.json")
            if pair_payload_bm:
                logging.info(f"[{gene_name}|{region_label}] Using cached PAIR result for branch_model")
                bm_result = pair_payload_bm["result"]
            else:
                # Run H0 and H1 codeml attempts concurrently to utilize multiple cores per gene×region.
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
                    with _with_lock(_fanout_dir(PAML_CACHE_DIR, pair_key_bm)):
                        cache_write_json(PAML_CACHE_DIR, pair_key_bm, "pair.json", {"key": pair_key_dict_bm, "result": bm_result})
                    logging.info(f"[{gene_name}|{region_label}] Cached LRT pair 'branch_model' (df=1)")
                else:
                    # Cache an invalid or non-improving LRT result to prevent re-runs. Represent failure with NaN statistics.
                    bm_result = {
                        "bm_p_value": np.nan,
                        "bm_lrt_stat": np.nan,
                        "bm_lnl_h0": lnl0,
                        "bm_lnl_h1": lnl1,
                        "bm_h0_key": h0_bm_key,
                        "bm_h1_key": h1_bm_key
                    }
                    with _with_lock(_fanout_dir(PAML_CACHE_DIR, pair_key_bm)):
                        cache_write_json(PAML_CACHE_DIR, pair_key_bm, "pair.json", {"key": pair_key_dict_bm, "result": bm_result})
                    logging.info(f"[{gene_name}|{region_label}] Cached LRT pair 'branch_model' (invalid or non-improvement)")
        else:
            logging.info(f"[{gene_name}|{region_label}] Skipping branch-model test as per configuration.")
            bm_result = {"bm_p_value": np.nan, "bm_lrt_stat": np.nan}
        
        # --- 4. Process Clade-Model LRT ---
        cmc_result = {}
        if RUN_CLADE_MODEL_TEST:
            pair_key_cmc, pair_key_dict_cmc = _hash_key_pair(h0_cmc_key, h1_cmc_key, "clade_model_c", 1, exe_fp)
            pair_payload_cmc = cache_read_json(PAML_CACHE_DIR, pair_key_cmc, "pair.json")
            if pair_payload_cmc:
                logging.info(f"[{gene_name}|{region_label}] Using cached PAIR result for clade_model_c")
                cmc_result = dict(pair_payload_cmc["result"])

                # Back-fill cmc_* parameters from raw artifacts when any are missing.
                h1_key = cmc_result.get("cmc_h1_key") or pair_payload_cmc.get("key", {}).get("h1_attempt_key")
                def _bad(x): return (x is None) or (isinstance(x, float) and np.isnan(x))
                healed = {}

                if h1_key:
                    art_dir_h1 = os.path.join(_fanout_dir(PAML_CACHE_DIR, h1_key), "artifacts")
                    candidates = [os.path.join(art_dir_h1, "H1_cmc.out"), os.path.join(art_dir_h1, "mlc")]
                    for raw_h1 in candidates:
                        if os.path.exists(raw_h1):
                            try:
                                healed = parse_h1_cmc_paml_output(raw_h1) or {}
                            except Exception as _e:
                                logging.debug(f"[{gene_name}|{region_label}] parse_h1_cmc_paml_output failed: {_e}")
                            if healed:
                                break

                if healed:
                    changed = False
                    for k, v in healed.items():
                        if k.startswith("cmc_") and (_bad(cmc_result.get(k)) or (k not in cmc_result)):
                            cmc_result[k] = v
                            changed = True
                    if changed:
                        with _with_lock(_fanout_dir(PAML_CACHE_DIR, pair_key_cmc)):
                            cache_write_json(PAML_CACHE_DIR, pair_key_cmc, "pair.json",
                                             {"key": pair_payload_cmc["key"], "result": cmc_result})
                        logging.info(f"[{gene_name}|{region_label}] Back-filled cmc_* in cached pair.json")

                    # Keep the H1 attempt.json consistent with parsed parameters
                    try:
                        if h1_key:
                            att = cache_read_json(PAML_CACHE_DIR, h1_key, "attempt.json")
                            if isinstance(att, dict):
                                aparams = att.get("params", {}) or {}
                                a_changed = False
                                for k, v in healed.items():
                                    if k.startswith("cmc_") and (_bad(aparams.get(k)) or (k not in aparams)):
                                        aparams[k] = v
                                        a_changed = True
                                if a_changed:
                                    att["params"] = aparams
                                    cache_write_json(PAML_CACHE_DIR, h1_key, "attempt.json", att)
                                    logging.info(f"[{gene_name}|{region_label}] Back-filled cmc_* in H1 attempt.json")
                    except Exception as _e:
                        logging.debug(f"[{gene_name}|{region_label}] attempt.json back-fill skipped: {_e}")

            else:
                with ThreadPoolExecutor(max_workers=2) as ex:
                    fut_h0 = ex.submit(get_attempt_result, h0_cmc_key, h0_tree, "H0_cmc.out",
                                       {"model": 0, "NSsites": 22, "ncatG": 3}, None)
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
                    with _with_lock(_fanout_dir(PAML_CACHE_DIR, pair_key_cmc)):
                        cache_write_json(PAML_CACHE_DIR, pair_key_cmc, "pair.json",
                                         {"key": pair_key_dict_cmc, "result": cmc_result})
                    logging.info(f"[{gene_name}|{region_label}] Cached LRT pair 'clade_model_c' (df=1)")
                else:
                    # Cache an invalid or non-improving LRT result to prevent re-runs. Represent failure with NaN statistics.
                    cmc_result = {
                        "cmc_p_value": np.nan,
                        "cmc_lrt_stat": np.nan,
                        "cmc_lnl_h0": lnl0,
                        "cmc_lnl_h1": lnl1,
                        "cmc_h0_key": h0_cmc_key,
                        "cmc_h1_key": h1_cmc_key
                    }
                    with _with_lock(_fanout_dir(PAML_CACHE_DIR, pair_key_cmc)):
                        cache_write_json(PAML_CACHE_DIR, pair_key_cmc, "pair.json", {"key": pair_key_dict_cmc, "result": cmc_result})
                    logging.info(f"[{gene_name}|{region_label}] Cached LRT pair 'clade_model_c' (invalid or non-improvement)")

        else:
            logging.info(f"[{gene_name}|{region_label}] Skipping clade-model test as per configuration.")
            cmc_result = {"cmc_p_value": np.nan, "cmc_lrt_stat": np.nan}

        # --- 5. Combine results ---
        # A run is OK if the test was skipped OR if it ran and produced a p-value.
        bm_ok = not RUN_BRANCH_MODEL_TEST or not np.isnan(bm_result.get("bm_p_value", np.nan))
        cmc_ok = not RUN_CLADE_MODEL_TEST or not np.isnan(cmc_result.get("cmc_p_value", np.nan))

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

        # --- Post-computation/cache-hit processing ---
        if KEEP_PAML_OUT and final_result.get('status') == 'success':
            try:
                safe_region = re.sub(r'[^A-Za-z0-9_.-]+', '_', region_label)
                safe_gene   = re.sub(r'[^A-Za-z0-9_.-]+', '_', gene_name)
                dest_dir = os.path.join(PAML_OUT_DIR, f"{safe_gene}__{safe_region}")
                os.makedirs(dest_dir, exist_ok=True)
                
                for key in ["bm_h0_key", "bm_h1_key", "cmc_h0_key", "cmc_h1_key"]:
                    if final_result.get(key):
                        artifact_dir = os.path.join(_fanout_dir(PAML_CACHE_DIR, final_result[key]), "artifacts")
                        if os.path.isdir(artifact_dir):
                            for f in os.listdir(artifact_dir):
                                shutil.copy(os.path.join(artifact_dir, f), dest_dir)
                
                if os.path.exists(h1_tree): shutil.copy(h1_tree, dest_dir)
                if os.path.exists(h0_tree): shutil.copy(h0_tree, dest_dir)
                if os.path.exists(pruned_tree): shutil.copy(pruned_tree, dest_dir)

            except Exception as e:
                logging.error(f"[{gene_name}|{region_label}] Failed to copy artifacts for KEEP_PAML_OUT: {e}")

        if final_result['status'] == 'success':
            try:
                # The figure visualizes the branch-model results.
                bm_params = {
                    'omega_direct': final_result.get('bm_omega_direct'),
                    'omega_inverted': final_result.get('bm_omega_inverted'),
                    'omega_background': final_result.get('bm_omega_background'),
                }
                generate_omega_result_figure(gene_name, region_label, status_tree, bm_params)
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
        # Per user request, do not delete the temporary directories.
        if temp_dir:
            logging.info(f"[{gene_name}|{region_label}] PAML run directory available at: {temp_dir}")

# ==============================================================================
# === SYSTEM MONITORING THREAD =================================================
# ==============================================================================
_prev_cpu_times = None
_prev_cpu_time_ts = None

def _get_procfs_cpu_times():
    """Returns (user, nice, system, idle) from /proc/stat."""
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        parts = line.split()
        # user, nice, system, idle, iowait, irq, softirq
        return tuple(map(int, parts[1:5]))
    except (IOError, IndexError, ValueError):
        return (0, 0, 0, 0)

def _get_cpu_usage_procfs():
    """Computes system-wide CPU usage % from /proc/stat deltas."""
    global _prev_cpu_times, _prev_cpu_time_ts

    if _prev_cpu_times is None:
        _prev_cpu_times = _get_procfs_cpu_times()
        _prev_cpu_time_ts = time.time()
        time.sleep(1) # sleep to get a delta on the next call

    now = time.time()
    current_times = _get_procfs_cpu_times()

    delta_times = tuple(c - p for c, p in zip(current_times, _prev_cpu_times))
    delta_ts = now - _prev_cpu_time_ts

    _prev_cpu_times = current_times
    _prev_cpu_time_ts = now

    if delta_ts == 0:
        return 0.0

    total_time = sum(delta_times)
    idle_time = delta_times[3]

    usage_pct = 100.0 * (total_time - idle_time) / total_time if total_time else 0.0
    return usage_pct

def _get_mem_info_procfs():
    """Returns (MemTotal, MemAvailable) in KB from /proc/meminfo."""
    mem_total, mem_avail = 0, 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_avail = int(line.split()[1])
                    break # Available is usually after Total
    except (IOError, IndexError, ValueError):
        pass
    return mem_total, mem_avail

def _get_load_avg_procfs():
    """Returns the 1-minute load average from /proc/loadavg."""
    try:
        with open("/proc/loadavg") as f:
            return float(f.readline().split()[0])
    except (IOError, IndexError, ValueError):
        return 0.0

def _get_process_counts_procfs():
    """Counts running iqtree3 and codeml processes by scanning /proc."""
    iqtree_count = 0
    codeml_count = 0
    try:
        for pid in os.listdir('/proc'):
            if not pid.isdigit():
                continue
            try:
                with open(f'/proc/{pid}/cmdline', 'rb') as f:
                    cmdline = f.read().split(b'\x00')
                if not cmdline:
                    continue

                exe_path = os.path.basename(cmdline[0].decode('utf-8', 'ignore'))
                if 'iqtree3' in exe_path:
                    iqtree_count += 1
                elif 'codeml' in exe_path:
                    codeml_count += 1
            except (IOError, UnicodeDecodeError):
                continue
    except IOError:
        pass
    return iqtree_count, codeml_count

def monitor_thread(status_dict, stop_event, interval=12):
    """A thread that periodically logs system and pipeline utilization."""
    logging.info("MONITOR: Starting utilization monitor thread.")

    # Prime the CPU usage calculator
    _get_cpu_usage_procfs()

    while not stop_event.is_set():
        try:
            cpu_pct = _get_cpu_usage_procfs()
            mem_total_kb, mem_avail_kb = _get_mem_info_procfs()
            mem_pct = 100.0 * (mem_total_kb - mem_avail_kb) / mem_total_kb if mem_total_kb else 0.0
            load_avg = _get_load_avg_procfs()
            iqtree_pids, codeml_pids = _get_process_counts_procfs()

            # --- Queue Stats ---
            regions_done = status_dict.get('regions_done', 0)
            regions_total = status_dict.get('regions_total', 0)
            paml_done = status_dict.get('paml_done', 0)
            paml_running = status_dict.get('paml_running', 0)
            paml_total = status_dict.get('paml_total', 0)

            msg = (
                f"MONITOR: CPU: {cpu_pct:.1f}%, Mem: {mem_pct:.1f}%, Load: {load_avg:.2f}, "
                f"PIDs(iq/paml): {iqtree_pids}/{codeml_pids} | "
                f"Regions: {regions_done}/{regions_total} | "
                f"PAML: {paml_done}/{paml_total} (running: {paml_running}) | "
                f"ETA: {status_dict.get('eta_str', 'N/A')}"
            )
            logging.info(msg)

        except Exception as e:
            logging.error(f"MONITOR: Error in monitor thread: {e}\n{traceback.format_exc()}")

        wait_time = 0
        while wait_time < interval and not stop_event.is_set():
            time.sleep(1)
            wait_time += 1

    logging.info("MONITOR: Stopping utilization monitor thread.")


# ==============================================================================
# === MAIN EXECUTION AND REPORTING =============================================
# ==============================================================================

def submit_with_cap(exec, fn, args, inflight, cap):
    """Submits a task to the executor and manages the inflight queue to enforce a cap."""
    fut = exec.submit(fn, *args)
    inflight.append(fut)
    
    # If the queue is full, wait for the next future to complete
    if len(inflight) >= cap:
        done = next(as_completed(inflight))
        inflight.remove(done)
        return [done]
    return []

def run_overlapped(region_infos, region_gene_map, log_q, status_dict):
    """
    Runs the full pipeline with overlapped IQ-TREE and PAML execution,
    using ProcessPoolExecutors and a cap on in-flight PAML jobs for back-pressure.
    """
    all_results = []
    inflight = deque()
    cap = PAML_WORKERS * 4
    completed_count = 0

    status_dict['regions_total'] = len(region_infos)
    status_dict['regions_done'] = 0
    total_paml_jobs = sum(len(genes) for r_label, genes in region_gene_map.items() if r_label in {r['label'] for r in region_infos})
    status_dict['paml_total'] = total_paml_jobs
    status_dict['paml_done'] = 0
    paml_start_time = None

    # Ensure workers use the 'spawn' context and our queue logger
    mpctx = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=PAML_WORKERS, mp_context=mpctx,
                             initializer=worker_logging_init, initargs=(log_q,)) as paml_exec, \
         ProcessPoolExecutor(max_workers=REGION_WORKERS, mp_context=mpctx,
                             initializer=worker_logging_init, initargs=(log_q,)) as region_exec:

        iqtree_threads = max(1, CPU_COUNT // REGION_WORKERS)
        logging.info(f"Submitting {len(region_infos)} region tasks to pool (using {iqtree_threads} threads per job)...")
        region_futs = {region_exec.submit(region_worker, r, iqtree_threads) for r in region_infos}

        region_pbar = tqdm(as_completed(region_futs), total=len(region_futs), desc="Processing regions")
        for region_future in region_pbar:
            status_dict['regions_done'] += 1
            try:
                label, tree, reason = region_future.result()
            except Exception as e:
                logging.error(f"A region task failed with an exception: {e}")
                continue

            if tree is None:
                logging.warning(f"Region {label} skipped: {reason}")
                # This region failed, so its PAML jobs will never run.
                # We need to adjust the total PAML job count for an accurate ETA.
                genes_for_failed_region = region_gene_map.get(label, [])
                total_paml_jobs -= len(genes_for_failed_region)
                status_dict['paml_total'] = total_paml_jobs
                continue
            
            genes_for_region = region_gene_map.get(label, [])
            if not genes_for_region:
                continue

            if paml_start_time is None:
                paml_start_time = time.time()

            logging.info(f"Region {label} complete. Submitting {len(genes_for_region)} PAML jobs.")
            for gene_info in genes_for_region:
                flushed = submit_with_cap(
                    paml_exec, codeml_worker, (gene_info, tree, label), inflight, cap)
                status_dict['paml_running'] = len(inflight)
                for paml_future in flushed:
                    try:
                        res = paml_future.result()
                        all_results.append(res)
                        completed_count += 1
                        status_dict['paml_done'] = completed_count
                        if paml_start_time and completed_count > 2:
                            elapsed = time.time() - paml_start_time
                            rate = completed_count / elapsed
                            if rate > 0:
                                remaining = total_paml_jobs - completed_count
                                eta_s = remaining / rate
                                status_dict['eta_str'] = f"{int(eta_s // 60)}m{int(eta_s % 60)}s"

                        if (completed_count % 25 == 0) or (res.get('status') != 'success'):
                            logging.info(f"Completed {completed_count}/{total_paml_jobs}: {res.get('gene')} in {res.get('region')} -> {res.get('status')}")
                        if completed_count % CHECKPOINT_EVERY == 0:
                            logging.info(f"--- Checkpointing {len(all_results)} results to {CHECKPOINT_FILE} ---")
                            pd.DataFrame(all_results).to_csv(CHECKPOINT_FILE, sep="\t", index=False, float_format='%.6g')
                    except Exception as e:
                        logging.error(f"A PAML job failed with an exception: {e}")
                status_dict['paml_running'] = len(inflight)

        # Drain any remaining PAML jobs
        logging.info(f"All regions processed. Draining {len(inflight)} remaining PAML jobs...")
        paml_pbar = tqdm(as_completed(list(inflight)), total=len(inflight), desc="Finalizing PAML jobs")
        for paml_future in paml_pbar:
            status_dict['paml_running'] = len(inflight) - paml_pbar.n - 1
            try:
                res = paml_future.result()
                all_results.append(res)
                completed_count += 1
                status_dict['paml_done'] = completed_count
                if paml_start_time and completed_count > 2:
                    elapsed = time.time() - paml_start_time
                    rate = completed_count / elapsed
                    if rate > 0:
                        remaining = total_paml_jobs - completed_count
                        eta_s = remaining / rate
                        status_dict['eta_str'] = f"{int(eta_s // 60)}m{int(eta_s % 60)}s"

                if (completed_count % 25 == 0) or (res.get('status') != 'success'):
                    logging.info(f"Completed {completed_count}/{total_paml_jobs}: {res.get('gene')} in {res.get('region')} -> {res.get('status')}")
                if completed_count % CHECKPOINT_EVERY == 0:
                    logging.info(f"--- Checkpointing {len(all_results)} results to {CHECKPOINT_FILE} ---")
                    pd.DataFrame(all_results).to_csv(CHECKPOINT_FILE, sep="\t", index=False, float_format='%.6g')
            except Exception as e:
                logging.error(f"A PAML job failed with an exception during drain: {e}")
        status_dict['paml_running'] = 0

    # Final checkpoint save
    if all_results:
        logging.info(f"--- Final checkpoint of {len(all_results)} results to {CHECKPOINT_FILE} ---")
        pd.DataFrame(all_results).to_csv(CHECKPOINT_FILE, sep="\t", index=False, float_format='%.6g')
        
    return all_results


def main():
    """Run region-first pipeline: IQ-TREE on regions, codeml on genes."""
    log_q, listener = start_logging()
    # Configure logging for the main process to also use the queue
    root = logging.getLogger()
    root.handlers[:] = [QueueHandler(log_q)]
    root.setLevel(logging.INFO)

    status_dict = {}
    stop_event = threading.Event()
    mon_thread = threading.Thread(target=monitor_thread, args=(status_dict, stop_event))
    mon_thread.daemon = True
    mon_thread.start()

    try:
        logging.info("--- Starting Region→Gene Differential Selection Pipeline ---")

        if not (os.path.exists(IQTREE_PATH) and os.access(IQTREE_PATH, os.X_OK)):
            logging.critical(f"FATAL: IQ-TREE not found or not executable at '{IQTREE_PATH}'")
            sys.exit(1)
        if not (os.path.exists(PAML_PATH) and os.access(PAML_PATH, os.X_OK)):
            logging.critical(f"FATAL: PAML codeml not found or not executable at '{PAML_PATH}'")
            sys.exit(1)

        logging.info("Checking external tool versions...")
        iqtree_ver = subprocess.run([IQTREE_PATH, '--version'], capture_output=True, text=True, check=True).stdout.strip().split('\n')[0]
        logging.info(f"IQ-TREE version: {iqtree_ver}")
        logging.info(f"PAML executable: {PAML_PATH}")
        logging.info(f"CPUs: {CPU_COUNT} | REGION_WORKERS={REGION_WORKERS} | PAML_WORKERS={PAML_WORKERS}")
        if PAML_WORKERS > CPU_COUNT:
            logging.warning(
                f"PAML_WORKERS ({PAML_WORKERS}) exceeds available CPUs ({CPU_COUNT}); performance may suffer"
            )

        os.makedirs(FIGURE_DIR, exist_ok=True)
        os.makedirs(ANNOTATED_FIGURE_DIR, exist_ok=True)
        os.makedirs(REGION_TREE_DIR, exist_ok=True)

        logging.info("Searching for alignment files...")
        region_files = glob.glob('combined_inversion_*.phy')
        gene_files = [f for f in glob.glob('combined_*.phy') if 'inversion' not in os.path.basename(f)]
        logging.info(f"Found {len(region_files)} region alignments and {len(gene_files)} gene alignments")

        if not region_files:
            logging.critical("FATAL: No region alignment files found.")
            sys.exit(1)
        if not gene_files:
            logging.critical("FATAL: No gene alignment files found.")
            sys.exit(1)

        logging.info("Loading gene metadata...")
        metadata = load_gene_metadata()
        logging.info(f"Loaded metadata for {len(metadata)} genes")

        logging.info("Parsing region and gene filenames...")
        region_infos, bad_regions = [], []
        for f in region_files:
            try:
                region_infos.append(parse_region_filename(f))
            except Exception as e:
                bad_regions.append((f, str(e)))
        if bad_regions:
            logging.warning(
                f"Skipping {len(bad_regions)} region files with bad names: " +
                "; ".join(os.path.basename(b) for b, _ in bad_regions))
        # Whitelist filter: retain only regions explicitly listed in ALLOWED_REGIONS.
        if len(ALLOWED_REGIONS) > 0:
            before = len(region_infos)
            region_infos = [
                r for r in region_infos
                #if (r['chrom'], r['start'], r['end']) in ALLOWED_REGIONS
            ]
            dropped = before - len(region_infos)
            if dropped:
                logging.info(f"Whitelist active: kept {len(region_infos)} region(s); dropped {dropped} non-whitelisted region(s).")
            present = {(r['chrom'], r['start'], r['end']) for r in region_infos}
            missing = [t for t in ALLOWED_REGIONS if t not in present]
            if missing:
                logging.warning(f"Whitelist regions not found among available files: {sorted(missing)}")

        gene_infos, bad_genes = [], []
        for f in gene_files:
            try:
                gene_infos.append(parse_gene_filename(f, metadata))
            except Exception as e:
                bad_genes.append((f, str(e)))
        if bad_genes:
            logging.warning(
                f"Skipping {len(bad_genes)} gene files with missing/ambiguous coords or bad names. "
                f"Example: {os.path.basename(bad_genes[0][0])} -> {bad_genes[0][1]}")
        logging.info("Mapping genes to overlapping regions...")
        region_gene_map = build_region_gene_map(region_infos, gene_infos)
        for label, genes in region_gene_map.items():
            logging.info(f"Region {label} overlaps {len(genes)} genes")

        all_results = run_overlapped(region_infos, region_gene_map, log_q, status_dict)
        results_df = pd.DataFrame(all_results)

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

        # Handle the no-task / empty-results case safely
        if results_df.empty:
            results_df = results_df[ordered_columns]
            results_df.to_csv(RESULTS_TSV, sep='\t', index=False, float_format='%.6g')
            logging.info(f"All results saved to: {RESULTS_TSV}")
            logging.warning("No results produced (no valid region trees or gene×region tasks).")
            logging.info("\n\n" + "="*75)
            logging.info("--- FINAL PIPELINE REPORT ---")
            logging.info(f"Total tests: {len(results_df)}")
            logging.info("="*75 + "\n")
            logging.info("No significant tests.")
            logging.info("\nPipeline finished.")
            return

        successful = results_df[results_df['status'] == 'success'].copy()
        if not successful.empty:
            # FDR for branch-model test
            bm_pvals = successful['bm_p_value'].dropna()
            if not bm_pvals.empty:
                _, qvals = fdrcorrection(bm_pvals, alpha=FDR_ALPHA, method='indep')
                results_df['bm_q_value'] = results_df.index.map(pd.Series(qvals, index=bm_pvals.index))
                logging.info(f"Applied FDR correction to {len(bm_pvals)} branch-model tests.")

            # FDR for clade-model test
            cmc_pvals = successful['cmc_p_value'].dropna()
            if not cmc_pvals.empty:
                _, qvals = fdrcorrection(cmc_pvals, alpha=FDR_ALPHA, method='indep')
                results_df['cmc_q_value'] = results_df.index.map(pd.Series(qvals, index=cmc_pvals.index))
                logging.info(f"Applied FDR correction to {len(cmc_pvals)} clade-model tests.")

        results_df = results_df[ordered_columns]
        results_df.to_csv(RESULTS_TSV, sep='\t', index=False, float_format='%.6g')
        logging.info(f"All results saved to: {RESULTS_TSV}")

        counts = results_df['status'].value_counts().to_dict()
        logging.info("\n\n" + "="*75)
        logging.info("--- FINAL PIPELINE REPORT ---")
        logging.info(f"Total tests: {len(results_df)}")
        for status, count in counts.items():
            logging.info(f"  - {status}: {count}")
        logging.info("="*75 + "\n")

        sig = results_df[(results_df['status'] == 'success') & ((results_df['bm_q_value'] < FDR_ALPHA) | (results_df['cmc_q_value'] < FDR_ALPHA))]
        if not sig.empty:
            logging.info(f"Significant gene×region tests (q < {FDR_ALPHA}):")
            # Sort by the minimum q-value of the two tests
            min_q = sig[['bm_q_value', 'cmc_q_value']].min(axis=1)
            for idx, row in sig.loc[min_q.sort_values().index].iterrows():
                logging.info(f"{row['region']} - {row['gene']}: bm_q={row['bm_q_value']:.4g}, cmc_q={row['cmc_q_value']:.4g}")
        else:
            logging.info("No significant tests.")

        logging.info("\nPipeline finished.")
    finally:
        stop_event.set()
        if 'mon_thread' in locals() and mon_thread.is_alive():
            mon_thread.join(timeout=2.0)
        listener.stop()

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
