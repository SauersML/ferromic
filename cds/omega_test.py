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
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque

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

# === PAML CACHE CONFIG (ADD THIS) ============================================
import hashlib, json, time, random

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
PAML_TIMEOUT   = int(os.environ.get("PAML_TIMEOUT", "3600"))     # 1h default

# Prevent hidden multi-threading from MKL/OpenBLAS in child processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Optional: speed up H0 runs by fixing branch lengths to the input tree's values
PAML_FIX_BLENGTH_H0 = bool(int(os.environ.get("PAML_FIX_BLENGTH_H0", "0")))

# ==============================================================================
# === GENERIC HELPER FUNCTIONS (UNCHANGED CORE LOGIC) ==========================
# ==============================================================================

def run_command(command_list, work_dir, timeout=None, env=None):
    try:
        subprocess.run(
            command_list, cwd=work_dir, check=True,
            capture_output=True, text=True, shell=False,
            timeout=timeout, env=env
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
                m = re.search(r'kappa \(ts/tv\) = \s*([\d\.]+)', line)
                if m: params['kappa'] = float(m.group(1))
            elif re.search(r'\bw\b.*\(dN/dS\)', line) or re.search(r'\bw\b for branch', line):
                m = re.search(r'=\s*([\d\.]+)|type 0:\s*([\d\.]+)', line)
                if m:
                    params['omega_background'] = float(m.group(1) or m.group(2))
    return params

def _ctl_string(seqfile, treefile, outfile, model_num, init_kappa, init_omega, fix_blength, base_opts: dict):
    # Mirrors generate_paml_ctl, but returns a normalized string for hashing
    return (
f"""seqfile = {seqfile}
treefile = {treefile}
outfile = {outfile}
noisy = 0
verbose = 0
runmode = 0
seqtype = 1
CodonFreq = 2
model = {model_num}
NSsites = 0
icode = 0
cleandata = 0
fix_kappa = 0
kappa = {init_kappa if init_kappa is not None else 2.0}
fix_omega = 0
omega = {init_omega if init_omega is not None else 0.5}
fix_blength = {fix_blength}
method = {base_opts.get('method', 0)}
getSE = 0
RateAncestor = 0
""").strip()

def _hash_key_meta(gene_phy_sha, h0_tree_str, h1_tree_str, taxa_used_list, seeds_list, exe_fp, base_opts: dict):
    key_dict = {
        "schema": CACHE_SCHEMA_VERSION,
        "gene_phy_sha": gene_phy_sha,
        "h0_tree_sha": _sha256_bytes(h0_tree_str.encode("utf-8")),
        "h1_tree_sha": _sha256_bytes(h1_tree_str.encode("utf-8")),
        "taxa_used": sorted(taxa_used_list),
        "seeds": [(float(k or 2.0), float(w or 0.5)) for k, w in seeds_list],
        "codeml": exe_fp["sha256"],
        "opts": {
            "PAML_FIX_BLENGTH_H0": int(PAML_FIX_BLENGTH_H0),
            "method": 0, "seqtype":1, "CodonFreq":2, "NSsites":0, "cleandata":0, "icode":0,
        },
        "logic_version": 1  # bump when logic meaningfully changes
    }
    return _sha256_bytes(json.dumps(key_dict, sort_keys=True).encode("utf-8")), key_dict

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

def create_paml_tree_files(iqtree_file, work_dir, gene_name):
    logging.info(f"[{gene_name}] Labeling internal branches conservatively...")
    t = Tree(iqtree_file, format=1)

    # Step 1: Propagate status up from the leaves ("post-order" traversal).
    # A temporary attribute 'group_status' is added to each node.
    for node in t.traverse("postorder"):
        if node.is_leaf():
            if node.name.startswith('0'):
                node.add_feature("group_status", "direct")
            elif node.name.startswith('1'):
                node.add_feature("group_status", "inverted")
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
    h0_tree_path = os.path.join(work_dir, f"{gene_name}_H0.tree")
    with open(h0_tree_path, 'w') as f:
        f.write("1\n" + h0_paml_str + "\n")

    # Return the tree object 't' which now has the 'group_status' features attached.
    return h1_tree_path, h0_tree_path, analysis_is_informative, t

def generate_paml_ctl(ctl_path, phy_file, tree_file, out_file, model_num,
                      init_kappa=None, init_omega=None, fix_blength=0):
    """
    model_num: 0 (one-ratio) or 2 (branch models)
    fix_blength: 0 = estimate branch lengths, 1/2 = keep as fixed (speed trade-off; keep 0 if unsure)
    """
    os.makedirs(os.path.dirname(ctl_path), exist_ok=True)
    kappa = init_kappa if init_kappa is not None else 2.0
    omega = init_omega if init_omega is not None else 0.5
    ctl_content = f"""
      seqfile = {phy_file}
      treefile = {tree_file}
      outfile = {out_file}

        noisy = 0
      verbose = 0
      runmode = 0

      seqtype = 1
    CodonFreq = 2
        model = {model_num}
      NSsites = 0
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
    """
    with open(ctl_path, 'w') as f:
        f.write(ctl_content.strip())

def parse_paml_lnl(outfile_path):
    """Extracts the log-likelihood (lnL) value from a PAML output file."""
    with open(outfile_path, 'r') as f:
        for line in f:
            if 'lnL' in line:
                match = re.search(r'lnL\(.*\):\s*([-\d\.]+)', line)
                if match:
                    return float(match.group(1))
    raise ValueError(f"Could not parse lnL from {outfile_path}")

def parse_h1_paml_output(outfile_path):
    params = {'kappa': np.nan, 'omega_background': np.nan, 'omega_direct': np.nan, 'omega_inverted': np.nan}
    omega_lines = []
    with open(outfile_path, 'r') as f:
        for line in f:
            if line.lstrip().startswith('kappa'):
                m = re.search(r'kappa \(ts/tv\)\s*=\s*([\d\.]+)', line)
                if m: params['kappa'] = float(m.group(1))
            # be permissive about indentation and wording
            if re.search(r'\bw\s*\(dN/dS\)', line) or re.search(r'w\s*for\s*branch\s*type', line) or re.search(r'w\s*ratios?\s*for\s*branches?', line):
                omega_lines.append(line)

    for line in omega_lines:
        if re.search(r'branch type\s*0', line):
            m = re.search(r'type\s*0:\s*([\d\.]+)', line)
            if m: params['omega_background'] = float(m.group(1))
        elif re.search(r'branch type\s*1', line):
            m = re.search(r'type\s*1:\s*([\d\.]+)', line)
            if m: params['omega_direct'] = float(m.group(1))
        elif re.search(r'branch type\s*2', line):
            m = re.search(r'type\s*2:\s*([\d\.]+)', line)
            if m: params['omega_inverted'] = float(m.group(1))
        else:
            m = re.search(r'=\s*([\d\.]+)|branches:\s*([\d\.]+)', line)
            if m:
                v = m.group(1) or m.group(2)
                if v: params['omega_background'] = float(v)
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


def region_worker(region):
    """Run IQ-TREE for a region after basic QC and cache its tree."""
    label = region['label']
    path = region['path']
    start_time = datetime.now()
    logging.info(f"[{label}] START IQ-TREE")
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

        temp_dir = tempfile.mkdtemp(prefix=f"{label}_")
        prefix = os.path.join(temp_dir, label)
        cmd = [IQTREE_PATH, '-s', os.path.abspath(path), '-m', 'MFP', '-T', '1', '--prefix', prefix, '-quiet', '-o', chimp]
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
    # Belt-and-suspenders: remove any PAML detritus if present from a failed previous run
    for pat in ('rst*', 'rub*', '2NG*', '2ML*', 'lnf', 'mlc'):
        for f in glob.glob(os.path.join(run_dir, pat)):
            try:
                os.remove(f)
            except OSError:
                pass
    run_command([PAML_PATH, ctl_path], run_dir, timeout=timeout)

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

        temp_dir = tempfile.mkdtemp(prefix=f"{gene_name}_", dir=os.getenv("PAML_TMPDIR"))

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

        # 1. Handle one-ratio run and its cache.
        one_ratio_ctl_str = _ctl_string(phy_abs, h0_tree, "one.out", 0, None, None, 0, {})
        one_ratio_key, _ = _hash_key_attempt(gene_phy_sha, h0_tree_str, taxa_used, one_ratio_ctl_str, exe_fp)
        
        one_payload = cache_read_json(PAML_CACHE_DIR, one_ratio_key, "attempt.json")
        if not one_payload:
            one_dir = os.path.join(temp_dir, "one_ratio")
            one_ctl = os.path.join(one_dir, f"{gene_name}_one.ctl")
            one_out = os.path.join(one_dir, f"{gene_name}_one.out")
            generate_paml_ctl(one_ctl, phy_abs, h0_tree, one_out, model_num=0)
            run_codeml_in(one_dir, one_ctl, PAML_TIMEOUT)
            _log_tail(one_out, 25, prefix=f"[{gene_name}|{region_label}] ONE out (recomputed)")
            seed = parse_simple_paml_output(one_out)
            one_payload = {"type":"one", "lnl": float(parse_paml_lnl(one_out)), "seed": seed}
            cache_write_json(PAML_CACHE_DIR, one_ratio_key, "attempt.json", one_payload)
            artifact_dir = os.path.join(_fanout_dir(PAML_CACHE_DIR, one_ratio_key), "artifacts")
            os.makedirs(artifact_dir, exist_ok=True)
            shutil.copy(one_out, os.path.join(artifact_dir, "one.out"))
            shutil.copy(one_ctl, os.path.join(artifact_dir, "one.ctl"))
        else:
            logging.info(f"[{gene_name}|{region_label}] Using cached ONE-RATIO result")
        seed = one_payload['seed']

        # 2. Define seeds and check for a cached META result.
        seeds_to_try = [(seed.get('kappa'), seed.get('omega_background')), (2.0, 0.2), (2.0, 1.0), (5.0, 2.0)]
        meta_key_hex, meta_key_dict = _hash_key_meta(gene_phy_sha, h0_tree_str, h1_tree_str, taxa_used, seeds_to_try, exe_fp, {})
        
        cached_meta = cache_read_json(PAML_CACHE_DIR, meta_key_hex, "meta.json")
        if cached_meta and cached_meta.get("result"):
            logging.info(f"[{gene_name}|{region_label}] Using cached META result")
            final_result.update(cached_meta["result"])
        else:
            with _with_lock(_fanout_dir(PAML_CACHE_DIR, meta_key_hex)):
                cached_meta = cache_read_json(PAML_CACHE_DIR, meta_key_hex, "meta.json")
                if cached_meta and cached_meta.get("result"):
                    logging.info(f"[{gene_name}|{region_label}] Using cached META (post-lock)")
                    final_result.update(cached_meta["result"])
                else:
                    best = {"lnl_h1": -np.inf, "lnl_h0": np.nan, "params": {}, "attempt_idx": None, "h0_key": None, "h1_key": None}
                    for i, (k_init, w_init) in enumerate(seeds_to_try):
                        attempt_dir = os.path.join(temp_dir, f"attempt_{i}")
                        h0_ctl_str  = _ctl_string(phy_abs, h0_tree, "h0.out", 2, k_init, w_init, (1 if PAML_FIX_BLENGTH_H0 else 0), {})
                        h1_ctl_str  = _ctl_string(phy_abs, h1_tree, "h1.out", 2, k_init, w_init, 0, {})
                        h0_key_hex, _  = _hash_key_attempt(gene_phy_sha, h0_tree_str, taxa_used, h0_ctl_str, exe_fp)
                        h1_key_hex, _  = _hash_key_attempt(gene_phy_sha, h1_tree_str, taxa_used, h1_ctl_str, exe_fp)
                        
                        h0_payload = cache_read_json(PAML_CACHE_DIR, h0_key_hex, "attempt.json")
                        if not h0_payload:
                            h0_ctl, h0_out = os.path.join(attempt_dir, f"{gene_name}_H0.ctl"), os.path.join(attempt_dir, f"{gene_name}_H0.out")
                            generate_paml_ctl(h0_ctl, phy_abs, h0_tree, h0_out, 2, k_init, w_init, 1 if PAML_FIX_BLENGTH_H0 else 0)
                            run_codeml_in(attempt_dir, h0_ctl, PAML_TIMEOUT)
                            _log_tail(h0_out, 20, prefix=f"{gene_name}|{region_label} H0 out (attempt {i+1})")
                            h0_payload = {"type":"h0", "lnl": float(parse_paml_lnl(h0_out))}
                            cache_write_json(PAML_CACHE_DIR, h0_key_hex, "attempt.json", h0_payload)
                            artifact_dir = os.path.join(_fanout_dir(PAML_CACHE_DIR, h0_key_hex), "artifacts")
                            os.makedirs(artifact_dir, exist_ok=True)
                            shutil.copy(h0_out, os.path.join(artifact_dir, "h0.out"))
                            shutil.copy(h0_ctl, os.path.join(artifact_dir, "h0.ctl"))

                        h1_payload = cache_read_json(PAML_CACHE_DIR, h1_key_hex, "attempt.json")
                        if not h1_payload:
                            h1_ctl, h1_out = os.path.join(attempt_dir, f"{gene_name}_H1.ctl"), os.path.join(attempt_dir, f"{gene_name}_H1.out")
                            generate_paml_ctl(h1_ctl, phy_abs, h1_tree, h1_out, 2, k_init, w_init)
                            run_codeml_in(attempt_dir, h1_ctl, PAML_TIMEOUT)
                            _log_tail(h1_out, 20, prefix=f"{gene_name}|{region_label} H1 out (attempt {i+1})")
                            h1_payload = {"type":"h1", "lnl": float(parse_paml_lnl(h1_out)), "params": parse_h1_paml_output(h1_out)}
                            cache_write_json(PAML_CACHE_DIR, h1_key_hex, "attempt.json", h1_payload)
                            artifact_dir = os.path.join(_fanout_dir(PAML_CACHE_DIR, h1_key_hex), "artifacts")
                            os.makedirs(artifact_dir, exist_ok=True)
                            shutil.copy(h1_out, os.path.join(artifact_dir, "h1.out"))
                            shutil.copy(h1_ctl, os.path.join(artifact_dir, "h1.ctl"))
                            mlc_path = os.path.join(attempt_dir, "mlc")
                            if os.path.exists(mlc_path): shutil.copy(mlc_path, os.path.join(artifact_dir, "mlc"))

                        if h1_payload["lnl"] >= h0_payload["lnl"] and h1_payload["lnl"] > best["lnl_h1"]:
                            best.update({"lnl_h1": h1_payload["lnl"], "lnl_h0": h0_payload["lnl"], "params": h1_payload.get("params", {}), "attempt_idx": i, "h0_key": h0_key_hex, "h1_key": h1_key_hex})
                    
                    computed_result = {}
                    if not np.isfinite(best["lnl_h1"]) or best["lnl_h1"] <= best["lnl_h0"]:
                        computed_result = {"status":"paml_optim_fail", "reason":"No attempt achieved H1 lnL >= H0 lnL."}
                    else:
                        lrt_stat = 2 * (best["lnl_h1"] - best["lnl_h0"])
                        p_value = chi2.sf(lrt_stat, df=1)
                        computed_result = {
                            "status":"success", "p_value": float(p_value), "lrt_stat": float(lrt_stat),
                            "lnl_h1": float(best["lnl_h1"]), "lnl_h0": float(best["lnl_h0"]), **best["params"],
                            "h0_key": best["h0_key"], "h1_key": best["h1_key"],
                            "n_leaves_region": len(region_taxa), "n_leaves_gene": len(gene_taxa), "n_leaves_pruned": len(taxa_used),
                            "chimp_in_region": any('pantro' in n.lower() for n in region_taxa),
                            "chimp_in_pruned": any('pantro' in n.lower() for n in t.get_leaf_names()),
                            "taxa_used": ';'.join(taxa_used)
                        }
                    final_result.update(computed_result)
                    meta_payload = {"schema": CACHE_SCHEMA_VERSION, "created": datetime.now().isoformat(timespec="seconds"), "key": meta_key_dict, "exe": exe_fp, "result": final_result}
                    cache_write_json(PAML_CACHE_DIR, meta_key_hex, "meta.json", meta_payload)

        # --- Post-computation/cache-hit processing ---
        if KEEP_PAML_OUT and final_result.get('status') == 'success':
            try:
                safe_region = re.sub(r'[^A-Za-z0-9_.-]+', '_', region_label)
                safe_gene   = re.sub(r'[^A-Za-z0-9_.-]+', '_', gene_name)
                dest_dir = os.path.join(PAML_OUT_DIR, f"{safe_gene}__{safe_region}")
                os.makedirs(dest_dir, exist_ok=True)
                
                for key in ["h0_key", "h1_key"]:
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
                generate_omega_result_figure(gene_name, region_label, status_tree, final_result)
            except Exception as fig_exc:
                logging.error(f"[{gene_name}] Failed to generate PAML results figure: {fig_exc}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        if final_result.get('status') == 'success':
            logging.info(f"[{gene_name}|{region_label}] LRT={final_result['lrt_stat']:.3f} p={final_result['p_value']:.3g}")
        logging.info(f"[{gene_name}|{region_label}] END codeml ({elapsed:.1f}s) status={final_result['status']}")
        
        return final_result

    except Exception as e:
        logging.error(f"FATAL ERROR for gene '{gene_name}' under region '{region_label}'.\n{traceback.format_exc()}")
        final_result.update({'status': 'runtime_error', 'reason': str(e)})
        return final_result
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

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

def run_overlapped(region_infos, region_gene_map, log_q):
    """
    Runs the full pipeline with overlapped IQ-TREE and PAML execution,
    using ProcessPoolExecutors and a cap on in-flight PAML jobs for back-pressure.
    """
    all_results = []
    inflight = deque()
    cap = PAML_WORKERS * 4
    completed_count = 0

    # Ensure workers use the 'spawn' context and our queue logger
    mpctx = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=PAML_WORKERS, mp_context=mpctx,
                             initializer=worker_logging_init, initargs=(log_q,)) as paml_exec, \
         ProcessPoolExecutor(max_workers=REGION_WORKERS, mp_context=mpctx,
                             initializer=worker_logging_init, initargs=(log_q,)) as region_exec:

        logging.info(f"Submitting {len(region_infos)} region tasks to pool...")
        region_futs = {region_exec.submit(region_worker, r) for r in region_infos}

        for rf in tqdm(as_completed(region_futs), total=len(region_futs), desc="Processing regions"):
            try:
                label, tree, reason = rf.result()
            except Exception as e:
                logging.error(f"A region task failed with an exception: {e}")
                continue

            if tree is None:
                logging.warning(f"Region {label} skipped: {reason}")
                continue
            
            genes_for_region = region_gene_map.get(label, [])
            if not genes_for_region:
                continue

            logging.info(f"Region {label} complete. Submitting {len(genes_for_region)} PAML jobs.")
            for gene_info in genes_for_region:
                flushed = submit_with_cap(
                    paml_exec, codeml_worker, (gene_info, tree, label), inflight, cap)
                for f in flushed:
                    try:
                        res = f.result()
                        all_results.append(res)
                        completed_count += 1
                        if (completed_count % 25 == 0) or (res.get('status') != 'success'):
                            logging.info(f"Completed {completed_count}: {res.get('gene')} in {res.get('region')} -> {res.get('status')}")
                        if completed_count % CHECKPOINT_EVERY == 0:
                            logging.info(f"--- Checkpointing {len(all_results)} results to {CHECKPOINT_FILE} ---")
                            pd.DataFrame(all_results).to_csv(CHECKPOINT_FILE, sep="\t", index=False, float_format='%.6g')
                    except Exception as e:
                        logging.error(f"A PAML job failed with an exception: {e}")

        # Drain any remaining PAML jobs
        logging.info(f"All regions processed. Draining {len(inflight)} remaining PAML jobs...")
        for f in tqdm(as_completed(list(inflight)), total=len(inflight), desc="Finalizing PAML jobs"):
            try:
                res = f.result()
                all_results.append(res)
                completed_count += 1
                if (completed_count % 25 == 0) or (res.get('status') != 'success'):
                    logging.info(f"Completed {completed_count}: {res.get('gene')} in {res.get('region')} -> {res.get('status')}")
                if completed_count % CHECKPOINT_EVERY == 0:
                    logging.info(f"--- Checkpointing {len(all_results)} results to {CHECKPOINT_FILE} ---")
                    pd.DataFrame(all_results).to_csv(CHECKPOINT_FILE, sep="\t", index=False, float_format='%.6g')
            except Exception as e:
                logging.error(f"A PAML job failed with an exception during drain: {e}")

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

        all_results = run_overlapped(region_infos, region_gene_map, log_q)
        results_df = pd.DataFrame(all_results)

        ordered_columns = ['region', 'gene', 'status', 'p_value', 'q_value', 'lrt_stat',
                           'omega_inverted', 'omega_direct', 'omega_background', 'kappa',
                           'lnl_h1', 'lnl_h0', 'n_leaves_region', 'n_leaves_gene',
                           'n_leaves_pruned', 'chimp_in_region', 'chimp_in_pruned',
                           'taxa_used', 'reason']
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
            pvals = successful['p_value'].dropna()
            if not pvals.empty:
                rejected, qvals = fdrcorrection(pvals, alpha=FDR_ALPHA, method='indep')
                qmap = {pvals.index[i]: q for i, q in enumerate(qvals)}
                results_df['q_value'] = results_df.index.map(qmap)
                logging.info(f"Applied FDR correction across {len(pvals)} gene×region tests")

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

        sig = results_df[(results_df['status'] == 'success') & (results_df['q_value'] < FDR_ALPHA)]
        if not sig.empty:
            logging.info(f"Significant gene×region tests (q < {FDR_ALPHA}):")
            for _, row in sig.sort_values('q_value').iterrows():
                logging.info(f"{row['region']} - {row['gene']}: q={row['q_value']:.4g}")
        else:
            logging.info("No significant tests.")

        logging.info("\nPipeline finished.")
    finally:
        listener.stop()

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
