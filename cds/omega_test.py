import os
import re
import sys
import glob
import subprocess
import multiprocessing
import tempfile
import getpass
import logging
import traceback
from datetime import datetime
import shutil

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

# ==============================================================================
# === CONFIGURATION & SETUP ====================================================
# ==============================================================================

# --- Centralized Logging ---
# A unique log file is created for each pipeline run.
LOG_FILE = f"pipeline_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout) # Also print logs to the console
    ]
)

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

# ==============================================================================
# === GENERIC HELPER FUNCTIONS (UNCHANGED CORE LOGIC) ==========================
# ==============================================================================

def run_command(command_list, work_dir):
    """
    Executes a shell command and raises a detailed error on failure.
    
    Args:
        command_list (list): The command and its arguments as a list of strings.
        work_dir (str): The directory in which to execute the command.
    """
    try:
        subprocess.run(
            command_list, cwd=work_dir, check=True,
            capture_output=True, text=True, shell=False
        )
    except subprocess.CalledProcessError as e:
        cmd_str = ' '.join(e.cmd)
        error_message = (
            f"\n--- COMMAND FAILED ---\n"
            f"COMMAND: '{cmd_str}'\n"
            f"EXIT CODE: {e.returncode}\n"
            f"WORKING DIR: {work_dir}\n"
            f"--- STDOUT ---\n{e.stdout}\n"
            f"--- STDERR ---\n{e.stderr}\n"
            f"--- END OF ERROR ---"
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
    h1_paml_str = re.sub(r"\[&&NHX:paml_mark=(#[01])\]", r" \1", h1_newick)
    h1_tree_path = os.path.join(work_dir, f"{gene_name}_H1.tree")
    with open(h1_tree_path, 'w') as f:
        f.write(f"{len(t_h1)} 1\n{h1_paml_str}")


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
        f.write(f"{len(t_h0)} 1\n{h0_paml_str}")

    # Return the tree object 't' which now has the 'group_status' features attached.
    return h1_tree_path, h0_tree_path, analysis_is_informative, t

def generate_paml_ctl(ctl_path, phy_file, tree_file, out_file, model_num):
    """
    Generates a codeml.ctl file with a specified evolutionary model.

    Args:
        ctl_path (str): The full path where the control file will be written.
        phy_file (str): The absolute path to the input sequence file (phylip format).
        tree_file (str): The absolute path to the input tree file.
        out_file (str): The absolute path for the PAML output file.
        model_num (int): The PAML model number to use (e.g., 0 for one-ratio, 2 for branch).
    """
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
        kappa = 2
      fix_omega = 0
        omega = 0.5
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
    """
    Robustly parses the H1 output file for estimated kappa and omega values.
    Handles cases where one foreground group might be missing.
    """
    params = {'kappa': np.nan, 'omega_background': np.nan, 'omega_direct': np.nan, 'omega_inverted': np.nan}
    omega_lines = []
    
    with open(outfile_path, 'r') as f:
        for line in f:
            if line.startswith('kappa'):
                match = re.search(r'kappa \(ts/tv\) = \s*([\d\.]+)', line)
                if match: params['kappa'] = float(match.group(1))
            elif re.match(r'\s*w\s*\(dN/dS\)|w for branch type', line):
                omega_lines.append(line.strip())

    for line in omega_lines:
        if "w for branch type 0" in line:
            match = re.search(r'type 0:\s*([\d\.]+)', line)
            if match: params['omega_background'] = float(match.group(1))
        elif "w for branch type 1" in line:
            match = re.search(r'type 1:\s*([\d\.]+)', line)
            if match: params['omega_direct'] = float(match.group(1))
        elif "w for branch type 2" in line:
            match = re.search(r'type 2:\s*([\d\.]+)', line)
            if match: params['omega_inverted'] = float(match.group(1))
        else:
            # This regex handles both "w (dN/dS) = ..." and "w for branches: ..."
            match = re.search(r'=\s*([\d\.]+)|branches:\s*([\d\.]+)', line)
            if match:
                value_str = match.group(1) or match.group(2)
                if value_str:
                    params['omega_background'] = float(value_str)
                
    return params

# ============================================================================
# === REGION/GENE HELPER FUNCTIONS ===========================================
# ============================================================================

def parse_region_filename(path):
    """Extract chromosome and coordinates from a region filename."""
    name = os.path.basename(path)
    m = re.match(r"combined_inversion_(chr[^_]+)_start(\d+)_end(\d+)\.phy", name)
    if not m:
        m = re.match(r"combined_inversion_(chr[^_]+)_(\d+)_(\d+)\.phy", name)
    if not m:
        raise ValueError(f"Unrecognized region filename format: {name}")
    chrom, start, end = m.groups()
    return {
        'path': path,
        'chrom': chrom,
        'start': int(start),
        'end': int(end),
        'label': f"{chrom}_{start}_{end}"
    }


def load_gene_metadata(tsv_path='phy_metadata.tsv'):
    """Load gene coordinate metadata from a TSV file."""
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(
            "Metadata file 'phy_metadata.tsv' not found; cannot map genes to regions.")
    df = pd.read_csv(tsv_path, sep='\t')
    meta = {}
    for _, row in df.iterrows():
        key = (row['gene'], row['enst'])
        meta[key] = {
            'chrom': row['chr'],
            'start': int(row['start']),
            'end': int(row['end'])
        }
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


# ============================================================================
# === GENE WORKER USING REGION TOPOLOGY ======================================
# ============================================================================

def codeml_worker(gene_info, region_tree_file, region_label):
    """Run codeml for a gene using the provided region tree."""
    gene_name = gene_info['label']
    result = {'gene': gene_name, 'region': region_label, 'status': 'runtime_error', 'reason': 'Unknown failure'}
    temp_dir = None

    try:
        qc_passed, qc_message = perform_qc(gene_info['path'])
        if not qc_passed:
            result.update({'status': 'qc_fail', 'reason': qc_message})
            return result

        temp_dir = tempfile.mkdtemp(prefix=f"{gene_name}_")

        region_taxa = Tree(region_tree_file, format=1).get_leaf_names()
        gene_taxa = read_taxa_from_phy(gene_info['path'])
        pruned_tree = os.path.join(temp_dir, f"{gene_name}_pruned.tree")
        prune_region_tree(region_tree_file, gene_taxa, pruned_tree)

        t = Tree(pruned_tree, format=1)
        chimp_name = next((n for n in t.get_leaf_names() if 'pantro' in n.lower() or 'pan_troglodytes' in n.lower()), None)
        if len(t.get_leaf_names()) < 4:
            result.update({'status': 'uninformative_topology', 'reason': 'Fewer than four taxa after pruning'})
            return result

        h1_tree, h0_tree, informative, status_tree = create_paml_tree_files(pruned_tree, temp_dir, gene_name)
        if not informative:
            result.update({'status': 'uninformative_topology', 'reason': 'No pure internal branches found for both direct and inverted groups.'})
            return result

        phy_abs = os.path.abspath(gene_info['path'])

        h1_ctl = os.path.join(temp_dir, f"{gene_name}_H1.ctl")
        h1_out = os.path.join(temp_dir, f"{gene_name}_H1.out")
        generate_paml_ctl(h1_ctl, phy_abs, h1_tree, h1_out, model_num=2)
        run_command([PAML_PATH, h1_ctl], temp_dir)
        lnl_h1 = parse_paml_lnl(h1_out)
        paml_params = parse_h1_paml_output(h1_out)

        try:
            generate_omega_result_figure(gene_name, region_label, status_tree, paml_params)
        except Exception as fig_exc:
            logging.error(f"[{gene_name}] Failed to generate PAML results figure: {fig_exc}")

        h0_ctl = os.path.join(temp_dir, f"{gene_name}_H0.ctl")
        h0_out = os.path.join(temp_dir, f"{gene_name}_H0.out")
        generate_paml_ctl(h0_ctl, phy_abs, h0_tree, h0_out, model_num=2)
        run_command([PAML_PATH, h0_ctl], temp_dir)
        lnl_h0 = parse_paml_lnl(h0_out)

        if lnl_h1 < lnl_h0:
            reason = f"lnL_H1({lnl_h1}) < lnL_H0({lnl_h0})"
            result.update({'status': 'paml_optim_fail', 'reason': reason})
            return result

        lrt_stat = 2 * (lnl_h1 - lnl_h0)
        p_value = chi2.sf(lrt_stat, df=1)

        result.update({
            'status': 'success', 'p_value': p_value, 'lrt_stat': lrt_stat,
            'lnl_h1': lnl_h1, 'lnl_h0': lnl_h0, **paml_params,
            'n_leaves_region': len(region_taxa),
            'n_leaves_gene': len(gene_taxa),
            'n_leaves_pruned': len(t.get_leaf_names()),
            'chimp_in_region': any('pantro' in n.lower() or 'pan_troglodytes' in n.lower() for n in region_taxa),
            'chimp_in_pruned': chimp_name is not None,
            'taxa_used': ';'.join(t.get_leaf_names())
        })
        return result

    except Exception as e:
        logging.error(f"FATAL ERROR for gene '{gene_name}' under region '{region_label}'.\n{traceback.format_exc()}")
        result.update({'status': 'runtime_error', 'reason': str(e)})
        return result

# ==============================================================================
# === MAIN EXECUTION AND REPORTING =============================================
# ==============================================================================

def main():
    """Run region-first pipeline: IQ-TREE on regions, codeml on genes."""

    logging.info("--- Starting Region→Gene Differential Selection Pipeline ---")

    if not (os.path.exists(IQTREE_PATH) and os.access(IQTREE_PATH, os.X_OK)):
        logging.critical(f"FATAL: IQ-TREE not found or not executable at '{IQTREE_PATH}'")
        sys.exit(1)
    if not (os.path.exists(PAML_PATH) and os.access(PAML_PATH, os.X_OK)):
        logging.critical(f"FATAL: PAML codeml not found or not executable at '{PAML_PATH}'")
        sys.exit(1)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(ANNOTATED_FIGURE_DIR, exist_ok=True)

    region_files = glob.glob('combined_inversion_*.phy')
    gene_files = [f for f in glob.glob('combined_*.phy') if 'inversion' not in os.path.basename(f)]

    if not region_files:
        logging.critical("FATAL: No region alignment files found.")
        sys.exit(1)
    if not gene_files:
        logging.critical("FATAL: No gene alignment files found.")
        sys.exit(1)

    metadata = load_gene_metadata()
    region_infos = [parse_region_filename(f) for f in region_files]
    gene_infos = [parse_gene_filename(f, metadata) for f in gene_files]
    region_gene_map = build_region_gene_map(region_infos, gene_infos)

    all_results = []
    tasks = []
    for region in region_infos:
        region_label = region['label']
        region_path = region['path']
        logging.info(f"Processing region {region_label}")

        region_taxa = read_taxa_from_phy(region_path)
        chimp_name = next((t for t in region_taxa if 'pantro' in t.lower() or 'pan_troglodytes' in t.lower()), None)
        if not chimp_name:
            logging.error(f"Skipping region {region_label} because chimp outgroup is missing")
            continue
        if len(region_taxa) < 6 or not any(t.startswith('0') for t in region_taxa) or not any(t.startswith('1') for t in region_taxa):
            logging.warning(f"Skipping region {region_label} due to insufficient taxa or haplotype diversity")
            continue

        temp_dir = tempfile.mkdtemp(prefix=f"{region_label}_")
        prefix = os.path.join(temp_dir, region_label)
        iqtree_cmd = [IQTREE_PATH, '-s', os.path.abspath(region_path), '-m', 'MFP', '-T', '1', '--prefix', prefix, '-quiet', '-o', chimp_name]
        run_command(iqtree_cmd, temp_dir)
        region_tree = f"{prefix}.treefile"
        if not os.path.exists(region_tree):
            logging.error(f"Region tree not found for {region_label}")
            continue
        try:
            generate_tree_figure(region_tree, region_label)
        except Exception as e:
            logging.error(f"Failed to generate region tree figure for {region_label}: {e}")

        for gene_info in region_gene_map.get(region_label, []):
            tasks.append((gene_info, region_tree, region_label))

    if tasks:
        def _worker(args):
            return codeml_worker(*args)
        with multiprocessing.Pool() as pool:
            for res in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks)):
                all_results.append(res)

    results_df = pd.DataFrame(all_results)

    successful = results_df[results_df['status'] == 'success'].copy()
    if not successful.empty:
        pvals = successful['p_value'].dropna()
        if not pvals.empty:
            rejected, qvals = fdrcorrection(pvals, alpha=FDR_ALPHA, method='indep')
            qmap = {pvals.index[i]: q for i, q in enumerate(qvals)}
            results_df['q_value'] = results_df.index.map(qmap)

    ordered_columns = ['region', 'gene', 'status', 'p_value', 'q_value', 'lrt_stat',
                       'omega_inverted', 'omega_direct', 'omega_background', 'kappa',
                       'lnl_h1', 'lnl_h0', 'n_leaves_region', 'n_leaves_gene',
                       'n_leaves_pruned', 'chimp_in_region', 'chimp_in_pruned',
                       'taxa_used', 'reason']
    for col in ordered_columns:
        if col not in results_df.columns:
            results_df[col] = np.nan
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

if __name__ == '__main__':
    main()
