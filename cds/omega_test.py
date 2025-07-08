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
IQTREE_PATH = os.path.abspath('./iqtree-3.0.1-Linux/bin/iqtree3')
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

def generate_tree_figure(tree_file, gene_name):
    """Creates a publication-quality phylogenetic tree figure using ete3."""
    t = Tree(tree_file, format=1)
    ts = TreeStyle()
    ts.layout_fn = _tree_layout
    ts.show_leaf_name = False
    ts.show_scale = False
    ts.branch_vertical_margin = 8
    ts.title.add_face(TextFace(f"Phylogeny of {gene_name}", fsize=16, ftype="Arial"), column=0)
    
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
    
    figure_path = os.path.join(FIGURE_DIR, f"{gene_name}.png")
    t.render(figure_path, w=200, units="mm", dpi=300, tree_style=ts)

def generate_colored_branch_figure(gene_name, t):
    """
    Creates a tree figure with branches colored by their inferred group status.
    This uses the tree object directly after statuses have been assigned.
    """
    # Define colors for each selection group category
    BRANCH_COLORS = {
        'direct': '#0072B2',   # Blue
        'inverted': '#D55E00',  # Vermillion
        'both': '#E69F00',      # Orange (for mixed/ancestral)
        'outgroup': '#56B4E9'   # Sky Blue
    }

    # A layout function to color branches based on the 'group_status' feature
    def _branch_color_layout(node):
        # Default style for all branches
        nstyle = NodeStyle()
        nstyle["hz_line_width"] = 2
        nstyle["vt_line_width"] = 2
        
        # Get the group status computed earlier; default to 'both' if not present
        status = getattr(node, "group_status", "both")
        color = BRANCH_COLORS.get(status, "#000000") # Default to black if status is unknown

        nstyle["hz_line_color"] = color
        nstyle["vt_line_color"] = color

        # Style leaves to show their population identity
        if node.is_leaf():
            name = node.name
            pop_match = re.search(r'_(AFR|EUR|EAS|SAS|AMR)_', name)
            pop = pop_match.group(1) if pop_match else 'CHIMP'
            leaf_color = POP_COLORS.get(pop, "#C0C0C0")
            nstyle["fgcolor"] = leaf_color
            nstyle["size"] = 5
        else:
            nstyle["size"] = 0 # Make internal nodes invisible for a clean look

        node.set_style(nstyle)

    ts = TreeStyle()
    ts.layout_fn = _branch_color_layout
    ts.show_leaf_name = False
    ts.show_scale = False
    ts.branch_vertical_margin = 8
    ts.title.add_face(TextFace(f"Selection Groups for {gene_name}", fsize=16, ftype="Arial"), column=0)
    
    # Create a legend describing the branch colors
    ts.legend.add_face(TextFace("Branch Category", fsize=10, ftype="Arial", fstyle="Bold"), column=0)
    for status, color in BRANCH_COLORS.items():
        ts.legend.add_face(RectFace(10, 10, fgcolor=color, bgcolor=color), column=0)
        ts.legend.add_face(TextFace(f" {status.capitalize()}", fsize=9), column=1)
    ts.legend_position = 4 # Position the legend in the top-right

    figure_path = os.path.join(ANNOTATED_FIGURE_DIR, f"{gene_name}_branches.png")
    t.render(figure_path, w=200, units="mm", dpi=300, tree_style=ts)

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
            node.add_feature("paml_mark", "#0")
        elif status == "inverted":
            node.add_feature("paml_mark", "#1")
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
            if match: params['omega_direct'] = float(match.group(1))
        elif "w for branch type 1" in line:
            match = re.search(r'type 1:\s*([\d\.]+)', line)
            if match: params['omega_inverted'] = float(match.group(1))
        else:
            # This regex handles both "w (dN/dS) = ..." and "w for branches: ..."
            match = re.search(r'=\s*([\d\.]+)|branches:\s*([\d\.]+)', line)
            if match:
                value_str = match.group(1) or match.group(2)
                if value_str:
                    params['omega_background'] = float(value_str)
                
    return params

# ==============================================================================
# === MAIN WORKER FUNCTION =====================================================
# ==============================================================================

def worker_function(phy_filepath, status_summary):
    """
    Worker process for a single gene.
    This function is now designed to be called via functools.partial to include
    a shared status_summary dictionary from a multiprocessing.Manager.
    """
    gene_name = os.path.basename(phy_filepath).replace('combined_', '').replace('.phy', '')
    result = {'gene': gene_name, 'status': 'runtime_error', 'reason': 'Unknown failure'}
    temp_dir = None
    # This flag ensures we only decrement the 'running_paml' counter if it was first incremented.
    paml_was_started = False

    try:
        qc_passed, qc_message = perform_qc(phy_filepath)
        if not qc_passed:
            result.update({'status': 'qc_fail', 'reason': qc_message})
            return result

        temp_dir = tempfile.mkdtemp(prefix=f"{gene_name}_")
        phy_file_abs_path = os.path.abspath(phy_filepath)

        # --- 1. Build Phylogeny with IQ-TREE ---
        logging.info(f"[{gene_name}] Starting IQ-TREE...")
        iqtree_out_prefix = os.path.join(temp_dir, gene_name)
        chimp_name = next((line.split()[0] for line in open(phy_filepath) if 'pantro' in line.lower() or 'pan_troglodytes' in line.lower()), None)
        if chimp_name is None: raise ValueError("Chimp outgroup name not found.")
        run_command([IQTREE_PATH, '-s', phy_file_abs_path, '-o', chimp_name, '-m', 'MFP', '-T', '1', '--prefix', iqtree_out_prefix, '-quiet'], temp_dir)
        tree_file = f"{iqtree_out_prefix}.treefile"
        if not os.path.exists(tree_file): raise FileNotFoundError("IQ-TREE did not produce a treefile.")

        logging.info(f"[{gene_name}] Generating figure...")
        generate_tree_figure(tree_file, gene_name)

        # --- 2. Prepare Trees for PAML and check if informative ---
        h1_tree, h0_tree, is_informative, status_annotated_tree = create_paml_tree_files(tree_file, temp_dir, gene_name)

        # --- Generate the colored branch figure for diagnostics ---
        # This is done for all genes that pass QC, regardless of whether they are
        # informative for the LRT, as it provides a useful visual.
        logging.info(f"[{gene_name}] Generating colored branch figure...")
        generate_colored_branch_figure(gene_name, status_annotated_tree)

        if not is_informative:
            reason = "No pure internal branches found for both direct and inverted groups."
            result.update({'status': 'uninformative_topology', 'reason': reason})
            logging.info(f"Intermediate files for uninformative gene '{gene_name}' are in: {temp_dir}")
            return result

        # --- This gene PASSED the filter. Update the live counter and proceed. ---
        logging.info(f"[{gene_name}] Topology is INFORMATIVE. Proceeding to PAML analysis.")
        # Safely increment the shared 'running_paml' counter.
        status_summary['running_paml'] = status_summary.get('running_paml', 0) + 1
        paml_was_started = True

        # --- 3. Run H1 (Alternative Model) ---
        logging.info(f"[{gene_name}] Running PAML H1 (Alternative)...")
        h1_ctl = os.path.join(temp_dir, f"{gene_name}_H1.ctl")
        h1_out = os.path.join(temp_dir, f"{gene_name}_H1.out")
        generate_paml_ctl(h1_ctl, phy_file_abs_path, h1_tree, h1_out, model_num=2)
        run_command([PAML_PATH, h1_ctl], temp_dir)
        lnl_h1 = parse_paml_lnl(h1_out)
        paml_params = parse_h1_paml_output(h1_out)

        # --- 4. Run H0 (Null Model) ---
        logging.info(f"[{gene_name}] Running PAML H0 (Null)...")
        h0_ctl = os.path.join(temp_dir, f"{gene_name}_H0.ctl")
        h0_out = os.path.join(temp_dir, f"{gene_name}_H0.out")
        generate_paml_ctl(h0_ctl, phy_file_abs_path, h0_tree, h0_out, model_num=2)
        run_command([PAML_PATH, h0_ctl], temp_dir)
        lnl_h0 = parse_paml_lnl(h0_out)

        # --- 5. Perform Likelihood Ratio Test ---
        if lnl_h1 < lnl_h0:
            reason = f'lnL_H1({lnl_h1}) < lnL_H0({lnl_h0})'
            logging.warning(f"[{gene_name}] PAML optimization issue: {reason}. Skipping.")
            result.update({'status': 'paml_optim_fail', 'reason': reason})
            logging.info(f"Intermediate files for failed gene '{gene_name}' are in: {temp_dir}")
            return result

        lrt_stat = 2 * (lnl_h1 - lnl_h0)
        p_value = chi2.sf(lrt_stat, df=1)

        result.update({
            'status': 'success', 'p_value': p_value, 'lrt_stat': lrt_stat,
            'lnl_h1': lnl_h1, 'lnl_h0': lnl_h0, 'reason': 'OK', **paml_params
        })
        return result

    except Exception as e:
        logging.error(f"FATAL ERROR in worker for gene '{gene_name}'.\n{traceback.format_exc()}")
        result.update({'status': 'runtime_error', 'reason': str(e)})
        if temp_dir:
            logging.info(f"Intermediate files for failed gene '{gene_name}' are in: {temp_dir}")
        return result
    
    finally:
        # This 'finally' block ensures that the running counter is always decremented
        # when a worker finishes, regardless of whether it succeeded or failed.
        if paml_was_started:
            status_summary['running_paml'] = status_summary.get('running_paml', 1) - 1

# ==============================================================================
# === MAIN EXECUTION AND REPORTING =============================================
# ==============================================================================

def main():
    """Main function to discover files, run the pipeline in parallel, and report results."""
    # This import is needed here for the partial function wrapper.
    from functools import partial

    logging.info("--- Starting Differential Selection Pipeline (Branch Model) ---")

    if not (os.path.exists(IQTREE_PATH) and os.access(IQTREE_PATH, os.X_OK)):
        logging.critical(f"FATAL: IQ-TREE not found or not executable at '{IQTREE_PATH}'")
        sys.exit(1)
    if not (os.path.exists(PAML_PATH) and os.access(PAML_PATH, os.X_OK)):
        logging.critical(f"FATAL: PAML codeml not found or not executable at '{PAML_PATH}'")
        sys.exit(1)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    phy_files = glob.glob('combined_*.phy')
    if not phy_files:
        logging.critical("FATAL: No 'combined_*.phy' files found in the current directory.")
        sys.exit(1)

    mapt_file_path = None
    for f in phy_files:
        if 'MAPT' in os.path.basename(f):
            mapt_file_path = f
            break
    if mapt_file_path:
        logging.info(f"Prioritizing gene MAPT found at: {mapt_file_path}")
        phy_files.remove(mapt_file_path)
        phy_files.insert(0, mapt_file_path)
    else:
        logging.warning("Could not find a specific file for gene MAPT to prioritize.")

    logging.info(f"Found {len(phy_files)} CDS files to process.")
    cpu_cores = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
    logging.info(f"Using {cpu_cores} CPU cores for parallel processing.")

    all_results = []
    
    # Use a Manager to create a shared dictionary for live status tracking.
    with multiprocessing.Manager() as manager:
        status_summary = manager.dict({
            'running_paml': 0,
            'success': 0,
            'qc_fail': 0,
            'uninformative_topology': 0,
            'paml_optim_fail': 0,
            'runtime_error': 0,
        })

        # Use functools.partial to create a new function-like object that has the
        # status_summary dictionary "baked in". This is the clean way to pass
        # extra, constant arguments to a worker function in a map call.
        worker_with_summary = partial(worker_function, status_summary=status_summary)

        with multiprocessing.Pool(processes=cpu_cores) as pool:
            with tqdm(total=len(phy_files), desc="Processing CDSs", file=sys.stdout) as pbar:
                for result in pool.imap_unordered(worker_with_summary, phy_files):
                    all_results.append(result)
                    pbar.update(1)

                    status = result.get('status', 'runtime_error')
                    status_summary[status] = status_summary.get(status, 0) + 1

                    # --- Format and Log the Live Aggregate Summary ---
                    summary_str = (
                        f"Running PAML: {status_summary['running_paml']} | "
                        f"Success: {status_summary['success']} | "
                        f"Filtered (Uninformative): {status_summary['uninformative_topology']} | "
                        f"Failed (QC/PAML/Error): {status_summary['qc_fail']}/{status_summary['paml_optim_fail']}/{status_summary['runtime_error']}"
                    )
                    logging.info(f"[PIPELINE STATUS] {summary_str}")

                    # --- Log the Detailed Report for the INDIVIDUAL Job That Just Finished ---
                    gene = result.get('gene', 'unknown_gene')
                    report_header = f"\n{'='*10} Job Finished {'='*10}\nGene: {gene}\nStatus: {status.upper()}"
                    report_body = ""
                    if status == 'success':
                        pval = result.get('p_value')
                        w_inv = result.get('omega_inverted', 'N/A')
                        w_dir = result.get('omega_direct', 'N/A')
                        report_body = (
                            f"p-value: {pval:.4g}\n"
                            f"Omega Inverted: {w_inv:.4f}\n"
                            f"Omega Direct:   {w_dir:.4f}"
                        )
                    else:
                        reason = result.get('reason', 'No reason provided.')
                        if len(reason) > 150:
                            reason = reason[:150] + "..."
                        report_body = f"Reason: {reason}"
                    
                    full_report = f"{report_header}\n{report_body}\n{'='*35}"
                    logging.info(full_report)

        # --- Final Report Section (after all jobs are complete) ---
        # Convert the managed dict back to a regular dict for the final report.
        final_status_counts = dict(status_summary)
        # We don't need the 'running_paml' in the final summary table.
        del final_status_counts['running_paml']


    # --- Create a comprehensive DataFrame and perform FDR Correction ---
    results_df = pd.DataFrame(all_results)
    
    successful_runs = results_df[results_df['status'] == 'success'].copy()
    if not successful_runs.empty:
        pvals = successful_runs['p_value'].dropna()
        if not pvals.empty:
            rejected, qvals = fdrcorrection(pvals, alpha=FDR_ALPHA, method='indep')
            qval_map = {pvals.index[i]: q for i, q in enumerate(qvals)}
            results_df['q_value'] = results_df.index.map(qval_map)

    # --- Write Full TSV Report ---
    ordered_columns = [
        'gene', 'status', 'p_value', 'q_value', 'lrt_stat', 
        'omega_inverted', 'omega_direct', 'omega_background', 'kappa', 
        'lnl_h1', 'lnl_h0', 'reason'
    ]
    for col in ordered_columns:
        if col not in results_df.columns:
            results_df[col] = np.nan
            
    results_df = results_df[ordered_columns]
    results_df.to_csv(RESULTS_TSV, sep='\t', index=False, float_format='%.6g')
    logging.info(f"All results, including failures, saved to: {RESULTS_TSV}")

    # --- Generate Final Console Report ---
    logging.info("\n\n" + "="*75)
    logging.info("--- FINAL PIPELINE REPORT ---")
    logging.info(f"Total CDSs Found: {len(phy_files)}")
    for status, count in final_status_counts.items():
        logging.info(f"  - {status.replace('_', ' ').title()}: {count}")
    logging.info("="*75 + "\n")

    significant_df = results_df[(results_df['status'] == 'success') & (results_df['q_value'] < FDR_ALPHA)]
    
    logging.info(f"--- Genes with Significant Differential Selection (q < {FDR_ALPHA}) ---")
    if not significant_df.empty:
        sorted_sig = significant_df.sort_values('q_value')
        logging.info(f"{'Gene':<25} {'p-value':<10} {'q-value':<10} {'w_inverted':<12} {'w_direct':<12}")
        logging.info("-" * 75)
        for _, row in sorted_sig.iterrows():
            logging.info(f"{row['gene']:<25} {row['p_value']:<10.4g} {row['q_value']:<10.4g} {row['omega_inverted']:<12.4f} {row['omega_direct']:<12.4f}")
    else:
        logging.info("  None found.")
    logging.info("")

    for status_type, status_name in [('qc_fail', 'QC Failures'), ('paml_optim_fail', 'PAML Optimization Failures'), ('runtime_error', 'Runtime Errors')]:
        failures = results_df[results_df['status'] == status_type]
        if not failures.empty:
            logging.warning(f"--- List of {status_name} ---")
            for _, row in failures.sort_values('gene').iterrows():
                logging.warning(f"  - {row['gene']:<40} Reason: {row.get('reason', 'N/A')}")
            logging.warning("")
    
    logging.info(f"\nPipeline finished successfully.")
    logging.info(f"Full details saved to log file: '{LOG_FILE}'.")
    logging.info(f"Tree figures saved to directory: '{FIGURE_DIR}/'.")
    logging.info(f"A comprehensive TSV of all results is at: '{RESULTS_TSV}'.")

if __name__ == '__main__':
    main()
