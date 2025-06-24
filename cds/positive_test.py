import os
import re
import sys
import glob
import subprocess
import multiprocessing
import tempfile
import warnings
import getpass
import logging
import traceback
from datetime import datetime
from collections import defaultdict

# --- Scientific Computing Imports ---
import numpy as np
from scipy.stats import chi2
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm

# --- ETE3 and QT Configuration for Headless Environments ---
os.environ["QT_QPA_PLATFORM"] = "offscreen"
user = getpass.getuser()
runtime_dir = f"/tmp/runtime-{user}"
os.makedirs(runtime_dir, exist_ok=True, mode=0o700)
os.environ['XDG_RUNTIME_DIR'] = runtime_dir
from ete3 import Tree
from ete3.treeview import TreeStyle, NodeStyle, TextFace, CircleFace, RectFace

# --- Centralized Logging Setup ---
# Create a unique log file for each run.
LOG_FILE = f"pipeline_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout) # Also print logs to the console
    ]
)

# --- Configuration ---
# Relative paths to the executable files from where this script is run.
IQTREE_PATH = os.path.abspath('./iqtree-3.0.1-Linux/bin/iqtree3')
PAML_PATH = os.path.abspath('../paml/bin/codeml')

# QC and analysis parameters
DIVERGENCE_THRESHOLD = 0.10
FDR_ALPHA = 0.05

# Superpopulation colors for tree figures
POP_COLORS = {
    'AFR': '#F05031', 'EUR': '#3173F0', 'EAS': '#35A83A',
    'SAS': '#F031D3', 'AMR': '#B345F0', 'CHIMP': '#808080'
}

# --- Output Directories ---
FIGURE_DIR = "tree_figures"

# --- Main Script Logic ---

def run_command(command_list, work_dir):
    """
    Executes a command and raises a detailed error on failure.
    """
    try:
        # shell=False is critical for security and correct argument handling
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
        # This exception will be caught by the fail-safe worker
        raise RuntimeError(error_message) from e

def perform_qc(phy_file_path):
    """
    Performs quality control. Returns (False, reason) if QC fails,
    (True, "QC Passed") otherwise. Catches its own errors.
    """
    with open(phy_file_path, 'r') as f:
        lines = f.readlines()

    if not lines or len(lines[0].strip().split()) < 2:
        return False, f"File is empty or header is missing/malformed."

    header = lines[0].strip().split()
    seq_length = int(header[1])

    if seq_length % 3 != 0:
        return False, f"Sequence length {seq_length} not divisible by 3."

    human_seqs = [line.strip().split()[1] for line in lines[1:] if line.strip().split()[0].startswith(('0', '1'))]
    chimp_seq = None
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) > 0:
            name_lower = parts[0].lower()
            if 'pantro' in name_lower or 'pan_troglodytes' in name_lower:
                chimp_seq = parts[1]
                break

    if not human_seqs or not chimp_seq:
        return False, "Could not find both human and chimp sequences."

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
    """A layout function to dynamically style nodes for the tree figure."""
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
    # Haplotype Status Legend
    ts.legend.add_face(TextFace("Haplotype Status", fsize=10, ftype="Arial", fstyle="Bold"), column=0)
    ts.legend.add_face(CircleFace(5, "black", style="circle"), column=0); ts.legend.add_face(TextFace(" Direct", fsize=9), column=1)
    ts.legend.add_face(CircleFace(5, "black", style="sphere"), column=0); ts.legend.add_face(TextFace(" Inverted", fsize=9), column=1)
    ts.legend.add_face(RectFace(10, 10, "black", "black"), column=0); ts.legend.add_face(TextFace(" Chimpanzee (Outgroup)", fsize=9), column=1)
    ts.legend.add_face(TextFace(" "), column=2) # Spacer
    # Population Legend
    ts.legend.add_face(TextFace("Super-population", fsize=10, ftype="Arial", fstyle="Bold"), column=3)
    for pop, color in POP_COLORS.items():
        ts.legend.add_face(CircleFace(10, color), column=3); ts.legend.add_face(TextFace(f" {pop}", fsize=9), column=4)
    ts.legend_position = 1
    figure_path = os.path.join(FIGURE_DIR, f"{gene_name}.png")
    t.render(figure_path, w=200, units="mm", dpi=300, tree_style=ts)

def annotate_tree_for_paml(tree_file, work_dir, gene_name):
    """Robustly annotates a tree file for PAML's branch-site test."""
    t = Tree(tree_file, format=1)
    direct_fg_tree_path, inverted_fg_tree_path = None, None
    # Direct Haplotypes
    direct_leaves = [leaf.name for leaf in t if leaf.name.startswith('0')]
    if len(direct_leaves) > 1:
        t_direct = t.copy("newick")
        mrca = t_direct.get_common_ancestor(direct_leaves)
        if mrca and not mrca.is_root() and not mrca.is_leaf():
            mrca.add_feature("paml_mark", "#1")
            direct_fg_tree_path = os.path.join(work_dir, f"{gene_name}_direct_fg.tree")
            newick_str = t_direct.write(format=1, features=["paml_mark"])
            paml_friendly_str = re.sub(r"\[&&NHX:paml_mark=#1\]", " #1", newick_str)
            with open(direct_fg_tree_path, 'w') as f:
                f.write(f"{len(t_direct)} 1\n{paml_friendly_str}")
    # Inverted Haplotypes
    inverted_leaves = [leaf.name for leaf in t if leaf.name.startswith('1')]
    if len(inverted_leaves) > 1:
        t_inverted = t.copy("newick")
        mrca = t_inverted.get_common_ancestor(inverted_leaves)
        if mrca and not mrca.is_root() and not mrca.is_leaf():
            mrca.add_feature("paml_mark", "#1")
            inverted_fg_tree_path = os.path.join(work_dir, f"{gene_name}_inverted_fg.tree")
            newick_str = t_inverted.write(format=1, features=["paml_mark"])
            paml_friendly_str = re.sub(r"\[&&NHX:paml_mark=#1\]", " #1", newick_str)
            with open(inverted_fg_tree_path, 'w') as f:
                f.write(f"{len(t_inverted)} 1\n{paml_friendly_str}")
    return direct_fg_tree_path, inverted_fg_tree_path

def generate_paml_ctl(ctl_path, phy_file, tree_file, out_file, is_null_model):
    """Programmatically generates a codeml.ctl file."""
    ctl_content = f"""
      seqfile = {phy_file}
      treefile = {tree_file}
      outfile = {out_file}
      noisy = 0
      verbose = 0
      runmode = 0
      seqtype = 1
      CodonFreq = 2
      model = 2
      NSsites = 2
      icode = 0
      fix_kappa = 0
      kappa = 2
      fix_omega = {'1' if is_null_model else '0'}
      omega = {'1.0' if is_null_model else '1.5'}
      cleandata = 0
    """
    with open(ctl_path, 'w') as f: f.write(ctl_content.strip())

def parse_paml_lnl(outfile_path):
    """Extracts the log-likelihood (lnL) from a PAML output file."""
    with open(outfile_path, 'r') as f:
        for line in f:
            if 'lnL' in line:
                match = re.search(r'lnL\(.*\):\s*([-\d\.]+)', line)
                if match: return float(match.group(1))
    raise ValueError(f"Could not parse lnL from {outfile_path}")

def run_paml_test(phy_file_abs_path, annotated_tree, work_dir):
    """Runs a full PAML alt vs null test and returns the p-value."""
    base_name = os.path.basename(annotated_tree).replace('.tree', '')
    # Alternative Model
    alt_ctl = os.path.join(work_dir, f"{base_name}_alt.ctl")
    alt_out = os.path.join(work_dir, f"{base_name}_alt.out")
    generate_paml_ctl(alt_ctl, phy_file_abs_path, annotated_tree, alt_out, is_null_model=False)
    run_command([PAML_PATH, alt_ctl], work_dir)
    lnl_alt = parse_paml_lnl(alt_out)
    # Null Model
    null_ctl = os.path.join(work_dir, f"{base_name}_null.ctl")
    null_out = os.path.join(work_dir, f"{base_name}_null.out")
    generate_paml_ctl(null_ctl, phy_file_abs_path, annotated_tree, null_out, is_null_model=True)
    run_command([PAML_PATH, null_ctl], work_dir)
    lnl_null = parse_paml_lnl(null_out)

    if lnl_alt < lnl_null:
        logging.warning(f"PAML optimization issue in {base_name}: "
                        f"Alternative model Likelihood ({lnl_alt}) is less than Null ({lnl_null}). Skipping.")
        return np.nan

    lr_stat = 2 * (lnl_alt - lnl_null)
    p_value = 0.5 * chi2.sf(lr_stat, df=1)
    return p_value

def worker_function(phy_filepath):
    """
    FAIL-SAFE main worker process. Runs the entire pipeline for one CDS file.
    Catches ALL exceptions, logs them, and returns a failure status
    without crashing the main script.
    """
    gene_name = os.path.basename(phy_filepath).replace('combined_', '').replace('.phy', '')
    try:
        phy_file_abs_path = os.path.abspath(phy_filepath)
        
        qc_passed, qc_message = perform_qc(phy_filepath)
        if not qc_passed:
            return {'status': 'qc_fail', 'gene': gene_name, 'reason': qc_message}

        with tempfile.TemporaryDirectory(prefix=f"{gene_name}_") as temp_dir:
            iqtree_out_prefix = os.path.join(temp_dir, gene_name)

            chimp_name = None
            with open(phy_filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        name_lower = parts[0].lower()
                        if 'pantro' in name_lower or 'pan_troglodytes' in name_lower:
                            chimp_name = parts[0]
                            break
            
            if chimp_name is None:
                raise ValueError(f"Chimp sequence name not found in {phy_filepath}")

            iqtree_cmd_list = [
                IQTREE_PATH, '-s', phy_file_abs_path, '-o', chimp_name, '-m', 'MFP',
                '-T', '1', '--prefix', iqtree_out_prefix, '-quiet'
            ]
            run_command(iqtree_cmd_list, temp_dir)
            
            tree_file = f"{iqtree_out_prefix}.treefile"
            if not os.path.exists(tree_file):
                raise FileNotFoundError(f"IQ-TREE did not produce expected treefile: {tree_file}")

            logging.info(f"Generating figure for {gene_name}...")
            generate_tree_figure(tree_file, gene_name)
            
            direct_fg_tree, inverted_fg_tree = annotate_tree_for_paml(tree_file, temp_dir, gene_name)
            
            p_direct, p_inverted = np.nan, np.nan
            if direct_fg_tree:
                p_direct = run_paml_test(phy_file_abs_path, direct_fg_tree, temp_dir)
            if inverted_fg_tree:
                p_inverted = run_paml_test(phy_file_abs_path, inverted_fg_tree, temp_dir)

            return {
                'status': 'success', 'gene': gene_name,
                'p_direct': p_direct, 'p_inverted': p_inverted
            }
    except Exception as e:
        # Catch ANY exception from this worker, log it, and return an error status.
        logging.error(f"FATAL ERROR in worker for gene '{gene_name}'. Skipping.")
        logging.error(f"REASON: {e}\n{traceback.format_exc()}")
        return {'status': 'runtime_error', 'gene': gene_name, 'reason': str(e)}

def main():
    """Main function to discover files, run the pipeline, and report results."""
    logging.info("--- Starting Full Parallel Selection Pipeline ---")

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
    
    logging.info(f"Found {len(phy_files)} CDS files to process.")
    # Use all available cores except one, but at least one.
    cpu_cores = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
    logging.info(f"Using {cpu_cores} CPU cores for parallel processing.")
    
    results = []
    # Use a multiprocessing pool to run worker_function in parallel
    with multiprocessing.Pool(processes=cpu_cores) as pool:
        with tqdm(total=len(phy_files), desc="Processing CDSs", file=sys.stdout) as pbar:
            # imap_unordered is great for getting results as they complete.
            for result in pool.imap_unordered(worker_function, phy_files):
                results.append(result)
                pbar.update(1)

    logging.info("\n--- Analysis Complete. Aggregating results... ---")
    
    successful_runs = [r for r in results if r['status'] == 'success']
    qc_failures = [r for r in results if r['status'] == 'qc_fail']
    runtime_errors = [r for r in results if r['status'] == 'runtime_error']
    
    pvals_direct, genes_direct = [], []
    valid_direct_runs = [r for r in successful_runs if 'p_direct' in r and not np.isnan(r['p_direct'])]
    if valid_direct_runs:
        genes_direct, pvals_direct = zip(*[(r['gene'], r['p_direct']) for r in valid_direct_runs])

    pvals_inverted, genes_inverted = [], []
    valid_inverted_runs = [r for r in successful_runs if 'p_inverted' in r and not np.isnan(r['p_inverted'])]
    if valid_inverted_runs:
        genes_inverted, pvals_inverted = zip(*[(r['gene'], r['p_inverted']) for r in valid_inverted_runs])

    significant_direct, significant_inverted = {}, {}
    if pvals_direct:
        rejected, qvals = fdrcorrection(list(pvals_direct), alpha=FDR_ALPHA, method='indep')
        for i, is_sig in enumerate(rejected):
            if is_sig: significant_direct[genes_direct[i]] = (pvals_direct[i], qvals[i])
            
    if pvals_inverted:
        rejected, qvals = fdrcorrection(list(pvals_inverted), alpha=FDR_ALPHA, method='indep')
        for i, is_sig in enumerate(rejected):
            if is_sig: significant_inverted[genes_inverted[i]] = (pvals_inverted[i], qvals[i])

    # --- FINAL REPORT ---
    logging.info("\n\n" + "="*50)
    logging.info("--- FINAL POSITIVE SELECTION PIPELINE REPORT ---")
    logging.info(f"Total CDSs found: {len(phy_files)}")
    logging.info(f"  - Successfully processed: {len(successful_runs)}")
    logging.info(f"  - Failed QC: {len(qc_failures)}")
    logging.info(f"  - Failed during runtime: {len(runtime_errors)}")
    logging.info("="*50 + "\n")

    logging.info(f"--- SIGNIFICANT POSITIVE SELECTION in DIRECT Haplotypes (q < {FDR_ALPHA}) ---")
    if significant_direct:
        for gene, (p, q) in sorted(significant_direct.items(), key=lambda item: item[1][1]):
            logging.info(f"  - {gene:<40} (p={p:.4g}, q={q:.4g})")
    else:
        logging.info("  None")
    logging.info("")
    
    logging.info(f"--- SIGNIFICANT POSITIVE SELECTION in INVERTED Haplotypes (q < {FDR_ALPHA}) ---")
    if significant_inverted:
        for gene, (p, q) in sorted(significant_inverted.items(), key=lambda item: item[1][1]):
            logging.info(f"  - {gene:<40} (p={p:.4g}, q={q:.4g})")
    else:
        logging.info("  None")
    logging.info("")

    if qc_failures:
        logging.warning("--- CDSs that FAILED Quality Control ---")
        for r in sorted(qc_failures, key=lambda x: x['gene']):
            logging.warning(f"  - {r['gene']:<40} Reason: {r['reason']}")
        logging.warning("")
        
    if runtime_errors:
        logging.error("--- CDSs that FAILED during RUNTIME ANALYSIS ---")
        for r in sorted(runtime_errors, key=lambda x: x['gene']):
            logging.error(f"  - {r['gene']:<40} Reason: {r['reason']}")
        logging.error("")
    
    logging.info(f"\nPipeline finished. Full details saved to '{LOG_FILE}'.")
    logging.info(f"Tree figures have been saved to the '{FIGURE_DIR}/' directory.")

if __name__ == '__main__':
    main()
