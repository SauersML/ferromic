#DRAFT

import os
import re
import sys
import glob
import subprocess
import multiprocessing
import tempfile
import shutil
import warnings
from collections import defaultdict
from ete3 import Tree, NodeStyle, TreeStyle
import numpy as np
from scipy.stats import chi2
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm

# --- Configuration ---
# Relative paths to the executable files from where this script is run.
IQTREE_PATH = './iqtree-3.0.1-Linux/bin/iqtree3'
PAML_PATH = '../paml/bin/codeml'

# QC and analysis parameters
DIVERGENCE_THRESHOLD = 0.10  # 10%
FDR_ALPHA = 0.05             # Significance level for FDR-corrected p-values

# Superpopulation colors for tree figures
POP_COLORS = {
    'AFR': '#F05031', 'EUR': '#3173F0', 'EAS': '#35A83A',
    'SAS': '#F031D3', 'AMR': '#B345F0', 'CHIMP': '#808080'
}

# --- Output Directories ---
# These will be created in the current working directory.
FIGURE_DIR = "tree_figures"
FINAL_REPORT_FILE = "final_selection_report.txt"

# --- Main Script Logic ---

def run_command(command, work_dir):
    """Executes a command in a specified directory and handles errors."""
    try:
        result = subprocess.run(
            command,
            cwd=work_dir,
            check=True,
            capture_output=True,
            text=True,
            shell=True # Using shell=True for simpler command strings
        )
        return result
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Command '{e.cmd}' failed with exit code {e.returncode}.\n"
            f"Working Directory: {work_dir}\n"
            f"--- STDOUT ---\n{e.stdout}\n"
            f"--- STDERR ---\n{e.stderr}"
        )
        raise RuntimeError(error_message) from e

def perform_qc(phy_file_path):
    """
    Performs quality control: checks for length divisibility by 3 and that
    the median human-chimp divergence is below the threshold.
    """
    with open(phy_file_path, 'r') as f:
        lines = f.readlines()
    
    header = lines[0].strip().split()
    seq_length = int(header[1])

    if seq_length % 3 != 0:
        return False, f"Sequence length {seq_length} not divisible by 3.", None

    human_seqs = [line.strip().split()[1] for line in lines[1:] if line.strip().split()[0].startswith(('0', '1'))]
    chimp_seq = None
    for line in lines[1:]:
        parts = line.strip().split()
        if 'pantro' in parts[0].lower():
            chimp_seq = parts[1]
            break
    
    if not human_seqs or not chimp_seq:
        return False, "Could not find both human and chimp sequences.", None

    divergences = []
    for human_seq in human_seqs:
        diffs = 0
        comparable_sites = 0
        for h_base, c_base in zip(human_seq, chimp_seq):
            if h_base != '-' and c_base != '-':
                comparable_sites += 1
                if h_base != c_base:
                    diffs += 1
        divergence = (diffs / comparable_sites) if comparable_sites > 0 else 0
        divergences.append(divergence)

    median_divergence = np.median(divergences)
    if median_divergence > DIVERGENCE_THRESHOLD:
        return False, f"Median divergence {median_divergence:.2%} > {DIVERGENCE_THRESHOLD:.0%}.", None

    return True, "QC Passed", seq_length

def generate_tree_figure(tree_file, gene_name):
    """Creates a styled phylogenetic tree figure using ete3."""
    try:
        t = Tree(tree_file, format=1)
        
        direct_style = NodeStyle(shape="circle", size=8)
        inverted_style = NodeStyle(shape="triangle", size=8)
        chimp_style = NodeStyle(shape="square", size=8)
        
        for leaf in t.iter_leaves():
            name = leaf.name
            pop_match = re.search(r'_(AFR|EUR|EAS|SAS|AMR)_', name)
            pop = pop_match.group(1) if pop_match else 'CHIMP'
            
            if name.startswith('0'):
                style = direct_style.copy()
            elif name.startswith('1'):
                style = inverted_style.copy()
            else:
                style = chimp_style.copy()
            
            style["fgcolor"] = POP_COLORS.get(pop, "grey")
            leaf.set_style(style)
        
        # Use TreeStyle for overall tree layout
        ts = TreeStyle()
        ts.show_leaf_name = False
        ts.scale = 10 * t.get_farthest_leaf()[1] # Auto-scale
        
        figure_path = os.path.join(FIGURE_DIR, f"{gene_name}.png")
        t.render(figure_path, w=183, units="mm", tree_style=ts)
    except Exception as e:
        return f"Figure generation failed for {gene_name}: {e}"
    return None

def annotate_tree_for_paml(tree_file, work_dir, gene_name):
    """Annotates a tree file correctly for PAML's branch-site test."""
    t = Tree(tree_file, format=1)
    
    direct_leaves = [leaf.name for leaf in t if leaf.name.startswith('0')]
    inverted_leaves = [leaf.name for leaf in t if leaf.name.startswith('1')]
    
    direct_fg_tree_path = os.path.join(work_dir, f"{gene_name}_direct_fg.tree")
    if direct_leaves:
        t_direct = t.copy()
        mrca = t_direct.get_common_ancestor(direct_leaves)
        if mrca and not mrca.is_root():
            mrca_newick = mrca.write(format=1)
            # Create a pattern including the branch length to ensure correct replacement
            pattern_to_replace = f"{mrca_newick}:{mrca.dist}"
            replacement = f"{pattern_to_replace} #1"
            
            full_tree_newick = t_direct.write(format=1)
            marked_newick = full_tree_newick.replace(pattern_to_replace, replacement)

            with open(direct_fg_tree_path, 'w') as f:
                f.write(marked_newick)
        else: # Handle case where all leaves form the root
             direct_fg_tree_path = None

    inverted_fg_tree_path = os.path.join(work_dir, f"{gene_name}_inverted_fg.tree")
    if inverted_leaves:
        t_inverted = t.copy()
        mrca = t_inverted.get_common_ancestor(inverted_leaves)
        if mrca and not mrca.is_root():
            mrca_newick = mrca.write(format=1)
            pattern_to_replace = f"{mrca_newick}:{mrca.dist}"
            replacement = f"{pattern_to_replace} #1"

            full_tree_newick = t_inverted.write(format=1)
            marked_newick = full_tree_newick.replace(pattern_to_replace, replacement)

            with open(inverted_fg_tree_path, 'w') as f:
                f.write(marked_newick)
        else:
             inverted_fg_tree_path = None

    return direct_fg_tree_path if direct_leaves else None, inverted_fg_tree_path if inverted_leaves else None

def generate_paml_ctl(ctl_path, phy_file, tree_file, out_file, is_null_model):
    """Programmatically generates a codeml.ctl file."""
    omega_val = "1.0" if is_null_model else "1.5"
    fix_omega_val = "1" if is_null_model else "0"
    
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
      fix_omega = {fix_omega_val}
      omega = {omega_val}
      cleandata = 0
    """
    with open(ctl_path, 'w') as f:
        f.write(ctl_content)

def parse_paml_lnl(outfile_path):
    """Extracts the log-likelihood (lnL) from a PAML output file robustly."""
    with open(outfile_path, 'r') as f:
        for line in f:
            if 'lnL' in line:
                match = re.search(r'lnL\(.*\):\s*([-\d\.]+)', line)
                if match:
                    return float(match.group(1))
    raise ValueError(f"Could not parse lnL from {outfile_path}")

def run_paml_test(phy_file_abs_path, annotated_tree, work_dir):
    """Runs a full PAML alt vs null test and returns the p-value."""
    base_name = os.path.basename(annotated_tree).replace('.tree', '')
    
    alt_ctl = os.path.join(work_dir, f"{base_name}_alt.ctl")
    alt_out = os.path.join(work_dir, f"{base_name}_alt.out")
    generate_paml_ctl(alt_ctl, phy_file_abs_path, annotated_tree, alt_out, is_null_model=False)
    run_command(f"{PAML_PATH} {alt_ctl}", work_dir)
    lnl_alt = parse_paml_lnl(alt_out)

    null_ctl = os.path.join(work_dir, f"{base_name}_null.ctl")
    null_out = os.path.join(work_dir, f"{base_name}_null.out")
    generate_paml_ctl(null_ctl, phy_file_abs_path, annotated_tree, null_out, is_null_model=True)
    run_command(f"{PAML_PATH} {null_ctl}", work_dir)
    lnl_null = parse_paml_lnl(null_out)

    if lnl_alt < lnl_null:
        warnings.warn(f"PAML optimization issue in {base_name}: "
                      f"Alternative model Likelihood ({lnl_alt}) is less than Null ({lnl_null}). Skipping.")
        return np.nan
        
    lr_stat = 2 * (lnl_alt - lnl_null)
    # p-value from a 50:50 mixture of chi-squared (df=0) and chi-squared (df=1)
    p_value = 0.5 * chi2.sf(lr_stat, df=1)
    
    return p_value

def worker_function(phy_filepath):
    """The main worker process that runs the entire pipeline for one CDS file."""
    gene_name = os.path.basename(phy_filepath).replace('combined_', '').replace('.phy', '')
    phy_file_abs_path = os.path.abspath(phy_filepath)
    
    try:
        qc_passed, qc_message, _ = perform_qc(phy_filepath)
        if not qc_passed:
            return {'status': 'qc_fail', 'gene': gene_name, 'reason': qc_message}

        with tempfile.TemporaryDirectory(prefix=f"{gene_name}_") as temp_dir:
            iqtree_out_prefix = os.path.join(temp_dir, gene_name)
            
            with open(phy_filepath, 'r') as f:
                chimp_name = [line.strip().split()[0] for line in f if 'pantro' in line.lower()][0]

            iqtree_cmd = (
                f"{IQTREE_PATH} -s {phy_file_abs_path} -o {chimp_name} "
                f"-m MFP -T 1 --prefix {iqtree_out_prefix}"
            )
            run_command(iqtree_cmd, temp_dir)
            
            tree_file = f"{iqtree_out_prefix}.treefile"
            if not os.path.exists(tree_file):
                raise FileNotFoundError("IQ-TREE did not produce a .treefile")

            figure_error = generate_tree_figure(tree_file, gene_name)
            if figure_error:
                warnings.warn(f"For {gene_name}: {figure_error}")
            
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
        return {'status': 'error', 'gene': gene_name, 'reason': str(e)}

def main():
    """Main function to discover files, run the pipeline, and report results."""
    print("--- Starting Full Parallel Selection Pipeline ---")

    if not (os.path.exists(IQTREE_PATH) and os.access(IQTREE_PATH, os.X_OK)):
        sys.exit(f"FATAL: IQ-TREE not found or not executable at '{IQTREE_PATH}'")
    if not (os.path.exists(PAML_PATH) and os.access(PAML_PATH, os.X_OK)):
        sys.exit(f"FATAL: PAML codeml not found or not executable at '{PAML_PATH}'")

    os.makedirs(FIGURE_DIR, exist_ok=True)

    phy_files = glob.glob('combined_*.phy')
    if not phy_files:
        sys.exit("FATAL: No 'combined_*.phy' files found in the current directory.")
    
    print(f"Found {len(phy_files)} CDS files to process.")

    results = []
    cpu_cores = os.cpu_count()
    print(f"Using {cpu_cores} CPU cores for parallel processing.")
    
    with multiprocessing.Pool(processes=cpu_cores) as pool:
        with tqdm(total=len(phy_files), desc="Processing CDSs") as pbar:
            for result in pool.imap_unordered(worker_function, phy_files):
                results.append(result)
                pbar.update(1)

    print("\n--- Analysis Complete. Aggregating results... ---")
    
    successful_runs = [r for r in results if r['status'] == 'success']
    qc_failures = [r for r in results if r['status'] == 'qc_fail']
    errors = [r for r in results if r['status'] == 'error']
    
    genes_direct, pvals_direct = zip(*[(r['gene'], r['p_direct']) for r in successful_runs if not np.isnan(r.get('p_direct'))])
    genes_inverted, pvals_inverted = zip(*[(r['gene'], r['p_inverted']) for r in successful_runs if not np.isnan(r.get('p_inverted'))])

    significant_direct, significant_inverted = {}, {}
    if pvals_direct:
        rejected, qvals = fdrcorrection(pvals_direct, alpha=FDR_ALPHA, method='indep')
        for i, is_sig in enumerate(rejected):
            if is_sig:
                significant_direct[genes_direct[i]] = (pvals_direct[i], qvals[i])
                
    if pvals_inverted:
        rejected, qvals = fdrcorrection(pvals_inverted, alpha=FDR_ALPHA, method='indep')
        for i, is_sig in enumerate(rejected):
            if is_sig:
                significant_inverted[genes_inverted[i]] = (pvals_inverted[i], qvals[i])

    with open(FINAL_REPORT_FILE, 'w') as f:
        f.write("--- FINAL POSITIVE SELECTION PIPELINE REPORT ---\n\n")
        f.write(f"Total CDSs processed: {len(phy_files)}\n")
        f.write(f"  - Successful runs: {len(successful_runs)}\n")
        f.write(f"  - Failed QC: {len(qc_failures)}\n")
        f.write(f"  - Errored during analysis: {len(errors)}\n\n")

        f.write(f"--- SIGNIFICANT POSITIVE SELECTION in DIRECT Haplotypes (q < {FDR_ALPHA}) ---\n")
        if significant_direct:
            for gene, (p, q) in sorted(significant_direct.items(), key=lambda item: item[1][1]):
                f.write(f"  - {gene:<40} (p={p:.4g}, q={q:.4g})\n")
        else:
            f.write("  None\n")
        f.write("\n")
        
        f.write(f"--- SIGNIFICANT POSITIVE SELECTION in INVERTED Haplotypes (q < {FDR_ALPHA}) ---\n")
        if significant_inverted:
            for gene, (p, q) in sorted(significant_inverted.items(), key=lambda item: item[1][1]):
                f.write(f"  - {gene:<40} (p={p:.4g}, q={q:.4g})\n")
        else:
            f.write("  None\n")
        f.write("\n")

        if qc_failures:
            f.write("--- CDSs that FAILED Quality Control ---\n")
            for r in sorted(qc_failures, key=lambda x: x['gene']):
                f.write(f"  - {r['gene']:<40} Reason: {r['reason']}\n")
            f.write("\n")
            
        if errors:
            f.write("--- CDSs that ERRORED During Analysis ---\n")
            for r in sorted(errors, key=lambda x: x['gene']):
                f.write(f"  - {r['gene']:<40} Reason: {r['reason']}\n")
            f.write("\n")

    print(f"\nPipeline finished. A detailed summary has been written to '{FINAL_REPORT_FILE}'.")
    print(f"Tree figures have been saved to the '{FIGURE_DIR}/' directory.")

if __name__ == '__main__':
    main()
