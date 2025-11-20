import os
import sys
import logging
import glob
import traceback

# Ensure pipeline_lib is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import pipeline_lib as lib
except ImportError:
    print("Error: Could not import pipeline_lib.", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 1. Get Target Region from Env Var
    target_region = os.environ.get("TARGET_REGION")
    if not target_region:
        logging.error("Environment variable TARGET_REGION is not set.")
        sys.exit(1)

    logging.info(f"Starting IQ-TREE Runner for Region: {target_region}")

    # 2. Setup Tools (Expect them to be present/unzipped)
    base_dir = os.getcwd()
    # setup_external_tools will set permissions if files exist
    iqtree_bin, _ = lib.setup_external_tools(base_dir)

    # 3. Locate the .phy file for this region
    # The label is typically "chr_start_end" or similar.
    # We need to find the file that matches this label.
    # pipeline_lib.parse_region_filename logic:
    # label: f"{chrom}_{start}_{end}"

    # We will scan files and match the label.
    region_files = glob.glob('combined_inversion_*.phy')
    target_file = None
    target_info = None

    for f in region_files:
        try:
            info = lib.parse_region_filename(f)
            if info['label'] == target_region:
                target_file = f
                target_info = info
                break
        except Exception:
            continue

    if not target_file:
        logging.error(f"Could not find combined PHYLIP file for region identifier: {target_region}")
        sys.exit(1)

    # 4. Run IQ-TREE
    # Determine CPU count for this runner
    try:
        threads = len(os.sched_getaffinity(0))
    except AttributeError:
        threads = os.cpu_count() or 1

    # Ensure output directory exists
    os.makedirs(lib.REGION_TREE_DIR, exist_ok=True)

    logging.info(f"Running IQ-TREE on {target_file} with {threads} threads...")

    try:
        label, tree_path, error = lib.run_iqtree_task(
            target_info,
            iqtree_bin,
            threads,
            lib.REGION_TREE_DIR,
            timeout=lib.IQTREE_TIMEOUT,
            make_figures=False # Figures not strictly needed for pipeline success, can be enabled if desired
        )

        if error:
            logging.error(f"IQ-TREE Failed: {error}")
            # We explicitly mark this as a failure.
            # Phase 3 will handle missing trees by skipping genes.
            # We might want to write a "FAILED" placeholder file so Phase 3 knows it failed vs just missing?
            # The prompt says: "Action: The runner does not upload a .treefile for Region X (or uploads a tiny file named Region_X.FAILED)."
            fail_marker = os.path.join(lib.REGION_TREE_DIR, f"{target_region}.FAILED")
            with open(fail_marker, "w") as f:
                f.write(f"Reason: {error}")
            sys.exit(0) # Exit 0 so GHA job doesn't turn red? Or exit 1?
            # User said "The fact that IQ-TREE might reject a region does not mean you have to change the scheduling strategy."
            # Phase 3 handles it. So let's exit 0 but leave the FAILED file.

        if tree_path and os.path.exists(tree_path):
            logging.info(f"Success. Tree file generated: {tree_path}")
            # The artifact uploader in GHA will look for this file.
        else:
            logging.warning("IQ-TREE finished but no tree file found (and no error returned?).")
            fail_marker = os.path.join(lib.REGION_TREE_DIR, f"{target_region}.FAILED")
            with open(fail_marker, "w") as f:
                f.write("Reason: Unknown (no tree file)")

    except Exception as e:
        logging.error(f"Exception during IQ-TREE execution: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
