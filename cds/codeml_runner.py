import os
import sys
import logging
import multiprocessing
import glob
import traceback
import pandas as pd
import time
import re

# Ensure pipeline_lib is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import pipeline_lib as lib
except ImportError:
    print("Error: Could not import pipeline_lib.", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _execute_task(args):
    gene_info, region_label, paml_bin, target_model = args
    gene_name = gene_info['label']

    tree_file = os.path.join(lib.REGION_TREE_DIR, f"{region_label}.treefile")
    failed_marker = os.path.join(lib.REGION_TREE_DIR, f"{region_label}.FAILED")

    if os.path.exists(failed_marker):
        logging.info(f"[{gene_name}] Skipping because region {region_label} FAILED topology check.")
        return {
            'gene': gene_name,
            'region': region_label,
            'status': 'skipped_region_failed',
            'reason': 'Region topology check failed (FAILED file found)'
        }

    if not os.path.exists(tree_file):
        logging.info(f"[{gene_name}] Skipping because region tree {region_label}.treefile not found.")
        return {
            'gene': gene_name,
            'region': region_label,
            'status': 'skipped_no_tree',
            'reason': 'Region tree file missing'
        }

    try:
        return lib.analyze_single_gene(
            gene_info,
            tree_file,
            region_label,
            paml_bin,
            lib.PAML_CACHE_DIR,
            timeout=lib.PAML_TIMEOUT,
            run_branch_model=lib.RUN_BRANCH_MODEL_TEST,
            run_clade_model=lib.RUN_CLADE_MODEL_TEST,
            proceed_on_terminal_only=lib.PROCEED_ON_TERMINAL_ONLY,
            keep_paml_out=lib.KEEP_PAML_OUT,
            paml_out_dir=lib.PAML_OUT_DIR,
            make_figures=False,  # No figures for batch runs usually, or maybe yes? prompt said "And logs."
            target_clade_model=target_model,
        )
    except Exception as e:
        logging.error(f"[{gene_name}] Unexpected error: {e}")
        # Clean the error message to keep TSV rows intact
        clean_reason = str(e).replace('\n', ' | ').replace('\r', '').replace('\t', ' ')
        return {
            'gene': gene_name,
            'region': region_label,
            'status': 'runtime_error',
            'reason': clean_reason
        }

def main():
    # 1. Get Gene Batch from Env Var
    # Expected format: "GENE_A,GENE_B,GENE_C"
    batch_str = os.environ.get("GENE_BATCH")
    if not batch_str:
        logging.error("Environment variable GENE_BATCH is not set.")
        sys.exit(1)

    if batch_str == "EMPTY_BATCH":
        logging.info("Received EMPTY_BATCH. Skipping PAML run.")
        sys.exit(0)

    gene_labels = [g.strip() for g in batch_str.split(',') if g.strip()]
    target_model = os.environ.get("PAML_MODEL")
    if not target_model:
        logging.error("Environment variable PAML_MODEL is not set. Expected 'h0', 'h1', or 'both'.")
        sys.exit(1)
    target_model = target_model.lower()
    if target_model not in {"h0", "h1", "both"}:
        logging.error(f"Invalid PAML_MODEL value '{target_model}'. Expected one of ['h0', 'h1', 'both'].")
        sys.exit(1)

    logging.info(f"Starting PAML Runner for Batch: {gene_labels} | model={target_model}")

    # 2. Setup Tools
    base_dir = os.getcwd()
    _, paml_bin = lib.setup_external_tools(base_dir)

    # 3. Load Metadata & Map Genes to Regions
    # We need metadata to find the region for each gene.
    try:
        metadata = lib.load_gene_metadata()
    except Exception as e:
        logging.error(f"Failed to load metadata: {e}")
        sys.exit(1)

    # We need to find the gene files.
    # Globbing all combined_*.phy is efficient enough if not too many,
    # but we can also target specific files if we knew the exact path.
    # Since lib.parse_gene_filename parses the filename to get the label,
    # we first need to map label -> filepath.

    # Optimization: Instead of globbing 20k files, assume file naming convention matches label?
    # Label is "{gene}_{enst}". File is "combined_{gene}_{enst}.phy" usually.
    # However, parse_gene_filename handles variations.
    # Let's glob once.
    all_gene_files = [f for f in glob.glob('combined_*.phy') if 'inversion' not in os.path.basename(f)]
    label_to_path = {}
    for f in all_gene_files:
        try:
            # We don't need full parse, just enough to match label
            # But let's use library function to be safe
            info = lib.parse_gene_filename(f, metadata)
            label_to_path[info['label']] = info
        except:
            pass

    # 4. Process Genes
    results = []

    # Pre-calculate region mapping for all genes (or just look up in loop)
    # We need to know which region a gene belongs to.
    # pipeline_lib.build_region_gene_map goes Region -> [Genes]
    # We want Gene -> Region.
    # Let's reconstruct the region list to do the mapping.
    # This assumes we have the region PHY files available OR we rely on metadata.
    # Actually, metadata has chrom/start/end. We need to match that to the allowed regions list
    # or the region definitions.
    # The library logic `build_region_gene_map` iterates region_infos and checks overlap.
    # So we need region_infos.

    # We can scan for region files (they might not be downloaded in this phase?
    # wait, prompt says "downloads all artifacts from Phase 2").
    # But Phase 2 artifacts are TREES, not PHY files.
    # However, `combine_phy` (Phase 1) output `combined_inversion_*.phy`.
    # If Phase 3 downloads `combined_phy` artifacts from Phase 1, we have them.
    region_files = glob.glob('combined_inversion_*.phy')
    region_infos = []
    for rf in region_files:
        try:
            region_infos.append(lib.parse_region_filename(rf))
        except:
            pass

    # Filter by ALLOWED_REGIONS if needed
    # If REGION_OVERRIDE_FILTER is set (from GHA inputs), we bypass this check
    # to allow running a region not in the whitelist.
    override_filter = os.environ.get("REGION_OVERRIDE_FILTER")
    if override_filter:
        logging.info(f"Region override active: {override_filter}. Bypassing whitelist.")
        # We assume the matrix generation step has already ensured we are only running
        # relevant things, or we let everything pass here and rely on the fact that
        # generate_gha_matrix only scheduled the specific region.
        # However, we should probably filter `region_infos` to ONLY contain the override region
        # just to be safe and avoid accidentally running other regions if they exist on disk.

        # Parse override string manually or matching label?
        # Easier to just let it pass all found regions, because generate_gha_matrix
        # ensures only the specific region is in the job matrix if we were running per-region.
        # But here we are running per-GENE-BATCH.
        # A gene might overlap multiple regions.
        # If we override for Region A, we only want to run Region A analysis for these genes.

        # So, let's parse the override and filter `region_infos` to match ONLY that region.
        try:
            m = re.match(r"^(chr[0-9a-zA-Z]+):(\d+)-(\d+)$", override_filter)
            if m:
                o_chrom, o_start, o_end = m.groups()
                o_start, o_end = int(o_start), int(o_end)
                if o_start > o_end: o_start, o_end = o_end, o_start

                region_infos = [
                    r for r in region_infos
                    if r['chrom'] == o_chrom and r['start'] == o_start and r['end'] == o_end
                ]
                logging.info(f"Filtered region_infos to override target: {len(region_infos)} regions kept.")
        except Exception as e:
            logging.error(f"Failed to parse override filter '{override_filter}': {e}. Aborting to avoid running everything.")
            sys.exit(1)

    elif lib.ALLOWED_REGIONS:
        allowed_set = set(lib.ALLOWED_REGIONS)
        region_infos = [r for r in region_infos if (r['chrom'], r['start'], r['end']) in allowed_set]

    # Build map
    # Note: A gene might overlap multiple regions? usually just one.
    # The library `build_region_gene_map` creates a dict: region_label -> [gene_infos]
    # Let's reverse it.
    gene_to_region_map = {}

    # We need 'gene_infos' for the batch
    batch_gene_infos = []
    for label in gene_labels:
        if label in label_to_path:
            batch_gene_infos.append(label_to_path[label])
        else:
            logging.warning(f"File for gene {label} not found.")

    # Run mapping
    r_map = lib.build_region_gene_map(region_infos, batch_gene_infos)
    # Invert r_map: region -> [genes]  =>  gene_label -> region_label
    # Note: if a gene is in multiple regions, we might run it multiple times?
    # The original omega_test.py iterates regions, then genes.
    # So if a gene is in Region A and Region B, it runs for both.

    tasks = []
    for region_label, genes in r_map.items():
        for g in genes:
            tasks.append((g, region_label))

    logging.info(f"Mapped {len(gene_labels)} genes to {len(tasks)} gene-region pairs.")

    # 5. Execute sequentially to allow per-gene restart parallelism inside pipeline_lib
    for gene_info, region_label in tasks:
        gene_name = gene_info['label']
        try:
            res = _execute_task((gene_info, region_label, paml_bin, target_model))
            results.append(res)
        except Exception as e:
            logging.error(f"[{gene_name}] Worker crashed: {e}")
            clean_reason = str(e).replace('\n', ' | ').replace('\r', '').replace('\t', ' ')
            results.append({
                'gene': gene_name,
                'region': region_label,
                'status': 'runtime_error',
                'reason': f'Worker failure: {clean_reason}'
            })

    # 6. Save Results
    if results:
        df = pd.DataFrame(results)
        # Sanitize batch name for filename
        # Use the first gene name to ensure uniqueness across parallel batches
        if gene_labels:
            safe_batch_label = gene_labels[0].replace(".", "_")
        else:
            safe_batch_label = "batch"

        safe_timestamp = str(int(time.time() * 1000))
        out_name = f"partial_results_{safe_batch_label}_{safe_timestamp}.tsv"

        df.to_csv(out_name, sep='\t', index=False)
        logging.info(f"Saved {len(results)} results to {out_name}")
    else:
        logging.warning("No results generated for this batch.")

if __name__ == "__main__":
    # FIX: Force 'spawn' to prevent deadlocks with ete3/Qt and logging locks
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Context might already be set

    main()
