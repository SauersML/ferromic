import os
import sys
import glob
import multiprocessing
import threading
import time
import logging
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
import subprocess
import pandas as pd
import numpy as np
from logging.handlers import QueueHandler, QueueListener
from tqdm import tqdm

# Add current directory to path to find pipeline_lib if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import pipeline_lib as lib
except ImportError:
    print("Error: Could not import pipeline_lib. Ensure it is in the same directory.")
    sys.exit(1)

# --- Configuration ---

def _detect_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1

CPU_COUNT = _detect_cpus()
REGION_WORKERS = int(os.environ.get("REGION_WORKERS", max(1, min(CPU_COUNT // 3, 4))))
default_paml = max(1, CPU_COUNT - REGION_WORKERS)
if CPU_COUNT >= 4:
    default_paml = max(2, default_paml)
PAML_WORKERS = int(os.environ.get("PAML_WORKERS", default_paml))

_REQUESTED_FIGURES = bool(int(os.environ.get("MAKE_FIGURES", "1")))

LOG_FILE = f"pipeline_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
RESULTS_TSV = f"full_paml_results_{datetime.now().strftime('%Y-%m-%d')}.tsv"

# --- Logging ---

def start_logging():
    log_q = multiprocessing.Queue(-1)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    listener = QueueListener(log_q, file_handler, stream_handler)
    listener.start()
    return log_q, listener

def worker_logging_init(log_q):
    root = logging.getLogger()
    root.handlers[:] = [QueueHandler(log_q)]
    root.setLevel(logging.INFO)

# --- Monitoring ---

def _get_procfs_cpu_times():
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        parts = line.split()
        return tuple(map(int, parts[1:5]))
    except (IOError, IndexError, ValueError):
        return (0, 0, 0, 0)

_prev_cpu_times = None
_prev_cpu_time_ts = None

def _get_cpu_usage_procfs():
    global _prev_cpu_times, _prev_cpu_time_ts
    if _prev_cpu_times is None:
        _prev_cpu_times = _get_procfs_cpu_times()
        _prev_cpu_time_ts = time.time()
        time.sleep(1)
    now = time.time()
    current_times = _get_procfs_cpu_times()
    delta_times = tuple(c - p for c, p in zip(current_times, _prev_cpu_times))
    delta_ts = now - _prev_cpu_time_ts
    _prev_cpu_times = current_times
    _prev_cpu_time_ts = now
    if delta_ts == 0: return 0.0
    total_time = sum(delta_times)
    idle_time = delta_times[3]
    return 100.0 * (total_time - idle_time) / total_time if total_time else 0.0

def _get_mem_info_procfs():
    mem_total, mem_avail = 0, 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_avail = int(line.split()[1])
                    break
    except (IOError, IndexError, ValueError): pass
    return mem_total, mem_avail

def _get_load_avg_procfs():
    try:
        with open("/proc/loadavg") as f:
            return float(f.readline().split()[0])
    except (IOError, IndexError, ValueError): return 0.0

def _get_process_counts_procfs():
    iqtree_count = 0
    codeml_count = 0
    try:
        for pid in os.listdir('/proc'):
            if not pid.isdigit(): continue
            try:
                with open(f'/proc/{pid}/cmdline', 'rb') as f:
                    cmdline = f.read().split(b'\x00')
                if not cmdline: continue
                exe_path = os.path.basename(cmdline[0].decode('utf-8', 'ignore'))
                if 'iqtree3' in exe_path: iqtree_count += 1
                elif 'codeml' in exe_path: codeml_count += 1
            except (IOError, UnicodeDecodeError): continue
    except IOError: pass
    return iqtree_count, codeml_count

def monitor_thread(status_dict, stop_event, interval=12):
    logging.info("MONITOR: Starting utilization monitor thread.")
    _get_cpu_usage_procfs()
    while not stop_event.is_set():
        try:
            cpu_pct = _get_cpu_usage_procfs()
            mem_total_kb, mem_avail_kb = _get_mem_info_procfs()
            mem_pct = 100.0 * (mem_total_kb - mem_avail_kb) / mem_total_kb if mem_total_kb else 0.0
            load_avg = _get_load_avg_procfs()
            iqtree_pids, codeml_pids = _get_process_counts_procfs()
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
            logging.error(f"MONITOR: Error in monitor thread: {e}")
        wait_time = 0
        while wait_time < interval and not stop_event.is_set():
            time.sleep(1)
            wait_time += 1
    logging.info("MONITOR: Stopping utilization monitor thread.")


# --- Wrappers for ProcessPoolExecutor ---

def _run_iqtree(r, bin, th):
    return lib.run_iqtree_task(r, bin, th, lib.REGION_TREE_DIR, lib.IQTREE_TIMEOUT, make_figures=_REQUESTED_FIGURES)

def _run_codeml(g, t, l, bin):
    return lib.analyze_single_gene(
        g, t, l, bin, lib.PAML_CACHE_DIR,
        timeout=lib.PAML_TIMEOUT,
        run_branch_model=lib.RUN_BRANCH_MODEL_TEST,
        run_clade_model=lib.RUN_CLADE_MODEL_TEST,
        proceed_on_terminal_only=lib.PROCEED_ON_TERMINAL_ONLY,
        keep_paml_out=lib.KEEP_PAML_OUT,
        paml_out_dir=lib.PAML_OUT_DIR,
        make_figures=_REQUESTED_FIGURES
    )

# --- Main Pipeline Logic ---

def submit_with_cap(exec, fn, args, inflight, cap):
    fut = exec.submit(fn, *args)
    inflight.append(fut)
    flushed = []
    if len(inflight) >= cap:
        done = next(as_completed(inflight))
        inflight.remove(done)
        flushed.append(done)
    return fut, flushed

def run_overlapped(region_infos, region_gene_map, log_q, status_dict, iqtree_bin, paml_bin):
    all_results = []
    inflight = deque()
    future_args = {}
    cap = PAML_WORKERS * 4 if PAML_WORKERS > 0 else 1
    completed_count = 0

    status_dict['regions_total'] = len(region_infos)
    status_dict['regions_done'] = 0
    total_paml_jobs = sum(len(genes) for r_label, genes in region_gene_map.items() if r_label in {r['label'] for r in region_infos})
    status_dict['paml_total'] = total_paml_jobs
    status_dict['paml_done'] = 0
    paml_start_time = None

    mpctx = multiprocessing.get_context("spawn")

    def record_result(res):
        nonlocal completed_count, paml_start_time
        all_results.append(res)
        completed_count += 1
        status_dict['paml_done'] = completed_count
        if paml_start_time and completed_count > 2:
            elapsed = time.time() - paml_start_time
            rate = completed_count / elapsed if elapsed > 0 else 0
            if rate > 0:
                remaining = max(total_paml_jobs - completed_count, 0)
                eta_s = remaining / rate if rate else 0
                status_dict['eta_str'] = f"{int(eta_s // 60)}m{int(eta_s % 60)}s"
        if (completed_count % 25 == 0) or (res.get('status') != 'success'):
            logging.info(f"Completed {completed_count}/{total_paml_jobs}: {res.get('gene')} in {res.get('region')} -> {res.get('status')}")
        if lib.CHECKPOINT_EVERY and completed_count % lib.CHECKPOINT_EVERY == 0:
            logging.info(f"--- Checkpointing {len(all_results)} results to {lib.CHECKPOINT_FILE} ---")
            pd.DataFrame(all_results).to_csv(lib.CHECKPOINT_FILE, sep="\t", index=False, float_format='%.6g')

    with ProcessPoolExecutor(max_workers=PAML_WORKERS or 1, mp_context=mpctx, initializer=worker_logging_init, initargs=(log_q,)) as paml_exec, \
         ProcessPoolExecutor(max_workers=REGION_WORKERS or 1, mp_context=mpctx, initializer=worker_logging_init, initargs=(log_q,)) as region_exec:

        paml_pool_alive = PAML_WORKERS > 0
        iqtree_threads = max(1, CPU_COUNT // (REGION_WORKERS or 1))
        logging.info(f"Submitting {len(region_infos)} region tasks to pool (using {iqtree_threads} threads per job)...")

        region_futs = {region_exec.submit(_run_iqtree, r, iqtree_bin, iqtree_threads) for r in region_infos}
        region_pbar = tqdm(as_completed(region_futs), total=len(region_futs), desc="Processing regions")

        for region_future in region_pbar:
            status_dict['regions_done'] += 1
            try:
                label, tree, reason = region_future.result()
            except Exception as e:
                logging.error(f"A region task failed: {e}")
                continue

            if tree is None:
                logging.warning(f"Region {label} skipped: {reason}")
                genes_for_failed_region = region_gene_map.get(label, [])
                total_paml_jobs -= len(genes_for_failed_region)
                status_dict['paml_total'] = total_paml_jobs
                continue
            
            genes_for_region = region_gene_map.get(label, [])
            if not genes_for_region: continue

            if paml_start_time is None: paml_start_time = time.time()

            logging.info(f"Region {label} complete. Submitting {len(genes_for_region)} PAML jobs.")
            for gene_info in genes_for_region:
                if paml_pool_alive:
                    try:
                        future, flushed = submit_with_cap(paml_exec, _run_codeml, (gene_info, tree, label, paml_bin), inflight, cap)
                        future_args[future] = (gene_info, tree, label, paml_bin)
                        status_dict['paml_running'] = len(inflight)
                    except BrokenProcessPool as pool_exc:
                        logging.critical(f"PAML worker pool crashed ({pool_exc}); switching to in-process.")
                        paml_pool_alive = False
                        status_dict['paml_running'] = 0
                        while inflight:
                            fut = inflight.popleft()
                            args = future_args.pop(fut, None)
                            try:
                                res = fut.result()
                                record_result(res)
                            except Exception:
                                if args: record_result(_run_codeml(*args))
                        flushed = []
                        record_result(_run_codeml(gene_info, tree, label, paml_bin))
                        continue

                    for paml_future in flushed:
                        args = future_args.pop(paml_future, None)
                        try:
                            res = paml_future.result()
                            record_result(res)
                        except Exception as e:
                            logging.error(f"PAML job failed: {e}")
                            if isinstance(e, BrokenProcessPool):
                                paml_pool_alive = False
                                status_dict['paml_running'] = 0
                            if args: record_result(_run_codeml(*args))
                    status_dict['paml_running'] = len(inflight)
                else:
                    record_result(_run_codeml(gene_info, tree, label, paml_bin))
                    status_dict['paml_running'] = 0

        if paml_pool_alive:
            logging.info(f"Draining {len(inflight)} remaining PAML jobs...")
            pending = list(inflight)
            paml_pbar = tqdm(as_completed(pending), total=len(pending), desc="Finalizing PAML jobs")
            for paml_future in paml_pbar:
                if paml_future in inflight: inflight.remove(paml_future)
                status_dict['paml_running'] = len(inflight)
                args = future_args.pop(paml_future, None)
                try:
                    res = paml_future.result()
                    record_result(res)
                except Exception as e:
                    logging.error(f"PAML job failed during drain: {e}")
                    if args: record_result(_run_codeml(*args))
            inflight.clear()
            future_args.clear()
            status_dict['paml_running'] = 0

    if all_results:
        logging.info(f"--- Final checkpoint of {len(all_results)} results to {lib.CHECKPOINT_FILE} ---")
        pd.DataFrame(all_results).to_csv(lib.CHECKPOINT_FILE, sep="\t", index=False, float_format='%.6g')
        
    return all_results

def main():
    log_q, listener = start_logging()
    root = logging.getLogger()
    root.handlers[:] = [QueueHandler(log_q)]
    root.setLevel(logging.INFO)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    iqtree_bin, paml_bin = lib.setup_external_tools(base_dir)

    status_dict = {}
    stop_event = threading.Event()
    mon_thread = threading.Thread(target=monitor_thread, args=(status_dict, stop_event))
    mon_thread.daemon = True
    mon_thread.start()

    try:
        logging.info("--- Starting Region→Gene Differential Selection Pipeline ---")
        if lib.TREEVIEW_IMPORT_ERROR:
            logging.warning(f"Tree figure rendering disabled: {lib.TREEVIEW_IMPORT_ERROR}")

        os.makedirs(lib.FIGURE_DIR, exist_ok=True)
        os.makedirs(lib.ANNOTATED_FIGURE_DIR, exist_ok=True)
        os.makedirs(lib.REGION_TREE_DIR, exist_ok=True)

        logging.info("Searching for files...")
        region_files = glob.glob('combined_inversion_*.phy')
        gene_files = [f for f in glob.glob('combined_*.phy') if 'inversion' not in os.path.basename(f)]
        logging.info(f"Found {len(region_files)} region alignments and {len(gene_files)} gene alignments")

        if not region_files or not gene_files:
            logging.critical("FATAL: Missing alignment files.")
            sys.exit(1)

        logging.info("Loading metadata...")
        metadata = lib.load_gene_metadata()

        logging.info("Parsing filenames...")
        region_infos = []
        for f in region_files:
            try: region_infos.append(lib.parse_region_filename(f))
            except Exception: pass

        if len(lib.ALLOWED_REGIONS) > 0:
            allowed_set = set(lib.ALLOWED_REGIONS)
            region_infos = [r for r in region_infos if (r['chrom'], r['start'], r['end']) in allowed_set]

        gene_infos = []
        for f in gene_files:
            try: gene_infos.append(lib.parse_gene_filename(f, metadata))
            except Exception: pass

        logging.info("Mapping genes...")
        region_gene_map = lib.build_region_gene_map(region_infos, gene_infos)

        all_results = run_overlapped(region_infos, region_gene_map, log_q, status_dict, iqtree_bin, paml_bin)
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

        if results_df.empty:
            results_df = pd.DataFrame(columns=ordered_columns)
        else:
            results_df = lib.compute_fdr(results_df)

        remaining_cols = [c for c in results_df.columns if c not in ordered_columns]
        ordered_with_dynamic = ordered_columns + sorted(remaining_cols)
        results_df = results_df[ordered_with_dynamic]
        results_df.to_csv(RESULTS_TSV, sep='\t', index=False, float_format='%.6g')
        logging.info(f"Results saved to {RESULTS_TSV}")

        counts = results_df['status'].value_counts().to_dict()
        logging.info("Status counts: " + str(counts))

        sig = results_df[(results_df['status'] == 'success') & ((results_df['bm_q_value'] < lib.FDR_ALPHA) | (results_df['cmc_q_value'] < lib.FDR_ALPHA))]
        if not sig.empty:
             logging.info(f"Significant tests: {len(sig)}")
        else:
             logging.info("No significant tests.")

    finally:
        stop_event.set()
        mon_thread.join(timeout=2.0)
        listener.stop()

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()
