import threading
import sys
from functools import partial
from multiprocessing import get_context, cpu_count
import os
import json
import math

import models
import time
import random
import queue

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

MP_CONTEXT = 'fork' if sys.platform == 'linux' else 'spawn'

class MemoryMonitor(threading.Thread):
    def __init__(self, interval=1):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.available_memory_gb = 0
        self.rss_gb = 0

    def run(self):
        while not self.stop_event.is_set():
            if PSUTIL_AVAILABLE:
                self.available_memory_gb = psutil.virtual_memory().available / (1024**3)
                self.rss_gb = psutil.Process().memory_info().rss / (1024**3)
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()


def run_fits(pheno_queue, core_df_with_const, allowed_mask_by_cat, target_inversion, results_cache_dir, ctx, min_available_memory_gb):
    """
    Creates the process pool with models.init_worker and submits models.run_single_model_worker
    with the same callback/progress bar logic.
    """
    worker_func = partial(
        models.run_single_model_worker, target_inversion=target_inversion, results_cache_dir=results_cache_dir
    )

    monitor = MemoryMonitor()
    monitor.start()
    try:
        print(f"\n--- Starting parallel model fitting with {cpu_count()} worker processes ({MP_CONTEXT} context) ---")
        with get_context(MP_CONTEXT).Pool(
            processes=max(1, min(cpu_count(), 8)),
            initializer=models.init_worker,
            initargs=(core_df_with_const, allowed_mask_by_cat, ctx),
            maxtasksperchild=50,
        ) as pool:
            bar_len = 40
            queued = 0
            done = 0
            lock = threading.Lock()

            def _print_bar(q, d):
                q = int(q)
                d = int(d)
                pct = int((d * 100) / q) if q else 0
                filled = int(bar_len * (d / q)) if q else 0
                bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
                mem_info = f"| Mem RSS: {monitor.rss_gb:.2f}GB Avail: {monitor.available_memory_gb:.2f}GB" if PSUTIL_AVAILABLE else ""
                print(f"\r[Fit] {bar} {d}/{q} ({pct}%) {mem_info}", end="", flush=True)

            def _cb(_):
                nonlocal done, queued
                with lock:
                    done += 1
                    _print_bar(queued, done)

            failed_tasks = []
            def _err_cb(e):
                nonlocal failed_tasks
                print(f"[pool ERR] Worker failed: {e}", flush=True)
                failed_tasks.append(e)

            # Consume items until sentinel is received
            while True:
                item = pheno_queue.get()
                if item is None:
                    break  # producer finished
                if PSUTIL_AVAILABLE and 0 < monitor.available_memory_gb < min_available_memory_gb:
                    print(f"\n[gov WARN] Low memory detected (avail: {monitor.available_memory_gb:.2f}GB), pausing task submission...", flush=True)
                    while PSUTIL_AVAILABLE and 0 < monitor.available_memory_gb < min_available_memory_gb:
                        time.sleep(2)
                # Cache policy: if a previous result exists but has an invalid or NA P_Value and the
                # association was attempted (no Skip_Reason), evict the meta to force a fresh run.
                try:
                    res_path = os.path.join(results_cache_dir, f"{item['name']}.json")
                    meta_path = os.path.join(results_cache_dir, f"{item['name']}.meta.json")
                    if os.path.exists(res_path) and os.path.exists(meta_path):
                        with open(res_path, "r") as _rf:
                            _res_obj = json.load(_rf)
                        _skip_reason = _res_obj.get("Skip_Reason", None)
                        if not _skip_reason:
                            _pv = _res_obj.get("P_Value", None)
                            _valid_p = False
                            try:
                                _pvf = float(_pv)
                                _valid_p = math.isfinite(_pvf) and (0.0 < _pvf < 1.0)
                            except Exception:
                                _valid_p = False
                            if not _valid_p:
                                try:
                                    os.remove(meta_path)
                                    print(f"\n[cache POLICY] Invalid or missing P_Value for '{item['name']}'. Forcing re-run by removing meta.", flush=True)
                                except Exception:
                                    pass
                except Exception:
                    pass
                queued += 1
                pool.apply_async(worker_func, (item,), callback=_cb, error_callback=_err_cb)
                _print_bar(queued, done)

            pool.close()
            pool.join()  # callbacks keep updating the bar while we wait
            _print_bar(queued, done)  # ensure 100% line
            print("")  # newline after the bar
    finally:
        monitor.stop()


def run_lrt_overall(core_df_with_const, allowed_mask_by_cat, phenos_list, name_to_cat, cdr_codename, target_inversion, ctx, min_available_memory_gb):
    """
    Same pool pattern; submits models.lrt_overall_worker.
    """
    tasks = [{"name": s, "category": name_to_cat.get(s, None), "cdr_codename": cdr_codename, "target": target_inversion} for s in phenos_list]
    random.shuffle(tasks)

    monitor = MemoryMonitor()
    monitor.start()
    try:
        print(f"[LRT-Stage1] Scheduling {len(tasks)} phenotypes for overall LRT with atomic caching.", flush=True)
        bar_len = 40
        queued = 0
        done = 0
        lock = threading.Lock()

        def _print_bar(q, d, label):
            q = int(q)
            d = int(d)
            pct = int((d * 100) / q) if q else 0
            filled = int(bar_len * (d / q)) if q else 0
            bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
            mem_info = f"| Mem RSS: {monitor.rss_gb:.2f}GB Avail: {monitor.available_memory_gb:.2f}GB" if PSUTIL_AVAILABLE else ""
            print(f"\r[{label}] {bar} {d}/{q} ({pct}%) {mem_info}", end="", flush=True)

        with get_context(MP_CONTEXT).Pool(
            processes=max(1, min(cpu_count(), 8)),
            initializer=models.init_worker,
            initargs=(core_df_with_const, allowed_mask_by_cat, ctx),
            maxtasksperchild=50,
        ) as pool:

            def _cb(_):
                nonlocal done, queued
                with lock:
                    done += 1
                    _print_bar(queued, done, "LRT-Stage1")

            failed_tasks = []
            def _err_cb(e):
                nonlocal failed_tasks
                print(f"[pool ERR] Worker failed: {e}", flush=True)
                failed_tasks.append(e)

            for task in tasks:
                if PSUTIL_AVAILABLE and 0 < monitor.available_memory_gb < min_available_memory_gb:
                    print(f"\n[gov WARN] Low memory detected (avail: {monitor.available_memory_gb:.2f}GB), pausing task submission...", flush=True)
                    while PSUTIL_AVAILABLE and 0 < monitor.available_memory_gb < min_available_memory_gb:
                        time.sleep(2)

                # Cache policy: if a previous Stage-1 LRT result exists but has an invalid or NA P_LRT_Overall,
                # evict the meta to force a fresh run. LRT tasks are only scheduled for non-skipped models.
                try:
                    _res_path = os.path.join(ctx["LRT_OVERALL_CACHE_DIR"], f"{task['name']}.json")
                    _meta_path = os.path.join(ctx["LRT_OVERALL_CACHE_DIR"], f"{task['name']}.meta.json")
                    if os.path.exists(_res_path) and os.path.exists(_meta_path):
                        with open(_res_path, "r") as _rf:
                            _res_obj = json.load(_rf)
                        _p = _res_obj.get("P_LRT_Overall", None)
                        _valid = False
                        try:
                            _pf = float(_p)
                            _valid = math.isfinite(_pf) and (0.0 < _pf < 1.0)
                        except Exception:
                            _valid = False
                        if not _valid:
                            try:
                                os.remove(_meta_path)
                                print(f"\n[cache POLICY] Invalid or missing P_LRT_Overall for '{task['name']}'. Forcing re-run by removing meta.", flush=True)
                            except Exception:
                                pass
                except Exception:
                    pass

                queued += 1
                pool.apply_async(models.lrt_overall_worker, (task,), callback=_cb, error_callback=_err_cb)
                _print_bar(queued, done, "LRT-Stage1")

            pool.close()
            pool.join()
            _print_bar(queued, done, "LRT-Stage1")
            print("")
    finally:
        monitor.stop()


def run_lrt_followup(core_df_with_const, allowed_mask_by_cat, anc_series, hit_names, name_to_cat, cdr_codename, target_inversion, ctx, min_available_memory_gb):
    if len(hit_names) > 0:
        tasks_follow = [{"name": s, "category": name_to_cat.get(s, None), "cdr_codename": cdr_codename, "target": target_inversion} for s in hit_names]
        random.shuffle(tasks_follow)
        print(f"[Ancestry] Scheduling follow-up for {len(tasks_follow)} FDR-significant phenotypes.", flush=True)

        # NEW: start monitor (symmetry with Stage-1)
        monitor = MemoryMonitor()
        monitor.start()
        try:
            bar_len = 40
            queued = 0
            done = 0
            lock = threading.Lock()

            def _print_bar(q, d, label):
                q = int(q); d = int(d)
                pct = int((d * 100) / q) if q else 0
                filled = int(bar_len * (d / q)) if q else 0
                bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
                # NEW: show memory info like Stage-1
                mem_info = ""
                if PSUTIL_AVAILABLE:
                    mem_info = f" | Mem RSS: {monitor.rss_gb:.2f}GB Avail: {monitor.available_memory_gb:.2f}GB"
                print(f"\r[{label}] {bar} {d}/{q} ({pct}%)" + mem_info, end="", flush=True)

            with get_context(MP_CONTEXT).Pool(
                processes=max(1, min(cpu_count(), 8)),
                initializer=models.init_lrt_worker,
                initargs=(core_df_with_const, allowed_mask_by_cat, anc_series, ctx),
                maxtasksperchild=50,
            ) as pool:
                def _cb2(_):
                    nonlocal done, queued
                    with lock:
                        done += 1
                        _print_bar(queued, done, "Ancestry")

                failed_tasks = []
                def _err_cb(e):
                    nonlocal failed_tasks
                    print(f"[pool ERR] Worker failed: {e}", flush=True)
                    failed_tasks.append(e)

                for task in tasks_follow:
                    if PSUTIL_AVAILABLE and 0 < monitor.available_memory_gb < min_available_memory_gb:
                        print(f"\n[gov WARN] Low memory detected (avail: {monitor.available_memory_gb:.2f}GB), pausing task submission...", flush=True)
                        while PSUTIL_AVAILABLE and 0 < monitor.available_memory_gb < min_available_memory_gb:
                            time.sleep(2)

                    queued += 1
                    pool.apply_async(models.lrt_followup_worker, (task,), callback=_cb2, error_callback=_err_cb)
                    _print_bar(queued, done, "Ancestry")

                pool.close()
                pool.join()
                _print_bar(queued, done, "Ancestry")
                print("")
        finally:
            monitor.stop()
