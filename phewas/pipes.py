import threading
from functools import partial
from multiprocessing import get_context, cpu_count

from phewas import models


def run_fits(pheno_queue, core_df_with_const, allowed_mask_by_cat, target_inversion, results_cache_dir, ctx):
    """
    Creates the process pool with models.init_worker and submits models.run_single_model_worker
    with the same callback/progress bar logic.
    """
    worker_func = partial(
        models.run_single_model_worker, target_inversion=target_inversion, results_cache_dir=results_cache_dir
    )

    print(f"\n--- Starting parallel model fitting with {cpu_count()} worker processes ---")
    with get_context('spawn').Pool(
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
            print(f"\r[Fit] {bar} {d}/{q} ({pct}%)", end="", flush=True)

        def _cb(_):
            nonlocal done, queued
            with lock:
                done += 1
                _print_bar(queued, done)

        # Drain queue → submit jobs with a completion callback
        while True:
            pheno_data = pheno_queue.get()
            if pheno_data is None:
                break
            queued += 1
            pool.apply_async(worker_func, (pheno_data,), callback=_cb)
            _print_bar(queued, done)  # show progress while queuing too

        pool.close()
        pool.join()  # callbacks keep updating the bar while we wait
        _print_bar(queued, done)  # ensure 100% line
        print("")  # newline after the bar


def run_lrt_overall(core_df_with_const, allowed_mask_by_cat, phenos_list, name_to_cat, cdr_codename, target_inversion, ctx):
    """
    Same pool pattern; submits models.lrt_overall_worker.
    """
    tasks = [{"name": s, "category": name_to_cat.get(s, None), "cdr_codename": cdr_codename, "target": target_inversion} for s in phenos_list]

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
        print(f"\r[{label}] {bar} {d}/{q} ({pct}%)", end="", flush=True)

    with get_context('spawn').Pool(
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

        for task in tasks:
            queued += 1
            pool.apply_async(models.lrt_overall_worker, (task,), callback=_cb)
            _print_bar(queued, done, "LRT-Stage1")
        pool.close()
        pool.join()
        _print_bar(queued, done, "LRT-Stage1")
        print("")


def run_lrt_followup(core_df_with_const, allowed_mask_by_cat, anc_series, hit_names, name_to_cat, cdr_codename, target_inversion, ctx):
    """
    Same pool pattern; initializer is models.init_lrt_worker(...); submits models.lrt_followup_worker.
    """
    if len(hit_names) > 0:
        tasks_follow = [{"name": s, "category": name_to_cat.get(s, None), "cdr_codename": cdr_codename, "target": target_inversion} for s in hit_names]
        print(f"[Ancestry] Scheduling follow-up for {len(tasks_follow)} FDR-significant phenotypes.", flush=True)

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
            print(f"\r[{label}] {bar} {d}/{q} ({pct}%)", end="", flush=True)

        with get_context('spawn').Pool(
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

            for task in tasks_follow:
                queued += 1
                pool.apply_async(models.lrt_followup_worker, (task,), callback=_cb2)
                _print_bar(queued, done, "Ancestry")
            pool.close()
            pool.join()
            _print_bar(queued, done, "Ancestry")
            print("")
