import os
import json
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from . import models
from . import iox as io

DEFAULTS = {
    "MODE": "lrt_bh",
    "SELECTION": "lrt_bh",
    "BOOTSTRAP_B": 1000, # not nearly enough for many tests... lrt_bh is better unless there is a ton of compute
    "BOOT_SEED_BASE": 2025,
    "MAX_CONCURRENT_INVERSIONS_DEFAULT": 8,
    "MAX_CONCURRENT_INVERSIONS_BOOT": 8,
}


def get_testing_ctx(overrides=None):
    cfg = DEFAULTS.copy()
    if overrides:
        cfg.update(overrides)
    return cfg


def run_overall(core_df_with_const, allowed_mask_by_cat, anc_series,
                phenos_list, name_to_cat, cdr_codename, target_inversion, ctx,
                min_available_memory_gb, on_pool_started=None, mode=None):
    """Dispatch Stage-1 tests based on mode."""
    mode = (mode or DEFAULTS["MODE"]).lower()
    if mode == "lrt_bh":
        from .pipes import run_lrt_overall
        return run_lrt_overall(core_df_with_const, allowed_mask_by_cat, anc_series,
                               phenos_list, name_to_cat, cdr_codename,
                               target_inversion, ctx, min_available_memory_gb,
                               on_pool_started=on_pool_started)
    else:
        from .pipes import run_bootstrap_overall
        return run_bootstrap_overall(core_df_with_const, allowed_mask_by_cat,
                                     anc_series, phenos_list, name_to_cat,
                                     cdr_codename, target_inversion, ctx,
                                     min_available_memory_gb,
                                     on_pool_started=on_pool_started)


def consolidate_and_select(df, inversions, cache_root, alpha=0.05,
                           mode=None, selection=None, ctx_tags=None,
                           cdr_codename=None):
    ctx_tags = ctx_tags or {}
    mode = (mode or DEFAULTS["MODE"]).lower()
    selection = (selection or DEFAULTS["SELECTION"]).lower()
    if mode == "lrt_bh":
        rows = []
        for inv in inversions:
            lrt_dir = os.path.join(cache_root, models.safe_basename(inv), "lrt_overall")
            if not os.path.isdir(lrt_dir):
                continue
            for fn in os.listdir(lrt_dir):
                if fn.endswith(".json") and not fn.endswith(".meta.json"):
                    meta_path = os.path.join(lrt_dir, fn.replace(".json", ".meta.json"))
                    meta = io.read_meta_json(meta_path)
                    expected_tag = ctx_tags.get(inv)
                    if not meta:
                        continue
                    if expected_tag and meta.get("ctx_tag") != expected_tag:
                        continue
                    if cdr_codename and meta.get("cdr_codename") != cdr_codename:
                        continue
                    if meta.get("target") != inv:
                        continue
                    rec = pd.read_json(os.path.join(lrt_dir, fn), typ="series").to_dict()
                    rows.append({
                        "Phenotype": os.path.splitext(fn)[0],
                        "Inversion": inv,
                        "P_LRT_Overall": pd.to_numeric(rec.get("P_LRT_Overall"), errors="coerce")
                    })
        if rows:
            lrt_df = pd.DataFrame(rows)
            df = df.merge(lrt_df, on=["Phenotype", "Inversion"], how="left")
        else:
            df["P_LRT_Overall"] = np.nan
        mask = pd.to_numeric(df["P_LRT_Overall"], errors="coerce").notna()
        df["Q_GLOBAL"] = np.nan

        if int(mask.sum()) > 0:
            _, q, _, _ = multipletests(df.loc[mask, "P_LRT_Overall"], alpha=alpha, method="fdr_bh")
            df.loc[mask, "Q_GLOBAL"] = q
        df["Sig_Global"] = df["Q_GLOBAL"] < alpha
        return df, {}

    if selection != "bh_empirical":
        raise ValueError(f"unknown selection: {selection}")

    df["Q_GLOBAL"] = np.nan
    df["Sig_Global"] = False

    rows = []
        for inv in inversions:
            boot_dir = os.path.join(cache_root, models.safe_basename(inv), "boot_overall")
            if not os.path.isdir(boot_dir):
                continue
            for fn in os.listdir(boot_dir):
                if fn.endswith(".json") and not fn.endswith(".meta.json"):
                    meta_path = os.path.join(boot_dir, fn.replace(".json", ".meta.json"))
                    meta = io.read_meta_json(meta_path)
                    expected_tag = ctx_tags.get(inv)
                    if not meta:
                        continue
                    if expected_tag and meta.get("ctx_tag") != expected_tag:
                        continue
                    if cdr_codename and meta.get("cdr_codename") != cdr_codename:
                        continue
                    if meta.get("target") != inv:
                        continue
                    rec = pd.read_json(os.path.join(boot_dir, fn), typ="series").to_dict()
                    rows.append({
                        "Phenotype": os.path.splitext(fn)[0],
                        "Inversion": inv,
                        "P_EMP": pd.to_numeric(rec.get("P_EMP"), errors="coerce"),
                    "T_OBS": pd.to_numeric(rec.get("T_OBS"), errors="coerce"),
                    "B": int(rec.get("B", 0)),
                })
    if rows:
        boot_df = pd.DataFrame(rows)
        df = df.merge(boot_df, on=["Phenotype", "Inversion"], how="left")
    else:
        df["P_EMP"] = np.nan
    mask = pd.to_numeric(df["P_EMP"], errors="coerce").notna()

    if int(mask.sum()) > 0:
        _, q, _, _ = multipletests(df.loc[mask, "P_EMP"], alpha=alpha, method="fdr_bh")
        df.loc[mask, "Q_GLOBAL"] = q
    df["Sig_Global"] = df["Q_GLOBAL"] < alpha
    return df, {}


def apply_followup_fdr(df, alpha_global, lrt_select_alpha):
    """Apply within-ancestry FDR and annotate final interpretation."""
    import numpy as np
    from statsmodels.stats.multitest import multipletests
    import pandas as pd

    pcol_overall = "P_LRT_Overall" if "P_LRT_Overall" in df.columns else ("P_EMP" if "P_EMP" in df.columns else None)
    m_total = int(pd.to_numeric(df[pcol_overall], errors="coerce").notna().sum()) if pcol_overall else 0
    R_selected = int(pd.to_numeric(df.get("Sig_Global"), errors="coerce").fillna(False).astype(bool).sum())
    alpha_within = (alpha_global * (R_selected / m_total)) if m_total > 0 else 0.0

    if R_selected > 0 and alpha_within > 0.0 and "P_LRT_AncestryxDosage" in df.columns:
        selected_idx = df.index[df["Sig_Global"] == True].tolist()
        for idx in selected_idx:
            p_lrt = df.at[idx, "P_LRT_AncestryxDosage"]
            if (not pd.notna(p_lrt)) or (p_lrt >= lrt_select_alpha):
                continue
            levels_str = str(df.at[idx, "LRT_Ancestry_Levels"]) if "LRT_Ancestry_Levels" in df.columns else ""
            anc_levels = [s for s in levels_str.split(",") if s]
            pvals, keys = [], []
            for anc in map(str.upper, anc_levels):
                pcol, rcol = f"{anc}_P", f"{anc}_REASON"
                if pcol in df.columns:
                    pval = df.at[idx, pcol]
                    reason = df.at[idx, rcol] if rcol in df.columns else ""
                    if pd.notna(pval) and reason not in ("insufficient_stratum_counts", "not_selected_by_LRT"):
                        pvals.append(float(pval))
                        keys.append(anc)
            if pvals:
                _, p_adj_vals, _, _ = multipletests(pvals, alpha=alpha_within, method="fdr_bh")
                for anc_key, adj_val in zip(keys, p_adj_vals):
                    df.at[idx, f"{anc_key}_P_FDR"] = float(adj_val)

    if "Sig_Global" in df.columns:
        df["FINAL_INTERPRETATION"] = ""
        for idx in df.index[df["Sig_Global"] == True].tolist():
            p_lrt = df.at[idx, "P_LRT_AncestryxDosage"] if "P_LRT_AncestryxDosage" in df.columns else np.nan
            if pd.isna(p_lrt) or p_lrt >= lrt_select_alpha:
                df.at[idx, "FINAL_INTERPRETATION"] = "overall"
                continue
            levels_str = str(df.at[idx, "LRT_Ancestry_Levels"]) if "LRT_Ancestry_Levels" in df.columns else ""
            anc_levels = [s.upper() for s in levels_str.split(",") if s]
            sig_groups = []
            for anc in anc_levels:
                adj_col, rcol = f"{anc}_P_FDR", f"{anc}_REASON"
                p_adj = df.at[idx, adj_col] if adj_col in df.columns else np.nan
                reason = df.at[idx, rcol] if rcol in df.columns else ""
                if pd.notna(p_adj) and p_adj < alpha_within and reason not in ("insufficient_stratum_counts", "not_selected_by_LRT"):
                    sig_groups.append(anc)
            df.at[idx, "FINAL_INTERPRETATION"] = ",".join(sig_groups) if sig_groups else "unable to determine"

    return df

