import os
import sys
import json
import logging
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from cyvcf2 import VCF

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    confusion_matrix,
)
from scipy.stats import pearsonr

# -----------------------------
# Hard-coded inputs
# -----------------------------
GROUND_TRUTH_FILE = "../variants_freeze4inv_sv_inv_hg38_processed_arbigent_filtered_manualDotplot_filtered_PAVgenAdded_withInvCategs_syncWithWH.fixedPH.simpleINV.mod.tsv"
VCF_PATH = "../vcfs/chr17.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz"
OUTPUT_DIR = "three_snp_results"

# -----------------------------
# Subsetting controls
# -----------------------------
SUBSET_TO_GROUP = True  # When True, restrict analysis to VCF samples whose names begin with group prefix
GROUP_PREFIX = "EUR_"        # Prefix that denotes group samples in VCF sample names.


# Target SNPs and alleles (H1/H2 per your instruction)
CHROM = "chr17"
TARGET_LOCI_INFO_3SNP = {
    "chr17:45996523": {"H1": "A", "H2": "G"},  # rs1052553
    "chr17:45974480": {"H1": "A", "H2": "G"},  # rs1800547
    "chr17:46024197": {"H1": "T", "H2": "C"},  # rs9468
}
TARGET_SNPS = [
    {"id": "chr17:45996523", "pos": 45996523, "rsid": "rs1052553"},
    {"id": "chr17:45974480", "pos": 45974480, "rsid": "rs1800547"},
    {"id": "chr17:46024197", "pos": 46024197, "rsid": "rs9468"},
]

# -----------------------------
# Logging setup
# -----------------------------
def setup_logging():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "run_log.txt")
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.info("=== Three-SNP Regression + Heuristic Evaluation ===")

# -----------------------------
# Utilities
# -----------------------------
def complement_base(b: str) -> str:
    """Return the DNA complement for a single base."""
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    return comp.get(b.upper(), "N")

def parse_gt_label(gt_str: str):
    """
    Map ground-truth genotype strings to dosage: 0 (H1/H1), 1 (H1/H2), 2 (H2/H2).
    Preserve a high/low confidence flag similar to existing script.
    """
    if pd.isna(gt_str):
        return None, None, None
    s = str(gt_str).strip()
    high_conf_map = {"0|0": 0, "1|0": 1, "0|1": 1, "1|1": 2}
    low_conf_map  = {"0|0_lowconf": 0, "1|0_lowconf": 1, "0|1_lowconf": 1, "1|1_lowconf": 2}

    if s in high_conf_map:
        return high_conf_map[s], True, s
    if s in low_conf_map:
        return low_conf_map[s], False, s.replace("_lowconf", "")
    return None, None, None

def select_inversion_row(config_df: pd.DataFrame) -> pd.Series:
    """
    Hard-coded to pick the chr17 inversion row with a specific ID.
    If multiple match, choose the first. If none match, exit with an error.
    """
    if "seqnames" not in config_df.columns or "start" not in config_df.columns or "end" not in config_df.columns:
        logging.critical("Ground-truth file missing required columns: seqnames/start/end.")
        sys.exit(1)

    df = config_df.copy()
    df = df[(df["verdict"] == "pass") & (df["seqnames"] == CHROM)]
    if df.empty:
        logging.critical("No 'pass' rows on chr17 in ground-truth TSV.")
        sys.exit(1)

    TARGET_INVERSION_ID = "chr17-45585160-INV-706887"
    logging.info(f"Hard-coded to find inversion ID: {TARGET_INVERSION_ID}")

    # Ensure the 'orig_ID' column exists before trying to search it
    if "orig_ID" not in df.columns:
        logging.critical(f"Ground-truth file is missing the 'orig_ID' column needed to find '{TARGET_INVERSION_ID}'.")
        sys.exit(1)

    # Search for the specific row by its unique ID
    cand = df[df["orig_ID"] == TARGET_INVERSION_ID]

    # Check if we found the inversion
    if cand.empty:
        logging.critical(f"Could not find inversion with ID '{TARGET_INVERSION_ID}' on chr17 with verdict='pass'. Check the ID and the ground-truth file.")
        sys.exit(1)
    else:
        # Select the first (and likely only) match
        row = cand.iloc[0]

    inv_id = row.get("orig_ID", "Unknown_ID")
    logging.info(f"Selected inversion row: id={inv_id} chr={row['seqnames']} start={row['start']} end={row['end']}")
    return row
def map_samples_to_vcf(tsv_row: pd.Series, vcf_samples: list) -> dict:
    """
    Map TSV sample columns (HG/NA) to VCF sample names by substring match (same as your other script).
    Returns dict: tsv_sample -> vcf_sample
    """
    tsv_samples = [c for c in tsv_row.index if str(c).startswith(("HG", "NA"))]
    sample_map = {}
    for ts in tsv_samples:
        if pd.isna(tsv_row[ts]):
            continue
        matches = [vs for vs in vcf_samples if ts in vs]
        if len(matches) == 1:
            sample_map[ts] = matches[0]
        elif len(matches) > 1:
            logging.warning(f"Ambiguous mapping for TSV sample {ts}: {matches} -> skipping")
        else:
            logging.warning(f"No VCF sample match for TSV sample {ts}")
    if not sample_map:
        logging.critical("No TSV samples could be mapped to VCF samples.")
        sys.exit(1)
    logging.info(f"Mapped {len(sample_map)} TSV samples to VCF.")
    return sample_map

def fetch_snp_variant(vcf_reader: VCF, chrom: str, pos: int):
    """
    Fetch the variant at (chrom:pos) if it's a biallelic SNP; else return None.
    """
    region = f"{chrom}:{pos}-{pos}"
    try:
        for var in vcf_reader(region):
            if var.POS != pos:
                continue
            if not var.is_snp:
                continue
            if len(var.ALT) != 1:
                continue
            return var
    except Exception as e:
        logging.warning(f"Error fetching {chrom}:{pos}: {e}")
    return None

def align_h2_to_variant(h2_base: str, ref: str, alt: str):
    """
    Decide how to compute H2 dosage relative to REF/ALT, possibly via strand complement.
    Returns (mode, strand_flipped) where mode in {'ALT','REF',None}.
    """
    h2 = h2_base.upper()
    ref_u = ref.upper()
    alt_u = alt.upper()

    if h2 == alt_u:
        return "ALT", False
    if h2 == ref_u:
        return "REF", False
    if complement_base(h2) == alt_u:
        return "ALT", True
    if complement_base(h2) == ref_u:
        return "REF", True
    return None, False

def build_y_from_tsv_row(tsv_row: pd.Series, sample_map: dict):
    """
    Build vectors: y_true (0/1/2), is_high_conf (bool), raw_gt (string) for mapped VCF samples (ordered).
    """
    y, conf, raw = [], [], []
    keep_vcf_samples = []
    for ts, vs in sample_map.items():
        dosage, is_high, raw_gt = parse_gt_label(tsv_row[ts])
        if dosage is None:
            continue
        y.append(int(dosage))
        conf.append(bool(is_high))
        raw.append(raw_gt)
        keep_vcf_samples.append(vs)
    if not y:
        logging.critical("No valid ground-truth dosages after parsing TSV row.")
        sys.exit(1)
    return np.array(y, dtype=int), np.array(conf, dtype=bool), np.array(raw, dtype=str), keep_vcf_samples

def compute_h2_dosages_for_snp(var, mode: str):
    """
    Compute H2 dosage per sample for a single variant given mode:
      - 'ALT': dosage = # of ALT alleles = g0 + g1
      - 'REF': dosage = # of H2 alleles when H2==REF = (2 - (g0 + g1))
    Returns array of shape (n_samples,) with values in {0,1,2} or np.nan where missing genotype.
    """
    gts = var.genotypes  # list of [a1, a2, phased?, ...] per VCF sample order
    out = []
    for g in gts:
        if g is None or len(g) < 2:
            out.append(np.nan)
            continue
        a1, a2 = g[0], g[1]
        if a1 < 0 or a2 < 0:
            out.append(np.nan)
            continue
        alt_count = (1 if a1 == 1 else 0) + (1 if a2 == 1 else 0)
        if mode == "ALT":
            out.append(float(alt_count))
        elif mode == "REF":
            out.append(float(2 - alt_count))
        else:
            out.append(np.nan)
    return np.array(out, dtype=float)

def choose_present_snps(snp_status):
    """
    Determine which SNPs are globally usable (variant found + allele mapping valid).
    Returns a list of snp_ids.
    """
    present = [sid for sid, st in snp_status.items() if st["usable"]]
    missing = [sid for sid in snp_status if sid not in present]
    if missing:
        logging.warning(f"SNPs not usable and will be excluded: {missing}")
    if not present:
        logging.error("None of the three SNPs are usable in the VCF. Regression will be skipped; heuristic may still use per-sample observed SNPs if any exist.")
    else:
        logging.info(f"Using SNPs for modeling: {present}")
    return present

def safe_pearsonr(y_true, y_pred):
    """Pearson r with guards; returns (r, r2)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan"), 0.0
    r, _ = pearsonr(y_true, y_pred)
    r2 = 0.0 if np.isnan(r) else r**2
    return float(r), float(r2)

def classification_metrics(y_true, y_pred_cls, label=""):
    """Compute classification metrics with robust handling."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred_cls = np.asarray(y_pred_cls, dtype=int)
    acc = accuracy_score(y_true, y_pred_cls)
    bal_acc = balanced_accuracy_score(y_true, y_pred_cls)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred_cls, labels=[0,1,2], average=None, zero_division=0
    )
    macro_f1 = np.mean(f1)
    cm = confusion_matrix(y_true, y_pred_cls, labels=[0,1,2])
    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_F1": macro_f1,
        "per_class": {
            "labels": [0,1,2],
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        }
    }
    logging.info(f"[{label}] acc={acc:.4f} bal_acc={bal_acc:.4f} macroF1={macro_f1:.4f}")
    return metrics, cm

def regression_metrics(y_true, y_pred, label=""):
    """Compute regression metrics (and return rounded predictions for classification-style eval)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r, r2_from_r = safe_pearsonr(y_true, y_pred)
    try:
        r2_sklearn = float(r2_score(y_true, y_pred))
    except Exception:
        r2_sklearn = float("nan")
    resid = y_true - y_pred
    resid_summary = {
        "mean": float(np.nanmean(resid)),
        "std": float(np.nanstd(resid)),
        "min": float(np.nanmin(resid)),
        "max": float(np.nanmax(resid)),
    }
    logging.info(f"[{label}] RMSE={rmse:.4f} MAE={mae:.4f} r={r if not np.isnan(r) else 0.0:.4f} r2(pearson)={r2_from_r:.4f} r2(sklearn)={r2_sklearn if not np.isnan(r2_sklearn) else 0.0:.4f}")
    return {
        "rmse": rmse,
        "mae": mae,
        "pearson_r": r,
        "pearson_r2": r2_from_r,
        "r2_sklearn": r2_sklearn,
        "residual_summary": resid_summary,
    }

# -----------------------------
# Main workflow
# -----------------------------
def main():
    setup_logging()

    # 0) Sanity checks
    if not os.path.exists(GROUND_TRUTH_FILE):
        logging.critical(f"Ground-truth TSV not found: {GROUND_TRUTH_FILE}")
        sys.exit(1)
    if not os.path.exists(VCF_PATH):
        logging.critical(f"VCF not found: {VCF_PATH}")
        sys.exit(1)

    # 1) Load ground-truth and pick the inversion row
    config_df = pd.read_csv(GROUND_TRUTH_FILE, sep="\t", dtype={"seqnames": str}, on_bad_lines="warn", low_memory=False)
    row = select_inversion_row(config_df)

    # 2) Open VCF and map samples
    vcf_full = VCF(VCF_PATH, lazy=True)
    vcf_samples_all = vcf_full.samples

    # Limit to group samples when requested via flag. This relies on the explicit prefix present in VCF sample names.
    if SUBSET_TO_GROUP:
        vcf_samples_all = [s for s in vcf_samples_all if isinstance(s, str) and s.startswith(GROUP_PREFIX)]
        if len(vcf_samples_all) == 0:
            logging.critical(f"No VCF samples found with prefix '{GROUP_PREFIX}' while SUBSET_TO_GROUP is True.")
            sys.exit(1)
        logging.info(f"Subsetting enabled: using {len(vcf_samples_all)} group VCF samples with prefix '{GROUP_PREFIX}'.")
    else:
        logging.info("Subsetting disabled: using all VCF samples.")

    sample_map = map_samples_to_vcf(row, vcf_samples_all)


    # Build y (ordered by VCF sample names we will actually read)
    y_true_list, is_high_conf_list, raw_gt_list, vcf_samples_used = build_y_from_tsv_row(row, sample_map)

    # Open a restricted VCF with only the mapped samples, to keep ordering stable
    vcf = VCF(VCF_PATH, samples=vcf_samples_used, lazy=True)

    # 3) For each target SNP, fetch variant and compute H2 dosages per sample
    snp_status = {}  # per SNP id: dict with info & flags
    snp_dosages = {} # per SNP id: numpy array (n_samples,) with dosage or NaN

    for snp in TARGET_SNPS:
        snp_id = snp["id"]
        pos = snp["pos"]
        rsid = snp["rsid"]
        H1 = TARGET_LOCI_INFO_3SNP[snp_id]["H1"].upper()
        H2 = TARGET_LOCI_INFO_3SNP[snp_id]["H2"].upper()

        var = fetch_snp_variant(vcf, CHROM, pos)
        info = {
            "rsid": rsid,
            "pos": pos,
            "usable": False,
            "reason": "",
            "ref": None,
            "alt": None,
            "mode": None,
            "strand_flipped": False,
        }

        if var is None:
            info["reason"] = "Variant not found or not a biallelic SNP"
            snp_status[snp_id] = info
            snp_dosages[snp_id] = np.full(len(vcf_samples_used), np.nan)
            logging.warning(f"{snp_id} ({rsid}) not usable: {info['reason']}")
            continue

        ref = var.REF
        alt = var.ALT[0]
        info["ref"] = ref
        info["alt"] = alt

        mode, flipped = align_h2_to_variant(H2, ref, alt)
        info["mode"] = mode
        info["strand_flipped"] = bool(flipped)

        if mode is None:
            info["reason"] = f"Allele mismatch (H2={H2}, REF={ref}, ALT={alt}, and complements)"
            snp_status[snp_id] = info
            snp_dosages[snp_id] = np.full(len(vcf_samples_used), np.nan)
            logging.warning(f"{snp_id} ({rsid}) not usable: {info['reason']}")
            continue

        # Compute H2 dosage
        dos = compute_h2_dosages_for_snp(var, mode)
        snp_dosages[snp_id] = dos
        # Usable if at least some non-missing calls exist
        if np.isfinite(dos).any():
            info["usable"] = True
        else:
            info["reason"] = "All genotypes missing for this SNP across selected samples"
        snp_status[snp_id] = info

        flip_note = " (strand-flipped)" if flipped else ""
        logging.info(f"{snp_id} ({rsid}) usable via mode={mode}{flip_note}; REF={ref} ALT={alt}; non-missing calls: {int(np.isfinite(dos).sum())}/{len(dos)}")

    present_snp_ids = choose_present_snps(snp_status)

    # 4) Assemble per-sample table baseline
    samples_series = pd.Series(vcf_samples_used, name="vcf_sample")
    df_per_sample = pd.DataFrame({
        "vcf_sample": vcf_samples_used,
        "y_true": y_true_list,
        "is_high_conf": is_high_conf_list,
        "raw_gt": raw_gt_list
    })

    # Attach SNP dosages columns (H2-dosage: 0/1/2, NaN if missing)
    for snp in TARGET_SNPS:
        sid = snp["id"]
        col = f"dosage_{sid.replace(':', '_')}"
        df_per_sample[col] = snp_dosages[sid] if sid in snp_dosages else np.full(len(vcf_samples_used), np.nan)

    # 4.5) Single-SNP correlation diagnostics
    # For each SNP, compute Pearson correlation between H2-dosage and the inversion dosage (y_true),
    # along with raw counts by dosage and mean y_true per dosage group. Direction indicates which
    # allele is associated with higher inversion dosage given the H2-encoding for that SNP.
    single_snp_stats = {}
    for snp in TARGET_SNPS:
        sid = snp["id"]
        rsid = snp["rsid"]
        pos = snp["pos"]
        col = f"dosage_{sid.replace(':', '_')}"
        x = df_per_sample[col].astype(float)
        mask = x.notna()
        y_vec = df_per_sample.loc[mask, "y_true"].astype(float).values
        x_vec = x.loc[mask].astype(float).values
        if mask.sum() >= 2 and np.std(x_vec) > 0 and np.std(y_vec) > 0:
            r_val, _ = pearsonr(x_vec, y_vec)
        else:
            r_val = float("nan")
        dose0_mask = x_vec == 0.0
        dose1_mask = x_vec == 1.0
        dose2_mask = x_vec == 2.0
        counts = {
            "dose0_n": int(np.sum(dose0_mask)),
            "dose1_n": int(np.sum(dose1_mask)),
            "dose2_n": int(np.sum(dose2_mask)),
        }
        means = {
            "y_mean_dose0": float(np.mean(y_vec[dose0_mask])) if counts["dose0_n"] > 0 else float("nan"),
            "y_mean_dose1": float(np.mean(y_vec[dose1_mask])) if counts["dose1_n"] > 0 else float("nan"),
            "y_mean_dose2": float(np.mean(y_vec[dose2_mask])) if counts["dose2_n"] > 0 else float("nan"),
        }
        direction = "H2 increases dosage (positive correlation)" if np.isfinite(r_val) and r_val > 0 else ("H1 increases dosage (negative correlation)" if np.isfinite(r_val) and r_val < 0 else "No direction (undefined or zero correlation)")
        st = snp_status.get(sid, {})
        single_snp_stats[sid] = {
            "rsid": rsid,
            "position": pos,
            "ref": st.get("ref"),
            "alt": st.get("alt"),
            "mode": st.get("mode"),
            "strand_flipped": st.get("strand_flipped"),
            "pearson_r": float(r_val) if np.isfinite(r_val) else None,
            "n": int(mask.sum()),
            "counts": counts,
            "y_means_by_dosage": means,
            "direction": direction
        }
        logging.info(f"SNP {sid} ({rsid}) correlation with dosage: r={(0.0 if not np.isfinite(r_val) else r_val):.4f} using n={int(mask.sum())}")
        logging.info(f"SNP {sid} counts by H2-dosage: 0={counts['dose0_n']} 1={counts['dose1_n']} 2={counts['dose2_n']}")
        logging.info(f"SNP {sid} y-mean by H2-dosage: dose0={means['y_mean_dose0']:.4f} dose1={means['y_mean_dose1']:.4f} dose2={means['y_mean_dose2']:.4f}")
        logging.info(f"SNP {sid} direction: {direction}")

    # 5) Heuristic predictions (resilient to partial availability)
    # Rule on observed SNPs only: all observed 0 -> 0; all observed 2 -> 2; else -> 1
    def heuristic_pred_for_row(row_vals):
        observed = [v for v in row_vals if np.isfinite(v)]
        if len(observed) == 0:
            return np.nan
        if all(v == 0.0 for v in observed):
            return 0
        if all(v == 2.0 for v in observed):
            return 2
        return 1

    dosage_cols = [f"dosage_{s['id'].replace(':', '_')}" for s in TARGET_SNPS]
    df_per_sample["heur_pred"] = df_per_sample[dosage_cols].apply(lambda r: heuristic_pred_for_row(list(r.values)), axis=1)

    # 6) Linear regression using the subset of SNPs that are globally usable
    regression_info = {"used_snps": present_snp_ids, "n_features": len(present_snp_ids)}
    if len(present_snp_ids) == 0:
        logging.warning("Regression skipped: no usable SNPs.")
        df_per_sample["reg_pred"] = np.nan
        df_per_sample["reg_pred_round"] = np.nan
        reg_metrics = None
        reg_cls_metrics = None
        cm_reg = None
    else:
        X_cols = [f"dosage_{sid.replace(':', '_')}" for sid in present_snp_ids]
        # We'll fit on rows with ALL chosen features present
        mask_reg = df_per_sample[X_cols].notna().all(axis=1)
        n_all = len(df_per_sample)
        n_fit = int(mask_reg.sum())
        if n_fit == 0:
            logging.warning("Regression skipped: chosen SNPs have no rows with complete data.")
            df_per_sample["reg_pred"] = np.nan
            df_per_sample["reg_pred_round"] = np.nan
            reg_metrics = None
            reg_cls_metrics = None
            cm_reg = None
        else:
            if n_fit < n_all:
                logging.warning(f"Regression will use only {n_fit}/{n_all} samples with complete data for selected SNPs.")
            X = df_per_sample.loc[mask_reg, X_cols].values.astype(float)
            y = df_per_sample.loc[mask_reg, "y_true"].values.astype(float)

            # Fit simple LinearRegression
            lr = LinearRegression()
            lr.fit(X, y)

            # Predict for those same rows
            y_pred = lr.predict(X)
            # Map predictions back into the full per-sample table; NaN for others
            df_per_sample["reg_pred"] = np.nan
            df_per_sample.loc[mask_reg, "reg_pred"] = y_pred

            # Clip to [0,2] and round for classification-style metrics
            df_per_sample["reg_pred_round"] = df_per_sample["reg_pred"].clip(lower=0.0, upper=2.0).round()

            # Regression metrics on fitted rows
            reg_metrics = regression_metrics(y_true=y, y_pred=y_pred, label="REG")
            # Classification-style metrics for rounded predictions (on same subset)
            reg_cls = df_per_sample.loc[mask_reg, ["y_true", "reg_pred_round"]].dropna()
            reg_cls_metrics, cm_reg = classification_metrics(
                y_true=reg_cls["y_true"].astype(int).values,
                y_pred_cls=reg_cls["reg_pred_round"].astype(int).values,
                label="REG(cls-rounded)"
            )

            # Save coefficients for reference
            regression_info["coefficients"] = dict(zip(X_cols, [float(c) for c in lr.coef_.tolist()]))
            regression_info["intercept"] = float(lr.intercept_)
            regression_info["n_samples_used"] = n_fit

    # 7) Heuristic classification metrics (use rows where a heuristic prediction exists)
    mask_heur = df_per_sample["heur_pred"].notna()
    n_heur = int(mask_heur.sum())
    if n_heur == 0:
        logging.warning("Heuristic metrics skipped: no rows with any observed SNP.")
        heur_metrics = None
        cm_heur = None
    else:
        heur_df = df_per_sample.loc[mask_heur, ["y_true", "heur_pred"]]
        heur_metrics, cm_heur = classification_metrics(
            y_true=heur_df["y_true"].astype(int).values,
            y_pred_cls=heur_df["heur_pred"].astype(int).values,
            label="HEUR"
        )

    # 8) Package metrics and write outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Confusion matrices
    if cm_reg is not None:
        cm_reg_df = pd.DataFrame(cm_reg, index=["true_0","true_1","true_2"], columns=["pred_0","pred_1","pred_2"])
        cm_reg_df.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix_regression.csv"))
    if cm_heur is not None:
        cm_heur_df = pd.DataFrame(cm_heur, index=["true_0","true_1","true_2"], columns=["pred_0","pred_1","pred_2"])
        cm_heur_df.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix_heuristic.csv"))

    # Metrics summary JSON
    metrics_summary = {
        "inputs": {
            "ground_truth_file": GROUND_TRUTH_FILE,
            "vcf_path": VCF_PATH,
            "chrom": CHROM,
            "target_snps": TARGET_SNPS,
            "target_loci_info": TARGET_LOCI_INFO_3SNP,
        },
        "snp_status": snp_status,
        "single_snp_correlations": single_snp_stats,
        "regression": {
            "used_snps": regression_info.get("used_snps"),
            "n_features": regression_info.get("n_features"),
            "n_samples_used": regression_info.get("n_samples_used", 0),
            "coefficients": regression_info.get("coefficients"),
            "intercept": regression_info.get("intercept"),
            "metrics_regression": reg_metrics,
            "metrics_classification_from_rounded": reg_cls_metrics,
        },
        "heuristic": {
            "rule": "all observed 0 -> 0; all observed 2 -> 2; else -> 1",
            "metrics": heur_metrics,
        },
        "coverage": {
            "total_samples_in_tsv_row": int(len(sample_map)),
            "mapped_vcf_samples": int(len(vcf_samples_used)),
            "per_snp_nonmissing_calls": {
                sid: int(np.isfinite(snp_dosages[sid]).sum()) for sid in snp_dosages
            },
            "class_counts_y_true": dict(Counter(df_per_sample["y_true"].astype(int).tolist())),
            "rows_with_any_observed_snp_for_heuristic": int(mask_heur.sum()),
        }
    }
    with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), "w") as fh:
        json.dump(metrics_summary, fh, indent=2)

    # Per-sample CSV
    # Contains: sample id, y_true, each SNP dosage, heuristic pred, regression pred (float), regression rounded, correctness flags, abs errors
    df_per_sample["is_correct_heur"] = np.where(
        df_per_sample["heur_pred"].notna(),
        (df_per_sample["heur_pred"] == df_per_sample["y_true"]).astype(int),
        np.nan
    )
    df_per_sample["is_correct_reg_round"] = np.where(
        df_per_sample["reg_pred_round"].notna(),
        (df_per_sample["reg_pred_round"] == df_per_sample["y_true"]).astype(int),
        np.nan
    )
    df_per_sample["abs_err_reg"] = np.where(
        df_per_sample["reg_pred"].notna(),
        np.abs(df_per_sample["y_true"] - df_per_sample["reg_pred"]),
        np.nan
    )
    df_per_sample["abs_err_heur"] = np.where(
        df_per_sample["heur_pred"].notna(),
        np.abs(df_per_sample["y_true"] - df_per_sample["heur_pred"].astype(float)),
        np.nan
    )

    per_sample_cols = ["vcf_sample", "y_true", "is_high_conf", "raw_gt"] + dosage_cols + [
        "heur_pred", "reg_pred", "reg_pred_round", "abs_err_reg", "abs_err_heur", "is_correct_heur", "is_correct_reg_round"
    ]
    df_per_sample[per_sample_cols].to_csv(os.path.join(OUTPUT_DIR, "per_sample_results.csv"), index=False)

    # Final summary to logs
    logging.info("=== DONE ===")
    logging.info(f"Results written to: {OUTPUT_DIR}")
    if metrics_summary["regression"]["metrics_regression"]:
        mm = metrics_summary["regression"]["metrics_regression"]
        logging.info(f"REG final: RMSE={mm['rmse']:.4f} MAE={mm['mae']:.4f} r2(pearson)={mm['pearson_r2']:.4f}")
    if metrics_summary["regression"]["metrics_classification_from_rounded"]:
        mc = metrics_summary["regression"]["metrics_classification_from_rounded"]
        logging.info(f"REG(cls-rounded): acc={mc['accuracy']:.4f} macroF1={mc['macro_F1']:.4f}")
    if metrics_summary["heuristic"]["metrics"]:
        mh = metrics_summary["heuristic"]["metrics"]
        logging.info(f"HEUR: acc={mh['accuracy']:.4f} macroF1={mh['macro_F1']:.4f}")

if __name__ == "__main__":
    main()
