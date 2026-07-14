"""Transferable pop-gen feature set: the diversity/differentiation features computable
identically on simulations (via features.popgen_group_stats, per-bp) and on the real
inversions (via ferromic output.csv, per-bp). This is what the real-data classifier
uses, so a sim-fit transfers without a domain shift in feature definitions.
"""
from __future__ import annotations

import numpy as np

PI_FLOOR = 1e-7
TRANSFERABLE_FEATURES = [
    "log_pi_ratio", "log_pi_inv", "log_pi_dir",
    "hudson_fst", "hudson_dxy",
    "log_theta_ratio", "log_seg_sites", "inv_freq",
]


def build_transferable(pi_inv, pi_dir, theta_inv, theta_dir, dxy, fst, seg_total, inv_freq):
    pii = max(float(pi_inv), PI_FLOOR)
    pid = max(float(pi_dir), PI_FLOOR)
    thi = max(float(theta_inv), PI_FLOOR)
    thd = max(float(theta_dir), PI_FLOOR)
    return {
        "log_pi_ratio": float(np.log(pii / pid)),
        "log_pi_inv": float(np.log(pii)),
        "log_pi_dir": float(np.log(pid)),
        "hudson_fst": float(fst) if fst == fst else np.nan,
        "hudson_dxy": float(dxy) if dxy == dxy else np.nan,
        "log_theta_ratio": float(np.log(thi / thd)),
        "log_seg_sites": float(np.log1p(max(0.0, float(seg_total)))),
        "inv_freq": float(inv_freq) if inv_freq == inv_freq else np.nan,
    }


def from_group_stats(s):
    """s = features.popgen_group_stats(...) dict (simulations)."""
    return build_transferable(s["pi_inv_bp"], s["pi_dir_bp"], s["theta_inv_bp"],
                              s["theta_dir_bp"], s["dxy_bp"], s["fst"],
                              (s["seg_inv"] + s["seg_dir"]), s["inv_freq"])


def from_output_csv_row(r):
    """r = a row of ferromic output.csv (Series). Uses the filtered tracks."""
    seg = (float(r.get("0_segregating_sites_filtered", 0) or 0)
           + float(r.get("1_segregating_sites_filtered", 0) or 0))
    # Watterson theta per bp: output.csv gives 0/1_w_theta_filtered (already per bp)
    return build_transferable(
        pi_inv=r["1_pi_filtered"], pi_dir=r["0_pi_filtered"],
        theta_inv=r.get("1_w_theta_filtered", np.nan), theta_dir=r.get("0_w_theta_filtered", np.nan),
        dxy=r.get("hudson_dxy_hap_group_0v1", np.nan), fst=r.get("hudson_fst_hap_group_0v1", np.nan),
        seg_total=seg, inv_freq=r.get("inversion_freq_filter", np.nan),
    )
