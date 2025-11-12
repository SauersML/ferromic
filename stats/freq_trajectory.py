"""Visualize the allele-frequency trajectory downloaded from the AGES dataset."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.request import urlopen

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for runtime usability
    raise SystemExit(
        "matplotlib is required to plot trajectories. Install it with 'pip install matplotlib'."
    ) from exc

TRAJECTORY_URL = (
    "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/"
    "Trajectory-12_47296118_A_G.tsv"
)
OUTPUT_IMAGE = Path("allele_frequency_trajectory.png")

# Column descriptions supplied by the AGES project. These comments double as
# in-code documentation for anyone reusing the downloaded table.
#
# Time fields
# * date_left / date_right — The bounds of the sliding time window (in years
#   before present) used to compute that row’s estimates. Think “the bin starts
#   here, ends there.”
# * date_center — The midpoint of that window; this is the x-coordinate used to
#   plot the point for that window.
# * date_mean — The average sampling date (BP) of the individuals that actually
#   contribute data inside that window; if the window spans a millennium but
#   only mid-period samples exist, this will sit near those sample dates.
# * date_normalized — The same time value mapped to the model’s internal scale
#   (e.g., converted to generations and centered so “today” is 0). In the AGES
#   GLMM, time enters on the logit scale of allele frequency with units tied to
#   the generation interval (≈29 years/generation).
#
# Counts and raw frequencies (empirical within each window)
# * num_allele — Haploid sample size in the window: the number of allele copies
#   with calls (≈ 2 × number of diploid individuals contributing data there,
#   after imputation/QC).
# * num_alt_allele — Count of alternative-allele copies among those calls in the
#   window.
# * af — Empirical allele frequency for the window: num_alt_allele / num_allele.
# * af_low / af_up — The uncertainty band for that empirical frequency in the
#   window (a binomial-likelihood–based confidence/credible interval around af).
#
# Model-predicted trajectory (smoothed / structure-adjusted)
# * pt — The model-predicted allele frequency at date_center (“pₜ”), after
#   smoothing and correction for population structure; this is the trajectory
#   the browser draws through the noisy points. It differs from af in that af is
#   the raw window estimate, whereas pt is the fitted value from the trajectory
#   model.
# * pt_low / pt_up — The model’s uncertainty band for pt at that time (the
#   fitted trajectory’s lower/upper interval).


def download_trajectory(url: str = TRAJECTORY_URL) -> List[Dict[str, float]]:
    """Download the allele-frequency trajectory TSV file and parse it."""

    with urlopen(url) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise RuntimeError(f"Failed to download trajectory (status {status}).")
        payload = response.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(payload), delimiter="\t")
    rows: List[Dict[str, float]] = []
    for row in reader:
        parsed_row = {key: float(value) for key, value in row.items()}
        rows.append(parsed_row)

    if not rows:
        raise RuntimeError("Trajectory file is empty.")

    return rows


def rows_to_columns(rows: Iterable[Dict[str, float]]) -> Dict[str, List[float]]:
    """Convert a row-oriented table into column-oriented lists for plotting."""

    columns: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            columns.setdefault(key, []).append(value)
    return columns


def plot_trajectory(columns: Dict[str, List[float]], output: Path) -> None:
    """Plot empirical and model allele-frequency trajectories with uncertainty."""

    dates = columns["date_center"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Empirical allele frequencies with confidence interval shading.
    ax.fill_between(
        dates,
        columns["af_low"],
        columns["af_up"],
        color="#b3cde3",
        alpha=0.45,
        label="Empirical frequency interval",
    )
    ax.plot(
        dates,
        columns["af"],
        color="#045a8d",
        linewidth=2.5,
        label="Empirical allele frequency",
    )

    # Model-predicted trajectory with its interval.
    ax.fill_between(
        dates,
        columns["pt_low"],
        columns["pt_up"],
        color="#ccebc5",
        alpha=0.45,
        label="Model trajectory interval",
    )
    ax.plot(
        dates,
        columns["pt"],
        color="#238b45",
        linewidth=2.5,
        label="Modelled allele frequency",
    )

    ax.set_xlabel("Years before present (window center)")
    ax.set_ylabel("Allele frequency")
    ax.set_ylim(0, 1)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(axis="both", labelsize=11)

    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    rows = download_trajectory()
    columns = rows_to_columns(rows)
    plot_trajectory(columns, OUTPUT_IMAGE)
    print(f"Saved allele frequency trajectory to {OUTPUT_IMAGE.resolve()}")


if __name__ == "__main__":
    main()
