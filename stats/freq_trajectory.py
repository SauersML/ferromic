"""Visualize the allele-frequency trajectory downloaded from the AGES dataset."""

from __future__ import annotations

import csv
import io
from bisect import bisect_left
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.request import urlopen

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
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


def _prepare_interpolator(
    dates: List[float], values: List[float]
) -> Tuple[List[float], List[float]]:
    """Return the date/value series sorted for interpolation."""

    paired = sorted(zip(dates, values))
    sorted_dates = [date for date, _ in paired]
    sorted_values = [value for _, value in paired]
    return sorted_dates, sorted_values


def _interpolate(date: float, dates: List[float], values: List[float]) -> float:
    """Linearly interpolate the value at ``date`` within the sorted series."""

    if date <= dates[0]:
        return values[0]
    if date >= dates[-1]:
        return values[-1]

    idx = bisect_left(dates, date)
    if dates[idx] == date:
        return values[idx]

    left_date = dates[idx - 1]
    right_date = dates[idx]
    left_value = values[idx - 1]
    right_value = values[idx]

    span = right_date - left_date
    if span == 0:
        return left_value
    weight = (date - left_date) / span
    return left_value + weight * (right_value - left_value)


def _find_largest_window_change(
    dates: List[float],
    values: List[float],
    window: float,
) -> Optional[Tuple[float, float, float]]:
    """Return the start, end, and magnitude of the largest change in ``window`` years."""

    if not dates:
        return None

    sorted_dates, sorted_values = _prepare_interpolator(dates, values)
    min_date = sorted_dates[0]
    max_date = sorted_dates[-1]
    if max_date - min_date < window:
        return None

    best_start = None
    best_change = -1.0
    best_end = None
    
    start = min_date
    while start <= max_date - window:
        end = start + window
        start_val = _interpolate(start, sorted_dates, sorted_values)
        end_val = _interpolate(end, sorted_dates, sorted_values)
        change = abs(end_val - start_val)
        
        if change > best_change:
            best_change = change
            best_start = start
            best_end = end
        
        start += 1.0  # Check every year

    if best_start is None or best_end is None:
        return None

    return best_start, best_end, best_change


def plot_trajectory(
    columns: Dict[str, List[float]], output: Path
) -> Optional[Tuple[float, float, float]]:
    """Plot empirical and model allele-frequency trajectories with uncertainty."""

    dates = columns["date_center"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.fill_between(
        dates,
        columns["af_low"],
        columns["af_up"],
        color="#b3cde3",
        alpha=0.45,
        label="",
    )
    ax.plot(
        dates,
        columns["af"],
        color="#045a8d",
        linewidth=2.5,
        label="Empirical allele frequency",
    )

    ax.fill_between(
        dates,
        columns["pt_low"],
        columns["pt_up"],
        color="#ccebc5",
        alpha=0.45,
        label="",
    )
    ax.plot(
        dates,
        columns["pt"],
        color="#238b45",
        linewidth=2.5,
        label="Model allele frequency",
    )

    ax.set_xlabel("Years before present (window center)", fontsize=20, fontweight='bold')

    def _format_year(value: float, _: float) -> str:
        if abs(value) >= 100:
            formatted = f"{value:,.0f}"
        elif abs(value) >= 10:
            formatted = f"{value:,.1f}".rstrip("0").rstrip(".")
        else:
            formatted = f"{value:,.2f}".rstrip("0").rstrip(".")
        return formatted

    ax.xaxis.set_major_formatter(FuncFormatter(_format_year))
    ax.set_ylabel('Derived allele "G" frequency (rs34666797)', fontsize=20, fontweight='bold')

    series_for_ylim = [
        columns["af_low"],
        columns["af_up"],
        columns["af"],
        columns["pt_low"],
        columns["pt_up"],
        columns["pt"],
    ]
    all_values = [value for series in series_for_ylim for value in series]
    ymin = min(all_values)
    ymax = max(all_values)
    padding = (ymax - ymin) * 0.05 if ymax > ymin else 0.05
    ax.set_ylim(ymin - padding, ymax + padding)

    highlight = _find_largest_window_change(dates, columns["af"], window=1_000.0)
    if highlight is not None:
        start_year, end_year, change = highlight
        ax.axvspan(
            min(start_year, end_year),
            max(start_year, end_year),
            color="#fdd49e",
            alpha=0.35,
            label="Largest 1,000-year change",
        )
    
    ax.invert_xaxis()
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=16)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(axis="both", labelsize=18)

    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)

    return highlight


def main() -> None:
    rows = download_trajectory()
    columns = rows_to_columns(rows)
    highlight = plot_trajectory(columns, OUTPUT_IMAGE)
    if highlight is not None:
        start_year, end_year, change = highlight
        print(
            "Largest 1,000-year change window: "
            f"start={start_year:g} BP ({start_year/1000:.3f} kya), "
            f"end={end_year:g} BP ({end_year/1000:.3f} kya), "
            f"|Δf|={change:.4f}"
        )
    print(f"Saved allele frequency trajectory to {OUTPUT_IMAGE.resolve()}")


if __name__ == "__main__":
    main()
