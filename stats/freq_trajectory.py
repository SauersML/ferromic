"""
Generate a four-panel figure (A, B, C, D) showing allele frequency
trajectories of the inverted haplotype for four selected inversion polymorphisms.
"""

from __future__ import annotations

import csv
import shutil
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data"
OUTPUT_FILE = Path("data/inversion_trajectories_combined.pdf")

SUBPLOTS_CONFIG = [
    {
        "panel_label": "A",
        "title": "10q22.3 (chr10:79.5–80.2 Mb)",
        "filename": "Trajectory-10_81319354_C_T.tsv",
        "flip": False,
    },
    {
        "panel_label": "B",
        "title": "8p23.1 (chr8:7.3–12.6 Mb)",
        "filename": "Trajectory-8_9261356_T_A.tsv",
        "flip": True,
    },
    {
        "panel_label": "C",
        "title": "12q13.11 (chr12:46.90–46.92 Mb)",
        "filename": "Trajectory-12_47295449_A_G.tsv",
        "flip": False,
    },
    {
        "panel_label": "D",
        "title": "7p11.2 (chr7:54.23–54.31 Mb)",
        "filename": "Trajectory-7_54318757_A_G.tsv",
        "flip": False,
    },
]

# Colors
COLOR_EMPIRICAL_LINE = "#045a8d"  # Dark blue
COLOR_EMPIRICAL_FILL = "#b3cde3"  # Light blue
COLOR_MODEL_LINE = "#238b45"      # Dark green
COLOR_MODEL_FILL = "#ccebc5"      # Light green
COLOR_HIGHLIGHT = "#fdd49e"       # Light orange

# Font sizes
FONT_AXIS_LABEL_X = 24
FONT_AXIS_LABEL_Y = 24
FONT_PANEL_LABEL = 32
FONT_TICKS = 18
FONT_TITLE = 26
FONT_LEGEND = 12


# ---------------------------------------------------------------------------
# Data I/O and processing
# ---------------------------------------------------------------------------

def ensure_file_exists(filename: str) -> Path:
    """Ensure the given data file exists locally; download it if necessary."""
    local_path = Path("data") / filename
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        return local_path

    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename} from {url}...")

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(request) as response, local_path.open("wb") as out_file:
            if response.status != 200:
                raise HTTPError(url, response.status, "Bad status", response.headers, None)
            shutil.copyfileobj(response, out_file)
    except (URLError, HTTPError) as exc:
        if local_path.exists():
            local_path.unlink()
        raise RuntimeError(f"Failed to download {url}") from exc

    return local_path


def load_trajectory(filename: str) -> List[Dict[str, float]]:
    """Load and parse a trajectory TSV file into a list of dictionaries."""
    path = ensure_file_exists(filename)

    records: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=2):
            parsed_row: Dict[str, float] = {}
            for key, value in row.items():
                if value in ("", "NA", "nan"):
                    parsed_row[key] = float("nan")
                else:
                    try:
                        parsed_row[key] = float(value)
                    except ValueError:
                        parsed_row = {}
                        break
            if parsed_row:
                records.append(parsed_row)

    if not records:
        raise RuntimeError(f"Trajectory file {filename} is empty or invalid.")

    return records


def rows_to_columns(rows: Iterable[Dict[str, float]]) -> Dict[str, List[float]]:
    """Convert row-oriented dictionaries into column-oriented lists."""
    columns: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            columns.setdefault(key, []).append(value)
    return columns


def invert_allele_frequencies(columns: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Invert frequencies p -> 1 - p and swap corresponding confidence intervals."""
    inverted: Dict[str, List[float]] = {}

    if "date_center" in columns:
        inverted["date_center"] = list(columns["date_center"])

    if "af" in columns:
        inverted["af"] = [1.0 - v for v in columns["af"]]
    if "af_low" in columns and "af_up" in columns:
        inverted["af_low"] = [1.0 - v for v in columns["af_up"]]
        inverted["af_up"] = [1.0 - v for v in columns["af_low"]]

    if "pt" in columns:
        inverted["pt"] = [1.0 - v for v in columns["pt"]]
    if "pt_low" in columns and "pt_up" in columns:
        inverted["pt_low"] = [1.0 - v for v in columns["pt_up"]]
        inverted["pt_up"] = [1.0 - v for v in columns["pt_low"]]

    return inverted


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _prepare_interpolator(
    dates: List[float],
    values: List[float],
) -> Tuple[List[float], List[float]]:
    """Return dates and values sorted by date."""
    pairs = sorted(zip(dates, values))
    sorted_dates = [d for d, _ in pairs]
    sorted_values = [v for _, v in pairs]
    return sorted_dates, sorted_values


def _interpolate(
    date: float,
    dates: List[float],
    values: List[float],
) -> float:
    """Piecewise-linear interpolation of value at a given date."""
    if not dates:
        raise ValueError("Cannot interpolate with an empty date series.")

    if date <= dates[0]:
        return values[0]
    if date >= dates[-1]:
        return values[-1]

    index = bisect_left(dates, date)

    if dates[index] == date:
        return values[index]

    left_date = dates[index - 1]
    right_date = dates[index]
    left_value = values[index - 1]
    right_value = values[index]

    span = right_date - left_date
    if span == 0:
        return left_value

    weight = (date - left_date) / span
    return left_value + weight * (right_value - left_value)


def _find_largest_window_change(
    dates: List[float],
    af: List[float],
    af_low: List[float],
    af_up: List[float],
    window: float,
) -> Optional[Tuple[float, float, float, bool]]:
    """Find the 1,000-year window with the largest credible empirical change.

    For each window, we:
      - interpolate the empirical allele frequency (af) at the start and end,
      - interpolate the 95% CIs (af_low, af_up) at the start and end,
      - check whether the two 95% intervals overlap,
      - prioritize windows with non-overlapping intervals (“statistically
        distinguishable”), and within that set choose the largest |Δ|.
      - if no window has non-overlapping intervals, fall back to the window
        with the largest |Δ| overall.
    Returns (start_year, end_year, delta, is_significant).
    """
    if not dates:
        return None

    # Sort dates and keep af/CI aligned
    quad = sorted(zip(dates, af, af_low, af_up))
    sorted_dates = [d for d, _, _, _ in quad]
    sorted_af = [p for _, p, _, _ in quad]
    sorted_low = [l for _, _, l, _ in quad]
    sorted_up = [u for _, _, _, u in quad]

    min_date = sorted_dates[0]
    max_date = sorted_dates[-1]

    if max_date - min_date < window:
        return None

    best_start: Optional[float] = None
    best_end: Optional[float] = None
    best_delta: float = 0.0
    best_abs_change: float = -1.0
    best_significant: bool = False  # non-overlapping 95% CIs

    start = min_date
    while start <= max_date - window:
        end = start + window

        # Interpolate af and 95% CIs at the window boundaries
        f_start = _interpolate(start, sorted_dates, sorted_af)
        f_end = _interpolate(end, sorted_dates, sorted_af)
        delta = f_end - f_start
        abs_change = abs(delta)

        low_start = _interpolate(start, sorted_dates, sorted_low)
        up_start = _interpolate(start, sorted_dates, sorted_up)
        low_end = _interpolate(end, sorted_dates, sorted_low)
        up_end = _interpolate(end, sorted_dates, sorted_up)

        # Non-overlapping 95% intervals => “statistically distinguishable”
        intervals_disjoint = (up_start < low_end) or (up_end < low_start)
        is_significant = bool(intervals_disjoint)

        # Primary: significance; secondary: magnitude
        if (is_significant, abs_change) > (best_significant, best_abs_change):
            best_significant = is_significant
            best_abs_change = abs_change
            best_delta = delta
            best_start = start
            best_end = end

        start += 10.0

    if best_start is None or best_end is None:
        return None

    return best_start, best_end, best_delta, best_significant

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trajectory_on_axis(
    ax: Axes,
    columns: Dict[str, List[float]],
    config: Dict[str, Any],
    show_ylabel: bool,
    show_xlabel: bool,
    is_panel_a: bool,
) -> None:
    """Plot empirical and model trajectories for a single panel."""
    dates = columns["date_center"]

    # Empirical data
    ax.fill_between(
        dates,
        columns["af_low"],
        columns["af_up"],
        color=COLOR_EMPIRICAL_FILL,
        alpha=0.45,
        edgecolor="none",
    )
    ax.plot(
        dates,
        columns["af"],
        color=COLOR_EMPIRICAL_LINE,
        linewidth=1.5,
        alpha=0.8,
    )

    # Model data
    ax.fill_between(
        dates,
        columns["pt_low"],
        columns["pt_up"],
        color=COLOR_MODEL_FILL,
        alpha=0.45,
        edgecolor="none",
    )
    ax.plot(
        dates,
        columns["pt"],
        color=COLOR_MODEL_LINE,
        linewidth=2.5,
    )

    # Highlight 1 kyr window with largest credible empirical change
    highlight = _find_largest_window_change(
        dates,
        columns["af"],
        columns["af_low"],
        columns["af_up"],
        window=1000.0,
    )
    if highlight is not None:
        start_year, end_year, delta, is_significant = highlight
        ax.axvspan(
            min(start_year, end_year),
            max(start_year, end_year),
            color=COLOR_HIGHLIGHT,
            alpha=0.4,
            edgecolor="none",
        )
        status = "95% non-overlapping CIs" if is_significant else "overlapping CIs"
        print(
            f"Panel {config['panel_label']}: strongest 1 kyr change "
            f"{start_year:.0f}–{end_year:.0f} years ago "
            f"(Δ={delta:+.3f}, {status})"
        )

    # Panel label (A/B/C/D)
    ax.text(
        -0.1,
        1.1,
        config["panel_label"],
        transform=ax.transAxes,
        fontsize=FONT_PANEL_LABEL,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    ax.set_title(
        config["title"],
        fontsize=FONT_TITLE,
        fontweight="bold",
        loc="center",
    )

    if show_xlabel:
        ax.set_xlabel("Years before present", fontsize=FONT_AXIS_LABEL_X)
    if show_ylabel:
        ax.set_ylabel(
            "Inversion-tagging allele frequency",
            fontsize=FONT_AXIS_LABEL_Y,
        )

    def _format_year(value: float, _: float) -> str:
        if abs(value) >= 1000:
            return f"{value / 1000:.0f}k"
        return f"{value:,.0f}"

    ax.xaxis.set_major_formatter(FuncFormatter(_format_year))
    ax.set_xlim(14000, 0)

    all_values: List[float] = []
    for key in ("af", "pt", "af_low", "af_up"):
        if key in columns:
            all_values.extend(columns[key])

    if all_values:
        ymin = min(all_values)
        ymax = max(all_values)
        yrange = ymax - ymin
        if yrange < 0.01:
            yrange = 0.1

        padding = yrange * 0.05
        ax.set_ylim(
            max(0.0, ymin - padding),
            min(1.0, ymax + padding),
        )
    else:
        ax.set_ylim(0.0, 1.0)

    ax.tick_params(axis="both", labelsize=FONT_TICKS)
    ax.grid(False)

    if is_panel_a:
        legend_elements = [
            Line2D(
                [0],
                [0],
                color=COLOR_EMPIRICAL_LINE,
                lw=2,
                label="Empirical frequency",
            ),
            Line2D(
                [0],
                [0],
                color=COLOR_MODEL_LINE,
                lw=2,
                label="Model frequency",
            ),
            Patch(
                facecolor=COLOR_HIGHLIGHT,
                alpha=0.4,
                label="1,000 year period of largest credible change",
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            frameon=True,
            fontsize=FONT_LEGEND,
            framealpha=0.9,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating combined inversion trajectory plot...")

    plt.style.use("seaborn-v0_8-white")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(16, 12),
        constrained_layout=True,
    )

    axes_flat = axes.flatten()

    for index, subplot_config in enumerate(SUBPLOTS_CONFIG):
        ax = axes_flat[index]
        filename = subplot_config["filename"]

        row, col = divmod(index, 2)
        show_ylabel = col == 0
        show_xlabel = row == 1
        is_panel_a = index == 0

        print(f"Processing panel {subplot_config['panel_label']}: {filename}")

        try:
            rows = load_trajectory(filename)
            columns = rows_to_columns(rows)

            if subplot_config["flip"]:
                print("  Inverting allele frequencies.")
                columns = invert_allele_frequencies(columns)

            plot_trajectory_on_axis(
                ax=ax,
                columns=columns,
                config=subplot_config,
                show_ylabel=show_ylabel,
                show_xlabel=show_xlabel,
                is_panel_a=is_panel_a,
            )
        except Exception as exc:
            print(
                f"  Error processing panel {subplot_config['panel_label']}: {exc}"
            )
            ax.text(
                0.5,
                0.5,
                "Data unavailable",
                ha="center",
                va="center",
                fontsize=20,
                color="red",
            )
            ax.set_title(subplot_config["title"], fontsize=FONT_TITLE)
            ax.set_axis_off()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=300)
    print(f"Saved 4-panel figure to {OUTPUT_FILE.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()
