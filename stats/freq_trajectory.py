"""
Generate a four-panel figure (A, B, C, D) showing the allele frequency 
trajectories of the inverted haplotype for four selected inversion polymorphisms.
"""

from __future__ import annotations

import csv
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import shutil

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required. Install it with 'pip install matplotlib'."
    ) from exc


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data"
OUTPUT_FILE = Path("data/inversion_trajectories_combined.pdf")

# Configuration for the four subplots
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

# Style Constants
COLOR_EMPIRICAL_LINE = "#045a8d"  # Dark Blue
COLOR_EMPIRICAL_FILL = "#b3cde3"  # Light Blue
COLOR_MODEL_LINE = "#238b45"      # Dark Green
COLOR_MODEL_FILL = "#ccebc5"      # Light Green
COLOR_HIGHLIGHT = "#fdd49e"       # Light Orange

# Font Sizes (Calculated based on previous ~12pt base)
FONT_AXIS_LABEL = 15     # +20%
FONT_PANEL_LABEL = 32    # +50% (was ~20)
FONT_TICKS = 18          # +80% (was ~10)
FONT_TITLE = 16
FONT_LEGEND = 12


# -----------------------------------------------------------------------------
# Data I/O and Processing
# -----------------------------------------------------------------------------

def ensure_file_exists(filename: str) -> Path:
    """Check if file exists in 'data/'; if not, download it."""
    local_path = Path("data") / filename
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        return local_path

    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")
    
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req) as response:
            if response.status != 200:
                raise HTTPError(url, response.status, "Bad Status", response.headers, None)
            with local_path.open("wb") as out_file:
                shutil.copyfileobj(response, out_file)
    except (URLError, HTTPError) as e:
        if local_path.exists():
            local_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}")

    return local_path


def load_trajectory(filename: str) -> List[Dict[str, float]]:
    """Load and parse the TSV file."""
    path = ensure_file_exists(filename)
    
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows: List[Dict[str, float]] = []
        
        for line_num, row in enumerate(reader, start=2):
            try:
                parsed_row = {
                    k: float(v) if v not in ("", "NA", "nan") else float("nan") 
                    for k, v in row.items()
                }
                rows.append(parsed_row)
            except ValueError:
                continue

    if not rows:
        raise RuntimeError(f"Trajectory file {filename} is empty or invalid.")

    return rows


def rows_to_columns(rows: Iterable[Dict[str, float]]) -> Dict[str, List[float]]:
    """Convert row-oriented dicts to column-oriented lists."""
    columns: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            columns.setdefault(key, []).append(value)
    return columns


def invert_allele_frequencies(columns: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Invert frequencies (p -> 1-p) and swap confidence intervals."""
    inverted = {}
    
    if "date_center" in columns:
        inverted["date_center"] = list(columns["date_center"])

    if "af" in columns:
        inverted["af"] = [1.0 - x for x in columns["af"]]
    if "af_low" in columns and "af_up" in columns:
        inverted["af_low"] = [1.0 - x for x in columns["af_up"]]
        inverted["af_up"] = [1.0 - x for x in columns["af_low"]]

    if "pt" in columns:
        inverted["pt"] = [1.0 - x for x in columns["pt"]]
    if "pt_low" in columns and "pt_up" in columns:
        inverted["pt_low"] = [1.0 - x for x in columns["pt_up"]]
        inverted["pt_up"] = [1.0 - x for x in columns["pt_low"]]

    return inverted


# -----------------------------------------------------------------------------
# Analysis Helpers
# -----------------------------------------------------------------------------

def _prepare_interpolator(
    dates: List[float], values: List[float]
) -> Tuple[List[float], List[float]]:
    paired = sorted(zip(dates, values))
    return [d for d, _ in paired], [v for _, v in paired]


def _interpolate(date: float, dates: List[float], values: List[float]) -> float:
    if date <= dates[0]: return values[0]
    if date >= dates[-1]: return values[-1]
    idx = bisect_left(dates, date)
    if dates[idx] == date: return values[idx]
    
    left_date, right_date = dates[idx - 1], dates[idx]
    left_val, right_val = values[idx - 1], values[idx]
    
    span = right_date - left_date
    if span == 0: return left_val
    weight = (date - left_date) / span
    return left_val + weight * (right_val - left_val)


def _find_largest_window_change(
    dates: List[float],
    values: List[float],
    window: float,
) -> Optional[Tuple[float, float, float]]:
    if not dates: return None
    sorted_dates, sorted_values = _prepare_interpolator(dates, values)
    min_date, max_date = sorted_dates[0], sorted_dates[-1]
    if max_date - min_date < window: return None

    best_start, best_change, best_end = None, -1.0, None
    start = min_date
    while start <= max_date - window:
        end = start + window
        s_val = _interpolate(start, sorted_dates, sorted_values)
        e_val = _interpolate(end, sorted_dates, sorted_values)
        change = abs(e_val - s_val)
        if change > best_change:
            best_change, best_start, best_end = change, start, end
        start += 10.0
    
    if best_start is None: return None
    return best_start, best_end, best_change


# -----------------------------------------------------------------------------
# Plotting Logic
# -----------------------------------------------------------------------------

def plot_trajectory_on_axis(
    ax: Axes, 
    columns: Dict[str, List[float]], 
    config: Dict[str, Any],
    show_ylabel: bool,
    show_xlabel: bool,
    is_panel_a: bool
) -> None:
    dates = columns["date_center"]
    
    # 1. Empirical Data
    ax.fill_between(
        dates, columns["af_low"], columns["af_up"],
        color=COLOR_EMPIRICAL_FILL, alpha=0.45, edgecolor="none"
    )
    ax.plot(
        dates, columns["af"],
        color=COLOR_EMPIRICAL_LINE, linewidth=1.5, alpha=0.8
    )

    # 2. Model Data
    ax.fill_between(
        dates, columns["pt_low"], columns["pt_up"],
        color=COLOR_MODEL_FILL, alpha=0.45, edgecolor="none"
    )
    ax.plot(
        dates, columns["pt"],
        color=COLOR_MODEL_LINE, linewidth=2.5
    )

    # 3. Highlight
    highlight = _find_largest_window_change(dates, columns["pt"], window=1000.0)
    if highlight is not None:
        s_yr, e_yr, _ = highlight
        ax.axvspan(
            min(s_yr, e_yr), max(s_yr, e_yr),
            color=COLOR_HIGHLIGHT, alpha=0.4, edgecolor="none"
        )

    # 4. Panel Label (A, B, C, D)
    ax.text(
        -0.1, 1.1, config["panel_label"], 
        transform=ax.transAxes, 
        fontsize=FONT_PANEL_LABEL, 
        fontweight='bold', 
        va='bottom', ha='right'
    )

    # 5. Titles and Labels
    ax.set_title(config["title"], fontsize=FONT_TITLE, fontweight="bold", loc='center')
    
    if show_xlabel:
        ax.set_xlabel("Years before present (BP)", fontsize=FONT_AXIS_LABEL)
    if show_ylabel:
        ax.set_ylabel("Inversion-tagging allele frequency", fontsize=FONT_AXIS_LABEL)

    # 6. Formatting X-Axis (Exactly 14000 to 0)
    def _format_year(value: float, _: float) -> str:
        if abs(value) >= 1000: return f"{value/1000:.0f}k"
        return f"{value:,.0f}"

    ax.xaxis.set_major_formatter(FuncFormatter(_format_year))
    ax.set_xlim(14000, 0) # Force range, Left=14k, Right=0

    # 7. Dynamic Y-Scaling Per Subplot
    # Gather all visible data points to determine bounds
    all_vals = []
    if "af" in columns: all_vals.extend(columns["af"])
    if "pt" in columns: all_vals.extend(columns["pt"])
    if "af_low" in columns: all_vals.extend(columns["af_low"])
    if "af_up" in columns: all_vals.extend(columns["af_up"])
    
    if all_vals:
        ymin, ymax = min(all_vals), max(all_vals)
        yrange = ymax - ymin
        # If range is effectively zero (flat line), give it some room
        if yrange < 0.01: yrange = 0.1
        
        padding = yrange * 0.05
        ax.set_ylim(max(0.0, ymin - padding), min(1.0, ymax + padding))
    else:
        ax.set_ylim(0, 1)

    # 8. Ticks and Grid
    ax.tick_params(axis="both", labelsize=FONT_TICKS)
    ax.grid(False) # No grid lines

    # 9. Legend (Only for Panel A)
    if is_panel_a:
        legend_elements = [
            Line2D([0], [0], color=COLOR_EMPIRICAL_LINE, lw=2, label='Empirical frequency'),
            Line2D([0], [0], color=COLOR_MODEL_LINE, lw=2, label='Model frequency'),
            Patch(facecolor=COLOR_HIGHLIGHT, alpha=0.4, label='Max 1 kyr change'),
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            frameon=True,
            fontsize=FONT_LEGEND,
            framealpha=0.9
        )


def main() -> None:
    print("Generating combined inversion trajectory plot...")
    
    plt.style.use("seaborn-v0_8-white") # Clean style, no grid default
    
    # Constrained layout helps with the large text sizes
    fig, axes = plt.subplots(
        nrows=2, ncols=2, 
        figsize=(16, 12), 
        constrained_layout=True
    )
    
    axes_flat = axes.flatten()

    for idx, config in enumerate(SUBPLOTS_CONFIG):
        ax = axes_flat[idx]
        filename = config["filename"]
        
        row, col = divmod(idx, 2)
        show_ylabel = (col == 0)
        show_xlabel = (row == 1)
        is_panel_a = (idx == 0)
        
        print(f"Processing Panel {config['panel_label']}: {filename}")
        
        try:
            rows = load_trajectory(filename)
            columns = rows_to_columns(rows)
            
            if config["flip"]:
                print("  -> Flipping allele frequencies")
                columns = invert_allele_frequencies(columns)
                
            plot_trajectory_on_axis(
                ax, columns, config, 
                show_ylabel, show_xlabel, is_panel_a
            )
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            ax.text(0.5, 0.5, "Data Unavailable", ha='center', fontsize=20, color='red')
            ax.set_title(config["title"], fontsize=FONT_TITLE)
            ax.set_axis_off()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=300)
    print(f"\n✓ Saved 4-panel figure to {OUTPUT_FILE.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()
