"""Shared utilities for PheWAS volcano plots.

This module contains common functions used by both volcano.py and ranged_volcano.py
to ensure consistent styling, colors, and behavior across all volcano plots.
"""

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

# --------------------------- Appearance Constants ---------------------------

NON_SIG_SIZE = 18
NON_SIG_ALPHA = 0.25  # More transparent for non-significant points
NON_SIG_DESAT = 0.70  # Desaturation factor (0=gray, 1=original color)
SIG_POINT_SIZE = 85
EXTREME_ARROW_SIZE = 110

# --------------------------- Color Functions ---------------------------

def desaturate_color(color, desat_factor=0.70):
    """
    Desaturate a color by blending it toward gray.
    
    Parameters
    ----------
    color : color-like
        Input color (any format matplotlib accepts)
    desat_factor : float, default=0.70
        Desaturation factor: 0.0 = fully gray, 1.0 = original color
        
    Returns
    -------
    tuple
        RGB tuple (r, g, b) with values in [0, 1]
    """
    import colorsys
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    # Reduce saturation
    s = s * desat_factor
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (r2, g2, b2)


def non_orange_colors(n, seed=21):
    """Generate n highly distinct colors using the full color spectrum.

    Parameters
    ----------
    n : int
        Number of colors to generate
    seed : int, default=21
        Random seed (currently unused, kept for API compatibility)

    Returns
    -------
    list of tuple
        List of RGB tuples
    """
    if n <= 0:
        return []
    
    # Custom color mapping for specific inversions (sorted order):
    # 0: chr10 -> Purple
    # 1: chr12 -> Orange
    # 2: chr17 -> Green
    # 3: chr4 -> Red
    # 4: chr6:141,866,310-141,898,728 -> Blue
    # 5: chr6:167,209,001-167,357,782 -> Brown
    distinct_colors = [
        (0.58, 0.24, 0.75),  # Purple (chr10)
        (1.00, 0.55, 0.00),  # Orange (chr12)
        (0.20, 0.70, 0.20),  # Green (chr17)
        (0.89, 0.10, 0.11),  # Red (chr4)
        (0.12, 0.47, 0.80),  # Blue (chr6 first)
        (0.60, 0.40, 0.20),  # Brown (chr6 second)
        (0.99, 0.75, 0.05),  # Gold
        (0.00, 0.62, 0.45),  # Teal
        (0.90, 0.10, 0.29),  # Crimson
        (0.55, 0.34, 0.29),  # Tan
        (0.13, 0.55, 0.13),  # Forest Green
        (0.00, 0.45, 0.70),  # Steel Blue
        (0.83, 0.15, 0.16),  # Fire Brick
        (0.42, 0.24, 0.60),  # Dark Violet
        (0.72, 0.53, 0.04),  # Dark Goldenrod
    ]
    
    if n <= len(distinct_colors):
        return [distinct_colors[i] for i in range(n)]
    
    # For more than predefined colors, generate additional ones
    cols = list(distinct_colors)
    for i in range(len(distinct_colors), n):
        # Use golden ratio for hue distribution to maximize perceptual difference
        h = (i * 0.618033988749895) % 1.0
        # Alternate between high and low saturation/value
        if i % 2 == 0:
            s, v = 0.85, 0.75
        else:
            s, v = 0.65, 0.90
        cols.append(tuple(mcolors.hsv_to_rgb((h, s, v))))
    
    return cols


def assign_colors_and_markers(levels):
    """Assign consistent colors and markers to inversion levels.
    
    Parameters
    ----------
    levels : list
        List of inversion level identifiers
        
    Returns
    -------
    color_map : dict
        Mapping from level to RGB color tuple
    marker_map : dict
        Mapping from level to matplotlib marker string
    """
    n = len(levels)
    colors = non_orange_colors(n)
    # Don't rotate colors - keep direct mapping
    marker_cycle = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'h', 'H', 'd', 'p', '8']
    marker_map = {lvl: marker_cycle[i % len(marker_cycle)] for i, lvl in enumerate(levels)}
    color_map  = {lvl: colors[i] for i, lvl in enumerate(levels)}
    return color_map, marker_map


# --------------------------- Statistics ---------------------------

def bh_fdr_cutoff(pvals, alpha=0.05):
    """Calculate Benjamini-Hochberg FDR cutoff p-value.
    
    Parameters
    ----------
    pvals : array-like
        Array of p-values
    alpha : float, default=0.05
        FDR level
        
    Returns
    -------
    float or nan
        Maximum p-value that passes FDR threshold, or nan if none pass
    """
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p)]
    m = p.size
    if m == 0:
        return np.nan
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    crit = ranks / m * alpha
    ok = p_sorted <= crit
    if not np.any(ok):
        return np.nan
    return p_sorted[np.where(ok)[0].max()]


# --------------------------- Legend Creation ---------------------------

def create_legend_handles(inv_levels, color_map, marker_map, fdr_label, y_fdr):
    """Create consistent legend handles for volcano plots.
    
    Parameters
    ----------
    inv_levels : list
        All inversion levels
    color_map : dict
        Mapping from level to color
    marker_map : dict
        Mapping from level to marker
    fdr_label : str
        Label for FDR line
    y_fdr : float
        FDR threshold value (or nan)
        
    Returns
    -------
    handles : list
        List of legend handle objects
    ncol : int
        Recommended number of columns for legend
    """
    # Create inversion handles
    inv_handles = [
        Line2D([], [], linestyle='None', marker=marker_map[inv], markersize=9,
               markerfacecolor=color_map[inv], markeredgecolor="black", markeredgewidth=0.6,
               label=str(inv))
        for inv in inv_levels
    ]

    # Create non-significant handle (use first inversion color as example)
    example_non_sig_color = desaturate_color(
        color_map[inv_levels[0]], NON_SIG_DESAT
    ) if inv_levels else '#b8b8b8'

    non_sig_handle = Line2D(
        [], [], linestyle='None', marker='o', markersize=6,
        markerfacecolor=example_non_sig_color, markeredgecolor='none', alpha=NON_SIG_ALPHA,
        label='Not significant (desaturated)'
    )
    
    # Create FDR handle
    fdr_handle = Line2D([], [], linestyle=':', color='black', linewidth=1.2, label=fdr_label)
    
    # Assemble handles in consistent order
    handles = [non_sig_handle] + inv_handles + ([fdr_handle] if np.isfinite(y_fdr) else [])
    
    # Determine number of columns
    n_items = len(handles)
    ncol = 1 if n_items <= 12 else (2 if n_items <= 30 else 3)
    
    return handles, ncol
