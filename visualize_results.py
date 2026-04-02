#!/usr/bin/env python3
"""
gem5 Cache Simulation Results Visualization Script

This script parses gem5 simulation results and generates publication-quality plots
for analyzing cache performance metrics across different configurations.

Metrics plotted:
- Miss Rate (L1D, L2, L3)
- MPKI (Misses Per Kilo Instructions)
- IPC (Instructions Per Cycle) / CPI (Cycles Per Instruction)
- Memory Stalls (total miss latency as proxy)
- Relative Speedup

Parameters varied:
- Cache sizes (predefined configurations)
- Cache line sizes
- L1/L2/L3 associativity
"""

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-not-found]
import scienceplots  # type: ignore[import-not-found]
from matplotlib import cm
from matplotlib import colors as mcolors

assert (
    scienceplots  # ensure the imported style package is recognized as used by linters
)


# Preferred style list (used with context where possible)
STYLE_LIST: List[str] = ["science", "ieee"]

# X-axis label mapping used across all figures
XLABEL_MAP: Dict[str, str] = {
    "cache_config": "",  # intentionally empty: categories already label the x-axis
    "cache_line": "Cache Line Size (bytes)",
    "l1_assoc": "L1 Associativity (ways)",
    "l2_assoc": "L2 Associativity (ways)",
    "l3_assoc": "L3 Associativity (ways)",
}


def validate_active_style() -> None:
    """Validate that the currently active style (plt.rcParams) meets IEEE rules.

    Raises RuntimeError on failure so the caller can fail-fast within a style context.
    """
    prop: Any = plt.rcParams.get("axes.prop_cycle")
    if prop is None:
        raise RuntimeError(
            "Active matplotlib style must define an 'axes.prop_cycle' with colors. Use SciencePlots 'science'+'ieee'."
        )

    try:
        prop_by_key = prop.by_key()
    except Exception:
        raise RuntimeError(
            "Failed to read 'axes.prop_cycle' from the active style; ensure it's a cycler with named keys."
        )

    if "color" not in prop_by_key:
        raise RuntimeError(
            "Active style must provide a 'color' cycle in 'axes.prop_cycle'. No default colors are allowed."
        )

    # Ensure LaTeX and Palatino are enabled
    if not plt.rcParams.get("text.usetex", False):
        raise RuntimeError(
            "Active style must enable LaTeX (rcParam 'text.usetex'=True). Install a TeX distribution and use a style that enables it."
        )

    font_serif = plt.rcParams.get("font.serif", [])
    serif_lower = [str(f).lower() for f in font_serif]
    if not any(("palatino" in f or "times" in f) for f in serif_lower):
        raise RuntimeError(
            "Active style must use a serif font appropriate for IEEE (Palatino or Times)."
        )

    # Ensure either linestyles or markers are present so plots are distinguishable in monochrome
    has_marker = "marker" in prop_by_key
    has_linestyle = "linestyle" in prop_by_key
    if not (has_marker or has_linestyle):
        raise RuntimeError(
            "Active style must provide either 'marker' or 'linestyle' entries in 'axes.prop_cycle' to ensure distinguishable lines in monochrome."
        )


def get_style_params() -> Dict[str, Any]:
    """Read style-derived plotting parameters from plt.rcParams.

    Returns a dict with keys: colors, markers, linestyles, linewidth, markersize, font sizes.
    """
    validate_active_style()
    prop: Any = plt.rcParams.get("axes.prop_cycle")
    prop_by_key = prop.by_key()
    colors = prop_by_key["color"]
    markers = prop_by_key.get("marker", ["o", "s", "^", "D", "v", "<", ">", "p", "h"])
    linestyles = prop_by_key.get("linestyle", None)
    linewidth = plt.rcParams.get("lines.linewidth", 1.5)
    markersize = plt.rcParams.get("lines.markersize", 6)
    font_title = plt.rcParams.get("axes.titlesize", 12)
    font_label = plt.rcParams.get("axes.labelsize", 10)
    font_tick = plt.rcParams.get("xtick.labelsize", 9)
    font_legend = plt.rcParams.get("legend.fontsize", 9)

    return {
        "colors": colors,
        "markers": markers,
        "linestyles": linestyles,
        "linewidth": linewidth,
        "markersize": markersize,
        "font_title": font_title,
        "font_label": font_label,
        "font_tick": font_tick,
        "font_legend": font_legend,
    }


def _series_style_kwargs(
    index: int,
    *,
    linestyles: Optional[Sequence[str]],
    markers: Optional[Sequence[str]],
    line_width: float,
    marker_size: float,
) -> Dict[str, Any]:
    """Compute per-series plot kwargs from style-derived arrays."""
    kw: Dict[str, Any] = {
        "linewidth": float(line_width),
        "markersize": float(marker_size),
    }
    if linestyles:
        kw["linestyle"] = linestyles[index % len(linestyles)]
    if markers:
        kw["marker"] = markers[index % len(markers)]
    return kw


def _hex_color(c: Any) -> str:
    """Normalize any Matplotlib color to a hex string (no alpha)."""
    # mcolors.to_hex safely handles named colors, RGB(A) tuples, etc.
    return mcolors.to_hex(c, keep_alpha=False).lower()


def expand_color_cycle(base_colors: Sequence[Any], required_count: int) -> List[str]:
    """Return a color list at least `required_count` long.

    - Preserves the original style colors order for the first N entries
    - Appends additional distinct colors sourced from qualitative colormaps
      (tab10/tab20/Set2/Dark2/Paired/Accent)
    - Falls back to simple HSV sampling if still short
    This keeps IEEE style intact while adding extra distinct colors when
    comparing > len(style.colors) series (e.g., 5 workloads vs 4 colors).
    """
    palette: List[str] = [_hex_color(c) for c in base_colors]
    if len(palette) >= required_count:
        return palette[:required_count]

    candidate_cmaps = ["tab10", "tab20", "Set2", "Dark2", "Paired", "Accent"]
    for cmap_name in candidate_cmaps:
        try:
            cmap = cm.get_cmap(cmap_name)
        except Exception:
            continue
        n = getattr(cmap, "N", 10)
        # Sample all discrete entries in the map
        for i in range(n):
            col = cmap(i / max(1, n - 1))
            hx = _hex_color(col)
            if hx not in palette:
                palette.append(hx)
                if len(palette) >= required_count:
                    return palette

    # Fallback: generate additional colors by sampling HSV with golden ratio
    # This ensures reasonably spaced hues if qualitative maps were insufficient
    phi = 0.61803398875
    h = 0.0
    while len(palette) < required_count:
        h = (h + phi) % 1.0
        rgb = mcolors.hsv_to_rgb((h, 0.55, 0.9))
        hx = _hex_color(tuple(rgb))
        if hx not in palette:
            palette.append(hx)
    return palette


# Cache size configurations from cache_config.py (total cache size for sorting)
CACHE_SIZE_CONFIGS = {
    "baseline": ("32KiB", "32KiB", "256KiB", "8MiB"),
    "intel_core_i9_9900k": ("32KiB", "32KiB", "256KiB", "16MiB"),
    "amd_ryzen_5600x": ("32KiB", "32KiB", "512KiB", "32MiB"),
    "amd_ryzen_7700x": ("32KiB", "32KiB", "1MiB", "32MiB"),
    "apple_m1": ("64KiB", "64KiB", "4MiB", "8MiB"),
    "apple_m2": ("64KiB", "64KiB", "4MiB", "16MiB"),
    "intel_atom": ("32KiB", "32KiB", "1MiB", "4MiB"),
    "arm_cortex_a78": ("32KiB", "32KiB", "512KiB", "4MiB"),
    "ibm_power10": ("32KiB", "32KiB", "2MiB", "8MiB"),
    "small_embedded": ("16KiB", "16KiB", "128KiB", "1MiB"),
    "large_server": ("64KiB", "64KiB", "2MiB", "64MiB"),
}

# Display names for configurations
CONFIG_DISPLAY_NAMES = {
    "baseline": "Baseline",
    "intel_core_i9_9900k": "i9-9900K",
    "amd_ryzen_5600x": "Ryzen 5600X",
    "amd_ryzen_7700x": "Ryzen 7700X",
    "apple_m1": "Apple M1",
    "apple_m2": "Apple M2",
    "intel_atom": "Intel Atom",
    "arm_cortex_a78": "Cortex A78",
    "ibm_power10": "POWER10",
    "small_embedded": "Small Embedded",
    "large_server": "Large Server",
}

# Polybench workloads (those with corresponding .h files)
POLYBENCH_WORKLOADS = ["atax", "floyd-warshall", "gemm", "jacobi-2d", "seidel-2d"]

# Parameter types and their values
CACHE_LINE_SIZES = [32, 64, 128, 256]
ASSOCIATIVITIES = [1, 2, 4, 8, 16]


@dataclass
class CacheStats:
    """Statistics for a single cache level."""

    hits: int = 0
    misses: int = 0
    miss_rate: float = 0.0
    miss_latency: float = 0.0  # Total miss latency in ticks


@dataclass
class SimulationStats:
    """Complete statistics from a single simulation run."""

    workload: str = ""
    config_type: str = (
        ""  # 'cache_config', 'cache_line', 'l1_assoc', 'l2_assoc', 'l3_assoc', 'baseline'
    )
    config_value: str = ""  # The actual value

    # CPU stats
    sim_insts: int = 0
    sim_cycles: int = 0
    ipc: float = 0.0
    cpi: float = 0.0

    # Cache stats
    l1d_cache: CacheStats = field(default_factory=CacheStats)
    l1i_cache: CacheStats = field(default_factory=CacheStats)
    l2_cache: CacheStats = field(default_factory=CacheStats)
    l3_cache: CacheStats = field(default_factory=CacheStats)

    # Computed metrics
    l1d_mpki: float = 0.0
    l2_mpki: float = 0.0
    l3_mpki: float = 0.0
    total_miss_latency: float = 0.0  # Sum of all cache miss latencies


def parse_size_to_bytes(size_str: str) -> int:
    """Convert size string (e.g., '32KiB', '8MiB') to bytes."""
    size_str = size_str.strip()
    match = re.match(r"(\d+)\s*(KiB|MiB|GiB|KB|MB|GB)", size_str, re.IGNORECASE)
    if not match:
        return 0
    value = int(match.group(1))
    unit = match.group(2).upper()
    multipliers = {
        "KIB": 1024,
        "KB": 1024,
        "MIB": 1024**2,
        "MB": 1024**2,
        "GIB": 1024**3,
        "GB": 1024**3,
    }
    return value * multipliers.get(unit, 1)


def get_total_cache_size(config_name: str) -> int:
    """Get total cache size for a configuration for sorting."""
    if config_name not in CACHE_SIZE_CONFIGS:
        return 0
    sizes = CACHE_SIZE_CONFIGS[config_name]
    return sum(parse_size_to_bytes(s) for s in sizes)


def parse_stats_file(stats_path: Path) -> Optional[SimulationStats]:
    """Parse a gem5 stats.txt file and extract relevant metrics."""
    if not stats_path.exists():
        return None

    stats = SimulationStats()

    # Parse directory name to get workload and config
    dir_name = stats_path.parent.name

    # Extract workload and configuration from directory name
    # Format: workload_configtype_value or workload_baseline
    for workload in POLYBENCH_WORKLOADS:
        if dir_name.startswith(workload + "_"):
            stats.workload = workload
            rest = dir_name[len(workload) + 1 :]

            if rest == "baseline":
                stats.config_type = "baseline"
                stats.config_value = "baseline"
            elif rest.startswith("cache_config_"):
                stats.config_type = "cache_config"
                stats.config_value = rest[len("cache_config_") :]
            elif rest.startswith("cache_line_"):
                stats.config_type = "cache_line"
                stats.config_value = rest[len("cache_line_") :]
            elif rest.startswith("l1_assoc_"):
                stats.config_type = "l1_assoc"
                stats.config_value = rest[len("l1_assoc_") :]
            elif rest.startswith("l2_assoc_"):
                stats.config_type = "l2_assoc"
                stats.config_value = rest[len("l2_assoc_") :]
            elif rest.startswith("l3_assoc_"):
                stats.config_type = "l3_assoc"
                stats.config_value = rest[len("l3_assoc_") :]
            break

    if not stats.workload:
        return None

    # Read and parse stats file
    with open(stats_path, "r") as f:
        content = f.read()

    # Helper function to extract numeric value
    def extract_value(pattern: str, default: float = 0.0) -> float:
        match = re.search(pattern + r"\s+([\d.]+(?:e[+-]?\d+)?)", content, re.MULTILINE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return default
        return default

    # CPU statistics (use first occurrence - before "Begin Simulation Statistics" repeat)
    stats.sim_insts = int(extract_value(r"^simInsts"))
    stats.sim_cycles = int(extract_value(r"^system\.cpu\.numCycles"))
    stats.ipc = extract_value(r"^system\.cpu\.ipc")
    stats.cpi = extract_value(r"^system\.cpu\.cpi")

    # L1 Data Cache
    stats.l1d_cache.hits = int(
        extract_value(r"^system\.cpu\.dcache\.demandHits::total")
    )
    stats.l1d_cache.misses = int(
        extract_value(r"^system\.cpu\.dcache\.demandMisses::total")
    )
    stats.l1d_cache.miss_rate = extract_value(
        r"^system\.cpu\.dcache\.demandMissRate::total"
    )
    stats.l1d_cache.miss_latency = extract_value(
        r"^system\.cpu\.dcache\.overallMissLatency::total"
    )

    # L1 Instruction Cache
    stats.l1i_cache.hits = int(
        extract_value(r"^system\.cpu\.icache\.demandHits::total")
    )
    stats.l1i_cache.misses = int(
        extract_value(r"^system\.cpu\.icache\.demandMisses::total")
    )
    stats.l1i_cache.miss_rate = extract_value(
        r"^system\.cpu\.icache\.demandMissRate::total"
    )
    stats.l1i_cache.miss_latency = extract_value(
        r"^system\.cpu\.icache\.overallMissLatency::total"
    )

    # L2 Cache (note: L2 may not have demandHits in gem5 depending on configuration)
    stats.l2_cache.hits = int(extract_value(r"^system\.l2cache\.demandHits::total"))
    stats.l2_cache.misses = int(extract_value(r"^system\.l2cache\.demandMisses::total"))
    stats.l2_cache.miss_rate = extract_value(r"^system\.l2cache\.demandMissRate::total")
    stats.l2_cache.miss_latency = extract_value(
        r"^system\.l2cache\.overallMissLatency::total"
    )

    # L3 Cache
    stats.l3_cache.hits = int(extract_value(r"^system\.l3cache\.demandHits::total"))
    stats.l3_cache.misses = int(extract_value(r"^system\.l3cache\.demandMisses::total"))
    stats.l3_cache.miss_rate = extract_value(r"^system\.l3cache\.demandMissRate::total")
    stats.l3_cache.miss_latency = extract_value(
        r"^system\.l3cache\.overallMissLatency::total"
    )

    # Compute MPKI (Misses Per Kilo Instructions)
    if stats.sim_insts > 0:
        stats.l1d_mpki = (stats.l1d_cache.misses / stats.sim_insts) * 1000
        stats.l2_mpki = (stats.l2_cache.misses / stats.sim_insts) * 1000
        stats.l3_mpki = (stats.l3_cache.misses / stats.sim_insts) * 1000

    # Total miss latency (proxy for memory stalls)
    stats.total_miss_latency = (
        stats.l1d_cache.miss_latency
        + stats.l1i_cache.miss_latency
        + stats.l2_cache.miss_latency
        + stats.l3_cache.miss_latency
    )

    return stats


# -----------------------------------------------------------------------------
# Small typed extractors to help static type checkers (avoid unknown-lambda warnings)
# -----------------------------------------------------------------------------


def l1d_miss_rate(s: SimulationStats) -> float:
    return s.l1d_cache.miss_rate


def l2_miss_rate(s: SimulationStats) -> float:
    return s.l2_cache.miss_rate


def l3_miss_rate(s: SimulationStats) -> float:
    return s.l3_cache.miss_rate


def l1d_mpki(s: SimulationStats) -> float:
    return s.l1d_mpki


def l2_mpki(s: SimulationStats) -> float:
    return s.l2_mpki


def l3_mpki(s: SimulationStats) -> float:
    return s.l3_mpki


def collect_all_results(results_dir: Path) -> Dict[str, List[SimulationStats]]:
    """Collect all simulation results, organized by workload."""
    results = defaultdict(list)

    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue

        stats_file = subdir / "stats.txt"
        stats = parse_stats_file(stats_file)

        if stats and stats.workload in POLYBENCH_WORKLOADS:
            results[stats.workload].append(stats)

    return results


def get_sorted_data(
    stats_list: List[SimulationStats], config_type: str
) -> Tuple[Sequence[Union[str, int]], List[SimulationStats]]:
    """Filter and sort data by configuration type."""
    filtered = [s for s in stats_list if s.config_type == config_type]

    if config_type == "cache_config":
        # Sort by total cache size
        filtered.sort(key=lambda s: get_total_cache_size(s.config_value))
        x_values = [
            CONFIG_DISPLAY_NAMES.get(s.config_value, s.config_value) for s in filtered
        ]
    elif config_type == "cache_line":
        # Sort by cache line size (numeric)
        filtered.sort(key=lambda s: int(s.config_value))
        x_values = [int(s.config_value) for s in filtered]
    elif config_type in ["l1_assoc", "l2_assoc", "l3_assoc"]:
        # Sort by associativity (numeric)
        filtered.sort(key=lambda s: int(s.config_value))
        x_values = [int(s.config_value) for s in filtered]
    else:
        x_values = [s.config_value for s in filtered]

    return x_values, filtered


def _delta_vs_baseline(
    filtered: List[SimulationStats], extractor: Callable[[SimulationStats], float]
) -> Tuple[List[float], Optional[int]]:
    """Return absolute deltas to baseline and the baseline value.

    Baseline detection order within the filtered set:
    - config_value == 'baseline'
    - config_type == 'baseline'
    - first element (fallback)
    """
    baseline_idx: Optional[int] = None
    for i, s in enumerate(filtered):
        if s.config_value == "baseline":
            baseline_idx = i
            break
    if baseline_idx is None:
        for i, s in enumerate(filtered):
            if s.config_type == "baseline":
                baseline_idx = i
                break
    base_val = (
        extractor(filtered[baseline_idx])
        if baseline_idx is not None
        else extractor(filtered[0])
    )
    deltas = [extractor(s) - base_val for s in filtered]
    return deltas, baseline_idx


def create_workload_figures(
    workload: str,
    stats_list: List[SimulationStats],
    figures_dir: Path,
) -> None:
    """Create all figures for a single workload."""

    # Get baseline stats for speedup calculation
    baseline_stats = [s for s in stats_list if s.config_type == "baseline"]
    if not baseline_stats:
        # Use first cache_config_baseline if no dedicated baseline
        baseline_stats = [s for s in stats_list if s.config_value == "baseline"]

    baseline_ipc = baseline_stats[0].ipc if baseline_stats else 1.0

    # Parameter types to iterate over
    param_types = ["cache_config", "cache_line", "l1_assoc", "l2_assoc", "l3_assoc"]

    # =========================================================================
    # Figure 1: Miss Rates for each param
    # - For associativity sweeps we only plot the relevant cache level (L1 for l1_assoc,
    #   L2 for l2_assoc, L3 for l3_assoc). For other params (cache_config, cache_line)
    #   we keep all three levels stacked for comparison.
    # =========================================================================
    for param_type in param_types:
        x_values, filtered = get_sorted_data(stats_list, param_type)
        if not filtered:
            continue

        with plt.style.context(STYLE_LIST):
            style = get_style_params()
            colors = style["colors"]
            markers = style["markers"]
            linestyles = style["linestyles"]
            line_width = style["linewidth"]
            marker_size = style["markersize"]

            # Choose which miss-rate series to plot depending on parameter type
            if param_type == "l1_assoc":
                miss_rate_metrics = [("L1D", l1d_miss_rate, colors[0])]
            elif param_type == "l2_assoc":
                miss_rate_metrics = [("L2", l2_miss_rate, colors[1])]
            elif param_type == "l3_assoc":
                miss_rate_metrics = [("L3", l3_miss_rate, colors[2])]
            elif param_type == "cache_line":
                # For cache_line: put all three levels on a single plot with log scale
                miss_rate_metrics = [
                    ("L1D", l1d_miss_rate, colors[0]),
                    ("L2", l2_miss_rate, colors[1]),
                    ("L3", l3_miss_rate, colors[2]),
                ]
            else:
                # cache_config: stacked subplots per level (deltas vs baseline)
                miss_rate_metrics = [
                    ("L1D", l1d_miss_rate, colors[0]),
                    ("L2", l2_miss_rate, colors[1]),
                    ("L3", l3_miss_rate, colors[2]),
                ]

            is_categorical = param_type == "cache_config"

            # Special-case cache_line: single combined line plot with log y-scale
            if param_type == "cache_line":
                fig, ax = plt.subplots(figsize=(3.45, 2.5))
                for i, (name, extractor, color) in enumerate(miss_rate_metrics):
                    y_values = [extractor(s) for s in filtered]
                    kwargs = _series_style_kwargs(
                        i,
                        linestyles=linestyles,
                        markers=markers,
                        line_width=line_width,
                        marker_size=marker_size,
                    )
                    ax.plot(
                        [float(x) for x in x_values],
                        y_values,
                        color=color,
                        label=name,
                        **kwargs,
                    )
                ax.set_yscale("log")
                ax.set_ylabel("Miss Rate")
                xlabel = XLABEL_MAP.get(param_type, param_type)
                if xlabel:
                    ax.set_xlabel(xlabel)
                ax.legend(loc="best")
                plt.tight_layout()
                fig.savefig(figures_dir / f"{workload}_miss_rate_vs_{param_type}.pdf")
                fig.savefig(figures_dir / f"{workload}_miss_rate_vs_{param_type}.png")
                plt.close(fig)
                continue

            # Default path: one subplot per metric (used for cache_config and per-level assoc)
            n_metrics = len(miss_rate_metrics)
            fig, axes = plt.subplots(
                n_metrics,
                1,
                figsize=(3.45, 2.5 * n_metrics + 1.0),
                sharex=(param_type != "cache_config"),
            )

            # Ensure axes is always indexable
            if n_metrics == 1:
                axes = np.atleast_1d(axes)

            for idx, (name, extractor, color) in enumerate(miss_rate_metrics):
                ax = axes[idx]
                if is_categorical:
                    deltas, baseline_idx = _delta_vs_baseline(filtered, extractor)
                    if baseline_idx is not None:
                        x_vals_plot = [
                            v for i, v in enumerate(x_values) if i != baseline_idx
                        ]
                        deltas_plot = [
                            d for i, d in enumerate(deltas) if i != baseline_idx
                        ]
                    else:
                        x_vals_plot = x_values
                        deltas_plot = deltas
                    x_numeric = np.arange(len(x_vals_plot))
                    ax.bar(
                        x_numeric,
                        deltas_plot,
                        color=color,
                        alpha=0.9,
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
                    ax.set_xticks(x_numeric)
                    ax.set_xticklabels(
                        [str(x) for x in x_vals_plot], rotation=45, ha="right"
                    )
                else:
                    y_values = [extractor(s) for s in filtered]
                    kwargs = _series_style_kwargs(
                        idx,
                        linestyles=linestyles,
                        markers=markers,
                        line_width=line_width,
                        marker_size=marker_size,
                    )
                    ax.plot(
                        [float(x) for x in x_values], y_values, color=color, **kwargs
                    )

                ax.set_ylabel(
                    (f"$\\Delta$ {name} Miss Rate vs baseline")
                    if is_categorical
                    else f"{name} Miss Rate"
                )

            # Set x-label on bottom subplot
            xlabel = XLABEL_MAP.get(param_type, param_type)
            if xlabel:
                axes[-1].set_xlabel(xlabel)

            plt.tight_layout()
            fig.savefig(figures_dir / f"{workload}_miss_rate_vs_{param_type}.pdf")
            fig.savefig(figures_dir / f"{workload}_miss_rate_vs_{param_type}.png")
            plt.close(fig)

    # =========================================================================
    # Figure 2: MPKI for each param
    # - For associativity sweeps only plot the corresponding level's MPKI.
    # =========================================================================
    for param_type in param_types:
        x_values, filtered = get_sorted_data(stats_list, param_type)
        if not filtered:
            continue

        with plt.style.context(STYLE_LIST):
            style = get_style_params()
            colors = style["colors"]
            markers = style["markers"]
            linestyles = style["linestyles"]
            line_width = style["linewidth"]
            marker_size = style["markersize"]

            if param_type == "l1_assoc":
                mpki_metrics = [("L1D", l1d_mpki, colors[0])]
            elif param_type == "l2_assoc":
                mpki_metrics = [("L2", l2_mpki, colors[1])]
            elif param_type == "l3_assoc":
                mpki_metrics = [("L3", l3_mpki, colors[2])]
            elif param_type == "cache_line":
                # For cache_line: single combined plot with three MPKI series on log-scale
                mpki_metrics = [
                    ("L1D", l1d_mpki, colors[0]),
                    ("L2", l2_mpki, colors[1]),
                    ("L3", l3_mpki, colors[2]),
                ]
            else:
                mpki_metrics = [
                    ("L1D", l1d_mpki, colors[0]),
                    ("L2", l2_mpki, colors[1]),
                    ("L3", l3_mpki, colors[2]),
                ]

            is_categorical = param_type == "cache_config"

            # Special-case cache_line: draw all MPKI series on one plot with log y-scale
            if param_type == "cache_line":
                fig, ax = plt.subplots(figsize=(3.45, 2.5))
                for i, (name, extractor, color) in enumerate(mpki_metrics):
                    y_values = [extractor(s) for s in filtered]
                    kwargs = _series_style_kwargs(
                        i,
                        linestyles=linestyles,
                        markers=markers,
                        line_width=line_width,
                        marker_size=marker_size,
                    )
                    ax.plot(
                        [float(x) for x in x_values],
                        y_values,
                        color=color,
                        label=name,
                        **kwargs,
                    )
                ax.set_yscale("log")
                ax.set_ylabel("MPKI")
                xlabel = XLABEL_MAP.get(param_type, param_type)
                if xlabel:
                    ax.set_xlabel(xlabel)
                ax.legend(loc="best")
                plt.tight_layout()
                fig.savefig(figures_dir / f"{workload}_mpki_vs_{param_type}.pdf")
                fig.savefig(figures_dir / f"{workload}_mpki_vs_{param_type}.png")
                plt.close(fig)
                continue

            # Default: stacked subplots for categorical or per-level plots
            n_metrics = len(mpki_metrics)
            fig, axes = plt.subplots(
                n_metrics,
                1,
                figsize=(3.45, 2.5 * n_metrics + 1.0),
                sharex=(param_type != "cache_config"),
            )
            if n_metrics == 1:
                axes = np.atleast_1d(axes)

            for idx, (name, extractor, color) in enumerate(mpki_metrics):
                ax = axes[idx]
                if is_categorical:
                    deltas, baseline_idx = _delta_vs_baseline(filtered, extractor)
                    if baseline_idx is not None:
                        x_vals_plot = [
                            v for i, v in enumerate(x_values) if i != baseline_idx
                        ]
                        deltas_plot = [
                            d for i, d in enumerate(deltas) if i != baseline_idx
                        ]
                    else:
                        x_vals_plot = x_values
                        deltas_plot = deltas
                    x_numeric = np.arange(len(x_vals_plot))
                    ax.bar(
                        x_numeric,
                        deltas_plot,
                        color=color,
                        alpha=0.9,
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
                    ax.set_xticks(x_numeric)
                    ax.set_xticklabels(
                        [str(x) for x in x_vals_plot], rotation=45, ha="right"
                    )
                else:
                    y_values = [extractor(s) for s in filtered]
                    kwargs = _series_style_kwargs(
                        idx,
                        linestyles=linestyles,
                        markers=markers,
                        line_width=line_width,
                        marker_size=marker_size,
                    )
                    ax.plot(
                        [float(x) for x in x_values], y_values, color=color, **kwargs
                    )

                ax.set_ylabel(
                    (f"$\\Delta$ {name} MPKI vs baseline")
                    if is_categorical
                    else f"{name} MPKI"
                )

            xlabel = XLABEL_MAP.get(param_type, param_type)
            if xlabel:
                axes[-1].set_xlabel(xlabel)

            plt.tight_layout()
            fig.savefig(figures_dir / f"{workload}_mpki_vs_{param_type}.pdf")
            fig.savefig(figures_dir / f"{workload}_mpki_vs_{param_type}.png")
            plt.close(fig)

    # =========================================================================
    # Figure 3: CPI only (single plot) for each param
    # =========================================================================
    for param_type in param_types:
        x_values, filtered = get_sorted_data(stats_list, param_type)
        if not filtered:
            continue

        with plt.style.context(STYLE_LIST):
            style = get_style_params()
            colors = style["colors"]
            markers = style["markers"]
            linestyles = style["linestyles"]
            line_width = style["linewidth"]
            marker_size = style["markersize"]

            fig, ax = plt.subplots(figsize=(3.45, 2.5))

            is_categorical = param_type == "cache_config"

            if is_categorical:
                deltas, baseline_idx = _delta_vs_baseline(filtered, lambda s: s.cpi)
                if baseline_idx is not None:
                    x_vals_plot = [
                        v for i, v in enumerate(x_values) if i != baseline_idx
                    ]
                    deltas_plot = [d for i, d in enumerate(deltas) if i != baseline_idx]
                else:
                    x_vals_plot = x_values
                    deltas_plot = deltas
                x_numeric = np.arange(len(x_vals_plot))
                ax.bar(
                    x_numeric,
                    deltas_plot,
                    color=colors[0],
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
                ax.set_xticks(x_numeric)
                ax.set_xticklabels(
                    [str(x) for x in x_vals_plot], rotation=45, ha="right"
                )
            else:
                # Numeric parameter: line plot of CPI
                y_values = [s.cpi for s in filtered]
                kwargs = _series_style_kwargs(
                    0,
                    linestyles=linestyles,
                    markers=markers,
                    line_width=line_width,
                    marker_size=marker_size,
                )
                ax.plot(
                    [float(x) for x in x_values], y_values, color=colors[0], **kwargs
                )

            ax.set_ylabel(("$\\Delta$ CPI vs baseline") if is_categorical else "CPI")
            xlabel = XLABEL_MAP.get(param_type, param_type)
            if xlabel:
                ax.set_xlabel(xlabel)

            plt.tight_layout()
            fig.savefig(
                figures_dir / f"{workload}_cpi_vs_{param_type}.pdf",
            )
            fig.savefig(
                figures_dir / f"{workload}_cpi_vs_{param_type}.png",
            )
            plt.close(fig)

    # =========================================================================
    # Figure 4: Memory Stalls for each param
    # =========================================================================
    for param_type in param_types:
        x_values, filtered = get_sorted_data(stats_list, param_type)
        if not filtered:
            continue

        with plt.style.context(STYLE_LIST):
            style = get_style_params()
            colors = style["colors"]
            markers = style["markers"]
            linestyles = style["linestyles"]
            line_width = style["linewidth"]
            marker_size = style["markersize"]

            fig, ax = plt.subplots(figsize=(3.45, 2.5))

            is_categorical = param_type == "cache_config"

            if is_categorical:

                def _stall_extractor(s: SimulationStats) -> float:
                    return s.total_miss_latency / 1e9

                deltas, baseline_idx = _delta_vs_baseline(filtered, _stall_extractor)
                if baseline_idx is not None:
                    x_vals_plot = [
                        v for i, v in enumerate(x_values) if i != baseline_idx
                    ]
                    deltas_plot = [d for i, d in enumerate(deltas) if i != baseline_idx]
                else:
                    x_vals_plot = x_values
                    deltas_plot = deltas
                x_numeric = np.arange(len(x_vals_plot))
                ax.bar(
                    x_numeric,
                    deltas_plot,
                    color=colors[0],
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
                ax.set_xticks(x_numeric)
                ax.set_xticklabels(
                    [str(x) for x in x_vals_plot], rotation=45, ha="right"
                )
            else:
                y_values = [
                    s.total_miss_latency / 1e9 for s in filtered
                ]  # Convert to billions
                kwargs = _series_style_kwargs(
                    0,
                    linestyles=linestyles,
                    markers=markers,
                    line_width=line_width,
                    marker_size=marker_size,
                )
                ax.plot(
                    [float(x) for x in x_values], y_values, color=colors[0], **kwargs
                )

            # Use concise y-labels; values are scaled to billions (1e9) of ticks
            ax.set_ylabel(
                ("$\\Delta$ Miss Latency ($10^9$ ticks)")
                if is_categorical
                else "Total miss latency ($10^9$ ticks)"
            )
            xlabel = XLABEL_MAP.get(param_type, param_type)
            if xlabel:
                ax.set_xlabel(xlabel)

            plt.tight_layout()
            fig.savefig(
                figures_dir / f"{workload}_memory_stalls_vs_{param_type}.pdf",
            )
            fig.savefig(
                figures_dir / f"{workload}_memory_stalls_vs_{param_type}.png",
            )
            plt.close(fig)

    # =========================================================================
    # Figure 5: Relative Speedup for each param
    # =========================================================================
    for param_type in param_types:
        x_values, filtered = get_sorted_data(stats_list, param_type)
        if not filtered:
            continue

        with plt.style.context(STYLE_LIST):
            style = get_style_params()
            colors = style["colors"]
            markers = style["markers"]
            linestyles = style["linestyles"]
            line_width = style["linewidth"]
            marker_size = style["markersize"]

            fig, ax = plt.subplots(figsize=(3.45, 2.5))

            is_categorical = param_type == "cache_config"

            if is_categorical:

                def _speedup_extractor(s: SimulationStats) -> float:
                    return (s.ipc / baseline_ipc) if baseline_ipc > 0 else 1.0

                deltas, baseline_idx = _delta_vs_baseline(filtered, _speedup_extractor)
                if baseline_idx is not None:
                    x_vals_plot = [
                        v for i, v in enumerate(x_values) if i != baseline_idx
                    ]
                    deltas_plot = [d for i, d in enumerate(deltas) if i != baseline_idx]
                else:
                    x_vals_plot = x_values
                    deltas_plot = deltas
                x_numeric = np.arange(len(x_vals_plot))
                ax.bar(
                    x_numeric,
                    deltas_plot,
                    color=colors[0],
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.set_xticks(x_numeric)
                ax.set_xticklabels(
                    [str(x) for x in x_vals_plot], rotation=45, ha="right"
                )
            else:
                y_values = [
                    s.ipc / baseline_ipc if baseline_ipc > 0 else 1.0 for s in filtered
                ]
                kwargs = _series_style_kwargs(
                    0,
                    linestyles=linestyles,
                    markers=markers,
                    line_width=line_width,
                    marker_size=marker_size,
                )
                ax.plot(
                    [float(x) for x in x_values], y_values, color=colors[0], **kwargs
                )

            # Reference line only for categorical delta plots; remove for numeric plots
            if is_categorical:
                # Ensure no style grids/gray backgrounds interfere and draw a thin black baseline
                ax.grid(False)
                try:
                    ax.set_facecolor("white")
                except Exception:
                    pass
                for spine in ax.spines.values():
                    spine.set_color("black")
                ax.axhline(
                    y=0.0, color="black", linestyle="-", linewidth=0.5, zorder=10
                )

            ax.set_ylabel(
                "Relative Speedup $(\\frac{\\mathrm{IPC}}{\\mathrm{IPC}_{\\text{baseline}}} - 1)$"
                if is_categorical
                else "Speedup $(\\frac{\\mathrm{IPC}}{\\mathrm{IPC}_{\\text{baseline}}})$"
            )
            xlabel = XLABEL_MAP.get(param_type, param_type)
            if xlabel:
                ax.set_xlabel(xlabel)

            plt.tight_layout()
            fig.savefig(
                figures_dir / f"{workload}_speedup_vs_{param_type}.pdf",
            )
            fig.savefig(
                figures_dir / f"{workload}_speedup_vs_{param_type}.png",
            )
            plt.close(fig)

    # =========================================================================
    # Combined Figure: All Miss Rates on same plot (for comparison)
    # - Only produced for cache_config and cache_line sweeps. For associativity
    #   sweeps, the per-level plots above already show the relevant metric.
    # =========================================================================
    for param_type in param_types:
        # Skip combined multi-level plot for associativity-only sweeps
        if param_type in ("l1_assoc", "l2_assoc", "l3_assoc"):
            continue

        x_values, filtered = get_sorted_data(stats_list, param_type)
        if not filtered:
            continue

        with plt.style.context(STYLE_LIST):
            style = get_style_params()
            colors = style["colors"]
            markers = style["markers"]
            linestyles = style["linestyles"]
            line_width = style["linewidth"]
            marker_size = style["markersize"]

            fig, ax = plt.subplots(figsize=(3.45, 2.5))

            combined_miss_rate_metrics = [
                ("L1D", l1d_miss_rate, colors[0]),
                ("L2", l2_miss_rate, colors[1]),
                ("L3", l3_miss_rate, colors[2]),
            ]

            is_categorical = param_type == "cache_config"

            if is_categorical:
                # Grouped absolute deltas vs baseline for categorical data
                n_groups = len(x_values)
                n_bars = len(combined_miss_rate_metrics)
                bar_width = 0.25
                x_numeric = np.arange(n_groups)

                baseline_idx = next(
                    (
                        i
                        for i, s in enumerate(filtered)
                        if s.config_value == "baseline" or s.config_type == "baseline"
                    ),
                    None,
                )
                if baseline_idx is not None:
                    x_vals_plot = [
                        v for i, v in enumerate(x_values) if i != baseline_idx
                    ]
                else:
                    x_vals_plot = x_values

                x_numeric = np.arange(len(x_vals_plot))

                for i, (name, extractor, color) in enumerate(
                    combined_miss_rate_metrics
                ):
                    deltas, _ = _delta_vs_baseline(filtered, extractor)
                    if baseline_idx is not None:
                        deltas_plot = [
                            d for j, d in enumerate(deltas) if j != baseline_idx
                        ]
                    else:
                        deltas_plot = deltas
                    offset = (i - n_bars / 2 + 0.5) * bar_width
                    ax.bar(
                        x_numeric + offset,
                        deltas_plot,
                        width=bar_width,
                        color=color,
                        label=name,
                        alpha=0.9,
                        edgecolor="black",
                        linewidth=0.5,
                    )

                ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
                ax.set_xticks(x_numeric)
                ax.set_xticklabels(
                    [str(x) for x in x_vals_plot], rotation=45, ha="right"
                )
            else:
                # Use line plot for numeric data
                for i, (name, extractor, color) in enumerate(
                    combined_miss_rate_metrics
                ):
                    y_values = [extractor(s) for s in filtered]
                    kwargs = _series_style_kwargs(
                        i,
                        linestyles=linestyles,
                        markers=markers,
                        line_width=line_width,
                        marker_size=marker_size,
                    )
                    ax.plot(
                        [float(x) for x in x_values],
                        y_values,
                        color=color,
                        label=name,
                        **kwargs,
                    )

            ax.set_ylabel(
                ("$\\Delta$ Miss Rate vs baseline") if is_categorical else "Miss Rate"
            )
            ax.set_yscale("log")
            xlabel = XLABEL_MAP.get(param_type, param_type)
            if xlabel:
                ax.set_xlabel(xlabel)
            ax.legend(loc="best")

            plt.tight_layout()
            fig.savefig(figures_dir / f"{workload}_all_miss_rates_vs_{param_type}.pdf")
            fig.savefig(figures_dir / f"{workload}_all_miss_rates_vs_{param_type}.png")
            plt.close(fig)


def create_comparison_figures(
    all_results: Dict[str, List[SimulationStats]],
    figures_dir: Path,
) -> None:
    """Create comparison figures across all workloads."""

    param_types = ["cache_config", "cache_line", "l1_assoc", "l2_assoc", "l3_assoc"]

    xlabel_map = {
        "cache_config": "Cache Configuration",
        "cache_line": "Cache Line Size (bytes)",
        "l1_assoc": "L1 Associativity (ways)",
        "l2_assoc": "L2 Associativity (ways)",
        "l3_assoc": "L3 Associativity (ways)",
    }

    for param_type in param_types:
        is_categorical = param_type == "cache_config"

        # IPC comparison across workloads
        with plt.style.context(STYLE_LIST):
            st = get_style_params()
            fig, ax = plt.subplots(figsize=(3.45, 2.5))
            if is_categorical:
                all_x_values: Optional[Sequence[Union[str, int]]] = None
                workload_data: Dict[str, List[float]] = {}
                baseline_idx_global: Optional[int] = None
                for workload in POLYBENCH_WORKLOADS:
                    if workload not in all_results:
                        continue
                    x_values, filtered = get_sorted_data(
                        all_results[workload], param_type
                    )
                    if filtered:
                        # Capture the canonical x_values from the first workload encountered
                        if all_x_values is None:
                            all_x_values = x_values
                            baseline_idx_global = next(
                                (
                                    i
                                    for i, s in enumerate(filtered)
                                    if s.config_value == "baseline"
                                    or s.config_type == "baseline"
                                ),
                                None,
                            )

                        def _ipc_extractor(s: SimulationStats) -> float:
                            return s.ipc

                        y_vals, _ = _delta_vs_baseline(filtered, _ipc_extractor)
                        workload_data[workload] = y_vals

                if all_x_values and workload_data:
                    # Remove baseline column from x and from each workload's y-values
                    if baseline_idx_global is not None:
                        x_vals_plot = [
                            v
                            for i, v in enumerate(all_x_values)
                            if i != baseline_idx_global
                        ]
                    else:
                        x_vals_plot = list(all_x_values)

                    n_groups = len(x_vals_plot)
                    n_bars = len(workload_data)
                    bar_width = 0.8 / n_bars
                    x_numeric = np.arange(n_groups)
                    # Extend IEEE style colors to cover all workloads if needed
                    st["colors"] = expand_color_cycle(st["colors"], n_bars)
                    for idx, (workload, y_values) in enumerate(workload_data.items()):
                        if baseline_idx_global is not None:
                            y_plot = [
                                yv
                                for i, yv in enumerate(y_values)
                                if i != baseline_idx_global
                            ]
                        else:
                            y_plot = y_values
                        offset = (idx - n_bars / 2 + 0.5) * bar_width
                        ax.bar(
                            x_numeric + offset,
                            y_plot,
                            width=bar_width,
                            color=st["colors"][idx % len(st["colors"])],
                            label=workload,
                            alpha=0.8,
                            edgecolor="black",
                            linewidth=0.5,
                        )
                    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
                    ax.set_xticks(x_numeric)
                    ax.set_xticklabels(
                        [str(x) for x in x_vals_plot], rotation=45, ha="right"
                    )
            else:
                for idx, workload in enumerate(POLYBENCH_WORKLOADS):
                    if workload not in all_results:
                        continue
                    x_values, filtered = get_sorted_data(
                        all_results[workload], param_type
                    )
                    if not filtered:
                        continue
                    y_values = [s.ipc for s in filtered]
                    kw = _series_style_kwargs(
                        idx,
                        linestyles=st["linestyles"],
                        markers=st["markers"],
                        line_width=st["linewidth"],
                        marker_size=st["markersize"],
                    )
                    ax.plot(
                        x_values,
                        y_values,
                        color=st["colors"][idx % len(st["colors"])],
                        label=workload,
                        **kw,
                    )

            ax.set_ylabel(("$\\Delta$ IPC vs baseline") if is_categorical else "IPC")
            xlabel = XLABEL_MAP.get(param_type, param_type)
            if xlabel:
                ax.set_xlabel(xlabel)
            ax.legend(loc="best")
            plt.tight_layout()
            fig.savefig(
                figures_dir / f"comparison_ipc_vs_{param_type}.pdf",
            )
            fig.savefig(
                figures_dir / f"comparison_ipc_vs_{param_type}.png",
            )
            plt.close(fig)

        # L1D Miss Rate comparison across workloads
        # Skip generating L1D comparison plots for parameter types that are
        # unlikely to affect L1D (e.g. changing L3 associativity). This avoids
        # producing misleading or irrelevant files such as
        # 'comparison_l1d_miss_rate_vs_l3_assoc'. Keep only params that can
        # meaningfully affect L1D: cache_config, cache_line, l1_assoc.
        if param_type not in ("cache_config", "cache_line", "l1_assoc"):
            continue

        # NOTE: The loop above also produces IPC comparison figures; those are
        # still emitted for all param_types because IPC can be influenced by
        # any cache-level change. We only early-continue for the L1D miss-rate
        # comparison block below, so ensure IPC block remains above this check
        # if you reorder code.

        # L1D Miss Rate comparison across workloads
        with plt.style.context(STYLE_LIST):
            st = get_style_params()
            fig, ax = plt.subplots(figsize=(3.45, 2.5))

            if is_categorical:
                all_x_values = None
                workload_data = {}
                for workload in POLYBENCH_WORKLOADS:
                    if workload not in all_results:
                        continue
                    x_values, filtered = get_sorted_data(
                        all_results[workload], param_type
                    )
                    if filtered:
                        all_x_values = x_values

                        def _mr_extractor(s: SimulationStats) -> float:
                            return s.l1d_cache.miss_rate

                        y_vals, _ = _delta_vs_baseline(filtered, _mr_extractor)
                        workload_data[workload] = y_vals
                if all_x_values and workload_data:
                    n_groups = len(all_x_values)
                    n_bars = len(workload_data)
                    bar_width = 0.8 / n_bars
                    x_numeric = np.arange(n_groups)
                    # Extend IEEE style colors to cover all workloads if needed
                    st["colors"] = expand_color_cycle(st["colors"], n_bars)
                    for idx, (workload, y_values) in enumerate(workload_data.items()):
                        offset = (idx - n_bars / 2 + 0.5) * bar_width
                        ax.bar(
                            x_numeric + offset,
                            y_values,
                            width=bar_width,
                            color=st["colors"][idx % len(st["colors"])],
                            label=workload,
                            alpha=0.8,
                            edgecolor="black",
                            linewidth=0.5,
                        )
                    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
                    ax.set_xticks(x_numeric)
                    ax.set_xticklabels(
                        [str(x) for x in all_x_values], rotation=45, ha="right"
                    )
            else:
                for idx, workload in enumerate(POLYBENCH_WORKLOADS):
                    if workload not in all_results:
                        continue
                    x_values, filtered = get_sorted_data(
                        all_results[workload], param_type
                    )
                    if not filtered:
                        continue
                    y_values = [s.l1d_cache.miss_rate for s in filtered]
                    kw = _series_style_kwargs(
                        idx,
                        linestyles=st["linestyles"],
                        markers=st["markers"],
                        line_width=st["linewidth"],
                        marker_size=st["markersize"],
                    )
                    ax.plot(
                        x_values,
                        y_values,
                        color=st["colors"][idx % len(st["colors"])],
                        label=workload,
                        **kw,
                    )

            ax.set_ylabel(
                ("$\\Delta$ L1D Miss Rate vs baseline")
                if is_categorical
                else "L1D Miss Rate"
            )
            ax.set_xlabel(xlabel_map.get(param_type, param_type))
            ax.legend(loc="best")
            plt.tight_layout()
            fig.savefig(
                figures_dir / f"comparison_l1d_miss_rate_vs_{param_type}.pdf",
            )
            fig.savefig(
                figures_dir / f"comparison_l1d_miss_rate_vs_{param_type}.png",
            )
            plt.close(fig)


def save_results_csv(
    all_results: Dict[str, List[SimulationStats]], out_dir: Path
) -> None:
    """Save all_results into a pandas DataFrame and write to CSV.

    Uses pandas nullable integer dtype `Int64` for integer columns and
    `float64` for floating columns. String columns use pandas `string` dtype.
    """
    records: List[Dict[str, Any]] = []

    for wl, stats_list in all_results.items():
        for s in stats_list:
            rec: Dict[str, Any] = {
                "workload": wl,
                "config_type": s.config_type,
                "config_value": s.config_value,
                # CPU stats
                "sim_insts": int(s.sim_insts),
                "sim_cycles": int(s.sim_cycles),
                "ipc": float(s.ipc),
                "cpi": float(s.cpi),
                # L1D
                "l1d_hits": int(s.l1d_cache.hits),
                "l1d_misses": int(s.l1d_cache.misses),
                "l1d_miss_rate": float(s.l1d_cache.miss_rate),
                "l1d_miss_latency": float(s.l1d_cache.miss_latency),
                # L1I
                "l1i_hits": int(s.l1i_cache.hits),
                "l1i_misses": int(s.l1i_cache.misses),
                "l1i_miss_rate": float(s.l1i_cache.miss_rate),
                "l1i_miss_latency": float(s.l1i_cache.miss_latency),
                # L2
                "l2_hits": int(s.l2_cache.hits),
                "l2_misses": int(s.l2_cache.misses),
                "l2_miss_rate": float(s.l2_cache.miss_rate),
                "l2_miss_latency": float(s.l2_cache.miss_latency),
                # L3
                "l3_hits": int(s.l3_cache.hits),
                "l3_misses": int(s.l3_cache.misses),
                "l3_miss_rate": float(s.l3_cache.miss_rate),
                "l3_miss_latency": float(s.l3_cache.miss_latency),
                # Computed
                "l1d_mpki": float(s.l1d_mpki),
                "l2_mpki": float(s.l2_mpki),
                "l3_mpki": float(s.l3_mpki),
                "total_miss_latency": float(s.total_miss_latency),
            }

            rec["config_display_name"] = CONFIG_DISPLAY_NAMES.get(s.config_value, "")
            rec["total_cache_bytes"] = (
                get_total_cache_size(s.config_value)
                if s.config_type == "cache_config"
                else None
            )

            records.append(rec)

    if not records:
        print("No records to save; skipping CSV export")
        return

    df = pd.DataFrame.from_records(records)

    # Columns by intended dtype
    int_cols = [
        "sim_insts",
        "sim_cycles",
        "l1d_hits",
        "l1d_misses",
        "l1i_hits",
        "l1i_misses",
        "l2_hits",
        "l2_misses",
        "l3_hits",
        "l3_misses",
        "total_cache_bytes",
    ]
    float_cols = [
        "ipc",
        "cpi",
        "l1d_miss_rate",
        "l1d_miss_latency",
        "l1i_miss_rate",
        "l1i_miss_latency",
        "l2_miss_rate",
        "l2_miss_latency",
        "l3_miss_rate",
        "l3_miss_latency",
        "l1d_mpki",
        "l2_mpki",
        "l3_mpki",
        "total_miss_latency",
    ]
    str_cols = ["workload", "config_type", "config_value", "config_display_name"]

    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")

    out_path = out_dir / "simulation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved combined simulation results to: {out_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main():
    """Main entry point."""
    # Determine paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    figures_dir = script_dir / "figures"

    # Create figures directory
    figures_dir.mkdir(exist_ok=True)

    print("gem5 Cache Simulation Results Visualization")
    print("=" * 50)
    print(f"Results directory: {results_dir}")
    print(f"Figures directory: {figures_dir}")
    print()

    # Collect all results
    print("Collecting simulation results...")
    all_results = collect_all_results(results_dir)

    if not all_results:
        print("ERROR: No results found in results directory!")
        sys.exit(1)

    print(f"Found results for {len(all_results)} workloads:")
    for workload, stats in all_results.items():
        print(f"  - {workload}: {len(stats)} configurations")
    print()

    # Generate figures for each workload
    for workload in POLYBENCH_WORKLOADS:
        if workload not in all_results:
            print(f"WARNING: No results found for {workload}")
            continue

        print(f"Generating figures for {workload}...")
        create_workload_figures(workload, all_results[workload], figures_dir)

    # Generate comparison figures
    print("Generating comparison figures...")
    create_comparison_figures(all_results, figures_dir)

    # Save all numerical results to CSV using pandas
    try:
        save_results_csv(all_results, figures_dir)
    except Exception as e:
        print("WARNING: failed to save simulation CSV:", e)

    print()
    print("=" * 50)
    print("Done! Figures saved to:", figures_dir)
    print()
    print("Generated figure types for each workload:")
    print("  - miss_rate_vs_<param>.pdf/png  : L1D/L2/L3 miss rates (stacked)")
    print("  - mpki_vs_<param>.pdf/png       : L1D/L2/L3 MPKI (stacked)")
    print("  - ipc_cpi_vs_<param>.pdf/png    : IPC and CPI")
    print("  - memory_stalls_vs_<param>.pdf/png : Total miss latency")
    print("  - speedup_vs_<param>.pdf/png    : Relative speedup vs baseline")
    print("  - all_miss_rates_vs_<param>.pdf/png : All miss rates on log scale")
    print()
    print("Cross-workload comparison figures:")
    print("  - comparison_ipc_vs_<param>.pdf/png")
    print("  - comparison_l1d_miss_rate_vs_<param>.pdf/png")


if __name__ == "__main__":
    main()
