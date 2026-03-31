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
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Sequence, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# STYLE CONFIGURATION - Easy to edit for publication
# =============================================================================

# Figure sizes (in inches) - designed to fit US Letter page
FIGURE_SIZE_SINGLE = (6.5, 4.5)      # Single plot
FIGURE_SIZE_WIDE = (6.5, 3.5)        # Wide single plot
FIGURE_SIZE_MULTI = (6.5, 8.0)       # Multiple subplots stacked

# Font sizes
FONT_SIZE_TITLE = 12
FONT_SIZE_AXIS_LABEL = 10
FONT_SIZE_TICK = 9
FONT_SIZE_LEGEND = 9

# Line styles
LINE_WIDTH = 1.5
MARKER_SIZE = 6
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h']

# Color palette (colorblind-friendly)
COLORS = [
    '#0077BB',  # Blue
    '#EE7733',  # Orange
    '#009988',  # Teal
    '#CC3311',  # Red
    '#33BBEE',  # Cyan
    '#EE3377',  # Magenta
    '#BBBBBB',  # Grey
    '#000000',  # Black
]

# Grid style
GRID_ALPHA = 0.3
GRID_STYLE = '--'

# DPI for saved figures
SAVE_DPI = 300

# Apply global matplotlib settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': FONT_SIZE_TICK,
    'axes.titlesize': FONT_SIZE_TITLE,
    'axes.labelsize': FONT_SIZE_AXIS_LABEL,
    'xtick.labelsize': FONT_SIZE_TICK,
    'ytick.labelsize': FONT_SIZE_TICK,
    'legend.fontsize': FONT_SIZE_LEGEND,
    'figure.dpi': 100,
    'savefig.dpi': SAVE_DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': GRID_ALPHA,
    'grid.linestyle': GRID_STYLE,
})


# =============================================================================
# CONFIGURATION MAPPINGS
# =============================================================================

# Cache size configurations from cache_config.py (total cache size for sorting)
CACHE_SIZE_CONFIGS = {
    'baseline': ('32KiB', '32KiB', '256KiB', '8MiB'),
    'intel_core_i7_6700k': ('32KiB', '32KiB', '256KiB', '8MiB'),
    'intel_core_i9_9900k': ('32KiB', '32KiB', '256KiB', '16MiB'),
    'amd_ryzen_5600x': ('32KiB', '32KiB', '512KiB', '32MiB'),
    'amd_ryzen_7700x': ('32KiB', '32KiB', '1MiB', '32MiB'),
    'intel_xeon_gold': ('32KiB', '32KiB', '1MiB', '32MiB'),
    'apple_m1': ('64KiB', '64KiB', '4MiB', '8MiB'),
    'apple_m2': ('64KiB', '64KiB', '4MiB', '16MiB'),
    'intel_atom': ('32KiB', '32KiB', '1MiB', '4MiB'),
    'arm_cortex_a78': ('32KiB', '32KiB', '512KiB', '4MiB'),
    'ibm_power10': ('32KiB', '32KiB', '2MiB', '8MiB'),
    'small_embedded': ('16KiB', '16KiB', '128KiB', '1MiB'),
    'large_server': ('64KiB', '64KiB', '2MiB', '64MiB'),
}

# Display names for configurations
CONFIG_DISPLAY_NAMES = {
    'baseline': 'Baseline',
    'intel_core_i7_6700k': 'i7-6700K',
    'intel_core_i9_9900k': 'i9-9900K',
    'amd_ryzen_5600x': 'Ryzen 5600X',
    'amd_ryzen_7700x': 'Ryzen 7700X',
    'intel_xeon_gold': 'Xeon Gold',
    'apple_m1': 'Apple M1',
    'apple_m2': 'Apple M2',
    'intel_atom': 'Intel Atom',
    'arm_cortex_a78': 'Cortex A78',
    'ibm_power10': 'POWER10',
    'small_embedded': 'Small Embedded',
    'large_server': 'Large Server',
}

# Polybench workloads (those with corresponding .h files)
POLYBENCH_WORKLOADS = ['atax', 'floyd-warshall', 'gemm', 'jacobi-2d', 'seidel-2d']

# Parameter types and their values
CACHE_LINE_SIZES = [32, 64, 128, 256]
ASSOCIATIVITIES = [1, 2, 4, 8, 16]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

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
    config_type: str = ""  # 'cache_config', 'cache_line', 'l1_assoc', 'l2_assoc', 'l3_assoc', 'baseline'
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


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_size_to_bytes(size_str: str) -> int:
    """Convert size string (e.g., '32KiB', '8MiB') to bytes."""
    size_str = size_str.strip()
    match = re.match(r'(\d+)\s*(KiB|MiB|GiB|KB|MB|GB)', size_str, re.IGNORECASE)
    if not match:
        return 0
    value = int(match.group(1))
    unit = match.group(2).upper()
    multipliers = {
        'KIB': 1024, 'KB': 1024,
        'MIB': 1024**2, 'MB': 1024**2,
        'GIB': 1024**3, 'GB': 1024**3,
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
        if dir_name.startswith(workload + '_'):
            stats.workload = workload
            rest = dir_name[len(workload) + 1:]

            if rest == 'baseline':
                stats.config_type = 'baseline'
                stats.config_value = 'baseline'
            elif rest.startswith('cache_config_'):
                stats.config_type = 'cache_config'
                stats.config_value = rest[len('cache_config_'):]
            elif rest.startswith('cache_line_'):
                stats.config_type = 'cache_line'
                stats.config_value = rest[len('cache_line_'):]
            elif rest.startswith('l1_assoc_'):
                stats.config_type = 'l1_assoc'
                stats.config_value = rest[len('l1_assoc_'):]
            elif rest.startswith('l2_assoc_'):
                stats.config_type = 'l2_assoc'
                stats.config_value = rest[len('l2_assoc_'):]
            elif rest.startswith('l3_assoc_'):
                stats.config_type = 'l3_assoc'
                stats.config_value = rest[len('l3_assoc_'):]
            break

    if not stats.workload:
        return None

    # Read and parse stats file
    with open(stats_path, 'r') as f:
        content = f.read()

    # Helper function to extract numeric value
    def extract_value(pattern: str, default: float = 0.0) -> float:
        match = re.search(pattern + r'\s+([\d.]+(?:e[+-]?\d+)?)', content, re.MULTILINE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return default
        return default

    # CPU statistics (use first occurrence - before "Begin Simulation Statistics" repeat)
    stats.sim_insts = int(extract_value(r'^simInsts'))
    stats.sim_cycles = int(extract_value(r'^system\.cpu\.numCycles'))
    stats.ipc = extract_value(r'^system\.cpu\.ipc')
    stats.cpi = extract_value(r'^system\.cpu\.cpi')

    # L1 Data Cache
    stats.l1d_cache.hits = int(extract_value(r'^system\.cpu\.dcache\.demandHits::total'))
    stats.l1d_cache.misses = int(extract_value(r'^system\.cpu\.dcache\.demandMisses::total'))
    stats.l1d_cache.miss_rate = extract_value(r'^system\.cpu\.dcache\.demandMissRate::total')
    stats.l1d_cache.miss_latency = extract_value(r'^system\.cpu\.dcache\.overallMissLatency::total')

    # L1 Instruction Cache
    stats.l1i_cache.hits = int(extract_value(r'^system\.cpu\.icache\.demandHits::total'))
    stats.l1i_cache.misses = int(extract_value(r'^system\.cpu\.icache\.demandMisses::total'))
    stats.l1i_cache.miss_rate = extract_value(r'^system\.cpu\.icache\.demandMissRate::total')
    stats.l1i_cache.miss_latency = extract_value(r'^system\.cpu\.icache\.overallMissLatency::total')

    # L2 Cache (note: L2 may not have demandHits in gem5 depending on configuration)
    stats.l2_cache.hits = int(extract_value(r'^system\.l2cache\.demandHits::total'))
    stats.l2_cache.misses = int(extract_value(r'^system\.l2cache\.demandMisses::total'))
    stats.l2_cache.miss_rate = extract_value(r'^system\.l2cache\.demandMissRate::total')
    stats.l2_cache.miss_latency = extract_value(r'^system\.l2cache\.overallMissLatency::total')

    # L3 Cache
    stats.l3_cache.hits = int(extract_value(r'^system\.l3cache\.demandHits::total'))
    stats.l3_cache.misses = int(extract_value(r'^system\.l3cache\.demandMisses::total'))
    stats.l3_cache.miss_rate = extract_value(r'^system\.l3cache\.demandMissRate::total')
    stats.l3_cache.miss_latency = extract_value(r'^system\.l3cache\.overallMissLatency::total')

    # Compute MPKI (Misses Per Kilo Instructions)
    if stats.sim_insts > 0:
        stats.l1d_mpki = (stats.l1d_cache.misses / stats.sim_insts) * 1000
        stats.l2_mpki = (stats.l2_cache.misses / stats.sim_insts) * 1000
        stats.l3_mpki = (stats.l3_cache.misses / stats.sim_insts) * 1000

    # Total miss latency (proxy for memory stalls)
    stats.total_miss_latency = (
        stats.l1d_cache.miss_latency +
        stats.l1i_cache.miss_latency +
        stats.l2_cache.miss_latency +
        stats.l3_cache.miss_latency
    )

    return stats


def collect_all_results(results_dir: Path) -> Dict[str, List[SimulationStats]]:
    """Collect all simulation results, organized by workload."""
    results = defaultdict(list)

    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue

        stats_file = subdir / 'stats.txt'
        stats = parse_stats_file(stats_file)

        if stats and stats.workload in POLYBENCH_WORKLOADS:
            results[stats.workload].append(stats)

    return results


# =============================================================================
# PLOTTING HELPER FUNCTIONS
# =============================================================================

def get_sorted_data(stats_list: List[SimulationStats], config_type: str) -> Tuple[Sequence[Union[str, int]], List[SimulationStats]]:
    """Filter and sort data by configuration type."""
    filtered = [s for s in stats_list if s.config_type == config_type]

    if config_type == 'cache_config':
        # Sort by total cache size
        filtered.sort(key=lambda s: get_total_cache_size(s.config_value))
        x_values = [CONFIG_DISPLAY_NAMES.get(s.config_value, s.config_value) for s in filtered]
    elif config_type == 'cache_line':
        # Sort by cache line size (numeric)
        filtered.sort(key=lambda s: int(s.config_value))
        x_values = [int(s.config_value) for s in filtered]
    elif config_type in ['l1_assoc', 'l2_assoc', 'l3_assoc']:
        # Sort by associativity (numeric)
        filtered.sort(key=lambda s: int(s.config_value))
        x_values = [int(s.config_value) for s in filtered]
    else:
        x_values = [s.config_value for s in filtered]

    return x_values, filtered


def plot_data(
    ax: Any,
    x_vals: Sequence[Union[str, int]],
    y_vals: Sequence[float],
    color: str,
    marker: str,
    label: str,
    is_categorical: bool = False,
    bar_width: float = 0.8,
    bar_offset: float = 0.0,
) -> None:
    """Plot data - bar chart for categorical, line plot for numeric."""
    if is_categorical:
        # For categorical data, use bar chart
        x_numeric = np.arange(len(x_vals)) + bar_offset
        ax.bar(x_numeric, y_vals, width=bar_width, color=color, label=label, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_xticklabels(x_vals, rotation=45, ha='right')
    else:
        # For numeric data, use line plot with markers
        ax.plot(x_vals, y_vals, color=color, marker=marker,
                linestyle='-', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=label)


# =============================================================================
# MAIN PLOTTING FUNCTIONS
# =============================================================================

def create_workload_figures(
    workload: str,
    stats_list: List[SimulationStats],
    figures_dir: Path,
) -> None:
    """Create all figures for a single workload."""

    # Get baseline stats for speedup calculation
    baseline_stats = [s for s in stats_list if s.config_type == 'baseline']
    if not baseline_stats:
        # Use first cache_config_baseline if no dedicated baseline
        baseline_stats = [s for s in stats_list if s.config_value == 'baseline']

    baseline_ipc = baseline_stats[0].ipc if baseline_stats else 1.0

    # Parameter types to iterate over
    param_types = ['cache_config', 'cache_line', 'l1_assoc', 'l2_assoc', 'l3_assoc']

    # =========================================================================
    # Figure 1: Miss Rates (3 subplots stacked - L1D, L2, L3) for each param
    # =========================================================================
    for param_type in param_types:
        x_values, filtered = get_sorted_data(stats_list, param_type)
        if not filtered:
            continue

        fig, axes = plt.subplots(3, 1, figsize=FIGURE_SIZE_MULTI, sharex=(param_type != 'cache_config'))
        fig.suptitle(f'{workload.upper()}: Cache Miss Rates vs {param_type.replace("_", " ").title()}')

        miss_rate_metrics: List[Tuple[str, Callable[[SimulationStats], float], str]] = [
            ('L1D', lambda s: s.l1d_cache.miss_rate, COLORS[0]),
            ('L2', lambda s: s.l2_cache.miss_rate, COLORS[1]),
            ('L3', lambda s: s.l3_cache.miss_rate, COLORS[2]),
        ]

        is_categorical = param_type == 'cache_config'

        for idx, (name, extractor, color) in enumerate(miss_rate_metrics):
            y_values = [extractor(s) for s in filtered]

            if is_categorical:
                x_numeric = np.arange(len(x_values))
                axes[idx].bar(x_numeric, y_values, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                axes[idx].set_xticks(x_numeric)
                axes[idx].set_xticklabels(x_values, rotation=45, ha='right')
            else:
                axes[idx].plot(x_values, y_values, color=color, marker=MARKERS[idx],
                              linestyle='-', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

            axes[idx].set_ylabel(f'{name} Miss Rate')
            axes[idx].grid(True, alpha=GRID_ALPHA, linestyle=GRID_STYLE, axis='y' if is_categorical else 'both')

        # Set x-label on bottom subplot
        xlabel_map = {
            'cache_config': 'Cache Configuration',
            'cache_line': 'Cache Line Size (bytes)',
            'l1_assoc': 'L1 Associativity (ways)',
            'l2_assoc': 'L2 Associativity (ways)',
            'l3_assoc': 'L3 Associativity (ways)',
        }
        axes[2].set_xlabel(xlabel_map.get(param_type, param_type))

        plt.tight_layout()
        fig.savefig(figures_dir / f'{workload}_miss_rate_vs_{param_type}.pdf')
        fig.savefig(figures_dir / f'{workload}_miss_rate_vs_{param_type}.png')
        plt.close(fig)

    # =========================================================================
    # Figure 2: MPKI (3 subplots stacked - L1D, L2, L3) for each param
    # =========================================================================
    for param_type in param_types:
        x_values, filtered = get_sorted_data(stats_list, param_type)
        if not filtered:
            continue

        fig, axes = plt.subplots(3, 1, figsize=FIGURE_SIZE_MULTI, sharex=(param_type != 'cache_config'))
        fig.suptitle(f'{workload.upper()}: MPKI vs {param_type.replace("_", " ").title()}')

        mpki_metrics: List[Tuple[str, Callable[[SimulationStats], float], str]] = [
            ('L1D', lambda s: s.l1d_mpki, COLORS[0]),
            ('L2', lambda s: s.l2_mpki, COLORS[1]),
            ('L3', lambda s: s.l3_mpki, COLORS[2]),
        ]

        is_categorical = param_type == 'cache_config'

        for idx, (name, extractor, color) in enumerate(mpki_metrics):
            y_values = [extractor(s) for s in filtered]

            if is_categorical:
                x_numeric = np.arange(len(x_values))
                axes[idx].bar(x_numeric, y_values, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                axes[idx].set_xticks(x_numeric)
                axes[idx].set_xticklabels(x_values, rotation=45, ha='right')
            else:
                axes[idx].plot(x_values, y_values, color=color, marker=MARKERS[idx],
                              linestyle='-', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

            axes[idx].set_ylabel(f'{name} MPKI')
            axes[idx].grid(True, alpha=GRID_ALPHA, linestyle=GRID_STYLE, axis='y' if is_categorical else 'both')

        xlabel_map = {
            'cache_config': 'Cache Configuration',
            'cache_line': 'Cache Line Size (bytes)',
            'l1_assoc': 'L1 Associativity (ways)',
            'l2_assoc': 'L2 Associativity (ways)',
            'l3_assoc': 'L3 Associativity (ways)',
        }
        axes[2].set_xlabel(xlabel_map.get(param_type, param_type))

        plt.tight_layout()
        fig.savefig(figures_dir / f'{workload}_mpki_vs_{param_type}.pdf')
        fig.savefig(figures_dir / f'{workload}_mpki_vs_{param_type}.png')
        plt.close(fig)

    # =========================================================================
    # Figure 3: IPC/CPI (2 subplots) for each param
    # =========================================================================
    for param_type in param_types:
        x_values, filtered = get_sorted_data(stats_list, param_type)
        if not filtered:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(FIGURE_SIZE_WIDE[0], 6), sharex=(param_type != 'cache_config'))
        fig.suptitle(f'{workload.upper()}: Performance vs {param_type.replace("_", " ").title()}')

        perf_metrics: List[Tuple[str, Callable[[SimulationStats], float], str]] = [
            ('IPC', lambda s: s.ipc, COLORS[0]),
            ('CPI', lambda s: s.cpi, COLORS[1]),
        ]

        is_categorical = param_type == 'cache_config'

        for idx, (name, extractor, color) in enumerate(perf_metrics):
            y_values = [extractor(s) for s in filtered]

            if is_categorical:
                x_numeric = np.arange(len(x_values))
                axes[idx].bar(x_numeric, y_values, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                axes[idx].set_xticks(x_numeric)
                axes[idx].set_xticklabels(x_values, rotation=45, ha='right')
            else:
                axes[idx].plot(x_values, y_values, color=color, marker=MARKERS[idx],
                              linestyle='-', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

            axes[idx].set_ylabel(name)
            axes[idx].grid(True, alpha=GRID_ALPHA, linestyle=GRID_STYLE, axis='y' if is_categorical else 'both')

        xlabel_map = {
            'cache_config': 'Cache Configuration',
            'cache_line': 'Cache Line Size (bytes)',
            'l1_assoc': 'L1 Associativity (ways)',
            'l2_assoc': 'L2 Associativity (ways)',
            'l3_assoc': 'L3 Associativity (ways)',
        }
        axes[1].set_xlabel(xlabel_map.get(param_type, param_type))

        plt.tight_layout()
        fig.savefig(figures_dir / f'{workload}_ipc_cpi_vs_{param_type}.pdf')
        fig.savefig(figures_dir / f'{workload}_ipc_cpi_vs_{param_type}.png')
        plt.close(fig)

    # =========================================================================
    # Figure 4: Memory Stalls for each param
    # =========================================================================
    for param_type in param_types:
        x_values, filtered = get_sorted_data(stats_list, param_type)
        if not filtered:
            continue

        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)
        fig.suptitle(f'{workload.upper()}: Memory Stalls vs {param_type.replace("_", " ").title()}')

        y_values = [s.total_miss_latency / 1e9 for s in filtered]  # Convert to billions
        is_categorical = param_type == 'cache_config'

        if is_categorical:
            x_numeric = np.arange(len(x_values))
            ax.bar(x_numeric, y_values, color=COLORS[3], alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x_numeric)
            ax.set_xticklabels(x_values, rotation=45, ha='right')
        else:
            ax.plot(x_values, y_values, color=COLORS[3], marker=MARKERS[0],
                   linestyle='-', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

        ax.set_ylabel('Total Miss Latency (Billion Ticks)')
        xlabel_map = {
            'cache_config': 'Cache Configuration',
            'cache_line': 'Cache Line Size (bytes)',
            'l1_assoc': 'L1 Associativity (ways)',
            'l2_assoc': 'L2 Associativity (ways)',
            'l3_assoc': 'L3 Associativity (ways)',
        }
        ax.set_xlabel(xlabel_map.get(param_type, param_type))
        ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_STYLE, axis='y' if is_categorical else 'both')

        plt.tight_layout()
        fig.savefig(figures_dir / f'{workload}_memory_stalls_vs_{param_type}.pdf')
        fig.savefig(figures_dir / f'{workload}_memory_stalls_vs_{param_type}.png')
        plt.close(fig)

    # =========================================================================
    # Figure 5: Relative Speedup for each param
    # =========================================================================
    for param_type in param_types:
        x_values, filtered = get_sorted_data(stats_list, param_type)
        if not filtered:
            continue

        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)
        fig.suptitle(f'{workload.upper()}: Relative Speedup vs {param_type.replace("_", " ").title()}')

        y_values = [s.ipc / baseline_ipc if baseline_ipc > 0 else 1.0 for s in filtered]
        is_categorical = param_type == 'cache_config'

        if is_categorical:
            x_numeric = np.arange(len(x_values))
            ax.bar(x_numeric, y_values, color=COLORS[4], alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x_numeric)
            ax.set_xticklabels(x_values, rotation=45, ha='right')
        else:
            ax.plot(x_values, y_values, color=COLORS[4], marker=MARKERS[0],
                   linestyle='-', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

        # Add reference line at speedup = 1.0
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_ylabel('Relative Speedup (vs Baseline)')
        xlabel_map = {
            'cache_config': 'Cache Configuration',
            'cache_line': 'Cache Line Size (bytes)',
            'l1_assoc': 'L1 Associativity (ways)',
            'l2_assoc': 'L2 Associativity (ways)',
            'l3_assoc': 'L3 Associativity (ways)',
        }
        ax.set_xlabel(xlabel_map.get(param_type, param_type))
        ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_STYLE, axis='y' if is_categorical else 'both')

        plt.tight_layout()
        fig.savefig(figures_dir / f'{workload}_speedup_vs_{param_type}.pdf')
        fig.savefig(figures_dir / f'{workload}_speedup_vs_{param_type}.png')
        plt.close(fig)

    # =========================================================================
    # Combined Figure: All Miss Rates on same plot (for comparison)
    # =========================================================================
    for param_type in param_types:
        x_values, filtered = get_sorted_data(stats_list, param_type)
        if not filtered:
            continue

        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)
        fig.suptitle(f'{workload.upper()}: All Cache Miss Rates vs {param_type.replace("_", " ").title()}')

        combined_miss_rate_metrics: List[Tuple[str, Callable[[SimulationStats], float], str, str]] = [
            ('L1D', lambda s: s.l1d_cache.miss_rate, COLORS[0], MARKERS[0]),
            ('L2', lambda s: s.l2_cache.miss_rate, COLORS[1], MARKERS[1]),
            ('L3', lambda s: s.l3_cache.miss_rate, COLORS[2], MARKERS[2]),
        ]

        is_categorical = param_type == 'cache_config'

        if is_categorical:
            # Use grouped bar chart for categorical data
            n_groups = len(x_values)
            n_bars = len(combined_miss_rate_metrics)
            bar_width = 0.25
            x_numeric = np.arange(n_groups)

            for i, (name, extractor, color, marker) in enumerate(combined_miss_rate_metrics):
                y_values = [extractor(s) for s in filtered]
                offset = (i - n_bars / 2 + 0.5) * bar_width
                ax.bar(x_numeric + offset, y_values, width=bar_width, color=color,
                       label=name, alpha=0.8, edgecolor='black', linewidth=0.5)

            ax.set_xticks(x_numeric)
            ax.set_xticklabels(x_values, rotation=45, ha='right')
        else:
            # Use line plot for numeric data
            for name, extractor, color, marker in combined_miss_rate_metrics:
                y_values = [extractor(s) for s in filtered]
                ax.plot(x_values, y_values, color=color, marker=marker,
                       linestyle='-', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=name)

        ax.set_ylabel('Miss Rate')
        ax.set_yscale('log')  # Use log scale for better visibility across cache levels
        xlabel_map = {
            'cache_config': 'Cache Configuration',
            'cache_line': 'Cache Line Size (bytes)',
            'l1_assoc': 'L1 Associativity (ways)',
            'l2_assoc': 'L2 Associativity (ways)',
            'l3_assoc': 'L3 Associativity (ways)',
        }
        ax.set_xlabel(xlabel_map.get(param_type, param_type))
        ax.legend(loc='best')
        ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_STYLE, axis='y' if is_categorical else 'both')

        plt.tight_layout()
        fig.savefig(figures_dir / f'{workload}_all_miss_rates_vs_{param_type}.pdf')
        fig.savefig(figures_dir / f'{workload}_all_miss_rates_vs_{param_type}.png')
        plt.close(fig)


def create_comparison_figures(
    all_results: Dict[str, List[SimulationStats]],
    figures_dir: Path,
) -> None:
    """Create comparison figures across all workloads."""

    param_types = ['cache_config', 'cache_line', 'l1_assoc', 'l2_assoc', 'l3_assoc']

    xlabel_map = {
        'cache_config': 'Cache Configuration',
        'cache_line': 'Cache Line Size (bytes)',
        'l1_assoc': 'L1 Associativity (ways)',
        'l2_assoc': 'L2 Associativity (ways)',
        'l3_assoc': 'L3 Associativity (ways)',
    }

    for param_type in param_types:
        is_categorical = param_type == 'cache_config'

        # =====================================================================
        # IPC comparison across workloads
        # =====================================================================
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)
        fig.suptitle(f'IPC Comparison: All Workloads vs {param_type.replace("_", " ").title()}')

        if is_categorical:
            # Use grouped bar chart for categorical data
            # First, collect all data to determine x_values
            all_x_values: Optional[Sequence[Union[str, int]]] = None
            workload_data: Dict[str, List[float]] = {}
            for workload in POLYBENCH_WORKLOADS:
                if workload not in all_results:
                    continue
                x_values, filtered = get_sorted_data(all_results[workload], param_type)
                if filtered:
                    all_x_values = x_values  # All should be the same
                    workload_data[workload] = [s.ipc for s in filtered]

            if all_x_values and workload_data:
                n_groups = len(all_x_values)
                n_bars = len(workload_data)
                bar_width = 0.8 / n_bars
                x_numeric = np.arange(n_groups)

                for idx, (workload, y_values) in enumerate(workload_data.items()):
                    offset = (idx - n_bars / 2 + 0.5) * bar_width
                    ax.bar(x_numeric + offset, y_values, width=bar_width,
                           color=COLORS[idx % len(COLORS)], label=workload,
                           alpha=0.8, edgecolor='black', linewidth=0.5)

                ax.set_xticks(x_numeric)
                ax.set_xticklabels(all_x_values, rotation=45, ha='right')
        else:
            # Use line plot for numeric data
            for idx, workload in enumerate(POLYBENCH_WORKLOADS):
                if workload not in all_results:
                    continue
                x_values, filtered = get_sorted_data(all_results[workload], param_type)
                if not filtered:
                    continue
                y_values = [s.ipc for s in filtered]
                ax.plot(x_values, y_values, color=COLORS[idx % len(COLORS)],
                       marker=MARKERS[idx % len(MARKERS)], linestyle='-',
                       linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=workload)

        ax.set_ylabel('IPC')
        ax.set_xlabel(xlabel_map.get(param_type, param_type))
        ax.legend(loc='best')
        ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_STYLE, axis='y' if is_categorical else 'both')

        plt.tight_layout()
        fig.savefig(figures_dir / f'comparison_ipc_vs_{param_type}.pdf')
        fig.savefig(figures_dir / f'comparison_ipc_vs_{param_type}.png')
        plt.close(fig)

        # =====================================================================
        # L1D Miss Rate comparison across workloads
        # =====================================================================
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)
        fig.suptitle(f'L1D Miss Rate Comparison: All Workloads vs {param_type.replace("_", " ").title()}')

        if is_categorical:
            # Use grouped bar chart for categorical data
            all_x_values: Optional[Sequence[Union[str, int]]] = None
            workload_data: Dict[str, List[float]] = {}
            for workload in POLYBENCH_WORKLOADS:
                if workload not in all_results:
                    continue
                x_values, filtered = get_sorted_data(all_results[workload], param_type)
                if filtered:
                    all_x_values = x_values
                    workload_data[workload] = [s.l1d_cache.miss_rate for s in filtered]

            if all_x_values and workload_data:
                n_groups = len(all_x_values)
                n_bars = len(workload_data)
                bar_width = 0.8 / n_bars
                x_numeric = np.arange(n_groups)

                for idx, (workload, y_values) in enumerate(workload_data.items()):
                    offset = (idx - n_bars / 2 + 0.5) * bar_width
                    ax.bar(x_numeric + offset, y_values, width=bar_width,
                           color=COLORS[idx % len(COLORS)], label=workload,
                           alpha=0.8, edgecolor='black', linewidth=0.5)

                ax.set_xticks(x_numeric)
                ax.set_xticklabels(all_x_values, rotation=45, ha='right')
        else:
            # Use line plot for numeric data
            for idx, workload in enumerate(POLYBENCH_WORKLOADS):
                if workload not in all_results:
                    continue
                x_values, filtered = get_sorted_data(all_results[workload], param_type)
                if not filtered:
                    continue
                y_values = [s.l1d_cache.miss_rate for s in filtered]
                ax.plot(x_values, y_values, color=COLORS[idx % len(COLORS)],
                       marker=MARKERS[idx % len(MARKERS)], linestyle='-',
                       linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=workload)

        ax.set_ylabel('L1D Miss Rate')
        ax.set_xlabel(xlabel_map.get(param_type, param_type))
        ax.legend(loc='best')
        ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_STYLE, axis='y' if is_categorical else 'both')

        plt.tight_layout()
        fig.savefig(figures_dir / f'comparison_l1d_miss_rate_vs_{param_type}.pdf')
        fig.savefig(figures_dir / f'comparison_l1d_miss_rate_vs_{param_type}.png')
        plt.close(fig)


def save_results_csv(all_results: Dict[str, List[SimulationStats]], out_dir: Path) -> None:
    """Save all_results into a pandas DataFrame and write to CSV.

    Uses pandas nullable integer dtype `Int64` for integer columns and
    `float64` for floating columns. String columns use pandas `string` dtype.
    """
    records: List[Dict[str, Any]] = []

    for wl, stats_list in all_results.items():
        for s in stats_list:
            rec: Dict[str, Any] = {
                'workload': wl,
                'config_type': s.config_type,
                'config_value': s.config_value,

                # CPU stats
                'sim_insts': int(s.sim_insts),
                'sim_cycles': int(s.sim_cycles),
                'ipc': float(s.ipc),
                'cpi': float(s.cpi),

                # L1D
                'l1d_hits': int(s.l1d_cache.hits),
                'l1d_misses': int(s.l1d_cache.misses),
                'l1d_miss_rate': float(s.l1d_cache.miss_rate),
                'l1d_miss_latency': float(s.l1d_cache.miss_latency),

                # L1I
                'l1i_hits': int(s.l1i_cache.hits),
                'l1i_misses': int(s.l1i_cache.misses),
                'l1i_miss_rate': float(s.l1i_cache.miss_rate),
                'l1i_miss_latency': float(s.l1i_cache.miss_latency),

                # L2
                'l2_hits': int(s.l2_cache.hits),
                'l2_misses': int(s.l2_cache.misses),
                'l2_miss_rate': float(s.l2_cache.miss_rate),
                'l2_miss_latency': float(s.l2_cache.miss_latency),

                # L3
                'l3_hits': int(s.l3_cache.hits),
                'l3_misses': int(s.l3_cache.misses),
                'l3_miss_rate': float(s.l3_cache.miss_rate),
                'l3_miss_latency': float(s.l3_cache.miss_latency),

                # Computed
                'l1d_mpki': float(s.l1d_mpki),
                'l2_mpki': float(s.l2_mpki),
                'l3_mpki': float(s.l3_mpki),
                'total_miss_latency': float(s.total_miss_latency),
            }

            rec['config_display_name'] = CONFIG_DISPLAY_NAMES.get(s.config_value, '')
            rec['total_cache_bytes'] = get_total_cache_size(s.config_value) if s.config_type == 'cache_config' else None

            records.append(rec)

    if not records:
        print('No records to save; skipping CSV export')
        return

    df = pd.DataFrame.from_records(records)

    # Columns by intended dtype
    int_cols = [
        'sim_insts', 'sim_cycles',
        'l1d_hits', 'l1d_misses', 'l1i_hits', 'l1i_misses',
        'l2_hits', 'l2_misses', 'l3_hits', 'l3_misses',
        'total_cache_bytes',
    ]
    float_cols = [
        'ipc', 'cpi',
        'l1d_miss_rate', 'l1d_miss_latency', 'l1i_miss_rate', 'l1i_miss_latency',
        'l2_miss_rate', 'l2_miss_latency', 'l3_miss_rate', 'l3_miss_latency',
        'l1d_mpki', 'l2_mpki', 'l3_mpki', 'total_miss_latency',
    ]
    str_cols = ['workload', 'config_type', 'config_value', 'config_display_name']

    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')

    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')

    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype('string')

    out_path = out_dir / 'simulation_results.csv'
    df.to_csv(out_path, index=False)
    print(f'Saved combined simulation results to: {out_path}')


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point."""
    # Determine paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'
    figures_dir = script_dir / 'figures'

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
        print('WARNING: failed to save simulation CSV:', e)

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


if __name__ == '__main__':
    main()
