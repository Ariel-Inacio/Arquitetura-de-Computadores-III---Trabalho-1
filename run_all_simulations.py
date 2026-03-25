#!/usr/bin/env python3
"""
Wrapper script to run gem5 simulations with all parameter variations for a given workload.
Usage: ./run_all_simulations.py <gem5_executable> <workload_binary> [workload_args...]
"""

import sys
import subprocess
import argparse
from pathlib import Path
import time
from datetime import datetime
from typing import List, Optional, Tuple

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BG_GREEN = '\033[42m'
    BG_BLUE = '\033[44m'
    BG_YELLOW = '\033[43m'
    BG_CYAN = '\033[46m'
    BLACK = '\033[30m'

def print_simulation_header(current: int, total: int, workload_name: str, config_desc: str) -> None:
    """Print a colored header for the current simulation."""
    percentage = (current / total) * 100
    header = f" SIMULATION {current}/{total} ({percentage:.1f}%) - {workload_name} "
    config_line = f" Configuration: {config_desc} "

    # Choose color based on progress
    if percentage < 33:
        bg_color = Colors.BG_YELLOW
    elif percentage < 66:
        bg_color = Colors.BG_CYAN
    else:
        bg_color = Colors.BG_GREEN

    # Create a box around the header
    max_len = max(len(header), len(config_line))
    border = "=" * (max_len + 2)

    print(f"\n{Colors.BOLD}{bg_color}{Colors.BLACK}")
    print(f" {border} ")
    print(f" {header.center(max_len)} ")
    print(f" {config_line.center(max_len)} ")
    print(f" {border} ")
    print(f"{Colors.ENDC}\n")

def check_simulation_completed(output_dir: Path) -> bool:
    """Check if a simulation has completed successfully."""
    completed_file = output_dir / '.completed'
    stats_file = output_dir / 'stats.txt'

    # If .completed marker exists, trust it
    if completed_file.exists():
        return True

    # Otherwise, check if stats.txt exists and has content
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                content = f.read()
                # Check for key indicators that simulation completed
                if 'simTicks' in content and 'system.cpu.numCycles' in content:
                    # Mark as completed for future runs
                    completed_file.touch()
                    return True
        except:
            pass

    return False

def run_simulation(gem5_exe: str, cache_config_script: str, workload: str, workload_args: List[str], output_dir: Path, param_name: Optional[str] = None, param_value: Optional[str] = None) -> bool:
    """Run a single gem5 simulation with given parameters."""
    # Build the command
    cmd = [gem5_exe, '-d', str(output_dir), '--', cache_config_script]

    # Add cache configuration parameter if specified
    if param_name and param_value:
        cmd.append(f'--{param_name}={param_value}')

    # Add workload and its arguments
    cmd.append(workload)
    cmd.extend(workload_args)

    # Run the simulation
    print(f"{Colors.OKCYAN}Command: {' '.join(cmd)}{Colors.ENDC}")

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minute timeout
        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"{Colors.OKGREEN}✓ Simulation completed in {elapsed_time:.1f} seconds{Colors.ENDC}")
            # Mark as completed
            (output_dir / '.completed').touch()
            return True
        else:
            print(f"{Colors.FAIL}✗ Simulation failed with return code {result.returncode}{Colors.ENDC}")
            if result.stderr:
                print(f"Error output: {result.stderr[:500]}...")  # Print first 500 chars of error
            return False

    except subprocess.TimeoutExpired:
        print(f"{Colors.FAIL}✗ Simulation timed out after 600 seconds{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.FAIL}✗ Error running simulation: {e}{Colors.ENDC}")
        return False

def get_all_parameter_variations() -> List[Tuple[str, str, str]]:
    """Get all parameter variations to test."""
    variations: List[Tuple[str, str, str]] = []

    # Cache size configurations
    cache_configs = [
        'baseline', 'intel_core_i7_6700k', 'intel_core_i9_9900k',
        'amd_ryzen_5600x', 'amd_ryzen_7700x', 'intel_xeon_gold',
        'apple_m1', 'apple_m2', 'intel_atom', 'arm_cortex_a78',
        'ibm_power10', 'small_embedded', 'large_server'
    ]

    for config in cache_configs:
        variations.append(('cache-config', config, f'cache_config_{config}'))

    # Cache line sizes
    cache_line_sizes = [32, 64, 128, 256]
    for size in cache_line_sizes:
        variations.append(('cache-line-size', str(size), f'cache_line_{size}'))

    # Associativities - only powers of 2
    associativities = [1, 2, 4, 8, 16]

    # L1 associativity
    for assoc in associativities:
        variations.append(('l1-assoc', str(assoc), f'l1_assoc_{assoc}'))

    # L2 associativity
    for assoc in associativities:
        variations.append(('l2-assoc', str(assoc), f'l2_assoc_{assoc}'))

    # L3 associativity
    for assoc in associativities:
        variations.append(('l3-assoc', str(assoc), f'l3_assoc_{assoc}'))

    # Memory latencies - removed as per assignment requirements
    # mem_latencies = ['fast', 'baseline', 'slow', 'very_slow']
    # for latency in mem_latencies:
    #     variations.append(('mem-latency', latency, f'mem_latency_{latency}'))

    return variations

def main():
    parser = argparse.ArgumentParser(description='Run all gem5 cache parameter variations for a workload')
    parser.add_argument('gem5_exe', help='Path to gem5 executable (e.g., ./gem5/build/ALL/gem5.opt)')
    parser.add_argument('workload', help='Path to workload binary')
    parser.add_argument('workload_args', nargs='*', default=[], help='Arguments to pass to the workload')
    parser.add_argument('--results-dir', default='results', help='Base directory for results (default: results)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run without executing')

    args = parser.parse_args()

    # Validate inputs
    gem5_exe = Path(args.gem5_exe)
    if not gem5_exe.exists():
        print(f"{Colors.FAIL}Error: gem5 executable not found: {gem5_exe}{Colors.ENDC}")
        sys.exit(1)

    workload = Path(args.workload)
    if not workload.exists():
        print(f"{Colors.FAIL}Error: Workload binary not found: {workload}{Colors.ENDC}")
        sys.exit(1)

    # Get workload name for display and directory naming
    workload_name = workload.stem

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    # Get all parameter variations
    variations = get_all_parameter_variations()

    # Add baseline (no parameters) as the first simulation
    all_simulations: List[Tuple[str, Optional[str], Optional[str], str]] = [('baseline', None, None, 'baseline')]

    # Add all variations
    for param_name, param_value, dir_suffix in variations:
        all_simulations.append((f'{param_name}={param_value}', param_name, param_value, dir_suffix))

    total_simulations = len(all_simulations)
    completed = 0
    skipped = 0
    failed = 0

    print(f"\n{Colors.BOLD}{Colors.HEADER}=== GEM5 SIMULATION RUNNER ==={Colors.ENDC}")
    print(f"Workload: {Colors.BOLD}{workload_name}{Colors.ENDC} ({workload})")
    print(f"Workload args: {' '.join(args.workload_args) if args.workload_args else '(none)'}")
    print(f"Total simulations to run: {Colors.BOLD}{total_simulations}{Colors.ENDC}")
    print(f"Results directory: {results_dir.absolute()}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if args.dry_run:
        print(f"{Colors.WARNING}DRY RUN MODE - No simulations will be executed{Colors.ENDC}\n")

    # Run each simulation
    for i, (desc, param_name, param_value, dir_suffix) in enumerate(all_simulations, 1):
        # Create output directory for this simulation
        output_dir = results_dir / f"{workload_name}_{dir_suffix}"

        # Check if already completed
        if check_simulation_completed(output_dir):
            print(f"\n{Colors.OKGREEN}Skipping simulation {i}/{total_simulations}: {desc} (already completed){Colors.ENDC}")
            skipped += 1
            continue

        # Print header for this simulation
        print_simulation_header(i, total_simulations, workload_name, desc)

        if args.dry_run:
            print(f"Would create directory: {output_dir}")
            if param_name:
                print(f"Would run with parameter: --{param_name}={param_value}")
            continue

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run the simulation
        success = run_simulation(
            str(gem5_exe),
            'cache_config.py',
            str(workload),
            args.workload_args,
            output_dir,
            param_name,
            param_value
        )

        if success:
            completed += 1
        else:
            failed += 1

    # Print summary
    print(f"\n{Colors.BOLD}{Colors.HEADER}=== SIMULATION SUMMARY ==={Colors.ENDC}")
    print(f"Total simulations: {total_simulations}")
    print(f"{Colors.OKGREEN}Completed: {completed}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Skipped (already done): {skipped}{Colors.ENDC}")
    if failed > 0:
        print(f"{Colors.FAIL}Failed: {failed}{Colors.ENDC}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if completed + skipped == total_simulations:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓ All simulations completed successfully!{Colors.ENDC}")
    elif failed > 0:
        print(f"\n{Colors.WARNING}{Colors.BOLD}⚠ Some simulations failed. Check the output above for details.{Colors.ENDC}")
        sys.exit(1)

if __name__ == '__main__':
    main()
