#!/usr/bin/env python3
"""
Wrapper script to run gem5 simulations with all parameter variations for a given workload.
Supports parallel execution using multiprocessing.
Usage: ./run_all_simulations_parallel.py <gem5_executable> <workload_binary> [workload_args...]
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil as _psutil
    psutil_available = True
except ImportError:
    _psutil = None  # type: ignore
    psutil_available = False
    print("Warning: psutil not installed. CPU pinning disabled. Install with: pip install psutil", file=sys.stderr)

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
    BG_MAGENTA = '\033[45m'
    BLACK = '\033[30m'

# Global lock for synchronized printing
# Global lock for synchronized printing
print_lock: Optional[Any] = None


def init_worker(lock: Any, worker_id: int, cpu_core: Optional[int]) -> None:
    """Initialize worker process with shared lock and CPU affinity."""
    global print_lock
    print_lock = lock
    # Ignore SIGINT in worker processes
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Set CPU affinity if specified and psutil is available
    if cpu_core is not None and psutil_available:
        try:
            p = _psutil.Process()  # type: ignore
            p.cpu_affinity([cpu_core])
            synchronized_print(f"{Colors.OKCYAN}[Worker {worker_id}] Pinned to CPU core {cpu_core}{Colors.ENDC}")
        except Exception as e:
            synchronized_print(f"{Colors.WARNING}[Worker {worker_id}] Could not set CPU affinity: {e}{Colors.ENDC}")

def synchronized_print(*args: Any, **kwargs: Any) -> None:
    """Thread-safe printing function."""
    global print_lock
    if print_lock:
        with print_lock:
            print(*args, **kwargs)
            sys.stdout.flush()
    else:
        print(*args, **kwargs)
        sys.stdout.flush()

def print_simulation_header(current: int, total: int, workload_name: str, config_desc: str, parallel_info: str = "") -> None:
    """Print a colored header for the current simulation."""
    percentage = (current / total) * 100
    header = f" SIMULATION {current}/{total} ({percentage:.1f}%) - {workload_name} "
    config_line = f" Configuration: {config_desc} "

    # Choose background color based on status in parallel_info
    if "FAILED" in parallel_info or "ERROR" in parallel_info:
        bg_color = Colors.BG_MAGENTA  # Red/magenta for failures
    elif "SKIPPED" in parallel_info:
        bg_color = Colors.BG_CYAN  # Cyan for skipped
    elif percentage < 33:
        bg_color = Colors.BG_YELLOW
    elif percentage < 66:
        bg_color = Colors.BG_CYAN
    else:
        bg_color = Colors.BG_GREEN

    # Create a box around the header
    lines = [header, config_line]
    if parallel_info:
        lines.append(f" {parallel_info} ")

    max_len = max(len(line) for line in lines)
    border = "=" * (max_len)

    output = ['']
    output.append(f"{Colors.BOLD}{bg_color}{Colors.BLACK} {border} {Colors.ENDC}")
    for line in lines:
        output.append(f"{Colors.BOLD}{bg_color}{Colors.BLACK} {line.center(max_len)} {Colors.ENDC}")
    output.append(f"{Colors.BOLD}{bg_color}{Colors.BLACK} {border} {Colors.ENDC}")
    output.append('')

    synchronized_print('\n'.join(output))

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
        except Exception:
            # If parsing fails, assume incomplete
            return False

    return False

def run_single_simulation(sim_data: Dict[str, Any]) -> Tuple[bool, str, float]:
    """
    Worker function to run a single simulation.
    Returns: (success, description, elapsed_time)
    """
    # Extract simulation parameters
    gem5_exe = sim_data['gem5_exe']
    cache_config_script = sim_data['cache_config_script']
    workload = sim_data['workload']
    workload_args = sim_data['workload_args']
    output_dir = Path(sim_data['output_dir'])
    param_name = sim_data.get('param_name')
    param_value = sim_data.get('param_value')
    desc = sim_data['desc']
    sim_index = sim_data['sim_index']
    total_sims = sim_data['total_sims']
    workload_name = sim_data['workload_name']
    worker_id = sim_data.get('worker_id', 0)

    # Check if already completed
    if check_simulation_completed(output_dir):
        # Print completion header for skipped simulation
        parallel_info = f"Worker {worker_id} - SKIPPED (already done)"
        print_simulation_header(sim_index, total_sims, workload_name, desc, parallel_info)
        return (True, desc, 0.0)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the command
    cmd = [gem5_exe, '-d', str(output_dir), '--', cache_config_script]

    # Add cache configuration parameter if specified
    if param_name and param_value:
        cmd.append(f'--{param_name}={param_value}')

    # Add workload and its arguments
    cmd.append(workload)
    cmd.extend(workload_args)

    # Print starting message (without the big colored header)
    synchronized_print(f"{Colors.OKCYAN}[Worker {worker_id}] Starting: {desc}{Colors.ENDC}")
    synchronized_print(f"{Colors.OKCYAN}[Worker {worker_id}] Command: {' '.join(cmd)}{Colors.ENDC}")

    try:
        start_time = time.time()
        # No timeout - simulations can take days
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            # Print success header AFTER completion
            # Format elapsed time nicely (handle hours/days)
            if elapsed_time < 60:
                time_str = f"{elapsed_time:.1f}s"
            elif elapsed_time < 3600:
                time_str = f"{elapsed_time/60:.1f}min"
            elif elapsed_time < 86400:
                time_str = f"{elapsed_time/3600:.1f}h"
            else:
                time_str = f"{elapsed_time/86400:.1f}days"

            parallel_info = f"Worker {worker_id} - COMPLETED in {time_str}"
            print_simulation_header(sim_index, total_sims, workload_name, desc, parallel_info)
            synchronized_print(f"{Colors.OKGREEN}[Worker {worker_id}] ✓ Success{Colors.ENDC}")
            # Mark as completed
            (output_dir / '.completed').touch()
            return (True, desc, elapsed_time)
        else:
            # Print failure header AFTER completion
            parallel_info = f"Worker {worker_id} - FAILED (code {result.returncode})"
            print_simulation_header(sim_index, total_sims, workload_name, desc, parallel_info)
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            synchronized_print(f"{Colors.FAIL}[Worker {worker_id}] ✗ Error: {error_msg}...{Colors.ENDC}")
            return (False, desc, elapsed_time)

    except Exception as e:
        # Print error header AFTER error
        parallel_info = f"Worker {worker_id} - ERROR"
        print_simulation_header(sim_index, total_sims, workload_name, desc, parallel_info)
        synchronized_print(f"{Colors.FAIL}[Worker {worker_id}] ✗ Error: {e}{Colors.ENDC}")
        return (False, desc, 0.0)

def worker_wrapper(task_queue: Any, result_queue: Any, worker_id: int, lock: Any, cpu_core: Optional[int]) -> None:
    """Worker process that pulls tasks from queue."""
    init_worker(lock, worker_id, cpu_core)

    while True:
        task = task_queue.get()
        if task is None:  # Sentinel value to stop
            break
        task['worker_id'] = worker_id
        result = run_single_simulation(task)
        result_queue.put(result)

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

    return variations

def main():
    parser = argparse.ArgumentParser(description='Run all gem5 cache parameter variations for a workload')
    parser.add_argument('gem5_exe', help='Path to gem5 executable (e.g., ./gem5/build/ALL/gem5.opt)')
    parser.add_argument('workload', help='Path to workload binary')
    parser.add_argument('workload_args', nargs='*', default=[], help='Arguments to pass to the workload')
    parser.add_argument('--results-dir', default='results', help='Base directory for results (default: results)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run without executing')
    parser.add_argument('--parallel', '-j', type=int, default=1,
                        help='Number of parallel simulations to run (default: 1, use -1 for all CPU cores)')
    parser.add_argument('--pin-workers', action='store_true',
                        help='Pin worker processes to specific CPU cores (requires psutil)')

    args = parser.parse_args()

    # Determine number of workers
    if args.parallel == -1:
        num_workers = os.cpu_count() or 1
    else:
        num_workers = max(1, args.parallel)

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

    # Build list of all simulations
    all_simulations: List[Tuple[str, Optional[str], Optional[str], str]] = [('baseline', None, None, 'baseline')]

    # Add all variations
    for param_name, param_value, dir_suffix in variations:
        all_simulations.append((f'{param_name}={param_value}', param_name, param_value, dir_suffix))

    total_simulations = len(all_simulations)

    print(f"\n{Colors.BOLD}{Colors.HEADER}=== GEM5 SIMULATION RUNNER ==={Colors.ENDC}")
    print(f"Workload: {Colors.BOLD}{workload_name}{Colors.ENDC} ({workload})")
    print(f"Workload args: {' '.join(args.workload_args) if args.workload_args else '(none)'}")
    print(f"Total simulations to run: {Colors.BOLD}{total_simulations}{Colors.ENDC}")
    print(f"Parallel workers: {Colors.BOLD}{num_workers}{Colors.ENDC}")
    print(f"Results directory: {results_dir.absolute()}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if args.dry_run:
        print(f"{Colors.WARNING}DRY RUN MODE - No simulations will be executed{Colors.ENDC}\n")
        for i, (desc, param_name, param_value, dir_suffix) in enumerate(all_simulations, 1):
            output_dir = results_dir / f"{workload_name}_{dir_suffix}"
            print(f"{i}/{total_simulations}: {desc}")
            print(f"  Directory: {output_dir}")
            if param_name:
                print(f"  Parameter: --{param_name}={param_value}")
        return

    # Prepare simulation data for workers
    simulation_tasks: List[Dict[str, Any]] = []
    for i, (desc, param_name, param_value, dir_suffix) in enumerate(all_simulations, 1):
        output_dir = results_dir / f"{workload_name}_{dir_suffix}"

        sim_data = {
            'gem5_exe': str(gem5_exe),
            'cache_config_script': 'cache_config.py',
            'workload': str(workload),
            'workload_args': args.workload_args,
            'output_dir': str(output_dir),
            'param_name': param_name,
            'param_value': param_value,
            'desc': desc,
            'sim_index': i,
            'total_sims': total_simulations,
            'workload_name': workload_name,
        }
        simulation_tasks.append(sim_data)

    # Run simulations
    if num_workers == 1:
        # Sequential execution
        print(f"{Colors.OKCYAN}Running simulations sequentially...{Colors.ENDC}\n")
        completed = 0
        failed = 0
        skipped = 0

        for i, sim_data in enumerate(simulation_tasks):
            sim_data['worker_id'] = 0
            success, desc, elapsed = run_single_simulation(sim_data)
            if elapsed == 0.0:  # Skipped
                skipped += 1
            elif success:
                completed += 1
            else:
                failed += 1
    else:
        # Parallel execution
        print(f"{Colors.OKCYAN}Running simulations in parallel with {num_workers} workers...{Colors.ENDC}")

        # Determine CPU cores to use for pinning
        cpu_cores_to_use = None
        if psutil_available and args.pin_workers:
            try:
                available_cpus = list(range(os.cpu_count() or 1))
                if num_workers <= len(available_cpus):
                    cpu_cores_to_use = available_cpus[:num_workers]
                    print(f"{Colors.OKCYAN}CPU pinning enabled: Workers will be pinned to cores {cpu_cores_to_use}{Colors.ENDC}")
                else:
                    print(f"{Colors.WARNING}Warning: More workers ({num_workers}) than CPU cores ({len(available_cpus)}). CPU pinning disabled.{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.WARNING}Warning: Could not determine CPU cores: {e}. CPU pinning disabled.{Colors.ENDC}")
        elif args.pin_workers and not psutil_available:
            print(f"{Colors.WARNING}Warning: CPU pinning requested but psutil not installed. Install with: pip install psutil{Colors.ENDC}")

        print()  # Empty line for readability

        # Create manager for shared lock
        manager = Manager()
        lock = manager.Lock()

        # Set up global lock for main process
        global print_lock
        print_lock = lock

        # Create pool with custom worker initialization
        # We'll create workers manually to assign CPU cores
        from multiprocessing import Process, Queue

        # Create queues
        task_queue: Any = Queue()
        result_queue: Any = Queue()

        # Start worker processes
        workers = []
        for i in range(num_workers):
            worker_id = i + 1
            cpu_core = cpu_cores_to_use[i] if cpu_cores_to_use else None
            p = Process(target=worker_wrapper, args=(task_queue, result_queue, worker_id, lock, cpu_core))
            p.start()
            workers.append(p)

        # Add tasks to queue
        for sim_data in simulation_tasks:
            task_queue.put(sim_data)

        # Add sentinel values to stop workers
        for _ in range(num_workers):
            task_queue.put(None)

        # Collect results
        results: List[Tuple[bool, str, float]] = []
        for _ in range(len(simulation_tasks)):
            result = result_queue.get()
            results.append(result)

        # Wait for workers to finish
        for p in workers:
            p.join()

        # Count results
        completed = sum(1 for success, _, elapsed in results if success and elapsed > 0)
        failed = sum(1 for success, _, _ in results if not success)
        skipped = sum(1 for _, _, elapsed in results if elapsed == 0.0)

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
