import argparse

import m5
from m5.objects import *

# Define realistic cache configurations based on real architectures
# Each configuration is (L1I_size, L1D_size, L2_size, L3_size)
CACHE_SIZE_CONFIGS = {
    "baseline": ("32KiB", "32KiB", "256KiB", "8MiB"),
    "intel_core_i9_9900k": ("32KiB", "32KiB", "256KiB", "16MiB"),
    "amd_ryzen_5600x": ("32KiB", "32KiB", "512KiB", "32MiB"),
    "amd_ryzen_7700x": ("32KiB", "32KiB", "1MiB", "32MiB"),
    "apple_m1": ("64KiB", "64KiB", "4MiB", "8MiB"),  # Estimated
    "apple_m2": ("64KiB", "64KiB", "4MiB", "16MiB"),  # Estimated
    "intel_atom": ("32KiB", "32KiB", "1MiB", "4MiB"),  # Adjusted for power-of-2
    "arm_cortex_a78": ("32KiB", "32KiB", "512KiB", "4MiB"),
    "ibm_power10": ("32KiB", "32KiB", "2MiB", "8MiB"),  # Adjusted for power-of-2
    "small_embedded": ("16KiB", "16KiB", "128KiB", "1MiB"),
    "large_server": ("64KiB", "64KiB", "2MiB", "64MiB"),
}

# Define realistic cache line sizes (in bytes)
CACHE_LINE_SIZES = [32, 64, 128, 256]

# We don't use 12-way because Gem5 expects powers of 2
ASSOCIATIVITIES = [1, 2, 4, 8, 16]

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="gem5 cache configuration with customizable parameters"
)
parser.add_argument(
    "workload",
    nargs="?",
    default="./zig-out/bin/matrix_multiply",
    help="Path to workload binary",
)
parser.add_argument("workload_args", nargs="*", help="Arguments for the workload")
parser.add_argument(
    "--wait-gdb",
    action="store_true",
    help="Wait for GDB connection before starting simulation",
)
parser.add_argument(
    "--gdb-port", type=int, default=7000, help="Port for remote GDB server"
)

# Cache parameter arguments
cache_group = parser.add_mutually_exclusive_group()
cache_group.add_argument(
    "--cache-config",
    choices=list(CACHE_SIZE_CONFIGS.keys()),
    help="Use predefined cache size configuration",
)
cache_group.add_argument(
    "--cache-line-size",
    type=int,
    choices=CACHE_LINE_SIZES,
    help="Set cache line size in bytes",
)
cache_group.add_argument(
    "--l1-assoc", type=int, choices=ASSOCIATIVITIES, help="Set L1 cache associativity"
)
cache_group.add_argument(
    "--l2-assoc", type=int, choices=ASSOCIATIVITIES, help="Set L2 cache associativity"
)
cache_group.add_argument(
    "--l3-assoc", type=int, choices=ASSOCIATIVITIES, help="Set L3 cache associativity"
)
args = parser.parse_args()

# Set workload from arguments
workload_path = args.workload
workload_args = args.workload_args

# Initialize cache parameters with baseline
l1i_size, l1d_size, l2_size, l3_size = CACHE_SIZE_CONFIGS["baseline"]
cache_line_size = 64
l1_assoc = 8
l2_assoc = 8
l3_assoc = 16

# Apply configuration
config_desc = "Baseline configuration"
if args.cache_config:
    l1i_size, l1d_size, l2_size, l3_size = CACHE_SIZE_CONFIGS[args.cache_config]
    config_desc = f"Cache sizes: {args.cache_config}"
elif args.cache_line_size:
    cache_line_size = args.cache_line_size
    config_desc = f"Cache line size: {cache_line_size} bytes"
elif args.l1_assoc:
    l1_assoc = args.l1_assoc
    config_desc = f"L1 associativity: {l1_assoc}-way"
elif args.l2_assoc:
    l2_assoc = args.l2_assoc
    config_desc = f"L2 associativity: {l2_assoc}-way"
elif args.l3_assoc:
    l3_assoc = args.l3_assoc
    config_desc = f"L3 associativity: {l3_assoc}-way"

# Main system configuration
system = System()
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = "4GHz"
system.clk_domain.voltage_domain = VoltageDomain()
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("8GiB")]
system.cpu = X86TimingSimpleCPU()

# Caches
system.cpu.icache = Cache(
    size=l1i_size,
    assoc=l1_assoc,
    tag_latency=4,
    data_latency=4,
    response_latency=4,
    mshrs=8,
    tgts_per_mshr=20,
)
system.cpu.dcache = Cache(
    size=l1d_size,
    assoc=l1_assoc,
    tag_latency=4,
    data_latency=4,
    response_latency=4,
    mshrs=8,
    tgts_per_mshr=20,
)
system.l2cache = Cache(
    size=l2_size,
    assoc=l2_assoc,
    tag_latency=12,
    data_latency=12,
    response_latency=12,
    mshrs=20,
    tgts_per_mshr=12,
)
system.l3cache = Cache(
    size=l3_size,
    assoc=l3_assoc,
    tag_latency=36,
    data_latency=36,
    response_latency=36,
    mshrs=40,
    tgts_per_mshr=12,
)

# Size of one cache line
system.cache_line_size = cache_line_size

# Buses
system.l2bus = SystemXBar(
    frontend_latency=1, forward_latency=2, response_latency=2, width=32
)
system.l3bus = SystemXBar(
    frontend_latency=1, forward_latency=2, response_latency=2, width=32
)
system.membus = SystemXBar(
    frontend_latency=1, forward_latency=2, response_latency=2, width=32
)

# Connect CPU, caches, and memory
system.cpu.icache_port = system.cpu.icache.cpu_side
system.cpu.dcache_port = system.cpu.dcache.cpu_side
system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports
system.l2cache.cpu_side = system.l2bus.mem_side_ports
system.l2cache.mem_side = system.l3bus.cpu_side_ports
system.l3cache.cpu_side = system.l3bus.mem_side_ports
system.l3cache.mem_side = system.membus.cpu_side_ports

# Memory controller
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

# Interrupt controller
system.cpu.createInterruptController()
system.cpu.interrupts[0].pio = system.membus.mem_side_ports
system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports

# Set workload from command line arguments
system.workload = SEWorkload.init_compatible(workload_path)
system.workload.wait_for_remote_gdb = args.wait_gdb
system.workload.remote_gdb_port = args.gdb_port

# Create process
process = Process()
process.cmd = [workload_path] + workload_args
system.cpu.workload = process
system.cpu.createThreads()

# Create root and run
root = Root(full_system=False, system=system)
m5.instantiate()

print(f"Running workload: {workload_path} {' '.join(workload_args)}")
print(f"\nConfiguration: {config_desc}")
print("\nDetailed Cache Configuration:")
print(f"  L1I: {l1i_size}, {l1_assoc}-way associative")
print(f"  L1D: {l1d_size}, {l1_assoc}-way associative")
print(f"  L2: {l2_size}, {l2_assoc}-way associative")
print(f"  L3: {l3_size}, {l3_assoc}-way associative")
print(f"  Cache line size: {cache_line_size} bytes")
print("")

exit_event = m5.simulate()
print(f"\nSimulation complete at tick {m5.curTick()}")
print(f"Exit reason: {exit_event.getCause()}")
