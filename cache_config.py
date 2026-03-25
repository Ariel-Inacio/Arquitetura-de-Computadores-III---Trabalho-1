import argparse

import m5
from m5.objects import *

# Define realistic cache configurations based on real architectures
# Each configuration is (L1I_size, L1D_size, L2_size, L3_size)
CACHE_SIZE_CONFIGS = {
    'baseline': ('32KiB', '32KiB', '256KiB', '8MiB'),     # Current baseline
    'intel_core_i7_6700k': ('32KiB', '32KiB', '256KiB', '8MiB'),  # Skylake
    'intel_core_i9_9900k': ('32KiB', '32KiB', '256KiB', '16MiB'),  # Coffee Lake
    'amd_ryzen_5600x': ('32KiB', '32KiB', '512KiB', '32MiB'),  # Zen 3
    'amd_ryzen_7700x': ('32KiB', '32KiB', '1MiB', '32MiB'),  # Zen 4
    'intel_xeon_gold': ('32KiB', '32KiB', '1MiB', '24MiB'),  # Cascade Lake
    'apple_m1': ('64KiB', '64KiB', '4MiB', '8MiB'),  # M1 (estimated)
    'apple_m2': ('64KiB', '64KiB', '4MiB', '16MiB'),  # M2 (estimated)
    'intel_atom': ('24KiB', '24KiB', '1MiB', '4MiB'),  # Atom (low power)
    'arm_cortex_a78': ('32KiB', '32KiB', '512KiB', '4MiB'),  # ARM mobile
    'ibm_power10': ('48KiB', '32KiB', '2MiB', '8MiB'),  # POWER10
    'small_embedded': ('16KiB', '16KiB', '128KiB', '1MiB'),  # Small embedded
    'large_server': ('64KiB', '64KiB', '2MiB', '64MiB'),  # Large server config
}

# Define realistic cache line sizes (in bytes)
CACHE_LINE_SIZES = [32, 64, 128, 256]

# Define associativity options
ASSOCIATIVITIES = [1, 2, 4, 8, 12, 16]

# Parse command line arguments
parser = argparse.ArgumentParser(description='gem5 cache configuration with customizable parameters')
parser.add_argument('workload', nargs='?', default='./zig-out/bin/matrix_multiply',
                    help='Path to workload binary')
parser.add_argument('workload_args', nargs='*',
                    help='Arguments for the workload')

# Cache parameter arguments (only one should be specified at a time)
cache_group = parser.add_mutually_exclusive_group()
cache_group.add_argument('--cache-config', choices=list(CACHE_SIZE_CONFIGS.keys()),
                         help='Use predefined cache size configuration')
cache_group.add_argument('--cache-line-size', type=int, choices=CACHE_LINE_SIZES,
                         help='Set cache line size in bytes')
cache_group.add_argument('--l1-assoc', type=int, choices=ASSOCIATIVITIES,
                         help='Set L1 cache associativity')
cache_group.add_argument('--l2-assoc', type=int, choices=ASSOCIATIVITIES,
                         help='Set L2 cache associativity')
cache_group.add_argument('--l3-assoc', type=int, choices=ASSOCIATIVITIES,
                         help='Set L3 cache associativity')
args = parser.parse_args()

# Set workload from arguments
workload_path = args.workload
workload_args = args.workload_args

# Initialize cache parameters with baseline values
l1i_size, l1d_size, l2_size, l3_size = CACHE_SIZE_CONFIGS['baseline']
cache_line_size = 64
l1_assoc = 8
l2_assoc = 8
l3_assoc = 16

# Apply cache configuration
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

# System configuration
system = System()
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = '4GHz'
system.clk_domain.voltage_domain = VoltageDomain()
system.mem_mode = 'timing'
system.mem_ranges = [AddrRange('8GiB')]

# CPU: X86 TimingSimpleCPU
system.cpu = X86TimingSimpleCPU()

# L1 instruction cache
system.cpu.icache = Cache(
    size=l1i_size,
    assoc=l1_assoc,
    tag_latency=4,
    data_latency=4,
    response_latency=4,
    mshrs=8,
    tgts_per_mshr=20
)

# L1 data cache
system.cpu.dcache = Cache(
    size=l1d_size,
    assoc=l1_assoc,
    tag_latency=4,
    data_latency=4,
    response_latency=4,
    mshrs=8,
    tgts_per_mshr=20
)

# L2 cache
system.l2cache = Cache(
    size=l2_size,
    assoc=l2_assoc,
    tag_latency=12,
    data_latency=12,
    response_latency=12,
    mshrs=20,
    tgts_per_mshr=12
)

# L3 cache
system.l3cache = Cache(
    size=l3_size,
    assoc=l3_assoc,
    tag_latency=36,
    data_latency=36,
    response_latency=36,
    mshrs=40,
    tgts_per_mshr=12
)

# Size of one cache line
system.cache_line_size = cache_line_size

# Create buses
system.l2bus = SystemXBar(
    frontend_latency=1,
    forward_latency=2,
    response_latency=2,
    width=32
)
system.l3bus = SystemXBar(
    frontend_latency=1,
    forward_latency=2,
    response_latency=2,
    width=32
)
system.membus = SystemXBar(
    frontend_latency=1,
    forward_latency=2,
    response_latency=2,
    width=32
)

# Connect CPU to L1 caches
system.cpu.icache_port = system.cpu.icache.cpu_side
system.cpu.dcache_port = system.cpu.dcache.cpu_side

# Connect L1 to L2
system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports

# Connect L2
system.l2cache.cpu_side = system.l2bus.mem_side_ports
system.l2cache.mem_side = system.l3bus.cpu_side_ports

# Connect L3
system.l3cache.cpu_side = system.l3bus.mem_side_ports
system.l3cache.mem_side = system.membus.cpu_side_ports

# Memory controller
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

# Create interrupt controller for X86
system.cpu.createInterruptController()
system.cpu.interrupts[0].pio = system.membus.mem_side_ports
system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports

# Setup system workload
system.workload = SEWorkload.init_compatible(workload_path)
# system.workload.wait_for_remote_gdb = True

# Create process
process = Process()
process.cmd = [workload_path] + workload_args
system.cpu.workload = process
system.cpu.createThreads()

# Create root and run
root = Root(full_system = False, system = system)
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
