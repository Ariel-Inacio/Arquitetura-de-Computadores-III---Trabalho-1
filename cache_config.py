import sys

import m5
from m5.objects import *

# Simple cache hierarchy configuration for gem5
# Usage: ~/src/gem5/build/ALL/gem5.opt cache_config.py [workload_path] [workload_args...]

# Get workload from arguments
if len(sys.argv) > 1:
    workload_path = sys.argv[1]
    workload_args = sys.argv[2:] if len(sys.argv) > 2 else []
else:
    # Default workload
    workload_path = './zig-out/bin/matrix_multiply'
    workload_args = ['32']

# System configuration
system = System()
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = '4GHz'
system.clk_domain.voltage_domain = VoltageDomain()
system.mem_mode = 'timing'
system.mem_ranges = [AddrRange('8GB')]

# CPU: X86 TimingSimpleCPU
system.cpu = X86TimingSimpleCPU()

# L1 instruction cache - 32KB, 8-way
system.cpu.icache = Cache(
    size='32kB',
    assoc=8,
    tag_latency=2,
    data_latency=2,
    response_latency=2,
    mshrs=4,
    tgts_per_mshr=20
)

# L1 data cache - 32KB, 8-way
system.cpu.dcache = Cache(
    size='32kB',
    assoc=8,
    tag_latency=2,
    data_latency=2,
    response_latency=2,
    mshrs=4,
    tgts_per_mshr=20
)

# L2 cache - 256KB, 8-way
system.l2cache = Cache(
    size='256kB',
    assoc=8,
    tag_latency=20,
    data_latency=20,
    response_latency=20,
    mshrs=20,
    tgts_per_mshr=12
)

# L3 cache - 8MB, 16-way
system.l3cache = Cache(
    size='8MB',
    assoc=16,
    tag_latency=40,
    data_latency=40,
    response_latency=40,
    mshrs=40,
    tgts_per_mshr=12
)

# Size of one cache line
system.cache_line_size = 64

# Create buses
system.l2bus = NoncoherentXBar(
    frontend_latency=1,
    forward_latency=2,
    response_latency=2,
    width=32
)
system.l3bus = NoncoherentXBar(
    frontend_latency=1,
    forward_latency=2,
    response_latency=2,
    width=32
)
system.membus = NoncoherentXBar(
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

# Create process
process = Process()
process.cmd = [workload_path] + workload_args
system.cpu.workload = process
system.cpu.createThreads()

# Create root and run
root = Root(full_system = False, system = system)
m5.instantiate()

print(f"Running workload: {workload_path} {' '.join(workload_args)}")
print("Cache Configuration:")
print("  L1I/L1D: 32KB, 8-way associative")
print("  L2: 256KB, 8-way associative")
print("  L3: 8MB, 16-way associative")
print("")

exit_event = m5.simulate()
print(f"\nSimulation complete at tick {m5.curTick()}")
print(f"Exit reason: {exit_event.getCause()}")
