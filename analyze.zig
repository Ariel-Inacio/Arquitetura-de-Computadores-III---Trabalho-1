const std = @import("std");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Read stats file
    const stats_file = std.fs.cwd().openFile("m5out/stats.txt", .{}) catch {
        std.debug.print("Error: Could not open m5out/stats.txt\n", .{});
        std.debug.print("Run gem5 first with: ~/src/gem5/build/ALL/gem5.opt cache_config.py ./zig-out/bin/workload\n", .{});
        return;
    };
    defer stats_file.close();

    const content = try stats_file.readToEndAlloc(allocator, 10 * 1024 * 1024);
    defer allocator.free(content);

    // Stats we extract from gem5
    var sim_insts: u64 = 0;
    var cpu_cycles: u64 = 0;

    // Cache hit/miss counts
    var l1i_hits: u64 = 0;
    var l1i_misses: u64 = 0;
    var l1d_hits: u64 = 0;
    var l1d_misses: u64 = 0;
    var l2_hits: u64 = 0;
    var l2_misses: u64 = 0;
    var l3_hits: u64 = 0;
    var l3_misses: u64 = 0;

    // gem5 pre-computed miss rates
    var l1i_miss_rate_gem5: f64 = 0.0;
    var l1d_miss_rate_gem5: f64 = 0.0;
    var l2_miss_rate_gem5: f64 = 0.0;
    var l3_miss_rate_gem5: f64 = 0.0;

    // Access counts
    var l1i_accesses: u64 = 0;
    var l1d_accesses: u64 = 0;
    var l2_accesses: u64 = 0;
    var l3_accesses: u64 = 0;

    // Helper function to parse values
    const parseValue = struct {
        fn parseInt(line: []const u8) u64 {
            if (std.mem.indexOf(u8, line, " ")) |space_idx| {
                const value_str = std.mem.trim(u8, line[space_idx..], " \t");
                if (std.mem.indexOf(u8, value_str, " ")) |end_idx| {
                    return std.fmt.parseInt(u64, value_str[0..end_idx], 10) catch 0;
                }
            }
            return 0;
        }

        fn parseFloat(line: []const u8) f64 {
            if (std.mem.indexOf(u8, line, " ")) |space_idx| {
                const value_str = std.mem.trim(u8, line[space_idx..], " \t");
                if (std.mem.indexOf(u8, value_str, " ")) |end_idx| {
                    return std.fmt.parseFloat(f64, value_str[0..end_idx]) catch 0.0;
                }
            }
            return 0.0;
        }
    };

    // Parse stats file
    var lines = std.mem.tokenizeScalar(u8, content, '\n');
    while (lines.next()) |line| {
        // Skip comments
        if (line.len == 0 or line[0] == '#') continue;

        // Parse key stats
        if (std.mem.indexOf(u8, line, "simInsts")) |_| {
            sim_insts = parseValue.parseInt(line);
        } else if (std.mem.indexOf(u8, line, "system.cpu.numCycles")) |_| {
            cpu_cycles = parseValue.parseInt(line);
        }
        // L1I cache stats
        else if (std.mem.indexOf(u8, line, "system.cpu.icache.overallHits::total")) |_| {
            l1i_hits = parseValue.parseInt(line);
        } else if (std.mem.indexOf(u8, line, "system.cpu.icache.overallMisses::total")) |_| {
            l1i_misses = parseValue.parseInt(line);
        } else if (std.mem.indexOf(u8, line, "system.cpu.icache.overallAccesses::total")) |_| {
            l1i_accesses = parseValue.parseInt(line);
        } else if (std.mem.indexOf(u8, line, "system.cpu.icache.overallMissRate::total")) |_| {
            l1i_miss_rate_gem5 = parseValue.parseFloat(line);
        }
        // L1D cache stats
        else if (std.mem.indexOf(u8, line, "system.cpu.dcache.overallHits::total")) |_| {
            l1d_hits = parseValue.parseInt(line);
        } else if (std.mem.indexOf(u8, line, "system.cpu.dcache.overallMisses::total")) |_| {
            l1d_misses = parseValue.parseInt(line);
        } else if (std.mem.indexOf(u8, line, "system.cpu.dcache.overallAccesses::total")) |_| {
            l1d_accesses = parseValue.parseInt(line);
        } else if (std.mem.indexOf(u8, line, "system.cpu.dcache.overallMissRate::total")) |_| {
            l1d_miss_rate_gem5 = parseValue.parseFloat(line);
        }
        // L2 cache stats
        else if (std.mem.indexOf(u8, line, "system.l2cache.overallHits::total")) |_| {
            l2_hits = parseValue.parseInt(line);
        } else if (std.mem.indexOf(u8, line, "system.l2cache.overallMisses::total")) |_| {
            l2_misses = parseValue.parseInt(line);
        } else if (std.mem.indexOf(u8, line, "system.l2cache.overallAccesses::total")) |_| {
            l2_accesses = parseValue.parseInt(line);
        } else if (std.mem.indexOf(u8, line, "system.l2cache.overallMissRate::total")) |_| {
            l2_miss_rate_gem5 = parseValue.parseFloat(line);
        }
        // L3 cache stats
        else if (std.mem.indexOf(u8, line, "system.l3cache.overallHits::total")) |_| {
            l3_hits = parseValue.parseInt(line);
        } else if (std.mem.indexOf(u8, line, "system.l3cache.overallMisses::total")) |_| {
            l3_misses = parseValue.parseInt(line);
        } else if (std.mem.indexOf(u8, line, "system.l3cache.overallAccesses::total")) |_| {
            l3_accesses = parseValue.parseInt(line);
        } else if (std.mem.indexOf(u8, line, "system.l3cache.overallMissRate::total")) |_| {
            l3_miss_rate_gem5 = parseValue.parseFloat(line);
        }
    }

    // Calculate performance metrics
    const ipc = if (cpu_cycles > 0) @as(f64, @floatFromInt(sim_insts)) / @as(f64, @floatFromInt(cpu_cycles)) else 0.0;
    const cpi = if (sim_insts > 0) @as(f64, @floatFromInt(cpu_cycles)) / @as(f64, @floatFromInt(sim_insts)) else 0.0;

    // MPKI (Misses Per Kilo Instructions)
    const l1i_mpki = if (sim_insts > 0) @as(f64, @floatFromInt(l1i_misses)) * 1000.0 / @as(f64, @floatFromInt(sim_insts)) else 0.0;
    const l1d_mpki = if (sim_insts > 0) @as(f64, @floatFromInt(l1d_misses)) * 1000.0 / @as(f64, @floatFromInt(sim_insts)) else 0.0;
    const l2_mpki = if (sim_insts > 0) @as(f64, @floatFromInt(l2_misses)) * 1000.0 / @as(f64, @floatFromInt(sim_insts)) else 0.0;
    const l3_mpki = if (sim_insts > 0) @as(f64, @floatFromInt(l3_misses)) * 1000.0 / @as(f64, @floatFromInt(sim_insts)) else 0.0;

    // Print results
    std.debug.print("\n=== CACHE PERFORMANCE ANALYSIS ===\n", .{});
    std.debug.print("\n📊 Basic Metrics:\n", .{});
    std.debug.print("  Instructions: {}\n", .{sim_insts});
    std.debug.print("  CPU Cycles: {}\n", .{cpu_cycles});
    std.debug.print("  IPC: {d:.6}\n", .{ipc});
    std.debug.print("  CPI: {d:.6}\n", .{cpi});

    std.debug.print("\n🎯 Cache Performance (using gem5's pre-computed rates):\n", .{});
    std.debug.print("  L1I: {} hits, {} misses (miss rate: {d:.6} | {d:.2}% formatted)\n", .{ l1i_hits, l1i_misses, l1i_miss_rate_gem5, l1i_miss_rate_gem5 * 100.0 });
    std.debug.print("  L1D: {} hits, {} misses (miss rate: {d:.6} | {d:.2}% formatted)\n", .{ l1d_hits, l1d_misses, l1d_miss_rate_gem5, l1d_miss_rate_gem5 * 100.0 });
    std.debug.print("  L2:  {} hits, {} misses (miss rate: {d:.6} | {d:.2}% formatted)\n", .{ l2_hits, l2_misses, l2_miss_rate_gem5, l2_miss_rate_gem5 * 100.0 });
    std.debug.print("  L3:  {} hits, {} misses (miss rate: {d:.6} | {d:.2}% formatted)\n", .{ l3_hits, l3_misses, l3_miss_rate_gem5, l3_miss_rate_gem5 * 100.0 });

    std.debug.print("\n📈 MPKI (Misses Per Kilo Instructions):\n", .{});
    std.debug.print("  L1I MPKI: {d:.6}\n", .{l1i_mpki});
    std.debug.print("  L1D MPKI: {d:.4}\n", .{l1d_mpki});
    std.debug.print("  L2 MPKI: {d:.4}\n", .{l2_mpki});
    std.debug.print("  L3 MPKI: {d:.4}\n", .{l3_mpki});
    std.debug.print("  Total MPKI: {d:.4}\n", .{l1i_mpki + l1d_mpki});
}
