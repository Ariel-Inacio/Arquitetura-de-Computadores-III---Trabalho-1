// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: Copyright © 2026 Lucca M. A. Pellegrini <lucca@verticordia.com>

const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    // Target with static linking to avoid cross-platform simulation issues.
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .cpu_arch = .x86_64,
            .os_tag = .linux,
            .abi = .musl,
        },
    });

    const optimize = b.standardOptimizeOption(.{});

    // Check dependencies
    const check_deps = b.step("check-deps", "Check all required dependencies");
    check_deps.makeFn = checkDependencies;

    // Setup Python Environment
    const setup_python = b.step("setup-python", "Setup Python environment with pyenv");
    setup_python.dependOn(check_deps);
    setup_python.makeFn = setupPythonEnvironment;

    // Initialize gem5 submodule
    const init_gem5 = b.step("init-gem5", "Initialize and update gem5 submodule");
    init_gem5.dependOn(check_deps);
    init_gem5.makeFn = initGem5Submodule;

    // Build gem5 simulator
    const build_gem5 = b.step("gem5", "Build gem5 simulator (takes ~50 minutes)");
    build_gem5.dependOn(setup_python);
    build_gem5.dependOn(init_gem5);
    build_gem5.makeFn = buildGem5Simulator;

    // Build gem5 control library (m5)
    const build_m5 = b.step("m5", "Build gem5 m5 control library");
    build_m5.dependOn(setup_python);
    build_m5.dependOn(init_gem5);
    build_m5.makeFn = buildM5Library;

    // Build workloads

    // Set proper paths to headers and libraries.
    const misc_include = b.path("include");
    const gem5_include = b.path("gem5/include");
    const m5_lib = b.path("gem5/util/m5/build/x86/out/libm5.a");

    // Define dataset size configuration for each workload
    const workload_configs = [_]struct {
        name: []const u8,
        real_dataset: []const u8,
    }{
        .{ .name = "seidel-2d", .real_dataset = "MEDIUM_DATASET" },
        .{ .name = "jacobi-2d", .real_dataset = "MEDIUM_DATASET" },
        .{ .name = "floyd-warshall", .real_dataset = "SMALL_DATASET" },
        .{ .name = "gemm", .real_dataset = "MEDIUM_DATASET" },
        .{ .name = "atax", .real_dataset = "LARGE_DATASET" },
    };

    // Handwritten workloads that don't have specific settings
    const other_workloads = [_][]const u8{
        "array_stride",
        "matrix_multiply",
        "random_access",
    };

    const build_workloads = b.step("workloads", "Build and install all workloads");
    build_workloads.dependOn(build_m5);
    build_workloads.dependOn(b.getInstallStep()); // Ensure binaries are put in zig-out

    // Build debug PolyBench versions with MINI dataset and -test suffix
    for (workload_configs) |config| {
        const test_exe = b.addExecutable(.{
            .name = b.fmt("{s}-test", .{config.name}),
            .root_module = b.createModule(.{
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
        });

        test_exe.addCSourceFile(.{
            .file = b.path(b.fmt("workloads/{s}.c", .{config.name})),
            .flags = &[_][]const u8{
                "-std=c99",
                "-Wall",
                "-Wextra",
                "-pedantic",
                "-D_GNU_SOURCE",
                "-DMINI_DATASET",
                "-O3",
            },
        });
        test_exe.addCSourceFile(.{
            .file = b.path("workloads/polybench.c"),
            .flags = &[_][]const u8{
                "-std=c99",
                "-Wall",
                "-Wextra",
                "-pedantic",
                "-D_GNU_SOURCE",
                "-DMINI_DATASET",
                "-O3",
            },
        });

        test_exe.root_module.addIncludePath(misc_include);
        test_exe.root_module.addIncludePath(gem5_include);
        test_exe.root_module.addObjectFile(m5_lib);

        b.installArtifact(test_exe);
        test_exe.step.dependOn(build_m5); // Linking stage depends on libm5.a
        build_workloads.dependOn(&test_exe.step);

        // Build real run version with specified dataset
        const real_exe = b.addExecutable(.{
            .name = config.name,
            .root_module = b.createModule(.{
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
        });

        // Real run dataset executable.
        real_exe.addCSourceFile(.{
            .file = b.path(b.fmt("workloads/{s}.c", .{config.name})),
            .flags = &[_][]const u8{
                "-std=c99",
                "-Wall",
                "-Wextra",
                "-pedantic",
                "-D_GNU_SOURCE",
                b.fmt("-D{s}", .{config.real_dataset}),
                "-O3",
            },
        });
        real_exe.addCSourceFile(.{
            .file = b.path("workloads/polybench.c"),
            .flags = &[_][]const u8{
                "-std=c99",
                "-Wall",
                "-Wextra",
                "-pedantic",
                "-D_GNU_SOURCE",
                b.fmt("-D{s}", .{config.real_dataset}),
                "-O3",
            },
        });

        real_exe.root_module.addIncludePath(misc_include);
        real_exe.root_module.addIncludePath(gem5_include);
        real_exe.root_module.addObjectFile(m5_lib);

        b.installArtifact(real_exe);
        real_exe.step.dependOn(build_m5);
        build_workloads.dependOn(&real_exe.step);
    }

    // Build other workloads that don't have specific settings
    for (other_workloads) |workload_name| {
        const exe = b.addExecutable(.{
            .name = workload_name,
            .root_module = b.createModule(.{
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
        });

        exe.addCSourceFile(.{
            .file = b.path(b.fmt("workloads/{s}.c", .{workload_name})),
            .flags = &[_][]const u8{
                "-std=c99",
                "-Wall",
                "-Wextra",
                "-pedantic",
                "-D_GNU_SOURCE",
                "-O3",
            },
        });

        exe.root_module.addIncludePath(gem5_include);
        exe.root_module.addObjectFile(m5_lib);

        b.installArtifact(exe);
        exe.step.dependOn(build_m5); // Linking stage depends on libm5.a
        build_workloads.dependOn(&exe.step);
    }

    // Print a single confirmation once ALL workloads finished building
    build_workloads.makeFn = printWorkloadsBuilt;

    // Build analyzer utility
    const analyze = b.addExecutable(.{
        .name = "analyze",
        .root_module = b.createModule(.{
            .root_source_file = b.path("analyze.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(analyze);

    // Run analyze script on passed argument
    const run_analyze = b.addRunArtifact(analyze);
    if (b.args) |args| {
        run_analyze.addArgs(args);
    }
    const analyze_step = b.step("analyze", "Analyze gem5 cache statistics");
    analyze_step.dependOn(&run_analyze.step);

    // Run simulations
    const run_simulations = b.step("simulations", "Run all gem5 simulations");
    run_simulations.dependOn(build_gem5); // Ensure gem5 is built
    run_simulations.dependOn(build_workloads); // Ensure workloads are built
    build_m5.dependOn(setup_python); // Ensure venv is set up
    run_simulations.dependOn(b.getInstallStep()); // Ensure workloads are in zig-out
    run_simulations.makeFn = runSimulations;

    // Generate visualizations
    const visualize = b.step("visualize", "Generate visualization plots from results");
    visualize.dependOn(run_simulations);
    build_m5.dependOn(setup_python);
    visualize.makeFn = generateVisualizations;

    // Build the report
    const build_report = b.step("report", "Build the LaTeX report");
    build_report.dependOn(visualize);
    build_report.makeFn = buildReport;

    // Default step just builds workloads
    b.default_step = build_workloads;
}

fn checkDependencies(step: *std.Build.Step, _: std.Build.Step.MakeOptions) anyerror!void {
    const b = step.owner;
    const allocator = b.allocator;

    std.debug.print("\x1b[1;36m==> Checking dependencies...\x1b[0m\n", .{});

    // List of required commands
    const required_commands = [_][]const u8{
        "pyenv",
        "cc",
        "c++",
        "git",
        "make",
        "lualatex",
        "latexrun",
        "bibtex",
    };

    var missing_count: u32 = 0;

    for (required_commands) |cmd| {
        const result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &[_][]const u8{ "which", cmd },
        }) catch |err| {
            std.debug.print("  \x1b[1;31m✗ {s}: not found\x1b[0m (error: {})\n", .{ cmd, err });
            missing_count += 1;
            continue;
        };
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);

        if (result.term.Exited != 0) {
            std.debug.print("  \x1b[1;31m✗ {s}: not found\x1b[0m\n", .{cmd});
            missing_count += 1;
        } else {
            std.debug.print("  \x1b[1;32m✓ {s}: found\x1b[0m\n", .{cmd});
        }
    }

    if (missing_count > 0) {
        std.debug.print("\n\x1b[1;31mMissing dependencies:\x1b[0m\n", .{});
        for (required_commands) |cmd| {
            const r = std.process.Child.run(.{ .allocator = allocator, .argv = &[_][]const u8{ "which", cmd } }) catch null;
            if (r) |res| {
                defer allocator.free(res.stdout);
                defer allocator.free(res.stderr);
                if (res.term.Exited != 0) {
                    std.debug.print("  - {s}\n", .{cmd});
                }
            } else {
                std.debug.print("  - {s}\n", .{cmd});
            }
        }
        std.debug.print("\n\x1b[1mPlease install the missing dependencies before continuing.\x1b[0m\n", .{});
        return error.MissingDependencies;
    }
}

fn setupPythonEnvironment(step: *std.Build.Step, _: std.Build.Step.MakeOptions) anyerror!void {
    const b = step.owner;
    const allocator = b.allocator;

    std.debug.print("\n\x1b[1;36m==> Setting up Python environment...\x1b[0m\n", .{});

    // Install Python 3.14.3 if not already installed
    const install_result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{ "pyenv", "install", "--skip-existing", "3.14.3" },
    }) catch |err| {
        std.debug.print("Failed to install Python 3.14.3: {}\n", .{err});
        return err;
    };
    defer allocator.free(install_result.stdout);
    defer allocator.free(install_result.stderr);

    if (install_result.term.Exited != 0) {
        std.debug.print("Failed to install Python 3.14.3:\n{s}\n", .{install_result.stderr});
        return error.PythonInstallFailed;
    }

    // Initialize venv and install dependencies
    const venv_path = "gem5/venv";
    if (std.fs.cwd().statFile(venv_path)) |_| {} else |err| {
        if (err == error.FileNotFound) {
            std.debug.print("\x1b[1mCreating Python virtual environment...\x1b[0m\n", .{});
            const venv_result = std.process.Child.run(.{
                .allocator = allocator,
                .argv = &[_][]const u8{
                    "env",
                    "PYENV_VERSION=3.14.3",
                    "pyenv",
                    "exec",
                    "python",
                    "-m",
                    "venv",
                    "gem5/venv",
                },
            }) catch |venv_err| {
                std.debug.print("Failed to create venv: {}\n", .{venv_err});
                return venv_err;
            };
            defer allocator.free(venv_result.stdout);
            defer allocator.free(venv_result.stderr);

            if (venv_result.term.Exited != 0) {
                std.debug.print("Failed to create venv:\n{s}\n", .{venv_result.stderr});
                return error.VenvCreationFailed;
            }

            // Install dependencies
            std.debug.print("\x1b[1mInstalling Python dependencies...\x1b[0m\n", .{});
            var pip_child = std.process.Child.init(&[_][]const u8{
                "env",
                "PIP_PROGRESS_BAR=off",
                "./gem5/venv/bin/pip",
                "--disable-pip-version-check",
                "--require-virtualenv",
                "install",
                "-r",
                "requirements.txt",
            }, allocator);
            pip_child.stdout_behavior = .Inherit;
            pip_child.stderr_behavior = .Inherit;
            const pip_term = pip_child.spawnAndWait() catch |pip_err| {
                std.debug.print("Failed to install dependencies: {}\n", .{pip_err});
                return pip_err;
            };
            if (pip_term.Exited != 0) {
                std.debug.print("Failed to install dependencies (exit code {})\n", .{pip_term.Exited});
                return error.DependencyInstallFailed;
            }
        } else {
            return err;
        }
    }
}

fn initGem5Submodule(step: *std.Build.Step, _: std.Build.Step.MakeOptions) anyerror!void {
    const b = step.owner;
    const allocator = b.allocator;

    // Check if the submodule's been properly initialized
    if (std.fs.cwd().statFile("gem5/.git")) |_| {} else |err| {
        if (err == error.FileNotFound) {
            std.debug.print("\n\x1b[1;36m==> Initializing gem5 submodule...\x1b[0m\n", .{});
            const init_result = std.process.Child.run(.{
                .allocator = allocator,
                .argv = &[_][]const u8{ "git", "submodule", "update", "--init", "--recursive" },
            }) catch |init_err| {
                std.debug.print("Failed to initialize submodule: {}\n", .{init_err});
                return init_err;
            };
            defer allocator.free(init_result.stdout);
            defer allocator.free(init_result.stderr);

            if (init_result.term.Exited != 0) {
                std.debug.print("Failed to initialize submodule:\n{s}\n", .{init_result.stderr});
                return error.SubmoduleInitFailed;
            }
        } else {
            return err;
        }
    }

    std.debug.print("  \x1b[1;32m✓ gem5 submodule ready\x1b[0m\n", .{});
}

fn buildGem5Simulator(step: *std.Build.Step, _: std.Build.Step.MakeOptions) anyerror!void {
    const b = step.owner;
    const allocator = b.allocator;

    // Build gem5 simulator if not already built
    const gem5_binary = "gem5/build/ALL/gem5.opt";
    if (std.fs.cwd().statFile(gem5_binary)) |_| {} else |err| {
        if (err == error.FileNotFound) {
            std.debug.print("\n\x1b[1;36m==> Building gem5 simulator...\x1b[0m\n", .{});

            // Get number of processors for parallel build
            const nproc_result = std.process.Child.run(.{
                .allocator = allocator,
                .argv = &[_][]const u8{"nproc"},
            }) catch {
                std.debug.print("Warning: Could not determine number of processors, using 4\n", .{});
                return;
            };
            defer allocator.free(nproc_result.stdout);
            defer allocator.free(nproc_result.stderr);

            const nproc_str = std.mem.trimRight(u8, nproc_result.stdout, "\n");
            const nproc = std.fmt.parseInt(u32, nproc_str, 10) catch 4;
            const jobs = @max(1, nproc / 2); // Use half of the cores for better stability

            std.debug.print("\x1b[1mBuilding gem5 (this will take ~50 minutes)...\x1b[0m\n", .{});
            std.debug.print("\x1b[1mUsing {} parallel jobs\x1b[0m\n", .{jobs});

            const build_cmd = std.fmt.allocPrint(allocator, "-j{}", .{jobs}) catch return error.OutOfMemory;
            defer allocator.free(build_cmd);

            var child = std.process.Child.init(&[_][]const u8{
                "./gem5/venv/bin/scons",
                "-C",
                "gem5",
                "--ignore-style",
                "gem5/build/ALL/gem5.opt",
                build_cmd,
            }, allocator);
            child.stdout_behavior = .Inherit;
            child.stderr_behavior = .Inherit;
            const term = child.spawnAndWait() catch |build_err| {
                std.debug.print("Failed to build gem5: {}\n", .{build_err});
                return build_err;
            };

            if (term.Exited != 0) {
                std.debug.print("\x1b[1;31mFailed to build gem5 (exit code {})\x1b[0m\n", .{term.Exited});
                return error.Gem5BuildFailed;
            }
        } else {
            return err;
        }
    }

    std.debug.print("  \x1b[1;32m✓ gem5 simulator built\x1b[0m\n", .{});
}

fn buildM5Library(step: *std.Build.Step, _: std.Build.Step.MakeOptions) anyerror!void {
    const b = step.owner;
    const allocator = b.allocator;

    // Check if already built
    const m5_lib = "gem5/util/m5/build/x86/out/libm5.a";
    if (std.fs.cwd().statFile(m5_lib)) |_| {} else |err| {
        if (err == error.FileNotFound) {
            std.debug.print("\n\x1b[1;36m==> Building gem5 m5 control library...\x1b[0m\n", .{});

            const build_result = std.process.Child.run(.{
                .allocator = allocator,
                .argv = &[_][]const u8{
                    "./gem5/venv/bin/scons",
                    "-C",
                    "gem5/util/m5",
                    "build/x86/out/m5",
                },
            }) catch |build_err| {
                std.debug.print("Failed to build m5 library: {}\n", .{build_err});
                return build_err;
            };
            defer allocator.free(build_result.stdout);
            defer allocator.free(build_result.stderr);

            if (build_result.term.Exited != 0) {
                std.debug.print("\x1b[1;31mFailed to build m5 library:\x1b[0m\n{s}\n", .{build_result.stderr});
                return error.M5BuildFailed;
            }
        } else {
            return err;
        }
    }

    std.debug.print("  \x1b[1;32m✓ m5 control library built\x1b[0m\n", .{});
}

fn runSimulations(step: *std.Build.Step, _: std.Build.Step.MakeOptions) anyerror!void {
    const b = step.owner;
    const allocator = b.allocator;

    std.debug.print("\n\x1b[1;36m==> Running simulations...\x1b[0m\n", .{});

    // Get number of processors
    const nproc_result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{"nproc"},
    }) catch {
        std.debug.print("Warning: Could not determine number of processors, using 4\n", .{});
        return;
    };
    defer allocator.free(nproc_result.stdout);
    defer allocator.free(nproc_result.stderr);

    const nproc_str = std.mem.trimRight(u8, nproc_result.stdout, "\n");
    const nproc = std.fmt.parseInt(u32, nproc_str, 10) catch 4;
    const jobs = @min(9, nproc); // Do not exceed 9 parallel jobs (OOM likely)

    const polybench_workloads = [_][]const u8{
        "atax",
        "floyd-warshall",
        "gemm",
        "jacobi-2d",
        "seidel-2d",
    };

    std.debug.print("\x1b[1mRunning simulations with {} parallel jobs...\x1b[0m\n", .{jobs});
    std.debug.print("This may take several hours depending on your system.\n", .{});

    for (polybench_workloads) |workload| {
        std.debug.print("\n\x1b[1mRunning simulations for {s}...\x1b[0m\n", .{workload});

        const workload_path = std.fmt.allocPrint(allocator, "zig-out/bin/{s}", .{workload}) catch return error.OutOfMemory;
        defer allocator.free(workload_path);

        const jobs_str = std.fmt.allocPrint(allocator, "-j{}", .{jobs}) catch return error.OutOfMemory;
        defer allocator.free(jobs_str);

        const sim_result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &[_][]const u8{
                "./gem5/venv/bin/python",
                "./run_all_simulations.py",
                "--results-dir=results",
                "gem5/build/ALL/gem5.opt",
                workload_path,
                jobs_str,
                "--pin-workers",
            },
        }) catch |sim_err| {
            std.debug.print("\x1b[1;31mFailed to run simulations for {s}: {}\x1b[0m\n", .{ workload, sim_err });
            return sim_err;
        };
        defer allocator.free(sim_result.stdout);
        defer allocator.free(sim_result.stderr);

        if (sim_result.term.Exited != 0) {
            std.debug.print("\x1b[1;31mSimulations failed for {s}:\x1b[0m\n{s}\n", .{ workload, sim_result.stderr });
        } else {
            std.debug.print("  \x1b[1;32m✓ Simulations complete for {s}\x1b[0m\n", .{workload});
        }
    }
}

fn generateVisualizations(step: *std.Build.Step, _: std.Build.Step.MakeOptions) anyerror!void {
    const b = step.owner;
    const allocator = b.allocator;

    std.debug.print("\n\x1b[1;36m==> Generating visualization plots...\x1b[0m\n", .{});

    const viz_result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{
            "./gem5/venv/bin/python",
            "./visualize_results.py",
        },
    }) catch |viz_err| {
        std.debug.print("\x1b[1;31mFailed to generate visualizations: {}\x1b[0m\n", .{viz_err});
        return viz_err;
    };
    defer allocator.free(viz_result.stdout);
    defer allocator.free(viz_result.stderr);

    if (viz_result.term.Exited != 0) {
        std.debug.print("\x1b[1;31mVisualization generation failed:\x1b[0m\n{s}\n", .{viz_result.stderr});
        return error.VisualizationFailed;
    }

    std.debug.print("  \x1b[1;32m✓ Visualization plots generated\x1b[0m\n", .{});
}

fn buildReport(step: *std.Build.Step, _: std.Build.Step.MakeOptions) anyerror!void {
    const b = step.owner;
    const allocator = b.allocator;

    std.debug.print("\n\x1b[1;36m==> Building LaTeX report...\x1b[0m\n", .{});

    const report_result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{ "make", "-C", "report/", "release" },
    }) catch |report_err| {
        std.debug.print("\x1b[1;31mFailed to build report: {}\x1b[0m\n", .{report_err});
        return report_err;
    };
    defer allocator.free(report_result.stdout);
    defer allocator.free(report_result.stderr);

    if (report_result.term.Exited != 0) {
        std.debug.print("\x1b[1;31mReport build failed:\x1b[0m\n{s}\n", .{report_result.stderr});
        return error.ReportBuildFailed;
    }

    std.debug.print("  \x1b[1;32m✓ Report built successfully\x1b[0m\n", .{});
    std.debug.print("  ⚡ \x1b[0mThe report is available in \x1b[1mreport/\x1b[0m\n", .{});
}

fn printWorkloadsBuilt(step: *std.Build.Step, _: std.Build.Step.MakeOptions) anyerror!void {
    _ = step;
    // Final confirmation after all workload artifacts (and installs) complete
    std.debug.print("  \x1b[1;32m✓ All workloads compiled and installed\x1b[0m\n", .{});
}
