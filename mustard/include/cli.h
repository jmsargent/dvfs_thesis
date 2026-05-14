#pragma once

#include <cctype>
#include <iostream>
#include <string>
#include <utility>

#include "argh.h"
#include "utils.h"

enum class ScalerMode {
    Interval,
    WaittimeDowntune,
    GreedyNpiDowntune,
    CriticalPathRampUp,
    NpiGap,
    NpiGapRamp,
    CombinedSlackAware,
};

// Common configuration shared by all mustard executables.
struct MustardConfig
{
    size_t      N              = 15;
    size_t      B              = 3;
    size_t      T              = 5;
    int         myPE           = 0;
    int         verbose        = 0;
    int         workspace      = 256;  // cublas workspace in kb
    int         smLimit        = 20;
    int         runs           = 1;
    int         repeat         = 1;
    bool        staticMultiGPU = false;
    int         numStreams      = 32;
    int         debugKernels   = 0;
    std::string invocationPath = "";
    std::string measureFlags   = "";  // e.g. "task_wait_time,task_compute_time"
    bool        measureWait    = false;
    bool        measureCompute = false;
    std::string outputDir      = "";  // directory for per-PE output files
    std::string dbPath         = "";  // path to task profile CSV for DVFS tuner
    std::string goalSpec       = "edp";
    unsigned    goalN          = 1;
    unsigned    goalM          = 1;
    bool        logPlan        = false;
    ScalerMode  scalerMode     = ScalerMode::Interval;
    int         baselineFreq   = 2040;
    bool        fakeTuner      = false;
};

// Print common options shared by all executables.
inline void printCommonUsage()
{
    std::cerr << "\n  Common options:\n"
              << "    -n, -N=<int>         Matrix dimension N                       [default: 15]\n"
              << "    -t, -T=<int>         Number of tiles (N must be divisible)    [default: 5]\n"
              << "    --sm, --smLimit=<int> SM limit per kernel (1-108)             [default: 20]\n"
              << "    --ws, --workspace=<int> cuBLAS workspace in kB (1-1048576)   [default: 256]\n"
              << "    -r, --runs=<int>     Number of timing runs                    [default: 1]\n"
              << "    --streams=<int>      Max CUDA streams for concurrent launch    [default: 32]\n"
              << "                         Use --streams=1 for fully sequential execution\n"
              << "    --repeat=<int>       Repeat each compute kernel N times (no   [default: 1]\n"
              << "                         save/restore — result is incorrect)\n"
              << "    -v, --verbose        Enable verbose output\n"
              << "    --verify             Verify result correctness\n"
              << "    --dot                Dump execution graph in DOT format\n"
              << "    --invocations=<path> Log task IDs and names to the specified path\n"
              << "    --measure=<flags>    Comma-separated list of columns to emit (one flag = one\n"
              << "                         column). _ms = CUDA event duration; _ts = absolute\n"
              << "                         Unix nanosecond timestamp.\n"
              << "                           wait_ms       wait duration (ms, CUDA event)\n"
              << "                           compute_ms    compute duration (ms, CUDA event)\n"
              << "                           start_ts      compute start (absolute Unix ns)\n"
              << "                           end_ts        compute end   (absolute Unix ns)\n"
              << "                           wait_start_ts cross-GPU wait start (absolute Unix ns)\n"
              << "                           wait_end_ts   cross-GPU wait end   (absolute Unix ns)\n"
              << "                         wait_start_ts / wait_end_ts are 0 for tasks with no\n"
              << "                         cross-GPU dependency.\n"
              << "                         Example: --measure=wait_start_ts,wait_end_ts,start_ts\n"
              << "    --output=<prefix>    Write per-PE timing output to <prefix>_pe<N>.csv.\n"
              << "                         If omitted, timing CSV is printed to stdout.\n"
              << "    --log-plan           Log the DVFS frequency plan to <prefix>_pe<N>.log.\n"
              << "    --scaler=<name>      DVFS scaler to use                  [default: interval]\n"
              << "                           interval      Interval goal optimizer\n"
              << "                           waittime      Waittime-downtuner\n"
              << "                           greedy-npi    Greedy NPI downtuner\n"
              << "                           cp-ramp-up    Critical-path ramp-up\n"
              << "                           npi-gap       NPI gap flat downtune\n"
              << "                           npi-gap-ramp  NPI gap backward-walk ramp\n"
              << "                           combined-slack Combined slack-aware (prefix + gaps + suffix)\n"
              << "    --baseline-freq=<mhz> Baseline GPU frequency for slack-based scalers [default: 1980]\n"
              << "    --fake-tuner         Inject logging frequency controller (no real NVML calls)\n";
}

// Print usage for lu_mustard / cholesky_mustard (single-node).
inline void printSingleNodeUsage(const char* progName, const char* decomposition)
{
    std::cerr << "Usage: " << progName << " [options]\n"
              << "\n  " << decomposition
              << " decomposition on one or more GPUs using CUDA graphs.\n"
              << "  The number of GPUs is determined by the number of NVSHMEM PEs (MPI ranks).\n"
              << "\n  Mode (pick one; default is single-kernel if none given):\n"
              << "    --tiled              Tiled execution (one graph per tile step)\n"
              << "    --subgraph           Sub-graph (mustard) execution\n"
              << "    --static-multigpu    Static multi-GPU scheduling (round-robin, no atomics)\n";
    printCommonUsage();
    std::cerr << "\n  Examples:\n"
              << "    " << progName << " -n=600 -t=2 --tiled --verify\n"
              << "    nvshmrun -np 4 " << progName << " -n=6000 -t=10 --subgraph -r=5\n"

              << std::endl;
}

// Print usage for p_lu_mustard (partitioned multi-node LU).
inline void printPartitionedUsage(const char* progName)
{
    std::cerr << "Usage: " << progName << " [options]\n"
              << "\n  Partitioned LU decomposition across multiple GPUs / nodes.\n"
              << "  Requires at least 2 MPI ranks (NVSHMEM PEs).\n"
              << "\n  Additional options:\n"
              << "    -p, -P=<int>         PE index whose graph to print (-1=none) [default: 0]\n";
    printCommonUsage();
    std::cerr << "\n  Examples:\n"
              << "    nvshmrun -np 4 " << progName << " -n=6000 -t=12 --verify\n"
              << "    mpirun -np 8 " << progName << " -n=12000 -t=24 -r=5\n"
              << std::endl;
}

// Parse a goal spec string into (energyExponent, timeExponent).
// Supported formats:
//   energy        → (1, 0)
//   e[N] d[M] [p] → explicit exponents, e.g. "edp", "ed2p", "e2d3"
// Returns false and prints an error if the spec is unrecognised.
inline bool parseGoalExponents(const std::string& spec, unsigned& n, unsigned& m)
{
    if (spec == "energy") { n = 1; m = 0; return true; }

    auto parseUint = [](const std::string& s, size_t& i, unsigned& out) {
        unsigned val = 0;
        bool any = false;
        while (i < s.size() && std::isdigit(s[i])) { val = val * 10 + (s[i++] - '0'); any = true; }
        if (any) out = val;
        return any;
    };

    size_t i = 0;
    n = 0; m = 0;

    if (i < spec.size() && spec[i] == 'e') {
        ++i; n = 1;
        parseUint(spec, i, n);
    }

    if (i < spec.size() && spec[i] == 'd') {
        ++i; m = 1;
        parseUint(spec, i, m);
    }

    if (i < spec.size() && spec[i] == 'p') ++i;

    if (i != spec.size() || (n == 0 && m == 0)) {
        std::cerr << "Error: unrecognised goal spec '" << spec << "'. "
                  << "Examples: energy, edp, ed2p, e2d3\n";
        return false;
    }
    return true;
}

// Parse the common CLI arguments shared by all mustard executables.
// Returns false if validation fails (error already printed).
inline bool parseCommonArgs(argh::parser& cmdl, MustardConfig& cfg)
{
    if (cmdl[{"h", "help"}])
    {
        return false;  // caller checks and prints usage
    }
    if (!(cmdl({"N", "n"}, cfg.N) >> cfg.N))
    {
        std::cerr << "Error: Must provide a valid N value! Got '" << cmdl({"N", "n"}).str() << "'"
                  << std::endl;
        return false;
    }
    if (!(cmdl({"t", "T"}, cfg.T) >> cfg.T))
    {
        std::cerr << "Error: Must provide a valid T value! Got '" << cmdl({"T", "t"}).str() << "'"
                  << std::endl;
        return false;
    }
    if (cfg.N % cfg.T > 0)
    {
        std::cerr << "Error: N must be divisible by T! Got 'N=" << cfg.N << " & T=" << cfg.T << "'"
                  << std::endl;
        return false;
    }
    if (!(cmdl({"sm", "SM", "smLimit"}, cfg.smLimit) >> cfg.smLimit) || cfg.smLimit > 108 ||
        cfg.smLimit < 1)
    {
        std::cerr << "Error: Must provide a valid SM Limit value! Got '"
                  << cmdl({"sm", "SM", "smLimit"}).str() << "'" << std::endl;
        return false;
    }
    if (!(cmdl({"workspace", "ws", "w", "W"}, cfg.workspace) >> cfg.workspace) ||
        cfg.workspace > 1024 * 1024 || cfg.workspace < 1)
    {
        std::cerr << "Error: Must provide a valid workspace (in kBytes) value! Got '"
                  << cmdl({"workspace", "ws", "w"}).str() << "'" << std::endl;
        return false;
    }
    if (!(cmdl({"run", "runs", "r", "R"}, cfg.runs) >> cfg.runs) || cfg.runs < 1)
    {
        std::cerr << "Error: Must provide a valid number of runs! Got '"
                  << cmdl({"run", "r", "R"}).str() << "'" << std::endl;
        return false;
    }

    if (!(cmdl({"repeat", "rep"}, cfg.repeat) >> cfg.repeat) || cfg.repeat < 1)
    {
        std::cerr << "Error: Must provide a valid repeat count! Got '"
                  << cmdl({"repeat", "rep"}).str() << "'" << std::endl;
        return false;
    }

    cmdl("invocations", "") >> cfg.invocationPath;
    cfg.staticMultiGPU = cmdl["static-multigpu"];
    cmdl("streams", cfg.numStreams) >> cfg.numStreams;

    cmdl("measure", "") >> cfg.measureFlags;
    {
        auto has = [&](const char* f) { return cfg.measureFlags.find(f) != std::string::npos; };
        cfg.measureWait    = has("wait_ms") || has("wait_start_ts") || has("wait_end_ts");
        cfg.measureCompute = has("compute_ms") || has("start_ts") || has("end_ts");
    }
    cmdl("output", "") >> cfg.outputDir;
    cmdl("db", "") >> cfg.dbPath;
    cmdl("goal", "edp") >> cfg.goalSpec;
    cfg.logPlan = cmdl["log-plan"];

    if (!parseGoalExponents(cfg.goalSpec, cfg.goalN, cfg.goalM)) return false;

    {
        std::string scalerSpec;
        cmdl("scaler", "interval") >> scalerSpec;
        if      (scalerSpec == "interval")      cfg.scalerMode = ScalerMode::Interval;
        else if (scalerSpec == "waittime")      cfg.scalerMode = ScalerMode::WaittimeDowntune;
        else if (scalerSpec == "greedy-npi")    cfg.scalerMode = ScalerMode::GreedyNpiDowntune;
        else if (scalerSpec == "cp-ramp-up")    cfg.scalerMode = ScalerMode::CriticalPathRampUp;
        else if (scalerSpec == "npi-gap")       cfg.scalerMode = ScalerMode::NpiGap;
        else if (scalerSpec == "npi-gap-ramp")   cfg.scalerMode = ScalerMode::NpiGapRamp;
        else if (scalerSpec == "combined-slack") cfg.scalerMode = ScalerMode::CombinedSlackAware;
        else
        {
            std::cerr << "Error: unrecognised scaler '" << scalerSpec
                      << "'. Valid: interval, waittime, greedy-npi, cp-ramp-up, npi-gap, npi-gap-ramp, combined-slack\n";
            return false;
        }
    }

    cmdl("baseline-freq", cfg.baselineFreq) >> cfg.baselineFreq;
    cfg.fakeTuner = cmdl["fake-tuner"];

    return true;
}

// Initialize NVSHMEM and set the CUDA device.
// Sets cfg.myPE and cfg.verbose based on command-line flags.
inline bool initNvshmemDevice(argh::parser& cmdl, MustardConfig& cfg)
{
    int   rank           = -1;
    char* local_rank_str = getenv("OMPI_COMM_WORLD_LOCAL_RANK");

    if (local_rank_str)
    {
        rank = atoi(local_rank_str);
    }
    else
    {
        printf("could not find OMPI_COMM_WORLD_LOCAL_RANK \n");
        exit(1);
    }

    printf("[rank %d] init: got OMPI_COMM_WORLD_LOCAL_RANK\n", rank);
    fflush(stdout);

    int dev_count, using_device;
    checkCudaErrors(cudaGetDeviceCount(&dev_count));
    using_device = rank % dev_count;
    checkCudaErrors(cudaSetDevice(using_device));

    printf("[rank %d] init: cudaSetDevice(%d) ok (dev_count=%d)\n", rank, using_device, dev_count);
    fflush(stdout);

    printf("[rank %d] init: calling nvshmem_init()...\n", rank);
    fflush(stdout);
    nvshmem_init();
    printf("[rank %d] init: nvshmem_init() returned\n", rank);
    fflush(stdout);
    cfg.myPE = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    printf("[rank %d] init: myPE=%d, nPEs=%d\n", rank, cfg.myPE, nvshmem_n_pes());
    fflush(stdout);
    // checkCudaErrors(cudaSetDevice(cfg.myPE));

    cfg.verbose      = cmdl[{"v", "verbose"}] && cfg.myPE == 0;
    cfg.debugKernels = cmdl[{"v", "verbose"}];

    if (cfg.verbose)
    {
        int gpusAvailable = -1;
        checkCudaErrors(cudaGetDeviceCount(&gpusAvailable));
        printf("Hello from NVSHMEM_PE=%d/%d\n", cfg.myPE, nvshmem_n_pes());
        printf("%d GPUs detected, asked to use %d GPUs\n", gpusAvailable, nvshmem_n_pes());
    }
    return true;
}
