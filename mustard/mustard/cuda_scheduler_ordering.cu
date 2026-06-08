/**
 * Experiment 1: creation_order
 *   PE 0 only. Single CUDA stream. Builds a graph with N independent spin nodes —
 *   first inserted forward (0..N-1), then reversed (N-1..0). Reports whether
 *   CUDA's execution order tracks insertion order or is unrelated.
 *
 * Experiment 2: staggered
 *   Requires exactly 3 PEs.
 *   PE 0: one spin (long delay) → signals flag[0] on PE 1
 *   PE 2: one spin (short delay) → signals flag[1] on PE 1
 *   PE 1: two independent tasks — A waits on flag[0], B waits on flag[1].
 *         Both launched on separate streams simultaneously.
 *   Reports which of A/B starts first: does the node whose upstream finishes
 *   sooner start sooner, or does launch order dominate?
 */

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "argh.h"
#include "mustard.h"
#include "time_utils.cuh"
#include "utils.h"

int myPE;  // required by mustard.h

// ── kernels ───────────────────────────────────────────────────────────────────

__global__ void spin_ns(unsigned long long duration_ns)
{
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < duration_ns);
}

// ── Experiment 1: Creation Order ─────────────────────────────────────────────

static void run_order_trial(const char* label, const std::vector<int>& creation_order,
                            unsigned long long spin_duration_ns, int n_nodes,
                            unsigned long long* d_ts, cudaStream_t stream,
                            const gpu_clock::CalibrationRef& ref)
{
    checkCudaErrors(cudaMemset(d_ts, 0, 2 * n_nodes * sizeof(unsigned long long)));

    cudaGraph_t graph;
    checkCudaErrors(cudaGraphCreate(&graph, 0));

    for (int i = 0; i < n_nodes; i++)
    {
        int                 node_id   = creation_order[i];
        unsigned long long* ts_start  = d_ts + 2 * node_id;
        unsigned long long* ts_end    = d_ts + 2 * node_id + 1;

        cudaGraphNode_t n_start, n_spin, n_end;

        {
            cudaKernelNodeParams p = {};
            p.func         = (void*)mustard::kernel_record_timestamp;
            p.gridDim      = dim3(1);
            p.blockDim     = dim3(1);
            void* args[]   = {(void*)&ts_start};
            p.kernelParams = args;
            checkCudaErrors(cudaGraphAddKernelNode(&n_start, graph, nullptr, 0, &p));
        }
        {
            cudaKernelNodeParams p = {};
            p.func         = (void*)spin_ns;
            p.gridDim      = dim3(1);
            p.blockDim     = dim3(1);
            void* args[]   = {(void*)&spin_duration_ns};
            p.kernelParams = args;
            checkCudaErrors(cudaGraphAddKernelNode(&n_spin, graph, &n_start, 1, &p));
        }
        {
            cudaKernelNodeParams p = {};
            p.func         = (void*)mustard::kernel_record_timestamp;
            p.gridDim      = dim3(1);
            p.blockDim     = dim3(1);
            void* args[]   = {(void*)&ts_end};
            p.kernelParams = args;
            checkCudaErrors(cudaGraphAddKernelNode(&n_end, graph, &n_spin, 1, &p));
        }
    }

    cudaGraphExec_t exec;
    checkCudaErrors(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    checkCudaErrors(cudaGraphLaunch(exec, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    std::vector<unsigned long long> h_ts(2 * n_nodes);
    checkCudaErrors(cudaMemcpy(h_ts.data(), d_ts, 2 * n_nodes * sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost));

    std::vector<int> by_start(n_nodes);
    for (int i = 0; i < n_nodes; i++) by_start[i] = i;
    std::sort(by_start.begin(), by_start.end(),
              [&](int a, int b) { return h_ts[2 * a] < h_ts[2 * b]; });

    printf("trial=%s\ncreation_rank,node_id,start_ns,end_ns,exec_rank\n", label);
    for (int cr = 0; cr < n_nodes; cr++)
    {
        int node = creation_order[cr];
        int er   = (int)(std::find(by_start.begin(), by_start.end(), node) - by_start.begin());
        printf("%d,%d,%lld,%lld,%d\n", cr, node,
               (long long)gpu_clock::globaltimer_to_unix_ns(h_ts[2 * node], ref),
               (long long)gpu_clock::globaltimer_to_unix_ns(h_ts[2 * node + 1], ref), er);
    }
    printf("\n");

    checkCudaErrors(cudaGraphExecDestroy(exec));
    checkCudaErrors(cudaGraphDestroy(graph));
}

static void exp_creation_order(int n_nodes, unsigned long long spin_ns_val, int n_runs)
{
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    unsigned long long* d_ts;
    checkCudaErrors(cudaMalloc(&d_ts, 2 * n_nodes * sizeof(unsigned long long)));

    std::vector<int> fwd(n_nodes), rev(n_nodes);
    for (int i = 0; i < n_nodes; i++) { fwd[i] = i; rev[i] = n_nodes - 1 - i; }

    for (int r = 0; r < n_runs; r++)
    {
        printf("=== creation_order run %d ===\n", r);
        auto ref = gpu_clock::calibrate(stream);
        run_order_trial("forward", fwd, spin_ns_val, n_nodes, d_ts, stream, ref);

        ref = gpu_clock::calibrate(stream);
        run_order_trial("reverse", rev, spin_ns_val, n_nodes, d_ts, stream, ref);
    }

    checkCudaErrors(cudaFree(d_ts));
    checkCudaErrors(cudaStreamDestroy(stream));
}

// ── Experiment 2: Staggered Delays ───────────────────────────────────────────
// flag[0] = task belonging to PE 0 (long spin), flag[1] = task belonging to PE 2 (short spin).
// PE 1 task A waits flag[0], task B waits flag[1]. A is launched first on stream sA, B on sB.

static void exp_staggered(int myPE, unsigned long long long_ns, unsigned long long short_ns,
                          unsigned long long work_ns, int n_runs)
{
    if (nvshmem_n_pes() != 3)
    {
        if (myPE == 0) fprintf(stderr, "staggered requires exactly 3 PEs\n");
        return;
    }

    int* d_flags = (int*)nvshmem_malloc(2 * sizeof(int));
    checkCudaErrors(cudaMemset(d_flags, 0, 2 * sizeof(int)));

    int  h_notify = 1;
    int* d_notify;
    checkCudaErrors(cudaMalloc(&d_notify, sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_notify, &h_notify, sizeof(int), cudaMemcpyHostToDevice));

    int *d_dep0 = nullptr, *d_dep1 = nullptr;
    if (myPE == 1)
    {
        int h0 = 0, h1 = 1;
        checkCudaErrors(cudaMalloc(&d_dep0, sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_dep1, sizeof(int)));
        checkCudaErrors(cudaMemcpy(d_dep0, &h0, sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_dep1, &h1, sizeof(int), cudaMemcpyHostToDevice));
    }

    // d_ts1: [start_A, end_A, start_B, end_B]
    unsigned long long* d_ts1 = nullptr;
    if (myPE == 1)
        checkCudaErrors(cudaMalloc(&d_ts1, 4 * sizeof(unsigned long long)));

    auto build_sender = [&](cudaStream_t s, int task_id, unsigned long long dur)
    {
        int         debug = 0;
        cudaGraph_t sg;
        checkCudaErrors(cudaGraphCreate(&sg, 0));
        checkCudaErrors(cudaStreamBeginCaptureToGraph(s, sg, nullptr, nullptr, 0,
                                                      cudaStreamCaptureModeGlobal));
        spin_ns<<<1, 1, 0, s>>>(dur);
        mustard::kernel_signal_static<<<1, 1, 0, s>>>(task_id, d_flags, d_notify, 1, debug);
        checkCudaErrors(cudaStreamEndCapture(s, &sg));
        return sg;
    };

    auto build_waiter = [&](cudaStream_t s, int* d_dep, unsigned long long* ts_start,
                             unsigned long long* ts_end)
    {
        int         debug = 0;
        cudaGraph_t sg;
        checkCudaErrors(cudaGraphCreate(&sg, 0));
        checkCudaErrors(cudaStreamBeginCaptureToGraph(s, sg, nullptr, nullptr, 0,
                                                      cudaStreamCaptureModeGlobal));
        mustard::kernel_wait_static<<<1, 1, 0, s>>>(d_dep, 1, d_flags, debug);
        mustard::kernel_record_timestamp<<<1, 1, 0, s>>>(ts_start);
        spin_ns<<<1, 1, 0, s>>>(work_ns);
        mustard::kernel_record_timestamp<<<1, 1, 0, s>>>(ts_end);
        checkCudaErrors(cudaStreamEndCapture(s, &sg));
        return sg;
    };

    cudaStream_t s0;
    checkCudaErrors(cudaStreamCreate(&s0));

    // Build waiter graphs once — they don't depend on sender timing.
    cudaGraph_t     graphA = {}, graphB = {};
    cudaStream_t    sA = {}, sB = {};
    cudaGraphExec_t execA = {}, execB = {};
    if (myPE == 1)
    {
        checkCudaErrors(cudaStreamCreate(&sA));
        checkCudaErrors(cudaStreamCreate(&sB));
        graphA = build_waiter(sA, d_dep0, d_ts1 + 0, d_ts1 + 1);
        graphB = build_waiter(sB, d_dep1, d_ts1 + 2, d_ts1 + 3);
        checkCudaErrors(cudaGraphInstantiate(&execA, graphA, nullptr, nullptr, 0));
        checkCudaErrors(cudaGraphInstantiate(&execB, graphB, nullptr, nullptr, 0));
    }

    if (myPE == 1)
        printf("trial,run,task,start_ns,end_ns,exec_rank\n");

    // Two trials: normal (PE0=long, PE2=short) and inverse (PE0=short, PE2=long).
    // In normal: A waits the long dep, B waits the short dep → B should start first.
    // In inverse: A waits the short dep, B waits the long dep → A should start first.
    const char* trial_names[2] = {"normal", "inverse"};
    for (int t = 0; t < 2; t++)
    {
        unsigned long long dur0 = (t == 0) ? long_ns  : short_ns;
        unsigned long long dur2 = (t == 0) ? short_ns : long_ns;

        cudaGraph_t sender_graph = {};
        if (myPE == 0) sender_graph = build_sender(s0, 0, dur0);
        if (myPE == 2) sender_graph = build_sender(s0, 1, dur2);

        cudaGraphExec_t execSender = {};
        if (myPE == 0 || myPE == 2)
            checkCudaErrors(cudaGraphInstantiate(&execSender, sender_graph, nullptr, nullptr, 0));

        for (int r = 0; r < n_runs; r++)
        {
            nvshmem_barrier_all();
            if (myPE == 0)
            {
                std::vector<int> zeros(2, 0);
                for (int pe = 0; pe < 3; pe++) nvshmem_int_put(d_flags, zeros.data(), 2, pe);
                nvshmem_quiet();
            }
            nvshmem_barrier_all();

            if (myPE == 1)
                checkCudaErrors(cudaMemset(d_ts1, 0, 4 * sizeof(unsigned long long)));

            gpu_clock::CalibrationRef ref = {};
            if (myPE == 1) ref = gpu_clock::calibrate(sA);

            // PE 1: launch A first, then B — launch order is always A-first across both trials.
            if (myPE == 1)
            {
                checkCudaErrors(cudaGraphLaunch(execA, sA));
                checkCudaErrors(cudaGraphLaunch(execB, sB));
            }
            if (myPE == 0 || myPE == 2)
                checkCudaErrors(cudaGraphLaunch(execSender, s0));

            if (myPE == 0 || myPE == 2) checkCudaErrors(cudaStreamSynchronize(s0));
            if (myPE == 1)
            {
                checkCudaErrors(cudaStreamSynchronize(sA));
                checkCudaErrors(cudaStreamSynchronize(sB));
            }
            checkCudaErrors(cudaDeviceSynchronize());

            if (myPE == 1)
            {
                unsigned long long h_ts[4];
                checkCudaErrors(cudaMemcpy(h_ts, d_ts1, 4 * sizeof(unsigned long long),
                                           cudaMemcpyDeviceToHost));

                int64_t start_A = gpu_clock::globaltimer_to_unix_ns(h_ts[0], ref);
                int64_t end_A   = gpu_clock::globaltimer_to_unix_ns(h_ts[1], ref);
                int64_t start_B = gpu_clock::globaltimer_to_unix_ns(h_ts[2], ref);
                int64_t end_B   = gpu_clock::globaltimer_to_unix_ns(h_ts[3], ref);

                printf("%s,%d,A,%lld,%lld,%d\n", trial_names[t], r,
                       (long long)start_A, (long long)end_A, (start_A <= start_B) ? 0 : 1);
                printf("%s,%d,B,%lld,%lld,%d\n", trial_names[t], r,
                       (long long)start_B, (long long)end_B, (start_B < start_A) ? 0 : 1);
            }
        }

        if (myPE == 0 || myPE == 2)
        {
            cudaGraphExecDestroy(execSender);
            cudaGraphDestroy(sender_graph);
        }
    }

    if (myPE == 1)
    {
        cudaGraphExecDestroy(execA);  cudaGraphDestroy(graphA);
        cudaGraphExecDestroy(execB);  cudaGraphDestroy(graphB);
        cudaStreamDestroy(sA);        cudaStreamDestroy(sB);
        cudaFree(d_ts1);              cudaFree(d_dep0);  cudaFree(d_dep1);
    }
    cudaStreamDestroy(s0);
    cudaFree(d_notify);
    nvshmem_free(d_flags);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    auto cmdl = argh::parser(argc, argv);

    std::string experiment;
    cmdl("experiment", "creation_order") >> experiment;

    int    n_nodes  = 5;
    int    n_runs   = 3;
    double spin_ms  = 50.0;
    double long_ms  = 200.0;
    double short_ms = 50.0;
    double work_ms  = 10.0;

    cmdl("nodes",    n_nodes)  >> n_nodes;
    cmdl("runs",     n_runs)   >> n_runs;
    cmdl("spin-ms",  spin_ms)  >> spin_ms;
    cmdl("long-ms",  long_ms)  >> long_ms;
    cmdl("short-ms", short_ms) >> short_ms;
    cmdl("work-ms",  work_ms)  >> work_ms;

    char* lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!lr) { fprintf(stderr, "OMPI_COMM_WORLD_LOCAL_RANK not set\n"); return 1; }
    cudaSetDevice(atoi(lr));

    nvshmem_init();
    myPE = nvshmem_my_pe();

    auto to_ns = [](double ms) { return (unsigned long long)(ms * 1e6); };

    if (experiment == "creation_order")
    {
        if (myPE == 0) exp_creation_order(n_nodes, to_ns(spin_ms), n_runs);
    }
    else if (experiment == "staggered")
    {
        exp_staggered(myPE, to_ns(long_ms), to_ns(short_ms), to_ns(work_ms), n_runs);
    }
    else
    {
        if (myPE == 0)
            fprintf(stderr,
                    "Unknown --experiment='%s'. Options: creation_order, staggered\n",
                    experiment.c_str());
    }

    nvshmem_barrier_all();
    nvshmem_finalize();
    return 0;
}
