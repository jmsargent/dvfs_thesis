/**
 * Sample two distributions to decide if an NVML retune hides behind a panel
 * transfer (retune hidden iff tau <= transfer; see the paired run for the
 * T_retune = max(tau, transfer) model). Raw samples out; stats in analyze.py.
 *
 *   Phase 1: retune cost tau (setApplicationsClocks host block) under idle /
 *            send-in-flight / recv-in-flight -- the call can block longer when
 *            the GPU is busy, and the busy number is the deployment-valid one.
 *            The cover copy (--cover-b) is sized to outlast the worst retune.
 *   Phase 2: transfer WALL time, send & recv, with and without a concurrent
 *            retune, swept over panel width b -- the direct hiding test:
 *            retune-wall ~ plain-wall => the retune was hidden.
 *
 * CSV: kind,cond,dir,b,mb,sample,ms   (2 PEs; NVSHMEM_SYMMETRIC_SIZE >= N*N*8)
 */

#include <cuda_runtime.h>
#include <nvshmem.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "argh.h"
#include "frequency_controller.h"
#include "mustard.h"
#include "utils.h"

int myPE;  // required by mustard.h

using clk = std::chrono::steady_clock;
static inline double ms_since(clk::time_point a, clk::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// "256,1024,2048" -> {256, 1024, 2048}
static std::vector<int> parse_int_list(const std::string& s)
{
    std::vector<int> out;
    std::string      tok;
    for (char c : s)
    {
        if (c == ',')
        {
            if (!tok.empty())
            {
                out.push_back(std::atoi(tok.c_str()));
                tok.clear();
            }
        }
        else
        {
            tok.push_back(c);
        }
    }
    if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
    return out;
}

int main(int argc, char** argv)
{
    auto cmdl = argh::parser(argc, argv);

    int         n        = 16384;
    int         f_lo     = 240;
    int         f_hi     = 2040;
    int         n_retune = 2000;
    int         n_xfer   = 500;
    int         cover_b  = 8192;  // cover-copy width for in-flight retune (~52 ms, > worst tau)
    std::string b_list   = "256,1024,2048,3072,4096,8192,14336";

    cmdl("n", n) >> n;
    cmdl("f-lo", f_lo) >> f_lo;
    cmdl("f-hi", f_hi) >> f_hi;
    cmdl("retune-samples", n_retune) >> n_retune;
    cmdl("xfer-samples", n_xfer) >> n_xfer;
    cmdl("cover-b", cover_b) >> cover_b;
    cmdl("b-list", b_list) >> b_list;

    std::vector<int> bs = parse_int_list(b_list);

    char* lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!lr)
    {
        fprintf(stderr, "OMPI_COMM_WORLD_LOCAL_RANK not set\n");
        return 1;
    }
    cudaSetDevice(atoi(lr));

    nvshmem_init();
    myPE = nvshmem_my_pe();
    if (nvshmem_n_pes() != 2)
    {
        if (myPE == 0) fprintf(stderr, "requires exactly 2 PEs\n");
        nvshmem_finalize();
        return 1;
    }
    const int peer = 1 - myPE;

    int dev = 0;
    cudaGetDevice(&dev);
    NvmlFrequencyController freq(dev);

    // Symmetric NxN matrix on both PEs, as the factorizations allocate it.
    const size_t nbytes = (size_t)n * n * sizeof(double);
    double*      d_sym  = (double*)nvshmem_malloc(nbytes);
    if (!d_sym)
    {
        if (myPE == 0) fprintf(stderr, "nvshmem_malloc failed; raise NVSHMEM_SYMMETRIC_SIZE\n");
        nvshmem_finalize();
        return 1;
    }
    checkCudaErrors(cudaMemset(d_sym, 1, nbytes));
    nvshmem_barrier_all();

    if (myPE == 0)
    {
        printf("kind,cond,dir,b,mb,sample,ms\n");

        double* d_remote = (double*)nvshmem_ptr(d_sym, peer);  // P2P pointer; null if unavailable
        if (!d_remote)
            fprintf(stderr,
                    "nvshmem_ptr null -- P2P unavailable; idle retune only, no transfer phase\n");

        cudaStream_t s;
        checkCudaErrors(cudaStreamCreate(&s));
        const size_t pitch       = (size_t)n * sizeof(double);        // row stride of the matrix
        const size_t cover_width = (size_t)cover_b * sizeof(double);  // cover-copy width per row

        // ── Phase 1: retune cost under idle / send-in-flight / recv-in-flight ──
        // (commented out -- reusing samples from prior run)
        //
        // auto transferData = [&](int retuneFrequency, int cond) -> double { ... };
        //
        // struct Cond { int id; const char* name; };
        // std::vector<Cond> conds;
        // conds.push_back({0, "idle"});
        // if (d_remote) { conds.push_back({1, "send"}); conds.push_back({2, "recv"}); }
        //
        // for (auto& c : conds)
        // {
        //     freq.setFrequency(f_hi);
        //     for (int i = 0; i < n_retune; ++i)
        //     {
        //         int target = (i % 2 == 0) ? f_lo : f_hi;
        //         const char* dir = (i % 2 == 0) ? "down" : "up";
        //         double tau = transferData(target, c.id);
        //         printf("retune,%s,%s,0,0.0,%d,%.4f\n", c.name, dir, i, tau);
        //     }
        // }
        freq.setFrequency(f_hi);

        // ── Phase 2: transfer WALL time, send & recv, plain & retune-in-flight ─
        // Wall time = launch -> (optional setFrequency) -> sync, so the retune
        // cost is included when present; plain vs retune histograms directly show
        // hiding (they overlap when tau <= transfer).
        if (d_remote)
        {
            // dir 0 = send (push, local -> peer), 1 = recv (pull, peer -> local).
            auto one_transfer = [&](size_t width, int dir, bool retune, int target) -> double
            {
                double* dst;
                double* src;
                if (dir == 0)
                {
                    dst = d_remote;
                    src = d_sym;
                }
                else
                {
                    dst = d_sym;
                    src = d_remote;
                }

                auto t0 = clk::now();
                checkCudaErrors(cudaMemcpy2DAsync(dst, pitch, src, pitch, width, n,
                                                  cudaMemcpyDeviceToDevice, s));
                if (retune) freq.setFrequency(target);  // host blocks; copy runs concurrently
                checkCudaErrors(cudaStreamSynchronize(s));
                auto t1 = clk::now();
                return ms_since(t0, t1);
            };

            struct Dir
            {
                int         id;
                const char* name;
            };
            struct Mode
            {
                bool        retune;
                const char* name;
            };
            Dir  dirs[2]  = {{0, "send"}, {1, "recv"}};
            Mode modes[2] = {{false, "plain"}, {true, "retune"}};

            for (int b : bs)
            {
                const size_t width = (size_t)b * sizeof(double);
                const double mb    = (double)width * n / (1024.0 * 1024.0);

                for (auto& dr : dirs)
                {
                    for (auto& mo : modes)
                    {
                        freq.setFrequency(f_hi);
                        one_transfer(width, dr.id, false, f_hi);  // warm

                        for (int i = 0; i < n_xfer; ++i)
                        {
                            int         target;
                            const char* retune_dir;
                            if (i % 2 == 0)
                            {
                                target     = f_lo;
                                retune_dir = "down";
                            }
                            else
                            {
                                target     = f_hi;
                                retune_dir = "up";
                            }
                            double ms = one_transfer(width, dr.id, mo.retune, target);
                            if (mo.retune)
                                printf("transfer,retune-%s,%s,%d,%.1f,%d,%.4f\n", retune_dir,
                                       dr.name, b, mb, i, ms);
                            else
                                printf("transfer,plain,%s,%d,%.1f,%d,%.4f\n", dr.name, b, mb, i,
                                       ms);
                        }
                    }
                }
            }
        }

        checkCudaErrors(cudaStreamDestroy(s));
    }

    nvshmem_barrier_all();
    nvshmem_free(d_sym);
    nvshmem_finalize();
    return 0;
}
