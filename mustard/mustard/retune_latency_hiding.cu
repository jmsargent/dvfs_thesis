/**
 * Experiment: hide_retune
 * ------------------------------------------------------------------------------
 * Question: can the host latency of an NVML core-clock change on PE 0 be
 * overlapped with ("hidden behind") a panel transfer to/from PE 1, which lives
 * on a different GPU?
 *
 * The transfer uses the SAME primitive as panel Cholesky / LU: a strided
 * cudaMemcpy2DAsync (device-to-device) over an nvshmem_ptr() P2P pointer into
 * the peer's symmetric buffer (d_matrix <-> d_matrix_remote in those codes).
 *
 * Method (paired, per run, per direction):
 *     plain   : memcpy2D_async + stream_sync                      -> T_plain
 *     retune  : memcpy2D_async + setFrequency(f_to) + stream_sync -> T_retune
 *
 * cudaMemcpy2DAsync is host-non-blocking and copy-engine driven, so its
 * wall-clock duration does NOT depend on the SM clock. It is enqueued first, so
 * setFrequency()'s host blocking time (tau_call) runs while the bytes are in
 * flight on the link; cudaStreamSynchronize() then completes the copy.
 *
 *   delta = T_retune - T_plain  ~=  max(0, tau_call - T_transfer)
 *       delta ~= 0      -> retune fully hidden inside the transfer window
 *       delta  > 0      -> tau_call leaked past the window (panel too small)
 *
 * "send" = PE 0 pushes its panel into PE 1   (dst = peer).
 * "recv" = PE 0 pulls  a panel from PE 1     (src = peer).
 *
 * The panel is a strided N-row x B-col region (row stride N) of an NxN tiled
 * matrix -- the panel-broadcast primitive. With the defaults it is ~256 MB,
 * large enough that the retune fits comfortably inside the transfer window.
 *
 * Requires exactly 2 PEs and NVSHMEM_SYMMETRIC_SIZE >= N*N*8 bytes.
 *
 * Run, e.g.:
 *   NVSHMEM_SYMMETRIC_SIZE=4G mpirun -np 2 ./retune_latency_hiding \
 *       --n 16384 --b 2048 --f-lo 765 --f-hi 2040 --ramp both --dir both --runs 20
 *
 * Pick f-lo / f-hi from supported (mem,graphics) pairs:
 *   nvidia-smi -q -d SUPPORTED_CLOCKS
 */

#include <cuda_runtime.h>
#include <nvshmem.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "argh.h"
#include "mustard.h"
#include "utils.h"

#include "frequency_controller.h"

int myPE;  // required by mustard.h

using clk = std::chrono::steady_clock;
static inline double ms_since(clk::time_point a, clk::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ── the experiment ────────────────────────────────────────────────────────────

static void exp_hide_retune(int myPE, IFrequencyController& freq, int N, int B,
                            int f_lo, int f_hi, int runs, bool do_send, bool do_recv,
                            bool ramp_up, bool ramp_down)
{
    if (nvshmem_n_pes() != 2)
    {
        if (myPE == 0) fprintf(stderr, "hide_retune requires exactly 2 PEs\n");
        return;
    }
    const int peer = (myPE == 0) ? 1 : 0;

    // Symmetric NxN matrix on both PEs, exactly as the factorizations allocate it.
    const size_t nbytes = (size_t)N * N * sizeof(double);
    double* d_sym = (double*)nvshmem_malloc(nbytes);
    if (!d_sym) { if (myPE == 0) fprintf(stderr, "nvshmem_malloc failed; raise NVSHMEM_SYMMETRIC_SIZE\n"); return; }
    checkCudaErrors(cudaMemset(d_sym, 1, nbytes));
    nvshmem_barrier_all();

    // PE 0 drives. d_remote is a P2P pointer into PE 1's symmetric buffer; the
    // local side is PE 0's own buffer -- the d_matrix / d_matrix_remote split.
    double*      d_remote = nullptr;
    cudaStream_t s        = nullptr;

    if (myPE == 0)
    {
        d_remote = (double*)nvshmem_ptr(d_sym, peer);
        if (!d_remote) { fprintf(stderr, "nvshmem_ptr returned null -- P2P unavailable\n"); nvshmem_free(d_sym); return; }
        checkCudaErrors(cudaStreamCreate(&s));

        printf("# results: delta_ms ~ 0 means the retune was hidden\n");
        printf("direction,ramp,run,panel_MB,f_from,f_to,T_plain_ms,T_retune_ms,delta_ms,tau_call_ms\n");
    }

    const size_t pitch  = (size_t)N * sizeof(double);   // row stride of the matrix
    const size_t width  = (size_t)B * sizeof(double);   // B columns copied per row
    const int    height = N;                            // all N rows of the panel
    const double panel_mb = (double)width * height / (1024.0 * 1024.0);

    // One timed panel transfer on PE 0. The async copy is enqueued first so it is
    // in flight on the copy engine while setFrequency() blocks the host thread.
    auto transfer = [&](bool send, bool retune, int f_to) -> std::pair<double, double>
    {
        double* dst = send ? d_remote : d_sym;          // send: local -> peer
        double* src = send ? d_sym    : d_remote;       // recv: peer  -> local

        auto t0 = clk::now();
        checkCudaErrors(cudaMemcpy2DAsync(dst, pitch, src, pitch, width, height,
                                          cudaMemcpyDeviceToDevice, s));
        double tau_call = 0.0;
        if (retune)
        {
            auto c0 = clk::now();
            freq.setFrequency(f_to);     // host blocks here while bytes fly
            auto c1 = clk::now();
            tau_call = ms_since(c0, c1);
        }
        checkCudaErrors(cudaStreamSynchronize(s));   // completes the copy
        auto t1 = clk::now();
        return {ms_since(t0, t1), tau_call};
    };

    struct Combo { bool send; const char* name; };
    std::vector<Combo> dirs;
    if (do_send) dirs.push_back({true,  "send"});
    if (do_recv) dirs.push_back({false, "recv"});

    int ramps[2][2] = {{f_lo, f_hi}, {f_hi, f_lo}};
    const char* rnames[2] = {"up", "down"};
    bool ramp_on[2] = {ramp_up, ramp_down};

    for (int ri = 0; ri < 2; ++ri)
    {
        if (!ramp_on[ri]) continue;
        int f_from = ramps[ri][0], f_to = ramps[ri][1];

        for (auto& d : dirs)
        {
            if (myPE == 0) freq.setFrequency(f_from);

            // Warm the P2P path once (first transfer pays setup costs).
            nvshmem_barrier_all();
            if (myPE == 0) transfer(d.send, false, 0);
            nvshmem_barrier_all();

            for (int r = 0; r < runs; ++r)
            {
                double T_plain = 0, T_retune = 0, tau = 0;

                nvshmem_barrier_all();                       // A
                if (myPE == 0) { auto pr = transfer(d.send, false, 0); T_plain = pr.first; }

                nvshmem_barrier_all();                       // B
                if (myPE == 0)
                {
                    freq.setFrequency(f_from);               // ensure same start point
                    auto rr  = transfer(d.send, true, f_to);
                    T_retune = rr.first;
                    tau      = rr.second;
                    freq.setFrequency(f_from);               // reset for next run
                }

                nvshmem_barrier_all();                       // C
                if (myPE == 0)
                    printf("%s,%s,%d,%.1f,%d,%d,%.4f,%.4f,%.4f,%.4f\n",
                           d.name, rnames[ri], r, panel_mb, f_from, f_to,
                           T_plain, T_retune, T_retune - T_plain, tau);
            }
        }
    }

    if (myPE == 0)
    {
        freq.setFrequency(f_hi);
        cudaStreamDestroy(s);
    }
    nvshmem_free(d_sym);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    auto cmdl = argh::parser(argc, argv);

    int         n    = 16384;    // matrix dimension (row stride / panel height)
    int         b    = 2048;     // panel width in columns -> ~256 MB transfer
    int         f_lo = 765;
    int         f_hi = 2040;
    int         runs = 20;
    std::string dir  = "both";   // send | recv | both
    std::string ramp = "both";   // up | down | both

    cmdl("n",    n)    >> n;
    cmdl("b",    b)    >> b;
    cmdl("f-lo", f_lo) >> f_lo;
    cmdl("f-hi", f_hi) >> f_hi;
    cmdl("runs", runs) >> runs;
    cmdl("dir",  dir)  >> dir;
    cmdl("ramp", ramp) >> ramp;

    bool do_send = (dir == "send"  || dir == "both");
    bool do_recv = (dir == "recv"  || dir == "both");
    bool ramp_up = (ramp == "up"   || ramp == "both");
    bool ramp_dn = (ramp == "down" || ramp == "both");

    char* lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!lr) { fprintf(stderr, "OMPI_COMM_WORLD_LOCAL_RANK not set\n"); return 1; }
    int local_rank = atoi(lr);
    cudaSetDevice(local_rank);

    nvshmem_init();
    myPE = nvshmem_my_pe();

    int dev = 0;
    cudaGetDevice(&dev);
    NvmlFrequencyController freq(dev);

    exp_hide_retune(myPE, freq, n, b, f_lo, f_hi, runs, do_send, do_recv, ramp_up, ramp_dn);

    nvshmem_barrier_all();
    nvshmem_finalize();
    return 0;
}
