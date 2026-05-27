#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <vector>

// Measures the latency of the tile pull pattern from the benchmark:
//   cudaMemcpy2DAsync(local_buf, tile*sizeof(double),
//                    nvshmem_ptr(remote, 0), N*sizeof(double),
//                    tile*sizeof(double), tile, D2D, stream)
//
// PE 1 pulls tiles from PE 0. Timed with CUDA events.

int main(int argc, char* argv[])
{
    int         tile     = 1250;  // default: LU N=20000, T=16
    int         n        = 20000;
    int         n_iter   = 200;
    int         n_warmup = 20;
    const char* output   = nullptr;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--tile")   && i + 1 < argc) tile     = atoi(argv[++i]);
        if (!strcmp(argv[i], "--n")      && i + 1 < argc) n        = atoi(argv[++i]);
        if (!strcmp(argv[i], "--iter")   && i + 1 < argc) n_iter   = atoi(argv[++i]);
        if (!strcmp(argv[i], "--warmup") && i + 1 < argc) n_warmup = atoi(argv[++i]);
        if (!strcmp(argv[i], "--output") && i + 1 < argc) output   = argv[++i];
    }

    char* lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    cudaSetDevice(lr ? atoi(lr) : 0);

    nvshmem_init();
    int mype = nvshmem_my_pe();

    if (nvshmem_n_pes() != 2) {
        fprintf(stderr, "requires exactly 2 PEs\n");
        nvshmem_finalize();
        return 1;
    }

    // Allocate just enough symmetric memory for one row of tiles (stride N, height tile)
    size_t nbytes = (size_t)n * tile * sizeof(double);
    double* d_src = (double*)nvshmem_malloc(nbytes);
    cudaMemset(d_src, 0, nbytes);
    nvshmem_barrier_all();

    double* d_dst = nullptr;
    cudaMalloc(&d_dst, (size_t)tile * tile * sizeof(double));

    double* d_remote = nullptr;
    if (mype == 1) {
        d_remote = (double*)nvshmem_ptr(d_src, 0);
        if (!d_remote) {
            fprintf(stderr, "nvshmem_ptr returned null — P2P not available\n");
            nvshmem_finalize();
            return 1;
        }
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    nvshmem_barrier_all();

    if (mype == 1) {
        for (int i = 0; i < n_warmup; i++) {
            cudaMemcpy2DAsync(d_dst,    sizeof(double) * tile,
                              d_remote, sizeof(double) * n,
                              sizeof(double) * tile, tile,
                              cudaMemcpyDeviceToDevice, stream);
        }
        cudaStreamSynchronize(stream);

        std::vector<float> times(n_iter);
        for (int i = 0; i < n_iter; i++) {
            cudaEventRecord(ev_start, stream);
            cudaMemcpy2DAsync(d_dst,    sizeof(double) * tile,
                              d_remote, sizeof(double) * n,
                              sizeof(double) * tile, tile,
                              cudaMemcpyDeviceToDevice, stream);
            cudaEventRecord(ev_stop, stream);
            cudaStreamSynchronize(stream);
            cudaEventElapsedTime(&times[i], ev_start, ev_stop);
        }

        FILE* f = stdout;
        if (output) {
            f = fopen(output, "w");
            if (!f) { perror(output); f = stdout; }
        }
        fprintf(f, "iter,latency_ms\n");
        for (int i = 0; i < n_iter; i++)
            fprintf(f, "%d,%.6f\n", i, times[i]);
        if (output && f != stdout) fclose(f);
    }

    nvshmem_barrier_all();

    cudaFree(d_dst);
    nvshmem_free(d_src);
    nvshmem_finalize();
    return 0;
}
