#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <vector>

// Ping-pong using the exact same primitives as sync_kernels.cu:
//   sender:   nvshmem_quiet() + nvshmem_int_put()
//   receiver: nvshmem_int_wait_until()
//
// PE 0 drives the loop and records timestamps. PE 1 reflects.
// One-way latency = round-trip / 2.

__global__ void pe0_loop(int* flag, int peer, int n_total, int n_warmup,
                         unsigned long long* ts_send, unsigned long long* ts_recv)
{
    for (int i = 1; i <= n_total; i++) {
        unsigned long long t;
        if (i > n_warmup)
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts_send[i - n_warmup - 1]));

        int val = i;
        nvshmem_quiet();
        nvshmem_int_put(flag, &val, 1, peer);

        nvshmem_int_wait_until(flag, NVSHMEM_CMP_EQ, i);

        if (i > n_warmup)
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts_recv[i - n_warmup - 1]));
    }
}

__global__ void pe1_loop(int* flag, int peer, int n_total)
{
    for (int i = 1; i <= n_total; i++) {
        nvshmem_int_wait_until(flag, NVSHMEM_CMP_EQ, i);
        int val = i;
        nvshmem_quiet();
        nvshmem_int_put(flag, &val, 1, peer);
    }
}

int main(int argc, char* argv[])
{
    int         n_warmup = 50;
    int         n_iter   = 500;
    const char* output   = nullptr;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--warmup") && i + 1 < argc) n_warmup = atoi(argv[++i]);
        if (!strcmp(argv[i], "--iter")   && i + 1 < argc) n_iter   = atoi(argv[++i]);
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

    int peer    = 1 - mype;
    int n_total = n_warmup + n_iter;

    int* flag = (int*)nvshmem_malloc(sizeof(int));
    cudaMemset(flag, 0, sizeof(int));
    nvshmem_barrier_all();

    unsigned long long* d_ts_send = nullptr;
    unsigned long long* d_ts_recv = nullptr;
    if (mype == 0) {
        cudaMalloc(&d_ts_send, n_iter * sizeof(unsigned long long));
        cudaMalloc(&d_ts_recv, n_iter * sizeof(unsigned long long));
    }

    cudaStream_t s;
    cudaStreamCreate(&s);

    if (mype == 0)
        pe0_loop<<<1, 1, 0, s>>>(flag, peer, n_total, n_warmup, d_ts_send, d_ts_recv);
    else
        pe1_loop<<<1, 1, 0, s>>>(flag, peer, n_total);

    cudaStreamSynchronize(s);

    if (mype == 0) {
        std::vector<unsigned long long> h_send(n_iter), h_recv(n_iter);
        cudaMemcpy(h_send.data(), d_ts_send, n_iter * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_recv.data(), d_ts_recv, n_iter * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);

        std::vector<double> rtts(n_iter);
        double total = 0;
        for (int i = 0; i < n_iter; i++) {
            rtts[i] = (double)(h_recv[i] - h_send[i]);
            total += rtts[i];
        }
        FILE* f = stdout;
        if (output) {
            f = fopen(output, "w");
            if (!f) { perror(output); f = stdout; }
        }
        fprintf(f, "iter,latency_ns\n");
        for (int i = 0; i < n_iter; i++)
            fprintf(f, "%d,%.1f\n", i, rtts[i] / 2.0);
        if (output && f != stdout) fclose(f);

        cudaFree(d_ts_send);
        cudaFree(d_ts_recv);
    }

    nvshmem_free(flag);
    nvshmem_finalize();
    return 0;
}
