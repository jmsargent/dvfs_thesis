#include <stdio.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "mustard.h"


int main(int argc, char *argv[]) {
    // 1. Initialize NVSHMEM (this handles MPI internally)
    int rank = 0;
    // Get local rank from MPI or env to set device BEFORE init
    char* local_rank_str = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (local_rank_str) {
        rank = atoi(local_rank_str);
    }

    int dev_count;
    cudaGetDeviceCount(&dev_count);
    cudaSetDevice(rank % dev_count);

    // Now initialize
    nvshmem_init();

    // 3. Allocate a "Symmetric" integer on the GPU heap
    // This memory exists at the same virtual address on all GPUs
    int *data = (int *)nvshmem_malloc(sizeof(int));

    // Initialize to 0
    printf("I am rank %d, dev_count: %d \n", rank,dev_count );

    int zero = 0;
    cudaMemcpy(data, &zero, sizeof(int), cudaMemcpyHostToDevice);
    nvshmem_barrier_all();

    if (rank == 0) {
        int val = 42;
        printf("Rank 0: Putting value %d into Rank 1's memory...\n", val);
        // Rank 0 writes '42' directly into Rank 1's 'data' pointer
        nvshmem_int_p(data, val, 1);
    }

    // Sync to ensure Rank 0 is finished writing
    nvshmem_barrier_all();

    if (rank == 1) {
        int result;
        cudaMemcpy(&result, data, sizeof(int), cudaMemcpyDeviceToHost);
        printf("Rank 1: Received value %d from Rank 0! Success.\n", result);
    }

    nvshmem_free(data);
    nvshmem_finalize();
    return 0;
}