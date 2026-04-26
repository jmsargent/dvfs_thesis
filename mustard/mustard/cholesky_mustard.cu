#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "allocator.h"
#include "argh.h"
#include "cli.h"
#include "gen.h"
#include "injectors.h"
#include "mustard.h"
#include "pe_writer.h"
#include "time_utils.cuh"
#include "verify.h"

// Global configuration (populated from CLI in main).
static MustardConfig cfg;
static size_t&       N = cfg.N;
static size_t&       B = cfg.B;
static size_t&       T = cfg.T;
int                  myPE;
static int&          verbose   = cfg.verbose;
static int&          workspace = cfg.workspace;
static int&          smLimit   = cfg.smLimit;
static int&          runs      = cfg.runs;

void trivialCholesky(bool verify)
{
    // Initialize libaries
    cusolverDnHandle_t cusolverDnHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));

    cusolverDnParams_t cusolverDnParams;
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));

    // Initialize data
    double* h_A = (double*)malloc(N * N * sizeof(double));
    generateRandomSymmetricPositiveDefiniteMatrix(h_A, N);

    double* d_A;
    checkCudaErrors(cudaMalloc(&d_A, N * N * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));

    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;

    checkCudaErrors(cusolverDnXpotrf_bufferSize(
        cusolverDnHandle, cusolverDnParams, CUBLAS_FILL_MODE_LOWER, N, CUDA_R_64F, d_A, N,
        CUDA_R_64F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    void* h_workspace = malloc(workspaceInBytesOnHost);

    void* d_workspace;
    checkCudaErrors(cudaMalloc(&d_workspace, workspaceInBytesOnDevice));

    int* d_info;
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));

    CudaEventClock clock;

    clock.start();
    double totalTime = 0.0;

    // Calculate
    for (int i = 0; i < runs; i++)
    {
        checkCudaErrors(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
        clock.start();
        checkCudaErrors(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams, CUBLAS_FILL_MODE_LOWER,
                                         N, CUDA_R_64F, d_A, N, CUDA_R_64F, d_workspace,
                                         workspaceInBytesOnDevice, h_workspace,
                                         workspaceInBytesOnHost, d_info));
        checkCudaErrors(cudaStreamSynchronize(0));
        clock.end();
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemset(d_workspace, 0, workspaceInBytesOnDevice));
        float time = clock.getTimeInSeconds();
        printf("device %d | %d run | time (s): %4.4f\n", myPE, i, time);
        totalTime += time;
    }

    clock.end();

    // Check
    int h_info = 0;
    checkCudaErrors(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0)
    {
        std::cout << "Unsuccessful potrf execution\n\n"
                  << "d_info = " << h_info << "\n\n";
    }

    // Verify
    if (verify)
    {
        double* h_L = (double*)malloc(N * N * sizeof(double));
        checkCudaErrors(cudaMemcpy(h_L, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        cleanCusolverCholeskyDecompositionResult(h_L, N);
        printf("Result passes verification: %d\n",
               verifyCholeskyDecomposition(h_A, h_L, N, verbose));
        free(h_L);
    }

    printf("Total time used (s): %4.4f\n", totalTime);
    // Clean
    free(h_A);
    free(h_workspace);
    checkCudaErrors(cusolverDnDestroy(cusolverDnHandle));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_workspace));
    checkCudaErrors(cudaFree(d_info));
}

void tiledCholesky(bool verify, bool subgraph, bool dot)
{
    auto setup_start = std::chrono::high_resolution_clock::now();

    // Initialize data
    auto originalMatrix = std::make_unique<double[]>(N * N);  // Column-major
    generateRandomSymmetricPositiveDefiniteMatrix(originalMatrix.get(), N);

    // Copy to device
    double*       d_matrix;
    double*       d_matrices;
    double*       d_matrix_remote;
    volatile int* d_flags;
    if (subgraph)
    {
        d_flags    = (volatile int*)nvshmem_malloc(sizeof(int) * 32);
        d_matrices = (double*)nvshmem_malloc(N * N * sizeof(double));
        d_matrix   = (double*)nvshmem_ptr(d_matrices, myPE);
    }
    else
    {
        checkCudaErrors(cudaMalloc(&d_matrix, N * N * sizeof(double)));
    }
    checkCudaErrors(
        cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    if (myPE != 0) d_matrix_remote = (double*)nvshmem_ptr(d_matrices, 0);

    auto getMatrixBlock = [&](double* matrix, int i, int j) { return matrix + i * B + j * B * N; };

    // Initialize libraries
    cusolverDnHandle_t cusolverDnHandle;
    cusolverDnParams_t cusolverDnParams;
    cublasHandle_t     cublasHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
    checkCudaErrors(cublasCreate(&cublasHandle));
    // Prepare constants
    double one      = 1.0;
    double minusOne = -1.0;

    // Prepare buffer for potrf
    int workspaceInBytesOnDevice;

    checkCudaErrors(cusolverDnDpotrf_bufferSize(cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, B,
                                                d_matrix, N, &workspaceInBytesOnDevice));

    double* d_workspace_cusolver;
    int     workspaces         = T * T;
    void**  d_workspace_cublas = (void**)malloc(sizeof(void*) * workspaces);
    int*    d_info;
    workspaceInBytesOnDevice *= 8;
    checkCudaErrors(cudaMalloc(&d_workspace_cusolver, workspaceInBytesOnDevice));
    int cublasWorkspaceSize = 1024 * workspace;

    for (int i = 0; i < workspaces; i++)
    {
        checkCudaErrors(cudaMalloc(&d_workspace_cublas[i], cublasWorkspaceSize));
    }
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));

    cudaGraph_t graph;
    checkCudaErrors(cudaGraphCreate(&graph, 0));

    int totalNodes = T;

    for (int k = 0; k < T; k++)
        for (int i = k + 1; i < T; i++) totalNodes += 2 + (T - (i + 1));

    if (verbose)
    {
        std::cout << "totalNodes=" << totalNodes << std::endl;
        std::cout << "bufferSize=" << workspaceInBytesOnDevice << std::endl;
        std::cout << "tileSize=" << cublasWorkspaceSize << std::endl;
    }

    cudaStream_t s;
    checkCudaErrors(cudaStreamCreate(&s));

    checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
    checkCudaErrors(cublasSetStream(cublasHandle, s));
    checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));

    auto tiledCholeskyGraphCreator =
        std::make_unique<mustard::TiledGraphCreator>(s, graph, subgraph, totalNodes);

    for (int k = 0; k < T; k++)
    {
        // A[k][k] = GETRF(A[k][k])
        // L[k][k]*U[k][k] = A[k][k]
        checkCudaErrors(
            cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));
        tiledCholeskyGraphCreator->beginCaptureOperation(
            std::make_pair(k, k), {std::make_pair(k, k)},
            "POTRF(" + std::to_string(k) + "," + std::to_string(k) + ")");
        if (subgraph)
        {
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k), sizeof(double) * N,
                                  getMatrixBlock(d_matrix_remote, k, k), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
        }
        checkCudaErrors(cusolverDnDpotrf(cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, B,
                                         getMatrixBlock(d_matrix, k, k), N, d_workspace_cusolver,
                                         workspaceInBytesOnDevice, d_info));
        if (subgraph)
        {
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, k, k), sizeof(double) * N,
                                  getMatrixBlock(d_matrix, k, k), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
        }
        tiledCholeskyGraphCreator->endCaptureOperation();

        for (int i = k + 1; i < T; i++)
        {
            // L[i][k] = TRSM(A[i][k], A[k][k]) // the U part of A[k][k]
            // seems like only these need a separate workspace
            checkCudaErrors(
                cublasSetWorkspace(cublasHandle, d_workspace_cublas[i], cublasWorkspaceSize));
            tiledCholeskyGraphCreator->beginCaptureOperation(
                std::make_pair(i, k), {std::make_pair(k, k), std::make_pair(i, k)},
                "TRSM(" + std::to_string(i) + "," + std::to_string(k) + ")");
            if (subgraph)
            {
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
                if (myPE != 0 && k != 0)
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, i, k), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                if (myPE != 0)
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k), sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, k, k), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            }
            checkCudaErrors(cublasDtrsm(cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                        CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, B, B, &one,
                                        getMatrixBlock(d_matrix, k, k), N,    // k + k * N;
                                        getMatrixBlock(d_matrix, i, k), N));  // k + (i + B) * N;
            if (subgraph)
            {
                if (myPE != 0)
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, k), sizeof(double) * N,
                                      getMatrixBlock(d_matrix, i, k), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            }
            tiledCholeskyGraphCreator->endCaptureOperation();
        }

        for (int i = k + 1; i < T; i++)
        {
            // U[k][i] = TRSM(A[k][k], A[k][i]) // the L part of A[k][k]
            checkCudaErrors(
                cublasSetWorkspace(cublasHandle, d_workspace_cublas[i + T], cublasWorkspaceSize));
            tiledCholeskyGraphCreator->beginCaptureOperation(
                std::make_pair(i, i), {std::make_pair(i, i), std::make_pair(i, k)},
                "SYRK(" + std::to_string(i) + "," + std::to_string(i) + "," + std::to_string(k) +
                    ")");

            if (subgraph)
            {
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
                if (myPE != 0)
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, i, k), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                if (myPE != 0)
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, i), sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, i, i), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            }
            checkCudaErrors(cublasDsyrk(cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, B, B,
                                        &minusOne, getMatrixBlock(d_matrix, i, k), N, &one,
                                        getMatrixBlock(d_matrix, i, i), N));
            if (subgraph)
            {
                if (myPE != 0)
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, i), sizeof(double) * N,
                                      getMatrixBlock(d_matrix, i, i), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            }
            tiledCholeskyGraphCreator->endCaptureOperation();

            for (int j = i + 1; j < T; j++)
            {
                // A[j][i] = GEMM(A[j][k], A[i][k])
                // A[j][i] = A[j][i] - L[j][k] * L[i][k]^T
                checkCudaErrors(cublasSetWorkspace(
                    cublasHandle, d_workspace_cublas[2 * T + (i - 1) * T + (j - 1)],
                    cublasWorkspaceSize));
                tiledCholeskyGraphCreator->beginCaptureOperation(
                    std::make_pair(j, i),
                    {std::make_pair(j, i), std::make_pair(j, k), std::make_pair(i, k)},
                    "GEMM(" + std::to_string(j) + "," + std::to_string(i) + "," +
                        std::to_string(k) + ")");
                if (subgraph)
                {
                    mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
                    if (myPE != 0)
                    {
                        cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), sizeof(double) * N,
                                          getMatrixBlock(d_matrix_remote, i, k), sizeof(double) * N,
                                          sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                        cudaMemcpy2DAsync(getMatrixBlock(d_matrix, j, k), sizeof(double) * N,
                                          getMatrixBlock(d_matrix_remote, j, k), sizeof(double) * N,
                                          sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                        cudaMemcpy2DAsync(getMatrixBlock(d_matrix, j, i), sizeof(double) * N,
                                          getMatrixBlock(d_matrix_remote, j, i), sizeof(double) * N,
                                          sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                    }
                }
                checkCudaErrors(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, B, B, B,
                                             &minusOne, getMatrixBlock(d_matrix, j, k), CUDA_R_64F,
                                             N, getMatrixBlock(d_matrix, i, k), CUDA_R_64F, N, &one,
                                             getMatrixBlock(d_matrix, j, i), CUDA_R_64F, N,
                                             CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
                if (subgraph)
                {
                    if (myPE != 0)
                        cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, j, i), sizeof(double) * N,
                                          getMatrixBlock(d_matrix, j, i), sizeof(double) * N,
                                          sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                    mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
                }
                tiledCholeskyGraphCreator->endCaptureOperation();
            }
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());

    cudaGraphExec_t graphExec;
    CudaEventClock  clock;
    double          totalTime = 0.0;

    if (subgraph)
    {
        if (verbose) tiledCholeskyGraphCreator->printDeps();

        // volatile int *d_flags;
        int*      h_dependencies;  //, *d_dependencies;
        const int queue_size = totalNodes * 2;
        if (verbose) std::cout << "Creating queue..." << std::endl;
        BrokerWorkDistributor queue(queue_size);
        if (verbose) std::cout << "Allocating memory..." << std::endl;

        int* d_dependencies = (int*)nvshmem_malloc(sizeof(int) * totalNodes);
        checkCudaErrors(cudaMallocHost(&h_dependencies, sizeof(int) * totalNodes));
        if (verbose) std::cout << "Setting dependencies..." << std::endl;

        for (int i = 0; i < totalNodes; i++)
        {
            h_dependencies[i] = tiledCholeskyGraphCreator->subgraphDependencies[i].size();
        }
        if (verbose) std::cout << "Populating the queue..." << std::endl;

        checkCudaErrors(cudaMemcpy((void*)d_dependencies, (void*)h_dependencies,
                                   sizeof(int) * totalNodes, cudaMemcpyHostToDevice));
        if (myPE == 0)
            mustard::kernel_populate_queue<<<108, 1024>>>(queue, d_dependencies, totalNodes);
        checkCudaErrors(cudaDeviceSynchronize());
        if (verbose) std::cout << "Inserting dependency kernels..." << std::endl;

        for (int dst = 0; dst < totalNodes; dst++)
            for (int src_ind = 0; src_ind < h_dependencies[dst]; src_ind++)
                tiledCholeskyGraphCreator->insertDependencyKernel(
                    tiledCholeskyGraphCreator->subgraphDependencies[dst][src_ind], dst, queue,
                    d_dependencies);
        if (verbose) showMemUsage();
        if (verbose) std::cout << "Uploading graphs..." << std::endl;

        if (!cfg.invocationPath.empty())
        {
            tiledCholeskyGraphCreator->printInvocations(cfg.invocationPath, myPE);
        }

        cudaGraphExec_t* h_subgraphsExec = new cudaGraphExec_t[totalNodes];
        cudaGraphExec_t* d_subgraphsExec;
        for (int i = 0; i < totalNodes; i++)
        {
            char filename[20];
            sprintf(filename, "./graph_%d.dot", i);
            if (dot)
                checkCudaErrors(
                    cudaGraphDebugDotPrint(tiledCholeskyGraphCreator->subgraphs[i], filename, 0));
            checkCudaErrors(cudaGraphInstantiate(&h_subgraphsExec[i],
                                                 tiledCholeskyGraphCreator->subgraphs[i],
                                                 cudaGraphInstantiateFlagDeviceLaunch));
            cudaGraphUpload(h_subgraphsExec[i], s);
        }
        checkCudaErrors(cudaMalloc(&d_subgraphsExec, sizeof(cudaGraphExec_t) * totalNodes));
        checkCudaErrors(cudaMemcpy((void*)d_subgraphsExec, (void*)h_subgraphsExec,
                                   sizeof(cudaGraphExec_t) * totalNodes, cudaMemcpyHostToDevice));

        if (verbose) std::cout << "Initializing scheduler..." << std::endl;
        cudaGraph_t     schedulerGraph;
        cudaGraphExec_t schedulerExec;
        checkCudaErrors(cudaGraphCreate(&schedulerGraph, 0));
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        mustard::kernel_scheduler<<<1, 1, 0, s>>>(queue, d_flags, d_subgraphsExec, totalNodes,
                                                  myPE);
        cudaStreamEndCapture(s, &schedulerGraph);
        checkCudaErrors(cudaGraphInstantiate(&schedulerExec, schedulerGraph,
                                             cudaGraphInstantiateFlagDeviceLaunch));
        checkCudaErrors(cudaDeviceSynchronize());
        if (verbose) showMemUsage();
        if (verbose) std::cout << "Launching..." << std::endl;

        auto   setup_end  = std::chrono::high_resolution_clock::now();
        double setup_time = std::chrono::duration<double>(setup_end - setup_start).count();
        printf("device %d | Setup time (s): %4.4f\n", myPE, setup_time);

        for (int i = 0; i < runs; i++)
        {
            checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double),
                                       cudaMemcpyHostToDevice));
            nvshmem_barrier_all();
            clock.start(s);
            checkCudaErrors(cudaGraphLaunch(schedulerExec, s));
            checkCudaErrors(cudaStreamSynchronize(s));
            clock.end(s);
            checkCudaErrors(cudaDeviceSynchronize());
            nvshmem_barrier_all();
            if (myPE == 0)
            {
                checkCudaErrors(cudaMemset((void*)d_flags, 0, sizeof(int) * 32));
                checkCudaErrors(cudaMemcpy((void*)d_dependencies, (void*)h_dependencies,
                                           sizeof(int) * totalNodes, cudaMemcpyHostToDevice));
                mustard::kernel_populate_queue<<<108, 1024>>>(queue, d_dependencies, totalNodes);
                checkCudaErrors(cudaDeviceSynchronize());
            }
            float time = clock.getTimeInSeconds();
            printf("device %d | %d run | time (s): %4.4f\n", myPE, i, time);
            totalTime += time;
        }
        if (verbose) std::cout << "Done" << std::endl;

        free(h_subgraphsExec);
        checkCudaErrors(cudaFreeHost(h_dependencies));
        checkCudaErrors(cudaFree(d_subgraphsExec));
        nvshmem_free(d_dependencies);
        nvshmem_free((void*)d_flags);
        queue.free_mem();
    }
    else
    {
        if (dot) checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", 0));
        checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

        auto   setup_end  = std::chrono::high_resolution_clock::now();
        double setup_time = std::chrono::duration<double>(setup_end - setup_start).count();
        printf("device %d | Setup time (s): %4.4f\n", myPE, setup_time);

        for (int i = 0; i < runs; i++)
        {
            checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double),
                                       cudaMemcpyHostToDevice));
            clock.start(s);
            checkCudaErrors(cudaGraphLaunch(graphExec, s));
            checkCudaErrors(cudaStreamSynchronize(s));
            clock.end(s);
            checkCudaErrors(cudaDeviceSynchronize());
            float time = clock.getTimeInSeconds();
            printf("device %d | %d run | time (s): %4.4f\n", myPE, i, time);
            totalTime += time;
        }
    }

    if (verify)
    {
        double* h_L = (double*)malloc(N * N * sizeof(double));
        checkCudaErrors(cudaMemcpy(h_L, d_matrix, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        cleanCusolverCholeskyDecompositionResult(h_L, N);
        printf("Result passes verification: %d\n",
               verifyCholeskyDecomposition(originalMatrix.get(), h_L, N, verbose));

        free(h_L);
    }
    printf("Total time used (s): %4.4f\n", totalTime);

    if (!subgraph)
        checkCudaErrors(cudaFree(d_matrix));
    else
        nvshmem_free(d_matrices);
    checkCudaErrors(cudaFree(d_info));
    checkCudaErrors(cudaFree(d_workspace_cusolver));
    for (int i = 0; i < workspaces; i++)
    {
        checkCudaErrors(cudaFree(d_workspace_cublas[i]));
    }
}

void tiledCholeskyStatic(bool verify, bool dot)
{
    auto setup_start = std::chrono::high_resolution_clock::now();

    int nPEs = nvshmem_n_pes();

    // Initialize data
    auto originalMatrix = std::make_unique<double[]>(N * N);  // Column-major
    generateRandomSymmetricPositiveDefiniteMatrix(originalMatrix.get(), N);

    // NVSHMEM allocations (all PEs participate)
    volatile int* d_flags         = (volatile int*)nvshmem_malloc(sizeof(int) * 32);
    double*       d_matrices      = (double*)nvshmem_malloc(N * N * sizeof(double));
    double*       d_matrix        = (double*)nvshmem_ptr(d_matrices, myPE);
    double*       d_matrix_remote = nullptr;
    checkCudaErrors(
        cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    if (myPE != 0) d_matrix_remote = (double*)nvshmem_ptr(d_matrices, 0);

    auto getMatrixBlock = [&](double* matrix, int i, int j) { return matrix + i * B + j * B * N; };

    // Initialize libraries
    cusolverDnHandle_t cusolverDnHandle;
    cusolverDnParams_t cusolverDnParams;
    cublasHandle_t     cublasHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
    checkCudaErrors(cublasCreate(&cublasHandle));

    double one      = 1.0;
    double minusOne = -1.0;

    int workspaceInBytesOnDevice;
    checkCudaErrors(cusolverDnDpotrf_bufferSize(cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, B,
                                                d_matrix, N, &workspaceInBytesOnDevice));

    double* d_workspace_cusolver;
    int     workspaces         = T * T;
    void**  d_workspace_cublas = (void**)malloc(sizeof(void*) * workspaces);
    int*    d_info;
    workspaceInBytesOnDevice *= 8;
    checkCudaErrors(cudaMalloc(&d_workspace_cusolver, workspaceInBytesOnDevice));
    int cublasWorkspaceSize = 1024 * workspace;
    for (int i = 0; i < workspaces; i++)
        checkCudaErrors(cudaMalloc(&d_workspace_cublas[i], cublasWorkspaceSize));
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));

    cudaGraph_t graph;
    checkCudaErrors(cudaGraphCreate(&graph, 0));

    int totalNodes = T;
    for (int k = 0; k < T; k++)
        for (int i = k + 1; i < T; i++) totalNodes += 2 + (T - (i + 1));

    if (verbose)
    {
        std::cout << "totalNodes=" << totalNodes << std::endl;
        std::cout << "bufferSize=" << workspaceInBytesOnDevice << std::endl;
        std::cout << "tileSize=" << cublasWorkspaceSize << std::endl;
    }
    printf("device %d | tiledCholeskyStatic: building %d graphs\n", myPE, totalNodes);
    fflush(stdout);

    cudaStream_t s;
    checkCudaErrors(cudaStreamCreate(&s));
    checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
    checkCudaErrors(cublasSetStream(cublasHandle, s));
    checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));

    auto tiledCholeskyGraphCreator =
        std::make_unique<mustard::TiledGraphCreator>(s, graph, true, totalNodes);

    // Graph construction — verbatim copy of the subgraph path in tiledCholesky
    for (int k = 0; k < T; k++)
    {
        checkCudaErrors(
            cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));
        tiledCholeskyGraphCreator->beginCaptureOperation(
            std::make_pair(k, k), {std::make_pair(k, k)},
            "POTRF(" + std::to_string(k) + "," + std::to_string(k) + ")");
        mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
        if (myPE != 0)
            cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k), sizeof(double) * N,
                              getMatrixBlock(d_matrix_remote, k, k), sizeof(double) * N,
                              sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
        checkCudaErrors(cusolverDnDpotrf(cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, B,
                                         getMatrixBlock(d_matrix, k, k), N, d_workspace_cusolver,
                                         workspaceInBytesOnDevice, d_info));
        if (myPE != 0)
            cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, k, k), sizeof(double) * N,
                              getMatrixBlock(d_matrix, k, k), sizeof(double) * N,
                              sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
        mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
        tiledCholeskyGraphCreator->endCaptureOperation();

        for (int i = k + 1; i < T; i++)
        {
            checkCudaErrors(
                cublasSetWorkspace(cublasHandle, d_workspace_cublas[i], cublasWorkspaceSize));
            tiledCholeskyGraphCreator->beginCaptureOperation(
                std::make_pair(i, k), {std::make_pair(k, k), std::make_pair(i, k)},
                "TRSM(" + std::to_string(i) + "," + std::to_string(k) + ")");
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
            if (myPE != 0 && k != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), sizeof(double) * N,
                                  getMatrixBlock(d_matrix_remote, i, k), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k), sizeof(double) * N,
                                  getMatrixBlock(d_matrix_remote, k, k), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            checkCudaErrors(cublasDtrsm(cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                        CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, B, B, &one,
                                        getMatrixBlock(d_matrix, k, k), N,
                                        getMatrixBlock(d_matrix, i, k), N));
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, k), sizeof(double) * N,
                                  getMatrixBlock(d_matrix, i, k), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            tiledCholeskyGraphCreator->endCaptureOperation();
        }

        for (int i = k + 1; i < T; i++)
        {
            checkCudaErrors(
                cublasSetWorkspace(cublasHandle, d_workspace_cublas[i + T], cublasWorkspaceSize));
            tiledCholeskyGraphCreator->beginCaptureOperation(
                std::make_pair(i, i), {std::make_pair(i, i), std::make_pair(i, k)},
                "SYRK(" + std::to_string(i) + "," + std::to_string(i) + "," + std::to_string(k) +
                    ")");
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), sizeof(double) * N,
                                  getMatrixBlock(d_matrix_remote, i, k), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, i), sizeof(double) * N,
                                  getMatrixBlock(d_matrix_remote, i, i), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            checkCudaErrors(cublasDsyrk(cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, B, B,
                                        &minusOne, getMatrixBlock(d_matrix, i, k), N, &one,
                                        getMatrixBlock(d_matrix, i, i), N));
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, i), sizeof(double) * N,
                                  getMatrixBlock(d_matrix, i, i), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            tiledCholeskyGraphCreator->endCaptureOperation();

            for (int j = i + 1; j < T; j++)
            {
                checkCudaErrors(cublasSetWorkspace(
                    cublasHandle, d_workspace_cublas[2 * T + (i - 1) * T + (j - 1)],
                    cublasWorkspaceSize));
                tiledCholeskyGraphCreator->beginCaptureOperation(
                    std::make_pair(j, i),
                    {std::make_pair(j, i), std::make_pair(j, k), std::make_pair(i, k)},
                    "GEMM(" + std::to_string(j) + "," + std::to_string(i) + "," +
                        std::to_string(k) + ")");
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
                if (myPE != 0)
                {
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, i, k), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, j, k), sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, j, k), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, j, i), sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, j, i), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                }
                checkCudaErrors(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, B, B, B,
                                             &minusOne, getMatrixBlock(d_matrix, j, k), CUDA_R_64F,
                                             N, getMatrixBlock(d_matrix, i, k), CUDA_R_64F, N, &one,
                                             getMatrixBlock(d_matrix, j, i), CUDA_R_64F, N,
                                             CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
                if (myPE != 0)
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, j, i), sizeof(double) * N,
                                      getMatrixBlock(d_matrix, j, i), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
                tiledCholeskyGraphCreator->endCaptureOperation();
            }
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    printf("device %d | tiledCholeskyStatic: graph construction done\n", myPE);
    fflush(stdout);

    // Parse measure flags — each flag corresponds to exactly one output column.
    // _ms  = CUDA event elapsed duration in milliseconds
    // _ts  = absolute Unix nanosecond timestamp (wall clock)
    auto has_flag    = [&](const char* f) { return cfg.measureFlags.find(f) != std::string::npos; };
    bool col_wait_ms = has_flag("wait_ms");
    bool col_compute_ms    = has_flag("compute_ms");
    bool col_start_ts      = has_flag("start_ts");
    bool col_end_ts        = has_flag("end_ts");
    bool col_wait_start_ts = has_flag("wait_start_ts");
    bool col_wait_end_ts   = has_flag("wait_end_ts");

    // NVSHMEM completion flags
    int* d_completion_flags = (int*)nvshmem_malloc(sizeof(int) * totalNodes);
    checkCudaErrors(cudaMemset(d_completion_flags, 0, sizeof(int) * totalNodes));

    auto scheduler = std::make_unique<mustard::StaticRoundRobinScheduler>(
        nPEs, myPE, totalNodes, tiledCholeskyGraphCreator->subgraphDependencies);

    mustard::TaskAllocator alloc;
    for (int task : scheduler->getMyTasksOrdered())
    {
        const auto& d = scheduler->getDeps(task);
        scheduler->setTaskDeps(task, alloc.allocate(d), (int)d.size());
        const auto& n = scheduler->getNotifyPEs(task);
        scheduler->setTaskNotifyPEs(task, alloc.allocate(n), (int)n.size());
    }

    const std::vector<int>& my_tasks_sorted = scheduler->getMyTasksOrdered();

    // Build injector chain based on measurement flags
    mustard::InjectionContext ctx(totalNodes);
    {
        auto injector = std::unique_ptr<mustard::IInjector>(
            new mustard::SubgraphInjector(tiledCholeskyGraphCreator->subgraphs, *scheduler,
                                          d_completion_flags, cfg.debugKernels));
        if (col_wait_start_ts || col_wait_end_ts)
            injector = std::make_unique<mustard::WaitTimestampDecorator>(
                std::move(injector), tiledCholeskyGraphCreator->subgraphs);
        if (col_wait_ms || col_compute_ms)
            injector = std::make_unique<mustard::WaitTimeDecorator>(
                std::move(injector), tiledCholeskyGraphCreator->subgraphs);
        if (col_compute_ms)
            injector = std::make_unique<mustard::ComputeTimeDecorator>(
                std::move(injector), tiledCholeskyGraphCreator->subgraphs);
        if (col_start_ts || col_end_ts)
            injector = std::make_unique<mustard::TimestampDecorator>(
                std::move(injector), tiledCholeskyGraphCreator->subgraphs);
        injector->inject(my_tasks_sorted, ctx);
    }

    // Instantiate and upload owned subgraphs
    cudaGraphExec_t* h_subgraphsExec = new cudaGraphExec_t[totalNodes];
    for (int task : my_tasks_sorted)
    {
        if (dot)
        {
            char filename[32];
            sprintf(filename, "./graph_%d_%d.dot", task, myPE);
            checkCudaErrors(
                cudaGraphDebugDotPrint(tiledCholeskyGraphCreator->subgraphs[task], filename, 0));
        }
        checkCudaErrors(cudaGraphInstantiate(&h_subgraphsExec[task],
                                             tiledCholeskyGraphCreator->subgraphs[task], nullptr,
                                             nullptr, 0));
        cudaGraphUpload(h_subgraphsExec[task], s);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    printf("device %d | tiledCholeskyStatic: graphs instantiated, entering timing loop\n", myPE);
    fflush(stdout);

    if (!cfg.invocationPath.empty())
        tiledCholeskyGraphCreator->printInvocations(cfg.invocationPath, myPE);

    auto   setup_end  = std::chrono::high_resolution_clock::now();
    double setup_time = std::chrono::duration<double>(setup_end - setup_start).count();
    printf("device %d | Setup time (s): %4.4f\n", myPE, setup_time);
    fflush(stdout);

    struct TaskTiming
    {
        float              wait_ms       = 0.0f;
        float              compute_ms    = 0.0f;
        unsigned long long start_ns      = 0;
        unsigned long long end_ns        = 0;
        unsigned long long wait_start_ns = 0;
        unsigned long long wait_end_ns   = 0;
    };
    int                                  numMyTasks = (int)my_tasks_sorted.size();
    std::vector<std::vector<TaskTiming>> all_timings(runs, std::vector<TaskTiming>(numMyTasks));
    std::vector<unsigned long long>      h_timestamps(totalNodes * 2);
    std::vector<unsigned long long>      h_wait_timestamps(totalNodes * 2);

    double totalTime = 0.0;

    for (int i = 0; i < runs; i++)
    {
        checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double),
                                   cudaMemcpyHostToDevice));

        // Reset completion flags on all PEs
        nvshmem_barrier_all();
        if (myPE == 0)
        {
            std::vector<int> zeros(totalNodes, 0);
            for (int pe = 0; pe < nPEs; pe++)
                nvshmem_int_put(d_completion_flags, zeros.data(), totalNodes, pe);
            nvshmem_quiet();
        }
        nvshmem_barrier_all();

        int                       numStreams = std::min(numMyTasks, 32);
        std::vector<cudaStream_t> taskStreams(numStreams);
        for (int si = 0; si < numStreams; si++) checkCudaErrors(cudaStreamCreate(&taskStreams[si]));

        cudaEvent_t ev_ref = nullptr;
        if (col_wait_ms)
        {
            checkCudaErrors(cudaEventCreate(&ev_ref));
            checkCudaErrors(cudaEventRecord(ev_ref, taskStreams[0]));
        }

        gpu_clock::CalibrationRef ts_ref;
        if (col_start_ts || col_end_ts || col_wait_start_ts || col_wait_end_ts)
            ts_ref = gpu_clock::calibrate(taskStreams[0]);

        if (myPE == 0) print_timestamp("cholesky tiledStatic start_time", 7);
        auto t_start = std::chrono::high_resolution_clock::now();
        for (int idx = 0; idx < numMyTasks; idx++)
        {
            int task = my_tasks_sorted[idx];
            if (verbose)
            {
                printf("device %d | run %d | launching task %d/%d: %s\n", myPE, i, idx + 1,
                       numMyTasks, tiledCholeskyGraphCreator->subgraphOpNames[task].c_str());
                fflush(stdout);
            }
            checkCudaErrors(cudaGraphLaunch(h_subgraphsExec[task], taskStreams[idx % numStreams]));
        }
        for (int si = 0; si < numStreams; si++)
            checkCudaErrors(cudaStreamSynchronize(taskStreams[si]));
        checkCudaErrors(cudaDeviceSynchronize());
        auto t_end = std::chrono::high_resolution_clock::now();
        if (myPE == 0) print_timestamp("cholesky tiledStatic end_time", 7);

        if (col_wait_ms || col_compute_ms)
        {
            for (int idx = 0; idx < numMyTasks; idx++)
            {
                int         task = my_tasks_sorted[idx];
                TaskTiming& tt   = all_timings[i][idx];
                if (col_wait_ms)
                {
                    if (ctx.task_wait_node[task] != nullptr)
                        checkCudaErrors(
                            cudaEventElapsedTime(&tt.wait_ms, ev_ref, ctx.compute_start[task]));
                    else
                        tt.wait_ms = 0.0f;
                }
                if (col_compute_ms)
                    checkCudaErrors(cudaEventElapsedTime(&tt.compute_ms, ctx.compute_start[task],
                                                         ctx.compute_end[task]));
            }
        }
        if (col_start_ts || col_end_ts)
        {
            checkCudaErrors(cudaMemcpy(h_timestamps.data(), ctx.d_timestamps,
                                       sizeof(unsigned long long) * totalNodes * 2,
                                       cudaMemcpyDeviceToHost));
            for (int idx = 0; idx < numMyTasks; idx++)
            {
                int         task = my_tasks_sorted[idx];
                TaskTiming& tt   = all_timings[i][idx];
                if (col_start_ts)
                    tt.start_ns =
                        gpu_clock::globaltimer_to_unix_ns(h_timestamps[task * 2 + 0], ts_ref);
                if (col_end_ts)
                    tt.end_ns =
                        gpu_clock::globaltimer_to_unix_ns(h_timestamps[task * 2 + 1], ts_ref);
            }
        }
        if ((col_wait_start_ts || col_wait_end_ts) && ctx.d_wait_timestamps)
        {
            checkCudaErrors(cudaMemcpy(h_wait_timestamps.data(), ctx.d_wait_timestamps,
                                       sizeof(unsigned long long) * totalNodes * 2,
                                       cudaMemcpyDeviceToHost));
            for (int idx = 0; idx < numMyTasks; idx++)
            {
                int         task = my_tasks_sorted[idx];
                TaskTiming& tt   = all_timings[i][idx];
                if (col_wait_start_ts && h_wait_timestamps[task * 2 + 0] != 0)
                    tt.wait_start_ns =
                        gpu_clock::globaltimer_to_unix_ns(h_wait_timestamps[task * 2 + 0], ts_ref);
                if (col_wait_end_ts && h_wait_timestamps[task * 2 + 1] != 0)
                    tt.wait_end_ns =
                        gpu_clock::globaltimer_to_unix_ns(h_wait_timestamps[task * 2 + 1], ts_ref);
            }
        }

        if (ev_ref) checkCudaErrors(cudaEventDestroy(ev_ref));
        for (int si = 0; si < numStreams; si++) checkCudaErrors(cudaStreamDestroy(taskStreams[si]));
        nvshmem_barrier_all();

        double time = std::chrono::duration<double>(t_end - t_start).count();
        printf("device %d | %d run | time (s): %4.4f\n", myPE, i, time);
        totalTime += time;
    }
    printf("Total time used (s): %4.4f\n", totalTime);

    if (col_wait_ms || col_compute_ms || col_start_ts || col_end_ts || col_wait_start_ts ||
        col_wait_end_ts)
    {
        PEWriter out(cfg.outputPrefix, myPE);

        out.print("pe,run,task_id,op_name");
        if (col_wait_ms) out.print(",wait_ms");
        if (col_compute_ms) out.print(",compute_ms");
        if (col_start_ts) out.print(",start_ts");
        if (col_end_ts) out.print(",end_ts");
        if (col_wait_start_ts) out.print(",wait_start_ts");
        if (col_wait_end_ts) out.print(",wait_end_ts");
        out.print("\n");

        for (int i = 0; i < runs; i++)
        {
            for (int idx = 0; idx < numMyTasks; idx++)
            {
                int task = my_tasks_sorted[idx];
                out.print("%d,%d,%d,%s", myPE, i, task,
                          tiledCholeskyGraphCreator->subgraphOpNames[task].c_str());
                if (col_wait_ms) out.print(",%.4f", all_timings[i][idx].wait_ms);
                if (col_compute_ms) out.print(",%.4f", all_timings[i][idx].compute_ms);
                if (col_start_ts) out.print(",%lld", (long long)all_timings[i][idx].start_ns);
                if (col_end_ts) out.print(",%lld", (long long)all_timings[i][idx].end_ns);
                if (col_wait_start_ts)
                    out.print(",%lld", (long long)all_timings[i][idx].wait_start_ns);
                if (col_wait_end_ts) out.print(",%lld", (long long)all_timings[i][idx].wait_end_ns);
                out.print("\n");
            }
        }
        out.flush();
        // Events are destroyed by InjectionContext destructor
    }

    if (verify)
    {
        double* h_L = (double*)malloc(N * N * sizeof(double));
        checkCudaErrors(cudaMemcpy(h_L, d_matrix, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        cleanCusolverCholeskyDecompositionResult(h_L, N);
        printf("Result passes verification: %d\n",
               verifyCholeskyDecomposition(originalMatrix.get(), h_L, N, verbose));
        free(h_L);
    }

    // Cleanup (scheduler destructor frees d_task_deps and d_task_notify_pes)
    delete[] h_subgraphsExec;
    nvshmem_free(d_completion_flags);
    nvshmem_free(d_matrices);
    nvshmem_free((void*)d_flags);
    checkCudaErrors(cudaFree(d_info));
    checkCudaErrors(cudaFree(d_workspace_cusolver));
    for (int i = 0; i < workspaces; i++) checkCudaErrors(cudaFree(d_workspace_cublas[i]));
    free(d_workspace_cublas);
}

void tiledCholeskyStaticOneGraph(bool verify, bool dot)
{
    auto setup_start = std::chrono::high_resolution_clock::now();

    int nPEs = nvshmem_n_pes();

    // Initialize data
    auto originalMatrix = std::make_unique<double[]>(N * N);  // Column-major
    generateRandomSymmetricPositiveDefiniteMatrix(originalMatrix.get(), N);

    // NVSHMEM allocations (all PEs participate)
    volatile int* d_flags         = (volatile int*)nvshmem_malloc(sizeof(int) * 32);
    double*       d_matrices      = (double*)nvshmem_malloc(N * N * sizeof(double));
    double*       d_matrix        = (double*)nvshmem_ptr(d_matrices, myPE);
    double*       d_matrix_remote = nullptr;
    checkCudaErrors(
        cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    if (myPE != 0) d_matrix_remote = (double*)nvshmem_ptr(d_matrices, 0);

    auto getMatrixBlock = [&](double* matrix, int i, int j) { return matrix + i * B + j * B * N; };

    // Initialize libraries
    cusolverDnHandle_t cusolverDnHandle;
    cusolverDnParams_t cusolverDnParams;
    cublasHandle_t     cublasHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
    checkCudaErrors(cublasCreate(&cublasHandle));

    double one      = 1.0;
    double minusOne = -1.0;

    int workspaceInBytesOnDevice;
    checkCudaErrors(cusolverDnDpotrf_bufferSize(cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, B,
                                                d_matrix, N, &workspaceInBytesOnDevice));

    double* d_workspace_cusolver;
    int     workspaces         = T * T;
    void**  d_workspace_cublas = (void**)malloc(sizeof(void*) * workspaces);
    int*    d_info;
    workspaceInBytesOnDevice *= 8;
    checkCudaErrors(cudaMalloc(&d_workspace_cusolver, workspaceInBytesOnDevice));
    int cublasWorkspaceSize = 1024 * workspace;
    for (int i = 0; i < workspaces; i++)
        checkCudaErrors(cudaMalloc(&d_workspace_cublas[i], cublasWorkspaceSize));
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));

    cudaGraph_t graph;
    checkCudaErrors(cudaGraphCreate(&graph, 0));

    int totalNodes = T;
    for (int k = 0; k < T; k++)
        for (int i = k + 1; i < T; i++) totalNodes += 2 + (T - (i + 1));

    if (verbose)
    {
        std::cout << "totalNodes=" << totalNodes << std::endl;
        std::cout << "bufferSize=" << workspaceInBytesOnDevice << std::endl;
        std::cout << "tileSize=" << cublasWorkspaceSize << std::endl;
    }
    printf("device %d | tiledCholeskyStaticOneGraph: building %d graphs\n", myPE, totalNodes);
    fflush(stdout);

    cudaStream_t s;
    checkCudaErrors(cudaStreamCreate(&s));
    checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
    checkCudaErrors(cublasSetStream(cublasHandle, s));
    checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));

    auto tiledCholeskyGraphCreator =
        std::make_unique<mustard::TiledGraphCreator>(s, graph, true, totalNodes);

    // Graph construction — verbatim copy of the subgraph path in tiledCholesky
    for (int k = 0; k < T; k++)
    {
        checkCudaErrors(
            cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));
        tiledCholeskyGraphCreator->beginCaptureOperation(
            std::make_pair(k, k), {std::make_pair(k, k)},
            "POTRF(" + std::to_string(k) + "," + std::to_string(k) + ")");
        mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
        if (myPE != 0)
            cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k), sizeof(double) * N,
                              getMatrixBlock(d_matrix_remote, k, k), sizeof(double) * N,
                              sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
        checkCudaErrors(cusolverDnDpotrf(cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, B,
                                         getMatrixBlock(d_matrix, k, k), N, d_workspace_cusolver,
                                         workspaceInBytesOnDevice, d_info));
        if (myPE != 0)
            cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, k, k), sizeof(double) * N,
                              getMatrixBlock(d_matrix, k, k), sizeof(double) * N,
                              sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
        mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
        tiledCholeskyGraphCreator->endCaptureOperation();

        for (int i = k + 1; i < T; i++)
        {
            checkCudaErrors(
                cublasSetWorkspace(cublasHandle, d_workspace_cublas[i], cublasWorkspaceSize));
            tiledCholeskyGraphCreator->beginCaptureOperation(
                std::make_pair(i, k), {std::make_pair(k, k), std::make_pair(i, k)},
                "TRSM(" + std::to_string(i) + "," + std::to_string(k) + ")");
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
            if (myPE != 0 && k != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), sizeof(double) * N,
                                  getMatrixBlock(d_matrix_remote, i, k), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k), sizeof(double) * N,
                                  getMatrixBlock(d_matrix_remote, k, k), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            checkCudaErrors(cublasDtrsm(cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                        CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, B, B, &one,
                                        getMatrixBlock(d_matrix, k, k), N,
                                        getMatrixBlock(d_matrix, i, k), N));
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, k), sizeof(double) * N,
                                  getMatrixBlock(d_matrix, i, k), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            tiledCholeskyGraphCreator->endCaptureOperation();
        }

        for (int i = k + 1; i < T; i++)
        {
            checkCudaErrors(
                cublasSetWorkspace(cublasHandle, d_workspace_cublas[i + T], cublasWorkspaceSize));
            tiledCholeskyGraphCreator->beginCaptureOperation(
                std::make_pair(i, i), {std::make_pair(i, i), std::make_pair(i, k)},
                "SYRK(" + std::to_string(i) + "," + std::to_string(i) + "," + std::to_string(k) +
                    ")");
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), sizeof(double) * N,
                                  getMatrixBlock(d_matrix_remote, i, k), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, i), sizeof(double) * N,
                                  getMatrixBlock(d_matrix_remote, i, i), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            checkCudaErrors(cublasDsyrk(cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, B, B,
                                        &minusOne, getMatrixBlock(d_matrix, i, k), N, &one,
                                        getMatrixBlock(d_matrix, i, i), N));
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, i), sizeof(double) * N,
                                  getMatrixBlock(d_matrix, i, i), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            tiledCholeskyGraphCreator->endCaptureOperation();

            for (int j = i + 1; j < T; j++)
            {
                checkCudaErrors(cublasSetWorkspace(
                    cublasHandle, d_workspace_cublas[2 * T + (i - 1) * T + (j - 1)],
                    cublasWorkspaceSize));
                tiledCholeskyGraphCreator->beginCaptureOperation(
                    std::make_pair(j, i),
                    {std::make_pair(j, i), std::make_pair(j, k), std::make_pair(i, k)},
                    "GEMM(" + std::to_string(j) + "," + std::to_string(i) + "," +
                        std::to_string(k) + ")");
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
                if (myPE != 0)
                {
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, i, k), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, j, k), sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, j, k), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, j, i), sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, j, i), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                }
                checkCudaErrors(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, B, B, B,
                                             &minusOne, getMatrixBlock(d_matrix, j, k), CUDA_R_64F,
                                             N, getMatrixBlock(d_matrix, i, k), CUDA_R_64F, N, &one,
                                             getMatrixBlock(d_matrix, j, i), CUDA_R_64F, N,
                                             CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
                if (myPE != 0)
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, j, i), sizeof(double) * N,
                                      getMatrixBlock(d_matrix, j, i), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
                tiledCholeskyGraphCreator->endCaptureOperation();
            }
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    printf("device %d | tiledCholeskyStaticOneGraph: graph construction done\n", myPE);
    fflush(stdout);

    // Parse measure flags — each flag corresponds to exactly one output column.
    // _ms  = CUDA event elapsed duration in milliseconds
    // _ts  = absolute Unix nanosecond timestamp (wall clock)
    auto has_flag    = [&](const char* f) { return cfg.measureFlags.find(f) != std::string::npos; };
    bool col_wait_ms = has_flag("wait_ms");
    bool col_compute_ms    = has_flag("compute_ms");
    bool col_start_ts      = has_flag("start_ts");
    bool col_end_ts        = has_flag("end_ts");
    bool col_wait_start_ts = has_flag("wait_start_ts");
    bool col_wait_end_ts   = has_flag("wait_end_ts");

    // NVSHMEM completion flags
    int* d_completion_flags = (int*)nvshmem_malloc(sizeof(int) * totalNodes);
    checkCudaErrors(cudaMemset(d_completion_flags, 0, sizeof(int) * totalNodes));

    auto scheduler = std::make_unique<mustard::StaticRoundRobinScheduler>(
        nPEs, myPE, totalNodes, tiledCholeskyGraphCreator->subgraphDependencies);

    mustard::TaskAllocator alloc;
    for (int task : scheduler->getMyTasksOrdered())
    {
        const auto& d = scheduler->getDeps(task);
        scheduler->setTaskDeps(task, alloc.allocate(d), (int)d.size());
        const auto& n = scheduler->getNotifyPEs(task);
        scheduler->setTaskNotifyPEs(task, alloc.allocate(n), (int)n.size());
    }

    const std::vector<int>& my_tasks_sorted = scheduler->getMyTasksOrdered();

    // Build injector chain based on measurement flags
    mustard::InjectionContext ctx(totalNodes);
    {
        auto injector = std::unique_ptr<mustard::IInjector>(
            new mustard::SubgraphInjector(tiledCholeskyGraphCreator->subgraphs, *scheduler,
                                          d_completion_flags, cfg.debugKernels));
        if (col_wait_start_ts || col_wait_end_ts)
            injector = std::make_unique<mustard::WaitTimestampDecorator>(
                std::move(injector), tiledCholeskyGraphCreator->subgraphs);
        if (col_wait_ms || col_compute_ms)
            injector = std::make_unique<mustard::WaitTimeDecorator>(
                std::move(injector), tiledCholeskyGraphCreator->subgraphs);
        if (col_compute_ms)
            injector = std::make_unique<mustard::ComputeTimeDecorator>(
                std::move(injector), tiledCholeskyGraphCreator->subgraphs);
        if (col_start_ts || col_end_ts)
            injector = std::make_unique<mustard::TimestampDecorator>(
                std::move(injector), tiledCholeskyGraphCreator->subgraphs);
        injector->inject(my_tasks_sorted, ctx);
    }

    // Assemble one flat combined graph per PE: each task becomes a child graph node,
    // with explicit dependency edges for same-PE task dependencies.
    cudaGraph_t pe_graph;
    checkCudaErrors(cudaGraphCreate(&pe_graph, 0));
    std::vector<cudaGraphNode_t> task_node(totalNodes, nullptr);
    for (int task : my_tasks_sorted)
    {
        std::vector<cudaGraphNode_t> dep_nodes;
        for (int dep : scheduler->getDeps(task))
            if (scheduler->getTaskPE(dep) == myPE) dep_nodes.push_back(task_node[dep]);
        checkCudaErrors(cudaGraphAddChildGraphNode(&task_node[task], pe_graph, dep_nodes.data(),
                                                   dep_nodes.size(),
                                                   tiledCholeskyGraphCreator->subgraphs[task]));
    }

    if (dot)
    {
        char filename[64];
        sprintf(filename, "./pe_graph_%d.dot", myPE);
        checkCudaErrors(cudaGraphDebugDotPrint(pe_graph, filename, 0));
    }

    cudaGraphExec_t pe_graph_exec;
    checkCudaErrors(cudaGraphInstantiate(&pe_graph_exec, pe_graph, nullptr, nullptr, 0));
    checkCudaErrors(cudaGraphUpload(pe_graph_exec, s));
    checkCudaErrors(cudaDeviceSynchronize());
    printf("device %d | tiledCholeskyStaticOneGraph: PE graph instantiated, entering timing loop\n",
           myPE);
    fflush(stdout);

    if (!cfg.invocationPath.empty())
        tiledCholeskyGraphCreator->printInvocations(cfg.invocationPath, myPE);

    auto   setup_end  = std::chrono::high_resolution_clock::now();
    double setup_time = std::chrono::duration<double>(setup_end - setup_start).count();
    printf("device %d | Setup time (s): %4.4f\n", myPE, setup_time);
    fflush(stdout);

    struct TaskTiming
    {
        float              wait_ms       = 0.0f;
        float              compute_ms    = 0.0f;
        unsigned long long start_ns      = 0;
        unsigned long long end_ns        = 0;
        unsigned long long wait_start_ns = 0;
        unsigned long long wait_end_ns   = 0;
    };
    int                                  numMyTasks = (int)my_tasks_sorted.size();
    std::vector<std::vector<TaskTiming>> all_timings(runs, std::vector<TaskTiming>(numMyTasks));
    std::vector<unsigned long long>      h_timestamps(totalNodes * 2);
    std::vector<unsigned long long>      h_wait_timestamps(totalNodes * 2);

    double totalTime = 0.0;

    for (int i = 0; i < runs; i++)
    {
        checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double),
                                   cudaMemcpyHostToDevice));

        // Reset completion flags on all PEs
        nvshmem_barrier_all();
        if (myPE == 0)
        {
            std::vector<int> zeros(totalNodes, 0);
            for (int pe = 0; pe < nPEs; pe++)
                nvshmem_int_put(d_completion_flags, zeros.data(), totalNodes, pe);
            nvshmem_quiet();
        }
        nvshmem_barrier_all();

        cudaEvent_t ev_ref = nullptr;
        if (col_wait_ms)
        {
            checkCudaErrors(cudaEventCreate(&ev_ref));
            checkCudaErrors(cudaEventRecord(ev_ref, s));
        }

        gpu_clock::CalibrationRef ts_ref;
        if (col_start_ts || col_end_ts || col_wait_start_ts || col_wait_end_ts)
            ts_ref = gpu_clock::calibrate(s);

        if (myPE == 0) print_timestamp("cholesky tiledStaticOneGraph start_time", 7);
        auto t_start = std::chrono::high_resolution_clock::now();
        checkCudaErrors(cudaGraphLaunch(pe_graph_exec, s));
        checkCudaErrors(cudaStreamSynchronize(s));
        checkCudaErrors(cudaDeviceSynchronize());
        auto t_end = std::chrono::high_resolution_clock::now();
        if (myPE == 0) print_timestamp("cholesky tiledStaticOneGraph end_time", 7);

        if (col_wait_ms || col_compute_ms)
        {
            for (int idx = 0; idx < numMyTasks; idx++)
            {
                int         task = my_tasks_sorted[idx];
                TaskTiming& tt   = all_timings[i][idx];
                if (col_wait_ms)
                {
                    if (ctx.task_wait_node[task] != nullptr)
                        checkCudaErrors(
                            cudaEventElapsedTime(&tt.wait_ms, ev_ref, ctx.compute_start[task]));
                    else
                        tt.wait_ms = 0.0f;
                }
                if (col_compute_ms)
                    checkCudaErrors(cudaEventElapsedTime(&tt.compute_ms, ctx.compute_start[task],
                                                         ctx.compute_end[task]));
            }
        }
        if (col_start_ts || col_end_ts)
        {
            checkCudaErrors(cudaMemcpy(h_timestamps.data(), ctx.d_timestamps,
                                       sizeof(unsigned long long) * totalNodes * 2,
                                       cudaMemcpyDeviceToHost));
            for (int idx = 0; idx < numMyTasks; idx++)
            {
                int         task = my_tasks_sorted[idx];
                TaskTiming& tt   = all_timings[i][idx];
                if (col_start_ts)
                    tt.start_ns =
                        gpu_clock::globaltimer_to_unix_ns(h_timestamps[task * 2 + 0], ts_ref);
                if (col_end_ts)
                    tt.end_ns =
                        gpu_clock::globaltimer_to_unix_ns(h_timestamps[task * 2 + 1], ts_ref);
            }
        }
        if ((col_wait_start_ts || col_wait_end_ts) && ctx.d_wait_timestamps)
        {
            checkCudaErrors(cudaMemcpy(h_wait_timestamps.data(), ctx.d_wait_timestamps,
                                       sizeof(unsigned long long) * totalNodes * 2,
                                       cudaMemcpyDeviceToHost));
            for (int idx = 0; idx < numMyTasks; idx++)
            {
                int         task = my_tasks_sorted[idx];
                TaskTiming& tt   = all_timings[i][idx];
                if (col_wait_start_ts && h_wait_timestamps[task * 2 + 0] != 0)
                    tt.wait_start_ns =
                        gpu_clock::globaltimer_to_unix_ns(h_wait_timestamps[task * 2 + 0], ts_ref);
                if (col_wait_end_ts && h_wait_timestamps[task * 2 + 1] != 0)
                    tt.wait_end_ns =
                        gpu_clock::globaltimer_to_unix_ns(h_wait_timestamps[task * 2 + 1], ts_ref);
            }
        }

        if (ev_ref) checkCudaErrors(cudaEventDestroy(ev_ref));
        nvshmem_barrier_all();

        double time = std::chrono::duration<double>(t_end - t_start).count();
        printf("device %d | %d run | time (s): %4.4f\n", myPE, i, time);
        totalTime += time;
    }
    printf("Total time used (s): %4.4f\n", totalTime);

    if (col_wait_ms || col_compute_ms || col_start_ts || col_end_ts || col_wait_start_ts ||
        col_wait_end_ts)
    {
        PEWriter out(cfg.outputPrefix, myPE);

        out.print("pe,run,task_id,op_name");
        if (col_wait_ms) out.print(",wait_ms");
        if (col_compute_ms) out.print(",compute_ms");
        if (col_start_ts) out.print(",start_ts");
        if (col_end_ts) out.print(",end_ts");
        if (col_wait_start_ts) out.print(",wait_start_ts");
        if (col_wait_end_ts) out.print(",wait_end_ts");
        out.print("\n");

        for (int i = 0; i < runs; i++)
        {
            for (int idx = 0; idx < numMyTasks; idx++)
            {
                int task = my_tasks_sorted[idx];
                out.print("%d,%d,%d,%s", myPE, i, task,
                          tiledCholeskyGraphCreator->subgraphOpNames[task].c_str());
                if (col_wait_ms) out.print(",%.4f", all_timings[i][idx].wait_ms);
                if (col_compute_ms) out.print(",%.4f", all_timings[i][idx].compute_ms);
                if (col_start_ts) out.print(",%lld", (long long)all_timings[i][idx].start_ns);
                if (col_end_ts) out.print(",%lld", (long long)all_timings[i][idx].end_ns);
                if (col_wait_start_ts)
                    out.print(",%lld", (long long)all_timings[i][idx].wait_start_ns);
                if (col_wait_end_ts) out.print(",%lld", (long long)all_timings[i][idx].wait_end_ns);
                out.print("\n");
            }
        }
        out.flush();
        // Events are destroyed by InjectionContext destructor
    }

    if (verify)
    {
        double* h_L = (double*)malloc(N * N * sizeof(double));
        checkCudaErrors(cudaMemcpy(h_L, d_matrix, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        cleanCusolverCholeskyDecompositionResult(h_L, N);
        printf("Result passes verification: %d\n",
               verifyCholeskyDecomposition(originalMatrix.get(), h_L, N, verbose));
        free(h_L);
    }

    // Cleanup
    checkCudaErrors(cudaGraphExecDestroy(pe_graph_exec));
    checkCudaErrors(cudaGraphDestroy(pe_graph));
    nvshmem_free(d_completion_flags);
    nvshmem_free(d_matrices);
    nvshmem_free((void*)d_flags);
    checkCudaErrors(cudaFree(d_info));
    checkCudaErrors(cudaFree(d_workspace_cusolver));
    for (int i = 0; i < workspaces; i++) checkCudaErrors(cudaFree(d_workspace_cublas[i]));
    free(d_workspace_cublas);
}

void Cholesky(bool tiled, bool verify, bool subgraph, bool staticMultiGPU, bool oneGraphPerPE,
              bool dot)
{
    if (oneGraphPerPE)
        tiledCholeskyStaticOneGraph(verify, dot);
    else if (staticMultiGPU)
        tiledCholeskyStatic(verify, dot);
    else if (tiled && myPE == 0)
        tiledCholesky(verify, subgraph, dot);
    else if (subgraph)
        tiledCholesky(verify, subgraph, dot);
    else if (myPE == 0)
        trivialCholesky(verify);
}

int main(int argc, char** argv)
{
    auto wall_start    = std::chrono::system_clock::now();
    auto program_start = std::chrono::high_resolution_clock::now();

    auto cmdl = argh::parser(argc, argv);

    if (!parseCommonArgs(cmdl, cfg))
    {
        printSingleNodeUsage(argv[0], "Cholesky");
        return 1;
    }

    auto init_start = std::chrono::high_resolution_clock::now();
    initNvshmemDevice(cmdl, cfg);
    auto init_end = std::chrono::high_resolution_clock::now();

    myPE = cfg.myPE;
    if (myPE == 0) print_timestamp("Program start timestamp", wall_start);
    double init_time = std::chrono::duration<double>(init_end - init_start).count();
    printf("device %d | NVSHMEM init time (s): %4.4f\n", myPE, init_time);
    fflush(stdout);

    if (!(cmdl["tiled"] || cmdl["subgraph"] || cmdl["static-multigpu"] || cmdl["one-graph-per-pe"]))
        T = 1;
    B = N / T;

    if (myPE == 0)
    {
        if (cmdl["tiled"])
            std::cout << "TILED";
        else if (cmdl["subgraph"])
            std::cout << "SUBGRAPH";
        else if (cmdl["static-multigpu"])
            std::cout << "STATIC-MULTIGPU";
        else
            std::cout << "Single-kernel";
        std::cout << " with N=" << N << " (" << T << " of " << B << "x" << B << " tiles)"
                  << std::endl;

        if (cmdl[{"subgraph", "tiled"}] || cmdl["static-multigpu"])
        {
            std::cout << "SM Limit per kernel = " << smLimit << std::endl;
            std::cout << "cuBLAS workspace = " << workspace << " kB" << std::endl;
        }
    }

    Cholesky(cmdl["tiled"], cmdl["verify"] && myPE == 0, cmdl["subgraph"], cmdl["static-multigpu"],
             cmdl["one-graph-per-pe"], cmdl["dot"]);

    nvshmem_finalize();

    auto   program_end  = std::chrono::high_resolution_clock::now();
    double program_time = std::chrono::duration<double>(program_end - program_start).count();
    printf("device %d | Total program time (s): %4.4f\n", myPE, program_time);
    if (myPE == 0) print_timestamp("Program end timestamp");

    return 0;
}