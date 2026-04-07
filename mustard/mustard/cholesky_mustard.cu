#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <utility>

#include "argh.h"
#include "cli.h"
#include "gen.h"
#include "mustard.h"
#include "verify.h"

// Global configuration (populated from CLI in main).
static MustardConfig cfg;
static size_t& N = cfg.N;
static size_t& B = cfg.B;
static size_t& T = cfg.T;
int myPE;
static int& verbose = cfg.verbose;
static int& workspace = cfg.workspace;
static int& smLimit = cfg.smLimit;
static int& runs = cfg.runs;

void trivialCholesky(bool verify)
{
    // Initialize libaries
    cusolverDnHandle_t cusolverDnHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));

    cusolverDnParams_t cusolverDnParams;
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));

    // Initialize data
    double *h_A = (double *)malloc(N * N * sizeof(double));
    generateRandomSymmetricPositiveDefiniteMatrix(h_A, N);

    double *d_A;
    checkCudaErrors(cudaMalloc(&d_A, N * N * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));

    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;

    checkCudaErrors(cusolverDnXpotrf_bufferSize(
        cusolverDnHandle,
        cusolverDnParams,
        CUBLAS_FILL_MODE_LOWER,
        N,
        CUDA_R_64F,
        d_A,
        N,
        CUDA_R_64F,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));

    void *h_workspace = malloc(workspaceInBytesOnHost);

    void *d_workspace;
    checkCudaErrors(cudaMalloc(&d_workspace, workspaceInBytesOnDevice));

    int *d_info;
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));

    CudaEventClock clock;

    clock.start();
    double totalTime = 0.0;

    // Calculate
    for (int i = 0; i < runs; i++) {
        checkCudaErrors(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
        clock.start();
        checkCudaErrors(cusolverDnXpotrf(
            cusolverDnHandle,
            cusolverDnParams,
            CUBLAS_FILL_MODE_LOWER,
            N,
            CUDA_R_64F,
            d_A,
            N,
            CUDA_R_64F,
            d_workspace,
            workspaceInBytesOnDevice,
            h_workspace,
            workspaceInBytesOnHost,
            d_info));
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
        double *h_L = (double *)malloc(N * N * sizeof(double));
        checkCudaErrors(cudaMemcpy(h_L, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        cleanCusolverCholeskyDecompositionResult(h_L, N);
        printf("Result passes verification: %d\n", verifyCholeskyDecomposition(h_A, h_L, N, verbose));
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
    auto originalMatrix = std::make_unique<double[]>(N * N); // Column-major
    generateRandomSymmetricPositiveDefiniteMatrix(originalMatrix.get(), N);

    // Copy to device
    double *d_matrix;
    double *d_matrices;
    double *d_matrix_remote;
    volatile int *d_flags;
    if (subgraph) {
        d_flags = (volatile int *) nvshmem_malloc(sizeof(int) * 32);
        d_matrices = (double *) nvshmem_malloc(N * N * sizeof(double));
        d_matrix = (double *) nvshmem_ptr(d_matrices, myPE);
    } else {
        checkCudaErrors(cudaMalloc(&d_matrix, N * N * sizeof(double)));
    }
    checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    if (myPE != 0) 
        d_matrix_remote = (double *) nvshmem_ptr(d_matrices, 0);

    auto getMatrixBlock = [&](double* matrix, int i, int j)
    {
        return matrix + i * B + j * B * N;
    };

    // Initialize libraries
    cusolverDnHandle_t cusolverDnHandle;
    cusolverDnParams_t cusolverDnParams;
    cublasHandle_t cublasHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
    checkCudaErrors(cublasCreate(&cublasHandle));
    // Prepare constants
    double one = 1.0;
    double minusOne = -1.0;

    // Prepare buffer for potrf
    int workspaceInBytesOnDevice;

    checkCudaErrors(cusolverDnDpotrf_bufferSize(
                    cusolverDnHandle,
                    CUBLAS_FILL_MODE_LOWER,
                    B,
                    d_matrix,
                    N,
                    &workspaceInBytesOnDevice));

    double *d_workspace_cusolver;
    int workspaces = T*T;
    void **d_workspace_cublas = (void **)malloc(sizeof(void *)*workspaces);
    int *d_info;
    workspaceInBytesOnDevice*=8;
    checkCudaErrors(cudaMalloc(&d_workspace_cusolver, workspaceInBytesOnDevice));
    int cublasWorkspaceSize = 1024*workspace;

    for (int i = 0; i < workspaces; i++) {
        checkCudaErrors(cudaMalloc(&d_workspace_cublas[i], cublasWorkspaceSize));
    }
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));

    cudaGraph_t graph;
    checkCudaErrors(cudaGraphCreate(&graph, 0));

    int totalNodes = T;
    
    for (int k = 0; k < T; k++) 
        for (int i = k + 1; i < T; i++) 
            totalNodes += 2 + (T-(i+1));

    if (verbose) {
        std::cout << "totalNodes=" << totalNodes << std::endl;
        std::cout << "bufferSize=" << workspaceInBytesOnDevice << std::endl;
        std::cout << "tileSize=" << cublasWorkspaceSize << std::endl;
    }

    cudaStream_t s;
    checkCudaErrors(cudaStreamCreate(&s));

    checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
    checkCudaErrors(cublasSetStream(cublasHandle, s));
    checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));

    auto tiledCholeskyGraphCreator = std::make_unique<mustard::TiledGraphCreator>(s, graph, subgraph, totalNodes);

    for (int k = 0; k < T; k++)
    {
        // A[k][k] = GETRF(A[k][k])
        // L[k][k]*U[k][k] = A[k][k]
        checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));
        tiledCholeskyGraphCreator->beginCaptureOperation(
            std::make_pair(k, k),
            {std::make_pair(k, k)},
            "POTRF(" + std::to_string(k) + "," + std::to_string(k) + ")");
        if (subgraph) {
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
            if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k), 
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix_remote, k, k), 
                                            sizeof(double) * N, 
                                            sizeof(double) * B, 
                                            B, cudaMemcpyDeviceToDevice, s);
        }
        checkCudaErrors(cusolverDnDpotrf(
            cusolverDnHandle,
            CUBLAS_FILL_MODE_LOWER,
            B,
            getMatrixBlock(d_matrix, k, k),
            N,
            d_workspace_cusolver,
            workspaceInBytesOnDevice,
            d_info));
        if (subgraph) {
            if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, k, k), 
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix, k, k), 
                                            sizeof(double) * N, 
                                            sizeof(double) * B, 
                                            B, cudaMemcpyDeviceToDevice, s);
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
        }
        tiledCholeskyGraphCreator->endCaptureOperation();

        for (int i = k + 1; i < T; i++)
        {
            // L[i][k] = TRSM(A[i][k], A[k][k]) // the U part of A[k][k]
            // seems like only these need a separate workspace
            checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[i], cublasWorkspaceSize));
            tiledCholeskyGraphCreator->beginCaptureOperation(
                std::make_pair(i, k),
                {std::make_pair(k, k), std::make_pair(i, k)},
                "TRSM(" + std::to_string(i) + "," + std::to_string(k) + ")");
            if (subgraph) {
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
                if (myPE != 0 && k != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), 
                                                sizeof(double) * N,
                                                getMatrixBlock(d_matrix_remote, i, k), 
                                                sizeof(double) * N, 
                                                sizeof(double) * B, 
                                                B, cudaMemcpyDeviceToDevice, s);
                if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k), 
                                                sizeof(double) * N,
                                                getMatrixBlock(d_matrix_remote, k, k), 
                                                sizeof(double) * N, 
                                                sizeof(double) * B, 
                                                B, cudaMemcpyDeviceToDevice, s);
            }
            checkCudaErrors(cublasDtrsm(
                cublasHandle,
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT,
                B, B,
                &one,
                getMatrixBlock(d_matrix, k, k), N, // k + k * N;
                getMatrixBlock(d_matrix, i, k), N)); // k + (i + B) * N;
            if (subgraph) {
                if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, k), 
                                                sizeof(double) * N,
                                                getMatrixBlock(d_matrix, i, k), 
                                                sizeof(double) * N, 
                                                sizeof(double) * B, 
                                                B, cudaMemcpyDeviceToDevice, s);
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            }
            tiledCholeskyGraphCreator->endCaptureOperation();

        }

        for (int i = k + 1; i < T; i++)
        {
            // U[k][i] = TRSM(A[k][k], A[k][i]) // the L part of A[k][k]
            checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[i + T], cublasWorkspaceSize));
            tiledCholeskyGraphCreator->beginCaptureOperation(
                std::make_pair(i, i),
                {std::make_pair(i, i), std::make_pair(i, k)},
                "SYRK(" + std::to_string(i) + "," + std::to_string(i) + "," + std::to_string(k) + ")");
            
            if (subgraph) {
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
                if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), 
                                                sizeof(double) * N,
                                                getMatrixBlock(d_matrix_remote, i, k), 
                                                sizeof(double) * N, 
                                                sizeof(double) * B, 
                                                B, cudaMemcpyDeviceToDevice, s);
                if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, i), 
                                                sizeof(double) * N,
                                                getMatrixBlock(d_matrix_remote, i, i), 
                                                sizeof(double) * N, 
                                                sizeof(double) * B, 
                                                B, cudaMemcpyDeviceToDevice, s);
            }
            checkCudaErrors(cublasDsyrk(
                cublasHandle,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N,
                B, B,
                &minusOne, getMatrixBlock(d_matrix, i, k), N,
                &one, getMatrixBlock(d_matrix, i, i), N));
            if (subgraph) {
                if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, i), 
                                                sizeof(double) * N,
                                                getMatrixBlock(d_matrix, i, i), 
                                                sizeof(double) * N, 
                                                sizeof(double) * B, 
                                                B, cudaMemcpyDeviceToDevice, s);
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            }
            tiledCholeskyGraphCreator->endCaptureOperation();

            for (int j = i + 1; j < T; j++)
            {
                // A[j][i] = GEMM(A[j][k], A[i][k])
                // A[j][i] = A[j][i] - L[j][k] * L[i][k]^T
                checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[2*T+ (i-1)*T + (j-1)], cublasWorkspaceSize));
                tiledCholeskyGraphCreator->beginCaptureOperation(
                    std::make_pair(j, i),
                    {std::make_pair(j, i), std::make_pair(j, k), std::make_pair(i, k)},
                    "GEMM(" + std::to_string(j) + "," + std::to_string(i) + "," + std::to_string(k) + ")");
                if (subgraph) {
                    mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
                    if (myPE != 0) {
                        cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), 
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix_remote, i, k), 
                                            sizeof(double) * N, 
                                            sizeof(double) * B, 
                                            B, cudaMemcpyDeviceToDevice, s);
                        cudaMemcpy2DAsync(getMatrixBlock(d_matrix, j, k), 
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix_remote, j, k), 
                                            sizeof(double) * N, 
                                            sizeof(double) * B, 
                                            B, cudaMemcpyDeviceToDevice, s);
                        cudaMemcpy2DAsync(getMatrixBlock(d_matrix, j, i), 
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix_remote, j, i), 
                                            sizeof(double) * N, 
                                            sizeof(double) * B, 
                                            B, cudaMemcpyDeviceToDevice, s);
                    }
                }
                checkCudaErrors(cublasGemmEx(
                    cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    B, B, B,
                    &minusOne,
                    getMatrixBlock(d_matrix, j, k), CUDA_R_64F, N,
                    getMatrixBlock(d_matrix, i, k), CUDA_R_64F, N, 
                    &one,
                    getMatrixBlock(d_matrix, j, i), CUDA_R_64F, N, 
                    CUBLAS_COMPUTE_64F,
                    CUBLAS_GEMM_DEFAULT));
                if (subgraph) {
                    if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, j, i), 
                                                    sizeof(double) * N,
                                                    getMatrixBlock(d_matrix, j, i), 
                                                    sizeof(double) * N, 
                                                    sizeof(double) * B, 
                                                    B, cudaMemcpyDeviceToDevice, s);
                    mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
                }
                tiledCholeskyGraphCreator->endCaptureOperation();
            }
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
        
    cudaGraphExec_t graphExec;
    CudaEventClock clock;
    double totalTime = 0.0;
    
    if (subgraph) {
        if (verbose)
            tiledCholeskyGraphCreator->printDeps();
        
        // volatile int *d_flags;
        int *h_dependencies; //, *d_dependencies;
        const int queue_size = totalNodes * 2;
        if (verbose) std::cout << "Creating queue..." << std::endl;
        BrokerWorkDistributor queue(queue_size);
        if (verbose) std::cout << "Allocating memory..." << std::endl;

        int *d_dependencies = (int *) nvshmem_malloc(sizeof(int) * totalNodes);
        checkCudaErrors(cudaMallocHost(&h_dependencies, sizeof(int) * totalNodes));
        if (verbose) std::cout << "Setting dependencies..." << std::endl;

        for (int i = 0; i < totalNodes; i++)
        {
            h_dependencies[i] = tiledCholeskyGraphCreator->subgraphDependencies[i].size();
        }
        if (verbose) std::cout << "Populating the queue..." << std::endl;

        checkCudaErrors(cudaMemcpy((void *)d_dependencies, (void *)h_dependencies, 
                                sizeof(int) * totalNodes, cudaMemcpyHostToDevice));
        if (myPE == 0)
            mustard::kernel_populate_queue<<<108, 1024>>>(queue, d_dependencies, totalNodes);
        checkCudaErrors(cudaDeviceSynchronize());
        if (verbose) std::cout << "Inserting dependency kernels..." << std::endl;

        for (int dst = 0; dst < totalNodes; dst++)
            for (int src_ind = 0; src_ind < h_dependencies[dst]; src_ind++) 
                tiledCholeskyGraphCreator->insertDependencyKernel(tiledCholeskyGraphCreator->subgraphDependencies[dst][src_ind], 
                                                            dst, queue, d_dependencies);
        if (verbose) showMemUsage();
        if (verbose) std::cout << "Uploading graphs..." << std::endl;

        if (!cfg.invocationPath.empty()) {
            tiledCholeskyGraphCreator->printInvocations(cfg.invocationPath, myPE);
        }

        cudaGraphExec_t *h_subgraphsExec = new cudaGraphExec_t[totalNodes];
        cudaGraphExec_t *d_subgraphsExec;
        for (int i = 0; i < totalNodes; i++)
        {
            char filename[20];
            sprintf(filename, "./graph_%d.dot", i);
            if (dot)
                checkCudaErrors(cudaGraphDebugDotPrint(tiledCholeskyGraphCreator->subgraphs[i], filename, 0));
            checkCudaErrors(cudaGraphInstantiate(&h_subgraphsExec[i], tiledCholeskyGraphCreator->subgraphs[i], cudaGraphInstantiateFlagDeviceLaunch));
            cudaGraphUpload(h_subgraphsExec[i], s);
        }
        checkCudaErrors(cudaMalloc(&d_subgraphsExec, sizeof(cudaGraphExec_t) * totalNodes));
        checkCudaErrors(cudaMemcpy((void *)d_subgraphsExec, (void *)h_subgraphsExec,
                                    sizeof(cudaGraphExec_t) * totalNodes, cudaMemcpyHostToDevice));

        if (verbose) std::cout << "Initializing scheduler..." << std::endl;
        cudaGraph_t schedulerGraph;
        cudaGraphExec_t schedulerExec;
        checkCudaErrors(cudaGraphCreate(&schedulerGraph, 0));
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        mustard::kernel_scheduler<<<1, 1, 0, s>>>(queue, d_flags, d_subgraphsExec, totalNodes, myPE);
        cudaStreamEndCapture(s, &schedulerGraph);
        checkCudaErrors(cudaGraphInstantiate(&schedulerExec, schedulerGraph, cudaGraphInstantiateFlagDeviceLaunch));
        checkCudaErrors(cudaDeviceSynchronize());
        if (verbose) showMemUsage();
        if (verbose) std::cout << "Launching..." << std::endl;

        auto setup_end = std::chrono::high_resolution_clock::now();
        double setup_time = std::chrono::duration<double>(setup_end - setup_start).count();
        printf("device %d | Setup time (s): %4.4f\n", myPE, setup_time);

        for (int i = 0; i < runs; i++) {
            checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double), cudaMemcpyHostToDevice));
            nvshmem_barrier_all();
            clock.start(s);
            checkCudaErrors(cudaGraphLaunch(schedulerExec, s));
            checkCudaErrors(cudaStreamSynchronize(s));
            clock.end(s);
            checkCudaErrors(cudaDeviceSynchronize());
            nvshmem_barrier_all();
            if (myPE == 0) {
                checkCudaErrors(cudaMemset((void *)d_flags, 0, sizeof(int) * 32));
                checkCudaErrors(cudaMemcpy((void *)d_dependencies, (void *)h_dependencies, 
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
    } else {
        if (dot)
            checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", 0));
        checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

        auto setup_end = std::chrono::high_resolution_clock::now();
        double setup_time = std::chrono::duration<double>(setup_end - setup_start).count();
        printf("device %d | Setup time (s): %4.4f\n", myPE, setup_time);

        for (int i = 0; i < runs; i++) {
            checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double), cudaMemcpyHostToDevice));
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

    if (verify) {
        double *h_L = (double *)malloc(N * N * sizeof(double));
        checkCudaErrors(cudaMemcpy(h_L, d_matrix, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        cleanCusolverCholeskyDecompositionResult(h_L, N);
        printf("Result passes verification: %d\n", verifyCholeskyDecomposition(originalMatrix.get(), h_L, N, verbose));

        free(h_L);
    }
    printf("Total time used (s): %4.4f\n", totalTime);

    if (!subgraph) checkCudaErrors(cudaFree(d_matrix));
    else nvshmem_free(d_matrices);
    checkCudaErrors(cudaFree(d_info));
    checkCudaErrors(cudaFree(d_workspace_cusolver));
    for (int i = 0; i < workspaces; i++) {
        checkCudaErrors(cudaFree(d_workspace_cublas[i]));
    }
}

void tiledCholeskyStatic(bool verify, bool dot)
{
    auto setup_start = std::chrono::high_resolution_clock::now();

    int nPEs = nvshmem_n_pes();

    // Initialize data
    auto originalMatrix = std::make_unique<double[]>(N * N); // Column-major
    generateRandomSymmetricPositiveDefiniteMatrix(originalMatrix.get(), N);

    // NVSHMEM allocations (all PEs participate)
    volatile int *d_flags = (volatile int *) nvshmem_malloc(sizeof(int) * 32);
    double *d_matrices    = (double *) nvshmem_malloc(N * N * sizeof(double));
    double *d_matrix      = (double *) nvshmem_ptr(d_matrices, myPE);
    double *d_matrix_remote = nullptr;
    checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    if (myPE != 0)
        d_matrix_remote = (double *) nvshmem_ptr(d_matrices, 0);

    auto getMatrixBlock = [&](double* matrix, int i, int j) {
        return matrix + i * B + j * B * N;
    };

    // Initialize libraries
    cusolverDnHandle_t cusolverDnHandle;
    cusolverDnParams_t cusolverDnParams;
    cublasHandle_t cublasHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
    checkCudaErrors(cublasCreate(&cublasHandle));

    double one = 1.0;
    double minusOne = -1.0;

    int workspaceInBytesOnDevice;
    checkCudaErrors(cusolverDnDpotrf_bufferSize(cusolverDnHandle,
                                                CUBLAS_FILL_MODE_LOWER,
                                                B, d_matrix, N,
                                                &workspaceInBytesOnDevice));

    double *d_workspace_cusolver;
    int workspaces = T * T;
    void **d_workspace_cublas = (void **)malloc(sizeof(void *) * workspaces);
    int *d_info;
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
        for (int i = k + 1; i < T; i++)
            totalNodes += 2 + (T-(i+1));

    if (verbose) {
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

    auto tiledCholeskyGraphCreator = std::make_unique<mustard::TiledGraphCreator>(s, graph, true, totalNodes);

    // Graph construction — verbatim copy of the subgraph path in tiledCholesky
    for (int k = 0; k < T; k++)
    {
        checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));
        tiledCholeskyGraphCreator->beginCaptureOperation(
            std::make_pair(k, k),
            {std::make_pair(k, k)},
            "POTRF(" + std::to_string(k) + "," + std::to_string(k) + ")");
        mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
        if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k),
                                         sizeof(double) * N,
                                         getMatrixBlock(d_matrix_remote, k, k),
                                         sizeof(double) * N,
                                         sizeof(double) * B,
                                         B, cudaMemcpyDeviceToDevice, s);
        checkCudaErrors(cusolverDnDpotrf(
            cusolverDnHandle,
            CUBLAS_FILL_MODE_LOWER,
            B,
            getMatrixBlock(d_matrix, k, k),
            N,
            d_workspace_cusolver,
            workspaceInBytesOnDevice,
            d_info));
        if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, k, k),
                                         sizeof(double) * N,
                                         getMatrixBlock(d_matrix, k, k),
                                         sizeof(double) * N,
                                         sizeof(double) * B,
                                         B, cudaMemcpyDeviceToDevice, s);
        mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
        tiledCholeskyGraphCreator->endCaptureOperation();

        for (int i = k + 1; i < T; i++)
        {
            checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[i], cublasWorkspaceSize));
            tiledCholeskyGraphCreator->beginCaptureOperation(
                std::make_pair(i, k),
                {std::make_pair(k, k), std::make_pair(i, k)},
                "TRSM(" + std::to_string(i) + "," + std::to_string(k) + ")");
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
            if (myPE != 0 && k != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k),
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix_remote, i, k),
                                            sizeof(double) * N,
                                            sizeof(double) * B,
                                            B, cudaMemcpyDeviceToDevice, s);
            if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k),
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix_remote, k, k),
                                            sizeof(double) * N,
                                            sizeof(double) * B,
                                            B, cudaMemcpyDeviceToDevice, s);
            checkCudaErrors(cublasDtrsm(
                cublasHandle,
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT,
                B, B,
                &one,
                getMatrixBlock(d_matrix, k, k), N,
                getMatrixBlock(d_matrix, i, k), N));
            if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, k),
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix, i, k),
                                            sizeof(double) * N,
                                            sizeof(double) * B,
                                            B, cudaMemcpyDeviceToDevice, s);
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            tiledCholeskyGraphCreator->endCaptureOperation();
        }

        for (int i = k + 1; i < T; i++)
        {
            checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[i + T], cublasWorkspaceSize));
            tiledCholeskyGraphCreator->beginCaptureOperation(
                std::make_pair(i, i),
                {std::make_pair(i, i), std::make_pair(i, k)},
                "SYRK(" + std::to_string(i) + "," + std::to_string(i) + "," + std::to_string(k) + ")");
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
            if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k),
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix_remote, i, k),
                                            sizeof(double) * N,
                                            sizeof(double) * B,
                                            B, cudaMemcpyDeviceToDevice, s);
            if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, i),
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix_remote, i, i),
                                            sizeof(double) * N,
                                            sizeof(double) * B,
                                            B, cudaMemcpyDeviceToDevice, s);
            checkCudaErrors(cublasDsyrk(
                cublasHandle,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N,
                B, B,
                &minusOne, getMatrixBlock(d_matrix, i, k), N,
                &one,      getMatrixBlock(d_matrix, i, i), N));
            if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, i),
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix, i, i),
                                            sizeof(double) * N,
                                            sizeof(double) * B,
                                            B, cudaMemcpyDeviceToDevice, s);
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            tiledCholeskyGraphCreator->endCaptureOperation();

            for (int j = i + 1; j < T; j++)
            {
                checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[2*T+ (i-1)*T + (j-1)], cublasWorkspaceSize));
                tiledCholeskyGraphCreator->beginCaptureOperation(
                    std::make_pair(j, i),
                    {std::make_pair(j, i), std::make_pair(j, k), std::make_pair(i, k)},
                    "GEMM(" + std::to_string(j) + "," + std::to_string(i) + "," + std::to_string(k) + ")");
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
                if (myPE != 0) {
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k),
                                      sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, i, k),
                                      sizeof(double) * N,
                                      sizeof(double) * B,
                                      B, cudaMemcpyDeviceToDevice, s);
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, j, k),
                                      sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, j, k),
                                      sizeof(double) * N,
                                      sizeof(double) * B,
                                      B, cudaMemcpyDeviceToDevice, s);
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, j, i),
                                      sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, j, i),
                                      sizeof(double) * N,
                                      sizeof(double) * B,
                                      B, cudaMemcpyDeviceToDevice, s);
                }
                checkCudaErrors(cublasGemmEx(
                    cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    B, B, B,
                    &minusOne,
                    getMatrixBlock(d_matrix, j, k), CUDA_R_64F, N,
                    getMatrixBlock(d_matrix, i, k), CUDA_R_64F, N,
                    &one,
                    getMatrixBlock(d_matrix, j, i), CUDA_R_64F, N,
                    CUBLAS_COMPUTE_64F,
                    CUBLAS_GEMM_DEFAULT));
                if (myPE != 0) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, j, i),
                                                  sizeof(double) * N,
                                                  getMatrixBlock(d_matrix, j, i),
                                                  sizeof(double) * N,
                                                  sizeof(double) * B,
                                                  B, cudaMemcpyDeviceToDevice, s);
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
                tiledCholeskyGraphCreator->endCaptureOperation();
            }
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    printf("device %d | tiledCholeskyStatic: graph construction done\n", myPE);
    fflush(stdout);

    // Static task assignment — round-robin
    std::vector<std::vector<int>> pe_tasks(nPEs);
    for (int i = 0; i < totalNodes; i++)
        pe_tasks[i % nPEs].push_back(i);

    std::vector<int> task_pe(totalNodes);
    for (int i = 0; i < totalNodes; i++)
        task_pe[i] = i % nPEs;

    // Topological sort of myPE's tasks respecting same-PE dependencies
    std::vector<int> my_tasks_sorted;
    std::set<int> topo_done;
    while (my_tasks_sorted.size() < pe_tasks[myPE].size()) {
        bool progress = false;
        for (int task : pe_tasks[myPE]) {
            if (topo_done.count(task)) continue;
            bool ready = true;
            for (int dep : tiledCholeskyGraphCreator->subgraphDependencies[task]) {
                if (task_pe[dep] == myPE && !topo_done.count(dep)) {
                    ready = false;
                    break;
                }
            }
            if (ready) {
                my_tasks_sorted.push_back(task);
                topo_done.insert(task);
                progress = true;
            }
        }
        if (!progress) break;
    }

    // NVSHMEM completion flags
    int *d_completion_flags = (int *) nvshmem_malloc(sizeof(int) * totalNodes);
    checkCudaErrors(cudaMemset(d_completion_flags, 0, sizeof(int) * totalNodes));

    // Reverse dependency map
    std::vector<std::vector<int>> dependents(totalNodes);
    for (int j = 0; j < totalNodes; j++)
        for (int dep : tiledCholeskyGraphCreator->subgraphDependencies[j])
            dependents[dep].push_back(j);

    // Debug: print dependency and notify info
    for (int task : pe_tasks[myPE]) {
        std::set<int> notify_set;
        for (int d : dependents[task]) notify_set.insert(task_pe[d]);
        printf("PE %d task %d: dependents=[", myPE, task);
        for (int d : dependents[task]) printf("%d,", d);
        printf("] notify_pes=[");
        for (int p : notify_set) printf("%d,", p);
        printf("]\n");
    }
    fflush(stdout);

    // Allocate per-task device arrays for deps and notify PEs
    std::vector<int*> d_task_deps(totalNodes, nullptr);
    std::vector<int*> d_task_notify_pes(totalNodes, nullptr);

    for (int task : pe_tasks[myPE]) {
        auto& deps = tiledCholeskyGraphCreator->subgraphDependencies[task];
        if (!deps.empty()) {
            checkCudaErrors(cudaMalloc(&d_task_deps[task], sizeof(int) * deps.size()));
            checkCudaErrors(cudaMemcpy(d_task_deps[task], deps.data(),
                                       sizeof(int) * deps.size(), cudaMemcpyHostToDevice));
        }

        std::set<int> notify_set;
        for (int dep_task : dependents[task])
            notify_set.insert(task_pe[dep_task]);
        std::vector<int> notify_vec(notify_set.begin(), notify_set.end());
        if (!notify_vec.empty()) {
            checkCudaErrors(cudaMalloc(&d_task_notify_pes[task], sizeof(int) * notify_vec.size()));
            checkCudaErrors(cudaMemcpy(d_task_notify_pes[task], notify_vec.data(),
                                       sizeof(int) * notify_vec.size(), cudaMemcpyHostToDevice));
        }
    }

    // Helper to find tail node of a subgraph
    auto getSubgraphTail = [](cudaGraph_t g) -> cudaGraphNode_t {
        size_t numEdges;
        MUSTARD_cudaGraphGetEdges(g, nullptr, nullptr, &numEdges);
        if (numEdges == 0) {
            size_t numNodes = 1;
            cudaGraphNode_t node;
            cudaGraphGetNodes(g, &node, &numNodes);
            return node;
        }
        std::vector<cudaGraphNode_t> from(numEdges), to(numEdges);
        MUSTARD_cudaGraphGetEdges(g, from.data(), to.data(), &numEdges);
        std::map<cudaGraphNode_t, bool> hasOutgoing;
        std::set<cudaGraphNode_t> noOutgoing;
        for (size_t e = 0; e < numEdges; e++) {
            hasOutgoing[from[e]] = true;
            noOutgoing.erase(from[e]);
            if (!hasOutgoing[to[e]]) noOutgoing.insert(to[e]);
        }
        if (noOutgoing.size() == 1) return *noOutgoing.begin();
        size_t numNodes = 0;
        cudaGraphGetNodes(g, nullptr, &numNodes);
        std::vector<cudaGraphNode_t> nodes(numNodes);
        cudaGraphGetNodes(g, nodes.data(), &numNodes);
        return nodes.back();
    };

    // Inject wait and signal kernels into owned subgraphs
    for (int task : pe_tasks[myPE]) {
        cudaGraph_t sg = tiledCholeskyGraphCreator->subgraphs[task];
        auto& deps = tiledCholeskyGraphCreator->subgraphDependencies[task];

        // Prepend wait kernel (only if task has dependencies)
        int debug = cfg.debugKernels;
        if (!deps.empty()) {
            size_t numRoots;
            cudaGraphGetRootNodes(sg, nullptr, &numRoots);
            std::vector<cudaGraphNode_t> roots(numRoots);
            cudaGraphGetRootNodes(sg, roots.data(), &numRoots);

            cudaGraphNode_t waitNode;
            cudaKernelNodeParams waitParams = {0};
            waitParams.gridDim  = dim3(1);
            waitParams.blockDim = dim3(1);
            waitParams.func = (void *)mustard::kernel_wait_static;
            int n_deps = (int)deps.size();
            void *waitArgs[4] = {&d_task_deps[task], &n_deps, &d_completion_flags, &debug};
            waitParams.kernelParams = waitArgs;
            checkCudaErrors(cudaGraphAddKernelNode(&waitNode, sg, nullptr, 0, &waitParams));

            for (auto& root : roots)
                MUSTARD_cudaGraphAddDependencies(sg, &waitNode, &root, 1);
        }

        // Append signal kernel
        std::set<int> notify_set;
        for (int dep_task : dependents[task])
            notify_set.insert(task_pe[dep_task]);
        int n_notify = (int)notify_set.size();

        cudaGraphNode_t tail = getSubgraphTail(sg);
        cudaGraphNode_t signalNode;
        cudaKernelNodeParams signalParams = {0};
        signalParams.gridDim  = dim3(1);
        signalParams.blockDim = dim3(1);
        signalParams.func = (void *)mustard::kernel_signal_static;
        int task_id_val = task;
        void *signalArgs[5] = {&task_id_val, &d_completion_flags,
                               &d_task_notify_pes[task], &n_notify, &debug};
        signalParams.kernelParams = signalArgs;
        checkCudaErrors(cudaGraphAddKernelNode(&signalNode, sg, &tail, 1, &signalParams));
    }

    // Instantiate and upload owned subgraphs
    cudaGraphExec_t *h_subgraphsExec = new cudaGraphExec_t[totalNodes];
    for (int task : pe_tasks[myPE]) {
        if (dot) {
            char filename[32];
            sprintf(filename, "./graph_%d_%d.dot", task, myPE);
            checkCudaErrors(cudaGraphDebugDotPrint(tiledCholeskyGraphCreator->subgraphs[task], filename, 0));
        }
        checkCudaErrors(cudaGraphInstantiate(&h_subgraphsExec[task],
                                             tiledCholeskyGraphCreator->subgraphs[task],
                                             nullptr, nullptr, 0));
        cudaGraphUpload(h_subgraphsExec[task], s);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    printf("device %d | tiledCholeskyStatic: graphs instantiated, entering timing loop\n", myPE);
    fflush(stdout);

    if (!cfg.invocationPath.empty())
        tiledCholeskyGraphCreator->printInvocations(cfg.invocationPath, myPE);

    auto setup_end = std::chrono::high_resolution_clock::now();
    double setup_time = std::chrono::duration<double>(setup_end - setup_start).count();
    printf("device %d | Setup time (s): %4.4f\n", myPE, setup_time);
    fflush(stdout);

    CudaEventClock clock;
    double totalTime = 0.0;

    for (int i = 0; i < runs; i++) {
    checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double),
                               cudaMemcpyHostToDevice));

    // Reset completion flags on all PEs
    nvshmem_barrier_all();
    if (myPE == 0) {
        std::vector<int> zeros(totalNodes, 0);
        for (int pe = 0; pe < nPEs; pe++)
            nvshmem_int_put(d_completion_flags, zeros.data(), totalNodes, pe);
        nvshmem_quiet();
    }
    nvshmem_barrier_all();

    printf("device %d | run %d: launching %zu tasks\n", myPE, i, my_tasks_sorted.size());
    fflush(stdout);
    int numStreams = std::min((int)my_tasks_sorted.size(), 32);
    std::vector<cudaStream_t> taskStreams(numStreams);
    for (int si = 0; si < numStreams; si++)
        checkCudaErrors(cudaStreamCreate(&taskStreams[si]));

    auto t_start = std::chrono::high_resolution_clock::now();
    for (int idx = 0; idx < (int)my_tasks_sorted.size(); idx++) {
        int task = my_tasks_sorted[idx];
        checkCudaErrors(cudaGraphLaunch(h_subgraphsExec[task], taskStreams[idx % numStreams]));
    }
    for (int si = 0; si < numStreams; si++)
        checkCudaErrors(cudaStreamSynchronize(taskStreams[si]));
    auto t_end = std::chrono::high_resolution_clock::now();

    for (int si = 0; si < numStreams; si++)
        checkCudaErrors(cudaStreamDestroy(taskStreams[si]));
    checkCudaErrors(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    double time = std::chrono::duration<double>(t_end - t_start).count();
    printf("device %d | %d run | time (s): %4.4f\n", myPE, i, time);
    totalTime += time;
}
    printf("Total time used (s): %4.4f\n", totalTime);

    if (verify) {
        double *h_L = (double *)malloc(N * N * sizeof(double));
        checkCudaErrors(cudaMemcpy(h_L, d_matrix, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        cleanCusolverCholeskyDecompositionResult(h_L, N);
        printf("Result passes verification: %d\n",
               verifyCholeskyDecomposition(originalMatrix.get(), h_L, N, verbose));
        free(h_L);
    }

    // Cleanup
    for (int task : pe_tasks[myPE]) {
        if (d_task_deps[task])       checkCudaErrors(cudaFree(d_task_deps[task]));
        if (d_task_notify_pes[task]) checkCudaErrors(cudaFree(d_task_notify_pes[task]));
    }
    delete[] h_subgraphsExec;
    nvshmem_free(d_completion_flags);
    nvshmem_free(d_matrices);
    nvshmem_free((void*)d_flags);
    checkCudaErrors(cudaFree(d_info));
    checkCudaErrors(cudaFree(d_workspace_cusolver));
    for (int i = 0; i < workspaces; i++)
        checkCudaErrors(cudaFree(d_workspace_cublas[i]));
    free(d_workspace_cublas);
}

void Cholesky(bool tiled, bool verify, bool subgraph, bool staticMultiGPU, bool dot)
{
    if (staticMultiGPU)
        tiledCholeskyStatic(verify, dot);
    else if (tiled && myPE == 0)
        tiledCholesky(verify, subgraph, dot);
    else if (subgraph)
        tiledCholesky(verify, subgraph, dot);
    else if (myPE == 0)
        trivialCholesky(verify);
}

int main(int argc, char **argv)
{
    auto wall_start = std::chrono::system_clock::now();
    auto program_start = std::chrono::high_resolution_clock::now();

    auto cmdl = argh::parser(argc, argv);

    if (!parseCommonArgs(cmdl, cfg)) {
        printSingleNodeUsage(argv[0], "Cholesky");
        return 1;
    }

    auto init_start = std::chrono::high_resolution_clock::now();
    initNvshmemDevice(cmdl, cfg);
    auto init_end = std::chrono::high_resolution_clock::now();

    myPE = cfg.myPE;
    if (myPE == 0) {
        double unix_start = std::chrono::duration<double>(wall_start.time_since_epoch()).count();
        printf("Program start timestamp: %.6f\n", unix_start);
    }
    double init_time = std::chrono::duration<double>(init_end - init_start).count();
    printf("device %d | NVSHMEM init time (s): %4.4f\n", myPE, init_time);
    fflush(stdout);

    if (!(cmdl["tiled"] || cmdl["subgraph"] || cmdl["static-multigpu"]))
        T = 1;
    B = N / T;

    if (myPE == 0) {
        if (cmdl["tiled"])
            std::cout << "TILED";
        else if (cmdl["subgraph"])
            std::cout << "SUBGRAPH";
        else if (cmdl["static-multigpu"])
            std::cout << "STATIC-MULTIGPU";
        else
            std::cout << "Single-kernel";
        std::cout << " with N=" << N << " (" << T << " of " << B << "x" << B << " tiles)" << std::endl;

        if (cmdl[{"subgraph", "tiled"}] || cmdl["static-multigpu"]) {
            std::cout << "SM Limit per kernel = " << smLimit << std::endl;
            std::cout << "cuBLAS workspace = " << workspace << " kB" << std::endl;
        }
    }

    Cholesky(cmdl["tiled"], cmdl["verify"] && myPE == 0, cmdl["subgraph"], cmdl["static-multigpu"], cmdl["dot"]);
    
    nvshmem_finalize();

    auto program_end = std::chrono::high_resolution_clock::now();
    double program_time = std::chrono::duration<double>(program_end - program_start).count();
    printf("device %d | Total program time (s): %4.4f\n", myPE, program_time);
    if (myPE == 0) {
        auto wall_end = std::chrono::system_clock::now();
        double unix_end = std::chrono::duration<double>(wall_end.time_since_epoch()).count();
        printf("Program end timestamp: %.6f\n", unix_end);
    }

    return 0;
}