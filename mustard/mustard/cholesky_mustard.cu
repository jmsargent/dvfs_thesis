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
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "allocator.h"
#include "argh.h"
#include "cli.h"
#include "gen.h"
#include "graph_assembler.h"
#include "injectors.h"
#include "mustard.h"
#include "pe_writer.h"
#include "task_timing.h"
#include "time_utils.cuh"
#include "tuner.h"
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
static int&          repeat    = cfg.repeat;
// When --output is set, all diagnostic output goes here instead of stdout so
// it cannot interleave with the CSV written by PEWriter.
static FILE* g_log = stdout;

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
        fprintf(g_log, "device %d | %d run | time (s): %4.4f\n", myPE, i, time);
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

    auto creator = std::make_unique<mustard::TiledGraphCreator>(s, graph, subgraph, totalNodes);

    for (int k = 0; k < T; k++)
    {
        // A[k][k] = GETRF(A[k][k])
        // L[k][k]*U[k][k] = A[k][k]
        checkCudaErrors(
            cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));
        creator->beginCaptureOperation(
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
        creator->endCaptureOperation();

        for (int i = k + 1; i < T; i++)
        {
            // L[i][k] = TRSM(A[i][k], A[k][k]) // the U part of A[k][k]
            // seems like only these need a separate workspace
            checkCudaErrors(
                cublasSetWorkspace(cublasHandle, d_workspace_cublas[i], cublasWorkspaceSize));
            creator->beginCaptureOperation(
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
            creator->endCaptureOperation();
        }

        for (int i = k + 1; i < T; i++)
        {
            // U[k][i] = TRSM(A[k][k], A[k][i]) // the L part of A[k][k]
            checkCudaErrors(
                cublasSetWorkspace(cublasHandle, d_workspace_cublas[i + T], cublasWorkspaceSize));
            creator->beginCaptureOperation(std::make_pair(i, i),
                                           {std::make_pair(i, i), std::make_pair(i, k)},
                                           "SYRK(" + std::to_string(i) + "," + std::to_string(i) +
                                               "," + std::to_string(k) + ")");

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
            creator->endCaptureOperation();

            for (int j = i + 1; j < T; j++)
            {
                // A[j][i] = GEMM(A[j][k], A[i][k])
                // A[j][i] = A[j][i] - L[j][k] * L[i][k]^T
                checkCudaErrors(cublasSetWorkspace(
                    cublasHandle, d_workspace_cublas[2 * T + (i - 1) * T + (j - 1)],
                    cublasWorkspaceSize));
                creator->beginCaptureOperation(
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
                creator->endCaptureOperation();
            }
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());

    cudaGraphExec_t graphExec;
    CudaEventClock  clock;
    double          totalTime = 0.0;

    if (subgraph)
    {
        if (verbose) creator->printDeps();

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
            h_dependencies[i] = creator->subgraphDependencies[i].size();
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
                creator->insertDependencyKernel(creator->subgraphDependencies[dst][src_ind], dst,
                                                queue, d_dependencies);
        if (verbose) showMemUsage();
        if (verbose) std::cout << "Uploading graphs..." << std::endl;

        if (!cfg.invocationPath.empty())
        {
            creator->printInvocations(cfg.invocationPath, myPE);
        }

        cudaGraphExec_t* h_subgraphsExec = new cudaGraphExec_t[totalNodes];
        cudaGraphExec_t* d_subgraphsExec;
        for (int i = 0; i < totalNodes; i++)
        {
            char filename[20];
            sprintf(filename, "./graph_%d.dot", i);
            if (dot) checkCudaErrors(cudaGraphDebugDotPrint(creator->subgraphs[i], filename, 0));
            checkCudaErrors(cudaGraphInstantiate(&h_subgraphsExec[i], creator->subgraphs[i],
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
            fprintf(g_log, "device %d | %d run | time (s): %4.4f\n", myPE, i, time);
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
            fprintf(g_log, "device %d | %d run | time (s): %4.4f\n", myPE, i, time);
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

void tiledCholeskyPanel(bool verify, bool dot)
{
    auto setup_start = std::chrono::high_resolution_clock::now();

    int nPEs = nvshmem_n_pes();

    StridedHostPanel hostPanel(myPE, nPEs, N, B, T);
    hostPanel.fill();

    mustard::OccupancyTracker occupancyTracker(smLimit);
    StridedDevicePanels       panels(myPE, nPEs, N, B, T);
    hostPanel.copyToDevicePanel(panels.myPanel());

    int totalNodes = T;
    for (int k = 0; k < (int)T; k++)
        for (int i = k + 1; i < (int)T; i++) totalNodes += 2 + (T - (i + 1));

    int numMyTasks = 0;
    for (int p = myPE; p < T; p += nPEs) numMyTasks += (T - p) * (p + 1);

    cudaStream_t s;
    checkCudaErrors(cudaStreamCreate(&s));
    cudaGraph_t graph;
    checkCudaErrors(cudaGraphCreate(&graph, 0));

    if (verbose)
    {
        std::cout << "totalNodes=" << totalNodes << std::endl;
        std::cout << "numMyTasks=" << numMyTasks << std::endl;
        std::cout << "tileSize=" << 1024 * workspace << std::endl;
    }
    printf("device %d | tiledCholeskyPanel: building %d/%d graphs\n", myPE, numMyTasks, totalNodes);
    fflush(stdout);

    CholeskyCudaOperations ops = CholeskyCudaOperations::build(
        s, std::move(panels), B, N, nPEs, numMyTasks, workspace, occupancyTracker);

    CompletionFlags d_completion_flags(totalNodes);

    auto creator = std::make_unique<mustard::TiledGraphCreator>(s, graph, true, totalNodes);

    MeasureFlags      flags{cfg.measureWait, cfg.measureCompute};
    OperationCapturer capturer(*creator, d_completion_flags.data(), s);
    KernelStopWatch   sw(flags, capturer);
    DVFSSignalBuilder dvfsSignalBuilder(capturer);
    PartitionedCudaGraphBuilder<CholeskyCudaOperations> graphBuilder(myPE, ops, capturer, sw,
                                                                     dvfsSignalBuilder);

    for (int pivotColumn = 0; pivotColumn < (int)T; pivotColumn++)
    {
        graphBuilder.add(pivotColumn % nPEs, [=](auto& o) { return o.potrf(pivotColumn); });

        for (int column = pivotColumn + 1; column < (int)T; column++)
            graphBuilder.add(pivotColumn % nPEs,
                             [=](auto& o) { return o.trsm(column, pivotColumn); });

        for (int column = pivotColumn + 1; column < (int)T; column++)
        {
            graphBuilder.add(column % nPEs, [=](auto& o) { return o.syrk(column, pivotColumn); });

            for (int row = column + 1; row < (int)T; row++)
                graphBuilder.add(column % nPEs,
                                 [=](auto& o) { return o.gemm(row, column, pivotColumn); });
        }
    }

    auto                  executableGraphs = graphBuilder.build(*creator, s, dot);
    TaskProfileRepository repo(myPE, TaskProfileRepository::Algorithm::Cholesky);
    if (!cfg.dbPath.empty()) repo.loadFromCSV(cfg.dbPath);

    auto partitionedDag = graphBuilder.getPartitionedGraph();

    auto ts              = sw.buffers();
    auto my_tasks_sorted = partitionedDag.nodes(myPE);
    auto numTasksSorted  = (int)my_tasks_sorted.size();

    printf("device %d | tiledCholeskyPanel: graphs instantiated, entering timing loop\n", myPE);
    fflush(stdout);

    if (!cfg.invocationPath.empty()) creator->printInvocations(cfg.invocationPath, myPE);

    auto   setup_end  = std::chrono::high_resolution_clock::now();
    double setup_time = std::chrono::duration<double>(setup_end - setup_start).count();
    printf("device %d | Setup time (s): %4.4f\n", myPE, setup_time);
    fflush(stdout);

    std::vector<int> task_indices;
    for (auto& n : my_tasks_sorted) task_indices.push_back(n.index);
    TaskTimingCollector collector(ts, task_indices, runs, cfg.measureFlags);
    double              totalTime = 0.0;

    EDP   goal(cfg.goalN, cfg.goalM);
    Tuner tuner(repo, dvfsSignalBuilder.signals(), partitionedDag, myPE, goal);
    tuner.plan();

    for (int i = 0; i < runs; i++)
    {
        // This makes me think Panels should not be owned by anything else
        hostPanel.copyToDevicePanel(ops.myPanel());  // panels owned by ops

        nvshmem_barrier_all();
        if (myPE == 0) d_completion_flags.resetAll(nPEs);
        nvshmem_barrier_all();

        int                       numStreams = std::min(numTasksSorted, cfg.numStreams);
        std::vector<cudaStream_t> taskStreams(numStreams);
        for (int si = 0; si < numStreams; si++) checkCudaErrors(cudaStreamCreate(&taskStreams[si]));

        gpu_clock::CalibrationRef ts_ref;

        if (myPE == 0) print_timestamp("cholesky tiledPanel start_time", 7);

        auto t_start = std::chrono::high_resolution_clock::now();
        tuner.reset(taskStreams);
        if (collector.active()) ts_ref = gpu_clock::calibrate(taskStreams[0]);
        for (int idx = 0; idx < numTasksSorted; idx++)
            checkCudaErrors(cudaGraphLaunch(executableGraphs[idx], taskStreams[idx % numStreams]));

        tuner.run();

        for (int si = 0; si < numStreams; si++)
            checkCudaErrors(cudaStreamSynchronize(taskStreams[si]));

        checkCudaErrors(cudaDeviceSynchronize());
        auto t_end = std::chrono::high_resolution_clock::now();

        if (myPE == 0) print_timestamp("cholesky tiledPanel end_time", 7);

        collector.collect(i, ts_ref);
        for (int si = 0; si < numStreams; si++) checkCudaErrors(cudaStreamDestroy(taskStreams[si]));
        nvshmem_barrier_all();

        double time = std::chrono::duration<double>(t_end - t_start).count();
        fprintf(g_log, "device %d | %d run | time (s): %4.4f\n", myPE, i, time);
        totalTime += time;
    }
    printf("Total time used (s): %4.4f\n", totalTime);

    collector.write(cfg.outputPrefix, myPE, creator->subgraphOpNames);

    if (verify)
    {
        bool pass = hostPanel.verify(ops.myPanel());  // panels owned by ops
        printf("device %d | diagonal verification: %s\n", myPE, pass ? "PASS" : "FAIL");
    }

}  // ops destructor frees handles, workspaces, panels, and d_info; occupancyTracker frees d_flags

void Cholesky(bool tiled, bool verify, bool subgraph, bool staticMultiGPU, bool oneGraphPerPE,
              bool panel, bool dot)
{
    if (panel)
        tiledCholeskyPanel(verify, dot);
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
    if (!cfg.outputPrefix.empty())
    {
        char logname[512];
        snprintf(logname, sizeof(logname), "%s_pe%d.log", cfg.outputPrefix.c_str(), myPE);
        FILE* f = fopen(logname, "w");
        if (f) g_log = f;
    }
    if (myPE == 0) print_timestamp("Program start timestamp", wall_start, 7, g_log);
    double init_time = std::chrono::duration<double>(init_end - init_start).count();
    fprintf(g_log, "device %d | NVSHMEM init time (s): %4.4f\n", myPE, init_time);
    fflush(g_log);

    if (!(cmdl["tiled"] || cmdl["subgraph"] || cmdl["static-multigpu"] ||
          cmdl["one-graph-per-pe"] || cmdl["panel"]))
        T = 1;
    B = N / T;

    if (myPE == 0)
    {
        if (cmdl["panel"])
            std::cout << "PANEL";
        else if (cmdl["tiled"])
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
             cmdl["one-graph-per-pe"], cmdl["panel"], cmdl["dot"]);

    nvshmem_finalize();

    auto   program_end  = std::chrono::high_resolution_clock::now();
    double program_time = std::chrono::duration<double>(program_end - program_start).count();
    printf("device %d | Total program time (s): %4.4f\n", myPE, program_time);
    if (myPE == 0) print_timestamp("Program end timestamp");

    return 0;
}