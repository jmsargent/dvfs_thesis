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

#include "argh.h"
#include "cli.h"
#include "gen.h"
#include "mustard.h"
#include "verify.h"

// Global configuration (populated from CLI in main).
static MustardConfig cfg;
static size_t       &N = cfg.N;
static size_t       &B = cfg.B;
static size_t       &T = cfg.T;
int                  myPE;
static int          &verbose   = cfg.verbose;
static int          &workspace = cfg.workspace;
static int          &smLimit   = cfg.smLimit;
static int          &runs      = cfg.runs;
// mustard/lu_mustard.cu

void trivialLU(bool verify)
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

    checkCudaErrors(cusolverDnXgetrf_bufferSize(
        cusolverDnHandle, cusolverDnParams, N, N, CUDA_R_64F, d_A, N, CUDA_R_64F,
        &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    void *h_workspace = malloc(workspaceInBytesOnHost);

    void *d_workspace;
    checkCudaErrors(cudaMalloc(&d_workspace, workspaceInBytesOnDevice));

    int *d_info;
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));
    CudaEventClock clock;
    double         totalTime = 0.0;

    for (int i = 0; i < runs; i++)
    {
        checkCudaErrors(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
        clock.start();
        checkCudaErrors(
            cusolverDnXgetrf(cusolverDnHandle, cusolverDnParams, N, N, CUDA_R_64F, d_A, N,
                             NULL,  // no pivoting
                             CUDA_R_64F, d_workspace, workspaceInBytesOnDevice, NULL, 0, d_info));
        checkCudaErrors(cudaStreamSynchronize(0));
        clock.end();
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemset(d_workspace, 0, workspaceInBytesOnDevice));
        float time = clock.getTimeInSeconds();
        printf("device %d | %d run | time (s): %4.4f\n", myPE, i, time);
        totalTime += time;
    }

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
        double *h_U = (double *)malloc(N * N * sizeof(double));
        checkCudaErrors(cudaMemcpy(h_L, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        cleanCusolverLUDecompositionResult(h_L, h_U, N);
        printf("Result passes verification: %d\n",
               verifyLUDecomposition(h_A, h_L, h_U, N, verbose));

        // Clean
        free(h_L);
        free(h_U);
    }

    printf("Total time used (s): %4.4f\n", totalTime);

    free(h_A);
    free(h_workspace);
    checkCudaErrors(cusolverDnDestroy(cusolverDnHandle));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_workspace));
    checkCudaErrors(cudaFree(d_info));
}

cudaGraph_t recordSubgraph(double *subMatrix, int subT, cudaStream_t s,
                           cusolverDnHandle_t cusolverDnHandle, cublasHandle_t cublasHandle,
                           double *d_workspace_cusolver, void **d_workspace_cublas,
                           int cublasWorkspaceSize, int *d_info)
{
    int    subN     = B;
    int    subB     = subN / subT;
    double one      = 1.0;
    double minusOne = -1.0;

    std::cout << "PE " << myPE << ". Create subgraph with subN=" << subT << " (" << subT << " of "
              << subB << "x" << subB << " tiles)" << std::endl;

    if (subT > T)
    {
        std::cout << "Unable to create such a graphx" << std::endl;
        exit(0);
    }

    auto getMatrixBlock = [&](double *matrix, int i, int j)
    { return matrix + i * subB + j * subB * N; };

    int totalNodes = subT;

    for (int k = 0; k < subT; k++)
        for (int i = k + 1; i < subT; i++) totalNodes += 2 + (subT - (k + 1));

    cudaGraph_t subGraph;
    checkCudaErrors(cudaGraphCreate(&subGraph, 0));

    auto tiledLUGraphCreator =
        std::make_unique<mustard::TiledGraphCreator>(s, subGraph, false, totalNodes);

    for (int k = 0; k < subT; k++)
    {
        // A[k][k] = GETRF(A[k][k])
        // L[k][k]*U[k][k] = A[k][k]

        tiledLUGraphCreator->beginCaptureOperation(
            std::make_pair(k, k), {std::make_pair(k, k)},
            "GETRF_SUB(" + std::to_string(k) + "," + std::to_string(k) + ")");
        checkCudaErrors(cusolverDnDgetrf(cusolverDnHandle, subB, subB,
                                         getMatrixBlock(subMatrix, k, k), N, d_workspace_cusolver,
                                         NULL, d_info));
        tiledLUGraphCreator->endCaptureOperation();

        for (int i = k + 1; i < subT; i++)
        {
            checkCudaErrors(
                cublasSetWorkspace(cublasHandle, d_workspace_cublas[i - 1], cublasWorkspaceSize));
            // L[i][k] = TRSM(A[i][k], A[k][k]) // the U part of A[k][k]
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(k, i), {std::make_pair(k, k), std::make_pair(k, i)},
                "TRSM_L_SUB(" + std::to_string(k) + "," + std::to_string(i) + ")");
            checkCudaErrors(cublasDtrsm(cublasHandle,
                                        CUBLAS_SIDE_LEFT,  // used to be right for cholesky
                                        CUBLAS_FILL_MODE_LOWER,
                                        CUBLAS_OP_N,       // CUBLAS_OP_T for cholesky
                                        CUBLAS_DIAG_UNIT,  // CUBLAS_DIAG_NON_UNIT for cholesky
                                        subB, subB, &one, getMatrixBlock(subMatrix, k, k),
                                        N,                                     // k + k * N;
                                        getMatrixBlock(subMatrix, k, i), N));  // k + (i + B) * N;
            tiledLUGraphCreator->endCaptureOperation();
        }

        for (int i = k + 1; i < subT; i++)
        {
            checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[T + i - 1],
                                               cublasWorkspaceSize));
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(i, k), {std::make_pair(k, k), std::make_pair(i, k)},
                "TRSM_R_SUB(" + std::to_string(i) + "," + std::to_string(k) + ")");
            checkCudaErrors(cublasDtrsm(cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                                        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, subB, subB, &one,
                                        getMatrixBlock(subMatrix, k, k), N,    // k + k * N;
                                        getMatrixBlock(subMatrix, i, k), N));  // (i + B) + k * N;
            tiledLUGraphCreator->endCaptureOperation();

            for (int j = k + 1; j < subT; j++)
            {
                checkCudaErrors(cublasSetWorkspace(
                    cublasHandle, d_workspace_cublas[2 * subT + j - 1], cublasWorkspaceSize));
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(i, j),
                    {std::make_pair(i, k), std::make_pair(k, j), std::make_pair(i, j)},
                    "GEMM_SUB(" + std::to_string(i) + "," + std::to_string(j) + "," +
                        std::to_string(k) + ")");
                checkCudaErrors(
                    cublasGemmEx(cublasHandle, CUBLAS_OP_N,
                                 CUBLAS_OP_N,  // CUBLAS_OP_T
                                 subB, subB, subB, &minusOne, getMatrixBlock(subMatrix, i, k),
                                 CUDA_R_64F, N,                                         // i + k * N
                                 getMatrixBlock(subMatrix, k, j), CUDA_R_64F, N,        // j + i * N
                                 &one, getMatrixBlock(subMatrix, i, j), CUDA_R_64F, N,  // k + i * N
                                 CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
                tiledLUGraphCreator->endCaptureOperation();
            }
        }
    }

    return subGraph;
}

void tiledLU(bool verify, bool subgraph, bool dot)
{
    auto setup_start = std::chrono::high_resolution_clock::now();

    // Initialize data
    auto originalMatrix = std::make_unique<double[]>(N * N);  // Column-major
    generateRandomSymmetricPositiveDefiniteMatrix(originalMatrix.get(), N);

    // Copy to device
    double       *d_matrix;
    double       *d_matrices;
    double       *d_matrix_remote;
    volatile int *d_flags;
    if (subgraph)
    {
        d_flags    = (volatile int *)nvshmem_malloc(sizeof(int) * 32);
        d_matrices = (double *)nvshmem_malloc(N * N * sizeof(double));
        d_matrix   = (double *)nvshmem_ptr(d_matrices, myPE);
    }
    else
    {
        checkCudaErrors(cudaMalloc(&d_matrix, N * N * sizeof(double)));
    }
    checkCudaErrors(
        cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    if (myPE != 0) d_matrix_remote = (double *)nvshmem_ptr(d_matrices, 0);

    auto getMatrixBlock = [&](double *matrix, int i, int j) { return matrix + i * B + j * B * N; };

    // Initialize libraries
    cusolverDnHandle_t cusolverDnHandle;
    cusolverDnParams_t cusolverDnParams;
    cublasHandle_t     cublasHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cublasSetSmCountTarget(cublasHandle, smLimit));

    // Prepare constants
    double one      = 1.0;
    double minusOne = -1.0;

    int workspaceInBytesOnDevice;

    checkCudaErrors(cusolverDnDgetrf_bufferSize(cusolverDnHandle, B, B, d_matrix, N,
                                                &workspaceInBytesOnDevice));

    double *d_workspace_cusolver;
    int     workspaces         = T * T;
    void  **d_workspace_cublas = (void **)malloc(sizeof(void *) * workspaces);
    int    *d_info;
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
        for (int i = k + 1; i < T; i++) totalNodes += 2 + (T - (k + 1));

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

    auto tiledLUGraphCreator =
        std::make_unique<mustard::TiledGraphCreator>(s, graph, subgraph, totalNodes);

    for (int k = 0; k < T; k++)
    {
        // A[k][k] = GETRF(A[k][k])
        // L[k][k]*U[k][k] = A[k][k]
        checkCudaErrors(
            cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));
        tiledLUGraphCreator->beginCaptureOperation(
            std::make_pair(k, k), {std::make_pair(k, k)},
            "GETRF(" + std::to_string(k) + "," + std::to_string(k) + ")");
        if (subgraph)
        {
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k), sizeof(double) * N,
                                  getMatrixBlock(d_matrix_remote, k, k), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
        }
        if (B <= MAX_TILE || !subgraph)
            checkCudaErrors(cusolverDnDgetrf(cusolverDnHandle, B, B, getMatrixBlock(d_matrix, k, k),
                                             N, d_workspace_cusolver, NULL, d_info));
        if (subgraph)
        {
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, k, k), sizeof(double) * N,
                                  getMatrixBlock(d_matrix, k, k), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
        }
        tiledLUGraphCreator->endCaptureOperation();

        if (B > MAX_TILE && subgraph)
        {
            int         subT  = ceil((float)B / (float)MAX_TILE);
            cudaGraph_t subLU = recordSubgraph(getMatrixBlock(d_matrix, k, k), subT, s,
                                               cusolverDnHandle, cublasHandle, d_workspace_cusolver,
                                               d_workspace_cublas, cublasWorkspaceSize, d_info);
            tiledLUGraphCreator->insertSubgraph(subLU);
        }

        for (int i = k + 1; i < T; i++)
        {
            // L[i][k] = TRSM(A[i][k], A[k][k]) // the U part of A[k][k]
            // seems like only these need a separate workspace
            checkCudaErrors(
                cublasSetWorkspace(cublasHandle, d_workspace_cublas[i], cublasWorkspaceSize));
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(k, i), {std::make_pair(k, k), std::make_pair(k, i)},
                "TRSM_L(" + std::to_string(k) + "," + std::to_string(i) + ")");
            if (subgraph)
            {
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
                if (myPE != 0 && k != 0)
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, i), sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, k, i), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                if (myPE != 0)
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k), sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, k, k), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            }
            checkCudaErrors(cublasDtrsm(cublasHandle,
                                        CUBLAS_SIDE_LEFT,  // used to be right for cholesky
                                        CUBLAS_FILL_MODE_LOWER,
                                        CUBLAS_OP_N,       // CUBLAS_OP_T for cholesky
                                        CUBLAS_DIAG_UNIT,  // CUBLAS_DIAG_NON_UNIT for cholesky
                                        B, B, &one, getMatrixBlock(d_matrix, k, k),
                                        N,                                    // k + k * N;
                                        getMatrixBlock(d_matrix, k, i), N));  // k + (i + B) * N;
            if (subgraph)
            {
                if (myPE != 0)
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, k, i), sizeof(double) * N,
                                      getMatrixBlock(d_matrix, k, i), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            }
            tiledLUGraphCreator->endCaptureOperation();
        }

        for (int i = k + 1; i < T; i++)
        {
            // U[k][i] = TRSM(A[k][k], A[k][i]) // the L part of A[k][k]
            checkCudaErrors(
                cublasSetWorkspace(cublasHandle, d_workspace_cublas[T + i], cublasWorkspaceSize));
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(i, k), {std::make_pair(k, k), std::make_pair(i, k)},
                "TRSM_R(" + std::to_string(i) + "," + std::to_string(k) + ")");

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
            checkCudaErrors(cublasDtrsm(cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                                        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, B, B, &one,
                                        getMatrixBlock(d_matrix, k, k), N,    // k + k * N;
                                        getMatrixBlock(d_matrix, i, k), N));  // (i + B) + k * N;
            if (subgraph)
            {
                if (myPE != 0)
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, k), sizeof(double) * N,
                                      getMatrixBlock(d_matrix, i, k), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            }
            tiledLUGraphCreator->endCaptureOperation();

            for (int j = k + 1; j < T; j++)
            {
                // A[j][i] = GEMM(A[j][k], A[i][k])
                // A[j][i] = A[j][i] - L[j][k] * L[i][k]^T
                checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[2 * T + j - 1],
                                                   cublasWorkspaceSize));
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(i, j),
                    {std::make_pair(i, k), std::make_pair(k, j), std::make_pair(i, j)},
                    "GEMM(" + std::to_string(i) + "," + std::to_string(j) + "," +
                        std::to_string(k) + ")");
                if (subgraph)
                {
                    mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
                    if (myPE != 0)
                    {
                        cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), sizeof(double) * N,
                                          getMatrixBlock(d_matrix_remote, i, k), sizeof(double) * N,
                                          sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                        cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, j), sizeof(double) * N,
                                          getMatrixBlock(d_matrix_remote, k, j), sizeof(double) * N,
                                          sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                        cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, j), sizeof(double) * N,
                                          getMatrixBlock(d_matrix_remote, i, j), sizeof(double) * N,
                                          sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                    }
                }
                checkCudaErrors(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, B, B, B,
                                             &minusOne, getMatrixBlock(d_matrix, i, k), CUDA_R_64F,
                                             N, getMatrixBlock(d_matrix, k, j), CUDA_R_64F, N, &one,
                                             getMatrixBlock(d_matrix, i, j), CUDA_R_64F, N,
                                             CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
                if (subgraph)
                {
                    if (myPE != 0)
                        cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, j), sizeof(double) * N,
                                          getMatrixBlock(d_matrix, i, j), sizeof(double) * N,
                                          sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                    mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
                }
                tiledLUGraphCreator->endCaptureOperation();
            }
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());

    cudaGraphExec_t graphExec;
    CudaEventClock  clock;
    double          totalTime = 0.0;

    if (subgraph)
    {
        if (verbose) tiledLUGraphCreator->printDeps();

        int      *h_dependencies;
        const int queue_size = totalNodes * 2;
        if (verbose) std::cout << "Creating queue..." << std::endl;
        BrokerWorkDistributor queue(queue_size);
        if (verbose) std::cout << "Allocating memory..." << std::endl;

        int *d_dependencies = (int *)nvshmem_malloc(sizeof(int) * totalNodes);
        checkCudaErrors(cudaMallocHost(&h_dependencies, sizeof(int) * totalNodes));
        if (verbose) std::cout << "Setting dependencies..." << std::endl;

        for (int i = 0; i < totalNodes; i++)
        {
            h_dependencies[i] = tiledLUGraphCreator->subgraphDependencies[i].size();
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
                tiledLUGraphCreator->insertDependencyKernel(
                    tiledLUGraphCreator->subgraphDependencies[dst][src_ind], dst, queue,
                    d_dependencies);
        if (verbose) showMemUsage();
        if (verbose) std::cout << "Uploading graphs..." << std::endl;

        if (!cfg.invocationPath.empty())
        {
            tiledLUGraphCreator->printInvocations(cfg.invocationPath, myPE);
        }

        cudaGraphExec_t *h_subgraphsExec = new cudaGraphExec_t[totalNodes];
        cudaGraphExec_t *d_subgraphsExec;
        for (int i = 0; i < totalNodes; i++)
        {
            char filename[20];
            sprintf(filename, "./graph_%d_%d.dot", i, myPE);
            if (dot)
                checkCudaErrors(
                    cudaGraphDebugDotPrint(tiledLUGraphCreator->subgraphs[i], filename, 0));
            checkCudaErrors(cudaGraphInstantiate(&h_subgraphsExec[i],
                                                 tiledLUGraphCreator->subgraphs[i],
                                                 cudaGraphInstantiateFlagDeviceLaunch));
            cudaGraphUpload(h_subgraphsExec[i], s);
        }
        checkCudaErrors(cudaMalloc(&d_subgraphsExec, sizeof(cudaGraphExec_t) * totalNodes));
        checkCudaErrors(cudaMemcpy((void *)d_subgraphsExec, (void *)h_subgraphsExec,
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
        nvshmem_free((void *)d_flags);
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
        double *h_L = (double *)malloc(N * N * sizeof(double));
        double *h_U = (double *)malloc(N * N * sizeof(double));
        checkCudaErrors(cudaMemcpy(h_L, d_matrix, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        memset(h_U, 0, N * N * sizeof(double));
        cleanCusolverLUDecompositionResult(h_L, h_U, N);
        printf("Result passes verification: %d\n",
               verifyLUDecomposition(originalMatrix.get(), h_L, h_U, N, verbose));

        free(h_L);
        free(h_U);
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

void tiledLUStatic(bool verify, bool dot)
{
    auto setup_start = std::chrono::high_resolution_clock::now();

    int nPEs = nvshmem_n_pes();

    // Parse measure flags
    bool measure_wait    = cfg.measureFlags.find("task_wait_time") != std::string::npos;
    bool measure_compute = cfg.measureFlags.find("task_compute_time") != std::string::npos;

    // Initialize data
    auto originalMatrix = std::make_unique<double[]>(N * N);
    generateRandomSymmetricPositiveDefiniteMatrix(originalMatrix.get(), N);

    // NVSHMEM allocations (all PEs participate)
    volatile int *d_flags         = (volatile int *)nvshmem_malloc(sizeof(int) * 32);
    double       *d_matrices      = (double *)nvshmem_malloc(N * N * sizeof(double));
    double       *d_matrix        = (double *)nvshmem_ptr(d_matrices, myPE);
    double       *d_matrix_remote = nullptr;
    checkCudaErrors(
        cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    if (myPE != 0) d_matrix_remote = (double *)nvshmem_ptr(d_matrices, 0);

    auto getMatrixBlock = [&](double *matrix, int i, int j) { return matrix + i * B + j * B * N; };

    // Initialize libraries
    cusolverDnHandle_t cusolverDnHandle;
    cusolverDnParams_t cusolverDnParams;
    cublasHandle_t     cublasHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cublasSetSmCountTarget(cublasHandle, smLimit));

    double one      = 1.0;
    double minusOne = -1.0;

    int workspaceInBytesOnDevice;
    checkCudaErrors(cusolverDnDgetrf_bufferSize(cusolverDnHandle, B, B, d_matrix, N,
                                                &workspaceInBytesOnDevice));

    double *d_workspace_cusolver;
    int     workspaces         = T * T;
    void  **d_workspace_cublas = (void **)malloc(sizeof(void *) * workspaces);
    int    *d_info;
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
        for (int i = k + 1; i < T; i++) totalNodes += 2 + (T - (k + 1));

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

    auto tiledLUGraphCreator =
        std::make_unique<mustard::TiledGraphCreator>(s, graph, true, totalNodes);

    // Graph construction
    for (int k = 0; k < T; k++)
    {
        checkCudaErrors(
            cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));
        tiledLUGraphCreator->beginCaptureOperation(
            std::make_pair(k, k), {std::make_pair(k, k)},
            "GETRF(" + std::to_string(k) + "," + std::to_string(k) + ")");
        mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
        if (myPE != 0)
            cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k), sizeof(double) * N,
                              getMatrixBlock(d_matrix_remote, k, k), sizeof(double) * N,
                              sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
        if (B <= MAX_TILE)
            checkCudaErrors(cusolverDnDgetrf(cusolverDnHandle, B, B, getMatrixBlock(d_matrix, k, k),
                                             N, d_workspace_cusolver, NULL, d_info));
        if (myPE != 0)
            cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, k, k), sizeof(double) * N,
                              getMatrixBlock(d_matrix, k, k), sizeof(double) * N,
                              sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
        mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
        tiledLUGraphCreator->endCaptureOperation();

        if (B > MAX_TILE)
        {
            int         subT  = ceil((float)B / (float)MAX_TILE);
            cudaGraph_t subLU = recordSubgraph(getMatrixBlock(d_matrix, k, k), subT, s,
                                               cusolverDnHandle, cublasHandle, d_workspace_cusolver,
                                               d_workspace_cublas, cublasWorkspaceSize, d_info);
            tiledLUGraphCreator->insertSubgraph(subLU);
        }

        for (int i = k + 1; i < T; i++)
        {
            checkCudaErrors(
                cublasSetWorkspace(cublasHandle, d_workspace_cublas[i], cublasWorkspaceSize));
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(k, i), {std::make_pair(k, k), std::make_pair(k, i)},
                "TRSM_L(" + std::to_string(k) + "," + std::to_string(i) + ")");
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
            if (myPE != 0 && k != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, i), sizeof(double) * N,
                                  getMatrixBlock(d_matrix_remote, k, i), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k), sizeof(double) * N,
                                  getMatrixBlock(d_matrix_remote, k, k), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            checkCudaErrors(cublasDtrsm(cublasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                        CUBLAS_OP_N, CUBLAS_DIAG_UNIT, B, B, &one,
                                        getMatrixBlock(d_matrix, k, k), N,
                                        getMatrixBlock(d_matrix, k, i), N));
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, k, i), sizeof(double) * N,
                                  getMatrixBlock(d_matrix, k, i), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            tiledLUGraphCreator->endCaptureOperation();
        }

        for (int i = k + 1; i < T; i++)
        {
            checkCudaErrors(
                cublasSetWorkspace(cublasHandle, d_workspace_cublas[T + i], cublasWorkspaceSize));
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(i, k), {std::make_pair(k, k), std::make_pair(i, k)},
                "TRSM_R(" + std::to_string(i) + "," + std::to_string(k) + ")");
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
            if (myPE != 0 && k != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), sizeof(double) * N,
                                  getMatrixBlock(d_matrix_remote, i, k), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k), sizeof(double) * N,
                                  getMatrixBlock(d_matrix_remote, k, k), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            checkCudaErrors(cublasDtrsm(cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                                        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, B, B, &one,
                                        getMatrixBlock(d_matrix, k, k), N,
                                        getMatrixBlock(d_matrix, i, k), N));
            if (myPE != 0)
                cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, k), sizeof(double) * N,
                                  getMatrixBlock(d_matrix, i, k), sizeof(double) * N,
                                  sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
            mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
            tiledLUGraphCreator->endCaptureOperation();

            for (int j = k + 1; j < T; j++)
            {
                checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[2 * T + j - 1],
                                                   cublasWorkspaceSize));
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(i, j),
                    {std::make_pair(i, k), std::make_pair(k, j), std::make_pair(i, j)},
                    "GEMM(" + std::to_string(i) + "," + std::to_string(j) + "," +
                        std::to_string(k) + ")");
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(smLimit, d_flags);
                if (myPE != 0)
                {
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, i, k), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, j), sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, k, j), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, j), sizeof(double) * N,
                                      getMatrixBlock(d_matrix_remote, i, j), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                }
                checkCudaErrors(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, B, B, B,
                                             &minusOne, getMatrixBlock(d_matrix, i, k), CUDA_R_64F,
                                             N, getMatrixBlock(d_matrix, k, j), CUDA_R_64F, N, &one,
                                             getMatrixBlock(d_matrix, i, j), CUDA_R_64F, N,
                                             CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
                if (myPE != 0)
                    cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, j), sizeof(double) * N,
                                      getMatrixBlock(d_matrix, i, j), sizeof(double) * N,
                                      sizeof(double) * B, B, cudaMemcpyDeviceToDevice, s);
                mustard::kernel_occupancy_update<<<1, 1, 0, s>>>(-smLimit, d_flags);
                tiledLUGraphCreator->endCaptureOperation();
            }
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());

    // Static task assignment — round-robin
    std::vector<std::vector<int>> pe_tasks(nPEs);
    for (int i = 0; i < totalNodes; i++) pe_tasks[i % nPEs].push_back(i);

    std::vector<int> task_pe(totalNodes);
    for (int i = 0; i < totalNodes; i++) task_pe[i] = i % nPEs;

    // Topological sort of myPE's tasks respecting same-PE dependencies
    std::vector<int> my_tasks_sorted;
    std::set<int>    topo_done;
    while (my_tasks_sorted.size() < pe_tasks[myPE].size())
    {
        bool progress = false;
        for (int task : pe_tasks[myPE])
        {
            if (topo_done.count(task)) continue;
            bool ready = true;
            for (int dep : tiledLUGraphCreator->subgraphDependencies[task])
            {
                if (task_pe[dep] == myPE && !topo_done.count(dep))
                {
                    ready = false;
                    break;
                }
            }
            if (ready)
            {
                my_tasks_sorted.push_back(task);
                topo_done.insert(task);
                progress = true;
            }
        }
        if (!progress) break;
    }

    // NVSHMEM completion flags
    int *d_completion_flags = (int *)nvshmem_malloc(sizeof(int) * totalNodes);
    checkCudaErrors(cudaMemset(d_completion_flags, 0, sizeof(int) * totalNodes));

    // Reverse dependency map
    std::vector<std::vector<int>> dependents(totalNodes);
    for (int j = 0; j < totalNodes; j++)
        for (int dep : tiledLUGraphCreator->subgraphDependencies[j]) dependents[dep].push_back(j);

    // Debug: print dependency and notify info
    for (int task : pe_tasks[myPE])
    {
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
    std::vector<int *> d_task_deps(totalNodes, nullptr);
    std::vector<int *> d_task_notify_pes(totalNodes, nullptr);

    for (int task : pe_tasks[myPE])
    {
        auto &deps = tiledLUGraphCreator->subgraphDependencies[task];
        if (!deps.empty())
        {
            checkCudaErrors(cudaMalloc(&d_task_deps[task], sizeof(int) * deps.size()));
            checkCudaErrors(cudaMemcpy(d_task_deps[task], deps.data(), sizeof(int) * deps.size(),
                                       cudaMemcpyHostToDevice));
        }

        std::set<int> notify_set;
        for (int dep_task : dependents[task]) notify_set.insert(task_pe[dep_task]);
        std::vector<int> notify_vec(notify_set.begin(), notify_set.end());
        if (!notify_vec.empty())
        {
            checkCudaErrors(cudaMalloc(&d_task_notify_pes[task], sizeof(int) * notify_vec.size()));
            checkCudaErrors(cudaMemcpy(d_task_notify_pes[task], notify_vec.data(),
                                       sizeof(int) * notify_vec.size(), cudaMemcpyHostToDevice));
        }
    }

    // Helper to find tail node of a subgraph
    auto getSubgraphTail = [](cudaGraph_t g) -> cudaGraphNode_t
    {
        size_t numEdges;
        MUSTARD_cudaGraphGetEdges(g, nullptr, nullptr, &numEdges);
        if (numEdges == 0)
        {
            size_t          numNodes = 1;
            cudaGraphNode_t node;
            cudaGraphGetNodes(g, &node, &numNodes);
            return node;
        }
        std::vector<cudaGraphNode_t> from(numEdges), to(numEdges);
        MUSTARD_cudaGraphGetEdges(g, from.data(), to.data(), &numEdges);
        std::map<cudaGraphNode_t, bool> hasOutgoing;
        std::set<cudaGraphNode_t>       noOutgoing;
        for (size_t e = 0; e < numEdges; e++)
        {
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

    // --- Loop 1: Inject wait and signal kernels ---
    int debug = cfg.debugKernels;
    // Track waitNode per task so loop 2 can reference it
    std::vector<cudaGraphNode_t> task_wait_node(totalNodes, nullptr);

    for (int task : pe_tasks[myPE])
    {
        cudaGraph_t sg   = tiledLUGraphCreator->subgraphs[task];
        auto       &deps = tiledLUGraphCreator->subgraphDependencies[task];

        if (!deps.empty())
        {
            size_t numRoots;
            cudaGraphGetRootNodes(sg, nullptr, &numRoots);
            std::vector<cudaGraphNode_t> roots(numRoots);
            cudaGraphGetRootNodes(sg, roots.data(), &numRoots);

            cudaGraphNode_t      waitNode;
            cudaKernelNodeParams waitParams = {0};
            waitParams.gridDim              = dim3(1);
            waitParams.blockDim             = dim3(1);
            waitParams.func                 = (void *)mustard::kernel_wait_static;
            int   n_deps                    = (int)deps.size();
            void *waitArgs[4]       = {&d_task_deps[task], &n_deps, &d_completion_flags, &debug};
            waitParams.kernelParams = waitArgs;
            checkCudaErrors(cudaGraphAddKernelNode(&waitNode, sg, nullptr, 0, &waitParams));
            task_wait_node[task] = waitNode;

            for (auto &root : roots) MUSTARD_cudaGraphAddDependencies(sg, &waitNode, &root, 1);
        }

        std::set<int> notify_set;
        for (int dep_task : dependents[task]) notify_set.insert(task_pe[dep_task]);
        int n_notify = (int)notify_set.size();

        cudaGraphNode_t      tail = getSubgraphTail(sg);
        cudaGraphNode_t      signalNode;
        cudaKernelNodeParams signalParams = {0};
        signalParams.gridDim              = dim3(1);
        signalParams.blockDim             = dim3(1);
        signalParams.func                 = (void *)mustard::kernel_signal_static;
        int   task_id_val                 = task;
        void *signalArgs[5]       = {&task_id_val, &d_completion_flags, &d_task_notify_pes[task],
                                     &n_notify, &debug};
        signalParams.kernelParams = signalArgs;
        checkCudaErrors(cudaGraphAddKernelNode(&signalNode, sg, &tail, 1, &signalParams));
    }

    // --- Loop 2: Inject measurement event nodes (separate concern) ---
    std::vector<cudaEvent_t> ev_compute_start(totalNodes, nullptr);
    std::vector<cudaEvent_t> ev_compute_end(totalNodes, nullptr);

    if (measure_wait || measure_compute)
    {
        for (int task : pe_tasks[myPE])
        {
            cudaGraph_t sg   = tiledLUGraphCreator->subgraphs[task];
            auto       &deps = tiledLUGraphCreator->subgraphDependencies[task];

            // Inject compute start event: after waitNode if present, else before roots
            checkCudaErrors(cudaEventCreate(&ev_compute_start[task]));
            cudaGraphNode_t computeStartNode;
            if (!deps.empty())
            {
                // waitNode is the single root — add event node after it
                cudaGraphNode_t waitNode = task_wait_node[task];
                checkCudaErrors(cudaGraphAddEventRecordNode(&computeStartNode, sg, &waitNode, 1,
                                                            ev_compute_start[task]));
                // rewire: children of waitNode (except computeStartNode) now depend on
                // computeStartNode
                size_t numDeps;
                MUSTARD_cudaGraphNodeGetDependentNodes(waitNode, nullptr, &numDeps);
                std::vector<cudaGraphNode_t> children(numDeps);
                MUSTARD_cudaGraphNodeGetDependentNodes(waitNode, children.data(), &numDeps);
                for (auto &child : children)
                {
                    if (child == computeStartNode) continue;
                    MUSTARD_cudaGraphAddDependencies(sg, &computeStartNode, &child, 1);
                    MUSTARD_cudaGraphRemoveDependencies(sg, &waitNode, &child, 1);
                }
            }
            else
            {
                // No wait node — insert before all current roots
                size_t numRoots;
                cudaGraphGetRootNodes(sg, nullptr, &numRoots);
                std::vector<cudaGraphNode_t> roots(numRoots);
                cudaGraphGetRootNodes(sg, roots.data(), &numRoots);
                checkCudaErrors(cudaGraphAddEventRecordNode(&computeStartNode, sg, nullptr, 0,
                                                            ev_compute_start[task]));
                for (auto &root : roots)
                    MUSTARD_cudaGraphAddDependencies(sg, &computeStartNode, &root, 1);
            }

            // Inject compute end event: tail is signalNode, so insert before it
            if (measure_compute)
            {
                checkCudaErrors(cudaEventCreate(&ev_compute_end[task]));
                // signalNode is now the tail; its only parent is the previous tail (before signal)
                cudaGraphNode_t signalNode = getSubgraphTail(sg);
                size_t          numParents;
                MUSTARD_cudaGraphNodeGetDependencies(signalNode, nullptr, &numParents);
                std::vector<cudaGraphNode_t> parents(numParents);
                MUSTARD_cudaGraphNodeGetDependencies(signalNode, parents.data(), &numParents);
                // insert computeEndNode after all parents of signalNode, before signalNode
                cudaGraphNode_t computeEndNode;
                checkCudaErrors(cudaGraphAddEventRecordNode(&computeEndNode, sg, parents.data(),
                                                            numParents, ev_compute_end[task]));
                // rewire signalNode to depend on computeEndNode instead of its current parents
                for (auto &parent : parents)
                    MUSTARD_cudaGraphRemoveDependencies(sg, &parent, &signalNode, 1);
                MUSTARD_cudaGraphAddDependencies(sg, &computeEndNode, &signalNode, 1);
            }
        }
    }

    // Instantiate and upload owned subgraphs
    cudaGraphExec_t *h_subgraphsExec = new cudaGraphExec_t[totalNodes];
    for (int task : pe_tasks[myPE])
    {
        if (dot)
        {
            char filename[32];
            sprintf(filename, "./graph_%d_%d.dot", task, myPE);
            checkCudaErrors(
                cudaGraphDebugDotPrint(tiledLUGraphCreator->subgraphs[task], filename, 0));
        }
        checkCudaErrors(cudaGraphInstantiate(
            &h_subgraphsExec[task], tiledLUGraphCreator->subgraphs[task], nullptr, nullptr, 0));
        cudaGraphUpload(h_subgraphsExec[task], s);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    if (!cfg.invocationPath.empty())
        tiledLUGraphCreator->printInvocations(cfg.invocationPath, myPE);

    auto   setup_end  = std::chrono::high_resolution_clock::now();
    double setup_time = std::chrono::duration<double>(setup_end - setup_start).count();
    printf("device %d | Setup time (s): %4.4f\n", myPE, setup_time);

    // Storage for per-task timings: [run][task_idx]
    struct TaskTiming
    {
        float wait_ms    = 0.0f;
        float compute_ms = 0.0f;
    };
    int                                  numMyTasks = (int)my_tasks_sorted.size();
    std::vector<std::vector<TaskTiming>> all_timings(runs, std::vector<TaskTiming>(numMyTasks));

    double totalTime = 0.0;

    for (int i = 0; i < runs; i++)
    {
        checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double),
                                   cudaMemcpyHostToDevice));

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

        // Reference event for wait time measurement (option B: single ref before all launches)
        cudaEvent_t ev_ref = nullptr;
        if (measure_wait)
        {
            checkCudaErrors(cudaEventCreate(&ev_ref));
            checkCudaErrors(cudaEventRecord(ev_ref, taskStreams[0]));
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        for (int idx = 0; idx < numMyTasks; idx++)
        {
            int task = my_tasks_sorted[idx];
            checkCudaErrors(cudaGraphLaunch(h_subgraphsExec[task], taskStreams[idx % numStreams]));
        }
        for (int si = 0; si < numStreams; si++)
            checkCudaErrors(cudaStreamSynchronize(taskStreams[si]));
        checkCudaErrors(cudaDeviceSynchronize());
        auto t_end = std::chrono::high_resolution_clock::now();

        // Collect per-task timings after all GPU work is done
        if (measure_wait || measure_compute)
        {
            for (int idx = 0; idx < numMyTasks; idx++)
            {
                int         task = my_tasks_sorted[idx];
                auto       &deps = tiledLUGraphCreator->subgraphDependencies[task];
                TaskTiming &tt   = all_timings[i][idx];

                if (measure_wait)
                {
                    if (!deps.empty())
                        checkCudaErrors(
                            cudaEventElapsedTime(&tt.wait_ms, ev_ref, ev_compute_start[task]));
                    else
                        tt.wait_ms = 0.0f;
                }
                if (measure_compute)
                {
                    checkCudaErrors(cudaEventElapsedTime(&tt.compute_ms, ev_compute_start[task],
                                                         ev_compute_end[task]));
                }
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

    // Print CSV after all runs
    if (measure_wait || measure_compute)
    {
        printf("pe,run,task_id,op_name");
        if (measure_wait) printf(",wait_ms");
        if (measure_compute) printf(",compute_ms");
        printf("\n");

        for (int i = 0; i < runs; i++)
        {
            for (int idx = 0; idx < numMyTasks; idx++)
            {
                int task = my_tasks_sorted[idx];
                printf("%d,%d,%d,%s", myPE, i, task,
                       tiledLUGraphCreator->subgraphOpNames[task].c_str());
                if (measure_wait) printf(",%.4f", all_timings[i][idx].wait_ms);
                if (measure_compute) printf(",%.4f", all_timings[i][idx].compute_ms);
                printf("\n");
            }
        }
        fflush(stdout);

        // Cleanup events
        for (int task : pe_tasks[myPE])
        {
            if (ev_compute_start[task]) checkCudaErrors(cudaEventDestroy(ev_compute_start[task]));
            if (ev_compute_end[task]) checkCudaErrors(cudaEventDestroy(ev_compute_end[task]));
        }
    }

    if (verify && myPE == 0)
    {
        double *h_L = (double *)malloc(N * N * sizeof(double));
        double *h_U = (double *)malloc(N * N * sizeof(double));
        checkCudaErrors(cudaMemcpy(h_L, d_matrix, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        memset(h_U, 0, N * N * sizeof(double));
        cleanCusolverLUDecompositionResult(h_L, h_U, N);
        printf("Result passes verification: %d\n",
               verifyLUDecomposition(originalMatrix.get(), h_L, h_U, N, verbose));
        free(h_L);
        free(h_U);
    }

    // Cleanup
    for (int task : pe_tasks[myPE])
    {
        if (d_task_deps[task]) checkCudaErrors(cudaFree(d_task_deps[task]));
        if (d_task_notify_pes[task]) checkCudaErrors(cudaFree(d_task_notify_pes[task]));
    }
    delete[] h_subgraphsExec;
    nvshmem_free(d_completion_flags);
    nvshmem_free(d_matrices);
    nvshmem_free((void *)d_flags);
    checkCudaErrors(cudaFree(d_info));
    checkCudaErrors(cudaFree(d_workspace_cusolver));
    for (int i = 0; i < workspaces; i++) checkCudaErrors(cudaFree(d_workspace_cublas[i]));
    free(d_workspace_cublas);
}

void LU(bool tiled, bool verify, bool subgraph, bool staticMultiGPU, bool dot)
{
    if (staticMultiGPU)
        tiledLUStatic(verify, dot);
    else if (tiled && myPE == 0)
        tiledLU(verify, subgraph, dot);
    else if (subgraph)
        tiledLU(verify, subgraph, dot);
    else if (myPE == 0)
        trivialLU(verify);
}

int main(int argc, char **argv)
{
    auto wall_start    = std::chrono::system_clock::now();
    auto program_start = std::chrono::high_resolution_clock::now();

    auto cmdl = argh::parser(argc, argv);

    if (!parseCommonArgs(cmdl, cfg))
    {
        printSingleNodeUsage(argv[0], "LU");
        return 1;
    }

    auto init_start = std::chrono::high_resolution_clock::now();
    initNvshmemDevice(cmdl, cfg);
    auto init_end = std::chrono::high_resolution_clock::now();
    myPE          = cfg.myPE;
    if (myPE == 0)
    {
        double unix_start = std::chrono::duration<double>(wall_start.time_since_epoch()).count();
        printf("Program start timestamp: %.6f\n", unix_start);
    }
    double init_time = std::chrono::duration<double>(init_end - init_start).count();
    printf("device %d | NVSHMEM init time (s): %4.4f\n", myPE, init_time);

    if (!(cmdl["tiled"] || cmdl["subgraph"] || cmdl["static-multigpu"])) T = 1;
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

    LU(cmdl["tiled"], cmdl["verify"] && myPE == 0, cmdl["subgraph"], cmdl["static-multigpu"],
       cmdl["dot"]);

    nvshmem_finalize();

    auto   program_end  = std::chrono::high_resolution_clock::now();
    double program_time = std::chrono::duration<double>(program_end - program_start).count();
    printf("device %d | Total program time (s): %4.4f\n", myPE, program_time);
    if (myPE == 0)
    {
        auto   wall_end = std::chrono::system_clock::now();
        double unix_end = std::chrono::duration<double>(wall_end.time_since_epoch()).count();
        printf("Program end timestamp: %.6f\n", unix_end);
    }

    return 0;
}