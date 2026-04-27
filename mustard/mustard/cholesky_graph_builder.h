#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <memory>
#include <string>

#include "mustard.h"
#include "utils.h"

struct TiledCholeskyBuildContext
{
    cudaStream_t       s;
    cusolverDnHandle_t cusolverHandle;
    cublasHandle_t     cublasHandle;
    double*            d_matrix;
    double*            d_matrix_remote;
    double*            d_workspace_cusolver;
    void**             d_workspace_cublas;
    int*               d_info;
    volatile int*      d_flags;
    int                workspaceInBytesOnDevice;
    int                cublasWorkspaceSize;
    size_t             N;
    size_t             B;
    size_t             T;
    int                smLimit;
    int                myPE;
    int                totalNodes;
};

class TiledCholeskyGraphBuilder
{
public:
    std::unique_ptr<mustard::TiledGraphCreator> creator;

    TiledCholeskyGraphBuilder(const TiledCholeskyBuildContext& ctx, cudaGraph_t graph)
        : ctx(ctx), one(1.0), minusOne(-1.0)
    {
        creator = std::make_unique<mustard::TiledGraphCreator>(ctx.s, graph, true, ctx.totalNodes);
    }

    virtual ~TiledCholeskyGraphBuilder() = default;

    virtual void build()
    {
        for (int k = 0; k < (int)ctx.T; k++)
        {
            checkCudaErrors(
                cublasSetWorkspace(ctx.cublasHandle, ctx.d_workspace_cublas[0], ctx.cublasWorkspaceSize));
            creator->beginCaptureOperation(
                std::make_pair(k, k), {std::make_pair(k, k)},
                "POTRF(" + std::to_string(k) + "," + std::to_string(k) + ")");
            mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(ctx.smLimit, ctx.d_flags);
            if (ctx.myPE != 0)
                cudaMemcpy2DAsync(tile(ctx.d_matrix, k, k), sizeof(double) * ctx.N,
                                  tile(ctx.d_matrix_remote, k, k), sizeof(double) * ctx.N,
                                  sizeof(double) * ctx.B, ctx.B, cudaMemcpyDeviceToDevice, ctx.s);
            doPOTRF(k);
            if (ctx.myPE != 0)
                cudaMemcpy2DAsync(tile(ctx.d_matrix_remote, k, k), sizeof(double) * ctx.N,
                                  tile(ctx.d_matrix, k, k), sizeof(double) * ctx.N,
                                  sizeof(double) * ctx.B, ctx.B, cudaMemcpyDeviceToDevice, ctx.s);
            mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(-ctx.smLimit, ctx.d_flags);
            creator->endCaptureOperation();

            for (int i = k + 1; i < (int)ctx.T; i++)
            {
                checkCudaErrors(cublasSetWorkspace(ctx.cublasHandle, ctx.d_workspace_cublas[i],
                                                   ctx.cublasWorkspaceSize));
                creator->beginCaptureOperation(
                    std::make_pair(i, k), {std::make_pair(k, k), std::make_pair(i, k)},
                    "TRSM(" + std::to_string(i) + "," + std::to_string(k) + ")");
                mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(ctx.smLimit, ctx.d_flags);
                if (ctx.myPE != 0 && k != 0)
                    cudaMemcpy2DAsync(tile(ctx.d_matrix, i, k), sizeof(double) * ctx.N,
                                      tile(ctx.d_matrix_remote, i, k), sizeof(double) * ctx.N,
                                      sizeof(double) * ctx.B, ctx.B, cudaMemcpyDeviceToDevice, ctx.s);
                if (ctx.myPE != 0)
                    cudaMemcpy2DAsync(tile(ctx.d_matrix, k, k), sizeof(double) * ctx.N,
                                      tile(ctx.d_matrix_remote, k, k), sizeof(double) * ctx.N,
                                      sizeof(double) * ctx.B, ctx.B, cudaMemcpyDeviceToDevice, ctx.s);
                doTRSM(i, k);
                if (ctx.myPE != 0)
                    cudaMemcpy2DAsync(tile(ctx.d_matrix_remote, i, k), sizeof(double) * ctx.N,
                                      tile(ctx.d_matrix, i, k), sizeof(double) * ctx.N,
                                      sizeof(double) * ctx.B, ctx.B, cudaMemcpyDeviceToDevice, ctx.s);
                mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(-ctx.smLimit, ctx.d_flags);
                creator->endCaptureOperation();
            }

            for (int i = k + 1; i < (int)ctx.T; i++)
            {
                checkCudaErrors(cublasSetWorkspace(ctx.cublasHandle, ctx.d_workspace_cublas[i + ctx.T],
                                                   ctx.cublasWorkspaceSize));
                creator->beginCaptureOperation(
                    std::make_pair(i, i), {std::make_pair(i, i), std::make_pair(i, k)},
                    "SYRK(" + std::to_string(i) + "," + std::to_string(i) + "," +
                        std::to_string(k) + ")");
                mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(ctx.smLimit, ctx.d_flags);
                if (ctx.myPE != 0)
                {
                    cudaMemcpy2DAsync(tile(ctx.d_matrix, i, k), sizeof(double) * ctx.N,
                                      tile(ctx.d_matrix_remote, i, k), sizeof(double) * ctx.N,
                                      sizeof(double) * ctx.B, ctx.B, cudaMemcpyDeviceToDevice, ctx.s);
                    cudaMemcpy2DAsync(tile(ctx.d_matrix, i, i), sizeof(double) * ctx.N,
                                      tile(ctx.d_matrix_remote, i, i), sizeof(double) * ctx.N,
                                      sizeof(double) * ctx.B, ctx.B, cudaMemcpyDeviceToDevice, ctx.s);
                }
                doSYRK(i, k);
                if (ctx.myPE != 0)
                    cudaMemcpy2DAsync(tile(ctx.d_matrix_remote, i, i), sizeof(double) * ctx.N,
                                      tile(ctx.d_matrix, i, i), sizeof(double) * ctx.N,
                                      sizeof(double) * ctx.B, ctx.B, cudaMemcpyDeviceToDevice, ctx.s);
                mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(-ctx.smLimit, ctx.d_flags);
                creator->endCaptureOperation();

                for (int j = i + 1; j < (int)ctx.T; j++)
                {
                    checkCudaErrors(cublasSetWorkspace(
                        ctx.cublasHandle,
                        ctx.d_workspace_cublas[2 * ctx.T + (i - 1) * ctx.T + (j - 1)],
                        ctx.cublasWorkspaceSize));
                    creator->beginCaptureOperation(
                        std::make_pair(j, i),
                        {std::make_pair(j, i), std::make_pair(j, k), std::make_pair(i, k)},
                        "GEMM(" + std::to_string(j) + "," + std::to_string(i) + "," +
                            std::to_string(k) + ")");
                    mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(ctx.smLimit, ctx.d_flags);
                    if (ctx.myPE != 0)
                    {
                        cudaMemcpy2DAsync(tile(ctx.d_matrix, i, k), sizeof(double) * ctx.N,
                                          tile(ctx.d_matrix_remote, i, k), sizeof(double) * ctx.N,
                                          sizeof(double) * ctx.B, ctx.B, cudaMemcpyDeviceToDevice, ctx.s);
                        cudaMemcpy2DAsync(tile(ctx.d_matrix, j, k), sizeof(double) * ctx.N,
                                          tile(ctx.d_matrix_remote, j, k), sizeof(double) * ctx.N,
                                          sizeof(double) * ctx.B, ctx.B, cudaMemcpyDeviceToDevice, ctx.s);
                        cudaMemcpy2DAsync(tile(ctx.d_matrix, j, i), sizeof(double) * ctx.N,
                                          tile(ctx.d_matrix_remote, j, i), sizeof(double) * ctx.N,
                                          sizeof(double) * ctx.B, ctx.B, cudaMemcpyDeviceToDevice, ctx.s);
                    }
                    doGEMM(j, i, k);
                    if (ctx.myPE != 0)
                        cudaMemcpy2DAsync(tile(ctx.d_matrix_remote, j, i), sizeof(double) * ctx.N,
                                          tile(ctx.d_matrix, j, i), sizeof(double) * ctx.N,
                                          sizeof(double) * ctx.B, ctx.B, cudaMemcpyDeviceToDevice, ctx.s);
                    mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(-ctx.smLimit, ctx.d_flags);
                    creator->endCaptureOperation();
                }
            }
        }
    }

protected:
    TiledCholeskyBuildContext ctx;
    double                    one;
    double                    minusOne;

    double* tile(double* matrix, int i, int j)
    {
        return matrix + i * ctx.B + j * ctx.B * ctx.N;
    }

    virtual void doPOTRF(int k)
    {
        checkCudaErrors(cusolverDnDpotrf(ctx.cusolverHandle, CUBLAS_FILL_MODE_LOWER, ctx.B,
                                         tile(ctx.d_matrix, k, k), ctx.N,
                                         ctx.d_workspace_cusolver, ctx.workspaceInBytesOnDevice,
                                         ctx.d_info));
    }

    virtual void doTRSM(int i, int k)
    {
        checkCudaErrors(cublasDtrsm(ctx.cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, ctx.B, ctx.B, &one,
                                    tile(ctx.d_matrix, k, k), ctx.N,
                                    tile(ctx.d_matrix, i, k), ctx.N));
    }

    virtual void doSYRK(int i, int k)
    {
        checkCudaErrors(cublasDsyrk(ctx.cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                    ctx.B, ctx.B, &minusOne,
                                    tile(ctx.d_matrix, i, k), ctx.N, &one,
                                    tile(ctx.d_matrix, i, i), ctx.N));
    }

    virtual void doGEMM(int j, int i, int k)
    {
        checkCudaErrors(cublasGemmEx(ctx.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                     ctx.B, ctx.B, ctx.B, &minusOne,
                                     tile(ctx.d_matrix, j, k), CUDA_R_64F, ctx.N,
                                     tile(ctx.d_matrix, i, k), CUDA_R_64F, ctx.N, &one,
                                     tile(ctx.d_matrix, j, i), CUDA_R_64F, ctx.N,
                                     CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
    }
};

// Repeats each compute kernel `repeat` times inside the captured subgraph.
// The result after execution is incorrect (no save/restore between passes),
// but GPU occupancy and runtime scale linearly with repeat.
class RepeatingTiledCholeskyGraphBuilder : public TiledCholeskyGraphBuilder
{
public:
    RepeatingTiledCholeskyGraphBuilder(const TiledCholeskyBuildContext& ctx, cudaGraph_t graph,
                                       int repeat)
        : TiledCholeskyGraphBuilder(ctx, graph), repeat(repeat)
    {
    }

protected:
    void doPOTRF(int k) override
    {
        for (int r = 0; r < repeat; r++) TiledCholeskyGraphBuilder::doPOTRF(k);
    }

    void doTRSM(int i, int k) override
    {
        for (int r = 0; r < repeat; r++) TiledCholeskyGraphBuilder::doTRSM(i, k);
    }

    void doSYRK(int i, int k) override
    {
        for (int r = 0; r < repeat; r++) TiledCholeskyGraphBuilder::doSYRK(i, k);
    }

    void doGEMM(int j, int i, int k) override
    {
        for (int r = 0; r < repeat; r++) TiledCholeskyGraphBuilder::doGEMM(j, i, k);
    }

private:
    int repeat;
};
