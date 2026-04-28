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

class CholeskyGraphBuilder
{
   public:
    std::unique_ptr<mustard::TiledGraphCreator> creator;

    CholeskyGraphBuilder(const TiledCholeskyBuildContext& ctx, cudaGraph_t graph)
        : ctx(ctx), one(1.0), minusOne(-1.0)
    {
        creator = std::make_unique<mustard::TiledGraphCreator>(ctx.s, graph, true, ctx.totalNodes);
    }

    virtual ~CholeskyGraphBuilder() = default;

   protected:
    TiledCholeskyBuildContext ctx;
    double                    one;
    double                    minusOne;

    double* tile(double* matrix, int i, int j) { return matrix + i * ctx.B + j * ctx.B * ctx.N; }
    double* tile(int i, int j) { return tile(ctx.d_matrix, i, j); }
    double* remoteTile(int i, int j) { return tile(ctx.d_matrix_remote, i, j); }
    void copyTile(double* dst, double* src, cudaMemcpyKind kind)
    {
        cudaMemcpy2DAsync(dst, sizeof(double) * ctx.N, src, sizeof(double) * ctx.N,
                          sizeof(double) * ctx.B, ctx.B, kind, ctx.s);
    }

    template<typename... Args>
    static std::string opName(const std::string& name, Args... args)
    {
        std::string s = name + "(";
        bool first = true;
        ((s += (first ? "" : ",") + std::to_string(args), first = false), ...);
        return s + ")";
    }

    virtual void doPOTRF(int k)
    {
        checkCudaErrors(cusolverDnDpotrf(ctx.cusolverHandle, CUBLAS_FILL_MODE_LOWER, ctx.B,
                                         tile(k, k), ctx.N, ctx.d_workspace_cusolver,
                                         ctx.workspaceInBytesOnDevice, ctx.d_info));
    }

    virtual void doTRSM(int i, int k)
    {
        checkCudaErrors(cublasDtrsm(ctx.cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, ctx.B, ctx.B, &one,
                                    tile(k, k), ctx.N, tile(i, k),
                                    ctx.N));
    }

    virtual void doSYRK(int i, int k)
    {
        checkCudaErrors(cublasDsyrk(ctx.cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, ctx.B,
                                    ctx.B, &minusOne, tile(i, k), ctx.N, &one,
                                    tile(i, i), ctx.N));
    }

    virtual void doGEMM(int j, int i, int k)
    {
        checkCudaErrors(cublasGemmEx(ctx.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ctx.B, ctx.B,
                                     ctx.B, &minusOne, tile(j, k), CUDA_R_64F, ctx.N,
                                     tile(i, k), CUDA_R_64F, ctx.N, &one,
                                     tile(j, i), CUDA_R_64F, ctx.N,
                                     CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
    }
};

class TiledCholeskyGraphBuilder : public CholeskyGraphBuilder
{
   public:
    TiledCholeskyGraphBuilder(const TiledCholeskyBuildContext& ctx, cudaGraph_t graph)
        : CholeskyGraphBuilder(ctx, graph)
    {
    }

    virtual void build()
    {
        for (int k = 0; k < (int)ctx.T; k++)
        {
            checkCudaErrors(cublasSetWorkspace(ctx.cublasHandle, ctx.d_workspace_cublas[0],
                                               ctx.cublasWorkspaceSize));
            creator->beginCaptureOperation(
                std::make_pair(k, k), {std::make_pair(k, k)},
                opName("POTR",k,k));
            mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(ctx.smLimit, ctx.d_flags);
            if (ctx.myPE != 0)
                copyTile(tile(k, k), remoteTile(k, k), cudaMemcpyDeviceToDevice);
            doPOTRF(k);
            if (ctx.myPE != 0)
                copyTile(remoteTile(k, k), tile(k, k), cudaMemcpyDeviceToDevice);
            mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(-ctx.smLimit, ctx.d_flags);
            creator->endCaptureOperation();

            for (int i = k + 1; i < (int)ctx.T; i++)
            {
                checkCudaErrors(cublasSetWorkspace(ctx.cublasHandle, ctx.d_workspace_cublas[i],
                                                   ctx.cublasWorkspaceSize));
                creator->beginCaptureOperation(
                    std::make_pair(i, k), {std::make_pair(k, k), std::make_pair(i, k)},
                    opName("TRSM",i,k));
                mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(ctx.smLimit, ctx.d_flags);
                if (ctx.myPE != 0 && k != 0)
                    copyTile(tile(i, k), remoteTile(i, k), cudaMemcpyDeviceToDevice);
                if (ctx.myPE != 0)
                    copyTile(tile(k, k), remoteTile(k, k), cudaMemcpyDeviceToDevice);
                doTRSM(i, k);
                if (ctx.myPE != 0)
                    copyTile(remoteTile(i, k), tile(i, k), cudaMemcpyDeviceToDevice);
                mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(-ctx.smLimit, ctx.d_flags);
                creator->endCaptureOperation();
            }

            for (int i = k + 1; i < (int)ctx.T; i++)
            {
                checkCudaErrors(cublasSetWorkspace(
                    ctx.cublasHandle, ctx.d_workspace_cublas[i + ctx.T], ctx.cublasWorkspaceSize));
                creator->beginCaptureOperation(
                    std::make_pair(i, i), {std::make_pair(i, i), std::make_pair(i, k)},
                    opName("SYRK",i,i,k));
                mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(ctx.smLimit, ctx.d_flags);
                if (ctx.myPE != 0)
                {
                    copyTile(tile(i, k), remoteTile(i, k), cudaMemcpyDeviceToDevice);
                    copyTile(tile(i, i), remoteTile(i, i), cudaMemcpyDeviceToDevice);
                }
                doSYRK(i, k);
                if (ctx.myPE != 0)
                    copyTile(remoteTile(i, i), tile(i, i), cudaMemcpyDeviceToDevice);
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
                        opName("GEMM",j,i,k));
                    mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(ctx.smLimit, ctx.d_flags);
                    if (ctx.myPE != 0)
                    {
                        copyTile(tile(i, k), remoteTile(i, k), cudaMemcpyDeviceToDevice);
                        copyTile(tile(j, k), remoteTile(j, k), cudaMemcpyDeviceToDevice);
                        copyTile(tile(j, i), remoteTile(j, i), cudaMemcpyDeviceToDevice);
                    }
                    doGEMM(j, i, k);
                    if (ctx.myPE != 0)
                        copyTile(remoteTile(j, i), tile(j, i), cudaMemcpyDeviceToDevice);
                    mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(-ctx.smLimit, ctx.d_flags);
                    creator->endCaptureOperation();
                }
            }
        }
    }
};

class PanelLocalGraphBuilder : public CholeskyGraphBuilder
{
   public:
    PanelLocalGraphBuilder(const TiledCholeskyBuildContext& ctx, cudaGraph_t graph)
        : CholeskyGraphBuilder(ctx, graph)
    {
    }

    virtual void build() {
        for (int k = 0; k < (int)ctx.T; k++)
        {

        }
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
        for (int r = 0; r < repeat; r++) CholeskyGraphBuilder::doPOTRF(k);
    }

    void doTRSM(int i, int k) override
    {
        for (int r = 0; r < repeat; r++) CholeskyGraphBuilder::doTRSM(i, k);
    }

    void doSYRK(int i, int k) override
    {
        for (int r = 0; r < repeat; r++) CholeskyGraphBuilder::doSYRK(i, k);
    }

    void doGEMM(int j, int i, int k) override
    {
        for (int r = 0; r < repeat; r++) CholeskyGraphBuilder::doGEMM(j, i, k);
    }

   private:
    int repeat;
};
