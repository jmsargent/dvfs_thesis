#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <map>
#include <memory>
#include <string>

#include "StridedPanel.h"
#include "mustard.h"
#include "utils.h"

struct TiledCholeskyBuildContext
{
    cudaStream_t       s;
    cusolverDnHandle_t cusolverHandle;
    cublasHandle_t     cublasHandle;
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
    int                nPEs = 0;
};

class PanelingGraphAssembler
{
   public:
    std::unique_ptr<mustard::TiledGraphCreator> creator;

    PanelingGraphAssembler(const TiledCholeskyBuildContext& ctx, cudaGraph_t graph,
                           StridedPanels& panels)
        : ctx(ctx), one(1.0), minusOne(-1.0), panels_(panels)
    {
        creator = std::make_unique<mustard::TiledGraphCreator>(ctx.s, graph, true, ctx.totalNodes);
    }

    virtual ~PanelingGraphAssembler() = default;

    virtual void assemble() = 0;

    static std::vector<std::vector<int>> buildDependencies(int T)
    {
        std::map<std::pair<int, int>, int> lastWriter;
        std::vector<std::vector<int>>      deps;
        int                                idx = 0;

        auto addTask = [&](std::pair<int, int> write, std::vector<std::pair<int, int>> reads)
        {
            std::vector<int> d;
            for (auto& r : reads)
                if (lastWriter.count(r)) d.push_back(lastWriter[r]);
            deps.push_back(d);
            lastWriter[write] = idx++;
        };

        for (int k = 0; k < T; k++)
        {
            addTask({k, k}, {{k, k}});
            for (int i = k + 1; i < T; i++) addTask({i, k}, {{k, k}, {i, k}});
            for (int i = k + 1; i < T; i++)
            {
                addTask({i, i}, {{i, i}, {i, k}});
                for (int j = i + 1; j < T; j++) addTask({j, i}, {{j, i}, {j, k}, {i, k}});
            }
        }
        return deps;
    }

    // Returns task-index → panel mapping matching the capture loop order.
    // POTRF(k,k) and TRSM(i,k) belong to panel k; SYRK/GEMM updating column i belong to panel i.
    static std::vector<int> buildTaskToPanel(int T)
    {
        std::vector<int> v;
        for (int k = 0; k < T; k++)
        {
            v.push_back(k);
            for (int i = k + 1; i < T; i++) v.push_back(k);
            for (int i = k + 1; i < T; i++)
            {
                v.push_back(i);
                for (int j = i + 1; j < T; j++) v.push_back(i);
            }
        }
        return v;
    }

    template <typename... Args>
    static std::string opName(const std::string& name, Args... args)
    {
        std::string s     = name + "(";
        bool        first = true;
        ((s += (first ? "" : ",") + std::to_string(args), first = false), ...);
        return s + ")";
    }

   protected:
    TiledCholeskyBuildContext ctx;
    double                    one;
    double                    minusOne;
    StridedPanels&            panels_;

    void setWorkspace(int idx) const
    {
        checkCudaErrors(cublasSetWorkspace(ctx.cublasHandle, ctx.d_workspace_cublas[idx],
                                           ctx.cublasWorkspaceSize));
    }
};

/*
    potrf(k):    reads [(k,k)]  writes [(k,k)]
    trsm(i,k):   reads [(k,k), (i,k)]  writes [(i,k)]
    syrk(i,k):   reads [(i,k), (i,i)]  writes [(i,i)]
    gemm(j,i,k): reads [(j,k), (i,k), (j,i)]  writes [(j,i)]

    on remote:

    for k = 0 to T-1:
        copyFromRemote(k,k)
        potrf(k, k)

        for i = k+1 to T-1:
            copyFromRemote(i,k)
            trsm(i, k)

        for i = k+1 to T-1:
            copyFromRemote(i,i)
            syrk(i, k)
            copyToRemote(i,i)

            for j = i+1 to T-1:
                copyFromRemote(j,i)
                gemm(j, i, k)
                copyToRemote(j,i)


    Each PE owns matrix columns nr: myPE,mePE+2*NPEs,...
    Each PE owns scratchpad
*/

class PanelGraphAssembler : public PanelingGraphAssembler
{
   public:
    PanelGraphAssembler(const TiledCholeskyBuildContext& ctx, cudaGraph_t graph,
                        StridedPanels& panels, const std::vector<int>& taskToPanel)
        : PanelingGraphAssembler(ctx, graph, panels), taskToPanel_(taskToPanel)
    {
    }

    void assemble() override
    {
        for (int k = 0; k < (int)ctx.T; k++)
        {
            if (k % ctx.nPEs == ctx.myPE)
            {
                assemblePOTRFNode(k);
                for (int i = k + 1; i < (int)ctx.T; i++) assembleTRSMNode(i, k);
            }
            else
                creator->skip(ctx.T - k);  // 1 POTRF + (T-k-1) TRSMs

            for (int i = k + 1; i < (int)ctx.T; i++)
            {
                if (i % ctx.nPEs == ctx.myPE)
                {
                    assembleSYRKNode(i, k);
                    for (int j = i + 1; j < (int)ctx.T; j++) assembleGEMMNode(j, i, k);
                }
                else
                    creator->skip(ctx.T - i);  // 1 SYRK + (T-i-1) GEMMs
            }
        }
    }

   private:
    std::vector<int> taskToPanel_;
    int              wsIdx = -1;

    void doSYRK(int i, int k)
    {
        checkCudaErrors(cublasDsyrk(ctx.cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, ctx.B,
                                    ctx.B, &minusOne, panels_.otherPanel(k % ctx.nPEs).tile(i, k),
                                    ctx.N, &one, panels_.myPanel().tile(i, i), ctx.N));
    }

    void doGEMM(int j, int i, int k)
    {
        checkCudaErrors(cublasGemmEx(
            ctx.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ctx.B, ctx.B, ctx.B, &minusOne,
            panels_.otherPanel(k % ctx.nPEs).tile(j, k), CUDA_R_64F, ctx.N,
            panels_.otherPanel(k % ctx.nPEs).tile(i, k), CUDA_R_64F, ctx.N,
            &one, panels_.myPanel().tile(j, i), CUDA_R_64F, ctx.N,
            CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
    }

    void doPOTRF(int k)
    {
        checkCudaErrors(cusolverDnDpotrf(ctx.cusolverHandle, CUBLAS_FILL_MODE_LOWER, ctx.B,
                                         panels_.myPanel().tile(k, k), ctx.N,
                                         ctx.d_workspace_cusolver, ctx.workspaceInBytesOnDevice,
                                         ctx.d_info));
    }

    void doTRSM(int i, int k)
    {
        checkCudaErrors(cublasDtrsm(ctx.cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, ctx.B, ctx.B, &one,
                                    panels_.myPanel().tile(k, k), ctx.N,
                                    panels_.myPanel().tile(i, k), ctx.N));
    }

    void assemblePOTRFNode(int k)
    {
        creator->beginCaptureOperation(std::make_pair(k, k), {std::make_pair(k, k)},
                                       opName("POTRF", k, k));
        setWorkspace(++wsIdx);
        doPOTRF(k);
        creator->endCaptureOperation();
    }

    void assembleTRSMNode(int i, int k)
    {
        creator->beginCaptureOperation(std::make_pair(i, k),
                                       {std::make_pair(k, k), std::make_pair(i, k)},
                                       opName("TRSM", i, k));
        setWorkspace(++wsIdx);
        mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(ctx.smLimit, ctx.d_flags);
        doTRSM(i, k);
        mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(-ctx.smLimit, ctx.d_flags);
        creator->endCaptureOperation();
    }

    void assembleSYRKNode(int i, int k)
    {
        creator->beginCaptureOperation(std::make_pair(i, i),
                                       {std::make_pair(i, i), std::make_pair(i, k)},
                                       opName("SYRK", i, i, k));
        setWorkspace(++wsIdx);
        mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(ctx.smLimit, ctx.d_flags);
        doSYRK(i, k);
        mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(-ctx.smLimit, ctx.d_flags);
        creator->endCaptureOperation();
    }

    void assembleGEMMNode(int j, int i, int k)
    {
        creator->beginCaptureOperation(
            std::make_pair(j, i),
            {std::make_pair(j, i), std::make_pair(j, k), std::make_pair(i, k)},
            opName("GEMM", j, i, k));
        setWorkspace(++wsIdx);
        mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(ctx.smLimit, ctx.d_flags);
        doGEMM(j, i, k);
        mustard::kernel_occupancy_update<<<1, 1, 0, ctx.s>>>(-ctx.smLimit, ctx.d_flags);
        creator->endCaptureOperation();
    }
};
