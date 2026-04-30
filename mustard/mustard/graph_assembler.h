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

class PanelGraph
{
   public:
    std::unique_ptr<mustard::TiledGraphCreator> creator;

    PanelGraph(cudaStream_t s, cudaGraph_t graph, int totalNodes) : s_(s)
    {
        creator = std::make_unique<mustard::TiledGraphCreator>(s, graph, true, totalNodes);
    }

    virtual ~PanelGraph() = default;

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
    cudaStream_t s_;
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

// owns panels, cublas/cusolver handles, and all workspace memory
class CholeskyCudaOperations
{
   public:
    CholeskyCudaOperations(cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle,
                           double* d_workspace_cusolver, int workspaceInBytesOnDevice,
                           void** d_workspace_cublas, int cublasWorkspaceSize, int numWorkspaces,
                           int* d_info, StridedDevicePanels panels, int B, int N, int nPEs)
        : cublasHandle_(cublasHandle),
          cusolverHandle_(cusolverHandle),
          d_workspace_cusolver_(d_workspace_cusolver),
          workspaceInBytesOnDevice_(workspaceInBytesOnDevice),
          d_workspace_cublas_(d_workspace_cublas),
          cublasWorkspaceSize_(cublasWorkspaceSize),
          numWorkspaces_(numWorkspaces),
          d_info_(d_info),
          panels_(std::move(panels)),
          B_(B),
          N_(N),
          nPEs_(nPEs),
          one_(1.0),
          minusOne_(-1.0)
    {
    }

    ~CholeskyCudaOperations()
    {
        cublasDestroy(cublasHandle_);
        cusolverDnDestroy(cusolverHandle_);
        cudaFree(d_workspace_cusolver_);
        for (int i = 0; i < numWorkspaces_; i++) cudaFree(d_workspace_cublas_[i]);
        free(d_workspace_cublas_);
        cudaFree(d_info_);
    }

    CholeskyCudaOperations(const CholeskyCudaOperations&)            = delete;
    CholeskyCudaOperations& operator=(const CholeskyCudaOperations&) = delete;

    StridedDevicePanel& myPanel() { return panels_.myPanel(); }

    void setWorkspace()
    {
        checkCudaErrors(
            cublasSetWorkspace(cublasHandle_, d_workspace_cublas_[++wsIdx_], cublasWorkspaceSize_));
    }

    void doSYRK(int i, int k)
    {
        checkCudaErrors(cublasDsyrk(cublasHandle_, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, B_, B_,
                                    &minusOne_, panels_.otherPanel(k % nPEs_).tile(i, k), N_, &one_,
                                    panels_.myPanel().tile(i, i), N_));
    }

    void doGEMM(int j, int i, int k)
    {
        checkCudaErrors(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_T, B_, B_, B_,
                                     &minusOne_, panels_.otherPanel(k % nPEs_).tile(j, k),
                                     CUDA_R_64F, N_, panels_.otherPanel(k % nPEs_).tile(i, k),
                                     CUDA_R_64F, N_, &one_, panels_.myPanel().tile(j, i),
                                     CUDA_R_64F, N_, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
    }

    void doPOTRF(int k)
    {
        checkCudaErrors(cusolverDnDpotrf(cusolverHandle_, CUBLAS_FILL_MODE_LOWER, B_,
                                         panels_.myPanel().tile(k, k), N_, d_workspace_cusolver_,
                                         workspaceInBytesOnDevice_, d_info_));
    }

    void doTRSM(int i, int k)
    {
        checkCudaErrors(cublasDtrsm(cublasHandle_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, B_, B_, &one_,
                                    panels_.myPanel().tile(k, k), N_, panels_.myPanel().tile(i, k),
                                    N_));
    }

   private:
    cublasHandle_t      cublasHandle_;
    cusolverDnHandle_t  cusolverHandle_;
    double*             d_workspace_cusolver_;
    int                 workspaceInBytesOnDevice_;
    void**              d_workspace_cublas_;
    int                 cublasWorkspaceSize_;
    int                 numWorkspaces_;
    int*                d_info_;
    StridedDevicePanels panels_;
    int                 B_, N_, nPEs_;
    int                 wsIdx_ = -1;
    double              one_, minusOne_;
};

class CholeskyPanelGraph : public PanelGraph
{
   public:
    CholeskyPanelGraph(cudaStream_t s, cudaGraph_t graph, int totalNodes, size_t T, int myPE,
                       int nPEs, CholeskyCudaOperations& ops, mustard::OccupancyTracker& occupancy)
        : PanelGraph(s, graph, totalNodes), T_(T), myPE_(myPE), nPEs_(nPEs), ops_(ops),
          occupancy_(occupancy)
    {
    }

    void assemble() override
    {
        for (int k = 0; k < (int)T_; k++)
        {
            if (k % nPEs_ == myPE_)
            {
                assemblePOTRFNode(k);
                for (int i = k + 1; i < (int)T_; i++) assembleTRSMNode(i, k);
            }
            else
                creator->skip(T_ - k);  // 1 POTRF + (T-k-1) TRSMs

            for (int i = k + 1; i < (int)T_; i++)
            {
                if (i % nPEs_ == myPE_)
                {
                    assembleSYRKNode(i, k);
                    for (int j = i + 1; j < (int)T_; j++) assembleGEMMNode(j, i, k);
                }
                else
                    creator->skip(T_ - i);  // 1 SYRK + (T-i-1) GEMMs
            }
        }
    }

   private:
    size_t                     T_;
    int                        myPE_, nPEs_;
    CholeskyCudaOperations&    ops_;
    mustard::OccupancyTracker& occupancy_;

    template <typename F>
    void captureNode(std::pair<int, int> write, std::vector<std::pair<int, int>> reads,
                     const std::string& name, bool occupancy, F&& work)
    {
        creator->beginCaptureOperation(write, reads, name);
        ops_.setWorkspace();
        if (occupancy) occupancy_.incrementOccupancy(s_);
        work();
        if (occupancy) occupancy_.decrementOccupancy(s_);
        creator->endCaptureOperation();
    }

    void assemblePOTRFNode(int k)
    {
        captureNode({k, k}, {{k, k}}, opName("POTRF", k, k), false, [&] { ops_.doPOTRF(k); });
    }

    void assembleTRSMNode(int i, int k)
    {
        captureNode({i, k}, {{k, k}, {i, k}}, opName("TRSM", i, k), true,
                    [&] { ops_.doTRSM(i, k); });
    }

    void assembleSYRKNode(int i, int k)
    {
        captureNode({i, i}, {{i, i}, {i, k}}, opName("SYRK", i, i, k), true,
                    [&] { ops_.doSYRK(i, k); });
    }

    void assembleGEMMNode(int j, int i, int k)
    {
        captureNode({j, i}, {{j, i}, {j, k}, {i, k}}, opName("GEMM", j, i, k), true,
                    [&] { ops_.doGEMM(j, i, k); });
    }
};
