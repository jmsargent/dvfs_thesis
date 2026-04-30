#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cassert>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "StridedPanel.h"
#include "mustard.h"
#include "utils.h"

class HasLocation
{
   public:
    virtual void setCoordinate(std::vector<int> coords) = 0;
    virtual ~HasLocation()                              = default;
};

template <typename Ops>
class DAGBuilder
{
    struct Node
    {
        int                   owner;
        std::function<void()> op;
        std::vector<int>      waitTaskIds;
        std::vector<int>      signalPEs;
    };

    std::vector<Node>           nodes_;
    int                         myPE_;
    mustard::TiledGraphCreator& creator_;
    Ops&                        ops_;

   public:
    class NodeRef
    {
        Node& node_;

       public:
        NodeRef(Node& n) : node_(n) {}
        NodeRef& waitsFor(std::vector<int> ids) { node_.waitTaskIds = std::move(ids); return *this; }
        NodeRef& signals(std::vector<int> pes)  { node_.signalPEs   = std::move(pes);  return *this; }
    };

    class NodeBuilder
    {
        DAGBuilder&      dag_;
        std::vector<int> coords_;

       public:
        NodeBuilder(DAGBuilder& dag, std::vector<int> coords)
            : dag_(dag), coords_(std::move(coords)) {}

        template <typename F>
        NodeRef add(int owner, F&& f)
        {
            auto& ops = dag_.ops_;
            auto  c   = coords_;
            dag_.nodes_.push_back({owner, [&ops, c, f = std::forward<F>(f)]() mutable {
                ops.setCoordinate(c);
                f(ops);
            }, {}, {}});
            return NodeRef(dag_.nodes_.back());
        }
    };

    DAGBuilder(int myPE, mustard::TiledGraphCreator& creator, Ops& ops)
        : myPE_(myPE), creator_(creator), ops_(ops) {}

    template <typename... Args>
    NodeBuilder at(Args... args)
    {
        return NodeBuilder(*this, {args...});
    }

    void buildFor()
    {
        // idk if want to change this to phantom operation , then we could somehow point to the creator whenever there is a dependency
        for (auto& n : nodes_) n.owner == myPE_ ? n.op() : creator_.skip();
    }
};

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

class CholeskyPanelGraph : public PanelGraph, public HasLocation
{
   public:
    CholeskyPanelGraph(cudaStream_t s, cudaGraph_t graph, int totalNodes, size_t T, int myPE,
                       int nPEs, CholeskyCudaOperations& ops, mustard::OccupancyTracker& occupancy)
        : PanelGraph(s, graph, totalNodes),
          T_(T),
          myPE_(myPE),
          nPEs_(nPEs),
          ops_(ops),
          occupancy_(occupancy)
    {
    }

    void setCoordinate(std::vector<int> c) override { coords_ = std::move(c); }

    void assemble() override
    {
        // These dependencies are wierd, lol
        DAGBuilder<CholeskyPanelGraph> dag(myPE_, *creator, *this);
        
        for (int pivotColumn = 0; pivotColumn < (int)T_; pivotColumn++)
        {
            dag.at(pivotColumn).add(pivotColumn % nPEs_, [](auto& ops) { ops.potrf(); });

            for (int column = pivotColumn + 1; column < (int)T_; column++)
                dag.at(column, pivotColumn).add(pivotColumn % nPEs_, [](auto& ops) { ops.trsm(); })
                   .signals(signalPEs(column, pivotColumn));

            for (int column = pivotColumn + 1; column < (int)T_; column++)
            {
                dag.at(column, pivotColumn)
                   .add(column % nPEs_, [](auto& ops) { ops.syrk(); })
                   .waitsFor(writeAt(column, pivotColumn));

                for (int row = column + 1; row < (int)T_; row++)
                    dag.at(row, column, pivotColumn)
                       .add(column % nPEs_, [](auto& ops) { ops.gemm(); })
                       .waitsFor(writeAt(row, column, pivotColumn));
            }
        }
        dag.buildFor();
    }

   private:
    size_t                     T_;
    int                        myPE_, nPEs_;
    CholeskyCudaOperations&    ops_;
    mustard::OccupancyTracker& occupancy_;
    std::vector<int>           coords_;

    void potrf()
    {
        assert(coords_.size() == 1 && "potrf: expected 1 coordinate from at()");
        int k = coords_[0];
        captureNode({k, k}, {{k, k}}, opName("POTRF", k), false, [&] { ops_.doPOTRF(k); });
    }

    void trsm()
    {
        assert(coords_.size() == 2 && "trsm: expected 2 coordinates from at()");
        int i = coords_[0], k = coords_[1];
        captureNode({i, k}, {{k, k}, {i, k}}, opName("TRSM", i, k), true, [&] { ops_.doTRSM(i, k); });
    }

    void syrk()
    {
        assert(coords_.size() == 2 && "syrk: expected 2 coordinates from at()");
        int i = coords_[0], k = coords_[1];
        captureNode({i, i}, {{i, i}, {i, k}}, opName("SYRK", i, i, k), true, [&] { ops_.doSYRK(i, k); });
    }

    void gemm()
    {
        assert(coords_.size() == 3 && "gemm: expected 3 coordinates from at()");
        int j = coords_[0], i = coords_[1], k = coords_[2];
        captureNode({j, i}, {{j, i}, {j, k}, {i, k}}, opName("GEMM", j, i, k), true,
                    [&] { ops_.doGEMM(j, i, k); });
    }

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

    int trsmFlagIdx(int column, int pivotColumn)
    {
        return pivotColumn * (2 * (int)T_ - pivotColumn - 1) / 2 + (column - pivotColumn - 1);
    }

    std::vector<int> signalPEs(int column, int pivotColumn)
    {
        std::set<int> pes;
        if (column % nPEs_ != pivotColumn % nPEs_)
            pes.insert(column % nPEs_);
        for (int p = pivotColumn + 1; p < column; p++)
            if (p % nPEs_ != pivotColumn % nPEs_)
                pes.insert(p % nPEs_);
        return {pes.begin(), pes.end()};
    }

    std::vector<int> writeAt(int column, int pivotColumn)
    {
        if (pivotColumn % nPEs_ == column % nPEs_) return {};
        return {trsmFlagIdx(column, pivotColumn)};
    }

    std::vector<int> writeAt(int row, int column, int pivotColumn)
    {
        if (pivotColumn % nPEs_ == column % nPEs_) return {};
        return {trsmFlagIdx(column, pivotColumn), trsmFlagIdx(row, pivotColumn)};
    }
};
