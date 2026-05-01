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


struct TileAccess
{
    std::string             name;
    MatrixTile              write;
    std::vector<MatrixTile> reads;
    std::function<void()>   work;
};

template <typename Ops>
class DAGBuilder
{
    struct Node
    {
        int              owner;
        TileAccess       tiles;
        std::vector<int> dependsOn;

        Node(int o, TileAccess t, std::vector<int> d)
            : owner(o), tiles(std::move(t)), dependsOn(std::move(d))
        {
        }

        void findRAWDependencies(const std::map<MatrixTile, int>& tileToLastWriter)
        {
            for (const auto& readTile : tiles.reads)
            {
                auto it = tileToLastWriter.find(readTile);
                if (it != tileToLastWriter.end()) dependsOn.push_back(it->second);
            }
        }
    };

    std::vector<Node>           nodes_;
    std::map<MatrixTile, int>   lastWriterByTile_;
    int                         myPE_;
    mustard::TiledGraphCreator& creator_;
    Ops&                        ops_;
    int*                        completionFlags_;
    cudaStream_t                stream_;

   public:
    DAGBuilder(int myPE, mustard::TiledGraphCreator& creator, Ops& ops, int* completionFlags,
               cudaStream_t stream)
        : myPE_(myPE), creator_(creator), ops_(ops), completionFlags_(completionFlags),
          stream_(stream)
    {
    }

    template <typename F>
    void add(int owner, F&& f)
    {
        TileAccess       access    = f(ops_);
        int              nodeIndex = (int)nodes_.size();
        std::vector<int> dependencies;

        Node node(owner, std::move(access), std::move(dependencies));
        node.findRAWDependencies(lastWriterByTile_);

        lastWriterByTile_[node.tiles.write] = nodeIndex;
        nodes_.push_back(node);
    }

    void build()
    {
        int nNodes = (int)nodes_.size();

        // For each node: which PEs need to be notified when it completes
        std::vector<std::vector<int>> notifyPEs(nNodes);
        for (int i = 0; i < nNodes; i++)
            for (int dep : nodes_[i].dependsOn)
                if (nodes_[i].owner != nodes_[dep].owner)
                    notifyPEs[dep].push_back(nodes_[i].owner);
        for (auto& v : notifyPEs)
        {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }

        // For each node: which task indices (on other PEs) does it need to wait for
        std::vector<std::vector<int>> crossPEDeps(nNodes);
        for (int i = 0; i < nNodes; i++)
            for (int dep : nodes_[i].dependsOn)
                if (nodes_[dep].owner != nodes_[i].owner)
                    crossPEDeps[i].push_back(dep);

        // Upload to device before any graph capture begins
        std::vector<int*> d_crossPEDeps(nNodes, nullptr);
        std::vector<int*> d_notifyPEs(nNodes, nullptr);
        for (int i = 0; i < nNodes; i++)
        {
            if (!crossPEDeps[i].empty())
            {
                cudaMalloc(&d_crossPEDeps[i], crossPEDeps[i].size() * sizeof(int));
                cudaMemcpy(d_crossPEDeps[i], crossPEDeps[i].data(),
                           crossPEDeps[i].size() * sizeof(int), cudaMemcpyHostToDevice);
            }
            if (!notifyPEs[i].empty())
            {
                cudaMalloc(&d_notifyPEs[i], notifyPEs[i].size() * sizeof(int));
                cudaMemcpy(d_notifyPEs[i], notifyPEs[i].data(),
                           notifyPEs[i].size() * sizeof(int), cudaMemcpyHostToDevice);
            }
        }

        for (int i = 0; i < nNodes; i++)
        {
            auto& n = nodes_[i];
            if (n.owner == myPE_)
            {
                creator_.beginCaptureOperation(n.tiles.write, n.tiles.reads, n.tiles.name);
                if (d_crossPEDeps[i])
                    mustard::kernel_wait_static<<<1, 1, 0, stream_>>>(d_crossPEDeps[i],
                                                                    (int)crossPEDeps[i].size(),
                                                                    completionFlags_, 0);
                n.tiles.work();
                if (d_notifyPEs[i])
                    mustard::kernel_signal_static<<<1, 1, 0, stream_>>>(i, completionFlags_,
                                                                       d_notifyPEs[i],
                                                                       (int)notifyPEs[i].size(), 0);
                creator_.endCaptureOperation();
            }
            else
            {
                creator_.phantomOperation(n.tiles.write, n.tiles.reads, n.tiles.name);
            }
        }
    }
};

class PanelGraph
{
   public:
    PanelGraph(cudaGraph_t graph, int totalNodes) {}

    virtual ~PanelGraph() = default;

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
};

// owns panels, cublas/cusolver handles, and all workspace memory
class CholeskyCudaOperations
{
   public:
    CholeskyCudaOperations(cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle,
                           double* d_workspace_cusolver, int workspaceInBytesOnDevice,
                           void** d_workspace_cublas, int cublasWorkspaceSize, int numWorkspaces,
                           int* d_info, StridedDevicePanels panels, int B, int N, int nPEs,
                           mustard::OccupancyTracker& occupancy)
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
          minusOne_(-1.0),
          occupancy_(occupancy)
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

    TileAccess potrf(int k)
    {
        return {mustard::opName("POTRF", k),
                {k, k},
                {{k, k}},
                [this, k]()
                {
                    checkCudaErrors(cusolverDnDpotrf(
                        cusolverHandle_, CUBLAS_FILL_MODE_LOWER, B_, panels_.myPanel().tile(k, k),
                        N_, d_workspace_cusolver_, workspaceInBytesOnDevice_, d_info_));
                }};
    }

    TileAccess trsm(int i, int k)
    {
        return {mustard::opName("TRSM", i, k),
                {i, k},
                {{k, k}, {i, k}},
                [this, i, k]()
                {
                    setWorkspace();
                    checkCudaErrors(cublasDtrsm(
                        cublasHandle_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                        CUBLAS_DIAG_NON_UNIT, B_, B_, &one_, panels_.myPanel().tile(k, k), N_,
                        panels_.myPanel().tile(i, k), N_));
                }};
    }

    TileAccess syrk(int i, int k)
    {
        return {mustard::opName("SYRK", i, i, k),
                {i, i},
                {{i, i}, {i, k}},
                [this, i, k]()
                {
                    setWorkspace();
                    checkCudaErrors(cublasDsyrk(cublasHandle_, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                                B_, B_, &minusOne_,
                                                panels_.otherPanel(k % nPEs_).tile(i, k), N_, &one_,
                                                panels_.myPanel().tile(i, i), N_));
                }};
    }

    TileAccess gemm(int j, int i, int k)
    {
        return {mustard::opName("GEMM", j, i, k),
                {j, i},
                {{j, i}, {j, k}, {i, k}},
                [this, j, i, k]()
                {
                    setWorkspace();
                    checkCudaErrors(
                        cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_T, B_, B_, B_,
                                     &minusOne_, panels_.otherPanel(k % nPEs_).tile(j, k),
                                     CUDA_R_64F, N_, panels_.otherPanel(k % nPEs_).tile(i, k),
                                     CUDA_R_64F, N_, &one_, panels_.myPanel().tile(j, i),
                                     CUDA_R_64F, N_, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
                }};
    }

   private:
    cublasHandle_t             cublasHandle_;
    cusolverDnHandle_t         cusolverHandle_;
    double*                    d_workspace_cusolver_;
    int                        workspaceInBytesOnDevice_;
    void**                     d_workspace_cublas_;
    int                        cublasWorkspaceSize_;
    int                        numWorkspaces_;
    int*                       d_info_;
    StridedDevicePanels        panels_;
    int                        B_, N_, nPEs_;
    int                        wsIdx_ = -1;
    double                     one_, minusOne_;
    mustard::OccupancyTracker& occupancy_;
};
