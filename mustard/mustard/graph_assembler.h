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

struct Node
{
    int                     index = -1;
    int                     partition = -1;
    std::string             name;
    MatrixTile              write;
    std::vector<MatrixTile> reads;
    std::function<void()>   work;
};

struct Edge
{
    int from;
    int to;
};

class PartitionedDag
{
   public:
    void addNode(Node n)
    {
        n.index = (int)nodes_.size();
        nodes_.push_back(std::move(n));
    }
    void addEdge(Edge e) { edges_.push_back(e); }

    std::vector<Node>&       nodes()       { return nodes_; }
    const std::vector<Node>& nodes() const { return nodes_; }

    std::vector<int> nodes(int partition) const
    {
        std::vector<int> result;
        for (auto& n : nodes_)
            if (n.partition == partition) result.push_back(n.index);
        return result;
    }

    std::vector<int> crossIncomingNodeIndices(const Node& n) const
    {
        std::vector<int> result;
        for (auto& e : edges_)
            if (e.to == n.index && nodes_[e.from].partition != n.partition)
                result.push_back(e.from);
        return result;
    }

    std::vector<int> crossOutgoingPartitions(const Node& n) const
    {
        std::set<int> result;
        for (auto& e : edges_)
            if (e.from == n.index && nodes_[e.to].partition != n.partition)
                result.insert(nodes_[e.to].partition);
        return {result.begin(), result.end()};
    }

   private:
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
};

struct MeasureFlags
{
    bool measureKernelWait   = false;
    bool measureKernelLaunch = false;
};

struct TimestampBuffers
{
    unsigned long long* d_waitTs = nullptr;
    unsigned long long* d_compTs = nullptr;
};

class OperationCapturer
{
   public:
    OperationCapturer(mustard::TiledGraphCreator& creator, int* completionFlags, cudaStream_t stream)
        : creator_(creator), completionFlags_(completionFlags), stream_(stream)
    {}

    static int* upload(const std::vector<int>& v)
    {
        if (v.empty()) return nullptr;
        int* d;
        cudaMalloc(&d, v.size() * sizeof(int));
        cudaMemcpy(d, v.data(), v.size() * sizeof(int), cudaMemcpyHostToDevice);
        return d;
    }

    void beginCapture(MatrixTile write, const std::vector<MatrixTile>& reads,
                      const std::string& name)
    {
        creator_.beginCaptureOperation(write, reads, name);
    }

    void launchWait(int* d_wait, int n)
    {
        mustard::kernel_wait_static<<<1, 1, 0, stream_>>>(d_wait, n, completionFlags_, 0);
    }

    void launchSignal(int nodeIndex, int* d_signal, int n)
    {
        mustard::kernel_signal_static<<<1, 1, 0, stream_>>>(nodeIndex, completionFlags_, d_signal,
                                                            n, 0);
    }

    void launchTimestamp(unsigned long long* ptr)
    {
        mustard::kernel_record_timestamp<<<1, 1, 0, stream_>>>(ptr);
    }

    void endCapture() { creator_.endCaptureOperation(); }

    void phantom(MatrixTile write, const std::vector<MatrixTile>& reads, const std::string& name)
    {
        creator_.phantomOperation(write, reads, name);
    }

   private:
    mustard::TiledGraphCreator& creator_;
    int*                        completionFlags_;
    cudaStream_t                stream_;
};

class KernelStopWatch
{
   public:
    KernelStopWatch(MeasureFlags flags, OperationCapturer& capturer)
        : flags_(flags), capturer_(capturer)
    {}

    void allocateTimers(int nrNodes)
    {
        if (flags_.measureKernelWait)
        {
            cudaMalloc(&d_waitTs_, 2 * nrNodes * sizeof(unsigned long long));
            cudaMemset(d_waitTs_, 0, 2 * nrNodes * sizeof(unsigned long long));
        }
        if (flags_.measureKernelLaunch)
        {
            cudaMalloc(&d_compTs_, 2 * nrNodes * sizeof(unsigned long long));
            cudaMemset(d_compTs_, 0, 2 * nrNodes * sizeof(unsigned long long));
        }
    }

    void waitStart(int idx) { if (d_waitTs_) capturer_.launchTimestamp(d_waitTs_ + idx * 2 + 0); }
    void waitEnd(int idx)   { if (d_waitTs_) capturer_.launchTimestamp(d_waitTs_ + idx * 2 + 1); }
    void compStart(int idx) { if (d_compTs_) capturer_.launchTimestamp(d_compTs_ + idx * 2 + 0); }
    void compEnd(int idx)   { if (d_compTs_) capturer_.launchTimestamp(d_compTs_ + idx * 2 + 1); }

    TimestampBuffers buffers() const { return {d_waitTs_, d_compTs_}; }

   private:
    MeasureFlags        flags_;
    OperationCapturer&  capturer_;
    unsigned long long* d_waitTs_ = nullptr;
    unsigned long long* d_compTs_ = nullptr;
};

template <typename Ops>
class DAGBuilder
{
    PartitionedDag            dag_;
    std::map<MatrixTile, int> lastWriterByTile_;
    int                       myPE_;
    Ops&                      ops_;
    OperationCapturer&        capturer_;
    KernelStopWatch&          sw_;

   public:
    DAGBuilder(int myPE, Ops& ops, OperationCapturer& capturer, KernelStopWatch& sw)
        : myPE_(myPE), ops_(ops), capturer_(capturer), sw_(sw)
    {
    }

    template <typename F>
    void add(int owner, F&& f)
    {
        TileAccess access      = f(ops_);
        int        futureIndex = (int)dag_.nodes().size();

        for (const auto& readTile : access.reads)
        {
            auto it = lastWriterByTile_.find(readTile);
            if (it != lastWriterByTile_.end())
                dag_.addEdge({it->second, futureIndex});
        }

        lastWriterByTile_[access.write] = futureIndex;
        dag_.addNode({-1, owner, std::move(access.name), std::move(access.write),
                      std::move(access.reads), std::move(access.work)});
    }

    PartitionedDag build()
    {
        int localNodes = 0;
        for (auto& n : dag_.nodes())
            if (n.partition == myPE_) localNodes++;
        sw_.allocateTimers(localNodes);

        int localIdx = 0;
        for (auto& n : dag_.nodes())
        {
            if (n.partition == myPE_)
            {
                auto waitNodes   = dag_.crossIncomingNodeIndices(n);
                auto signalParts = dag_.crossOutgoingPartitions(n);
                int* d_wait      = OperationCapturer::upload(waitNodes);
                int* d_signal    = OperationCapturer::upload(signalParts);

                capturer_.beginCapture(n.write, n.reads, n.name);
                if (d_wait)
                {
                    sw_.waitStart(localIdx);
                    capturer_.launchWait(d_wait, (int)waitNodes.size());
                    sw_.waitEnd(localIdx);
                }
                sw_.compStart(localIdx);
                n.work();
                sw_.compEnd(localIdx);
                if (d_signal)
                    capturer_.launchSignal(n.index, d_signal, (int)signalParts.size());
                capturer_.endCapture();
                localIdx++;
            }
            else
            {
                capturer_.phantom(n.write, n.reads, n.name);
            }
        }
        return std::move(dag_);
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
