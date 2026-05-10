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
#include "partitioned_dag.h"
#include "tile_operation.h"
#include "utils.h"

struct TileAccess
{
    TileOperation          op;
    std::function<void()>  work;
};

struct MeasureFlags
{
    bool measureKernelWait   = false;
    bool measureKernelLaunch = false;
    bool notifyWaitReached   = false;
};

struct TimestampBuffers
{
    unsigned long long* d_waitTs = nullptr;
    unsigned long long* d_compTs = nullptr;
};

class CompletionFlags
{
   public:
    CompletionFlags(int n) : n_(n), ptr_((int*)nvshmem_malloc(sizeof(int) * n))
    {
        checkCudaErrors(cudaMemset(ptr_, 0, sizeof(int) * n));
    }
    ~CompletionFlags() { nvshmem_free(ptr_); }

    void resetAll(int nPEs)
    {
        std::vector<int> zeros(n_, 0);
        for (int pe = 0; pe < nPEs; pe++) nvshmem_int_put(ptr_, zeros.data(), n_, pe);
        nvshmem_quiet();
    }

    int* data() { return ptr_; }

   private:
    int  n_;
    int* ptr_;
};

class OperationCapturer
{
   public:
    OperationCapturer(mustard::TiledGraphCreator& creator, int* completionFlags,
                      cudaStream_t stream)
        : creator_(creator), completionFlags_(completionFlags), stream_(stream)
    {
    }

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

    void recordEvent(cudaEvent_t event) { cudaEventRecord(event, stream_); }

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
    {
    }

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

    void waitStart(int idx)
    {
        if (d_waitTs_) capturer_.launchTimestamp(d_waitTs_ + idx * 2 + 0);
    }
    void waitEnd(int idx)
    {
        if (d_waitTs_) capturer_.launchTimestamp(d_waitTs_ + idx * 2 + 1);
    }
    void compStart(int idx)
    {
        if (d_compTs_) capturer_.launchTimestamp(d_compTs_ + idx * 2 + 0);
    }
    void compEnd(int idx)
    {
        if (d_compTs_) capturer_.launchTimestamp(d_compTs_ + idx * 2 + 1);
    }

    TimestampBuffers buffers() const { return {d_waitTs_, d_compTs_}; }

   private:
    MeasureFlags        flags_;
    OperationCapturer&  capturer_;
    unsigned long long* d_waitTs_ = nullptr;
    unsigned long long* d_compTs_ = nullptr;
};

struct DVFSSignal
{
    cudaEvent_t   event;
    TileOperation op;
};

class DVFSSignalBuilder
{
   public:
    DVFSSignalBuilder(OperationCapturer& capturer) : capturer_(capturer) {}

    void newSignal(const TileOperation& op)
    {
        cudaEvent_t ev;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        capturer_.recordEvent(ev);
        signals_.push_back({ev, op});
    }

    const std::vector<DVFSSignal>& signals() const { return signals_; }

   private:
    OperationCapturer&      capturer_;
    std::vector<DVFSSignal> signals_;
};

template <typename Ops>
class PartitionedCudaGraphBuilder
{
    PartitionedDag<TileAccess> dag_;
    std::map<MatrixTile, int>  lastWriterByTile_;
    int                        myPE_;
    Ops&                       ops_;
    OperationCapturer&         capturer_;
    KernelStopWatch&           sw_;
    DVFSSignalBuilder&         dvfsSignalBuilder_;

   public:
    PartitionedCudaGraphBuilder(int myPE, Ops& ops, OperationCapturer& capturer,
                                KernelStopWatch& sw, DVFSSignalBuilder& dvfsSignalBuilder)
        : myPE_(myPE), ops_(ops), capturer_(capturer), sw_(sw),
          dvfsSignalBuilder_(dvfsSignalBuilder)
    {
    }

    template <typename F>
    void add(int owner, F&& f)
    {
        TileAccess access      = f(ops_);
        int        futureIndex = (int)dag_.nodes().size();

        // addNode must come before addEdge so that incoming_[futureIndex] exists
        auto reads = access.op.reads;
        auto write = access.op.write;
        dag_.addNode({-1, owner, true, std::move(access)});

        for (const auto& readTile : reads)
        {
            auto it = lastWriterByTile_.find(readTile);
            if (it != lastWriterByTile_.end()) dag_.addEdge({it->second, futureIndex});
        }

        lastWriterByTile_[write] = futureIndex;
    }

    std::vector<cudaGraphExec_t> build(mustard::TiledGraphCreator& creator, cudaStream_t stream,
                                       bool dot = false)
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
                auto waitNodes   = dag_.crossIncomingNodes(n);
                auto signalNodes = dag_.crossOutgoingNodes(n);
                std::vector<int> waitIndices;
                for (auto& wn : waitNodes) waitIndices.push_back(wn.index);
                std::set<int> signalPartSet;
                for (auto& sn : signalNodes) signalPartSet.insert(sn.partition);
                std::vector<int> signalParts(signalPartSet.begin(), signalPartSet.end());
                int* d_wait      = OperationCapturer::upload(waitIndices);
                int* d_signal    = OperationCapturer::upload(signalParts);

                capturer_.beginCapture(n.content.op.write, n.content.op.reads, n.content.op.name);
                if (d_wait)
                {
                    dvfsSignalBuilder_.newSignal(n.content.op);


                    sw_.waitStart(localIdx);
                    capturer_.launchWait(d_wait, (int)waitIndices.size());
                    sw_.waitEnd(localIdx);
                }
                sw_.compStart(localIdx);
                n.content.work();
                sw_.compEnd(localIdx);
                if (d_signal) capturer_.launchSignal(n.index, d_signal, (int)signalParts.size());
                capturer_.endCapture();
                localIdx++;
            }
            else
            {
                capturer_.phantom(n.content.op.write, n.content.op.reads, n.content.op.name);
            }
        }

        checkCudaErrors(cudaDeviceSynchronize());

        auto                         myTasks = dag_.nodes(myPE_);
        std::vector<cudaGraphExec_t> execs(myTasks.size());
        for (int idx = 0; idx < (int)myTasks.size(); idx++)
        {
            int task = myTasks[idx].index;
            if (dot)
            {
                char filename[32];
                sprintf(filename, "./graph_%d_%d.dot", task, myPE_);
                checkCudaErrors(cudaGraphDebugDotPrint(creator.subgraphs[task], filename, 0));
            }
            checkCudaErrors(
                cudaGraphInstantiate(&execs[idx], creator.subgraphs[task], nullptr, nullptr, 0));
            cudaGraphUpload(execs[idx], stream);
        }
        checkCudaErrors(cudaDeviceSynchronize());

        return execs;
    }


    PartitionedDag<TileAccess> getPartitionedGraph() { return std::move(dag_); }
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

    static CholeskyCudaOperations build(cudaStream_t stream, StridedDevicePanels panels, int B,
                                        int N, int nPEs, int numMyTasks, int workspace,
                                        mustard::OccupancyTracker& occupancy)
    {
        cublasHandle_t     cublasHandle;
        cusolverDnHandle_t cusolverHandle;
        checkCudaErrors(cublasCreate(&cublasHandle));
        checkCudaErrors(cusolverDnCreate(&cusolverHandle));
        checkCudaErrors(cublasSetStream(cublasHandle, stream));
        checkCudaErrors(cusolverDnSetStream(cusolverHandle, stream));

        int workspaceInBytesOnDevice;
        checkCudaErrors(cusolverDnDpotrf_bufferSize(cusolverHandle, CUBLAS_FILL_MODE_LOWER, B,
                                                    panels.myPanel().tile(0, 0), N,
                                                    &workspaceInBytesOnDevice));
        workspaceInBytesOnDevice *= 8;

        double* d_workspace_cusolver;
        checkCudaErrors(cudaMalloc(&d_workspace_cusolver, workspaceInBytesOnDevice));

        int* d_info;
        checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));

        int    cublasWorkspaceSize = 1024 * workspace;
        void** d_workspace_cublas  = (void**)malloc(sizeof(void*) * numMyTasks);
        for (int i = 0; i < numMyTasks; i++)
            checkCudaErrors(cudaMalloc(&d_workspace_cublas[i], cublasWorkspaceSize));

        return {cublasHandle,
                cusolverHandle,
                d_workspace_cusolver,
                workspaceInBytesOnDevice,
                d_workspace_cublas,
                cublasWorkspaceSize,
                numMyTasks,
                d_info,
                std::move(panels),
                B,
                N,
                nPEs,
                occupancy};
    }

    StridedDevicePanel& myPanel() { return panels_.myPanel(); }

    void setWorkspace()
    {
        checkCudaErrors(
            cublasSetWorkspace(cublasHandle_, d_workspace_cublas_[++wsIdx_], cublasWorkspaceSize_));
    }

    TileAccess potrf(int k)
    {
        return {TileOperation::potrf(k),
                [this, k]()
                {
                    checkCudaErrors(cusolverDnDpotrf(
                        cusolverHandle_, CUBLAS_FILL_MODE_LOWER, B_, panels_.myPanel().tile(k, k),
                        N_, d_workspace_cusolver_, workspaceInBytesOnDevice_, d_info_));
                }};
    }

    TileAccess trsm(int i, int k)
    {
        return {TileOperation::trsm(i, k),
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
        return {TileOperation::syrk(i, k),
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
        return {TileOperation::cholGemm(j, i, k),
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

class LUCudaOperations
{
   public:
    LUCudaOperations(cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle,
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

    ~LUCudaOperations()
    {
        cublasDestroy(cublasHandle_);
        cusolverDnDestroy(cusolverHandle_);
        cudaFree(d_workspace_cusolver_);
        for (int i = 0; i < numWorkspaces_; i++) cudaFree(d_workspace_cublas_[i]);
        free(d_workspace_cublas_);
        cudaFree(d_info_);
    }

    LUCudaOperations(const LUCudaOperations&)            = delete;
    LUCudaOperations& operator=(const LUCudaOperations&) = delete;

    static LUCudaOperations build(cudaStream_t stream, StridedDevicePanels panels, int B, int N,
                                  int nPEs, int numMyTasks, int workspace,
                                  mustard::OccupancyTracker& occupancy)
    {
        cublasHandle_t     cublasHandle;
        cusolverDnHandle_t cusolverHandle;
        checkCudaErrors(cublasCreate(&cublasHandle));
        checkCudaErrors(cusolverDnCreate(&cusolverHandle));
        checkCudaErrors(cublasSetStream(cublasHandle, stream));
        checkCudaErrors(cusolverDnSetStream(cusolverHandle, stream));

        int workspaceInBytesOnDevice;
        checkCudaErrors(cusolverDnDgetrf_bufferSize(
            cusolverHandle, B, B, panels.myPanel().tile(0, 0), N, &workspaceInBytesOnDevice));
        workspaceInBytesOnDevice *= 8;

        double* d_workspace_cusolver;
        checkCudaErrors(cudaMalloc(&d_workspace_cusolver, workspaceInBytesOnDevice));

        int* d_info;
        checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));

        int    cublasWorkspaceSize = 1024 * workspace;
        void** d_workspace_cublas  = (void**)malloc(sizeof(void*) * numMyTasks);
        for (int i = 0; i < numMyTasks; i++)
            checkCudaErrors(cudaMalloc(&d_workspace_cublas[i], cublasWorkspaceSize));

        return {cublasHandle,
                cusolverHandle,
                d_workspace_cusolver,
                workspaceInBytesOnDevice,
                d_workspace_cublas,
                cublasWorkspaceSize,
                numMyTasks,
                d_info,
                std::move(panels),
                B,
                N,
                nPEs,
                occupancy};
    }

    StridedDevicePanel& myPanel() { return panels_.myPanel(); }

    void setWorkspace()
    {
        checkCudaErrors(
            cublasSetWorkspace(cublasHandle_, d_workspace_cublas_[++wsIdx_], cublasWorkspaceSize_));
    }

    // Factor diagonal block: A[k][k] -> L[k][k], U[k][k]
    TileAccess getrf(int k)
    {
        return {TileOperation::getrf(k),
                [this, k]()
                {
                    checkCudaErrors(cusolverDnDgetrf(cusolverHandle_, B_, B_,
                                                     panels_.myPanel().tile(k, k), N_,
                                                     d_workspace_cusolver_, nullptr, d_info_));
                }};
    }

    // Solve L[k][k] * U[k][i] = A[k][i]  (update U row, assigned to PE i%nPEs)
    TileAccess trsm_l(int k, int i)
    {
        return {TileOperation::trsmL(k, i),
                [this, k, i]()
                {
                    setWorkspace();
                    checkCudaErrors(cublasDtrsm(
                        cublasHandle_, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                        CUBLAS_DIAG_UNIT, B_, B_, &one_, panels_.otherPanel(k % nPEs_).tile(k, k),
                        N_, panels_.myPanel().tile(k, i), N_));
                }};
    }

    // Solve L[i][k] * U[k][k] = A[i][k]  (update L column, assigned to PE k%nPEs)
    TileAccess trsm_r(int i, int k)
    {
        return {TileOperation::trsmR(i, k),
                [this, i, k]()
                {
                    setWorkspace();
                    checkCudaErrors(cublasDtrsm(
                        cublasHandle_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                        CUBLAS_DIAG_NON_UNIT, B_, B_, &one_, panels_.myPanel().tile(k, k), N_,
                        panels_.myPanel().tile(i, k), N_));
                }};
    }

    // A[i][j] -= L[i][k] * U[k][j]  (assigned to PE j%nPEs)
    TileAccess gemm(int i, int j, int k)
    {
        return {TileOperation::luGemm(i, j, k),
                [this, i, j, k]()
                {
                    setWorkspace();
                    checkCudaErrors(
                        cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N, B_, B_, B_,
                                     &minusOne_, panels_.otherPanel(k % nPEs_).tile(i, k),
                                     CUDA_R_64F, N_, panels_.myPanel().tile(k, j), CUDA_R_64F, N_,
                                     &one_, panels_.myPanel().tile(i, j), CUDA_R_64F, N_,
                                     CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
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
