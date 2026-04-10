#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "broker_queue.h"
#include "utils.h"

#define FLAGS_SUBG_COUNT 0
#define FLAGS_OCCUP 4
#define MAX_TILE 4000

// ---------- CUDA Graph API compatibility (12.x vs 13.0+) ----------
// CUDA 13.0 changed signatures of several graph functions to require a
// cudaGraphEdgeData* parameter; the old overloads were removed.
// These wrappers let the code compile with both API versions.
#if CUDART_VERSION >= 13000
#define MUSTARD_cudaGraphNodeGetDependentNodes(node, out, cnt) \
    cudaGraphNodeGetDependentNodes((node), (out), nullptr, (cnt))
#define MUSTARD_cudaGraphNodeGetDependencies(node, out, cnt) \
    cudaGraphNodeGetDependencies((node), (out), nullptr, (cnt))
#define MUSTARD_cudaGraphGetEdges(g, from, to, cnt) \
    cudaGraphGetEdges((g), (from), (to), nullptr, (cnt))
#define MUSTARD_cudaGraphAddDependencies(g, from, to, cnt) \
    cudaGraphAddDependencies((g), (from), (to), nullptr, (cnt))
#define MUSTARD_cudaGraphRemoveDependencies(g, from, to, cnt) \
    cudaGraphRemoveDependencies((g), (from), (to), nullptr, (cnt))
#else
#define MUSTARD_cudaGraphNodeGetDependentNodes(node, out, cnt) \
    cudaGraphNodeGetDependentNodes((node), (out), (cnt))
#define MUSTARD_cudaGraphNodeGetDependencies(node, out, cnt) \
    cudaGraphNodeGetDependencies((node), (out), (cnt))
#define MUSTARD_cudaGraphGetEdges(g, from, to, cnt) cudaGraphGetEdges((g), (from), (to), (cnt))
#define MUSTARD_cudaGraphAddDependencies(g, from, to, cnt) \
    cudaGraphAddDependencies((g), (from), (to), (cnt))
#define MUSTARD_cudaGraphRemoveDependencies(g, from, to, cnt) \
    cudaGraphRemoveDependencies((g), (from), (to), (cnt))
#endif

extern int myPE;

typedef std::pair<int, int> MatrixTile;

namespace mustard
{

__global__ void kernel_dep_wait(int* dependencies, int nodeIndex, int myPE)
{
    while (nvshmem_int_atomic_fetch(dependencies + nodeIndex, myPE) > 0)
    {
    }
}

__global__ void kernel_dep_update_noq(int* dependencies, int nodeIndex, int PE, int srcPE = -1)
{
    nvshmem_int_atomic_fetch_add(dependencies + nodeIndex, -1, PE);
}

__global__ void kernel_dep_update(BrokerWorkDistributor queue, int* dependencies, int nodeIndex)
{
    int old = nvshmem_int_atomic_fetch_add(dependencies + nodeIndex, -1, 0);  // on PE 0
    if (old == 1)
    {
        queue.enqueue(nodeIndex, 0);
    }
}

// only 1 thread runs this
// sm_count is negative when occupancy is reduced after a kernel has been completed
__global__ void kernel_occupancy_update(int sm_count, volatile int* flags)
{
    atomicAdd((int*)&flags[FLAGS_OCCUP], sm_count);
}

__global__ void kernel_populate_queue(BrokerWorkDistributor queue, int* dependencies,
                                      int totalNodes)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < totalNodes; i = i + gridDim.x * blockDim.x)
    {
        if (dependencies[i] == 0) queue.enqueue(i, 0);
    }
}

__global__ void kernel_test_dequeue(BrokerWorkDistributor queue)
{
    unsigned int placeholder      = UINT32_MAX;
    bool         placeholder_bool = false;
    while (queue.size(0) > 0)
    {
        queue.dequeue(placeholder_bool, placeholder, 0);
        printf("Dequeued %d\n", placeholder);
    }
}

__global__ void kernel_scheduler(BrokerWorkDistributor queue, volatile int* flags,
                                 cudaGraphExec_t* subgraphs, int totalSubgraphs, int device)
{
    unsigned int placeholder      = UINT32_MAX;
    bool         placeholder_bool = false;

    while (nvshmem_int_atomic_fetch((int*)&flags[FLAGS_SUBG_COUNT], 0) < totalSubgraphs)
    {
        if (flags[FLAGS_OCCUP] < (100) && queue.size(0) > 0)
        {
            queue.dequeue(placeholder_bool, placeholder, 0);
            if (placeholder_bool)
            {
                cudaGraphLaunch(subgraphs[placeholder], cudaStreamGraphFireAndForget);
                nvshmem_int_atomic_inc((int*)&flags[FLAGS_SUBG_COUNT], 0);
            }
        }
    }
}

__global__ void kernel_signal_static(int task_id, int* d_completion_flags, int* d_notify_pes,
                                     int n_notify_pes, int debug)
{
    int one = 1;
    if (debug) printf("[signal] task %d: signaling %d PEs\n", task_id, n_notify_pes);
    nvshmem_fence();
    for (int i = 0; i < n_notify_pes; i++)
    {
        if (debug)
            printf("[signal] task %d -> PE %d flag[%d]\n", task_id, d_notify_pes[i], task_id);
        nvshmem_int_put(&d_completion_flags[task_id], &one, 1, d_notify_pes[i]);
    }
    if (debug) printf("[signal] task %d: done\n", task_id);
}

__global__ void kernel_wait_static(int* d_deps, int n_deps, int* d_completion_flags, int debug)
{
    if (debug) printf("[wait] waiting on %d deps\n", n_deps);
    for (int i = 0; i < n_deps; i++)
    {
        if (debug)
            printf("[wait] polling flag[%d] (currently %d)\n", d_deps[i],
                   d_completion_flags[d_deps[i]]);
        nvshmem_int_wait_until(&d_completion_flags[d_deps[i]], NVSHMEM_CMP_EQ, 1);
        if (debug) printf("[wait] flag[%d] set\n", d_deps[i]);
    }
    if (debug) printf("[wait] all deps satisfied\n");
}

class TiledGraphCreator
{
   public:
    cudaGraph_t*                  subgraphs;
    cudaGraph_t                   graph;
    std::vector<std::vector<int>> subgraphDependencies;
    std::vector<std::string>      subgraphOpNames;

    TiledGraphCreator(cudaStream_t stream, cudaGraph_t graph, bool subgraph = false,
                      int totalNodes = 1)
        : stream(stream), graph(graph)
    {
        this->lastModifiedTile = std::make_pair(-1, -1);
        this->subgraph         = subgraph;
        this->subgraphs        = new cudaGraph_t[totalNodes];
        this->subgraphDependencies.resize(totalNodes);
        this->subgraphOpNames.resize(totalNodes);
        this->index_counter = 0;
    }

    void beginCaptureOperation(MatrixTile                        tileToWrite,
                               std::initializer_list<MatrixTile> tilesToRead,
                               const std::string&                opName = "")
    {
        auto tiles = std::vector<MatrixTile>(tilesToRead);
        tiles.push_back(tileToWrite);

        this->lastModifiedTile = tileToWrite;

        if (!this->subgraph)
        {
            auto dependencies      = this->getDependencies(tiles);
            this->lastDependencies = dependencies;
            checkCudaErrors(cudaStreamBeginCaptureToGraph(
                this->stream, this->graph, dependencies.data(), nullptr, dependencies.size(),
                cudaStreamCaptureModeGlobal));
        }
        else
        {
            this->subgraphDependencies[index_counter] = this->getSubgraphDependencies(tiles);
            this->subgraphOpNames[index_counter]      = opName;

            checkCudaErrors(cudaGraphCreate(&this->subgraphs[index_counter], 0));
            checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream,
                                                          this->subgraphs[index_counter], nullptr,
                                                          nullptr, 0, cudaStreamCaptureModeGlobal));
        }
    }

    void endCaptureOperation()
    {
        assert(this->lastModifiedTile.first != -1 && this->lastModifiedTile.second != -1);
        if (!this->subgraph)
        {
            checkCudaErrors(cudaStreamEndCapture(this->stream, &this->graph));
            this->tileLastModifiedByMap[this->lastModifiedTile] =
                this->getTailOfLastCapturedNodeChain();
        }
        else
        {
            checkCudaErrors(cudaStreamEndCapture(this->stream, &(this->subgraphs[index_counter])));
            this->tileIndexByMap[this->lastModifiedTile] = this->index_counter;
            this->index_counter++;
        }
        this->lastModifiedTile = std::make_pair(-1, -1);
    };

    void printDeps()
    {
        for (int i = 0; i < this->subgraphDependencies.size(); i++)
        {
            std::vector<int> deps = this->subgraphDependencies[i];
            std::cout << i << ":";
            for (int j = 0; j < deps.size(); j++)
            {
                std::cout << " " << deps[j];
            }
            std::cout << std::endl;
        }
    }

    void printInvocations(const std::string& path, int myPE)
    {
        std::string   filename = path + "_PE" + std::to_string(myPE) + ".txt";
        std::ofstream out(filename);
        if (!out.is_open())
        {
            std::cerr << "Error: Could not open file " << filename << " for writing invocations."
                      << std::endl;
            return;
        }
        for (int i = 0; i < index_counter; i++)
        {
            out << i << ": " << subgraphOpNames[i] << "\n";
        }
        out.close();
    }

    void insertDependencyKernel(int src, int dst, BrokerWorkDistributor queue, int* d_dependencies)
    {
        cudaGraphNode_t      dependencyUpdateNode;
        cudaKernelNodeParams params = {0};
        params.gridDim              = dim3(1, 1, 1);
        params.blockDim             = dim3(1, 1, 1);
        params.extra                = NULL;
        params.func                 = (void*)kernel_dep_update;
        void* kernelArgs[3]         = {&queue, &d_dependencies, &dst};
        params.kernelParams         = kernelArgs;
        std::vector<cudaGraphNode_t> deps;
        deps.push_back(getTail(this->subgraphs[src]));
        checkCudaErrors(cudaGraphAddKernelNode(&dependencyUpdateNode, this->subgraphs[src],
                                               deps.data(), deps.size(), &params));
    }

    // for inserting LU subgraphs that have to be constructed when the tile size is too big
    void insertSubgraph(cudaGraph_t getrfSubgraph)
    {
        cudaGraph_t                  g = this->subgraphs[index_counter - 1];
        std::vector<cudaGraphNode_t> deps;

        cudaGraphNode_t root, tail;
        root = getRoot(g);
        tail = getTail(g);

        if (myPE != 0)
        {
            size_t edge_count;
            checkCudaErrors(MUSTARD_cudaGraphNodeGetDependentNodes(root, NULL, &edge_count));
            if (edge_count > 0)
            {
                std::vector<cudaGraphNode_t> children(edge_count);
                checkCudaErrors(
                    MUSTARD_cudaGraphNodeGetDependentNodes(root, children.data(), &edge_count));
                root = children[0];
            }

            checkCudaErrors(MUSTARD_cudaGraphNodeGetDependencies(tail, NULL, &edge_count));
            if (edge_count > 0)
            {
                std::vector<cudaGraphNode_t> parents(edge_count);
                checkCudaErrors(
                    MUSTARD_cudaGraphNodeGetDependencies(tail, parents.data(), &edge_count));
                tail = parents[0];
            }
        }
        deps.push_back(root);

        cudaGraphNode_t node;
        checkCudaErrors(
            cudaGraphAddChildGraphNode(&node, g, deps.data(), deps.size(), getrfSubgraph));
        // if (myPE == 0) {
        MUSTARD_cudaGraphAddDependencies(g, &node, &tail, 1);  // add dep from subg to write memcpy
        MUSTARD_cudaGraphRemoveDependencies(g, &root, &tail,
                                            1);  // remove dep from read memcpy to write memcpy
        //}
    }

   private:
    std::map<MatrixTile, cudaGraphNode_t> tileLastModifiedByMap;
    std::map<MatrixTile, int>             tileIndexByMap;
    std::map<cudaGraphNode_t, bool>       visited;
    cudaStream_t                          stream;
    MatrixTile                            lastModifiedTile;
    std::vector<cudaGraphNode_t>          lastDependencies;
    int                                   index_counter;
    bool                                  subgraph;

    std::vector<cudaGraphNode_t> getDependencies(std::vector<MatrixTile> tiles)
    {
        std::vector<cudaGraphNode_t> dependencies;
        for (auto tile : tiles)
        {
            auto it = this->tileLastModifiedByMap.find(tile);
            if (it != this->tileLastModifiedByMap.end())
            {
                dependencies.push_back(it->second);
            }
        }

        auto dedupedEnd = std::unique(dependencies.begin(), dependencies.end());
        dependencies.resize(std::distance(dependencies.begin(), dedupedEnd));

        return dependencies;
    }

    std::vector<int> getSubgraphDependencies(std::vector<MatrixTile> tiles)
    {
        std::vector<int> dependencies;
        for (auto tile : tiles)
        {
            auto it = this->tileIndexByMap.find(tile);
            if (it != this->tileIndexByMap.end())
            {
                dependencies.push_back(it->second);
            }
        }

        std::sort(dependencies.begin(), dependencies.end());
        auto dedupedEnd = std::unique(dependencies.begin(), dependencies.end());
        dependencies.resize(std::distance(dependencies.begin(), dedupedEnd));

        return dependencies;
    }

    cudaGraphNode_t getRoot(cudaGraph_t graph)
    {
        size_t numNodes = 1;
        auto   nodes    = std::make_unique<cudaGraphNode_t[]>(1);
        checkCudaErrors(cudaGraphGetRootNodes(graph, nodes.get(), &numNodes));
        return nodes[0];
    }

    cudaGraphNode_t getTail(cudaGraph_t graph)
    {
        size_t numEdges;
        checkCudaErrors(MUSTARD_cudaGraphGetEdges(graph, nullptr, nullptr, &numEdges));
        if (numEdges == 0)
        {
            size_t numNodes = 1;
            auto   nodes    = std::make_unique<cudaGraphNode_t[]>(1);
            checkCudaErrors(cudaGraphGetNodes(graph, nodes.get(), &numNodes));
            return nodes[0];
        }
        auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);
        auto to   = std::make_unique<cudaGraphNode_t[]>(numEdges);
        checkCudaErrors(MUSTARD_cudaGraphGetEdges(graph, from.get(), to.get(), &numEdges));

        std::map<cudaGraphNode_t, bool> hasOutGoingEdge;
        std::set<cudaGraphNode_t>       noOutGoingEdgeNodes;
        for (int i = 0; i < numEdges; i++)
        {
            hasOutGoingEdge[from[i]] = true;
            noOutGoingEdgeNodes.erase(from[i]);
            if (!hasOutGoingEdge[to[i]]) noOutGoingEdgeNodes.insert(to[i]);
        }

        if (noOutGoingEdgeNodes.size() != 1)
        {
            size_t numNodes = 0;
            checkCudaErrors(cudaGraphGetNodes(graph, nullptr, &numNodes));

            std::vector<cudaGraphNode_t> nodes(numNodes);
            cudaGraphGetNodes(graph, nodes.data(), &numNodes);
            return nodes.back();
        }

        return *noOutGoingEdgeNodes.rbegin();
    }

    cudaGraphNode_t getTailOfLastCapturedNodeChain()
    {
        if (lastDependencies.size() == 0)
        {
            return getTail(this->graph);
        }
        else
        {
            auto   nodeBeforeChain = lastDependencies[0];
            size_t numDependentNodes;
            checkCudaErrors(MUSTARD_cudaGraphNodeGetDependentNodes(nodeBeforeChain, nullptr,
                                                                   &numDependentNodes));

            assert(numDependentNodes > 0);

            auto dependentNodes = std::make_unique<cudaGraphNode_t[]>(numDependentNodes);
            checkCudaErrors(MUSTARD_cudaGraphNodeGetDependentNodes(
                nodeBeforeChain, dependentNodes.get(), &numDependentNodes));

            cudaGraphNode_t chainBeginningNode;
            for (int i = 0; i < numDependentNodes; i++)
            {
                if (!visited[dependentNodes[i]])
                {
                    chainBeginningNode = dependentNodes[i];
                    break;
                }
            }

            auto u = chainBeginningNode;
            while (true)
            {
                visited[u] = true;
                checkCudaErrors(
                    MUSTARD_cudaGraphNodeGetDependentNodes(u, nullptr, &numDependentNodes));
                if (numDependentNodes == 0) break;

                std::vector<cudaGraphNode_t> nodes(numDependentNodes);
                checkCudaErrors(
                    MUSTARD_cudaGraphNodeGetDependentNodes(u, nodes.data(), &numDependentNodes));
                u = nodes.back();
            }

            return u;
        }
    }
};

inline cudaGraphNode_t getSubgraphTail(cudaGraph_t g)
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
}

struct InjectionContext
{
    std::vector<cudaGraphNode_t> task_wait_node;  // written by SubgraphInjector
    std::vector<cudaEvent_t>     compute_start;   // written by WaitTimeDecorator
    std::vector<cudaEvent_t>     compute_end;     // written by ComputeTimeDecorator

    explicit InjectionContext(int totalNodes)
        : task_wait_node(totalNodes, nullptr),
          compute_start(totalNodes, nullptr),
          compute_end(totalNodes, nullptr)
    {
    }

    ~InjectionContext()
    {
        for (auto &ev : compute_start)
            if (ev) cudaEventDestroy(ev);
        for (auto &ev : compute_end)
            if (ev) cudaEventDestroy(ev);
    }
};

class IInjector
{
   public:
    virtual void inject(const std::vector<int> &tasks, InjectionContext &ctx) = 0;
    virtual ~IInjector()                                                       = default;
};

class StaticRoundRobinScheduler
{
   public:
    StaticRoundRobinScheduler(int nPEs, int myPE, int totalNodes,
                               const std::vector<std::vector<int>>& subgraphDependencies)
        : nPEs_(nPEs),
          myPE_(myPE),
          totalNodes_(totalNodes),
          d_task_deps_(totalNodes, nullptr),
          d_task_notify_pes_(totalNodes, nullptr),
          task_dep_count_(totalNodes, 0),
          task_notify_count_(totalNodes, 0),
          iter_(0)
    {
        // Round-robin task assignment
        pe_tasks_.resize(nPEs);
        task_pe_.resize(totalNodes);
        for (int i = 0; i < totalNodes; i++)
        {
            pe_tasks_[i % nPEs].push_back(i);
            task_pe_[i] = i % nPEs;
        }

        // Topological sort of this PE's tasks respecting same-PE dependencies
        std::set<int> topo_done;
        while (my_tasks_sorted_.size() < pe_tasks_[myPE].size())
        {
            bool progress = false;
            for (int task : pe_tasks_[myPE])
            {
                if (topo_done.count(task)) continue;
                bool ready = true;
                for (int dep : subgraphDependencies[task])
                {
                    if (task_pe_[dep] == myPE && !topo_done.count(dep))
                    {
                        ready = false;
                        break;
                    }
                }
                if (ready)
                {
                    my_tasks_sorted_.push_back(task);
                    topo_done.insert(task);
                    progress = true;
                }
            }
            if (!progress) break;
        }

        // Build reverse dependency map
        dependents_.resize(totalNodes);
        for (int j = 0; j < totalNodes; j++)
            for (int dep : subgraphDependencies[j]) dependents_[dep].push_back(j);

        // Allocate per-task device arrays for deps and notify PEs
        for (int task : pe_tasks_[myPE])
        {
            const auto& deps      = subgraphDependencies[task];
            task_dep_count_[task] = (int)deps.size();
            if (!deps.empty())
            {
                checkCudaErrors(cudaMalloc(&d_task_deps_[task], sizeof(int) * deps.size()));
                checkCudaErrors(cudaMemcpy(d_task_deps_[task], deps.data(),
                                           sizeof(int) * deps.size(), cudaMemcpyHostToDevice));
            }

            std::set<int> notify_set;
            for (int dep_task : dependents_[task]) notify_set.insert(task_pe_[dep_task]);
            std::vector<int> notify_vec(notify_set.begin(), notify_set.end());
            task_notify_count_[task] = (int)notify_vec.size();
            if (!notify_vec.empty())
            {
                checkCudaErrors(
                    cudaMalloc(&d_task_notify_pes_[task], sizeof(int) * notify_vec.size()));
                checkCudaErrors(cudaMemcpy(d_task_notify_pes_[task], notify_vec.data(),
                                           sizeof(int) * notify_vec.size(),
                                           cudaMemcpyHostToDevice));
            }
        }
    }

    ~StaticRoundRobinScheduler()
    {
        for (int task : pe_tasks_[myPE_])
        {
            if (d_task_deps_[task]) cudaFree(d_task_deps_[task]);
            if (d_task_notify_pes_[task]) cudaFree(d_task_notify_pes_[task]);
        }
    }

    // Returns all tasks for this PE in a safe launch order
    const std::vector<int>& getMyTasksOrdered() const { return my_tasks_sorted_; }

    // Stateful iterator over tasks in launch order; returns false when exhausted
    bool getNextTask(int& task)
    {
        if (iter_ >= (int)my_tasks_sorted_.size()) return false;
        task = my_tasks_sorted_[iter_++];
        return true;
    }

    void resetIterator() { iter_ = 0; }

    int* getTaskDeps(int task) const { return d_task_deps_[task]; }
    int* getNotifyPEs(int task) const { return d_task_notify_pes_[task]; }
    int  getDepCount(int task) const { return task_dep_count_[task]; }
    int  getNotifyCount(int task) const { return task_notify_count_[task]; }

    const std::vector<std::vector<int>>& getDependents() const { return dependents_; }
    int                                  getTaskPE(int task) const { return task_pe_[task]; }

   private:
    int nPEs_, myPE_, totalNodes_;
    std::vector<std::vector<int>> pe_tasks_;
    std::vector<int>              task_pe_;
    std::vector<int>              my_tasks_sorted_;
    std::vector<std::vector<int>> dependents_;
    std::vector<int*>             d_task_deps_;
    std::vector<int*>             d_task_notify_pes_;
    std::vector<int>              task_dep_count_;
    std::vector<int>              task_notify_count_;
    int                           iter_;
};

class SubgraphInjector : public IInjector
{
   public:
    SubgraphInjector(cudaGraph_t *subgraphs, const StaticRoundRobinScheduler &scheduler,
                     int *d_completion_flags, int debug)
        : subgraphs_(subgraphs),
          scheduler_(scheduler),
          d_completion_flags_(d_completion_flags),
          debug_(debug)
    {
    }

    void inject(const std::vector<int> &tasks, InjectionContext &ctx) override
    {
        for (int task : tasks)
        {
            cudaGraph_t sg     = subgraphs_[task];
            int         n_deps = scheduler_.getDepCount(task);
            int        *d_deps = scheduler_.getTaskDeps(task);

            if (n_deps > 0)
            {
                size_t numRoots;
                cudaGraphGetRootNodes(sg, nullptr, &numRoots);
                std::vector<cudaGraphNode_t> roots(numRoots);
                cudaGraphGetRootNodes(sg, roots.data(), &numRoots);

                cudaGraphNode_t      waitNode;
                cudaKernelNodeParams waitParams = {0};
                waitParams.gridDim              = dim3(1);
                waitParams.blockDim             = dim3(1);
                waitParams.func                 = (void *)kernel_wait_static;
                void *waitArgs[4] = {&d_deps, &n_deps, &d_completion_flags_, &debug_};
                waitParams.kernelParams         = waitArgs;
                checkCudaErrors(cudaGraphAddKernelNode(&waitNode, sg, nullptr, 0, &waitParams));
                ctx.task_wait_node[task] = waitNode;

                for (auto &root : roots)
                    MUSTARD_cudaGraphAddDependencies(sg, &waitNode, &root, 1);
            }

            int  n_notify     = scheduler_.getNotifyCount(task);
            int *d_notify_pes = scheduler_.getNotifyPEs(task);

            cudaGraphNode_t      tail = getSubgraphTail(sg);
            cudaGraphNode_t      signalNode;
            cudaKernelNodeParams signalParams = {0};
            signalParams.gridDim              = dim3(1);
            signalParams.blockDim             = dim3(1);
            signalParams.func                 = (void *)kernel_signal_static;
            int   task_id_val                 = task;
            void *signalArgs[5] = {&task_id_val, &d_completion_flags_, &d_notify_pes, &n_notify,
                                   &debug_};
            signalParams.kernelParams         = signalArgs;
            checkCudaErrors(cudaGraphAddKernelNode(&signalNode, sg, &tail, 1, &signalParams));
        }
    }

   private:
    cudaGraph_t                     *subgraphs_;
    const StaticRoundRobinScheduler &scheduler_;
    int                             *d_completion_flags_;
    int                              debug_;
};

// Injects a compute-start event after the wait kernel (or before the first compute node if no
// wait). Required for both wait-time and compute-time measurement.
class WaitTimeDecorator : public IInjector
{
   public:
    WaitTimeDecorator(std::unique_ptr<IInjector> inner, cudaGraph_t *subgraphs)
        : inner_(std::move(inner)), subgraphs_(subgraphs)
    {
    }

    void inject(const std::vector<int> &tasks, InjectionContext &ctx) override
    {
        inner_->inject(tasks, ctx);

        for (int task : tasks)
        {
            cudaGraph_t     sg              = subgraphs_[task];
            cudaGraphNode_t computeStartNode;
            checkCudaErrors(cudaEventCreate(&ctx.compute_start[task]));

            if (ctx.task_wait_node[task] != nullptr)
            {
                cudaGraphNode_t waitNode = ctx.task_wait_node[task];
                checkCudaErrors(cudaGraphAddEventRecordNode(&computeStartNode, sg, &waitNode, 1,
                                                            ctx.compute_start[task]));
                size_t numChildren;
                MUSTARD_cudaGraphNodeGetDependentNodes(waitNode, nullptr, &numChildren);
                std::vector<cudaGraphNode_t> children(numChildren);
                MUSTARD_cudaGraphNodeGetDependentNodes(waitNode, children.data(), &numChildren);
                for (auto &child : children)
                {
                    if (child == computeStartNode) continue;
                    MUSTARD_cudaGraphAddDependencies(sg, &computeStartNode, &child, 1);
                    MUSTARD_cudaGraphRemoveDependencies(sg, &waitNode, &child, 1);
                }
            }
            else
            {
                size_t numRoots;
                cudaGraphGetRootNodes(sg, nullptr, &numRoots);
                std::vector<cudaGraphNode_t> roots(numRoots);
                cudaGraphGetRootNodes(sg, roots.data(), &numRoots);
                checkCudaErrors(cudaGraphAddEventRecordNode(&computeStartNode, sg, nullptr, 0,
                                                            ctx.compute_start[task]));
                for (auto &root : roots)
                    MUSTARD_cudaGraphAddDependencies(sg, &computeStartNode, &root, 1);
            }
        }
    }

   private:
    std::unique_ptr<IInjector> inner_;
    cudaGraph_t               *subgraphs_;
};

// Injects a compute-end event just before the signal kernel.
// Requires WaitTimeDecorator to have run first (ctx.compute_start must be populated).
class ComputeTimeDecorator : public IInjector
{
   public:
    ComputeTimeDecorator(std::unique_ptr<IInjector> inner, cudaGraph_t *subgraphs)
        : inner_(std::move(inner)), subgraphs_(subgraphs)
    {
    }

    void inject(const std::vector<int> &tasks, InjectionContext &ctx) override
    {
        inner_->inject(tasks, ctx);

        for (int task : tasks)
        {
            cudaGraph_t sg = subgraphs_[task];
            checkCudaErrors(cudaEventCreate(&ctx.compute_end[task]));

            // Signal kernel is the current tail; insert compute-end before it
            cudaGraphNode_t signalNode = getSubgraphTail(sg);
            size_t          numParents;
            MUSTARD_cudaGraphNodeGetDependencies(signalNode, nullptr, &numParents);
            std::vector<cudaGraphNode_t> parents(numParents);
            MUSTARD_cudaGraphNodeGetDependencies(signalNode, parents.data(), &numParents);

            cudaGraphNode_t computeEndNode;
            checkCudaErrors(cudaGraphAddEventRecordNode(&computeEndNode, sg, parents.data(),
                                                        numParents, ctx.compute_end[task]));
            for (auto &parent : parents)
                MUSTARD_cudaGraphRemoveDependencies(sg, &parent, &signalNode, 1);
            MUSTARD_cudaGraphAddDependencies(sg, &computeEndNode, &signalNode, 1);
        }
    }

   private:
    std::unique_ptr<IInjector> inner_;
    cudaGraph_t               *subgraphs_;
};

}  // namespace mustard