#pragma once

#include <cuda_runtime.h>

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "mustard.h"  // MUSTARD_* macros, kernel_wait_static, kernel_signal_static, checkCudaErrors
#include "scheduler.h"  // StaticRoundRobinScheduler

namespace mustard
{

inline cudaKernelNodeParams makeKernelParams(void* func, void** args)
{
    cudaKernelNodeParams p = {0};
    p.gridDim              = dim3(1);
    p.blockDim             = dim3(1);
    p.func                 = func;
    p.kernelParams         = args;
    return p;
}

inline std::vector<cudaGraphNode_t> getRootNodes(cudaGraph_t g)
{
    size_t numRoots;
    cudaGraphGetRootNodes(g, nullptr, &numRoots);
    std::vector<cudaGraphNode_t> roots(numRoots);
    cudaGraphGetRootNodes(g, roots.data(), &numRoots);
    return roots;
}

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
    // Device buffer written by TimestampDecorator: [task*2+0]=start ns, [task*2+1]=end ns
    // Values are raw __globaltimer() nanoseconds. Convert to wall time via a reference pair.
    unsigned long long* d_timestamps;  // written by TimestampDecorator
    // Device buffer written by WaitTimestampDecorator: [task*2+0]=wait_start ns,
    // [task*2+1]=wait_end ns Both are 0 for tasks with no cross-GPU dependency (no wait kernel).
    unsigned long long* d_wait_timestamps;  // written by WaitTimestampDecorator

    explicit InjectionContext(int totalNodes)
        : task_wait_node(totalNodes, nullptr),
          compute_start(totalNodes, nullptr),
          compute_end(totalNodes, nullptr),
          d_timestamps(nullptr),
          d_wait_timestamps(nullptr)
    {
    }

    ~InjectionContext()
    {
        for (auto& ev : compute_start)
            if (ev) cudaEventDestroy(ev);
        for (auto& ev : compute_end)
            if (ev) cudaEventDestroy(ev);
        if (d_timestamps) cudaFree(d_timestamps);
        if (d_wait_timestamps) cudaFree(d_wait_timestamps);
    }
};

class IInjector
{
   public:
    virtual void inject(const std::vector<int>& tasks, InjectionContext& ctx) = 0;
    virtual ~IInjector()                                                      = default;
};

class SubgraphInjector : public IInjector
{
   public:
    SubgraphInjector(cudaGraph_t* subgraphs, const StaticRoundRobinScheduler& scheduler,
                     int* d_completion_flags, int debug)
        : subgraphs_(subgraphs),
          scheduler_(scheduler),
          d_completion_flags_(d_completion_flags),
          debug_(debug)
    {
    }

    void inject(const std::vector<int>& tasks, InjectionContext& ctx) override
    {
        for (int task : tasks)
        {
            cudaGraph_t sg     = subgraphs_[task];
            int         n_deps = scheduler_.getDepCount(task);
            int*        d_deps = scheduler_.getTaskDeps(task);

            if (n_deps > 0)  // If there are dependencies on nodes on other GPUs
            {
                prependWaitNode(sg, d_deps, n_deps, ctx, task);
            }

            int n_notify = scheduler_.getNotifyCount(task);
            if (n_notify > 0)  // If other tasks (on other GPUs) are dependant on this Task
            {
                appendSignalNode(task, sg, n_notify);
            }
        }
    }

    void appendSignalNode(int task, cudaGraph_t sg, int n_notify)
    {
        int* d_notify_pes = scheduler_.getNotifyPEs(task);

        cudaGraphNode_t tail = getSubgraphTail(sg);
        cudaGraphNode_t signalNode;
        int             task_id_val = task;
        void* signalArgs[5]         = {&task_id_val, &d_completion_flags_, &d_notify_pes, &n_notify,
                                       &debug_};
        auto  signalParams          = makeKernelParams((void*)kernel_signal_static, signalArgs);
        checkCudaErrors(cudaGraphAddKernelNode(&signalNode, sg, &tail, 1, &signalParams));
    }

    void prependWaitNode(cudaGraph_t& sg, int*& d_deps, int& n_deps, mustard::InjectionContext& ctx,
                         int task)
    {
        // obtain root node(s)
        auto roots = getRootNodes(sg);

        // construct wait node
        cudaGraphNode_t waitNode;
        void*           waitArgs[4] = {&d_deps, &n_deps, &d_completion_flags_, &debug_};
        auto            waitParams  = makeKernelParams((void*)kernel_wait_static, waitArgs);

        // replace position of original root nodes with wait-node
        checkCudaErrors(cudaGraphAddKernelNode(&waitNode, sg, nullptr, 0, &waitParams));
        ctx.task_wait_node[task] = waitNode;

        // add old root(s) back into the DAG after wait-node
        for (auto& root : roots) MUSTARD_cudaGraphAddDependencies(sg, &waitNode, &root, 1);
    }

   private:
    cudaGraph_t*                     subgraphs_;
    const StaticRoundRobinScheduler& scheduler_;
    int*                             d_completion_flags_;
    int                              debug_;
};

// Injects a compute-start event after the wait kernel (or before the first compute node if no
// wait). Required for both wait-time and compute-time measurement.
class WaitTimeDecorator : public IInjector
{
   public:
    WaitTimeDecorator(std::unique_ptr<IInjector> inner, cudaGraph_t* subgraphs)
        : inner_(std::move(inner)), subgraphs_(subgraphs)
    {
    }

    void inject(const std::vector<int>& tasks, InjectionContext& ctx) override
    {
        inner_->inject(tasks, ctx);

        for (int task : tasks)
        {
            cudaGraph_t     sg = subgraphs_[task];
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
                for (auto& child : children)
                {
                    if (child == computeStartNode) continue;
                    MUSTARD_cudaGraphAddDependencies(sg, &computeStartNode, &child, 1);
                    MUSTARD_cudaGraphRemoveDependencies(sg, &waitNode, &child, 1);
                }
            }
            else
            {
                auto roots = getRootNodes(sg);
                checkCudaErrors(cudaGraphAddEventRecordNode(&computeStartNode, sg, nullptr, 0,
                                                            ctx.compute_start[task]));
                for (auto& root : roots)
                    MUSTARD_cudaGraphAddDependencies(sg, &computeStartNode, &root, 1);
            }
        }
    }

   private:
    std::unique_ptr<IInjector> inner_;
    cudaGraph_t*               subgraphs_;
};

// Injects a compute-end event just before the signal kernel.
// Requires WaitTimeDecorator to have run first (ctx.compute_start must be populated).
class ComputeTimeDecorator : public IInjector
{
   public:
    ComputeTimeDecorator(std::unique_ptr<IInjector> inner, cudaGraph_t* subgraphs)
        : inner_(std::move(inner)), subgraphs_(subgraphs)
    {
    }

    void inject(const std::vector<int>& tasks, InjectionContext& ctx) override
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
            for (auto& parent : parents)
                MUSTARD_cudaGraphRemoveDependencies(sg, &parent, &signalNode, 1);
            MUSTARD_cudaGraphAddDependencies(sg, &computeEndNode, &signalNode, 1);
        }
    }

   private:
    std::unique_ptr<IInjector> inner_;
    cudaGraph_t*               subgraphs_;
};

// Injects kernel_record_timestamp nodes immediately before and after the wait kernel to measure
// cross-GPU spin-wait duration. Must be placed first in the decorator chain (right after
// SubgraphInjector) so subsequent decorators see the rewired graph correctly.
//
// After inject(), ctx.d_wait_timestamps holds a device buffer with 2 entries per task:
//   ctx.d_wait_timestamps[task * 2 + 0]  = wait-start __globaltimer() ns (just before spin-wait)
//   ctx.d_wait_timestamps[task * 2 + 1]  = wait-end   __globaltimer() ns (just after spin-wait)
// Both entries are 0 for tasks with no cross-GPU dependency (no wait kernel).
class WaitTimestampDecorator : public IInjector
{
   public:
    WaitTimestampDecorator(std::unique_ptr<IInjector> inner, cudaGraph_t* subgraphs)
        : inner_(std::move(inner)), subgraphs_(subgraphs)
    {
    }

    void inject(const std::vector<int>& tasks, InjectionContext& ctx) override
    {
        inner_->inject(tasks, ctx);

        int totalNodes = (int)ctx.task_wait_node.size();
        checkCudaErrors(
            cudaMalloc(&ctx.d_wait_timestamps, sizeof(unsigned long long) * totalNodes * 2));
        checkCudaErrors(
            cudaMemset(ctx.d_wait_timestamps, 0, sizeof(unsigned long long) * totalNodes * 2));

        for (int task : tasks)
        {
            if (ctx.task_wait_node[task] == nullptr) continue;  // no cross-GPU dep, skip

            cudaGraph_t     sg       = subgraphs_[task];
            cudaGraphNode_t waitNode = ctx.task_wait_node[task];

            // --- Wait-start timestamp (before spin-wait kernel) ---
            cudaGraphNode_t     tsWaitStartNode;
            unsigned long long* wait_start_ptr   = ctx.d_wait_timestamps + task * 2 + 0;
            void*               waitStartArgs[1] = {&wait_start_ptr};
            auto                tsWaitStartParams =
                makeKernelParams((void*)kernel_record_timestamp, waitStartArgs);
            // Insert as a new root (no deps), then make waitNode depend on it
            checkCudaErrors(
                cudaGraphAddKernelNode(&tsWaitStartNode, sg, nullptr, 0, &tsWaitStartParams));
            MUSTARD_cudaGraphAddDependencies(sg, &tsWaitStartNode, &waitNode, 1);

            // --- Wait-end timestamp (after spin-wait kernel, before compute) ---
            cudaGraphNode_t     tsWaitEndNode;
            unsigned long long* wait_end_ptr   = ctx.d_wait_timestamps + task * 2 + 1;
            void*               waitEndArgs[1] = {&wait_end_ptr};
            auto tsWaitEndParams = makeKernelParams((void*)kernel_record_timestamp, waitEndArgs);
            // Insert after waitNode, then rewire waitNode's existing children through tsWaitEndNode
            checkCudaErrors(
                cudaGraphAddKernelNode(&tsWaitEndNode, sg, &waitNode, 1, &tsWaitEndParams));

            size_t numChildren;
            MUSTARD_cudaGraphNodeGetDependentNodes(waitNode, nullptr, &numChildren);
            std::vector<cudaGraphNode_t> children(numChildren);
            MUSTARD_cudaGraphNodeGetDependentNodes(waitNode, children.data(), &numChildren);
            for (auto& child : children)
            {
                if (child == tsWaitEndNode) continue;
                MUSTARD_cudaGraphAddDependencies(sg, &tsWaitEndNode, &child, 1);
                MUSTARD_cudaGraphRemoveDependencies(sg, &waitNode, &child, 1);
            }
        }
    }

   private:
    std::unique_ptr<IInjector> inner_;
    cudaGraph_t*               subgraphs_;
};

// Injects kernel_record_timestamp nodes at compute-start and compute-end positions.
//
// Start position: immediately after the wait kernel (or before the first compute node if there
// is no wait), so the timestamp captures when actual computation begins.
// End position: immediately before the signal kernel, so the timestamp captures when
// actual computation ends.
//
// Self-contained: does not depend on WaitTimeDecorator or ComputeTimeDecorator.
// Requires SubgraphInjector to be in the inner chain (ctx.task_wait_node must be populated).
//
// After inject(), ctx.d_timestamps holds a device buffer with 2 entries per task:
//   ctx.d_timestamps[task * 2 + 0]  = compute-start __globaltimer() in nanoseconds
//   ctx.d_timestamps[task * 2 + 1]  = compute-end   __globaltimer() in nanoseconds
// To get absolute wall-clock times, correlate with a reference pair recorded at run start:
//   wall_ns = base_wall_ns + (d_timestamps[...] - base_globaltimer)
class TimestampDecorator : public IInjector
{
   public:
    TimestampDecorator(std::unique_ptr<IInjector> inner, cudaGraph_t* subgraphs)
        : inner_(std::move(inner)), subgraphs_(subgraphs)
    {
    }

    void inject(const std::vector<int>& tasks, InjectionContext& ctx) override
    {
        inner_->inject(tasks, ctx);

        int totalNodes = (int)ctx.task_wait_node.size();
        checkCudaErrors(cudaMalloc(&ctx.d_timestamps, sizeof(unsigned long long) * totalNodes * 2));
        checkCudaErrors(
            cudaMemset(ctx.d_timestamps, 0, sizeof(unsigned long long) * totalNodes * 2));

        for (int task : tasks)
        {
            cudaGraph_t sg = subgraphs_[task];

            // --- Compute-start timestamp (after wait, before compute) ---
            cudaGraphNode_t     tsStartNode;
            unsigned long long* start_ptr    = ctx.d_timestamps + task * 2 + 0;
            void*               startArgs[1] = {&start_ptr};
            auto tsStartParams = makeKernelParams((void*)kernel_record_timestamp, startArgs);

            if (ctx.task_wait_node[task] != nullptr)
            {
                cudaGraphNode_t waitNode = ctx.task_wait_node[task];
                checkCudaErrors(
                    cudaGraphAddKernelNode(&tsStartNode, sg, &waitNode, 1, &tsStartParams));

                // Rewire waitNode's other dependents to go through tsStartNode instead
                size_t numChildren;
                MUSTARD_cudaGraphNodeGetDependentNodes(waitNode, nullptr, &numChildren);
                std::vector<cudaGraphNode_t> children(numChildren);
                MUSTARD_cudaGraphNodeGetDependentNodes(waitNode, children.data(), &numChildren);
                for (auto& child : children)
                {
                    if (child == tsStartNode) continue;
                    MUSTARD_cudaGraphAddDependencies(sg, &tsStartNode, &child, 1);
                    MUSTARD_cudaGraphRemoveDependencies(sg, &waitNode, &child, 1);
                }
            }
            else
            {
                // No wait node: insert before all current roots
                auto roots = getRootNodes(sg);
                checkCudaErrors(
                    cudaGraphAddKernelNode(&tsStartNode, sg, nullptr, 0, &tsStartParams));
                for (auto& root : roots)
                    MUSTARD_cudaGraphAddDependencies(sg, &tsStartNode, &root, 1);
            }

            // --- Compute-end timestamp (after compute, before signal) ---
            cudaGraphNode_t signalNode = getSubgraphTail(sg);
            size_t          numParents;
            MUSTARD_cudaGraphNodeGetDependencies(signalNode, nullptr, &numParents);
            std::vector<cudaGraphNode_t> parents(numParents);
            MUSTARD_cudaGraphNodeGetDependencies(signalNode, parents.data(), &numParents);

            cudaGraphNode_t     tsEndNode;
            unsigned long long* end_ptr    = ctx.d_timestamps + task * 2 + 1;
            void*               endArgs[1] = {&end_ptr};
            auto tsEndParams = makeKernelParams((void*)kernel_record_timestamp, endArgs);

            checkCudaErrors(
                cudaGraphAddKernelNode(&tsEndNode, sg, parents.data(), numParents, &tsEndParams));
            for (auto& parent : parents)
                MUSTARD_cudaGraphRemoveDependencies(sg, &parent, &signalNode, 1);
            MUSTARD_cudaGraphAddDependencies(sg, &tsEndNode, &signalNode, 1);
        }
    }

   private:
    std::unique_ptr<IInjector> inner_;
    cudaGraph_t*               subgraphs_;
};

}  // namespace mustard
