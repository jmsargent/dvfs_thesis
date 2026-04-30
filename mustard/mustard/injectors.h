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


class GraphInjector
{
   public:
    GraphInjector(cudaGraph_t sg) : sg_(sg) {}

    cudaGraph_t graph() const { return sg_; }

    // --- structural queries ---

    std::vector<cudaGraphNode_t> rootNodes()
    {
        return cudaGraphGetRootNodes(sg_);
    }

    // The tail node is the node with no outgoing edges
    // There can be multiple tail nodes
    cudaGraphNode_t tailNode()
    {
        // If there are no edges in graph => there is only one node
        auto nodes = cudaGraphGetNodes(sg_);
        if (nodes.size() == 1) return nodes.front();

        auto [from, to] = cudaGraphGetEdges(sg_);
        std::set<cudaGraphNode_t> hasOutgoing(from.begin(), from.end());
        
        for (auto& n : nodes)
            if (!hasOutgoing.count(n)) return n;
        
        
        return nodes.back();
    }

    // --- kernel node operations ---

    // Before:  original_node1 -> ...           After:  new_node -> original_node1 -> ...
    //          original_node2 -> ...                            -> original_node2 -> ...
    cudaGraphNode_t prependNode(const cudaKernelNodeParams& params)
    {
        auto            roots = rootNodes();
        cudaGraphNode_t node;
        checkCudaErrors(cudaGraphAddKernelNode(&node, sg_, nullptr, 0, &params));
        for (auto& root : roots) MUSTARD_cudaGraphAddDependencies(sg_, &node, &root, 1);
        return node;
    }

    // Before:  ... -> original_node1           After:  ... -> original_node1 -> new_node
    cudaGraphNode_t appendNode(const cudaKernelNodeParams& params)
    {
        cudaGraphNode_t tail = tailNode();
        cudaGraphNode_t node;
        checkCudaErrors(cudaGraphAddKernelNode(&node, sg_, &tail, 1, &params));
        return node;
    }

    // Before:  original_node1 -> original_node2 -> ...
    //                         -> original_node3 -> ...
    //
    // After:   original_node1 -> new_node -> original_node2 -> ...
    //                                     -> original_node3 -> ...
    cudaGraphNode_t insertAfterNode(cudaGraphNode_t             original_node1,
                                    const cudaKernelNodeParams& params)
    {
        cudaGraphNode_t node;
        checkCudaErrors(cudaGraphAddKernelNode(&node, sg_, &original_node1, 1, &params));
        auto children = cudaGraphNodeGetDependentNodes(original_node1);
        for (auto& child : children)
        {
            if (child == node) continue;
            MUSTARD_cudaGraphAddDependencies(sg_, &node, &child, 1);
            MUSTARD_cudaGraphRemoveDependencies(sg_, &original_node1, &child, 1);
        }
        return node;
    }

    // Before:  original_node1 -> original_node3 -> ...
    //          original_node2 ->
    //
    // After:   original_node1 -> new_node -> original_node3 -> ...
    //          original_node2 ->
    cudaGraphNode_t insertBeforeNode(cudaGraphNode_t             original_node3,
                                     const cudaKernelNodeParams& params)
    {
        auto            parents = cudaGraphNodeGetDependencies(original_node3);
        cudaGraphNode_t node;
        checkCudaErrors(cudaGraphAddKernelNode(&node, sg_, parents.data(), parents.size(), &params));
        for (auto& parent : parents)
            MUSTARD_cudaGraphRemoveDependencies(sg_, &parent, &original_node3, 1);
        MUSTARD_cudaGraphAddDependencies(sg_, &node, &original_node3, 1);
        return node;
    }

    // --- event record node operations (same structure as kernel variants above) ---

    // Before:  original_node1 -> ...           After:  new_node -> original_node1 -> ...
    //          original_node2 -> ...                            -> original_node2 -> ...
    cudaGraphNode_t prependEventNode(cudaEvent_t event)
    {
        auto            roots = rootNodes();
        cudaGraphNode_t node;
        checkCudaErrors(cudaGraphAddEventRecordNode(&node, sg_, nullptr, 0, event));
        for (auto& root : roots) MUSTARD_cudaGraphAddDependencies(sg_, &node, &root, 1);
        return node;
    }

    // Before:  original_node1 -> original_node2 -> ...
    //                         -> original_node3 -> ...
    //
    // After:   original_node1 -> new_node -> original_node2 -> ...
    //                                     -> original_node3 -> ...
    cudaGraphNode_t insertEventAfterNode(cudaGraphNode_t original_node1, cudaEvent_t event)
    {
        cudaGraphNode_t node;
        checkCudaErrors(cudaGraphAddEventRecordNode(&node, sg_, &original_node1, 1, event));
        auto children = cudaGraphNodeGetDependentNodes(original_node1);
        for (auto& child : children)
        {
            if (child == node) continue;
            MUSTARD_cudaGraphAddDependencies(sg_, &node, &child, 1);
            MUSTARD_cudaGraphRemoveDependencies(sg_, &original_node1, &child, 1);
        }
        return node;
    }

    // Before:  original_node1 -> original_node3 -> ...
    //          original_node2 ->
    //
    // After:   original_node1 -> new_node -> original_node3 -> ...
    //          original_node2 ->
    cudaGraphNode_t insertEventBeforeNode(cudaGraphNode_t original_node3, cudaEvent_t event)
    {
        auto            parents = cudaGraphNodeGetDependencies(original_node3);
        cudaGraphNode_t node;
        checkCudaErrors(
            cudaGraphAddEventRecordNode(&node, sg_, parents.data(), parents.size(), event));
        for (auto& parent : parents)
            MUSTARD_cudaGraphRemoveDependencies(sg_, &parent, &original_node3, 1);
        MUSTARD_cudaGraphAddDependencies(sg_, &node, &original_node3, 1);
        return node;
    }

   private:
    cudaGraph_t sg_;
};

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
    SubgraphInjector(cudaGraph_t* subgraphs, const StaticScheduler& scheduler,
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
            int*        d_deps = scheduler_.getDeviceDeps(task);

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
        int*  d_notify_pes  = scheduler_.getDeviceNotifyPEs(task);
        int   task_id_val   = task;
        void* signalArgs[5] = {&task_id_val, &d_completion_flags_, &d_notify_pes, &n_notify,
                               &debug_};
        auto  signalParams  = makeKernelParams((void*)kernel_signal_static, signalArgs);
        GraphInjector(sg).appendNode(signalParams);
    }

    void prependWaitNode(cudaGraph_t sg, int* d_deps, int n_deps, mustard::InjectionContext& ctx,
                         int task)
    {
        void* waitArgs[4]        = {&d_deps, &n_deps, &d_completion_flags_, &debug_};
        auto  waitParams         = makeKernelParams((void*)kernel_wait_static, waitArgs);
        ctx.task_wait_node[task] = GraphInjector(sg).prependNode(waitParams);
    }

   private:
    cudaGraph_t*           subgraphs_;
    const StaticScheduler& scheduler_;
    int*                   d_completion_flags_;
    int                    debug_;
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
            checkCudaErrors(cudaEventCreate(&ctx.compute_start[task]));
            GraphInjector gi(subgraphs_[task]);
            if (ctx.task_wait_node[task] != nullptr)
                gi.insertEventAfterNode(ctx.task_wait_node[task], ctx.compute_start[task]);
            else
                gi.prependEventNode(ctx.compute_start[task]);
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
            checkCudaErrors(cudaEventCreate(&ctx.compute_end[task]));
            GraphInjector gi(subgraphs_[task]);
            gi.insertEventBeforeNode(gi.tailNode(), ctx.compute_end[task]);
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

            GraphInjector   gi(subgraphs_[task]);
            cudaGraphNode_t waitNode = ctx.task_wait_node[task];

            unsigned long long* wait_start_ptr   = ctx.d_wait_timestamps + task * 2 + 0;
            void*               waitStartArgs[1] = {&wait_start_ptr};
            auto                tsWaitStartParams =
                makeKernelParams((void*)kernel_record_timestamp, waitStartArgs);
            gi.prependNode(tsWaitStartParams);

            unsigned long long* wait_end_ptr   = ctx.d_wait_timestamps + task * 2 + 1;
            void*               waitEndArgs[1] = {&wait_end_ptr};
            auto tsWaitEndParams = makeKernelParams((void*)kernel_record_timestamp, waitEndArgs);
            gi.insertAfterNode(waitNode, tsWaitEndParams);
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
            GraphInjector gi(subgraphs_[task]);

            unsigned long long* start_ptr    = ctx.d_timestamps + task * 2 + 0;
            void*               startArgs[1] = {&start_ptr};
            auto tsStartParams = makeKernelParams((void*)kernel_record_timestamp, startArgs);

            if (ctx.task_wait_node[task] != nullptr)
                gi.insertAfterNode(ctx.task_wait_node[task], tsStartParams);
            else
                gi.prependNode(tsStartParams);

            unsigned long long* end_ptr    = ctx.d_timestamps + task * 2 + 1;
            void*               endArgs[1] = {&end_ptr};
            auto tsEndParams = makeKernelParams((void*)kernel_record_timestamp, endArgs);
            gi.insertBeforeNode(gi.tailNode(), tsEndParams);
        }
    }

   private:
    std::unique_ptr<IInjector> inner_;
    cudaGraph_t*               subgraphs_;
};

}  // namespace mustard
