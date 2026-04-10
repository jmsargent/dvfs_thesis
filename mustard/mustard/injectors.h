#pragma once

#include <cuda_runtime.h>

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "mustard.h"    // MUSTARD_* macros, kernel_wait_static, kernel_signal_static, checkCudaErrors
#include "scheduler.h"  // StaticRoundRobinScheduler

namespace mustard
{

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
        for (auto& ev : compute_start)
            if (ev) cudaEventDestroy(ev);
        for (auto& ev : compute_end)
            if (ev) cudaEventDestroy(ev);
    }
};

class IInjector
{
   public:
    virtual void inject(const std::vector<int>& tasks, InjectionContext& ctx) = 0;
    virtual ~IInjector()                                                       = default;
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
                waitParams.func                 = (void*)kernel_wait_static;
                void* waitArgs[4] = {&d_deps, &n_deps, &d_completion_flags_, &debug_};
                waitParams.kernelParams         = waitArgs;
                checkCudaErrors(cudaGraphAddKernelNode(&waitNode, sg, nullptr, 0, &waitParams));
                ctx.task_wait_node[task] = waitNode;

                for (auto& root : roots)
                    MUSTARD_cudaGraphAddDependencies(sg, &waitNode, &root, 1);
            }

            int  n_notify     = scheduler_.getNotifyCount(task);
            int* d_notify_pes = scheduler_.getNotifyPEs(task);

            cudaGraphNode_t      tail = getSubgraphTail(sg);
            cudaGraphNode_t      signalNode;
            cudaKernelNodeParams signalParams = {0};
            signalParams.gridDim              = dim3(1);
            signalParams.blockDim             = dim3(1);
            signalParams.func                 = (void*)kernel_signal_static;
            int   task_id_val                 = task;
            void* signalArgs[5] = {&task_id_val, &d_completion_flags_, &d_notify_pes, &n_notify,
                                   &debug_};
            signalParams.kernelParams         = signalArgs;
            checkCudaErrors(cudaGraphAddKernelNode(&signalNode, sg, &tail, 1, &signalParams));
        }
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
                size_t numRoots;
                cudaGraphGetRootNodes(sg, nullptr, &numRoots);
                std::vector<cudaGraphNode_t> roots(numRoots);
                cudaGraphGetRootNodes(sg, roots.data(), &numRoots);
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

}  // namespace mustard
