#pragma once

#include <cuda_runtime.h>

#include <set>
#include <vector>

#include "utils.h"

namespace mustard
{

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

}  // namespace mustard
