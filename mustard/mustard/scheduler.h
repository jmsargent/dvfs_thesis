#pragma once

#include <queue>
#include <set>
#include <unordered_map>
#include <vector>

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
          task_deps_(subgraphDependencies),
          task_notify_pes_(totalNodes),
          d_task_deps_(totalNodes, nullptr),
          d_task_notify_pes_(totalNodes, nullptr),
          task_dep_count_(totalNodes, 0),
          task_notify_count_(totalNodes, 0),
          iter_(0)
    {
        distributeTasksRoundRobin(nPEs, totalNodes);
        buildDependentsMap(totalNodes, subgraphDependencies);
        topologicalSortDeviceTasks(myPE, subgraphDependencies);
        buildNotifyDeviceLists(myPE);
    }

    void distributeTasksRoundRobin(int nPEs, int totalNodes)
    {
        tasks_per_pe_.resize(nPEs);
        task_to_pe_.resize(totalNodes);
        for (int i = 0; i < totalNodes; i++)
        {
            tasks_per_pe_[i % nPEs].push_back(i);
            task_to_pe_[i] = i % nPEs;
        }
    }

    void buildDependentsMap(int totalNodes, const std::vector<std::vector<int>>& deps)
    {
        dependents_.resize(totalNodes);
        for (int j = 0; j < totalNodes; j++)
            for (int dep : deps[j]) dependents_[dep].push_back(j);
    }

    void topologicalSortDeviceTasks(int myPE, const std::vector<std::vector<int>>& deps)
    {
        std::unordered_map<int, int> in_degree;
        for (int task : tasks_per_pe_[myPE])
        {
            in_degree[task] = 0;
            for (int dep : deps[task])
                if (task_to_pe_[dep] == myPE) in_degree[task]++;
        }

        std::queue<int> ready;
        for (int task : tasks_per_pe_[myPE])
            if (in_degree[task] == 0) ready.push(task);

        while (!ready.empty())
        {
            int task = ready.front();
            ready.pop();
            sorted_pe_tasks_.push_back(task);
            for (int dependent : dependents_[task])
            {
                if (task_to_pe_[dependent] != myPE) continue;
                if (--in_degree[dependent] == 0) ready.push(dependent);
            }
        }
    }

    void buildNotifyDeviceLists(int myPE)
    {
        for (int task : tasks_per_pe_[myPE])
        {
            std::set<int> notify_set;
            for (int dep_task : dependents_[task]) notify_set.insert(task_to_pe_[dep_task]);
            task_notify_pes_[task].assign(notify_set.begin(), notify_set.end());
        }
    }

    const std::vector<int>& getMyTasksOrdered() const { return sorted_pe_tasks_; }

    bool getNextTask(int& task)
    {
        if (iter_ >= (int)sorted_pe_tasks_.size()) return false;
        task = sorted_pe_tasks_[iter_++];
        return true;
    }

    void resetIterator() { iter_ = 0; }

    // Host-side data exposed for the allocator
    const std::vector<int>& getDeps(int task) const { return task_deps_[task]; }
    const std::vector<int>& getNotifyPEs(int task) const { return task_notify_pes_[task]; }
    int                     getTotalNodes() const { return totalNodes_; }

    // Setters called by the allocator to wire in device pointers
    void setTaskDeps(int task, int* ptr, int count)
    {
        d_task_deps_[task]    = ptr;
        task_dep_count_[task] = count;
    }
    void setTaskNotifyPEs(int task, int* ptr, int count)
    {
        d_task_notify_pes_[task] = ptr;
        task_notify_count_[task] = count;
    }

    // Device pointer getters used by SubgraphInjector
    int* getDeviceDeps(int task) const { return d_task_deps_[task]; }
    int* getDeviceNotifyPEs(int task) const { return d_task_notify_pes_[task]; }
    int  getDepCount(int task) const { return task_dep_count_[task]; }
    int  getNotifyCount(int task) const { return task_notify_count_[task]; }

    int getTaskPE(int task) const { return task_to_pe_[task]; }

   private:
    int                           nPEs_, myPE_, totalNodes_;
    std::vector<std::vector<int>> tasks_per_pe_;
    std::vector<int>              task_to_pe_;
    std::vector<int>              sorted_pe_tasks_;
    std::vector<std::vector<int>> dependents_;
    std::vector<std::vector<int>> task_deps_;
    std::vector<std::vector<int>> task_notify_pes_;
    std::vector<int*>             d_task_deps_;
    std::vector<int*>             d_task_notify_pes_;
    std::vector<int>              task_dep_count_;
    std::vector<int>              task_notify_count_;
    int                           iter_;
};

}  // namespace mustard
