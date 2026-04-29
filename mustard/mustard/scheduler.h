#pragma once

#include <queue>
#include <set>
#include <unordered_map>
#include <vector>

namespace mustard
{

class StaticScheduler
{
   public:
    virtual ~StaticScheduler() = default;

    const std::vector<int>& getMyTasksOrdered() const { return sorted_pe_tasks_; }

    bool getNextTask(int& task)
    {
        if (iter_ >= (int)sorted_pe_tasks_.size()) return false;
        task = sorted_pe_tasks_[iter_++];
        return true;
    }

    void resetIterator() { iter_ = 0; }

    const std::vector<int>& getDeps(int task) const { return task_deps_[task]; }
    const std::vector<int>& getNotifyPEs(int task) const { return task_notify_pes_[task]; }
    int                     getTotalNodes() const { return totalNodes_; }

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

    int* getDeviceDeps(int task) const { return d_task_deps_[task]; }
    int* getDeviceNotifyPEs(int task) const { return d_task_notify_pes_[task]; }
    int  getDepCount(int task) const { return task_dep_count_[task]; }
    int  getNotifyCount(int task) const { return task_notify_count_[task]; }

    int getTaskPE(int task) const { return task_to_pe_[task]; }

   protected:
    StaticScheduler(int nPEs, int myPE, int totalNodes,
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
    }

    // Call after subclass has populated task_to_pe_ and tasks_per_pe_.
    void buildSchedule(const std::vector<std::vector<int>>& deps)
    {
        buildDependentsMap(deps);
        topologicalSortDeviceTasks(deps);
        buildNotifyDeviceLists();
    }

    int                           nPEs_, myPE_, totalNodes_;
    std::vector<std::vector<int>> tasks_per_pe_;
    std::vector<int>              task_to_pe_;

   private:
    void buildDependentsMap(const std::vector<std::vector<int>>& deps)
    {
        dependents_.resize(totalNodes_);
        for (int j = 0; j < totalNodes_; j++)
            for (int dep : deps[j]) dependents_[dep].push_back(j);
    }

    void topologicalSortDeviceTasks(const std::vector<std::vector<int>>& deps)
    {
        // Global Kahn's over all tasks so that cross-PE transitive deps are respected.
        // A task on another PE may depend on a same-PE task through a cross-PE chain;
        // restricting in_degree to same-PE direct edges lets those "hidden" dependencies
        // go undetected and can place a task before its transitive same-PE predecessor
        // on the same GPU stream, causing a spin-wait deadlock.
        std::vector<int> in_degree(totalNodes_, 0);
        for (int task = 0; task < totalNodes_; task++)
            for (int dep : deps[task]) in_degree[task]++;

        std::queue<int> ready;
        for (int task = 0; task < totalNodes_; task++)
            if (in_degree[task] == 0) ready.push(task);

        while (!ready.empty())
        {
            int task = ready.front();
            ready.pop();
            if (task_to_pe_[task] == myPE_) sorted_pe_tasks_.push_back(task);
            for (int dependent : dependents_[task])
                if (--in_degree[dependent] == 0) ready.push(dependent);
        }
    }

    void buildNotifyDeviceLists()
    {
        for (int task : tasks_per_pe_[myPE_])
        {
            std::set<int> notify_set;
            for (int dep_task : dependents_[task]) notify_set.insert(task_to_pe_[dep_task]);
            task_notify_pes_[task].assign(notify_set.begin(), notify_set.end());
        }
    }

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

class StaticRoundRobinScheduler : public StaticScheduler
{
   public:
    StaticRoundRobinScheduler(int nPEs, int myPE, int totalNodes,
                              const std::vector<std::vector<int>>& subgraphDependencies)
        : StaticScheduler(nPEs, myPE, totalNodes, subgraphDependencies)
    {
        tasks_per_pe_.resize(nPEs);
        task_to_pe_.resize(totalNodes);
        for (int i = 0; i < totalNodes; i++)
        {
            tasks_per_pe_[i % nPEs].push_back(i);
            task_to_pe_[i] = i % nPEs;
        }
        buildSchedule(subgraphDependencies);
    }
};

class StaticPanelScheduler : public StaticScheduler
{
   public:
    StaticPanelScheduler(int nPEs, int myPE, int totalNodes,
                         const std::vector<std::vector<int>>& subgraphDependencies,
                         const std::vector<int>&              taskToPanel)
        : StaticScheduler(nPEs, myPE, totalNodes, subgraphDependencies)
    {
        tasks_per_pe_.resize(nPEs);
        task_to_pe_.resize(totalNodes);
        for (int i = 0; i < totalNodes; i++)
        {
            int pe = taskToPanel[i] % nPEs;
            tasks_per_pe_[pe].push_back(i);
            task_to_pe_[i] = pe;
        }
        buildSchedule(subgraphDependencies);
    }
};

}  // namespace mustard
