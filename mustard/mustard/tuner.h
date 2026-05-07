#pragma once

#include <limits>
#include <map>
#include <optional>
#include <vector>

#include <nvml.h>

#include "goal.h"
#include "graph_assembler.h"
#include "partitioned_dag.h"
#include "task_profile_repository.h"

class Tuner
{
   public:
    Tuner(const TaskProfileRepository& repo, const std::vector<DVFSSignal>& signals,
          PartitionedDag<TileAccess> dag, const int pe, const DVFSGoal& goal)
        : repo_(repo), signals_(signals), dag_(std::move(dag)), pe_(pe), goal_(goal)
    {
        nvmlInit();
        nvmlDeviceGetHandleByIndex(pe_, &nvmlDevice_);
    }

    ~Tuner()
    {
        nvmlDeviceResetGpuLockedClocks(nvmlDevice_);
        nvmlShutdown();
    }

    struct IntervalPlan
    {
        int               bestFreq;
        std::optional<int> nextNodeIndex;
    };

    void plan()
    {
        plannedFreqs_.clear();

        std::optional<int> idx = dag_.getRoot(pe_).index;
        while (idx)
        {
            auto [bestFreq, next] = optimizeNextInterval(*idx);
            plannedFreqs_.push_back(bestFreq);
            idx = next;
        }
    }

   private:
    IntervalPlan optimizeNextInterval(int nodeIndex)
    {
        auto nodes = dag_.untilIncomingXEdge(dag_.nodes()[nodeIndex]);

        std::map<int, double> edpByFreq;
        for (auto& node : nodes)
            for (auto& p : *repo_.getProfiles(node.content.op))
                edpByFreq[p.frequency_mhz] += goal_(p.execution_time_ns, p.energy_uj);

        int bestFreq = std::min_element(edpByFreq.begin(), edpByFreq.end(),
                                        [](auto& a, auto& b) { return a.second < b.second; })->first;

        std::optional<int> next;
        if (!nodes.empty())
        {
            int idx = nodes.back().index;
            if (dag_.next(idx, nodes.back().partition)) next = idx;
        }

        return {bestFreq, next};
    }

   public:

    void run()
    {
        if (plannedFreqs_.empty()) return;

        nvmlDeviceSetGpuLockedClocks(nvmlDevice_, plannedFreqs_[0], plannedFreqs_[0]);

        for (size_t i = 1; i < plannedFreqs_.size(); ++i)
        {
            while (cudaEventQuery(signals_[i - 1].event) != cudaSuccess)
                ;
            nvmlDeviceSetGpuLockedClocks(nvmlDevice_, plannedFreqs_[i], plannedFreqs_[i]);
        }
    }

    void reset(const std::vector<cudaStream_t>& streams)
    {
        for (size_t i = 0; i < signals_.size(); ++i)
            cudaEventRecord(signals_[i].event, streams[i % streams.size()]);
    }

   private:
    const TaskProfileRepository&   repo_;
    const std::vector<DVFSSignal>& signals_;
    PartitionedDag<TileAccess>     dag_;
    int                            pe_;
    const DVFSGoal&                goal_;
    std::vector<int>               plannedFreqs_;
    nvmlDevice_t                   nvmlDevice_;
};

