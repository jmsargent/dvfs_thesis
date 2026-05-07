#pragma once

#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <vector>

#include <nvml.h>

#include "graph_assembler.h"
#include "partitioned_dag.h"
#include "task_profile_repository.h"

namespace Goal {
    inline double edp(uint n, uint m, double executionTime_ns, double energy_uj)
    {
        return std::pow(energy_uj, n) * std::pow(executionTime_ns / 1e9, m);
    }
}

class Tuner
{
   public:
    Tuner(const TaskProfileRepository& repo, const std::vector<DVFSSignal>& signals,
          PartitionedDag<TileAccess> dag, const int pe)
        : repo_(repo), signals_(signals), dag_(std::move(dag)), pe_(pe)
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
                edpByFreq[p.frequency_mhz] += Goal::edp(1, 1, p.execution_time_ns, p.energy_uj);

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
    std::vector<int>               plannedFreqs_;
    nvmlDevice_t                   nvmlDevice_;
};

