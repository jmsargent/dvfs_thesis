#pragma once

#include <limits>
#include <map>
#include <optional>
#include <vector>

#include <nvml.h>

#include "goal.h"
#include "graph_assembler.h"
#include "partitioned_dag.h"
#include "pe_writer.h"
#include "task_profile_repository.h"

class Tuner
{
   public:
    Tuner(const TaskProfileRepository& repo, const std::vector<DVFSSignal>& signals,
          PartitionedDag<TileAccess> dag, const int pe, const DVFSGoal& goal,
          std::optional<PEWriter> out = std::nullopt)
        : repo_(repo), signals_(signals), dag_(std::move(dag)), pe_(pe), goal_(goal), out_(std::move(out))
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
        int                      bestFreq;
        std::vector<std::string> nodeNames;
        std::optional<int>       nextNodeIndex;
    };

    void plan()
    {
        std::optional<int> idx = dag_.getRoot(pe_).index;
        while (idx)
        {
            auto [bestFreq, nodeNames, next] = optimizeNextInterval(*idx);
            plannedFreqs_.push_back(bestFreq);
            plannedNames_.push_back(std::move(nodeNames));
            idx = next;
        }

        if (out_) printPlan();
    }

    const std::vector<std::vector<std::string>>& plannedNames() const { return plannedNames_; }

   private:
    void printPlan()
    {
        auto& out = *out_;
        for (size_t i = 0; i < plannedFreqs_.size(); ++i)
        {
            out.print("interval %zu: %d MHz\n", i, plannedFreqs_[i]);
            for (const auto& node : plannedNames_[i])
                out.print("  * %s\n", node.c_str());
        }
    }


    IntervalPlan optimizeNextInterval(int nodeIndex)
    {
        auto nodes = dag_.untilIncomingXEdge(dag_.nodes()[nodeIndex]);

        std::map<int, std::pair<double, double>> totalsByFreq; // {energy_uj, time_ns}

        for (auto& node : nodes)
            for (auto& p : *repo_.getProfiles(node.content.op))
            {
                totalsByFreq[p.frequency_mhz].first  += p.energy_uj;
                totalsByFreq[p.frequency_mhz].second += p.execution_time_ns;
            }

        int bestFreq = std::min_element(totalsByFreq.begin(), totalsByFreq.end(),
                                        [&](auto& a, auto& b) {
                                            return goal_(a.second.second, a.second.first) <
                                                   goal_(b.second.second, b.second.first);
                                        })->first;

        std::vector<std::string> nodeNames;
        for (auto& node : nodes)
            nodeNames.push_back(node.content.op.toString());

        std::optional<int> next;
        if (!nodes.empty())
        {
            int idx = nodes.back().index;
            if (dag_.next(idx, nodes.back().partition)) next = idx;
        }

        return {bestFreq, std::move(nodeNames), next};
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
    const TaskProfileRepository&          repo_;
    const std::vector<DVFSSignal>&        signals_;
    PartitionedDag<TileAccess>            dag_;
    int                                   pe_;
    const DVFSGoal&                       goal_;
    std::optional<PEWriter>                         out_;
    std::vector<int>                      plannedFreqs_;
    std::vector<std::vector<std::string>> plannedNames_;
    nvmlDevice_t                   nvmlDevice_;
};

