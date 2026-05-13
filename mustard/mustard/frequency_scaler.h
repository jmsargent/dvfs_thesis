#pragma once

#include <nvml.h>

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "goal.h"
#include "graph_assembler.h"
#include "idle_power.h"
#include "partitioned_dag.h"
#include "pe_writer.h"
#include "retune_delay_tracker.h"
#include "spanned_partitioned_dag.h"
#include "task_profile_repository.h"

using namespace std::chrono_literals;

class IRuntimeEventHandle
{
   public:
    virtual ~IRuntimeEventHandle()                                 = default;
    virtual void init()                                            = 0;
    virtual void onSignal(size_t signalIdx, KernelStatusUpdate status) = 0;
};

class FrequencyUpdateScheduler
{
   public:
    FrequencyUpdateScheduler(nvmlDevice_t device) : device_(device) {}

    ~FrequencyUpdateScheduler() { cancel(); }

    FrequencyUpdateScheduler(const FrequencyUpdateScheduler&)            = delete;
    FrequencyUpdateScheduler& operator=(const FrequencyUpdateScheduler&) = delete;

    void scheduleFrequencyUpdate(int freqMhz, std::chrono::nanoseconds delay)
    {
        if (delay.count() == 0)
        {
            nvmlDeviceSetGpuLockedClocks(device_, freqMhz, freqMhz);
            return;
        }
        cancel();

        completed_.store(false);
        thread_ = std::thread(
            [this, freqMhz, delay]()
            {
                std::this_thread::sleep_for(delay);
                if (!completed_.load(std::memory_order_relaxed))
                    nvmlDeviceSetGpuLockedClocks(device_, freqMhz, freqMhz);
                completed_.store(true, std::memory_order_release);
            });
    }

    void cancel()
    {
        completed_.store(true);
        if (thread_.joinable()) thread_.join();
    }

    bool didComplete() const { return completed_.load(std::memory_order_acquire); }

   private:
    nvmlDevice_t      device_;
    std::thread       thread_;
    std::atomic<bool> completed_{true};
};

struct IntervalPlan
{
    int                      bestFreq;
    std::vector<std::string> nodeNames;
    std::optional<int>       nextNodeIndex;
};

class FrequencyScaler : public IRuntimeEventHandle
{
   public:
    FrequencyScaler(const TaskProfileRepository& repo, PartitionedDag<TileAccess> dag, int pe,
                    const DVFSGoal& goal, std::optional<PEWriter> out = std::nullopt)
        : repo_(repo), dag_(std::move(dag)), pe_(pe), goal_(goal), out_(std::move(out))
    {
        nvmlInit();
        nvmlDeviceGetHandleByIndex(pe_, &nvmlDevice_);
    }

    ~FrequencyScaler()
    {
        nvmlDeviceResetGpuLockedClocks(nvmlDevice_);
        nvmlShutdown();
    }

    FrequencyScaler(FrequencyScaler&&) = default;

    void init()
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

        if (!plannedFreqs_.empty())
            nvmlDeviceSetGpuLockedClocks(nvmlDevice_, plannedFreqs_[0], plannedFreqs_[0]);
    }

    void onSignal(size_t signalIdx, KernelStatusUpdate)
    {
        nvmlDeviceSetGpuLockedClocks(nvmlDevice_, plannedFreqs_[signalIdx + 1],
                                     plannedFreqs_[signalIdx + 1]);
    }


   private:
    void printPlan()
    {
        auto& out = *out_;
        for (size_t i = 0; i < plannedFreqs_.size(); ++i)
        {
            out.print("interval %zu: %d MHz\n", i, plannedFreqs_[i]);
            for (const auto& node : plannedNames_[i]) out.print("  * %s\n", node.c_str());
        }
    }

    IntervalPlan optimizeNextInterval(int nodeIndex)
    {
        auto nodes = dag_.untilIncomingXEdge(dag_.nodes()[nodeIndex]);

        std::map<int, std::pair<double, double>> totalsByFreq;  // {energy_uj, time_ns}

        for (auto& node : nodes)
            for (auto& p : *repo_.getProfiles(node.content.op))
            {
                totalsByFreq[p.frequency_mhz].first += p.energy_uj;
                totalsByFreq[p.frequency_mhz].second += p.execution_time_ns;
            }

        int bestFreq = std::min_element(totalsByFreq.begin(), totalsByFreq.end(),
                                        [&](auto& a, auto& b)
                                        {
                                            return goal_(a.second.second, a.second.first) <
                                                   goal_(b.second.second, b.second.first);
                                        })
                           ->first;

        std::vector<std::string> nodeNames;
        for (auto& node : nodes) nodeNames.push_back(node.content.op.toString());

        std::optional<int> next;
        if (!nodes.empty())
        {
            int idx = nodes.back().index;
            if (dag_.next(idx, nodes.back().partition)) next = idx;
        }

        return {bestFreq, std::move(nodeNames), next};
    }

    const TaskProfileRepository&          repo_;
    PartitionedDag<TileAccess>            dag_;
    int                                   pe_;
    const DVFSGoal&                       goal_;
    std::optional<PEWriter>               out_;
    std::vector<int>                      plannedFreqs_;
    std::vector<std::vector<std::string>> plannedNames_;
    nvmlDevice_t                          nvmlDevice_;
};

class FrequencyScalerWaittimeDowntune : public IRuntimeEventHandle
{
   public:
    FrequencyScalerWaittimeDowntune(const TaskProfileRepository& repo,
                                    PartitionedDag<TileAccess> dag, int pe, const DVFSGoal& goal,
                                    const RetuneDelayTracker& retuneDelayTracker,
                                    std::optional<PEWriter>   out = std::nullopt)
        : repo_(repo),
          dag_(std::move(dag)),
          pe_(pe),
          goal_(goal),
          retuneDelayTracker_(retuneDelayTracker),
          out_(std::move(out))
    {
        nvmlInit();
        nvmlDeviceGetHandleByIndex(pe_, &nvmlDevice_);
        scheduler_ = std::make_unique<FrequencyUpdateScheduler>(nvmlDevice_);
    }

    ~FrequencyScalerWaittimeDowntune()
    {
        nvmlDeviceResetGpuLockedClocks(nvmlDevice_);
        nvmlShutdown();
    }

    FrequencyScalerWaittimeDowntune(FrequencyScalerWaittimeDowntune&&) = default;

    void init()
    {
        std::optional<int> idx = dag_.getRoot(pe_).index;
        while (idx)
        {
            plannedNodeIndices_.push_back(*idx);
            auto [bestFreq, nodeNames, next] = optimizeNextInterval(*idx);
            plannedFreqs_.push_back(bestFreq);
            plannedNames_.push_back(std::move(nodeNames));
            idx = next;
        }

        if (out_) printPlan();

        idlePower_ = IdlePower::forDevice(nvmlDevice_);

        if (!plannedFreqs_.empty())
            nvmlDeviceSetGpuLockedClocks(nvmlDevice_, plannedFreqs_[0], plannedFreqs_[0]);
    }

    /*
        =====================================================================================================================
        On unexpectedly early return, minimize goal-function:

        total execution-time = \sum [ executionTime ] + retune_latency( from: CORE_FREQUENCY.MIN,
       to:freq ) total energy         = \sum [ energy        ] + static_energy( retune_latency(
       from: CORE_FREQUENCY.MIN, to: freq ) )

        =====================================================================================================================
        On wait-event:

        estimated_wait_time(node) retune_latency( from: current_freq, to: CORE_FREQUENCY.MIN )
                                + retune_latency( from: CORE_FREQUENCY.MIN, to: next_freq )
                                - estimated_wait_time(node)

        =====================================================================================================================
    */
    void onSignal(size_t signalIdx, KernelStatusUpdate status)
    {
        if (!scheduler_->didComplete() && status == KernelStatusUpdate::Running)
        {
            int bestFreq = reoptimizeCurrentInterval(signalIdx);
            scheduler_->scheduleFrequencyUpdate(bestFreq, 0ns);
        }
        if (status == KernelStatusUpdate::Waiting)
        {
            planNext(signalIdx);
        }
    }

    void planNext(size_t signalIdx)
    {
        const int FREQ_LOW_MHZ   = 210;
        const int currentFreqMhz = plannedFreqs_[signalIdx];
        const int nextFreqMhz    = plannedFreqs_[signalIdx + 1];

        const auto latencyTuneDown =
            retuneDelayTracker_.getRetuneDelay(currentFreqMhz, FREQ_LOW_MHZ);
        const auto latencyTuneUp = retuneDelayTracker_.getRetuneDelay(FREQ_LOW_MHZ, nextFreqMhz);

        const auto estWaitTime     = 0ns;
        const auto idleWaitingTime = latencyTuneUp + latencyTuneDown - estWaitTime;

        if (idleWaitingTime > 0ns)
        {
            scheduler_->scheduleFrequencyUpdate(nextFreqMhz, estWaitTime - latencyTuneUp);
            scheduler_->scheduleFrequencyUpdate(FREQ_LOW_MHZ, 0ns);
        }
        else
        {
            scheduler_->scheduleFrequencyUpdate(nextFreqMhz, 0ns);
        }
    }


   private:
    void printPlan()
    {
        auto& out = *out_;
        for (size_t i = 0; i < plannedFreqs_.size(); ++i)
        {
            out.print("interval %zu: %d MHz\n", i, plannedFreqs_[i]);
            for (const auto& node : plannedNames_[i]) out.print("  * %s\n", node.c_str());
        }
    }

    IntervalPlan optimizeNextInterval(int nodeIndex)
    {
        auto nodes = dag_.untilIncomingXEdge(dag_.nodes()[nodeIndex]);

        std::map<int, std::pair<double, double>> totalsByFreq;  // {energy_uj, time_ns}

        for (auto& node : nodes)
            for (auto& p : *repo_.getProfiles(node.content.op))
            {
                totalsByFreq[p.frequency_mhz].first += p.energy_uj;
                totalsByFreq[p.frequency_mhz].second += p.execution_time_ns;
            }

        int bestFreq = std::min_element(totalsByFreq.begin(), totalsByFreq.end(),
                                        [&](auto& a, auto& b)
                                        {
                                            return goal_(a.second.second, a.second.first) <
                                                   goal_(b.second.second, b.second.first);
                                        })
                           ->first;

        std::vector<std::string> nodeNames;
        for (auto& node : nodes) nodeNames.push_back(node.content.op.toString());

        std::optional<int> next;
        if (!nodes.empty())
        {
            int idx = nodes.back().index;
            if (dag_.next(idx, nodes.back().partition)) next = idx;
        }

        return {bestFreq, std::move(nodeNames), next};
    }

    int reoptimizeCurrentInterval(size_t intervalIdx)
    {
        const int FREQ_LOW_MHZ = 210;
        auto      nodes =
            dag_.untilIncomingXEdge(dag_.nodes()[plannedNodeIndices_[intervalIdx]]);

        std::map<int, std::pair<double, double>> totalsByFreq;  // {energy_uj, time_ns}
        for (auto& node : nodes)
            for (auto& p : *repo_.getProfiles(node.content.op))
            {
                totalsByFreq[p.frequency_mhz].first += p.energy_uj;
                totalsByFreq[p.frequency_mhz].second += p.execution_time_ns;
            }

        for (auto& [freq, totals] : totalsByFreq)
        {
            double retune_ns =
                static_cast<double>(retuneDelayTracker_.getRetuneDelay(FREQ_LOW_MHZ, freq).count());
            totals.second += retune_ns;
            totals.first += idlePower_->energyUj(retune_ns);
        }

        return std::min_element(totalsByFreq.begin(), totalsByFreq.end(),
                                [&](auto& a, auto& b) {
                                    return goal_(a.second.second, a.second.first) <
                                           goal_(b.second.second, b.second.first);
                                })
            ->first;
    }

    const TaskProfileRepository&              repo_;
    PartitionedDag<TileAccess>                dag_;
    int                                       pe_;
    const DVFSGoal&                           goal_;
    const RetuneDelayTracker&                 retuneDelayTracker_;
    std::optional<PEWriter>                   out_;
    std::vector<int>                          plannedFreqs_;
    std::vector<int>                          plannedNodeIndices_;
    std::vector<std::vector<std::string>>     plannedNames_;
    nvmlDevice_t                              nvmlDevice_;
    std::unique_ptr<FrequencyUpdateScheduler> scheduler_;
    std::unique_ptr<IdlePower>                idlePower_;
};


inline SpannedPartitionedDag<TileOperation, double>
makeSpannedDag(const PartitionedDag<TileAccess>& dag,
               const TaskProfileRepository&      repo,
               int                               baselineFreq)
{
    SpannedPartitionedDag<TileOperation, double> spanned;

    std::vector<double> est(dag.nodes().size(), 0.0);

    for (auto& n : dag.nodes())
    {
        for (int src : dag.incomingEdges(n.index))
        {
            auto p = repo.getProfile(dag.nodes()[src].content.op, baselineFreq);
            double parentEnd = est[src] + (p ? p->execution_time_ns : 0.0);
            est[n.index] = std::max(est[n.index], parentEnd);
        }

        auto p = repo.getProfile(n.content.op, baselineFreq);
        double dur = p ? p->execution_time_ns : 0.0;

        SpannedNode<TileOperation, double> sn;
        sn.partition = n.partition;
        sn.content   = n.content.op;
        sn.span      = {est[n.index], est[n.index] + dur};
        spanned.addNode(sn);
    }

    for (auto& e : dag.edges())
        spanned.addEdge(e);

    return spanned;
}

class SlackAwareFrequencyScaler : public IRuntimeEventHandle
{
   public:
    SlackAwareFrequencyScaler(const TaskProfileRepository& repo,
                               PartitionedDag<TileAccess>  dag,
                               int                         pe,
                               const RetuneDelayTracker&   retuneDelayTracker,
                               int                         baselineFreq)
        : repo_(repo),
          dag_(std::move(dag)),
          pe_(pe),
          retuneDelayTracker_(retuneDelayTracker),
          baselineFreq_(baselineFreq)
    {}

    void init() override
    {
        auto spanned = makeSpannedDag(dag_, repo_, baselineFreq_);
        spanned.computeSlack();
    }

    /*
        There are 2 kinds of intra-partition-intervals (IPI) in the PDAG
        PI: IPIs that preceede critical path
        NPI: IPIs that do not ( These are most interesting to retune )

        NPI: does not need to tune back before entering critical path

        We need a model and a strategy for retuning
    */

    void onSignal(size_t signalIdx, KernelStatusUpdate) override {}

   private:
    const TaskProfileRepository& repo_;
    PartitionedDag<TileAccess>   dag_;
    int                          pe_;
    const RetuneDelayTracker&    retuneDelayTracker_;
    int                          baselineFreq_;
};

