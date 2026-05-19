#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "frequency_controller.h"
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
    virtual void onSignal(size_t signalIdx, int nodeIdx, KernelStatusUpdate status) = 0;
};

class FrequencyUpdateScheduler
{
   public:
    FrequencyUpdateScheduler(IFrequencyController& ctrl) : ctrl_(ctrl) {}

    ~FrequencyUpdateScheduler() { cancel(); }

    FrequencyUpdateScheduler(const FrequencyUpdateScheduler&)            = delete;
    FrequencyUpdateScheduler& operator=(const FrequencyUpdateScheduler&) = delete;

    void scheduleFrequencyUpdate(int freqMhz, std::chrono::nanoseconds delay)
    {
        if (delay.count() == 0)
        {
            ctrl_.setFrequency(freqMhz);
            return;
        }
        cancel();

        completed_.store(false);
        thread_ = std::thread(
            [this, freqMhz, delay]()
            {
                std::this_thread::sleep_for(delay);
                if (!completed_.load(std::memory_order_relaxed))
                    ctrl_.setFrequency(freqMhz);
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
    IFrequencyController& ctrl_;
    std::thread           thread_;
    std::atomic<bool>     completed_{true};
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
                    const DVFSGoal& goal, std::unique_ptr<IFrequencyController> ctrl,
                    std::optional<PEWriter> out = std::nullopt)
        : repo_(repo), dag_(std::move(dag)), pe_(pe), goal_(goal),
          ctrl_(std::move(ctrl)), out_(std::move(out))
    {}

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
            ctrl_->setFrequency(plannedFreqs_[0]);
    }

    void onSignal(size_t signalIdx, int, KernelStatusUpdate)
    {
        ctrl_->setFrequency(plannedFreqs_[signalIdx + 1]);
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

    const TaskProfileRepository&              repo_;
    PartitionedDag<TileAccess>                dag_;
    int                                       pe_;
    const DVFSGoal&                           goal_;
    std::unique_ptr<IFrequencyController>     ctrl_;
    std::optional<PEWriter>                   out_;
    std::vector<int>                          plannedFreqs_;
    std::vector<std::vector<std::string>>     plannedNames_;
};

class FrequencyScalerWaittimeDowntune : public IRuntimeEventHandle
{
   public:
    FrequencyScalerWaittimeDowntune(const TaskProfileRepository& repo,
                                    PartitionedDag<TileAccess> dag, int pe, const DVFSGoal& goal,
                                    const RetuneDelayTracker&             retuneDelayTracker,
                                    std::unique_ptr<IFrequencyController> ctrl,
                                    std::optional<PEWriter>               out = std::nullopt)
        : repo_(repo),
          dag_(std::move(dag)),
          pe_(pe),
          goal_(goal),
          retuneDelayTracker_(retuneDelayTracker),
          ctrl_(std::move(ctrl)),
          out_(std::move(out))
    {
        scheduler_ = std::make_unique<FrequencyUpdateScheduler>(*ctrl_);
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

        if (auto dev = ctrl_->device())
            idlePower_ = IdlePower::forDevice(*dev);

        if (!plannedFreqs_.empty())
            ctrl_->setFrequency(plannedFreqs_[0]);
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
    void onSignal(size_t signalIdx, int, KernelStatusUpdate status)
    {
        if (!scheduler_->didComplete() && status == KernelStatusUpdate::Running)
        {
            int bestFreq = reoptimizeCurrentInterval(signalIdx);
            scheduler_->scheduleFrequencyUpdate(bestFreq, 0ns);
        }
        if (status == KernelStatusUpdate::Completed)
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
            if (idlePower_) totals.first += idlePower_->energyUj(retune_ns);
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
    std::unique_ptr<IFrequencyController>     ctrl_;
    std::optional<PEWriter>                   out_;
    std::vector<int>                          plannedFreqs_;
    std::vector<int>                          plannedNodeIndices_;
    std::vector<std::vector<std::string>>     plannedNames_;
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

// Ramp-up tuner: starts at the lowest feasible frequency and incrementally
// retuning up, arriving at baseline just as the critical path enters this PE.
// The inverse of GreedyNpiDowntuner.
class CriticalPathRampUpScaler : public IRuntimeEventHandle
{
   public:
    CriticalPathRampUpScaler(const TaskProfileRepository&          repo,
                              PartitionedDag<TileAccess>            dag,
                              int                                   pe,
                              const RetuneDelayTracker&             retuneDelayTracker,
                              int                                   baselineFreq,
                              std::unique_ptr<IFrequencyController> ctrl,
                              std::optional<PEWriter>               out = std::nullopt)
        : repo_(repo),
          dag_(std::move(dag)),
          pe_(pe),
          retuneDelayTracker_(retuneDelayTracker),
          baselineFreq_(baselineFreq),
          ctrl_(std::move(ctrl)),
          out_(std::move(out))
    {}

    void init() override
    {
        spanned_ = makeSpannedDag(dag_, repo_, baselineFreq_);
        spanned_.computeSlack();

        findCriticalPathEntry();
        buildRampUpSchedule();

        if (out_) printPlan();
        ctrl_->setFrequency(startFreq_);
    }

    void onSignal(size_t, int nodeIdx, KernelStatusUpdate status) override
    {
        if (status != KernelStatusUpdate::Completed) return;
        auto it = rampUpSchedule_.find(nodeIdx);
        if (it != rampUpSchedule_.end())
            ctrl_->setFrequency(it->second);
    }

   private:
    // Finds the first zero-slack node on this PE — the point where this PE
    // is on the critical path and must be running at baseline frequency.
    void findCriticalPathEntry()
    {
        int nodeIdx = dag_.getRoot(pe_).index;
        do {
            if (spanned_.slack(nodeIdx) <= 0.0)
            {
                criticalPathEntry_ = nodeIdx;
                return;
            }
        } while (dag_.next(nodeIdx, pe_));
    }

    // Builds a ramp-up schedule by walking backwards from the CP entry boundary.
    // At each boundary, find the lowest frequency from which we can afford the
    // retune-up to the next target frequency within the slack budget at that node.
    // The root interval's starting frequency becomes startFreq_.
    void buildRampUpSchedule()
    {
        startFreq_ = baselineFreq_;
        if (!criticalPathEntry_) return;

        // Collect interval starts from root up to (not including) criticalPathEntry_
        std::vector<int> intervalStarts;
        int nodeIdx = dag_.getRoot(pe_).index;
        while (nodeIdx != *criticalPathEntry_)
        {
            intervalStarts.push_back(nodeIdx);
            auto interval = dag_.untilIncomingXEdge(dag_.nodes()[nodeIdx]);
            if (interval.empty()) break;
            int last = interval.back().index;
            if (!dag_.next(last, pe_)) break;
            nodeIdx = last;
        }

        if (intervalStarts.empty()) return;

        // Get available frequencies in ascending order from any interval node's profiles
        std::vector<int> availableFreqs;
        if (auto profs = repo_.getProfiles(dag_.nodes()[intervalStarts[0]].content.op))
            for (auto& p : *profs)
                availableFreqs.push_back(p.frequency_mhz);

        if (availableFreqs.empty()) return;

        // Returns the lowest frequency from which a retune to targetFreq fits in budget.
        // Returns targetFreq itself if nothing is feasible (no downtuning possible).
        auto lowestFeasibleFrom = [&](int targetFreq, double budget) -> int
        {
            for (int freq : availableFreqs)
            {
                if (freq >= targetFreq) break;
                double cost = (double)retuneDelayTracker_.getRetuneDelay(freq, targetFreq).count();
                if (cost <= budget)
                    return freq;
            }
            return targetFreq;
        };

        int targetFreq = baselineFreq_;

        // Determine what frequency we must exit the last NPI interval at so
        // that the final retune to baseline fits within the CP entry's slack
        {
            int fromFreq = lowestFeasibleFrom(targetFreq, spanned_.slack(*criticalPathEntry_));
            if (fromFreq != targetFreq)
                rampUpSchedule_[*criticalPathEntry_] = targetFreq;
            targetFreq = fromFreq;
        }

        // Walk backwards through NPI intervals: at each boundary, find lowest
        // feasible starting frequency and schedule the retune-up to targetFreq
        for (int i = (int)intervalStarts.size() - 1; i > 0; --i)
        {
            int intervalNode = intervalStarts[i];
            int fromFreq = lowestFeasibleFrom(targetFreq, spanned_.slack(intervalNode));
            if (fromFreq != targetFreq)
                rampUpSchedule_[intervalNode] = targetFreq;
            targetFreq = fromFreq;
        }

        startFreq_ = targetFreq;
    }

    const TaskProfileRepository&                 repo_;
    PartitionedDag<TileAccess>                   dag_;
    int                                          pe_;
    const RetuneDelayTracker&                    retuneDelayTracker_;
    int                                          baselineFreq_;
    std::unique_ptr<IFrequencyController>        ctrl_;
    SpannedPartitionedDag<TileOperation, double> spanned_;
    void printPlan()
    {
        for (auto& sn : spanned_.spannedNodes())
        {
            if (sn.partition != pe_) continue;
            double slackMs = spanned_.slack(sn.index) / 1e6;
            out_->print("slack node %d (%s): %.2f ms\n", sn.index,
                        sn.content.toString().c_str(), slackMs);
        }
        out_->print("start_freq: %d MHz\n", startFreq_);
        for (auto& [node, freq] : rampUpSchedule_)
            out_->print("  node %d (%s) -> %d MHz\n", node,
                        dag_.nodes()[node].content.op.toString().c_str(), freq);
    }

    std::optional<int>                           criticalPathEntry_;  // nodeIdx
    int                                          startFreq_{};
    std::map<int, int>                           rampUpSchedule_;  // nodeIdx -> freq
    std::optional<PEWriter>                      out_;
};

// Handles NPI gaps sandwiched between PI segments on this PE.
// During each gap, the GPU downtunes and must return to baseline before
// the critical path re-enters. Unlike GreedyNpiDowntuner, this
// class must account for both the retune-down and retune-up costs fitting
// within the gap's slack budget.
class NpiGapScaler : public IRuntimeEventHandle
{
   public:
    NpiGapScaler(const TaskProfileRepository&          repo,
                  PartitionedDag<TileAccess>            dag,
                  int                                   pe,
                  const RetuneDelayTracker&             retuneDelayTracker,
                  int                                   baselineFreq,
                  std::unique_ptr<IFrequencyController> ctrl,
                  std::optional<PEWriter>               out = std::nullopt)
        : repo_(repo),
          dag_(std::move(dag)),
          pe_(pe),
          retuneDelayTracker_(retuneDelayTracker),
          baselineFreq_(baselineFreq),
          ctrl_(std::move(ctrl)),
          out_(std::move(out))
    {}

    void init() override
    {
        spanned_ = makeSpannedDag(dag_, repo_, baselineFreq_);
        spanned_.computeSlack();
        buildGapSchedule();
        if (out_) printPlan();
        ctrl_->setFrequency(baselineFreq_);
    }

    void onSignal(size_t, int nodeIdx, KernelStatusUpdate status) override
    {
        if (status != KernelStatusUpdate::Completed) return;
        auto it = retuneSchedule_.find(nodeIdx);
        if (it != retuneSchedule_.end())
            ctrl_->setFrequency(it->second);
    }

   private:
    struct NpiGap
    {
        int gapStart;     // first NPI node of the gap (slack > 0, has x-partition incoming edge)
        int piReentry;    // first PI node after the gap (must be at baseline when its signal fires)
    };

    // Walk this PE's nodes and identify all NPI gaps: contiguous sequences of
    // slack > 0 nodes preceded by a PI segment, ending at the first PI node
    // whose cross-partition incoming edge comes from a critical-path node.
    // Both gap boundaries must have signals (x-partition incoming edges) so
    // that onSignal can trigger the retune events.
    std::vector<NpiGap> findNpiGaps()
    {
        std::vector<NpiGap> gaps;
        int  nodeIdx  = dag_.getRoot(pe_).index;
        bool inNpi    = false;
        int  gapStart = -1;

        do {
            bool isNpi = spanned_.slack(nodeIdx) > 0.0;

            if (!inNpi && isNpi)
            {
                // NPI node: gap opens if it has a cross-partition signal to trigger the downtune
                if (dag_.hasIncomingXPartition(dag_.nodes()[nodeIdx]))
                {
                    inNpi    = true;
                    gapStart = nodeIdx;
                }
            }
            else if (inNpi && !isNpi)
            {
                // NPI → PI transition: gap closes — record if CP arrives here via signal
                inNpi = false;
                for (int src : dag_.incomingEdges(nodeIdx))
                {
                    if (dag_.nodes()[src].partition != pe_ && spanned_.slack(src) <= 0.0)
                    {
                        gaps.push_back({gapStart, nodeIdx});
                        break;
                    }
                }
            }
        } while (dag_.next(nodeIdx, pe_));

        return gaps;
    }

    // One downtune + one uptune per gap. Frequency is flat across the entire gap.
    // For each gap, find the lowest frequency where
    // retune(baseline→low) + execDelta(low) + retune(low→baseline) ≤ slack(gapStart),
    // then schedule the symmetric downtune/uptune pair.
    void buildGapSchedule()
    {
        for (auto& gap : findNpiGaps())
        {
            std::map<int, double> execByFreq;
            int n = gap.gapStart;
            while (n != gap.piReentry)
            {
                if (auto profs = repo_.getProfiles(dag_.nodes()[n].content.op))
                    for (auto& p : *profs)
                        execByFreq[p.frequency_mhz] += p.execution_time_ns;
                if (!dag_.next(n, pe_)) break;
            }

            auto baseIt = execByFreq.find(baselineFreq_);
            if (execByFreq.empty() || baseIt == execByFreq.end()) continue;

            double budget = spanned_.slack(gap.gapStart);

            int lowFreq = baselineFreq_;
            for (auto& [freq, execTime] : execByFreq)
            {
                if (freq >= baselineFreq_) break;
                double retuneDown = (double)retuneDelayTracker_.getRetuneDelay(baselineFreq_, freq).count();
                double retuneUp   = (double)retuneDelayTracker_.getRetuneDelay(freq, baselineFreq_).count();
                double execDelta  = execTime - baseIt->second;
                if (retuneDown + execDelta + retuneUp <= budget)
                {
                    lowFreq = freq;
                    break;
                }
            }

            if (lowFreq != baselineFreq_)
            {
                retuneSchedule_[gap.gapStart]  = lowFreq;
                retuneSchedule_[gap.piReentry] = baselineFreq_;
            }
        }
    }

    void printPlan()
    {
        for (auto& sn : spanned_.spannedNodes())
        {
            if (sn.partition != pe_) continue;
            double slackMs = spanned_.slack(sn.index) / 1e6;
            out_->print("slack node %d (%s): %.2f ms\n", sn.index,
                        sn.content.toString().c_str(), slackMs);
        }
        for (auto& [node, freq] : retuneSchedule_)
            out_->print("node %d (%s) -> %d MHz\n", node,
                        dag_.nodes()[node].content.op.toString().c_str(), freq);
    }

    const TaskProfileRepository&                 repo_;
    PartitionedDag<TileAccess>                   dag_;
    int                                          pe_;
    const RetuneDelayTracker&                    retuneDelayTracker_;
    int                                          baselineFreq_;
    std::unique_ptr<IFrequencyController>        ctrl_;
    SpannedPartitionedDag<TileOperation, double> spanned_;
    std::map<int, int>                           retuneSchedule_;  // nodeIdx -> freq
    std::optional<PEWriter>                      out_;
};

// Like NpiGapScaler but instead of one flat frequency per gap, applies a
// backward-walk ramp-up within each gap. Downtunes once at the gap start and
// spreads incremental uptunes across the gap's sub-interval boundaries.
// By amortizing retune costs across multiple boundaries, it can afford a lower
// initial frequency than the flat single-pair strategy.
class NpiGapRampScaler : public IRuntimeEventHandle
{
   public:
    NpiGapRampScaler(const TaskProfileRepository&          repo,
                      PartitionedDag<TileAccess>            dag,
                      int                                   pe,
                      const RetuneDelayTracker&             retuneDelayTracker,
                      int                                   baselineFreq,
                      std::unique_ptr<IFrequencyController> ctrl,
                      std::optional<PEWriter>               out = std::nullopt)
        : repo_(repo),
          dag_(std::move(dag)),
          pe_(pe),
          retuneDelayTracker_(retuneDelayTracker),
          baselineFreq_(baselineFreq),
          ctrl_(std::move(ctrl)),
          out_(std::move(out))
    {}

    void init() override
    {
        spanned_ = makeSpannedDag(dag_, repo_, baselineFreq_);
        spanned_.computeSlack();
        buildSchedule();
        if (out_) printPlan();
        ctrl_->setFrequency(baselineFreq_);
    }

    void onSignal(size_t, int nodeIdx, KernelStatusUpdate status) override
    {
        if (status != KernelStatusUpdate::Completed) return;
        auto it = retuneSchedule_.find(nodeIdx);
        if (it != retuneSchedule_.end())
            ctrl_->setFrequency(it->second);
    }

   private:
    struct NpiGap
    {
        int gapStart;
        int piReentry;
    };

    std::vector<NpiGap> findNpiGaps()
    {
        std::vector<NpiGap> gaps;
        int  nodeIdx  = dag_.getRoot(pe_).index;
        bool inNpi    = false;
        int  gapStart = -1;

        do {
            bool isNpi = spanned_.slack(nodeIdx) > 0.0;

            if (!inNpi && isNpi)
            {
                if (dag_.hasIncomingXPartition(dag_.nodes()[nodeIdx]))
                {
                    inNpi    = true;
                    gapStart = nodeIdx;
                }
            }
            else if (inNpi && !isNpi)
            {
                inNpi = false;
                for (int src : dag_.incomingEdges(nodeIdx))
                {
                    if (dag_.nodes()[src].partition != pe_ && spanned_.slack(src) <= 0.0)
                    {
                        gaps.push_back({gapStart, nodeIdx});
                        break;
                    }
                }
            }
        } while (dag_.next(nodeIdx, pe_));

        return gaps;
    }

    // For each gap, apply a backward walk across the gap's sub-intervals.
    // One downtune at gapStart, incremental uptunes at each sub-interval
    // boundary, arriving at baseline by piReentry.
    void buildSchedule()
    {
        for (auto& gap : findNpiGaps())
        {
            std::vector<int> intervalStarts;
            int n = gap.gapStart;
            while (n != gap.piReentry)
            {
                intervalStarts.push_back(n);
                auto interval = dag_.untilIncomingXEdge(dag_.nodes()[n]);
                if (interval.empty()) break;
                int last = interval.back().index;
                if (!dag_.next(last, pe_)) break;
                n = last;
            }

            if (intervalStarts.empty()) continue;

            std::vector<int> availableFreqs;
            if (auto profs = repo_.getProfiles(dag_.nodes()[intervalStarts[0]].content.op))
                for (auto& p : *profs)
                    availableFreqs.push_back(p.frequency_mhz);

            if (availableFreqs.empty()) continue;

            auto lowestFeasibleFrom = [&](int targetFreq, double budget) -> int
            {
                for (int freq : availableFreqs)
                {
                    if (freq >= targetFreq) break;
                    double cost = (double)retuneDelayTracker_.getRetuneDelay(freq, targetFreq).count();
                    if (cost <= budget)
                        return freq;
                }
                return targetFreq;
            };

            int targetFreq = baselineFreq_;

            // Retune to baseline at piReentry if its slack allows
            {
                int fromFreq = lowestFeasibleFrom(targetFreq, spanned_.slack(gap.piReentry));
                if (fromFreq != targetFreq)
                    retuneSchedule_[gap.piReentry] = targetFreq;
                targetFreq = fromFreq;
            }

            // Backward walk through sub-intervals: spread uptune cost across boundaries
            for (int i = (int)intervalStarts.size() - 1; i > 0; --i)
            {
                int intervalNode = intervalStarts[i];
                int fromFreq = lowestFeasibleFrom(targetFreq, spanned_.slack(intervalNode));
                if (fromFreq != targetFreq)
                    retuneSchedule_[intervalNode] = targetFreq;
                targetFreq = fromFreq;
            }

            // One big downtune at gap start
            if (targetFreq != baselineFreq_)
                retuneSchedule_[gap.gapStart] = targetFreq;
        }
    }

    void printPlan()
    {
        for (auto& sn : spanned_.spannedNodes())
        {
            if (sn.partition != pe_) continue;
            double slackMs = spanned_.slack(sn.index) / 1e6;
            out_->print("slack node %d (%s): %.2f ms\n", sn.index,
                        sn.content.toString().c_str(), slackMs);
        }
        for (auto& [node, freq] : retuneSchedule_)
            out_->print("node %d (%s) -> %d MHz\n", node,
                        dag_.nodes()[node].content.op.toString().c_str(), freq);
    }

    const TaskProfileRepository&                 repo_;
    PartitionedDag<TileAccess>                   dag_;
    int                                          pe_;
    const RetuneDelayTracker&                    retuneDelayTracker_;
    int                                          baselineFreq_;
    std::unique_ptr<IFrequencyController>        ctrl_;
    SpannedPartitionedDag<TileOperation, double> spanned_;
    std::map<int, int>                           retuneSchedule_;  // nodeIdx -> freq
    std::optional<PEWriter>                      out_;
};

class GreedyNpiDowntuner : public IRuntimeEventHandle
{
   public:
    GreedyNpiDowntuner(const TaskProfileRepository&          repo,
                       PartitionedDag<TileAccess>            dag,
                       int                                   pe,
                       const DVFSGoal&                       goal,
                       const RetuneDelayTracker&             retuneDelayTracker,
                       int                                   baselineFreq,
                       std::unique_ptr<IFrequencyController> ctrl,
                       std::optional<PEWriter>               out = std::nullopt)
        : repo_(repo),
          dag_(std::move(dag)),
          pe_(pe),
          goal_(goal),
          retuneDelayTracker_(retuneDelayTracker),
          baselineFreq_(baselineFreq),
          ctrl_(std::move(ctrl)),
          out_(std::move(out))
    {}

    void init() override
    {
        spanned_ = makeSpannedDag(dag_, repo_, baselineFreq_);
        spanned_.computeSlack();

        findFirstNpiNode();
        buildRetuneSchedule();

        if (out_) printPlan();

        // If firstNpiNode_ has no cross-partition incoming edges, no signal fires for it —
        // apply its scheduled frequency immediately. Otherwise onSignal handles it.
        int initFreq = baselineFreq_;
        if (firstNpiNode_ && !dag_.hasIncomingXPartition(dag_.nodes()[*firstNpiNode_]))
        {
            auto it = retuneSchedule_.find(*firstNpiNode_);
            if (it != retuneSchedule_.end())
                initFreq = it->second;
        }

        ctrl_->setFrequency(initFreq);
    }

    void onSignal(size_t, int nodeIdx, KernelStatusUpdate status) override
    {
        if (status != KernelStatusUpdate::Completed) return;
        auto it = retuneSchedule_.find(nodeIdx);
        if (it != retuneSchedule_.end())
            ctrl_->setFrequency(it->second);
    }

   private:
    // Find the first NPI node (slack > 0) reachable from the root of this PE.
    void findFirstNpiNode()
    {
        firstNpiNode_ = std::nullopt;
        int nodeIdx = dag_.getRoot(pe_).index;
        do {
            if (spanned_.slack(nodeIdx) > 0.0)
            {
                firstNpiNode_ = nodeIdx;
                break;
            }
        } while (dag_.next(nodeIdx, pe_));
    }

    // Build a retune schedule for the leading NPI region before the first PI node.
    // One downtune at the NPI region start, then a backward-walk ramp-up to arrive
    // back at baseline before the first zero-slack (PI) node.
    void buildRetuneSchedule()
    {
        if (!firstNpiNode_) return;

        // Collect interval start nodes in the NPI prefix, stopping at the first PI node.
        std::vector<int> intervalStarts;
        int nodeIdx = *firstNpiNode_;
        while (true)
        {
            if (spanned_.slack(nodeIdx) <= 0.0) break;
            intervalStarts.push_back(nodeIdx);
            auto interval = dag_.untilIncomingXEdge(dag_.nodes()[nodeIdx]);
            if (interval.empty()) break;
            int nextIdx = interval.back().index;
            if (!dag_.next(nextIdx, pe_)) break;
            nodeIdx = nextIdx;
        }

        if (intervalStarts.empty()) return;

        std::vector<int> availableFreqs;
        if (auto profs = repo_.getProfiles(dag_.nodes()[intervalStarts[0]].content.op))
            for (auto& p : *profs)
                availableFreqs.push_back(p.frequency_mhz);
        if (availableFreqs.empty()) return;

        // Returns the lowest frequency from which a retune to targetFreq fits in budget.
        auto lowestFeasibleFrom = [&](int targetFreq, double budget) -> int
        {
            for (int freq : availableFreqs)
            {
                if (freq >= targetFreq) break;
                double cost = (double)retuneDelayTracker_.getRetuneDelay(freq, targetFreq).count();
                if (cost <= budget)
                    return freq;
            }
            return targetFreq;
        };

        int targetFreq = baselineFreq_;

        // Walk backward through the NPI intervals, spreading uptune costs across boundaries.
        for (int i = (int)intervalStarts.size() - 1; i > 0; --i)
        {
            int node = intervalStarts[i];
            int fromFreq = lowestFeasibleFrom(targetFreq, spanned_.slack(node));
            if (fromFreq != targetFreq)
                retuneSchedule_[node] = targetFreq;
            targetFreq = fromFreq;
        }

        // One big downtune at the NPI region start.
        if (targetFreq != baselineFreq_)
            retuneSchedule_[intervalStarts[0]] = targetFreq;
    }

    void printPlan()
    {
        for (auto& sn : spanned_.spannedNodes())
        {
            if (sn.partition != pe_) continue;
            double slackMs = spanned_.slack(sn.index) / 1e6;
            out_->print("slack node %d (%s): %.2f ms\n", sn.index,
                        sn.content.toString().c_str(), slackMs);
        }
        for (auto& [node, freq] : retuneSchedule_)
            out_->print("node %d (%s) -> %d MHz\n", node,
                        dag_.nodes()[node].content.op.toString().c_str(), freq);
    }

    const TaskProfileRepository&                   repo_;
    PartitionedDag<TileAccess>                     dag_;
    int                                            pe_;
    const DVFSGoal&                                goal_;
    const RetuneDelayTracker&                      retuneDelayTracker_;
    int                                            baselineFreq_;
    std::unique_ptr<IFrequencyController>          ctrl_;
    SpannedPartitionedDag<TileOperation, double>   spanned_;
    std::optional<int>                             firstNpiNode_;
    std::map<int, int>                             retuneSchedule_;  // nodeIdx -> freq
    std::optional<PEWriter>                        out_;
};

// Combined slack-aware scaler implementing a three-zone strategy per PE.
// All frequency transitions are ordered events in freqQueue_ (deque of (nodeId, freq)).
// nodeId == -1 is a sentinel for the zone A downtune, consumed once in init().
// During runtime, onSignal pops the front of the queue when the signalling node matches.
//   Zone A (1x): NPI prefix before the first CP entry. Downtune at sentinel -1, applied
//                in init(); backward-walk ramp-up schedules uptunes on NPI nodes.
//   Zone B (0+): NPI gap(s) between consecutive CP segments. Downtune on region[0],
//                uptune(s) from backward walk. Closing PI node carries a safety reset.
//   Zone C (1x): trailing NPI suffix after the last CP exit. Step-downs assigned
//                to their estimated nodes; no uptune needed.
class CombinedSlackAwareFrequencyScaler : public IRuntimeEventHandle
{
   public:
    CombinedSlackAwareFrequencyScaler(const TaskProfileRepository&          repo,
                                      PartitionedDag<TileAccess>            dag,
                                      int                                   pe,
                                      const RetuneDelayTracker&             retuneDelayTracker,
                                      int                                   baselineFreq,
                                      std::unique_ptr<IFrequencyController> ctrl,
                                      std::optional<PEWriter>               out = std::nullopt)
        : repo_(repo),
          dag_(std::move(dag)),
          pe_(pe),
          retuneDelayTracker_(retuneDelayTracker),
          baselineFreq_(baselineFreq),
          ctrl_(std::move(ctrl)),
          out_(std::move(out))
    {}

    void init() override
    {
        spanned_ = makeSpannedDag(dag_, repo_, baselineFreq_);
        spanned_.computeSlack();
        buildSchedule();
        if (out_) printPlan();
        int zoneAFreq = baselineFreq_;
        if (!freqQueue_.empty() && freqQueue_.front().first == -1)
        {
            zoneAFreq = freqQueue_.front().second;
            freqQueue_.pop_front();
        }
        currentFreq_ = zoneAFreq;
        ctrl_->setFrequency(zoneAFreq);
    }

    void onSignal(size_t, int nodeIdx, KernelStatusUpdate status) override
    {
        if (status != KernelStatusUpdate::Completed) return;
        if (freqQueue_.empty() || freqQueue_.front().first != nodeIdx) return;
        int freq = freqQueue_.front().second;
        freqQueue_.pop_front();
        if (freq != currentFreq_)
        {
            currentFreq_ = freq;
            ctrl_->setFrequency(freq);
        }
    }

   private:
    using FreqEvent = std::pair<int, int>;  // (nodeId, freq); nodeId == -1 means zone A init

    std::vector<int> getAvailableFreqs(int nodeIdx) const
    {
        std::vector<int> freqs;
        if (auto profs = repo_.getProfiles(dag_.nodes()[nodeIdx].content.op))
            for (auto& p : *profs)
                freqs.push_back(p.frequency_mhz);
        return freqs;
    }

    std::chrono::nanoseconds effectiveBudget(int nodeIdx) const
    {
        double s = spanned_.slack(nodeIdx);
        if (dag_.hasXPartitionEdge(dag_[nodeIdx]))
            s /= dag_.nrPartitions();
        return std::chrono::nanoseconds(static_cast<long long>(s));
    }

    std::chrono::nanoseconds execTime(int nodeIdx, int freqMhz) const
    {
        auto profs = repo_.getProfiles(dag_[nodeIdx].content.op);
        if (!profs) return 0ns;
        for (auto& p : *profs)
            if (p.frequency_mhz == freqMhz)
                return std::chrono::nanoseconds(static_cast<long long>(p.execution_time_ns));
        return 0ns;
    }

    std::chrono::nanoseconds regionSlowdown(const std::vector<int>& region, int tuneFreq) const
    {
        std::chrono::nanoseconds delta = 0ns;
        for (int nodeIdx : region)
            delta += execTime(nodeIdx, tuneFreq) - execTime(nodeIdx, baselineFreq_);
        return std::max(0ns, delta);
    }

    // All PE nodes that have a cross-partition incoming edge, sorted by span.start.
    // This visits parallel/isolated nodes that dag_.next traversal would miss.
    std::vector<int> collectSchedulingPoints() const
    {
        std::vector<int> pts;
        for (auto& n : dag_.nodes())
        {
            if (n.partition != pe_) continue;
            if (!dag_.hasIncomingXPartition(n)) continue;
            pts.push_back(n.index);
        }
        std::sort(pts.begin(), pts.end(), [this](int a, int b) {
            return spanned_.spannedNodes()[a].span.start < spanned_.spannedNodes()[b].span.start;
        });
        return pts;
    }

    // Zone A: prefix before the first critical path entry.
    // Emits a (-1, freq) sentinel consumed in init(). Uses the same single downtune +
    // single uptune forward check as Zone B: downCost fits in region.front()'s budget,
    // total debt fits in the tune-up node's budget.
    void buildZoneA(const std::vector<int>& region, const std::vector<int>& availableFreqs,
                    int piNodeIdx, std::vector<FreqEvent>& out)
    {
        if (region.empty()) return;

        int tuneUpNode;
        if (region.size() > 1)
            tuneUpNode = region.back();
        else
            tuneUpNode = piNodeIdx;

        auto downBudget = effectiveBudget(region.front());
        auto upBudget   = effectiveBudget(tuneUpNode);

        int tuneFreq = baselineFreq_;
        for (int freq : availableFreqs)
        {
            if (freq >= baselineFreq_) break;
            auto downCost = retuneDelayTracker_.getRetuneDelay(baselineFreq_, freq);
            auto upCost   = retuneDelayTracker_.getRetuneDelay(freq, baselineFreq_);
            auto slowdown = regionSlowdown(region, freq);
            if (downCost <= downBudget && downCost + slowdown + upCost <= upBudget)
            {
                tuneFreq = freq;
                break;
            }
        }

        if (tuneFreq == baselineFreq_) return;
        out.push_back({-1, tuneFreq});
        if (region.size() > 1)
            out.push_back({region.back(), baselineFreq_});
    }

    /*

        Just exiting critical path => frequency is at baseline
        We want to downtune, we know there is another upcomming critical path.

        Therefore what we are attempting to optimize here is, how can we save as much energy as possible,
        within the slack budget that we have. The cost here will have to include a downtune and an uptune

        We do not want to tune up when the interval however we could assert that the previous scheduling decision was a tune to baseline

        Nodes with cross-partition edges use slack / nrPartitions() as their budget to bound the
        debt that propagates to other PEs. This is conservative but correct: at most nrPartitions()-1
        PEs can affect any given node, so each PE's share never exceeds total available slack.

    */
    void buildZoneB(const std::vector<int>& region, const std::vector<int>& availableFreqs,
                    int piNodeIdx, std::vector<FreqEvent>& out)
    {
        out.push_back({piNodeIdx, baselineFreq_});

        /*
            Find the biggest tune-down for the initial node such that we can match it with a tune-up
            at the last node of the slack interval.

            The backward walk previously used here was incorrect: slack values are static (pre-computed
            at baseline), but slowing down a node increases execution time and propagates debt forward
            through the DAG. A forward check on just the two endpoints is used instead, which is correct
            given that we only emit a single tune-down at region.front() and a single tune-up at
            region.back().

            For single-node regions region.front() == region.back(), but the two events fire at
            different nodes: tune-down on region[0] Completed, tune-up on piNodeIdx Completed.
            onSignal pops one event per Completed signal so this is safe.

            region can be empty when two consecutive critical path nodes appear with no slack nodes
            between them.
        */
        if (region.empty()) return;

        int tuneUpNode;
        if (region.size() > 1)
            tuneUpNode = region.back();
        else
            tuneUpNode = piNodeIdx;

        auto downBudget = effectiveBudget(region.front());
        auto upBudget   = effectiveBudget(tuneUpNode);

        int tuneFreq = baselineFreq_;
        for (int freq : availableFreqs)
        {
            if (freq >= baselineFreq_) break;
            auto downCost = retuneDelayTracker_.getRetuneDelay(baselineFreq_, freq);
            auto upCost   = retuneDelayTracker_.getRetuneDelay(freq, baselineFreq_);
            auto slowdown = regionSlowdown(region, freq);
            if (downCost <= downBudget && downCost + slowdown + upCost <= upBudget)
            {
                tuneFreq = freq;
                break;
            }
        }

        if (tuneFreq == baselineFreq_) return;
        out.push_back({region.front(), tuneFreq});
        if (region.back() != region.front())
            out.push_back({region.back(), baselineFreq_});
    }

    /*
        Here we do not have to worry about tuning up again.
        However we still have to worry about effective slack & we have to worry about how future slack is affected
        from spending slack 
    */
    void buildZoneC(const std::vector<int>& region, const std::vector<int>& availableFreqs,
                    std::vector<FreqEvent>& out)
    {
        if (region.empty()) return;
        auto budget  = effectiveBudget(region.front());
        int tuneFreq = baselineFreq_;
        for (int freq : availableFreqs)
        {
            if (freq >= baselineFreq_) break;
            auto downCost = retuneDelayTracker_.getRetuneDelay(baselineFreq_, freq);
            if (downCost <= budget)
            {
                tuneFreq = freq;
                break;
            }
        }
        if (tuneFreq == baselineFreq_) return;
        out.push_back({region.front(), tuneFreq});
    }

    void buildSchedule()
    {
        auto pts = collectSchedulingPoints();
        if (pts.empty()) return;

        auto availableFreqs = getAvailableFreqs(pts[0]);
        if (availableFreqs.empty()) return;

        std::vector<FreqEvent> events;
        std::vector<int>       region;
        bool                   seenPI = false;

        for (int nodeIdx : pts)
        {
            if (spanned_.slack(nodeIdx) > 0.0)
            {
                region.push_back(nodeIdx);
            }
            else
            {
                if (!seenPI)
                {
                    buildZoneA(region, availableFreqs, nodeIdx, events);
                    events.push_back({nodeIdx, baselineFreq_});  // safety for zone A boundary
                }
                else
                {
                    buildZoneB(region, availableFreqs, nodeIdx, events);
                }
                seenPI = true;
                region.clear();
            }
        }
        buildZoneC(region, availableFreqs, events);

        // Sort into execution order: -1 sentinel first, then by span.start.
        std::sort(events.begin(), events.end(), [this](const FreqEvent& a, const FreqEvent& b) {
            if (a.first == -1) return true;
            if (b.first == -1) return false;
            return spanned_.spannedNodes()[a.first].span.start
                 < spanned_.spannedNodes()[b.first].span.start;
        });

        freqQueue_.assign(events.begin(), events.end());
    }

    void printPlan()
    {
        for (auto& sn : spanned_.spannedNodes())
        {
            if (sn.partition != pe_) continue;
            double slackMs = spanned_.slack(sn.index) / 1e6;
            out_->print("slack node %d (%s): %.2f ms\n", sn.index,
                        sn.content.toString().c_str(), slackMs);
        }
        int zoneAFreq = baselineFreq_;
        if (!freqQueue_.empty() && freqQueue_.front().first == -1)
            zoneAFreq = freqQueue_.front().second;
        out_->print("zone_A_start_freq: %d MHz\n", zoneAFreq);

        auto pts = collectSchedulingPoints();
        out_->print("scheduling_points (execution order):\n");
        for (int idx : pts)
        {
            double      slackMs = spanned_.slack(idx) / 1e6;
            const char* kind    = (slackMs > 0.0) ? "NPI" : "PI ";
            out_->print("  %s node %d (%s) slack=%.2fms", kind, idx,
                        dag_.nodes()[idx].content.op.toString().c_str(), slackMs);
            for (auto& [nid, freq] : freqQueue_)
            {
                if (nid == idx)
                {
                    out_->print(" [retune=%d MHz]", freq);
                    break;
                }
            }
            out_->print("\n");
        }
    }

    const TaskProfileRepository&                 repo_;
    PartitionedDag<TileAccess>                   dag_;
    int                                          pe_;
    const RetuneDelayTracker&                    retuneDelayTracker_;
    int                                          baselineFreq_;
    std::unique_ptr<IFrequencyController>        ctrl_;
    SpannedPartitionedDag<TileOperation, double> spanned_;
    std::deque<FreqEvent>                        freqQueue_;  // ordered retune events; front consumed first
    int                                          currentFreq_{};
    std::optional<PEWriter>                      out_;
};