#pragma once

#include <chrono>
#include <limits>
#include <map>
#include <memory>
#include <optional>
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
    virtual void reset()                                           {}
    virtual void onSignal(size_t signalIdx, int nodeIdx, KernelStatusUpdate status) = 0;
};

class ScalerNone : public IRuntimeEventHandle
{
   public:
    void init()    override {}
    void reset()   override {}
    void onSignal(size_t, int, KernelStatusUpdate) override {}
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

class ScalerSyncPoints : public IRuntimeEventHandle
{
   public:
    ScalerSyncPoints(const TaskProfileRepository&          repo,
                     PartitionedDag<TileAccess>            dag,
                     int                                   pe,
                     int                                   baselineFreq,
                     const DVFSGoal&                       goal,
                     std::unique_ptr<IFrequencyController> ctrl)
        : repo_(repo), dag_(std::move(dag)), pe_(pe), baselineFreq_(baselineFreq), goal_(goal), ctrl_(std::move(ctrl))
    {}

    void init() override
    {
        spanned_ = makeSpannedDag(dag_, repo_, baselineFreq_);
        auto dependents = dag_.getNodesWithXPartitionDependencies(pe_);
        for (auto& dependent : dependents)
        {
            auto beforeDependent = spanned_.predecessor(spanned_.spannedNodes()[dependent.index], pe_);
            if (beforeDependent)
                syncPairs_.push_back({beforeDependent->index, dependent.index});
        }

        auto cur = std::optional{spanned_.spannedNodes()[dag_.getRoot(pe_).index]};
        for (auto& [before, dep] : syncPairs_)
        {
            std::vector<int> interval;
            for (; cur && cur->index != dep; cur = spanned_.successor(*cur))
                interval.push_back(cur->index);
            interval.push_back(dep);
            cur = spanned_.successor(*cur);
            intervals_.push_back(interval);
        }

        if (!intervals_.empty())
            initFreq_ = findBestFreq(intervals_[0]);
        for (size_t i = 0; i + 1 < intervals_.size(); i++)
            freqPlan_.push_back({syncPairs_[i].first, findBestFreq(intervals_[i + 1])});
        printPlan();
        reset();
    }

    void reset() override
    {
        eventIdx_ = 0;
        ctrl_->setFrequency(initFreq_);
    }

    void onSignal(size_t, int nodeIdx, KernelStatusUpdate status) override
    {
        if (status != KernelStatusUpdate::Completed) return;
        if (eventIdx_ < freqPlan_.size() && freqPlan_[eventIdx_].first == nodeIdx)
            ctrl_->setFrequency(freqPlan_[eventIdx_++].second);
    }

   private:
    int findBestFreq(const std::vector<int>& interval) const
    {
        int    bestFreq  = baselineFreq_;
        double bestScore = std::numeric_limits<double>::max();
        auto*  profs     = repo_.getProfiles(dag_.nodes()[interval[0]].content.op);
        if (!profs) return bestFreq;
        for (auto& candidate : *profs)
        {
            double totalTime = 0, totalEnergy = 0;
            bool   valid = true;
            for (int idx : interval)
            {
                auto p = repo_.getProfile(dag_.nodes()[idx].content.op, candidate.frequency_mhz);
                if (!p) { valid = false; break; }
                totalTime   += p->execution_time_ns;
                totalEnergy += p->energy_uj;
            }
            if (!valid) continue;
            double score = goal_(totalTime, totalEnergy);
            if (score < bestScore) { bestScore = score; bestFreq = candidate.frequency_mhz; }
        }
        return bestFreq;
    }

    void printPlan() const
    {
        printf("[ScalerSyncPoints pe=%d] initFreq=%d MHz, nodes:", pe_, initFreq_);
        for (int idx : intervals_[0]) printf(" %d", idx);
        printf("\n");
        for (size_t i = 0; i < freqPlan_.size(); i++)
        {
            printf("  trigger=node %d -> freq=%d MHz, nodes:", freqPlan_[i].first, freqPlan_[i].second);
            for (int idx : intervals_[i + 1]) printf(" %d", idx);
            printf("\n");
        }
    }

    const TaskProfileRepository&                 repo_;
    PartitionedDag<TileAccess>                   dag_;
    int                                          pe_;
    int                                          baselineFreq_;
    const DVFSGoal&                              goal_;
    std::unique_ptr<IFrequencyController>        ctrl_;
    SpannedPartitionedDag<TileOperation, double> spanned_;
    std::vector<std::pair<int,int>>              syncPairs_;  // {beforeDependent, dependent}
    std::vector<std::vector<int>>                intervals_;
    int                                          initFreq_{};
    std::vector<std::pair<int,int>>              freqPlan_;   // {beforeDependent, freq}
    size_t                                       eventIdx_{0};
};


class ScalerConstantFrequency : public IRuntimeEventHandle
{
   public:
    ScalerConstantFrequency(int freq, std::unique_ptr<IFrequencyController> ctrl)
        : freq_(freq), ctrl_(std::move(ctrl))
    {}

    void init() override { ctrl_->setFrequency(freq_); }

    void onSignal(size_t, int, KernelStatusUpdate) override {}

   private:
    int                                   freq_;
    std::unique_ptr<IFrequencyController> ctrl_;
};


class BoostingFrequencyScaler : public IRuntimeEventHandle
{
   public:
    BoostingFrequencyScaler(int freq, std::unique_ptr<IFrequencyController> ctrl)
        : freq_(freq), ctrl_(std::move(ctrl))
    {}

    void init() override { ctrl_->setFrequency(freq_); }

    void onSignal(size_t, int, KernelStatusUpdate) override {}

   private:
    int                                   freq_;
    std::unique_ptr<IFrequencyController> ctrl_;
};


// Combined slack-aware scaler implementing a three-zone strategy per PE.
// buildSchedule() populates freqEvents_ (a persistent ring buffer) and zoneAFreq_.
// onSignal() advances eventIdx_ through freqEvents_ each iteration; wrap is detected
// when nodeIdx resets, at which point eventIdx_ and the zone-A frequency are restored.
//   Zone A (1x): slack prefix before the first CP entry. Downtune at sentinel, applied
//                in init(); uptune resets at the closing CP node (cross-partition incoming).
//   Zone B (0+): slack gap(s) between consecutive CP segments. Downtune on the preceding
//                CP node (if it has cross-partition incoming); uptune at the closing CP node.
//   Zone C (1x): trailing slack suffix after the last CP exit. Downtune on the preceding
//                CP node (if it has cross-partition incoming); no uptune needed.
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
        reset();
    }

    void reset() override
    {
        eventIdx_    = 0;
        currentFreq_ = zoneAFreq_;
        ctrl_->setFrequency(zoneAFreq_);
        if(out_) out_->print("reset");
    }

    void onSignal(size_t, int nodeIdx, KernelStatusUpdate status) override
    {
        if (status != KernelStatusUpdate::Completed) return;
        if (eventIdx_ < freqEvents_.size() && freqEvents_[eventIdx_].nodeId == nodeIdx)
        {
            int freq = freqEvents_[eventIdx_++].freq;
            if (freq != currentFreq_)
            {
                currentFreq_ = freq;
                ctrl_->setFrequency(freq);
                if(out_) out_->print("setting frequency to: %d\n", freq);
            }
        }
    }

   private:
    static constexpr int kSentinelNodeId = -1;

    struct FreqEvent {
        int nodeId;  // kSentinelNodeId means zone-A downtune, applied in init()
        int freq;
    };

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

    std::chrono::nanoseconds executionTimeSlowdown(int nodeIdx, int freqMhz) const
    {
        return std::max(0ns, execTime(nodeIdx, freqMhz) - execTime(nodeIdx, baselineFreq_));
    }

    // All nodes on this PE, sorted by span.start (execution order from baseline schedule).
    std::vector<int> collectSchedulingPoints() const
    {
        std::vector<int> pts;
        for (auto& n : dag_.nodes())
        {
            if (n.partition != pe_) continue;
            pts.push_back(n.index);
        }
        std::sort(pts.begin(), pts.end(), [this](int a, int b) {
            return spanned_.spannedNodes()[a].span.start < spanned_.spannedNodes()[b].span.start;
        });
        return pts;
    }

    // Returns the lowest frequency where downCost + cumulative slowdown + upCost fits every
    // node's effectiveBudget. Pass includeDownCost=false for zone A (no preceding retune);
    // pass includeUpCost=false for zone C (no subsequent retune). Returns baselineFreq_ if
    // no frequency qualifies.
    int findLowestFeasibleFreq(const std::vector<int>& region,
                               const std::vector<int>& availableFreqs,
                               bool                    includeDownCost,
                               bool                    includeUpCost) const
    {
        for (int freq : availableFreqs)
        {
            if (freq >= baselineFreq_) break;
            auto downCost = includeDownCost
                ? retuneDelayTracker_.getRetuneDelay(baselineFreq_, freq) : 0ns;
            auto upCost   = includeUpCost
                ? retuneDelayTracker_.getRetuneDelay(freq, baselineFreq_) : 0ns;
            if (downCost > effectiveBudget(region.front())) continue;
            auto debt     = downCost;
            bool feasible = true;
            for (int nodeIdx : region)
            {
                debt += executionTimeSlowdown(nodeIdx, freq);
                if (debt + upCost > effectiveBudget(nodeIdx))
                {
                    feasible = false;
                    break;
                }
            }
            if (feasible) return freq;
        }
        return baselineFreq_;
    }

    // Zone A: slack prefix before the first CP entry. No downtune cost; must fit uptune.
    void buildZoneA(const std::vector<int>& region, const std::vector<int>& availableFreqs,
                    std::vector<FreqEvent>& out)
    {
        if (region.empty()) return;
        int tuneFreq = findLowestFeasibleFreq(region, availableFreqs, false, true);
        if (tuneFreq == baselineFreq_) return;
        out.push_back({kSentinelNodeId, tuneFreq});
        out.push_back({region.back(), baselineFreq_});
    }

    // Zone B: slack gap between two CP segments. Downtune when lastCpNode completes;
    // uptune when region.back() completes, restoring baseline before the next CP node.
    // Nodes with cross-partition edges use slack/nrPartitions() as their effective budget.
    void buildZoneB(const std::vector<int>& region, const std::vector<int>& availableFreqs,
                    int lastCpNode, std::vector<FreqEvent>& out)
    {
        if (region.empty()) return;
        int tuneFreq = findLowestFeasibleFreq(region, availableFreqs, true, true);
        if (tuneFreq == baselineFreq_) return;
        out.push_back({lastCpNode, tuneFreq});
        out.push_back({region.back(), baselineFreq_});
    }

    // Zone C: trailing slack after the last CP exit. Downtune when lastCpNode completes;
    // no uptune required. Nodes with cross-partition edges use slack/nrPartitions().
    void buildZoneC(const std::vector<int>& region, const std::vector<int>& availableFreqs,
                    int lastCpNode, std::vector<FreqEvent>& out)
    {
        if (region.empty()) return;
        int tuneFreq = findLowestFeasibleFreq(region, availableFreqs, true, false);
        if (tuneFreq == baselineFreq_) return;
        out.push_back({lastCpNode, tuneFreq});
    }

    void buildSchedule()
    {
        auto pts = collectSchedulingPoints();
        if (pts.empty()) return;

        auto availableFreqs = getAvailableFreqs(pts[0]);
        if (availableFreqs.empty()) return;

        std::vector<FreqEvent> events;
        std::vector<int>       region;
        bool                   isInitialSegment = true;
        int                    lastCpNode       = -1;

        for (int nodeIdx : pts)
        {
            if (spanned_.slack(nodeIdx) > 0.0)
            {
                region.push_back(nodeIdx);
            }
            else
            {
                if (isInitialSegment)
                    buildZoneA(region, availableFreqs, events);
                else
                    buildZoneB(region, availableFreqs, lastCpNode, events);
                isInitialSegment = false;
                lastCpNode       = nodeIdx;
                region.clear();
            }
        }
        if (!isInitialSegment)
            buildZoneC(region, availableFreqs, lastCpNode, events);

        // Sort into execution order: sentinel first, then by span.start.
        std::sort(events.begin(), events.end(), [this](const FreqEvent& a, const FreqEvent& b) {
            if (a.nodeId == kSentinelNodeId) return true;
            if (b.nodeId == kSentinelNodeId) return false;
            return spanned_.spannedNodes()[a.nodeId].span.start
                 < spanned_.spannedNodes()[b.nodeId].span.start;
        });

        zoneAFreq_ = baselineFreq_;
        for (auto& ev : events)
        {
            if (ev.nodeId == kSentinelNodeId)
                zoneAFreq_ = ev.freq;
            else
                freqEvents_.push_back(ev);
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
        out_->print("zone_A_start_freq: %d MHz\n", zoneAFreq_);

        auto pts = collectSchedulingPoints();
        out_->print("scheduling_points (execution order):\n");
        for (int idx : pts)
        {
            double      slackMs = spanned_.slack(idx) / 1e6;
            const char* kind    = (slackMs > 0.0) ? "NPI" : "PI ";
            out_->print("  %s node %d (%s) slack=%.2fms", kind, idx,
                        dag_.nodes()[idx].content.op.toString().c_str(), slackMs);
            for (auto& ev : freqEvents_)
            {
                if (ev.nodeId == idx)
                {
                    out_->print(" [retune=%d MHz]", ev.freq);
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
    std::vector<FreqEvent>                       freqEvents_;
    size_t                                       eventIdx_{0};
    int                                          zoneAFreq_{};
    int                                          currentFreq_{};
    std::optional<PEWriter>                      out_;
};


class CombinedSlackAwareFrequencyScaler2 : public IRuntimeEventHandle
{
   public:
    CombinedSlackAwareFrequencyScaler2(const TaskProfileRepository&          repo,
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
        reset();
    }

    void reset() override
    {
        eventIdx_    = 0;
        currentFreq_ = zoneAFreq_;
        ctrl_->setFrequency(zoneAFreq_);
        if(out_) out_->print("reset");
    }

    void onSignal(size_t, int nodeIdx, KernelStatusUpdate status) override
    {
        if (status != KernelStatusUpdate::Completed) return;
        if (eventIdx_ < freqEvents_.size() && freqEvents_[eventIdx_].nodeId == nodeIdx)
        {
            int freq = freqEvents_[eventIdx_++].freq;
            if (freq != currentFreq_)
            {
                currentFreq_ = freq;
                ctrl_->setFrequency(freq);
                if(out_) out_->print("setting frequency to: %d\n", freq);
            }
        }
    }

   private:
    static constexpr int kSentinelNodeId = -1;

    struct FreqEvent {
        int nodeId;  // kSentinelNodeId means zone-A downtune, applied in init()
        int freq;
    };

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

    std::chrono::nanoseconds executionTimeSlowdown(int nodeIdx, int freqMhz) const
    {
        return std::max(0ns, execTime(nodeIdx, freqMhz) - execTime(nodeIdx, baselineFreq_));
    }

    // All nodes on this PE, sorted by span.start (execution order from baseline schedule).
    std::vector<int> collectSchedulingPoints() const
    {
        std::vector<int> pts;
        for (auto& n : dag_.nodes())
        {
            if (n.partition != pe_) continue;
            pts.push_back(n.index);
        }
        std::sort(pts.begin(), pts.end(), [this](int a, int b) {
            return spanned_.spannedNodes()[a].span.start < spanned_.spannedNodes()[b].span.start;
        });
        return pts;
    }

    // Returns the lowest frequency where downCost + cumulative slowdown + upCost fits every
    // node's effectiveBudget. Pass includeDownCost=false for zone A (no preceding retune);
    // pass includeUpCost=false for zone C (no subsequent retune). Returns baselineFreq_ if
    // no frequency qualifies.
    int findLowestFeasibleFreq(const std::vector<int>& region,
                               const std::vector<int>& availableFreqs,
                               bool                    includeDownCost,
                               bool                    includeUpCost) const
    {
        for (int freq : availableFreqs)
        {
            if (freq >= baselineFreq_) break;
            auto downCost = includeDownCost
                ? retuneDelayTracker_.getRetuneDelay(baselineFreq_, freq) : 0ns;
            auto upCost   = includeUpCost
                ? retuneDelayTracker_.getRetuneDelay(freq, baselineFreq_) : 0ns;
            if (downCost > effectiveBudget(region.front())) continue;
            auto debt     = downCost;
            bool feasible = true;
            for (int nodeIdx : region)
            {
                debt += executionTimeSlowdown(nodeIdx, freq);
                if (debt + upCost > effectiveBudget(nodeIdx))
                {
                    feasible = false;
                    break;
                }
            }
            if (feasible) return freq;
        }
        return baselineFreq_;
    }

    // Zone A: slack prefix before the first CP entry. No downtune cost; must fit uptune.
    void buildZoneA(const std::vector<int>& region, const std::vector<int>& availableFreqs,
                    std::vector<FreqEvent>& out)
    {
        if (region.empty()) return;
        int tuneFreq = findLowestFeasibleFreq(region, availableFreqs, false, true);
        if (tuneFreq == baselineFreq_) return;
        out.push_back({kSentinelNodeId, tuneFreq});
        out.push_back({region.back(), baselineFreq_});
    }

    // Zone B: slack gap between two CP segments. Downtune when lastCpNode completes;
    // uptune when region.back() completes, restoring baseline before the next CP node.
    // Nodes with cross-partition edges use slack/nrPartitions() as their effective budget.
    void buildZoneB(const std::vector<int>& region, const std::vector<int>& availableFreqs,
                    int lastCpNode, std::vector<FreqEvent>& out)
    {
        if (region.empty()) return;
        int tuneFreq = findLowestFeasibleFreq(region, availableFreqs, true, true);
        if (tuneFreq == baselineFreq_) return;
        out.push_back({lastCpNode, tuneFreq});
        out.push_back({region.back(), baselineFreq_});
    }

    // Zone C: trailing slack after the last CP exit. Downtune when lastCpNode completes;
    // no uptune required. Nodes with cross-partition edges use slack/nrPartitions().
    void buildZoneC(const std::vector<int>& region, const std::vector<int>& availableFreqs,
                    int lastCpNode, std::vector<FreqEvent>& out)
    {
        if (region.empty()) return;
        int tuneFreq = findLowestFeasibleFreq(region, availableFreqs, true, false);
        if (tuneFreq == baselineFreq_) return;
        out.push_back({lastCpNode, tuneFreq});
    }

    void buildSchedule()
    {
        auto pts = collectSchedulingPoints();
        if (pts.empty()) return;

        auto availableFreqs = getAvailableFreqs(pts[0]);
        if (availableFreqs.empty()) return;

        std::vector<FreqEvent> events;
        std::vector<int>       region;
        bool                   isInitialSegment = true;
        int                    lastCpNode       = -1;

        for (int nodeIdx : pts)
        {
            if (spanned_.slack(nodeIdx) > 0.0)
            {
                region.push_back(nodeIdx);
            }
            else
            {
                if (isInitialSegment)
                    buildZoneA(region, availableFreqs, events);
                else
                    buildZoneB(region, availableFreqs, lastCpNode, events);
                isInitialSegment = false;
                lastCpNode       = nodeIdx;
                region.clear();
            }
        }
        if (!isInitialSegment)
            buildZoneC(region, availableFreqs, lastCpNode, events);

        // Sort into execution order: sentinel first, then by span.start.
        std::sort(events.begin(), events.end(), [this](const FreqEvent& a, const FreqEvent& b) {
            if (a.nodeId == kSentinelNodeId) return true;
            if (b.nodeId == kSentinelNodeId) return false;
            return spanned_.spannedNodes()[a.nodeId].span.start
                 < spanned_.spannedNodes()[b.nodeId].span.start;
        });

        zoneAFreq_ = baselineFreq_;
        for (auto& ev : events)
        {
            if (ev.nodeId == kSentinelNodeId)
                zoneAFreq_ = ev.freq;
            else
                freqEvents_.push_back(ev);
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
        out_->print("zone_A_start_freq: %d MHz\n", zoneAFreq_);

        auto pts = collectSchedulingPoints();
        out_->print("scheduling_points (execution order):\n");
        for (int idx : pts)
        {
            double      slackMs = spanned_.slack(idx) / 1e6;
            const char* kind    = (slackMs > 0.0) ? "NPI" : "PI ";
            out_->print("  %s node %d (%s) slack=%.2fms", kind, idx,
                        dag_.nodes()[idx].content.op.toString().c_str(), slackMs);
            for (auto& ev : freqEvents_)
            {
                if (ev.nodeId == idx)
                {
                    out_->print(" [retune=%d MHz]", ev.freq);
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
    std::vector<FreqEvent>                       freqEvents_;
    size_t                                       eventIdx_{0};
    int                                          zoneAFreq_{};
    int                                          currentFreq_{};
    std::optional<PEWriter>                      out_;
};
