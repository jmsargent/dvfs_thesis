#pragma once

#include <chrono>
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
                if(out_) out_->print("setting frequency to: %d", freq);
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