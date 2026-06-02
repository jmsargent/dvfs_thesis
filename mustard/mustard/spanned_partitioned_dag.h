#pragma once

#include <algorithm>

#include "partitioned_dag.h"

template <typename T>
struct Span
{
    T start;
    T end;
};

template <typename Content, typename SpanT>
struct SpannedNode : Node<Content>
{
    Span<SpanT> span;
};

template <typename Content, typename SpanT>
class SpannedPartitionedDag : public PartitionedDag<Content>
{
   public:
    using SpannedNodeType = SpannedNode<Content, SpanT>;

    void addNode(SpannedNodeType n)
    {
        int idx       = (int)spannedNodes_.size();
        n.index       = idx;
        spannedNodes_.push_back(n);
        maxEnd_ = std::max(maxEnd_, n.span.end);

        Node<Content> base;
        base.index     = idx;
        base.partition = n.partition;
        base.content   = n.content;
        PartitionedDag<Content>::addNode(base);
    }

    void addEdge(Edge e)
    {
        spannedNodes_[e.from].tail = false;
        PartitionedDag<Content>::addEdge(e);
    }

    void extendNode(int nodeIdx, SpanT amount)
    {
        spannedNodes_[nodeIdx].span.end += amount;
        if (spannedNodes_[nodeIdx].isTail())
            maxEnd_ = std::max(maxEnd_, spannedNodes_[nodeIdx].span.end);
        cascadeFrom_(nodeIdx);
    }

    void shortenNode(int nodeIdx, SpanT amount)
    {
        spannedNodes_[nodeIdx].span.end -= amount;
        cascadeFrom_(nodeIdx);
        recomputeMax_();
    }

    SpanT makespan() const { return maxEnd_; }

    std::vector<Span<SpanT>> idleSpans(int partition) const
    {
        std::vector<Span<SpanT>> gaps;
        for (auto& n : spannedNodes_)
        {
            if (n.partition != partition) continue;

            SpanT latestLocalEnd  = SpanT{};
            bool  hasCrossIncoming = false;

            for (auto& e : this->edges())
            {
                if (e.to != n.index) continue;
                auto& parent = spannedNodes_[e.from];
                if (parent.partition == partition)
                    latestLocalEnd = std::max(latestLocalEnd, parent.span.end);
                else
                    hasCrossIncoming = true;
            }

            if (hasCrossIncoming && n.span.start > latestLocalEnd)
                gaps.push_back({latestLocalEnd, n.span.start});
        }
        return gaps;
    }

    void computeSlack()
    {
        int n = (int)spannedNodes_.size();
        lst_.resize(n);
        SpanT ms = makespan();
        for (int i = n - 1; i >= 0; --i)
        {
            SpanT duration = spannedNodes_[i].span.end - spannedNodes_[i].span.start;
            auto& out      = this->outgoingEdges(i);
            if (out.empty())
            {
                lst_[i] = ms - duration;
            }
            else
            {
                SpanT earliest_child_lst = lst_[out[0]];
                for (int j = 1; j < (int)out.size(); ++j)
                    earliest_child_lst = std::min(earliest_child_lst, lst_[out[j]]);
                lst_[i] = earliest_child_lst - duration;
            }
        }
    }

    SpanT slack(int nodeIdx) const { return lst_[nodeIdx] - spannedNodes_[nodeIdx].span.start; }

    // Returns the same-partition predecessor with the latest span.end (i.e. the one
    // that executes immediately before n in execution order on that partition).
    std::optional<SpannedNodeType> predecessor(const SpannedNodeType& n, int partition) const
    {
        std::optional<SpannedNodeType> result;
        for (int src : this->incomingEdges(n.index))
        {
            if (spannedNodes_[src].partition != partition) continue;
            if (!result || spannedNodes_[src].span.end > result->span.end)
                result = spannedNodes_[src];
        }
        return result;
    }

    // Returns the same-partition successor with the earliest span.start.
    std::optional<SpannedNodeType> successor(const SpannedNodeType& n) const
    {
        std::optional<SpannedNodeType> result;
        for (int dst : this->outgoingEdges(n.index))
        {
            if (spannedNodes_[dst].partition != n.partition) continue;
            if (!result || spannedNodes_[dst].span.start < result->span.start)
                result = spannedNodes_[dst];
        }
        return result;
    }

    std::vector<SpannedNodeType>&       spannedNodes() { return spannedNodes_; }
    const std::vector<SpannedNodeType>& spannedNodes() const { return spannedNodes_; }

   private:
    void cascadeFrom_(int nodeIdx)
    {
        for (auto& e : this->edges())
        {
            if (e.from != nodeIdx) continue;

            SpanT maxParentEnd = latestParentEnd_(e.to);
            auto& child        = spannedNodes_[e.to];
            if (maxParentEnd == child.span.start) continue;

            SpanT delta      = maxParentEnd - child.span.start;
            child.span.start = maxParentEnd;
            child.span.end  += delta;
            if (child.isTail())
                maxEnd_ = std::max(maxEnd_, child.span.end);
            cascadeFrom_(e.to);
        }
    }

    SpanT latestParentEnd_(int nodeIdx) const
    {
        SpanT latest = SpanT{};
        for (auto& e : this->edges())
            if (e.to == nodeIdx)
                latest = std::max(latest, spannedNodes_[e.from].span.end);
        return latest;
    }

    void recomputeMax_()
    {
        maxEnd_ = SpanT{};
        for (auto& n : spannedNodes_)
            if (n.isTail()) maxEnd_ = std::max(maxEnd_, n.span.end);
    }

    std::vector<SpannedNodeType> spannedNodes_;
    std::vector<SpanT>           lst_;
    SpanT                        maxEnd_ = SpanT{};
};
