#pragma once

#include <set>
#include <vector>

struct Edge
{
    int from;
    int to;
};

template <typename Content>
struct Node
{
    int     index     = -1;
    int     partition = -1;
    Content content;
};

template <typename Content>
class PartitionedDag
{
   public:
    using NodeType = Node<Content>;

    void addNode(NodeType n)
    {
        n.index = (int)nodes_.size();
        nodes_.push_back(std::move(n));
    }
    void addEdge(Edge e) { edges_.push_back(e); }

    std::vector<NodeType>&       nodes() { return nodes_; }
    const std::vector<NodeType>& nodes() const { return nodes_; }

    std::vector<int> nodes(int partition) const
    {
        std::vector<int> result;
        for (auto& n : nodes_)
            if (n.partition == partition) result.push_back(n.index);
        return result;
    }

    std::vector<int> crossIncomingNodeIndices(const NodeType& n) const
    {
        std::vector<int> result;
        for (auto& e : edges_)
            if (e.to == n.index && nodes_[e.from].partition != n.partition)
                result.push_back(e.from);
        return result;
    }

    std::vector<int> crossOutgoingPartitions(const NodeType& n) const
    {
        std::set<int> result;
        for (auto& e : edges_)
            if (e.from == n.index && nodes_[e.to].partition != n.partition)
                result.insert(nodes_[e.to].partition);
        return {result.begin(), result.end()};
    }

   private:
    std::vector<NodeType> nodes_;
    std::vector<Edge>     edges_;
};
