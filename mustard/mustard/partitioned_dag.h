#pragma once

#include <optional>
#include <set>
#include <stdexcept>
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
    bool    tail      = true;
    Content content;

    bool isTail() const { return tail; }
};

template <typename Content>
class PartitionedDag
{
   public:
    using NodeType = Node<Content>;

    void addNode(NodeType n)
    {
        n.index = (int)nodes_.size();
        incoming_.push_back({});
        outgoing_.push_back({});
        nodes_.push_back(std::move(n));
    }
    void addEdge(Edge e)
    {
        nodes_[e.from].tail = false;
        incoming_[e.to].push_back(e.from);
        outgoing_[e.from].push_back(e.to);
        edges_.push_back(e);
    }

    const std::vector<int>& incomingEdges(int nodeIdx) const { return incoming_[nodeIdx]; }
    const std::vector<int>& outgoingEdges(int nodeIdx) const { return outgoing_[nodeIdx]; }

    std::vector<NodeType>&       nodes() { return nodes_; }
    const std::vector<NodeType>& nodes() const { return nodes_; }

    std::vector<Edge>&       edges() { return edges_; }
    const std::vector<Edge>& edges() const { return edges_; }

    std::vector<NodeType> nodes(int partition) const
    {
        std::vector<NodeType> result;
        for (auto& n : nodes_)
            if (n.partition == partition) result.push_back(n);
        return result;
    }

    bool hasIncomingXPartition(const NodeType& n) const
    {
        for (int src : incomingEdges(n.index))
            if (nodes_[src].partition != n.partition) return true;
        return false;
    }

    // Returns the first node in the partition with no within-partition incoming edges.
    NodeType getRoot(int partition) const
    {
        for (auto& n : nodes_)
        {
            if (n.partition != partition) continue;
            bool hasIncomingSamePartition = false;
            for (int src : incomingEdges(n.index))
                if (nodes_[src].partition == partition) { hasIncomingSamePartition = true; break; }
            if (!hasIncomingSamePartition) return n;
        }
        throw std::runtime_error("no root node for partition");
    }

    // Starting from n, follows same-partition successors and collects nodes
    // until (not including) the first one with a cross-partition incoming edge.
    std::vector<NodeType> untilIncomingXEdge(const NodeType& start) const
    {
        std::vector<NodeType> result;
        int current = start.index;
        while (current != -1)
        {
            auto& n = nodes_[current];
            if (current != start.index && hasIncomingXPartition(n)) return result;
            result.push_back(n);
            current = -1;
            for (int dst : outgoingEdges(n.index))
                if (nodes_[dst].partition == n.partition) { current = dst; break; }
        }
        return result;
    }

    // Advances nodeIndex to the first same-partition outgoing successor.
    // Returns true and updates nodeIndex if found, false if none exists.
    bool next(int& nodeIndex, int partition) const
    {
        for (int dst : outgoingEdges(nodeIndex))
            if (nodes_[dst].partition == partition) { nodeIndex = dst; return true; }
        return false;
    }

    std::optional<int> nextNode(int nodeIndex, int partition) const
    {
        for (int dst : outgoingEdges(nodeIndex))
            if (nodes_[dst].partition == partition) return dst;
        return std::nullopt;
    }

    std::vector<int> crossIncomingNodeIndices(const NodeType& n) const
    {
        std::vector<int> result;
        for (int src : incomingEdges(n.index))
            if (nodes_[src].partition != n.partition) result.push_back(src);
        return result;
    }

    std::vector<int> crossOutgoingPartitions(const NodeType& n) const
    {
        std::vector<int> result;
        for (int dst : outgoingEdges(n.index))
            if (nodes_[dst].partition != n.partition) result.push_back(nodes_[dst].partition);
        return result;
    }

    // Find all nodes on other partitions that have outgoing edge to this node
    std::vector<NodeType> crossIncomingNodes(const NodeType& n) const
    {
        std::vector<NodeType> result;
        for (int src : incomingEdges(n.index))
            if (nodes_[src].partition != n.partition) result.push_back(nodes_[src]);
        return result;
    }

    // Find all nodes on other partitions that are connected to outgoing edge from this one
    std::vector<NodeType> crossOutgoingNodes(const NodeType& n) const
    {
        std::vector<NodeType> result;
        for (int dst : outgoingEdges(n.index))
            if (nodes_[dst].partition != n.partition) result.push_back(nodes_[dst]);
        return result;
    }

   private:
    std::vector<NodeType>        nodes_;
    std::vector<Edge>            edges_;
    std::vector<std::vector<int>> incoming_;
    std::vector<std::vector<int>> outgoing_;
};
