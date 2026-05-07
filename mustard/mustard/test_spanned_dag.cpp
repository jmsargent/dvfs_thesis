#include <cstdio>

#include "spanned_partitioned_dag.h"

static int pass = 0, fail = 0;

#define CHECK(cond, msg)                                      \
    do {                                                      \
        if (cond) { printf("  PASS: %s\n", msg); pass++; }   \
        else      { printf("  FAIL: %s\n", msg); fail++; }   \
    } while (0)

int main()
{
    // ------------------------------------------------------------------
    // Test 1: nodes store span data correctly
    // ------------------------------------------------------------------
    printf("=== Test 1: span storage ===\n");
    {
        SpannedPartitionedDag<int, double> dag;

        SpannedNode<int, double> n;
        n.partition  = 0;
        n.content    = 42;
        n.span       = {1.0, 3.5};
        dag.addNode(n);

        auto& nodes = dag.spannedNodes();
        CHECK(nodes.size() == 1,           "one node stored");
        CHECK(nodes[0].content == 42,      "content preserved");
        CHECK(nodes[0].span.start == 1.0,  "span.start correct");
        CHECK(nodes[0].span.end   == 3.5,  "span.end correct");
        CHECK(nodes[0].index == 0,         "index assigned");
    }

    // ------------------------------------------------------------------
    // Test 2: partition membership
    // ------------------------------------------------------------------
    printf("=== Test 2: partitions ===\n");
    {
        SpannedPartitionedDag<int, double> dag;

        for (int i = 0; i < 3; i++)
        {
            SpannedNode<int, double> n;
            n.partition = i % 2;
            n.content   = i;
            n.span      = {(double)i, (double)i + 1.0};
            dag.addNode(n);
        }

        auto p0 = dag.nodes(0);
        auto p1 = dag.nodes(1);
        CHECK(p0.size() == 2, "partition 0 has 2 nodes");
        CHECK(p1.size() == 1, "partition 1 has 1 node");
    }

    // ------------------------------------------------------------------
    // Test 3: cross-partition edge traversal
    // ------------------------------------------------------------------
    printf("=== Test 3: cross-partition edges ===\n");
    {
        SpannedPartitionedDag<int, double> dag;

        SpannedNode<int, double> a, b, c;
        a.partition = 0; a.content = 0; a.span = {0.0, 1.0};
        b.partition = 1; b.content = 1; b.span = {1.0, 2.0};
        c.partition = 1; c.content = 2; c.span = {1.0, 2.0};
        dag.addNode(a);  // index 0
        dag.addNode(b);  // index 1
        dag.addNode(c);  // index 2

        dag.addEdge({0, 1});  // a -> b  (cross-partition)
        dag.addEdge({1, 2});  // b -> c  (same partition)

        auto& sn     = dag.spannedNodes();
        auto  inc_b  = dag.crossIncomingNodeIndices(sn[1]);
        auto  out_a  = dag.crossOutgoingPartitions(sn[0]);

        CHECK(inc_b.size() == 1 && inc_b[0] == 0, "b has one cross-partition incoming (a)");
        CHECK(out_a.size() == 1 && out_a[0] == 1, "a has one cross-partition outgoing (partition 1)");

        auto inc_c = dag.crossIncomingNodeIndices(sn[2]);
        CHECK(inc_c.empty(), "c has no cross-partition incoming");
    }

    printf("=== Test 4: extendNode cascades to dependents ===\n");
    {
        SpannedPartitionedDag<int, double> dag;

        SpannedNode<int, double> a, b, c;
        a.partition = 0; a.content = 0; a.span = {0.0, 1.0};
        b.partition = 1; b.content = 1; b.span = {1.0, 2.0};
        c.partition = 1; c.content = 2; c.span = {2.0, 3.0};
        dag.addNode(a);  // index 0
        dag.addNode(b);  // index 1
        dag.addNode(c);  // index 2

        dag.addEdge({0, 1});  // a -> b
        dag.addEdge({1, 2});  // b -> c

        dag.extendNode(0, 1.0);  // a now ends at 2.0

        auto& sn = dag.spannedNodes();
        CHECK(sn[0].span.end   == 2.0, "a end extended to 2.0");
        CHECK(sn[1].span.start == 2.0, "b start pushed to 2.0");
        CHECK(sn[1].span.end   == 3.0, "b end pushed to 3.0 (duration preserved)");
        CHECK(sn[2].span.start == 3.0, "c start pushed to 3.0");
        CHECK(sn[2].span.end   == 4.0, "c end pushed to 4.0 (duration preserved)");
    }

    // ------------------------------------------------------------------
    // Test 5: child with two parents — only delayed if extended node ends latest
    // ------------------------------------------------------------------
    printf("=== Test 5: child not delayed when another parent ends later ===\n");
    {
        //  a(0-1) --> b(1-2) --\
        //                       --> d(3-4)
        //  c(0-3) -------------/
        SpannedPartitionedDag<int, double> dag;

        SpannedNode<int, double> a, b, c, d;
        a.partition = 0; a.content = 0; a.span = {0.0, 1.0};
        b.partition = 1; b.content = 1; b.span = {1.0, 2.0};
        c.partition = 0; c.content = 2; c.span = {0.0, 3.0};
        d.partition = 1; d.content = 3; d.span = {3.0, 4.0};
        dag.addNode(a);  // index 0
        dag.addNode(b);  // index 1
        dag.addNode(c);  // index 2
        dag.addNode(d);  // index 3

        dag.addEdge({0, 1});  // a -> b
        dag.addEdge({1, 3});  // b -> d
        dag.addEdge({2, 3});  // c -> d

        // extend a by 1: a ends at 2, b shifts to {2,3}, but d's other parent c ends at 3
        dag.extendNode(0, 1.0);

        auto& sn = dag.spannedNodes();
        CHECK(sn[1].span.start == 2.0, "b pushed to start at 2.0");
        CHECK(sn[1].span.end   == 3.0, "b pushed to end at 3.0");
        CHECK(sn[3].span.start == 3.0, "d not delayed (c still ends at 3.0)");
        CHECK(sn[3].span.end   == 4.0, "d end unchanged");

        // now extend a further so b ends at 4, past c — d must follow
        dag.extendNode(0, 1.0);  // a ends at 3, b shifts to {3,4}

        CHECK(sn[3].span.start == 4.0, "d delayed when b overtakes c");
        CHECK(sn[3].span.end   == 5.0, "d end updated, duration preserved");
    }

    // ------------------------------------------------------------------
    // Test 6: shortenNode pulls dependents earlier, blocked by other parents
    // ------------------------------------------------------------------
    printf("=== Test 6: shortenNode cascades earlier ===\n");
    {
        //  a(0-2) --> b(2-4) --> c(4-6)
        SpannedPartitionedDag<int, double> dag;

        SpannedNode<int, double> a, b, c;
        a.partition = 0; a.content = 0; a.span = {0.0, 2.0};
        b.partition = 1; b.content = 1; b.span = {2.0, 4.0};
        c.partition = 0; c.content = 2; c.span = {4.0, 6.0};
        dag.addNode(a);
        dag.addNode(b);
        dag.addNode(c);

        dag.addEdge({0, 1});  // a -> b
        dag.addEdge({1, 2});  // b -> c

        dag.shortenNode(0, 1.0);  // a now ends at 1.0

        auto& sn = dag.spannedNodes();
        CHECK(sn[0].span.end   == 1.0, "a end shortened to 1.0");
        CHECK(sn[1].span.start == 1.0, "b pulled earlier to 1.0");
        CHECK(sn[1].span.end   == 3.0, "b end at 3.0 (duration preserved)");
        CHECK(sn[2].span.start == 3.0, "c pulled earlier to 3.0");
        CHECK(sn[2].span.end   == 5.0, "c end at 5.0 (duration preserved)");
    }

    printf("=== Test 7: shortenNode blocked by another parent ===\n");
    {
        //  a(0-2) --> b(2-3) --\
        //                       --> d(3-4)
        //  c(0-3) -------------/
        SpannedPartitionedDag<int, double> dag;

        SpannedNode<int, double> a, b, c, d;
        a.partition = 0; a.content = 0; a.span = {0.0, 2.0};
        b.partition = 1; b.content = 1; b.span = {2.0, 3.0};
        c.partition = 0; c.content = 2; c.span = {0.0, 3.0};
        d.partition = 1; d.content = 3; d.span = {3.0, 4.0};
        dag.addNode(a);
        dag.addNode(b);
        dag.addNode(c);
        dag.addNode(d);

        dag.addEdge({0, 1});  // a -> b
        dag.addEdge({1, 3});  // b -> d
        dag.addEdge({2, 3});  // c -> d

        dag.shortenNode(0, 1.0);  // a ends at 1.0, b shifts to {1,2}

        auto& sn = dag.spannedNodes();
        CHECK(sn[1].span.start == 1.0, "b pulled to 1.0");
        CHECK(sn[1].span.end   == 2.0, "b ends at 2.0");
        CHECK(sn[3].span.start == 3.0, "d not pulled (c still ends at 3.0)");
        CHECK(sn[3].span.end   == 4.0, "d end unchanged");
    }

    // ------------------------------------------------------------------
    printf("\n%d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
