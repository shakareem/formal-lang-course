import random

from networkx import MultiDiGraph

from project.all_pairs_rpq import tensor_based_rpq
from project.multiple_source_rpq import ms_bfs_based_rpq


def _graph_from_edges(edges):
    graph = MultiDiGraph()
    for source, label, target in edges:
        graph.add_edge(source, target, label=label)
    return graph


def test_ms_bfs_based_rpq_finds_paths_from_single_source():
    graph = _graph_from_edges(
        [
            (0, "a", 1),
            (1, "b", 2),
            (0, "a", 3),
            (3, "b", 4),
        ]
    )

    assert ms_bfs_based_rpq("a b", graph, {0}, {2, 3, 4}) == {(0, 2), (0, 4)}


def test_ms_bfs_based_rpq_handles_multiple_sources():
    graph = _graph_from_edges(
        [
            (0, "a", 1),
            (1, "b", 2),
            (5, "a", 1),
        ]
    )

    assert ms_bfs_based_rpq("a b", graph, {0, 5}, {2}) == {(0, 2), (5, 2)}


def test_ms_bfs_based_rpq_returns_empty_for_empty_sets():
    graph = _graph_from_edges([(0, "a", 1)])

    assert ms_bfs_based_rpq("a", graph, set(), {1}) == set()
    assert ms_bfs_based_rpq("a", graph, {0}, set()) == set()


def test_ms_bfs_based_rpq_includes_zero_length_paths_for_star():
    graph = MultiDiGraph()
    graph.add_nodes_from([4, 7])

    assert ms_bfs_based_rpq("a*", graph, {4, 7}, {4}) == {(4, 4)}


def test_ms_bfs_based_rpq_respects_labels_and_parallel_edges():
    graph = _graph_from_edges(
        [
            (3, "x", 5),
            (3, "y", 5),
            (5, "x", 8),
        ]
    )

    assert ms_bfs_based_rpq("x", graph, {3}, {5, 8}) == {(3, 5)}
    assert ms_bfs_based_rpq("y x", graph, {3}, {8}) == {(3, 8)}


def test_ms_bfs_based_rpq_ignores_nodes_absent_from_graph():
    graph = _graph_from_edges([(0, "a", 1)])

    assert ms_bfs_based_rpq("a", graph, {0, 99}, {1, 42}) == {(0, 1)}


def _random_graph(
    rng: random.Random, node_count: int, labels: list[str]
) -> MultiDiGraph:
    graph = MultiDiGraph()
    graph.add_nodes_from(range(node_count))
    for _ in range(node_count * 2):
        source = rng.randrange(node_count)
        target = rng.randrange(node_count)
        graph.add_edge(source, target, label=rng.choice(labels))
    return graph


def test_ms_bfs_based_rpq_matches_tensor_based_rpq_on_random_graphs():
    rng = random.Random(20240517)
    regexes = [
        "a",
        "a*",
        "a b",
        "(a|b)*",
        "a* b* c*",
        "(a b)*",
        "a+",
        "b c a*",
        "(a|b|c)+",
        "a b c*",
    ]

    for _ in range(30):
        node_count = rng.randint(1, 12)
        graph = _random_graph(rng, node_count, ["a", "b", "c"])
        nodes = list(graph.nodes)
        start_nodes = set(rng.sample(nodes, rng.randint(1, node_count)))
        final_nodes = set(rng.sample(nodes, rng.randint(1, node_count)))

        for regex in regexes:
            expected = tensor_based_rpq(regex, graph, start_nodes, final_nodes)
            actual = ms_bfs_based_rpq(regex, graph, start_nodes, final_nodes)

            assert actual == expected, (regex, start_nodes, final_nodes)
