import numpy as np
from networkx import MultiDiGraph
from pyformlang.finite_automaton import State
from scipy.sparse import csr_matrix

from project.all_pairs_rpq import AdjacencyMatrixFA, intersect_automata
from project.automaton_conversion import graph_to_nfa, regex_to_dfa


def _initial_frontier(
    starts: list[int],
    graph_automaton: AdjacencyMatrixFA,
    regex_automaton: AdjacencyMatrixFA,
    regex_state_count: int,
) -> csr_matrix:
    """Build a boolean matrix with one row per source and one column per product state.

    A row ``i`` has ``True`` in the columns corresponding to the product states
    ``(starts[i], regex_start)`` for every start state of the regular expression.
    """
    rows = []
    columns = []
    for row, node in enumerate(starts):
        graph_index = graph_automaton.state_to_index[State(node)]
        for regex_index in regex_automaton.start_states:
            rows.append(row)
            columns.append(graph_index * regex_state_count + regex_index)

    data = np.ones(len(rows), dtype=bool)
    product_state_count = len(graph_automaton.states) * regex_state_count
    return csr_matrix(
        (data, (rows, columns)),
        shape=(len(starts), product_state_count),
        dtype=bool,
    )


def _collect_reachable_pairs(
    starts: list[int],
    visited: csr_matrix,
    graph_automaton: AdjacencyMatrixFA,
    regex_automaton: AdjacencyMatrixFA,
    final_nodes: set[int],
    regex_state_count: int,
) -> set[tuple[int, int]]:
    """Turn the visited product states into pairs of source and final graph nodes."""
    final_by_index = {}
    for node in final_nodes:
        graph_index = graph_automaton.state_to_index[State(node)]
        for regex_index in regex_automaton.final_states:
            final_by_index[graph_index * regex_state_count + regex_index] = node

    visited = visited.tocsr()
    result = set()
    for row, node in enumerate(starts):
        for column in visited.indices[visited.indptr[row] : visited.indptr[row + 1]]:
            target = final_by_index.get(int(column))
            if target is not None:
                result.add((node, target))
    return result


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    """Regular path query evaluated with a multiple source BFS.

    The graph and the regular expression are converted to automata whose product
    is traversed with boolean sparse matrix operations. All sources are processed
    simultaneously: row ``i`` of the frontier matrix tracks the states reachable
    from ``start_nodes[i]``. A pair ``(source, target)`` is reported when a final
    product state ``(target, regex_final)`` is reachable from the source.
    """
    graph_nodes = set(graph.nodes)
    start_nodes = set(start_nodes) & graph_nodes
    final_nodes = set(final_nodes) & graph_nodes
    if not start_nodes or not final_nodes:
        return set()

    graph_automaton = AdjacencyMatrixFA(graph_to_nfa(graph, graph_nodes, graph_nodes))
    regex_automaton = AdjacencyMatrixFA(regex_to_dfa(regex))
    product = intersect_automata(graph_automaton, regex_automaton)

    regex_state_count = len(regex_automaton.states)
    starts = sorted(start_nodes)
    adjacency = product._adjacency_matrix.astype(bool)

    frontier = _initial_frontier(
        starts, graph_automaton, regex_automaton, regex_state_count
    )
    visited = frontier
    while frontier.nnz:
        reached = (frontier @ adjacency).astype(bool)
        frontier = reached > visited
        visited = visited.maximum(reached)

    return _collect_reachable_pairs(
        starts,
        visited,
        graph_automaton,
        regex_automaton,
        final_nodes,
        regex_state_count,
    )
