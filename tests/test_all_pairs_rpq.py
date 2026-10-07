from networkx import MultiDiGraph
from pyformlang.finite_automaton import Epsilon, EpsilonNFA, State, Symbol

from project.automaton_conversion import graph_to_nfa, regex_to_dfa
from project.all_pairs_rpq import (
    AdjacencyMatrixFA,
    intersect_automata,
    tensor_based_rpq,
)


def test_adjacency_matrix_fa_accepts_words_and_empty_word():
    automaton = AdjacencyMatrixFA(regex_to_dfa("a b*"))

    assert automaton.accepts([Symbol("a")])
    assert automaton.accepts(["a", "b", "b"])
    assert not automaton.accepts([])
    assert not automaton.accepts(["b"])


def test_adjacency_matrix_fa_detects_empty_and_nonempty_languages():
    accepting = AdjacencyMatrixFA(regex_to_dfa("a*"))
    rejecting = EpsilonNFA()
    rejecting.add_start_state(State("start"))
    rejecting.add_final_state(State("final"))

    assert not accepting.is_empty()
    assert AdjacencyMatrixFA(rejecting).is_empty()


def test_adjacency_matrix_fa_supports_epsilon_transitions():
    automaton = EpsilonNFA()
    automaton.add_start_state(State(0))
    automaton.add_transition(State(0), Epsilon(), State(1))
    automaton.add_transition(State(1), Symbol("x"), State(2))
    automaton.add_final_state(State(2))
    matrix_automaton = AdjacencyMatrixFA(automaton)

    assert matrix_automaton.accepts(["x"])
    assert not matrix_automaton.accepts([])
    assert not matrix_automaton.is_empty()


def test_intersection_accepts_only_common_language():
    first = AdjacencyMatrixFA(regex_to_dfa("a b*"))
    second = AdjacencyMatrixFA(regex_to_dfa("a* b"))
    intersection = intersect_automata(first, second)

    assert intersection.accepts(["a", "b"])
    assert not intersection.accepts(["a"])
    assert not intersection.accepts(["b"])
    assert not intersection.is_empty()


def test_intersection_handles_epsilon_moves_on_either_automaton():
    epsilon_automaton = EpsilonNFA()
    epsilon_automaton.add_start_state(State(0))
    epsilon_automaton.add_transition(State(0), Epsilon(), State(1))
    epsilon_automaton.add_transition(State(1), Symbol("a"), State(2))
    epsilon_automaton.add_final_state(State(2))

    first = AdjacencyMatrixFA(epsilon_automaton)
    second = AdjacencyMatrixFA(regex_to_dfa("a"))
    intersection = intersect_automata(first, second)

    assert intersection.accepts(["a"])


def test_tensor_based_rpq_returns_only_requested_endpoints():
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(0, 2, label="a")
    graph.add_edge(2, 2, label="b")

    assert tensor_based_rpq("a b*", graph, {0}, {1, 2}) == {(0, 1), (0, 2)}
    assert tensor_based_rpq("a b*", graph, set(), {1, 2}) == set()
    assert tensor_based_rpq("a b*", graph, {0}, set()) == set()


def test_tensor_based_rpq_includes_zero_length_paths_for_star():
    graph = MultiDiGraph()
    graph.add_nodes_from([4, 7])

    assert tensor_based_rpq("a*", graph, {4, 7}, {4}) == {(4, 4)}


def test_tensor_based_rpq_respects_graph_labels_and_parallel_edges():
    graph = MultiDiGraph()
    graph.add_edge(3, 5, label="x")
    graph.add_edge(3, 5, label="y")
    graph.add_edge(5, 8, label="x")

    assert tensor_based_rpq("x", graph, {3}, {5, 8}) == {(3, 5)}
    assert tensor_based_rpq("y x", graph, {3}, {8}) == {(3, 8)}


def test_constructor_preserves_start_and_final_states_from_graph_nfa():
    graph = MultiDiGraph()
    graph.add_edge(10, 11, label="edge")
    nfa = graph_to_nfa(graph, {10}, {11})
    matrix_automaton = AdjacencyMatrixFA(nfa)

    assert matrix_automaton.accepts(["edge"])
    assert not matrix_automaton.is_empty()
