from collections import defaultdict
from collections.abc import Iterable

from networkx import MultiDiGraph
from pyformlang.finite_automaton import Epsilon, Symbol
from scipy.sparse import coo_matrix, csr_matrix, eye, kron
from scipy.sparse.csgraph import breadth_first_order

from project.automaton_conversion import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    def __init__(self, automaton):
        states = list(automaton.states)
        state_to_index = {state: index for index, state in enumerate(states)}
        transitions = defaultdict(list)

        for source, symbol_targets in automaton.to_dict().items():
            for symbol, targets in symbol_targets.items():
                if not isinstance(targets, (set, frozenset, list, tuple)):
                    targets = (targets,)
                for target in targets:
                    if source not in state_to_index:
                        state_to_index[source] = len(states)
                        states.append(source)
                    if target not in state_to_index:
                        state_to_index[target] = len(states)
                        states.append(target)
                    transitions[symbol].append(
                        (state_to_index[source], state_to_index[target])
                    )

        start_states = {state_to_index[state] for state in automaton.start_states}
        final_states = {state_to_index[state] for state in automaton.final_states}
        matrices = {}
        state_count = len(states)

        for symbol, edges in transitions.items():
            rows, columns = zip(*edges)
            matrices[symbol] = coo_matrix(
                ([True] * len(edges), (rows, columns)),
                shape=(state_count, state_count),
                dtype=bool,
            ).tocsr()

        self._initialize(states, matrices, start_states, final_states)

    def _initialize(self, states, matrices, start_states, final_states):
        self.states = tuple(states)
        self.state_to_index = {state: index for index, state in enumerate(states)}
        self.adjacency_matrices = {
            symbol: matrix.tocsr().astype(bool) for symbol, matrix in matrices.items()
        }
        self.start_states = set(start_states)
        self.final_states = set(final_states)
        self._epsilon = Epsilon()
        self._adjacency_matrix = self._union_matrices(
            self.adjacency_matrices.values(), len(self.states)
        )

    @staticmethod
    def _union_matrices(matrices, state_count):
        result = csr_matrix((state_count, state_count), dtype=bool)
        for matrix in matrices:
            result = result.maximum(matrix)
        return result.tocsr().astype(bool)

    @classmethod
    def _from_components(cls, states, matrices, start_states, final_states):
        result = cls.__new__(cls)
        result._initialize(states, matrices, start_states, final_states)
        return result

    def _epsilon_closure(self, state_indices):
        epsilon_matrix = self.adjacency_matrices.get(self._epsilon)
        closure = set(state_indices)
        if epsilon_matrix is None:
            return closure

        frontier = list(closure)
        while frontier:
            source = frontier.pop()
            for target in epsilon_matrix.getrow(source).indices:
                target = int(target)
                if target not in closure:
                    closure.add(target)
                    frontier.append(target)
        return closure

    def accepts(self, word: Iterable[Symbol]) -> bool:
        current_states = self._epsilon_closure(self.start_states)

        for item in word:
            symbol = item if isinstance(item, (Symbol, Epsilon)) else Symbol(item)
            if isinstance(symbol, Epsilon):
                current_states = self._epsilon_closure(current_states)
                continue

            transition_matrix = self.adjacency_matrices.get(symbol)
            if transition_matrix is None or not current_states:
                return False

            next_states = set()
            for source in current_states:
                next_states.update(
                    int(target) for target in transition_matrix.getrow(source).indices
                )
            current_states = self._epsilon_closure(next_states)

        return bool(current_states & self.final_states)

    def is_empty(self) -> bool:
        if self.start_states & self.final_states:
            return False
        if not self.states or not self.start_states or not self.final_states:
            return True

        for start_state in self.start_states:
            reachable = breadth_first_order(
                self._adjacency_matrix,
                start_state,
                directed=True,
                return_predecessors=False,
            )
            if self.final_states.intersection(map(int, reachable)):
                return False
        return True


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    first_count = len(automaton1.states)
    second_count = len(automaton2.states)
    product_states = [
        (first_state, second_state)
        for first_state in automaton1.states
        for second_state in automaton2.states
    ]
    product_matrices = {}

    common_symbols = automaton1.adjacency_matrices.keys() & (
        automaton2.adjacency_matrices.keys()
    )
    for symbol in common_symbols:
        if isinstance(symbol, Epsilon):
            continue
        product_matrices[symbol] = kron(
            automaton1.adjacency_matrices[symbol],
            automaton2.adjacency_matrices[symbol],
            format="csr",
        ).astype(bool)

    epsilon = Epsilon()
    first_epsilon = automaton1.adjacency_matrices.get(
        epsilon, csr_matrix((first_count, first_count), dtype=bool)
    )
    second_epsilon = automaton2.adjacency_matrices.get(
        epsilon, csr_matrix((second_count, second_count), dtype=bool)
    )
    product_epsilon = kron(
        first_epsilon,
        eye(second_count, dtype=bool, format="csr"),
        format="csr",
    ).maximum(
        kron(
            eye(first_count, dtype=bool, format="csr"),
            second_epsilon,
            format="csr",
        )
    )
    if product_epsilon.nnz:
        product_matrices[epsilon] = product_epsilon.astype(bool)

    start_states = {
        first_index * second_count + second_index
        for first_index in automaton1.start_states
        for second_index in automaton2.start_states
    }
    final_states = {
        first_index * second_count + second_index
        for first_index in automaton1.final_states
        for second_index in automaton2.final_states
    }

    return AdjacencyMatrixFA._from_components(
        product_states, product_matrices, start_states, final_states
    )


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    graph_nodes = set(graph.nodes)
    start_nodes = set(start_nodes) & graph_nodes
    final_nodes = set(final_nodes) & graph_nodes
    if not start_nodes or not final_nodes:
        return set()

    graph_automaton = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_automaton = AdjacencyMatrixFA(regex_to_dfa(regex))
    product = intersect_automata(graph_automaton, regex_automaton)
    accepting_states = product.final_states
    result = set()

    for product_start in product.start_states:
        reachable = breadth_first_order(
            product._adjacency_matrix,
            product_start,
            directed=True,
            return_predecessors=False,
        )
        reached_accepting = accepting_states.intersection(map(int, reachable))
        origin = product.states[product_start][0].value
        for product_final in reached_accepting:
            destination = product.states[product_final][0].value
            result.add((origin, destination))

    return result
