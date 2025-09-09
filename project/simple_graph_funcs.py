import cfpq_data
from networkx.drawing import nx_pydot

def get_graph_info(graph_name: str) -> tuple[int, int, list[str]]:
    graph_path = cfpq_data.download(graph_name)
    graph = cfpq_data.graph_from_csv(graph_path)
    return graph.number_of_nodes(), graph.number_of_edges(), cfpq_data.get_sorted_labels(graph)

def create_and_save_two_cycled_graph(n: int, m: int, labels: tuple[str, str], file_path: str) -> None:
    graph = cfpq_data.labeled_two_cycles_graph(n, m, labels=labels)
    pydot_graph = nx_pydot.to_pydot(graph)
    pydot_graph.write_raw(file_path)
