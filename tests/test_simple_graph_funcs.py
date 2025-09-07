import os
import tempfile
import cfpq_data
from project.simple_graph_funcs import get_graph_info, create_and_save_two_cycled_graph

def test_get_graph_info():
    nodes, edges, labels = get_graph_info("wc")
    assert isinstance(nodes, int)
    assert isinstance(edges, int)
    assert isinstance(labels, list)
    assert "a" in labels or "b" in labels  # примерная проверка

def test_create_and_save_two_cycled_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "graph.dot")
        create_and_save_two_cycled_graph(3, 4, ("x", "y"), file_path)
        assert os.path.exists(file_path)
        with open(file_path) as f:
            dot_content = f.read()
        assert "x" in dot_content
        assert "y" in dot_content
