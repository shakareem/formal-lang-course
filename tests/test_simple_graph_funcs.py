import os
import tempfile
import cfpq_data
from project.simple_graph_funcs import get_graph_info, create_and_save_two_cycled_graph

def test_get_graph_info():
    nodes, edges, labels = get_graph_info("wc")
    assert nodes == 332
    assert edges == 269
    assert labels == {"a", "d"}

def test_create_and_save_two_cycled_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "graph.dot")
        create_and_save_two_cycled_graph(3, 4, ("x", "y"), file_path)
        assert os.path.exists(file_path)
        with open(file_path) as f:
            dot_content = f.read()
        assert "x" in dot_content
        assert "y" in dot_content
