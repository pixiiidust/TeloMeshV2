import os
import sys
import pytest
import networkx as nx
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions to test (will be implemented in analysis/event_chokepoints.py)
from analysis.event_chokepoints import (
    compute_fractal_participation,
    compute_hubness_score,
    build_decision_table
)

# Test fixtures for standard graph types
@pytest.fixture
def linear_graph():
    """Create a small linear graph for testing."""
    return nx.path_graph(5)

@pytest.fixture
def loop_graph():
    """Create a simple loop graph for testing."""
    G = nx.DiGraph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 0)
    return G

@pytest.fixture
def star_graph():
    """Create a small star graph for testing."""
    return nx.star_graph(5)

@pytest.fixture
def complex_graph():
    """Create a more complex graph with known structures."""
    G = nx.DiGraph()
    
    # Create a triangle (loop)
    G.add_edge('A', 'B')
    G.add_edge('B', 'C')
    G.add_edge('C', 'A')
    
    # Create a hub (star)
    G.add_edge('D', 'E')
    G.add_edge('D', 'F')
    G.add_edge('D', 'G')
    G.add_edge('D', 'H')
    
    # Create a chain
    G.add_edge('I', 'J')
    G.add_edge('J', 'K')
    G.add_edge('K', 'L')
    
    # Connect structures
    G.add_edge('C', 'D')
    G.add_edge('H', 'I')
    
    return G

def test_compute_fractal_participation_exists():
    """Test that compute_fractal_participation function exists and is callable."""
    assert callable(compute_fractal_participation), "compute_fractal_participation should be a callable function"

def test_compute_hubness_score_exists():
    """Test that compute_hubness_score function exists and is callable."""
    assert callable(compute_hubness_score), "compute_hubness_score should be a callable function"

def test_fractal_participation_loop(loop_graph):
    """Test fractal participation on a loop graph."""
    # Nodes in loops should have higher fractal participation
    for node in loop_graph.nodes():
        score = compute_fractal_participation(loop_graph, node)
        assert score > 0, f"Node {node} in loop should have positive fractal participation"

def test_fractal_participation_linear(linear_graph):
    """Test fractal participation on a linear graph."""
    # End nodes should have lower fractal participation
    end_nodes = [0, 4]  # First and last nodes in path_graph(5)
    middle_nodes = [1, 2, 3]
    
    for node in end_nodes:
        score = compute_fractal_participation(linear_graph, node)
        assert 0 <= score <= 0.5, f"End node {node} should have low fractal participation"
    
    # Middle nodes may have slightly higher participation but still relatively low
    for node in middle_nodes:
        score = compute_fractal_participation(linear_graph, node)
        assert 0 <= score <= 0.7, f"Middle node {node} should have moderate fractal participation"

def test_hubness_score_star(star_graph):
    """Test hubness score on a star graph."""
    # Central node (0) should have high hubness
    central_score = compute_hubness_score(star_graph, 0)
    assert central_score > 0.8, f"Central node should have high hubness, got {central_score}"
    
    # Leaf nodes should have low hubness
    for node in range(1, 6):  # Nodes 1-5 in star_graph(5)
        score = compute_hubness_score(star_graph, node)
        assert 0 <= score <= 0.3, f"Leaf node {node} should have low hubness, got {score}"

def test_hubness_score_linear(linear_graph):
    """Test hubness score on a linear graph."""
    # All nodes should have relatively low hubness, with middle nodes slightly higher
    for node in linear_graph.nodes():
        score = compute_hubness_score(linear_graph, node)
        assert 0 <= score <= 0.5, f"Node {node} in linear graph should have low-to-moderate hubness, got {score}"

def test_complex_graph_metrics(complex_graph):
    """Test both metrics on a more complex graph with known structures."""
    # Loop nodes (A, B, C) should have higher fractal participation
    for node in ['A', 'B', 'C']:
        fp_score = compute_fractal_participation(complex_graph, node)
        assert fp_score > 0.5, f"Loop node {node} should have high fractal participation, got {fp_score}"
    
    # Hub node (D) should have high hubness
    hub_score = compute_hubness_score(complex_graph, 'D')
    assert hub_score > 0.7, f"Hub node D should have high hubness, got {hub_score}"
    
    # End nodes (E, F, G, L) should have low hubness and low fractal participation
    for node in ['E', 'F', 'G', 'L']:
        h_score = compute_hubness_score(complex_graph, node)
        fp_score = compute_fractal_participation(complex_graph, node)
        assert h_score < 0.3, f"End node {node} should have low hubness, got {h_score}"
        assert fp_score < 0.3, f"End node {node} should have low fractal participation, got {fp_score}"

def test_build_decision_table_with_node_metrics(star_graph):
    """Test that build_decision_table includes both global and node-level metrics."""
    # Create test data
    D = 1.5  # Global fractal dimension
    alpha = 2.0  # Global power-law exponent
    FB = {str(node): 0.1 * node for node in star_graph.nodes}
    threshold = 0.5

    # Convert integer nodes to strings for the test
    string_graph = nx.Graph()
    for node in star_graph.nodes():
        string_graph.add_node(str(node))
    for u, v in star_graph.edges():
        string_graph.add_edge(str(u), str(v))

    # Mock chokepoints dataframe
    import pandas as pd
    chokepoints = pd.DataFrame({
        'page': [str(node) for node in star_graph.nodes],
        'event': ['click'] * len(star_graph.nodes),
        'WSJF_Friction_Score': [0.1 * node for node in star_graph.nodes]
    })

    # Call function
    table = build_decision_table(string_graph, D, alpha, FB, threshold, chokepoints)

    # Check result
    assert isinstance(table, pd.DataFrame), "Should return a DataFrame"
    assert len(table) == len(star_graph.nodes), "Should have a row for each node"

    # Both global and node-level metrics should be present
    assert "D" in table.columns, "Should contain D column (global metric)"
    assert "alpha" in table.columns, "Should contain alpha column (global metric)"
    assert "fractal_participation" in table.columns, "Should contain fractal_participation column (node metric)"
    assert "hubness" in table.columns, "Should contain hubness column (node metric)"

    # Global metrics should be the same for all nodes
    assert table["D"].nunique() == 1, "D should be same for all nodes (global property)"
    assert table["alpha"].nunique() == 1, "alpha should be same for all nodes (global property)"
    assert table["D"].iloc[0] == D, "D should match the global value"
    assert table["alpha"].iloc[0] == alpha, "alpha should match the global value"

    # Node-level metrics should be different
    assert table["fractal_participation"].nunique() > 1, "fractal_participation should vary by node"
    assert table["hubness"].nunique() > 1, "hubness should vary by node"

    # Central node (0) should have higher hubness
    central_node_row = table[table["node"] == "0"]
    leaf_node_row = table[table["node"] == "1"]  # Any leaf node

    assert float(central_node_row["hubness"].iloc[0]) > float(leaf_node_row["hubness"].iloc[0]), \
        "Central node should have higher hubness than leaf nodes" 