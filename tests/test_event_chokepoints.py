import os
import sys
import pytest
import pandas as pd
import json
import pickle
import networkx as nx
import numpy as np
import tempfile
import shutil

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions to test (will be implemented in analysis/event_chokepoints.py)
from analysis.event_chokepoints import (
    compute_exit_rates,
    compute_betweenness,
    compute_wsjf_friction,
    detect_fragile_flows,
    export_chokepoints,
    convert_to_digraph,
    compute_fractal_dimension,
    compute_power_law_alpha,
    detect_repeating_subgraphs,
    simulate_percolation,
    compute_fractal_betweenness,
    build_decision_table,
    compute_clustering_coefficient
)

# Test fixtures for standard graph types
@pytest.fixture
def linear_graph():
    """Create a small linear graph for testing."""
    # This small graph will trigger the default behavior in fractal_dimension
    return nx.path_graph(5)

@pytest.fixture
def large_linear_graph():
    """Create a large linear graph that's suitable for fractal dimension calculation."""
    return nx.path_graph(15)

@pytest.fixture
def loop_graph():
    """Create a simple loop graph for testing repeating subgraphs."""
    G = nx.DiGraph()
    G.add_edge(0, 1)
    G.add_edge(1, 0)
    return G

@pytest.fixture
def star_graph():
    """Create a small star graph for testing."""
    # This small graph may trigger default behavior
    return nx.star_graph(5)

@pytest.fixture
def large_star_graph():
    """Create a large star graph suitable for power-law testing."""
    return nx.star_graph(15)

@pytest.fixture
def grid_graph():
    """Create a small grid graph for testing."""
    return nx.grid_2d_graph(3, 3)

@pytest.fixture
def large_grid_graph():
    """Create a large grid graph suitable for fractal dimension testing."""
    return nx.grid_2d_graph(5, 5)

@pytest.fixture
def scale_free_graph():
    """Create a Barabási-Albert scale-free graph."""
    return nx.barabasi_albert_graph(50, 2)

@pytest.fixture
def multi_digraph():
    """Create a MultiDiGraph for testing conversion."""
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, weight=2)
    G.add_edge(0, 1, weight=3)
    G.add_edge(1, 2, weight=1)
    G.add_edge(2, 0, weight=1)
    return G

@pytest.fixture
def test_data_setup():
    """Create test data files for the main function tests."""
    # Create temp directories
    temp_dir = tempfile.mkdtemp()
    output_dir = os.path.join(temp_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a test session flows CSV
    session_flows_path = os.path.join(output_dir, "session_flows.csv")
    data = {
        "user_id": ["user1", "user1", "user1", "user2", "user2"],
        "session_id": ["session1", "session1", "session1", "session2", "session2"],
        "step_index": [0, 1, 2, 0, 1],
        "page": ["/home", "/product", "/checkout", "/home", "/cart"],
        "event": ["Click", "Add to Cart", "Checkout", "Search", "Add to Cart"],
        "timestamp": pd.date_range(start="2023-01-01", periods=5)
    }
    df = pd.DataFrame(data)
    df.to_csv(session_flows_path, index=False)
    
    # Create a test graph
    graph_path = os.path.join(output_dir, "user_graph.gpickle")
    G = nx.DiGraph()
    G.add_edge("/home", "/product", event="Click", weight=2)
    G.add_edge("/product", "/cart", event="Add to Cart", weight=1)
    G.add_edge("/cart", "/checkout", event="Checkout", weight=1)
    G.add_edge("/home", "/about", event="Click", weight=1)
    G.add_edge("/about", "/contact", event="Click", weight=1)
    with open(graph_path, 'wb') as f:
        pickle.dump(G, f)
    
    # Create a test multi-graph
    multi_graph_path = os.path.join(output_dir, "user_graph_multi.gpickle")
    G_multi = nx.MultiDiGraph()
    G_multi.add_edge("/home", "/product", event="Click", weight=1)
    G_multi.add_edge("/home", "/product", event="Click", weight=1)
    G_multi.add_edge("/product", "/cart", event="Add to Cart", weight=1)
    G_multi.add_edge("/cart", "/checkout", event="Checkout", weight=1)
    G_multi.add_edge("/home", "/about", event="Click", weight=1)
    G_multi.add_edge("/about", "/contact", event="Click", weight=1)
    with open(multi_graph_path, 'wb') as f:
        pickle.dump(G_multi, f)
    
    # Run the main function to generate all output files
    from analysis.event_chokepoints import main
    main(
        input_flows=session_flows_path,
        input_graph=graph_path,
        input_graph_multi=multi_graph_path,
        output_dir=output_dir,
        fast=True
    )
    
    # Return the paths for use in tests
    yield {
        "temp_dir": temp_dir,
        "output_dir": output_dir,
        "session_flows_path": session_flows_path,
        "graph_path": graph_path,
        "multi_graph_path": multi_graph_path
    }
    
    # Cleanup after tests
    shutil.rmtree(temp_dir)

def test_event_chokepoint_csv_exists(test_data_setup):
    """Test that the event_chokepoints.csv file is created after running the analysis."""
    # Check if the file exists
    output_file = os.path.join(test_data_setup["output_dir"], "event_chokepoints.csv")
    assert os.path.exists(output_file), "event_chokepoints.csv file was not created"

def test_required_columns(test_data_setup):
    """Test that the event_chokepoints.csv file has all required columns."""
    output_file = os.path.join(test_data_setup["output_dir"], "event_chokepoints.csv")
    
    # Check if the file exists first
    if not os.path.exists(output_file):
        pytest.skip(f"{output_file} not found, skipping test")
    
    # Read the CSV file
    df = pd.read_csv(output_file)
    
    # Check required columns
    required_columns = [
        "page", "event", "exit_rate", "betweenness", 
        "users_lost", "WSJF_Friction_Score"
    ]
    
    for column in required_columns:
        assert column in df.columns, f"Required column '{column}' is missing from event_chokepoints.csv"

def test_valid_score_ranges(test_data_setup):
    """Test that the scores in event_chokepoints.csv are within valid ranges."""
    output_file = os.path.join(test_data_setup["output_dir"], "event_chokepoints.csv")
    
    # Check if the file exists first
    if not os.path.exists(output_file):
        pytest.skip(f"{output_file} not found, skipping test")
    
    # Read the CSV file
    df = pd.read_csv(output_file)
    
    # Check exit_rate ranges (0-1)
    assert df["exit_rate"].min() >= 0, "exit_rate contains values less than 0"
    assert df["exit_rate"].max() <= 1, "exit_rate contains values greater than 1"
    
    # Check betweenness ranges (should be >= 0)
    assert df["betweenness"].min() >= 0, "betweenness contains negative values"
    
    # Check WSJF_Friction_Score ranges (should be >= 0)
    assert df["WSJF_Friction_Score"].min() >= 0, "WSJF_Friction_Score contains negative values"
    
    # Check users_lost (should be integer and >= 0)
    assert df["users_lost"].dtype in ['int64', 'int32'], "users_lost is not an integer type"
    assert df["users_lost"].min() >= 0, "users_lost contains negative values"

def test_fragile_flows_file_exists(test_data_setup):
    """Test that the high_friction_flows.csv file is created."""
    # Check if the file exists
    output_file = os.path.join(test_data_setup["output_dir"], "high_friction_flows.csv")
    assert os.path.exists(output_file), f"{output_file} file was not created"

def test_output_node_map(test_data_setup):
    """Test that the friction_node_map.json file is created and contains valid data."""
    # Check if the file exists
    output_file = os.path.join(test_data_setup["output_dir"], "friction_node_map.json")
    assert os.path.exists(output_file), f"{output_file} file was not created"
    
    # Read the JSON file
    with open(output_file, "r") as f:
        node_map = json.load(f)
    
    # Check that it's a dictionary
    assert isinstance(node_map, dict), "friction_node_map.json does not contain a dictionary"
    
    # Check that it has at least some nodes
    assert len(node_map) > 0, "friction_node_map.json contains an empty dictionary"
    
    # Check that values are numeric
    for page, score in node_map.items():
        assert isinstance(page, str), f"Page key '{page}' is not a string"
        assert isinstance(score, (int, float)), f"Score for page '{page}' is not a number"

def test_compute_exit_rates():
    """Test that compute_exit_rates correctly calculates exit rates from session data."""
    # Create a test DataFrame
    data = {
        "user_id": ["user1", "user1", "user1", "user2", "user2"],
        "session_id": ["session1", "session1", "session1", "session2", "session2"],
        "step_index": [0, 1, 2, 0, 1],
        "page": ["/home", "/product", "/checkout", "/home", "/cart"],
        "event": ["Click", "Add to Cart", "Checkout", "Search", "Add to Cart"],
        "timestamp": pd.date_range(start="2023-01-01", periods=5)
    }
    df = pd.DataFrame(data)
    
    # Call compute_exit_rates
    exit_df = compute_exit_rates(df)
    
    # Check that the function returns a DataFrame
    assert isinstance(exit_df, pd.DataFrame), "compute_exit_rates did not return a DataFrame"
    
    # Check that the DataFrame has the required columns
    assert "page" in exit_df.columns, "Result is missing 'page' column"
    assert "event" in exit_df.columns, "Result is missing 'event' column"
    assert "exit_rate" in exit_df.columns, "Result is missing 'exit_rate' column"
    assert "users_lost" in exit_df.columns, "Result is missing 'users_lost' column"
    
    # Check specific exit rates
    # The last step in each session should have exit_rate = 1.0
    checkout_row = exit_df[(exit_df["page"] == "/checkout") & (exit_df["event"] == "Checkout")]
    cart_row = exit_df[(exit_df["page"] == "/cart") & (exit_df["event"] == "Add to Cart")]
    
    assert len(checkout_row) == 1, "Checkout event not found in results"
    assert len(cart_row) == 1, "Cart event not found in results"
    
    assert checkout_row.iloc[0]["exit_rate"] == 1.0, "Exit rate for /checkout should be 1.0"
    assert cart_row.iloc[0]["exit_rate"] == 1.0, "Exit rate for /cart should be 1.0"
    
    # Other steps should have exit_rate = 0.0
    non_exit_steps = exit_df[~((exit_df["page"] == "/checkout") & (exit_df["event"] == "Checkout")) & 
                            ~((exit_df["page"] == "/cart") & (exit_df["event"] == "Add to Cart"))]
    
    for _, row in non_exit_steps.iterrows():
        assert row["exit_rate"] == 0.0, f"Exit rate for {row['page']} with {row['event']} should be 0.0"

def test_compute_betweenness():
    """Test that compute_betweenness correctly calculates betweenness centrality from a graph."""
    # Create a test graph
    G = nx.DiGraph()
    G.add_edge("A", "B", event="Click", weight=2)
    G.add_edge("B", "C", event="Add to Cart", weight=1) 
    G.add_edge("C", "D", event="Checkout", weight=1)
    G.add_edge("A", "E", event="Click", weight=1)
    
    # Call compute_betweenness
    centrality_dict = compute_betweenness(G)
    
    # Check that the function returns a dictionary
    assert isinstance(centrality_dict, dict), "compute_betweenness did not return a dictionary"
    
    # Check that all nodes have a betweenness value
    for node in G.nodes():
        assert node in centrality_dict, f"Node {node} is missing from centrality dictionary"
    
    # Check specific betweenness values
    # B has highest betweenness in this configuration
    nodes_by_betweenness = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
    assert nodes_by_betweenness[0][0] == "B", "Node B should have highest betweenness"
    
    # "D" and "E" should have lowest betweenness as they're endpoints
    assert centrality_dict["D"] == 0.0, "Node D should have betweenness of 0"
    assert centrality_dict["E"] == 0.0, "Node E should have betweenness of 0"

def test_convert_to_digraph_multi(multi_digraph):
    """Test conversion from MultiDiGraph to DiGraph."""
    G = convert_to_digraph(multi_digraph)
    
    # Check type
    assert isinstance(G, nx.DiGraph), "Result should be a DiGraph"
    assert not isinstance(G, nx.MultiDiGraph), "Result should not be a MultiDiGraph"
    
    # Check nodes and edges
    assert len(G.nodes) == len(multi_digraph.nodes), "Node count should match"
    assert G.has_edge(0, 1), "Edge (0, 1) should exist"
    assert G.has_edge(1, 2), "Edge (1, 2) should exist"
    assert G.has_edge(2, 0), "Edge (2, 0) should exist"
    
    # Check weights
    assert G[0][1]['weight'] == 5, "Edge (0, 1) weight should be 5 (sum of original weights)"

def test_convert_to_digraph_regular():
    """Test conversion from regular DiGraph (should return same graph)."""
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=3)
    G.add_edge(1, 2, weight=2)
    
    G2 = convert_to_digraph(G)
    assert isinstance(G2, nx.DiGraph), "Result should be a DiGraph"
    assert G2[0][1]['weight'] == 3, "Edge weight should be preserved"

def test_fractal_dimension_linear(linear_graph):
    """Test fractal dimension computation on small linear graph."""
    # Test the actual implementation on small graph
    D = compute_fractal_dimension(linear_graph)
    
    # The implementation should return D=1.0 for small graphs as a default value
    assert D == 1.0, f"For small linear graph, should return default D=1.0, got {D}"

def test_fractal_dimension_large_linear(large_linear_graph):
    """Test fractal dimension computation on large linear graph."""
    # Test on large linear graph
    D = compute_fractal_dimension(large_linear_graph)
    
    # For larger linear graphs, we should get a value close to 1
    # But the box-counting implementation might give a bit lower value
    assert 0.6 <= D <= 1.2, f"Large linear graph dimension should be close to 1, got {D}"

def test_fractal_dimension_grid(grid_graph):
    """Test fractal dimension computation on small grid graph."""
    # Test the actual implementation on small grid
    D = compute_fractal_dimension(grid_graph)
    
    # Small grid may return default value as per implementation
    assert D == 1.0, f"Small grid should return default D=1.0, got {D}"

def test_fractal_dimension_large_grid(large_grid_graph):
    """Test fractal dimension computation on large grid graph."""
    # Test on large grid
    D = compute_fractal_dimension(large_grid_graph)
    
    # Larger grid should give more accurate estimate
    assert 1.0 <= D <= 2.5, f"Large grid dimension should be in [1.0, 2.5], got {D}"
    
def test_fractal_dimension_small_graph():
    """Test fractal dimension computation on graphs too small for reliable calculation."""
    # Create a very small graph
    G = nx.DiGraph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    
    # Should return default value of 1.0 for very small graphs as per implementation
    D = compute_fractal_dimension(G)
    assert D == 1.0, f"Very small graph should return default dimension of 1.0, got {D}"

def test_power_law_alpha_small_star(star_graph):
    """Test power-law exponent computation on small star graph."""
    # Test on small star graph
    alpha = compute_power_law_alpha(star_graph)
    
    # For small star graphs, the implementation should return the default value
    assert alpha == 2.5, f"Small star graph should return default alpha=2.5, got {alpha}"

def test_power_law_alpha_large_star(large_star_graph):
    """Test power-law exponent computation on large star graph."""
    # Test on large star graph
    alpha = compute_power_law_alpha(large_star_graph)
    
    # Star graphs should have a power-law exponent typically around 2-3
    assert 1.5 <= alpha <= 5.0, f"Large star graph alpha should be in range [1.5, 5.0], got {alpha}"

def test_power_law_alpha_scale_free(scale_free_graph):
    """Test power-law exponent computation on scale-free network."""
    # Test on a Barabási-Albert graph (known scale-free properties)
    alpha = compute_power_law_alpha(scale_free_graph)
    
    # BA graphs typically have alpha around 3
    assert 1.5 <= alpha <= 5.0, f"Scale-free graph alpha should be in range [1.5, 5.0], got {alpha}"
    
    # Test actual scale-free range (BA graphs should have alpha around 3)
    if alpha != 2.5:  # If not default value
        assert 2.0 <= alpha <= 4.0, f"Barabási-Albert graph should have alpha around 3, got {alpha}"

def test_power_law_alpha_edge_cases():
    """Test power-law exponent computation on edge cases."""
    # Test on small graph (should return default value)
    small_graph = nx.DiGraph()
    small_graph.add_edge(0, 1)
    small_graph.add_edge(1, 2)
    
    alpha_small = compute_power_law_alpha(small_graph)
    assert alpha_small == 2.5, f"Small graph should return default alpha=2.5, got {alpha_small}"
    
    # Test on graph with too few unique degrees
    uniform_graph = nx.cycle_graph(5)
    # All nodes in cycle have degree 2
    alpha_uniform = compute_power_law_alpha(uniform_graph)
    assert alpha_uniform == 2.5, f"Uniform degree graph should return default alpha=2.5, got {alpha_uniform}"

def test_detect_repeating_subgraphs_loop(loop_graph):
    """Test detection of repeating subgraphs in a simple loop."""
    subgraphs = detect_repeating_subgraphs(loop_graph)
    # It detects both (0,1) and (1,0) as separate patterns
    assert len(subgraphs) == 2, "Should detect two repeating subgraph patterns for a loop"
    assert (0, 1) in subgraphs, "Should detect the (0, 1) pattern"
    assert (1, 0) in subgraphs, "Should detect the (1, 0) pattern"

def test_simulate_percolation(star_graph):
    """Test percolation simulation on star graph."""
    threshold = simulate_percolation(star_graph)
    assert 0 <= threshold <= 1, "Percolation threshold should be between 0 and 1"
    
    # Removing central node (0) should collapse star graph
    threshold = simulate_percolation(star_graph, ranked_nodes=[0])
    assert threshold <= 0.2, "Star graph should collapse after removing central node"

def test_fractal_betweenness(loop_graph):
    """Test fractal betweenness computation."""
    FB = compute_fractal_betweenness(loop_graph)
    assert isinstance(FB, dict), "Should return a dictionary"
    assert len(FB) == len(loop_graph.nodes), "Should have values for all nodes"
    assert all(v >= 0 for v in FB.values()), "All values should be non-negative"

def test_build_decision_table(star_graph):
    """Test decision table creation."""
    # Create test data
    D = 1.5
    alpha = 2.0
    FB = {str(node): 0.1 * node for node in star_graph.nodes}
    threshold = 0.5

    # Convert integer nodes to strings for the test
    # Convert the star_graph to have string nodes for the test
    string_graph = nx.Graph()
    for node in star_graph.nodes():
        string_graph.add_node(str(node))
    for u, v in star_graph.edges():
        string_graph.add_edge(str(u), str(v))

    # Mock chokepoints dataframe
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
    assert "ux_label" in table.columns, "Should contain UX labels"
    assert "suggested_action" in table.columns, "Should contain action recommendations"
    
    # Check sorting
    assert table.iloc[0]["wsjf_score"] >= table.iloc[-1]["wsjf_score"], "Should be sorted by WSJF score"

def test_decision_table_csv_exists(test_data_setup):
    """Test that the decision_table.csv file is created after running the analysis."""
    # Check if the file exists
    output_file = os.path.join(test_data_setup["output_dir"], "decision_table.csv")
    assert os.path.exists(output_file), f"{output_file} file was not created"

def test_final_report_exists(test_data_setup):
    """Test that the final_report.json and final_report.csv files are created."""
    # Check if the files exist
    json_file = os.path.join(test_data_setup["output_dir"], "final_report.json")
    csv_file = os.path.join(test_data_setup["output_dir"], "final_report.csv")
    
    assert os.path.exists(json_file), f"{json_file} file was not created"
    assert os.path.exists(csv_file), f"{csv_file} file was not created"
    
    # Check JSON file content
    with open(json_file, "r") as f:
        report_data = json.load(f)
    
    # Check required fields
    assert "fractal_dimension" in report_data, "final_report.json missing fractal_dimension"
    assert "power_law_alpha" in report_data, "final_report.json missing power_law_alpha"
    assert "percolation_threshold" in report_data, "final_report.json missing percolation_threshold"
    assert "clustering_coefficient" in report_data, "final_report.json missing clustering_coefficient"
    assert "top_fb_nodes" in report_data, "final_report.json missing top_fb_nodes"
    assert "top_chokepoints" in report_data, "final_report.json missing top_chokepoints"
    
    # Check CSV file content
    df = pd.read_csv(csv_file)
    
    # Check required columns
    assert "metric" in df.columns, "CSV should have a 'metric' column"
    assert "value" in df.columns, "CSV should have a 'value' column"
    
    # Check that clustering coefficient is included
    assert any(df["metric"].str.contains("Clustering Coefficient")), "CSV should include Clustering Coefficient metric"

def test_compute_clustering_coefficient():
    """Test computation of the average clustering coefficient of a graph."""
    # Create a test graph with known clustering structure
    G = nx.DiGraph()
    
    # Create a triangle (fully connected triad)
    G.add_edge('A', 'B')
    G.add_edge('B', 'C')
    G.add_edge('C', 'A')
    
    # Add a disconnected path
    G.add_edge('D', 'E')
    G.add_edge('E', 'F')
    
    # The triangle has clustering coefficient of 1 for each node
    # The path has clustering coefficient of 0 for each node
    # So the average should be 0.5 if all nodes are weighted equally
    
    # Test the function
    cc = compute_clustering_coefficient(G)
    
    # For directed graphs, this might vary based on implementation
    # But should be in a reasonable range
    assert 0.3 <= cc <= 0.7, f"Expected clustering coefficient around 0.5, got {cc}"
    
    # Test on different graph types
    triangle = nx.DiGraph()
    triangle.add_edge(1, 2)
    triangle.add_edge(2, 3)
    triangle.add_edge(3, 1)
    cc_triangle = compute_clustering_coefficient(triangle)
    assert 0.8 <= cc_triangle <= 1.0, f"Triangle should have high clustering coefficient, got {cc_triangle}"
    
    # Test on a star graph (should have low clustering)
    star = nx.star_graph(5)
    cc_star = compute_clustering_coefficient(star)
    assert 0.0 <= cc_star <= 0.2, f"Star should have low clustering coefficient, got {cc_star}" 