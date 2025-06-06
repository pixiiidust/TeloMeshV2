import os
import sys
import pytest
import pandas as pd
import json
import pickle
import networkx as nx

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions to test (will be implemented in analysis/event_chokepoints.py)
from analysis.event_chokepoints import (
    compute_exit_rates,
    compute_betweenness,
    compute_wsjf_friction,
    detect_fragile_flows,
    export_chokepoints
)

def test_event_chokepoint_csv_exists():
    """Test that the event_chokepoints.csv file is created after running the analysis."""
    from analysis.event_chokepoints import main
    # Run the main function to generate the outputs
    main(fast=True)
    # Check if the file exists
    assert os.path.exists("outputs/event_chokepoints.csv"), "event_chokepoints.csv file was not created"

def test_required_columns():
    """Test that the event_chokepoints.csv file has all required columns."""
    # Check if the file exists first
    if not os.path.exists("outputs/event_chokepoints.csv"):
        pytest.skip("event_chokepoints.csv not found, skipping test")
    
    # Read the CSV file
    df = pd.read_csv("outputs/event_chokepoints.csv")
    
    # Check required columns
    required_columns = [
        "page", "event", "exit_rate", "betweenness", 
        "users_lost", "WSJF_Friction_Score"
    ]
    
    for column in required_columns:
        assert column in df.columns, f"Required column '{column}' is missing from event_chokepoints.csv"

def test_valid_score_ranges():
    """Test that the scores in event_chokepoints.csv are within valid ranges."""
    # Check if the file exists first
    if not os.path.exists("outputs/event_chokepoints.csv"):
        pytest.skip("event_chokepoints.csv not found, skipping test")
    
    # Read the CSV file
    df = pd.read_csv("outputs/event_chokepoints.csv")
    
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

def test_fragile_flows_file_exists():
    """Test that the high_friction_flows.csv file is created."""
    # Check if the file exists
    assert os.path.exists("outputs/high_friction_flows.csv"), "high_friction_flows.csv file was not created"

def test_output_node_map():
    """Test that the friction_node_map.json file is created and contains valid data."""
    # Check if the file exists
    assert os.path.exists("outputs/friction_node_map.json"), "friction_node_map.json file was not created"
    
    # Read the JSON file
    with open("outputs/friction_node_map.json", "r") as f:
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
    G.add_edge("/home", "/product", event="Click", weight=2)
    G.add_edge("/product", "/cart", event="Add to Cart", weight=1)
    G.add_edge("/cart", "/checkout", event="Checkout", weight=1)
    G.add_edge("/home", "/about", event="Click", weight=1)
    G.add_edge("/about", "/contact", event="Click", weight=1)
    
    # Call compute_betweenness
    centrality_dict = compute_betweenness(G)
    
    # Check that the function returns a dictionary
    assert isinstance(centrality_dict, dict), "compute_betweenness did not return a dictionary"
    
    # Check that all nodes have a betweenness value
    for node in G.nodes():
        assert node in centrality_dict, f"Node {node} is missing from centrality dictionary"
    
    # Check specific betweenness values
    # "/home" should have highest betweenness as it's a starting point for multiple paths
    nodes_by_betweenness = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
    assert nodes_by_betweenness[0][0] == "/home", "Home page should have highest betweenness"
    
    # "/checkout" and "/contact" should have lowest betweenness as they're endpoints
    assert centrality_dict["/checkout"] == 0.0, "Checkout page should have betweenness of 0"
    assert centrality_dict["/contact"] == 0.0, "Contact page should have betweenness of 0" 