import os
import networkx as nx
import pytest
import pickle

def test_graph_file_exists():
    """Test that the output graph file is created."""
    assert os.path.exists("outputs/user_graph.gpickle")

def test_graph_is_directed():
    """Test that the graph is a directed graph."""
    with open("outputs/user_graph.gpickle", 'rb') as f:
        G = pickle.load(f)
    assert isinstance(G, nx.DiGraph)

def test_graph_has_nodes():
    """Test that the graph has a reasonable number of nodes."""
    with open("outputs/user_graph.gpickle", 'rb') as f:
        G = pickle.load(f)
    assert len(G.nodes) >= 5, f"Expected at least 5 nodes, got {len(G.nodes)}"

def test_graph_has_edges():
    """Test that the graph has a reasonable number of edges."""
    with open("outputs/user_graph.gpickle", 'rb') as f:
        G = pickle.load(f)
    assert len(G.edges) >= 10, f"Expected at least 10 edges, got {len(G.edges)}"

def test_edge_attributes():
    """Test that all edges have the required attributes."""
    with open("outputs/user_graph.gpickle", 'rb') as f:
        G = pickle.load(f)
    for u, v, data in G.edges(data=True):
        assert "event" in data, f"Edge ({u}, {v}) is missing 'event' attribute"
        assert "weight" in data, f"Edge ({u}, {v}) is missing 'weight' attribute"
        assert isinstance(data["weight"], int), f"Edge ({u}, {v}) weight is not an integer"
        assert data["weight"] > 0, f"Edge ({u}, {v}) weight is not positive"

def test_weights_match_frequency():
    """Test that edge weights match frequency of transitions in the session data."""
    import pandas as pd
    
    # Load the session flows data
    df = pd.read_csv("outputs/session_flows.csv")
    
    # Group by session to count transitions
    transition_counts = {}
    
    for session_id, group in df.groupby("session_id"):
        # Sort by step_index to ensure chronological order
        session_steps = group.sort_values("step_index")
        
        # Count transitions (from_page -> to_page, event)
        for i in range(len(session_steps) - 1):
            from_step = session_steps.iloc[i]
            to_step = session_steps.iloc[i+1]
            
            from_page = from_step["page"]
            to_page = to_step["page"]
            event = to_step["event"]
            
            key = (from_page, to_page, event)
            
            if key not in transition_counts:
                transition_counts[key] = 0
            
            transition_counts[key] += 1
    
    # Load the graph
    with open("outputs/user_graph.gpickle", 'rb') as f:
        G = pickle.load(f)
    
    # Check that edge weights match transition counts
    for (from_page, to_page, event), count in transition_counts.items():
        # Check if the edge exists
        if G.has_edge(from_page, to_page):
            edge_data = G.get_edge_data(from_page, to_page)
            # If there are multiple edges with different events, find the one with matching event
            if isinstance(edge_data, dict) and edge_data.get("event") == event:
                assert edge_data["weight"] == count, f"Edge ({from_page}, {to_page}, {event}) has weight {edge_data['weight']} but expected {count}" 