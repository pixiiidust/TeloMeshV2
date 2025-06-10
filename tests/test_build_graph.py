import os
import networkx as nx
import pytest
import pickle

def test_graph_file_exists():
    """Test that the output graph file is created."""
    assert os.path.exists("outputs/user_graph.gpickle")

def test_multi_graph_file_exists():
    """Test that the output multi-graph file is created."""
    assert os.path.exists("outputs/user_graph_multi.gpickle")

def test_graph_is_directed():
    """Test that the graph is a directed graph."""
    with open("outputs/user_graph.gpickle", 'rb') as f:
        G = pickle.load(f)
    assert isinstance(G, nx.DiGraph)

def test_multi_graph_is_multi_directed():
    """Test that the multi-graph is a multi-directed graph."""
    with open("outputs/user_graph_multi.gpickle", 'rb') as f:
        G = pickle.load(f)
    assert isinstance(G, nx.MultiDiGraph)

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

def test_multi_graph_has_same_nodes():
    """Test that the multi-graph has the same nodes as the regular graph."""
    with open("outputs/user_graph.gpickle", 'rb') as f:
        G = pickle.load(f)
    with open("outputs/user_graph_multi.gpickle", 'rb') as f:
        G_multi = pickle.load(f)
    assert set(G.nodes) == set(G_multi.nodes), "Multi-graph should have the same nodes as regular graph"

def test_multi_graph_has_at_least_same_edges():
    """Test that the multi-graph has at least as many edges as the regular graph."""
    with open("outputs/user_graph.gpickle", 'rb') as f:
        G = pickle.load(f)
    with open("outputs/user_graph_multi.gpickle", 'rb') as f:
        G_multi = pickle.load(f)
    assert G_multi.number_of_edges() >= len(G.edges), "Multi-graph should have at least as many edges as regular graph"

def test_edge_attributes():
    """Test that all edges have the required attributes."""
    with open("outputs/user_graph.gpickle", 'rb') as f:
        G = pickle.load(f)
    for u, v, data in G.edges(data=True):
        assert "event" in data, f"Edge ({u}, {v}) is missing 'event' attribute"
        assert "weight" in data, f"Edge ({u}, {v}) is missing 'weight' attribute"
        assert isinstance(data["weight"], int), f"Edge ({u}, {v}) weight is not an integer"
        assert data["weight"] > 0, f"Edge ({u}, {v}) weight is not positive"

def test_multi_graph_edge_attributes():
    """Test that all multi-graph edges have the required attributes."""
    with open("outputs/user_graph_multi.gpickle", 'rb') as f:
        G = pickle.load(f)
    for u, v, key, data in G.edges(data=True, keys=True):
        assert "event" in data, f"Edge ({u}, {v}, {key}) is missing 'event' attribute"
        assert "weight" in data, f"Edge ({u}, {v}, {key}) is missing 'weight' attribute"
        assert isinstance(data["weight"], int), f"Edge ({u}, {v}, {key}) weight is not an integer"
        assert data["weight"] > 0, f"Edge ({u}, {v}, {key}) weight is not positive"

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

def test_multi_graph_preserves_all_transitions():
    """Test that the multi-graph preserves all transitions with their events."""
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
    
    # Load the multi-graph
    with open("outputs/user_graph_multi.gpickle", 'rb') as f:
        G_multi = pickle.load(f)
    
    # Check that each transition type is preserved in the multi-graph
    for (from_page, to_page, event), count in transition_counts.items():
        # Check if edges between these nodes exist
        if G_multi.has_edge(from_page, to_page):
            # Get all edges between these nodes
            edges_data = G_multi.get_edge_data(from_page, to_page)
            
            # Find edges with matching event
            matching_edges = [data for key, data in edges_data.items() if data.get("event") == event]
            
            # In a MultiDiGraph, we should have one edge per transition
            assert len(matching_edges) > 0, f"No edge with event '{event}' found between {from_page} and {to_page}"
            
            # Total weight should match the transition count
            total_weight = sum(edge["weight"] for edge in matching_edges)
            assert total_weight == count, f"Total weight for edges ({from_page}, {to_page}, {event}) is {total_weight} but expected {count}" 