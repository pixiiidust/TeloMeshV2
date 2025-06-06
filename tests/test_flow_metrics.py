import os
import sys
import pytest
import json
import pandas as pd
import networkx as nx
import pickle

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingest.flow_metrics import validate_sessions, validate_graph, run_metrics

def test_stats_log_created():
    """Test that the stats log file is created."""
    # Run the metrics
    run_metrics()
    
    # Check if the log file exists
    assert os.path.exists("logs/session_stats.log")

def test_metrics_json_created():
    """Test that the metrics JSON file is created."""
    # Run the metrics
    run_metrics()
    
    # Check if the JSON file exists
    assert os.path.exists("logs/metrics.json")

def test_sessions_per_user_valid():
    """Test that sessions per user is at least 1."""
    # Run session validation
    stats = validate_sessions()
    
    # Check if sessions per user is at least 1
    assert stats["sessions_per_user"] >= 1, f"Expected sessions per user to be at least 1, got {stats['sessions_per_user']}"

def test_avg_flow_length_valid():
    """Test that average flow length is at least 3 steps."""
    # Run session validation
    stats = validate_sessions()
    
    # Check if average flow length is at least 3
    assert stats["avg_flow_length"] >= 3, f"Expected average flow length to be at least 3, got {stats['avg_flow_length']}"

def test_node_count_valid():
    """Test that the graph has at least 10 nodes."""
    # Run graph validation
    stats = validate_graph()
    
    # Check if node count is at least 10
    assert stats["node_count"] >= 10, f"Expected at least 10 nodes, got {stats['node_count']}"

def test_unique_edge_events_valid():
    """Test that the graph has at least 2 unique edge events."""
    # Run graph validation
    stats = validate_graph()
    
    # Check if there are at least 2 unique edge events
    assert stats["unique_edge_events"] >= 2, f"Expected at least 2 unique edge events, got {stats['unique_edge_events']}"

def test_connected_components_valid():
    """Test that the graph has at least 1 connected component."""
    # Run graph validation
    stats = validate_graph()
    
    # Check if there is at least 1 connected component
    assert stats["connected_components"] >= 1, f"Expected at least 1 connected component, got {stats['connected_components']}" 