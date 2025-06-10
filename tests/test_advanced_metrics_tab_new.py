import os
import sys
import pytest
import pandas as pd
import json
import networkx as nx
import pickle
import importlib.util
from unittest.mock import patch, MagicMock, call
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def test_data_setup():
    """Create a test environment with real data paths."""
    # Look for an existing dataset in outputs directory to use for testing
    outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    
    # Find the first valid dataset
    valid_dataset = None
    if os.path.exists(outputs_dir):
        for dataset in os.listdir(outputs_dir):
            dataset_dir = os.path.join(outputs_dir, dataset)
            if os.path.isdir(dataset_dir):
                required_files = [
                    "decision_table.csv",
                    "final_report.csv",
                    "final_report.json",
                    "event_chokepoints.csv",
                    "friction_node_map.json",
                    "user_graph.gpickle"
                ]
                if all(os.path.exists(os.path.join(dataset_dir, f)) for f in required_files):
                    valid_dataset = dataset
                    break
    
    if valid_dataset:
        dataset_dir = os.path.join(outputs_dir, valid_dataset)
        return {
            "dataset_name": valid_dataset,
            "dataset_dir": dataset_dir,
            "decision_table_path": os.path.join(dataset_dir, "decision_table.csv"),
            "final_report_csv_path": os.path.join(dataset_dir, "final_report.csv"),
            "final_report_json_path": os.path.join(dataset_dir, "final_report.json"),
            "event_chokepoints_path": os.path.join(dataset_dir, "event_chokepoints.csv"),
            "node_map_path": os.path.join(dataset_dir, "friction_node_map.json"),
            "graph_path": os.path.join(dataset_dir, "user_graph.gpickle")
        }
    else:
        pytest.skip("No valid dataset found for testing. Run the TeloMesh pipeline first.")

# Test the Advanced Metrics Tab Core Functions
def test_compute_fractal_dimension_exists():
    """Test that the compute_fractal_dimension function exists."""
    from analysis.event_chokepoints import compute_fractal_dimension
    assert callable(compute_fractal_dimension), "compute_fractal_dimension should be a callable function"

def test_compute_fractal_dimension_returns_float():
    """Test that compute_fractal_dimension returns a float value."""
    from analysis.event_chokepoints import compute_fractal_dimension
    
    # Create a simple test graph
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "A")
    
    # Call the function
    result = compute_fractal_dimension(G)
    
    # Assert that the result is a float
    assert isinstance(result, float), "compute_fractal_dimension should return a float"
    
    # Fractal dimension should be within a reasonable range for networks
    assert 0.0 <= result <= 3.0, "Fractal dimension should be between 0 and 3"

def test_compute_power_law_alpha_exists():
    """Test that the compute_power_law_alpha function exists."""
    from analysis.event_chokepoints import compute_power_law_alpha
    assert callable(compute_power_law_alpha), "compute_power_law_alpha should be a callable function"

def test_compute_power_law_alpha_returns_float():
    """Test that compute_power_law_alpha returns a float value."""
    from analysis.event_chokepoints import compute_power_law_alpha
    
    # Create a simple test graph
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "D")
    G.add_edge("A", "D")
    G.add_edge("A", "E")
    
    # Call the function
    result = compute_power_law_alpha(G)
    
    # Assert that the result is a float
    assert isinstance(result, float), "compute_power_law_alpha should return a float"
    
    # Power law alpha should be within a reasonable range for networks
    assert 1.0 <= result <= 5.0, "Power law alpha should be between 1 and 5"

def test_simulate_percolation_exists():
    """Test that the simulate_percolation function exists."""
    from analysis.event_chokepoints import simulate_percolation
    assert callable(simulate_percolation), "simulate_percolation should be a callable function"

def test_simulate_percolation_returns_float():
    """Test that simulate_percolation returns a float value."""
    from analysis.event_chokepoints import simulate_percolation
    
    # Create a simple test graph
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "D")
    G.add_edge("A", "D")
    G.add_edge("A", "E")
    
    # Call the function
    result = simulate_percolation(G)
    
    # Assert that the result is a float
    assert isinstance(result, float), "simulate_percolation should return a float"
    
    # Percolation threshold should be between 0 and 1
    assert 0.0 <= result <= 1.0, "Percolation threshold should be between 0 and 1"

def test_detect_repeating_subgraphs_exists():
    """Test that the detect_repeating_subgraphs function exists."""
    from analysis.event_chokepoints import detect_repeating_subgraphs
    assert callable(detect_repeating_subgraphs), "detect_repeating_subgraphs should be a callable function"

def test_detect_repeating_subgraphs_returns_dict():
    """Test that detect_repeating_subgraphs returns a dictionary with the expected keys."""
    from analysis.event_chokepoints import detect_repeating_subgraphs
    
    # Create a simple test graph with a cycle
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "A")
    
    # Call the function
    result = detect_repeating_subgraphs(G)
    
    # Assert that the result is a dictionary
    assert isinstance(result, dict), "detect_repeating_subgraphs should return a dictionary"
    
    # Check for expected keys
    assert "recurring_patterns" in result, "Result should contain 'recurring_patterns' key"
    assert "node_loop_counts" in result, "Result should contain 'node_loop_counts' key"
    assert "total_patterns" in result, "Result should contain 'total_patterns' key"
    
    # Check that at least one pattern was found
    assert result["total_patterns"] > 0, "At least one pattern should be found in the cycle graph"
    
    # Check that all nodes are in the node_loop_counts
    for node in G.nodes():
        assert node in result["node_loop_counts"], f"Node {node} should be in node_loop_counts"

def test_compute_fractal_betweenness_exists():
    """Test that the compute_fractal_betweenness function exists."""
    from analysis.event_chokepoints import compute_fractal_betweenness
    assert callable(compute_fractal_betweenness), "compute_fractal_betweenness should be a callable function"

def test_compute_fractal_betweenness_returns_dict():
    """Test that compute_fractal_betweenness returns a dictionary with node scores."""
    from analysis.event_chokepoints import compute_fractal_betweenness
    
    # Create a simple test graph
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "D")
    G.add_edge("A", "D")
    
    # Call the function
    result = compute_fractal_betweenness(G)
    
    # Assert that the result is a dictionary
    assert isinstance(result, dict), "compute_fractal_betweenness should return a dictionary"
    
    # Check that all nodes have a score
    for node in G.nodes():
        assert node in result, f"Node {node} should have a fractal betweenness score"
        assert isinstance(result[node], (int, float)), f"Score for node {node} should be a number"

def test_build_decision_table_exists():
    """Test that the build_decision_table function exists."""
    from analysis.event_chokepoints import build_decision_table
    assert callable(build_decision_table), "build_decision_table should be a callable function"

def test_build_decision_table_returns_dataframe():
    """Test that build_decision_table returns a DataFrame with the expected columns."""
    from analysis.event_chokepoints import build_decision_table, compute_fractal_betweenness
    
    # Create a simple test graph
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "D")
    G.add_edge("A", "D")
    
    # Create required inputs
    D = 1.5
    alpha = 2.3
    FB = compute_fractal_betweenness(G)
    threshold = 0.5
    chokepoints = pd.DataFrame({
        'page': ['A', 'B', 'C', 'D'],
        'event': ['click', 'view', 'scroll', 'click'],
        'WSJF_Friction_Score': [0.5, 0.6, 0.7, 0.8]
    })
    
    # Call the function
    result = build_decision_table(G, D, alpha, FB, threshold, chokepoints)
    
    # Assert that the result is a DataFrame
    assert isinstance(result, pd.DataFrame), "build_decision_table should return a DataFrame"
    
    # Check for expected columns
    expected_columns = ["node", "FB", "percolation_role", "wsjf_score", "ux_label", "suggested_action"]
    for col in expected_columns:
        assert col in result.columns, f"Result should contain column {col}"
    
    # Check that all nodes are in the table
    node_list = result["node"].tolist()
    for node in G.nodes():
        assert node in node_list, f"Node {node} should be in the decision table"

# Test the Advanced Metrics Tab UI Components
def test_load_advanced_metrics_function_exists():
    """Test that the load_advanced_metrics function exists."""
    from ui.dashboard import load_advanced_metrics
    assert callable(load_advanced_metrics), "load_advanced_metrics should be a callable function"

def test_render_top_metrics_function_exists():
    """Test that the render_top_metrics function exists."""
    from ui.dashboard import render_top_metrics
    assert callable(render_top_metrics), "render_top_metrics should be a callable function"

def test_render_top_metrics_displays_diagnostic_badges():
    """Test that render_top_metrics displays diagnostic badges based on metric values."""
    from ui.dashboard import render_top_metrics
    
    # Test cases with different threshold values
    test_cases = [
        # Good values
        {
            "fractal_dimension": 1.5,  # Good (1.0-2.2)
            "power_law_alpha": 2.0,    # Good (1.8-2.6)
            "percolation_threshold": 0.6,  # Good (>0.5)
            "expected_badges": ["✅", "✅", "✅"]
        },
        # Warning values
        {
            "fractal_dimension": 2.3,  # Warning (2.2-2.6)
            "power_law_alpha": 1.7,    # Warning (1.5-1.8)
            "percolation_threshold": 0.4,  # Warning (0.3-0.5)
            "expected_badges": ["⚠️", "⚠️", "⚠️"]
        },
        # Severe risk values
        {
            "fractal_dimension": 2.7,  # Severe (>2.6)
            "power_law_alpha": 1.4,    # Severe (<1.5)
            "percolation_threshold": 0.2,  # Severe (<0.3)
            "expected_badges": ["❌", "❌", "❌"]
        }
    ]
    
    for case in test_cases:
        metrics_data = {
            "fractal_dimension": case["fractal_dimension"],
            "power_law_alpha": case["power_law_alpha"],
            "percolation_threshold": case["percolation_threshold"],
            "network_metrics": {"critical_nodes_count": 3}
        }
        
        # Mock streamlit's columns and metric functions
        with patch("streamlit.columns", return_value=[MagicMock(), MagicMock(), MagicMock()]) as mock_columns, \
             patch("streamlit.metric") as mock_metric, \
             patch("streamlit.markdown") as mock_markdown:
            
            # Call the function
            render_top_metrics(metrics_data)
            
            # Assert that markdown was called for each badge
            for badge in case["expected_badges"]:
                found_badge = False
                for call_args in mock_markdown.call_args_list:
                    args, _ = call_args
                    if args and isinstance(args[0], str) and badge in args[0]:
                        found_badge = True
                        break
                assert found_badge, f"Badge {badge} should be displayed for values {metrics_data}"

def test_render_decision_table_function_exists():
    """Test that the render_decision_table function exists."""
    from ui.dashboard import render_decision_table
    assert callable(render_decision_table), "render_decision_table should be a callable function"

def test_render_decision_table_removes_global_metrics():
    """Test that render_decision_table removes global metrics columns."""
    from ui.dashboard import render_decision_table
    
    # Create a mock metrics data dictionary with a decision table that includes D and alpha
    decision_table = pd.DataFrame({
        'node': ['page1', 'page2', 'page3'],
        'D': [1.2, 1.2, 1.2],  # Global D (same for all nodes)
        'alpha': [2.1, 2.1, 2.1],  # Global alpha (same for all nodes)
        'FB': [0.1, 0.2, 0.3],
        'percolation_role': ['critical', 'standard', 'critical'],
        'wsjf_score': [0.5, 0.6, 0.7],
        'ux_label': ['label1', 'label2', 'label3'],
        'suggested_action': ['action1', 'action2', 'action3']
    })
    
    metrics_data = {
        "decision_table": decision_table,
        "fractal_dimension": 1.2,
        "power_law_alpha": 2.1,
        "percolation_threshold": 0.5
    }
    
    # Mock streamlit's dataframe function
    with patch("streamlit.dataframe", MagicMock()) as mock_st_dataframe, \
         patch("streamlit.subheader", MagicMock()), \
         patch("streamlit.write", MagicMock()), \
         patch("streamlit.selectbox", MagicMock(return_value="All")), \
         patch("streamlit.slider", MagicMock(return_value=0.0)):
        
        render_decision_table(metrics_data)
        
        # Get the DataFrame that was passed to st.dataframe
        mock_st_dataframe.assert_called_once()
        args, _ = mock_st_dataframe.call_args
        df = args[0]
        
        # Verify that the DataFrame doesn't contain global metrics
        assert "D" not in df.columns, "D column should be removed from decision table"
        assert "alpha" not in df.columns, "alpha column should be removed from decision table"

def test_render_fb_vs_wsjf_chart_function_exists():
    """Test that the render_fb_vs_wsjf_chart function exists."""
    from ui.dashboard import render_fb_vs_wsjf_chart
    assert callable(render_fb_vs_wsjf_chart), "render_fb_vs_wsjf_chart should be a callable function"

def test_render_fb_vs_wsjf_chart_has_quadrants():
    """Test that render_fb_vs_wsjf_chart includes quadrant shading and labels."""
    from ui.dashboard import render_fb_vs_wsjf_chart
    
    # Create a mock metrics data dictionary
    decision_table = pd.DataFrame({
        'node': ['page1', 'page2', 'page3', 'page4', 'page5', 'page6', 'page7', 'page8', 'page9', 'page10'],
        'FB': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
        'percolation_role': ['critical', 'critical', 'standard', 'standard', 'standard', 'standard', 'standard', 'standard', 'standard', 'standard'],
        'wsjf_score': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
        'ux_label': ['redundant bottleneck', 'complex hub', 'standard', 'standard', 'standard', 'standard', 'standard', 'standard', 'standard', 'standard'],
        'suggested_action': ['action1', 'action2', 'action3', 'action4', 'action5', 'action6', 'action7', 'action8', 'action9', 'action10']
    })
    
    metrics_data = {
        "decision_table": decision_table
    }
    
    # Mock plotly figure creation to capture the figure object
    import plotly.graph_objects as go
    original_figure = go.Figure
    mock_figure = MagicMock()
    
    try:
        go.Figure = MagicMock(return_value=mock_figure)
        mock_figure.add_trace = MagicMock()
        mock_figure.update_layout = MagicMock()
        mock_figure.add_shape = MagicMock()
        mock_figure.add_annotation = MagicMock()
        
        # Mock streamlit's plotly_chart function
        with patch("streamlit.plotly_chart") as mock_plotly_chart, \
             patch("streamlit.subheader", MagicMock()), \
             patch("streamlit.write", MagicMock()):
            
            render_fb_vs_wsjf_chart(metrics_data)
            
            # Assert that plotly_chart was called
            mock_plotly_chart.assert_called_once()
            
            # Check that add_shape was called multiple times for quadrant shading
            assert mock_figure.add_shape.call_count >= 4, "Should add shapes for quadrant shading"
            
            # Check that add_annotation was called for quadrant labels
            assert mock_figure.add_annotation.call_count >= 4, "Should add annotations for quadrant labels"
    finally:
        go.Figure = original_figure

def test_render_recurring_patterns_function_exists():
    """Test that the render_recurring_patterns function exists."""
    from ui.dashboard import render_recurring_patterns
    assert callable(render_recurring_patterns), "render_recurring_patterns should be a callable function"

def test_render_recurring_patterns_handles_missing_data():
    """Test that render_recurring_patterns handles missing data gracefully."""
    from ui.dashboard import render_recurring_patterns
    
    # Create a mock metrics data dictionary without recurring patterns
    metrics_data = {
        "network_metrics": {}
    }
    
    # Mock streamlit's functions
    with patch("streamlit.subheader", MagicMock()) as mock_subheader, \
         patch("streamlit.info", MagicMock()) as mock_info:
        
        render_recurring_patterns(metrics_data)
        
        # Assert that info was called to handle missing data
        mock_info.assert_called_once()

def test_render_percolation_collapse_function_exists():
    """Test that the render_percolation_collapse function exists."""
    from ui.dashboard import render_percolation_collapse
    assert callable(render_percolation_collapse), "render_percolation_collapse should be a callable function"

def test_render_glossary_sidebar_function_exists():
    """Test that the render_glossary_sidebar function exists."""
    from ui.dashboard import render_glossary_sidebar
    assert callable(render_glossary_sidebar), "render_glossary_sidebar should be a callable function"

def test_render_developer_controls_function_exists():
    """Test that the render_developer_controls function exists."""
    from ui.dashboard import render_developer_controls
    assert callable(render_developer_controls), "render_developer_controls should be a callable function"

def test_render_advanced_metrics_tab_function_exists():
    """Test that the render_advanced_metrics_tab function exists."""
    from ui.dashboard import render_advanced_metrics_tab
    assert callable(render_advanced_metrics_tab), "render_advanced_metrics_tab should be a callable function"

def test_advanced_metrics_tab_included_in_main():
    """Test that the Advanced Metrics tab is included in the main dashboard."""
    with patch("streamlit.tabs") as mock_tabs:
        from ui.dashboard import main
        
        # Create mock return values for tabs
        mock_tabs.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        
        # Mock various other functions to avoid actual execution
        with patch("ui.dashboard.configure_dark_theme"), \
             patch("ui.dashboard.discover_datasets", return_value=(["test_dataset"], "test_dataset")), \
             patch("ui.dashboard.is_valid_dataset", return_value=True), \
             patch("ui.dashboard.load_friction_data", return_value=(pd.DataFrame(), pd.DataFrame(), {}, nx.DiGraph())), \
             patch("ui.dashboard.load_advanced_metrics", return_value={}), \
             patch("ui.dashboard.render_friction_table"), \
             patch("ui.dashboard.render_flow_summaries"), \
             patch("ui.dashboard.render_graph_heatmap"), \
             patch("ui.dashboard.render_advanced_metrics_tab"), \
             patch("ui.dashboard.render_glossary_sidebar"), \
             patch("streamlit.sidebar.selectbox", return_value="test_dataset"), \
             patch("streamlit.sidebar.title"), \
             patch("streamlit.sidebar.markdown"), \
             patch("streamlit.sidebar.info"), \
             patch("pathlib.Path.exists", return_value=True):
            
            # Call the main function
            main()
            
            # Assert that the tabs were created with the expected names
            mock_tabs.assert_called_once()
            args, _ = mock_tabs.call_args
            assert len(args[0]) >= 4, "There should be at least 4 tabs"
            tab_names = args[0]
            assert "Friction Analysis" in tab_names, "Friction Analysis tab should exist"
            assert "Flow Analysis" in tab_names, "Flow Analysis tab should exist"
            assert "User Journey Graph" in tab_names, "User Journey Graph tab should exist"
            assert "Advanced Metrics" in tab_names, "Advanced Metrics tab should exist" 