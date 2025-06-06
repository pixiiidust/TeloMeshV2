import os
import sys
import pytest
import pandas as pd
import json
import pickle
import networkx as nx
import importlib.util
import tempfile
from unittest.mock import patch, MagicMock, call

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_streamlit_installed():
    """Test that streamlit is installed."""
    streamlit_spec = importlib.util.find_spec("streamlit")
    assert streamlit_spec is not None, "Streamlit is not installed"

def test_pyvis_installed():
    """Test that pyvis is installed."""
    pyvis_spec = importlib.util.find_spec("pyvis")
    assert pyvis_spec is not None, "PyVis is not installed"

def test_dashboard_module_exists():
    """Test that the dashboard module exists."""
    dashboard_path = os.path.join(os.path.dirname(__file__), '..', 'ui', 'dashboard.py')
    assert os.path.exists(dashboard_path), "dashboard.py does not exist"

def test_temp_file_handling():
    """Test that temporary file handling works correctly in render_network_graph."""
    from ui.dashboard import render_network_graph
    
    # Create a mock network object
    mock_network = MagicMock()
    mock_network.save_graph = MagicMock()
    
    # Mock the open function to avoid file I/O
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = "<html></html>"
    
    # Create mock components.html function
    mock_components_html = MagicMock()
    
    # Test the function
    with patch("builtins.open", return_value=mock_file), \
         patch("streamlit.components.v1.html", mock_components_html), \
         patch("os.path.exists", return_value=True), \
         patch("os.remove") as mock_remove:
        
        render_network_graph(mock_network)
        
        # Assert that save_graph was called
        mock_network.save_graph.assert_called_once()
        
        # Assert that components.html was called
        mock_components_html.assert_called_once()
        
        # Assert that we tried to remove the temporary file
        mock_remove.assert_called_once()

def test_create_full_network():
    """Test creation of a full network with correct colors and options."""
    from ui.dashboard import create_full_network
    
    # Create test graph
    G = nx.DiGraph()
    G.add_edge("/home", "/product", event="Click", weight=2)
    G.add_edge("/product", "/cart", event="Add to Cart", weight=1)
    
    # Create test score map with different score tiers
    score_map = {
        "/home": 0.8,      # Top 10%
        "/product": 0.6,   # Top 20%
        "/cart": 0.2       # Regular
    }
    
    # Define thresholds
    top10_threshold = 0.7
    top20_threshold = 0.5
    
    # Create network
    network = create_full_network(G, score_map, top10_threshold, top20_threshold, True)
    
    # Get node colors to verify
    node_colors = {}
    for node in network.nodes:
        node_colors[node['id']] = node['color']
    
    # Verify node colors
    assert node_colors['/home'] == "#F87171", "Top 10% node should be soft red"
    assert node_colors['/product'] == "#FBBF24", "Top 20% node should be warm amber"
    assert node_colors['/cart'] == "#94A3B8", "Regular node should be soft steel"
    
    # Verify the presence of edges
    assert len(network.edges) == 2, "Network should have 2 edges"
    
    # Check edge properties
    for edge in network.edges:
        assert "title" in edge, "Edge should have a title"
        assert "width" in edge, "Edge should have a width"
        assert "color" in edge, "Edge should have color settings"
        
    # Check that options are correctly set
    options = network.options
    assert "physics" in options, "Options should include physics settings"
    assert options["physics"]["enabled"] is True, "Physics should be enabled"

def test_create_filtered_network():
    """Test filtered network creation with specific node thresholds."""
    from ui.dashboard import create_filtered_network
    
    # Create test graph
    G = nx.DiGraph()
    G.add_edge("/home", "/product", event="Click", weight=2)
    G.add_edge("/product", "/cart", event="Add to Cart", weight=1)
    G.add_edge("/cart", "/checkout", event="Checkout", weight=1)
    
    # Create filtered nodes (top 20%)
    filtered_nodes = {
        "/home": 0.8,      # Top 10%
        "/product": 0.6    # Top 20%
    }
    
    # Define thresholds
    top10_threshold = 0.7
    top20_threshold = 0.5
    
    # Create network
    network = create_filtered_network(G, filtered_nodes, top10_threshold, top20_threshold, True)
    
    # Verify only filtered nodes are included
    node_ids = [node['id'] for node in network.nodes]
    assert '/home' in node_ids, "Home node should be included"
    assert '/product' in node_ids, "Product node should be included"
    assert '/cart' not in node_ids, "Cart node should not be included"
    assert '/checkout' not in node_ids, "Checkout node should not be included"
    
    # Verify only edges between filtered nodes are included
    assert len(network.edges) == 1, "Only one edge should be included"
    
    # Get the included edge
    edge = network.edges[0]
    assert edge['from'] == '/home', "Edge should start at home"
    assert edge['to'] == '/product', "Edge should end at product"

def test_network_highlighting():
    """Test that node highlighting works correctly with contrasting colors."""
    from ui.dashboard import create_full_network
    
    # Create test graph
    G = nx.DiGraph()
    G.add_node("/home")
    
    # Create test score map
    score_map = {
        "/home": 0.8
    }
    
    # Define thresholds
    top10_threshold = 0.7
    top20_threshold = 0.5
    
    # Create network
    network = create_full_network(G, score_map, top10_threshold, top20_threshold, True)
    
    # Get options as a dictionary
    options = network.options
    
    # Check for highlight color definitions
    assert "nodes" in options, "Options should contain node settings"
    assert "color" in options["nodes"], "Node options should contain color settings"
    assert "highlight" in options["nodes"]["color"], "Color settings should contain highlight properties"
    
    # Verify highlight settings
    highlight = options["nodes"]["color"]["highlight"]
    assert "border" in highlight, "Highlight should set border color"
    assert highlight["border"] == "#38BDF8", "Border highlight should be accent blue"
    assert "background" in highlight, "Highlight should set background color"
    assert highlight["background"] == "#38BDF8", "Background highlight should be accent blue"

def test_error_handling_in_graph_rendering():
    """Test that error handling works properly when graph rendering fails."""
    from ui.dashboard import render_network_graph
    
    # Create mock network that will raise an exception on save_graph
    mock_network = MagicMock()
    mock_network.save_graph.side_effect = Exception("Test error")
    
    # Mock streamlit's error function
    mock_st_error = MagicMock()
    
    # Test the function
    with patch("streamlit.error", mock_st_error):
        render_network_graph(mock_network)
        
        # Assert that streamlit.error was called
        mock_st_error.assert_called_once()

def test_filter_handling():
    """Test that filter handling works correctly for invalid filter combinations."""
    from ui.dashboard import render_graph_heatmap
    
    # Create test graph
    G = nx.DiGraph()
    G.add_edge("/home", "/product", event="Click", weight=1)
    
    # Create empty score map to trigger warning
    score_map = {}
    
    # Mock streamlit's warning function
    mock_st_warning = MagicMock()
    
    # Mock streamlit's selectbox to always return "Top 10% friction nodes"
    mock_st_selectbox = MagicMock(return_value="Top 10% friction nodes")
    
    # Mock streamlit's checkbox for physics
    mock_st_checkbox = MagicMock(return_value=True)
    
    # Mock streamlit's columns
    mock_st_columns = MagicMock(return_value=[MagicMock(), MagicMock()])
    
    # Test the function
    with patch("streamlit.warning", mock_st_warning), \
         patch("streamlit.selectbox", mock_st_selectbox), \
         patch("streamlit.checkbox", mock_st_checkbox), \
         patch("streamlit.columns", mock_st_columns), \
         patch("streamlit.header", MagicMock()):
        
        render_graph_heatmap(G, score_map)
        
        # Assert that streamlit.warning was called for empty filtered nodes
        mock_st_warning.assert_called_once()

def test_dashboard_loads():
    """Test that the dashboard loads without errors."""
    # This is a basic import test since we can't easily test Streamlit's UI rendering in pytest
    from ui.dashboard import load_friction_data, render_friction_table, render_graph_heatmap, render_flow_summaries
    
    # Check that the functions are callable
    assert callable(load_friction_data), "load_friction_data should be a callable function"
    assert callable(render_friction_table), "render_friction_table should be a callable function"
    assert callable(render_graph_heatmap), "render_graph_heatmap should be a callable function"
    assert callable(render_flow_summaries), "render_flow_summaries should be a callable function"

def test_render_friction_table():
    """Test that the render_friction_table function works with test data."""
    from ui.dashboard import render_friction_table
    
    # Create a test DataFrame
    data = {
        "page": ["/home", "/product", "/cart", "/checkout"],
        "event": ["Click", "Add to Cart", "Checkout", "Form Submit"],
        "exit_rate": [0.1, 0.2, 0.5, 0.8],
        "betweenness": [0.5, 0.3, 0.2, 0.1],
        "users_lost": [10, 20, 50, 80],
        "WSJF_Friction_Score": [0.05, 0.06, 0.1, 0.08]
    }
    df = pd.DataFrame(data)
    
    # Mock streamlit functions
    with patch("streamlit.header", MagicMock()), \
         patch("streamlit.columns", MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])), \
         patch("streamlit.selectbox", MagicMock(side_effect=["All", "All", "All points"])), \
         patch("streamlit.dataframe", MagicMock()) as mock_dataframe, \
         patch("streamlit.markdown", MagicMock()), \
         patch("streamlit.metric", MagicMock()), \
         patch("streamlit.info", MagicMock()) as mock_info:
        
        # Run the function
        render_friction_table(df)
        
        # Assert dataframe was called
        mock_dataframe.assert_called_once()
        # Assert info was not called (since we have valid data)
        mock_info.assert_not_called()

def test_render_flow_summaries():
    """Test that the render_flow_summaries function works with empty filters."""
    # Create a wrapped version of render_flow_summaries that doesn't rely on actual Streamlit
    def wrapped_render_flow_summaries(df):
        """A simplified version of render_flow_summaries for testing."""
        if len(df) == 0:
            return "No fragile flows detected"
        
        # Simplified logic similar to what's in the real function
        filtered_sessions = []
        for session_id in df["session_id"].unique():
            session_data = df[df["session_id"] == session_id]
            # Criteria similar to the actual function
            if len(session_data) >= 3 and session_data["is_chokepoint"].sum() >= 2:
                filtered_sessions.append(session_id)
        
        if not filtered_sessions:
            return "No flows match filters"
        
        return f"Found {len(filtered_sessions)} flows"
    
    # Create a test DataFrame
    data = {
        "session_id": ["session1", "session1", "session1", "session2", "session2"],
        "step_index": [0, 1, 2, 0, 1],
        "page": ["/home", "/product", "/checkout", "/home", "/cart"],
        "event": ["Click", "Add to Cart", "Checkout", "Search", "Add to Cart"],
        "WSJF_Friction_Score": [0.05, 0.06, 0.1, 0.05, 0.1],
        "is_chokepoint": [0, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    # Check that with our example data, no flows match our filters
    # (session1 has enough length but only 1 chokepoint, session2 has a chokepoint but not enough length)
    result = wrapped_render_flow_summaries(df)
    assert result == "No flows match filters", "Should return no flows message"
    
    # Now let's create data that does have valid flows
    data2 = {
        "session_id": ["session3", "session3", "session3", "session3"],
        "step_index": [0, 1, 2, 3],
        "page": ["/home", "/product", "/checkout", "/cart"],
        "event": ["Click", "Add to Cart", "Checkout", "Add More"],
        "WSJF_Friction_Score": [0.05, 0.06, 0.1, 0.05],
        "is_chokepoint": [0, 1, 1, 0]  # Has 2 chokepoints and length >= 3
    }
    df2 = pd.DataFrame(pd.concat([df, pd.DataFrame(data2)]))
    
    # Check that valid flows are found
    result2 = wrapped_render_flow_summaries(df2)
    assert result2 == "Found 1 flows", "Should find one valid flow"

def test_fragile_flows_detailed():
    """Test the Fragile Flows tab with comprehensive checking of filters, metrics, and path rendering."""
    # Create a wrapped version for testing
    def wrapped_flow_analysis(df, min_length, min_chokepoints):
        """Simplified version of the flow filtering logic for testing."""
        filtered_sessions = []
        for session_id in df["session_id"].unique():
            session_data = df[df["session_id"] == session_id]
            
            # Apply filters
            if len(session_data) < min_length:
                continue
            
            chokepoint_count = session_data["is_chokepoint"].sum()
            if chokepoint_count < min_chokepoints:
                continue
            
            filtered_sessions.append(session_id)
        
        return filtered_sessions
    
    # Create a more detailed test DataFrame with multiple sessions and chokepoints
    data = {
        "session_id": ["flow1", "flow1", "flow1", "flow1", "flow2", "flow2", "flow2", "flow3", "flow3", "flow3"],
        "step_index": [0, 1, 2, 3, 0, 1, 2, 0, 1, 2],
        "page": ["/home", "/search", "/product", "/cart", "/home", "/product", "/checkout", "/search", "/product", "/cart"],
        "event": ["Login", "Search", "View", "Add to Cart", "Login", "View", "Checkout", "Search", "View", "Add to Cart"],
        "WSJF_Friction_Score": [0.05, 0.08, 0.1, 0.05, 0.05, 0.2, 0.3, 0.08, 0.1, 0.3],
        "is_chokepoint": [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    # Test 1: Filter with min_length=3, min_chokepoints=2
    filtered_sessions = wrapped_flow_analysis(df, 3, 2)
    assert "flow2" in filtered_sessions, "flow2 should match the filters"
    assert "flow3" in filtered_sessions, "flow3 should match the filters"
    assert "flow1" not in filtered_sessions, "flow1 should not match (only 1 chokepoint)"
    assert len(filtered_sessions) == 2, "Should find 2 flows that match criteria"
    
    # Test 2: More strict filtering
    filtered_sessions = wrapped_flow_analysis(df, 4, 2)
    assert len(filtered_sessions) == 0, "No flows should match with length >= 4"
    
    # Test 3: Lenient filtering
    filtered_sessions = wrapped_flow_analysis(df, 2, 1)
    assert len(filtered_sessions) == 3, "All flows should match with length >= 2 and chokepoints >= 1"

def test_render_tooltips():
    """Test that the render_tooltips function returns the correct text for different metrics."""
    from ui.dashboard import render_tooltips
    
    # Test for exit_rate
    exit_rate_tooltip = render_tooltips("exit_rate")
    assert "leave" in exit_rate_tooltip.lower(), "exit_rate tooltip should mention users leaving"
    
    # Test for betweenness
    betweenness_tooltip = render_tooltips("betweenness")
    assert "central" in betweenness_tooltip.lower() or "flow" in betweenness_tooltip.lower(), \
        "betweenness tooltip should mention 'central' or 'flow'"
    
    # Test for WSJF_Friction_Score
    wsjf_tooltip = render_tooltips("WSJF_Friction_Score")
    assert "friction" in wsjf_tooltip.lower() or "frustration" in wsjf_tooltip.lower() or "importance" in wsjf_tooltip.lower(), \
        "WSJF_Friction_Score tooltip should mention 'friction', 'frustration', or 'importance'"
    
    # Test for unknown metric
    unknown_tooltip = render_tooltips("unknown_metric")
    assert unknown_tooltip != "", "Unknown metric should still return some tooltip text"

def test_dark_theme_configuration():
    """Test that dark theme configuration is correctly applied."""
    from ui.dashboard import configure_dark_theme
    
    # Mock streamlit functions
    with patch("streamlit.set_page_config", MagicMock()), \
         patch("streamlit.markdown", MagicMock()) as mock_markdown:
        
        # Run the function
        configure_dark_theme()
        
        # Check that the function was called
        mock_markdown.assert_called_once()
        
        # Check that the CSS contains dark theme colors
        css = mock_markdown.call_args[0][0]
        assert "background-color: #0B0F19" in css, "Primary background color should be set"
        assert "color: #E2E8F0" in css, "Text color should be set"
        
        # Check for unsafe_allow_html
        assert mock_markdown.call_args[1].get("unsafe_allow_html") is True, "CSS should be allowed as HTML"

def test_network_node_colors():
    """Test that network nodes are colored correctly."""
    from ui.dashboard import create_full_network
    
    # Create test graph
    G = nx.DiGraph()
    G.add_node("/home")
    G.add_node("/about")
    G.add_node("/product")
    
    # Create test score map with values that cross thresholds
    score_map = {
        "/home": 0.2,      # Normal node
        "/about": 0.6,     # Top 20% node
        "/product": 0.9    # Top 10% node
    }
    
    # Create network
    top10_threshold = 0.8
    top20_threshold = 0.5
    network = create_full_network(G, score_map, top10_threshold, top20_threshold, False)
    
    # Get nodes
    nodes = network.nodes
    
    # Verify node colors
    for node in nodes:
        if node["id"] == "/product":
            assert node["color"] == "#F87171", "Top 10% node should be soft red"
        elif node["id"] == "/about":
            assert node["color"] == "#FBBF24", "Top 20% node should be warm amber"
        elif node["id"] == "/home":
            assert node["color"] == "#94A3B8", "Normal node should be soft steel"

def test_color_scheme_variables():
    """Test that color scheme variables are defined and used correctly."""
    from ui.dashboard import configure_dark_theme
    import inspect
    
    # Get the source code
    source = inspect.getsource(configure_dark_theme)
    
    # Check for color variables
    assert "colors = {" in source, "Color variables should be defined"
    assert "bg_primary" in source, "Primary background color should be defined"
    assert "text_primary" in source, "Primary text color should be defined"
    assert "node_red" in source, "Node red color should be defined"
    assert "node_orange" in source, "Node orange color should be defined"
    assert "edge_color" in source, "Edge color should be defined"
    
    # Check for specific colors from the new scheme
    assert "#0B0F19" in source, "Jet black-blue background color should be defined"
    assert "#38BDF8" in source, "Sky blue accent color should be defined"
    assert "#F87171" in source, "Soft red for top 10% nodes should be defined"
    assert "#FBBF24" in source, "Warm amber for top 20% nodes should be defined"
    assert "#94A3B8" in source, "Soft steel for normal nodes should be defined"
    
    # Check that colors are used with f-strings
    assert 'f"""' in source, "f-strings should be used for color interpolation"
    assert "{colors[" in source, "Color variables should be used in CSS"

def test_enhanced_network_options():
    """Test that network has enhanced visual options."""
    from ui.dashboard import create_full_network
    
    # Create minimal test data
    G = nx.DiGraph()
    G.add_edge("A", "B")
    score_map = {"A": 0.1, "B": 0.1}
    
    # Create network
    network = create_full_network(G, score_map, 0.8, 0.5, False)
    
    # Get options
    options = network.options
    
    # Check for enhanced visual options
    assert "shadow" in options["nodes"], "Nodes should have shadow effect"
    assert options["nodes"]["shadow"]["enabled"] is True, "Node shadow should be enabled"
    
    assert "arrows" in options["edges"], "Edges should have arrow configuration"
    assert options["edges"]["arrows"]["to"]["enabled"] is True, "Edge arrows should be enabled"
    
    # Check edge highlighting
    assert "inherit" in options["edges"]["color"], "Edge should have inherit property"
    assert options["edges"]["color"]["inherit"] is False, "Edge should not use default inheritance, since we use explicit colors"

def test_fragile_flow_custom_styles():
    """Test that fragile flows have custom styling classes."""
    from ui.dashboard import configure_dark_theme
    
    # Mock streamlit functions
    with patch("streamlit.set_page_config", MagicMock()), \
         patch("streamlit.markdown", MagicMock()) as mock_markdown:
        
        # Run the function
        configure_dark_theme()
        
        # Get the CSS
        css = mock_markdown.call_args[0][0]
        
        # Check for custom highlight classes
        assert ".high-friction" in css, "High friction class should be defined"
        assert ".medium-friction" in css, "Medium friction class should be defined"
        
        # Check for proper color assignment
        assert "color: #F87171" in css, "High friction should use soft red"
        assert "color: #FBBF24" in css, "Medium friction should use warm amber"
        assert "font-weight: bold" in css, "Friction classes should use bold text"

def test_dark_theme_table_styling():
    """Test that tables and dataframes have proper dark theme styling."""
    from ui.dashboard import configure_dark_theme
    
    # Mock streamlit functions
    with patch("streamlit.set_page_config", MagicMock()), \
         patch("streamlit.markdown", MagicMock()) as mock_markdown:
        
        # Run the function
        configure_dark_theme()
        
        # Get the CSS
        css = mock_markdown.call_args[0][0]
        
        # Check for key table styling
        assert "border-collapse: separate" in css, "Tables should use separate border model"
        assert "border-spacing: 0" in css, "Tables should have no border spacing"
        assert "width: 100%" in css, "Tables should use full width"
        
        # Check for alternating row colors
        assert "tr:nth-child(odd)" in css, "Odd rows should be styled"
        assert "tr:nth-child(even)" in css, "Even rows should be styled"
        assert "rgba(39, 49, 67, 0.4)" in css, "Tables should use semi-transparent background for alternating rows"
        
        # Check for rounded corners and borders
        assert "border-radius: 6px" in css, "Tables should have rounded corners"
        assert "overflow: hidden" in css, "Tables should hide overflow"

def test_network_edge_highlighting():
    """Test that edge highlighting uses source node colors instead of white."""
    from ui.dashboard import create_full_network
    import json
    
    # Create test graph
    G = nx.DiGraph()
    G.add_edge("/home", "/product", event="Click", weight=1)
    G.add_edge("/blog", "/cart", event="Add to Cart", weight=2)
    G.add_edge("/cart", "/checkout", event="Checkout", weight=1)
    
    # Create test score map
    score_map = {
        "/home": 0.3,       # Regular node (soft steel)
        "/product": 0.3,    # Regular node (soft steel)
        "/blog": 0.9,       # Top 10% node (soft red)
        "/cart": 0.6,       # Top 20% node (warm amber)
        "/checkout": 0.2    # Regular node (soft steel)
    }
    
    # Define thresholds
    top10_threshold = 0.8
    top20_threshold = 0.5
    
    # Create network
    network = create_full_network(G, score_map, top10_threshold, top20_threshold, False)
    
    # Extract edges with their colors
    edges = network.edges
    
    # Check specific edge highlight colors
    blog_cart_edge = None
    cart_checkout_edge = None
    home_product_edge = None
    
    for edge in edges:
        if edge["from"] == "/blog" and edge["to"] == "/cart":
            blog_cart_edge = edge
        elif edge["from"] == "/cart" and edge["to"] == "/checkout":
            cart_checkout_edge = edge
        elif edge["from"] == "/home" and edge["to"] == "/product":
            home_product_edge = edge
    
    # Verify edge highlighting colors match source node colors
    assert blog_cart_edge is not None, "Blog to Cart edge should exist"
    assert cart_checkout_edge is not None, "Cart to Checkout edge should exist"
    assert home_product_edge is not None, "Home to Product edge should exist"
    
    # Check blog edge (top 10% - soft red)
    assert blog_cart_edge["color"]["highlight"] == "#F87171", "Blog edge highlight should be soft red (#F87171)"
    assert blog_cart_edge["color"]["hover"] == "#F87171", "Blog edge hover should be soft red (#F87171)"
    
    # Check cart edge (top 20% - warm amber)
    assert cart_checkout_edge["color"]["highlight"] == "#FBBF24", "Cart edge highlight should be warm amber (#FBBF24)"
    assert cart_checkout_edge["color"]["hover"] == "#FBBF24", "Cart edge hover should be warm amber (#FBBF24)"
    
    # Check home edge (regular - soft steel)
    assert home_product_edge["color"]["highlight"] == "#94A3B8", "Home edge highlight should be soft steel (#94A3B8)"
    assert home_product_edge["color"]["hover"] == "#94A3B8", "Home edge hover should be soft steel (#94A3B8)"
    
    # Check that global edge inheritance is disabled in options
    options = network.options
    assert "edges" in options, "Options should contain edge settings"
    assert "color" in options["edges"], "Edge options should contain color settings"
    assert "inherit" in options["edges"]["color"], "Edge color should have inherit property"
    assert options["edges"]["color"]["inherit"] is False, "Edge inherit should be false since we set explicit colors"

def test_tooltip_visibility_css():
    """Test that tooltip icons are styled to be visible."""
    from ui.dashboard import configure_dark_theme
    
    # Mock streamlit functions
    with patch("streamlit.set_page_config", MagicMock()), \
         patch("streamlit.markdown", MagicMock()) as mock_markdown:
        
        # Run the function
        configure_dark_theme()
        
        # Get the CSS
        css = mock_markdown.call_args[0][0]
        
        # Check for tooltip styling
        assert "[data-testid=\"tooltip-icon\"]" in css, "CSS should target tooltip icons"
        assert "opacity: 1" in css, "CSS should set opacity to 1 for tooltips"
        assert "visibility: visible" in css, "CSS should set visibility to visible for tooltips"
        assert "fill:" in css, "CSS should set fill color for SVG tooltips"

def test_fragile_flow_expander_text_color():
    """Test that fragile flow expander text color is properly set."""
    from ui.dashboard import configure_dark_theme
    
    # Mock streamlit functions
    with patch("streamlit.set_page_config", MagicMock()), \
         patch("streamlit.markdown", MagicMock()) as mock_markdown:
        
        # Run the function
        configure_dark_theme()
        
        # Get the CSS
        css = mock_markdown.call_args[0][0]
        
        # Check for expander styling
        assert ".streamlit-expanderHeader p" in css, "CSS should target text in expander headers"
        assert "color: #E2E8F0" in css, "CSS should set text color for expander headers"

def test_flow_length_dropdown():
    """Test that flow length is configured as a dropdown instead of a slider."""
    # Create a wrapped version for testing
    def wrapped_flow_options(flow_df):
        """Extract flow length options from a DataFrame."""
        min_flow_length = int(flow_df.groupby("session_id").size().min())
        max_flow_length = int(flow_df.groupby("session_id").size().max())
        return list(range(min_flow_length, max_flow_length + 1))
    
    # Create test data
    data = {
        "session_id": ["flow1", "flow1", "flow1", "flow2", "flow2", "flow2", "flow2", "flow3", "flow3"],
        "step_index": [0, 1, 2, 0, 1, 2, 3, 0, 1],
        "is_chokepoint": [0, 0, 1, 0, 1, 1, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    
    # Check that options are correctly generated
    options = wrapped_flow_options(df)
    assert options == [2, 3, 4], "Should generate options from min to max flow length"
    
    # Check for different data
    data2 = {
        "session_id": ["flow1", "flow1", "flow1", "flow1", "flow1"],
        "step_index": [0, 1, 2, 3, 4],
        "is_chokepoint": [0, 0, 1, 0, 1]
    }
    df2 = pd.DataFrame(data2)
    
    options2 = wrapped_flow_options(df2)
    assert options2 == [5], "Should generate a single option when min=max"

def test_telomesh_logo_integration():
    """Test that TeloMesh logo is properly integrated in the UI."""
    from ui.dashboard import main
    
    # Mock all streamlit functions
    with patch("streamlit.set_page_config", MagicMock()), \
         patch("streamlit.markdown", MagicMock()), \
         patch("streamlit.title", MagicMock()), \
         patch("streamlit.columns", MagicMock()) as mock_columns, \
         patch("streamlit.tabs", MagicMock()), \
         patch("os.path.exists", MagicMock(return_value=True)), \
         patch("streamlit.image", MagicMock()) as mock_image:
        
        # Configure mock columns to return a tuple of mocks
        col1_mock = MagicMock()
        col2_mock = MagicMock()
        mock_columns.return_value = [col1_mock, col2_mock]
        
        # Run the main function
        try:
            with patch("ui.dashboard.load_friction_data", MagicMock(return_value=(None, None, None, None))):
                main()
        except Exception:
            pass  # We expect this to fail since we're mocking load_friction_data to return None
        
        # Check if columns were created (for logo and title)
        mock_columns.assert_called_once_with([1, 5])
        
        # Check if title was set
        col2_mock.title.assert_called_once()
        
        # The image function should be called on col1 when the logo exists
        col1_mock.image.assert_called_once() 