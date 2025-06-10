"""
TeloMesh User Flow Intelligence Dashboard - LOVABLE Stage

This module provides a Streamlit dashboard for visualizing friction points in user journeys.
It displays event chokepoints, fragile flows, and a graph heatmap with interactive elements.

The dashboard is intended for product managers to identify and prioritize UX improvements.
"""

import os
import streamlit as st
import pandas as pd
import networkx as nx
import pickle
import json
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
import numpy as np
from typing import Tuple, Dict, List, Optional
import time
import logging
import traceback
import base64
from pathlib import Path
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Helper function for creating downloadable CSV
def get_csv_download_link(df, filename, button_text="Export CSV"):
    """
    Generate a download link for a CSV file from a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to export as CSV
        filename (str): Filename for the exported CSV
        button_text (str): Text to display on the download button
    
    Returns:
        str: HTML for the download link
    """
    try:
        # Add timestamp to filename to avoid duplicates
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}.csv"
        
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False)
        
        # Create download link
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">{button_text}</a>'
        
        return href
    except Exception as e:
        logger.error(f"Error creating CSV download link: {str(e)}")
        return f"<p>Error creating download link: {str(e)}</p>"

def get_html_download_link(html_content, filename, button_text="Export HTML"):
    """
    Generate a download link for an HTML file.
    
    Args:
        html_content (str): HTML content to export
        filename (str): Filename for the exported HTML
        button_text (str): Text to display on the download button
    
    Returns:
        str: HTML for the download link
    """
    try:
        # Add timestamp to filename to avoid duplicates
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}.html"
        
        # Create download link
        b64 = base64.b64encode(html_content.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="{filename}" class="download-button">{button_text}</a>'
        
        return href
    except Exception as e:
        logger.error(f"Error creating HTML download link: {str(e)}")
        return f"<p>Error creating download link: {str(e)}</p>"

# Tooltip Fix Tests (Manual Acceptance Criteria)
# 1. Hover over a metric label in the friction table (e.g., "Exit Rate")
# 2. âœ… Tooltip icon (â„¹ï¸) should be visible (white on dark)
# 3. âœ… Tooltip bubble should show with dark background and white text
# 4. âœ… No white-on-white or missing content
# 5. âœ… Tooltip border should be visible and styled (optional accent)
# 6. Regression: tooltips should still appear when hovering over table headers or metrics

def configure_dark_theme():
    """Configure dark theme using native Streamlit + minimal CSS."""
    # Set page config (NO theme parameter - use config.toml instead)
    logo_path = "logos/telomesh logo.png"
    if os.path.exists(logo_path):
        st.set_page_config(
            page_title="TeloMesh User Flow Intelligence",
            page_icon=logo_path,
            layout="wide",
            initial_sidebar_state="expanded"
        )
    else:
        st.set_page_config(
            page_title="TeloMesh User Flow Intelligence",
            page_icon="ðŸ”",
            layout="wide", 
            initial_sidebar_state="expanded"
        )
    
    # MINIMAL CSS - only fix what config.toml doesn't handle
    st.markdown("""
    <style>
        /* Fix tooltips (main issue) */
        [data-testid*="stTooltip"],
        [data-testid*="tooltip"],
        [role="tooltip"],
        div[aria-label*="tooltip"] {
            background-color: #1A2332 !important;
            color: #E2E8F0 !important;
            border: 1px solid #38BDF8 !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
            z-index: 9999 !important;
        }
        
        /* Remove box around tooltip icon */
        [data-testid="stTooltipIcon"] {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        
        /* Fix tooltip text */
        [data-testid*="stTooltip"] *,
        [data-testid*="tooltip"] *,
        [role="tooltip"] * {
            color: #E2E8F0 !important;
            background-color: transparent !important;
        }
        
        /* Ensure text visibility in main content */
        .main .block-container {
            color: #E2E8F0 !important;
        }
        
        .main .block-container p,
        .main .block-container div,
        .main .block-container span,
        .main .block-container li {
            color: inherit !important;
        }
        
        /* Tab text visibility fix */
        .stTabs [data-baseweb="tab"] {
            color: #E2E8F0 !important;
        }
        
        .stTabs [aria-selected="true"] {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        
        /* Custom classes for your app */
        .high-friction {
            color: #F87171 !important;
            font-weight: bold !important;
        }
        
        .medium-friction {
            color: #FBBF24 !important;
            font-weight: bold !important;
        }
        
        .telomesh-header {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            margin-bottom: 1.5rem !important;
            text-align: center !important;
        }
        
        .telomesh-logo {
            height: 8rem !important;
            margin-bottom: 1rem !important;
        }
        
        /* Download button styling */
        .download-button {
            display: inline-block;
            padding: 8px 16px;
            background-color: #38BDF8;
            color: #0B0F19 !important;
            font-weight: 600;
            text-decoration: none;
            border-radius: 4px;
            margin: 10px 0;
            border: none;
            cursor: pointer;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .download-button:hover {
            background-color: #0EA5E9;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .download-container {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 16px;
        }
    </style>
    """, unsafe_allow_html=True)

def load_logo_base64(path: str) -> str:
    """Load image file and return as base64 encoded string."""
    try:
        return base64.b64encode(Path(path).read_bytes()).decode()
    except Exception as e:
        logger.error(f"Error loading logo from {path}: {str(e)}")
        return ""

def discover_datasets():
    """
    Discover available datasets in the outputs directory.
    
    Returns:
        list: List of dataset names (directory names)
        str: Most recent dataset name or None if no datasets found
    """
    try:
        outputs_dir = Path("outputs")
        if not outputs_dir.exists():
            return [], None
        
        # Get all subdirectories in the outputs directory
        datasets = [d.name for d in outputs_dir.iterdir() if d.is_dir()]
        
        # If no datasets found, return empty list
        if not datasets:
            return [], None
        
        # Find the most recent dataset based on modification time
        dataset_times = [(d, os.path.getmtime(outputs_dir / d)) for d in datasets]
        dataset_times.sort(key=lambda x: x[1], reverse=True)
        
        most_recent = dataset_times[0][0] if dataset_times else None
        
        return datasets, most_recent
    except Exception as e:
        logger.error(f"Error discovering datasets: {str(e)}")
        return [], None

def is_valid_dataset(dataset_name):
    """
    Check if a dataset directory contains all the required files.
    
    Args:
        dataset_name (str): Name of the dataset directory
        
    Returns:
        bool: True if the dataset is valid, False otherwise
    """
    required_files = [
        "event_chokepoints.csv",
        "high_friction_flows.csv",
        "friction_node_map.json",
        "user_graph.gpickle"
    ]
    
    dataset_dir = Path(f"outputs/{dataset_name}")
    
    # Check if all required files exist
    for file in required_files:
        if not (dataset_dir / file).exists():
            return False
    
    return True

def load_friction_data(dataset=None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, nx.DiGraph]:
    """
    Load friction data from files.
    
    Args:
        dataset (str, optional): Name of the dataset to load. If None, uses the default dataset.
        
    Returns:
        Tuple containing:
        - DataFrame with event chokepoints
        - DataFrame with high friction flows
        - Dictionary mapping page names to friction scores
        - NetworkX DiGraph representing the user journey graph
    """
    try:
        # Determine dataset directory
        dataset_dir = Path(f"outputs/{dataset if dataset else 'default'}")
        
        # Check if dataset directory exists
        if not dataset_dir.exists() and dataset:
            st.error(f"Dataset directory not found: {dataset_dir}")
            st.info("Please select a valid dataset from the dropdown.")
            return None, None, None, None
        
        # Check if required files exist
        chokepoints_path = dataset_dir / "event_chokepoints.csv"
        flows_path = dataset_dir / "high_friction_flows.csv"
        node_map_path = dataset_dir / "friction_node_map.json"
        graph_path = dataset_dir / "user_graph.gpickle"
        
        if not chokepoints_path.exists():
            st.error(f"File not found: {chokepoints_path}")
            return None, None, None, None
            
        if not flows_path.exists():
            st.error(f"File not found: {flows_path}")
            return None, None, None, None
            
        if not node_map_path.exists():
            st.error(f"File not found: {node_map_path}")
            return None, None, None, None
            
        if not graph_path.exists():
            st.error(f"File not found: {graph_path}")
            return None, None, None, None
        
        # Load data
        friction_df = pd.read_csv(chokepoints_path)
        flow_df = pd.read_csv(flows_path)
        
        with open(node_map_path, 'r') as f:
            node_map = json.load(f)
        
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        
        return friction_df, flow_df, node_map, G
    
    except Exception as e:
        st.error(f"Error loading friction data: {str(e)}")
        st.error(traceback.format_exc())
        return None, None, None, None

def render_tooltips(metric: str) -> str:
    """
    Render tooltips for metrics in PM-friendly terms.
    
    Args:
        metric (str): The metric to render a tooltip for.
        
    Returns:
        str: The tooltip text.
    """
    tooltips = {
        "exit_rate": "% of users who leave after this event. Higher values indicate user abandonment points.",
        "betweenness": "How central this page is in user flows. Higher values indicate pages that many users pass through.",
        "users_lost": "Number of users who exited after this event and didn't return within the same session.",
        "WSJF_Friction_Score": "WSJF Friction Score is calculated as exit_rate Ã— betweenness, prioritizing high-impact friction points.",
        "wsjf": "WSJF Friction Score is calculated as exit_rate Ã— betweenness, prioritizing high-impact friction points.",
        "is_chokepoint": "Identifies high-friction points (top 10% by WSJF score)."
    }
    
    return tooltips.get(metric, "Hover for more information about this metric.")

def render_friction_table(df: pd.DataFrame):
    """
    Render a sortable, filterable table of friction points.
    
    Args:
        df (pd.DataFrame): DataFrame with event chokepoints.
    """
    # Top Pages Summary Section
    st.subheader("Top 3 Pages by Friction Metric")
    
    # Dropdown for metric selection
    metric_options = {
        "Users Lost": "users_lost",
        "Exit Rate": "exit_rate", 
        "WSJF Score": "WSJF_Friction_Score"
    }
    selected_metric = st.selectbox("Select metric:", list(metric_options.keys()))
    metric_column = metric_options[selected_metric]
    
    # Group by page and calculate aggregate metrics
    page_metrics = df.groupby("page").agg({
        "users_lost": "sum",
        "exit_rate": "mean",
        "WSJF_Friction_Score": "mean"
    }).reset_index()
    
    # Get top 3 pages by selected metric
    top_pages = page_metrics.sort_values(by=metric_column, ascending=False).head(3)
    
    # Display top pages in columns
    cols = st.columns(3)
    for i, (_, row) in enumerate(top_pages.iterrows()):
        page_name = row["page"]
        metric_value = row[metric_column]
        
        # Format the metric value based on the type
        if metric_column == "exit_rate":
            formatted_value = f"{metric_value:.2%}"
        elif metric_column == "WSJF_Friction_Score":
            formatted_value = f"{metric_value:.6f}"
        else:
            formatted_value = f"{int(metric_value):,}"
            
        cols[i].metric(
            f"#{i+1}: {page_name}", 
            formatted_value,
            help=f"Page with #{i+1} highest {selected_metric}"
        )
    
    # Main friction table
    st.header("ðŸ”¥ Friction Table")
    st.write("""
    * Event chokepoints ranked by friction metrics. 
    * Filter by page, event, and percentile. 
    * Mouse over column headers for metric definitions.
    """)
    try:
        # UI controls for filtering
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Create a list of unique pages, sorted alphabetically
            pages = ["All"] + sorted(df["page"].unique().tolist())
            selected_page = st.selectbox("Filter by page:", pages)
        
        with col2:
            # Create a list of unique events, sorted alphabetically
            events = ["All"] + sorted(df["event"].unique().tolist())
            selected_event = st.selectbox("Filter by event:", events)
        
        with col3:
            # Filter by top friction percentile
            percentile_options = {
                "All points": 0,
                "Top 25%": 75,
                "Top 10%": 90,
                "Top 5%": 95
            }
            selected_percentile_label = st.selectbox("Show:", list(percentile_options.keys()))
            selected_percentile = percentile_options[selected_percentile_label]
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_page != "All":
            filtered_df = filtered_df[filtered_df["page"] == selected_page]
        
        if selected_event != "All":
            filtered_df = filtered_df[filtered_df["event"] == selected_event]
        
        if selected_percentile > 0:
            threshold = df["WSJF_Friction_Score"].quantile(selected_percentile / 100)
            filtered_df = filtered_df[filtered_df["WSJF_Friction_Score"] >= threshold]
        
        # Add summary metrics at the top based on filtered data
        total_users_lost = filtered_df["users_lost"].sum()
        avg_exit_rate = filtered_df["exit_rate"].mean()
        avg_wsjf = filtered_df["WSJF_Friction_Score"].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Users Lost", f"{total_users_lost:,}")
        col2.metric("Avg. Exit Rate", f"{avg_exit_rate:.2%}")
        col3.metric("Avg. WSJF Score", f"{avg_wsjf:.6f}")
        
        # Display the filtered table
        if len(filtered_df) > 0:
            # Add export button for the filtered data
            export_container = st.container()
            with export_container:
                st.markdown(
                    f'<div class="download-container">{get_csv_download_link(filtered_df, "friction_points", "ðŸ“¥ Export Filtered Data")}</div>',
                    unsafe_allow_html=True
                )
            
            # Format the DataFrame for display
            display_df = filtered_df.copy()
            display_df["exit_rate"] = display_df["exit_rate"].apply(lambda x: f"{x:.2%}")
            display_df["betweenness"] = display_df["betweenness"].apply(lambda x: f"{x:.4f}")
            display_df["WSJF_Friction_Score"] = display_df["WSJF_Friction_Score"].apply(lambda x: f"{x:.6f}")
            
            # Add hover tooltips for column headers
            st.dataframe(
                display_df,
                column_config={
                    "page": st.column_config.TextColumn("Page", help="URL path of the page"),
                    "event": st.column_config.TextColumn("Event", help="User action on the page"),
                    "exit_rate": st.column_config.TextColumn("Exit Rate", help=render_tooltips("exit_rate")),
                    "betweenness": st.column_config.TextColumn("Betweenness", help=render_tooltips("betweenness")),
                    "users_lost": st.column_config.NumberColumn("Users Lost", help=render_tooltips("users_lost")),
                    "WSJF_Friction_Score": st.column_config.TextColumn("WSJF Score", help=render_tooltips("WSJF_Friction_Score"))
                }
            )
            
            # Add some summary metrics
            st.markdown(f"Showing **{len(filtered_df)}** friction points out of {len(df)} total.")
        else:
            st.info("No friction points match the selected filters.")
    except Exception as e:
        st.error(f"Error in friction table: {str(e)}")
        logger.error(f"Error in friction table: {str(e)}")
        logger.error(traceback.format_exc())

def render_network_graph(net, height=600):
    """
    Safely render a pyvis network graph in Streamlit.
    
    This function handles the temporary file creation and cleanup properly
    to avoid file access errors.
    
    Args:
        net (Network): PyVis network object to render
        height (int): Height of the graph in pixels
        
    Returns:
        str: The HTML content of the graph
    """
    # Create a unique filename to avoid conflicts
    unique_id = str(int(time.time() * 1000))
    html_file = f"temp_graph_{unique_id}.html"
    html_content = ""
    
    try:
        # Save the graph to the temporary file
        net.save_graph(html_file)
        
        # Read the file content
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Render the HTML content
        components.html(html_content, height=height, scrolling=False)
        
        return html_content
    except Exception as e:
        st.error(f"Error rendering graph: {str(e)}")
        logger.error(f"Error rendering graph: {str(e)}")
        logger.error(traceback.format_exc())
        return ""
    finally:
        # Clean up - try to remove the temporary file if it exists
        try:
            if os.path.exists(html_file):
                os.remove(html_file)
        except Exception as e:
            logger.warning(f"Could not remove temporary file {html_file}: {str(e)}")

# ============================================================================
# MULTIPARTITE LAYOUT FUNCTIONS - NEW FUNCTIONALITY
# ============================================================================

def setup_multipartite_layout(graph, score_map, layout_type="friction_levels"):
    """
    Set up multipartite layout by assigning nodes to different subsets.
    
    Args:
        graph: NetworkX graph
        score_map: Dictionary mapping nodes to WSJF scores
        layout_type: "friction_levels", "funnel_stages", or "betweenness_tiers"
    
    Returns:
        Graph with subset attributes added to nodes
    """
    
    if layout_type == "friction_levels":
        return setup_friction_multipartite(graph, score_map)
    elif layout_type == "funnel_stages":
        return setup_funnel_multipartite(graph, score_map)
    elif layout_type == "betweenness_tiers":
        return setup_betweenness_multipartite(graph, score_map)
    else:
        return setup_friction_multipartite(graph, score_map)

def setup_friction_multipartite(graph, score_map):
    """
    Arrange nodes in layers by friction level (high friction at top).
    Perfect for PM dashboards - shows problem areas prominently.
    """
    
    # Calculate thresholds
    scores = list(score_map.values())
    if not scores:
        return graph
    
    top10_threshold = np.percentile(scores, 90)
    top25_threshold = np.percentile(scores, 75)
    top50_threshold = np.percentile(scores, 50)
    
    # Assign nodes to subsets (layers)
    for node in graph.nodes():
        score = score_map.get(node, 0)
        
        if score >= top10_threshold:
            subset = 0  # Top layer - highest friction
        elif score >= top25_threshold:
            subset = 1  # Second layer
        elif score >= top50_threshold:
            subset = 2  # Third layer
        else:
            subset = 3  # Bottom layer - lowest friction
        
        graph.nodes[node]['subset'] = subset
    
    return graph

def setup_funnel_multipartite(graph, score_map):
    """
    Assign nodes into funnel stages based on known page patterns.
    0 = Awareness
    1 = Discovery
    2 = Consideration
    3 = Conversion
    4 = Retention
    5 = Exit
    """

    stage_keywords = {
        0: ["landing", "ads", "referral", "blog", "faq"],
        1: ["home", "search", "products", "categories"],
        2: ["product", "features", "pricing", "review"],
        3: ["login", "register", "checkout", "cart", "plan"],
        4: ["account", "settings", "dashboard", "orders"],
        5: ["help", "contact", "logout", "404"]
    }

    def infer_stage(node_name):
        name = node_name.lower()
        for stage, keywords in stage_keywords.items():
            if any(k in name for k in keywords):
                return stage
        return 1  # fallback = Discovery

    for node in graph.nodes():
        stage = infer_stage(str(node))
        graph.nodes[node]["subset"] = stage

    # Debug summary
    from collections import Counter
    stage_counts = Counter(nx.get_node_attributes(graph, "subset").values())
    print("Funnel Stage Counts:")
    for stage, count in sorted(stage_counts.items()):
        print(f"  Stage {stage}: {count} nodes")

    return graph

def setup_betweenness_multipartite(graph, score_map):
    """
    Arrange nodes by centrality (hub nodes in middle, peripheral on edges).
    Shows information flow patterns.
    """
    
    betweenness = nx.betweenness_centrality(graph)
    centrality_values = list(betweenness.values())
    
    if not centrality_values:
        return graph
    
    high_centrality = np.percentile(centrality_values, 80)
    med_centrality = np.percentile(centrality_values, 20)
    
    for node in graph.nodes():
        centrality = betweenness.get(node, 0)
        
        if centrality >= high_centrality:
            subset = 1  # Middle layer - hub nodes
        elif centrality >= med_centrality:
            subset = 0  # First layer
        else:
            subset = 2  # Outer layer - peripheral nodes
        
        graph.nodes[node]['subset'] = subset
    
    return graph

def apply_multipartite_layout(graph, score_map, layout_type="friction_levels", scale=800):
    """
    Apply layout to graph using multipartite or force-directed arrangement.
    
    - funnel_stages: leftâ†’right (vertical alignment â†’ x = stage)
    - friction_levels: topâ†’bottom (horizontal alignment â†’ y = friction)
    - betweenness_tiers: force-directed hub-spoke layout
    """

    # --------------------------------------------
    # Case 1: Force-directed layout for betweenness
    # --------------------------------------------
    if layout_type == "betweenness_tiers":
        try:
            pos = nx.kamada_kawai_layout(graph, weight='weight')
        except:
            pos = nx.spring_layout(graph, k=2, iterations=100, seed=42)

        for node in pos:
            x, y = pos[node]
            pos[node] = (x * scale * 0.4, y * scale * 0.4)

        return pos

    # --------------------------------------------
    # Case 2: Multipartite layout (funnel/friction)
    # --------------------------------------------
    graph_copy = graph.copy()

    if layout_type == "funnel_stages":
        graph_copy = setup_funnel_multipartite(graph_copy, score_map)
    else:
        graph_copy = setup_multipartite_layout(graph_copy, score_map, layout_type)

    try:
        # Choose alignment based on layout type
        align = "horizontal" if layout_type == "friction_levels" else "vertical"
        pos = nx.multipartite_layout(graph_copy, subset_key='subset', align=align)

        # Set correct axis-based scaling
        if align == "vertical":
            # Subset controls X-axis (leftâ†’right)
            scale_x = scale * 0.6
            scale_y = scale * 0.6
        else:
            # Subset controls Y-axis (topâ†’bottom)
            scale_x = scale * 0.4
            scale_y = scale * 0.6

        # Apply scaled and separated positions
        for node in pos:
            x, y = pos[node]
            pos[node] = (x * scale_x, y * scale_y)

        # Optional debug info
        if pos:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            print(f"[Layout: {layout_type}] X range: {min(xs):.1f} â†’ {max(xs):.1f}")
            print(f"[Layout: {layout_type}] Y range: {min(ys):.1f} â†’ {max(ys):.1f}")

        return pos

    except Exception as e:
        print(f"âš ï¸ Layout fallback due to error: {e}")
        return nx.spring_layout(graph, k=3, iterations=100, seed=42)


def render_enhanced_legend(layout_type="friction_levels", physics_enabled=False):
    """
    Render an enhanced legend that shows both color coding and layout arrangement.
    
    Args:
        layout_type: The selected multipartite layout type
        physics_enabled: Whether physics is currently enabled
    """
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### ðŸ“Š Node Colors")
        st.markdown("""
        - <span style="display:inline-block;width:15px;height:15px;border-radius:50%;background-color:#F87171;"></span> **Red**: Top 10% friction (highest WSJF scores)
        - <span style="display:inline-block;width:15px;height:15px;border-radius:50%;background-color:#FBBF24;"></span> **Yellow**: Top 20% friction
        - <span style="display:inline-block;width:15px;height:15px;border-radius:50%;background-color:#A3E635;"></span> **Green**: Top 50% friction  
        - <span style="display:inline-block;width:15px;height:15px;border-radius:50%;background-color:#94A3B8;"></span> **Gray**: Lower friction
        """, unsafe_allow_html=True)
    
    with col2:
        if not physics_enabled:
            st.markdown("### ðŸ“ Layout Arrangement")
            
            if layout_type == "friction_levels":
                st.markdown("""
                - **Top Layer**: Highest friction (urgent)
                - **Middle Layers**: Medium friction (monitor)  
                - **Lowest Layer**: Low friction (stable)
                """)
            elif layout_type == "funnel_stages":
                st.markdown("""
                **User Funnel Stages:**
                - ðŸšª **Left**: Entry points
                - âš™ï¸ **Middle**: Conversion steps
                - ðŸšª **Right**: Exit points
                """)
            elif layout_type == "betweenness_tiers":
                st.markdown("""
                **Journey Centrality Structure:**
                - ðŸ“ **Edges**: Peripheral nodes
                - ðŸŽ¯ **Center**: Hub nodes
                """)
        else:
            st.markdown("### â„¹ï¸ Physics Mode")
            st.markdown("Nodes move freely. Layout arrangement disabled.")

# ============================================================================
# UPDATED NETWORK CREATION FUNCTIONS WITH MULTIPARTITE SUPPORT
# ============================================================================

def create_full_network(graph, score_map, top10_threshold, top20_threshold, physics_enabled, top50_threshold=None, layout_type="friction_levels"):
    """
    Create a network with all nodes, now supporting multipartite layouts.
    
    Args:
        graph (nx.DiGraph): The user journey graph
        score_map (dict): Mapping of page names to WSJF scores
        top10_threshold (float): Threshold for top 10% nodes
        top20_threshold (float): Threshold for top 20% nodes
        physics_enabled (bool): Whether physics simulation is enabled
        top50_threshold (float, optional): Threshold for top 50% nodes
        layout_type (str): Type of multipartite layout to use
        
    Returns:
        Network: PyVis network object
    """
    # Create a pyvis network
    net = Network(height="600px", width="100%", bgcolor="#0B0F19", font_color="#E2E8F0")
    
    # Get positions using multipartite layout (when physics is disabled)
    if not physics_enabled:
        pos = apply_multipartite_layout(graph, score_map, layout_type)
    else:
        # Use default positioning for physics-enabled mode
        pos = {node: (0, 0) for node in graph.nodes()}
    
    # Create a mapping of node colors for use in highlighting edges
    node_colors = {}
    
    # Add nodes with colors based on WSJF score
    for node in graph.nodes():
        score = score_map.get(node, 0)
        
        if score >= top10_threshold:
            color = "#F87171"  # Soft red for top 10%
            title = f"{node} | WSJF Score: {score:.3f} Top 10%"
            node_colors[node] = color
        elif score >= top20_threshold:
            color = "#FBBF24"  # Warm amber for top 20%
            title = f"{node} | WSJF Score: {score:.3f} Top 20%"
            node_colors[node] = color
        elif top50_threshold and score >= top50_threshold:
            color = "#A3E635"  # Lime green for top 50%
            title = f"{node} | WSJF Score: {score:.3f} Top 50%"
            node_colors[node] = color
        else:
            color = "#94A3B8"  # Soft steel for low-friction nodes
            title = f"{node} | WSJF Score: {score:.3f}"
            node_colors[node] = color
        
        # Apply position
        x, y = pos.get(node, (0, 0))
        net.add_node(
            node, 
            title=title, 
            color=color, 
            font={"color": "#FFFFFF"},
            x=x, 
            y=y, 
            physics=physics_enabled  # Only use physics if enabled
        )
    
    # Add edges with event labels
    for source, target, data in graph.edges(data=True):
        event = data.get("event", "")
        weight = data.get("weight", 1)
        
        # Scale edge width based on weight, but keep it reasonable
        width = min(1 + (weight / 5), 5)
        
        # Define edge color to match source node's color when highlighted
        source_color = node_colors.get(source, "#94A3B8")
        
        net.add_edge(
            source, 
            target, 
            title=event, 
            width=width, 
            color={
                "color": "rgba(148, 163, 184, 0.35)",
                "highlight": source_color,  # Use source node color for highlighting
                "hover": source_color       # Also use it for hover effects
            },
            smooth={"type": "continuous", "roundness": 0.2}
        )
    
    # Configure network options with more precise color inheritance
    options = {
        "nodes": {
            "font": {"color": "#FFFFFF", "size": 16},
            "color": {
                "highlight": {"border": "#38BDF8", "background": "#38BDF8"}
            },
            "borderWidth": 2,
            "shadow": {
                "enabled": True,
                "color": "rgba(0,0,0,0.5)",
                "size": 10,
                "x": 0,
                "y": 0
            }
        },
        "edges": {
            "color": {
                "color": "rgba(148, 163, 184, 0.35)",
                "inherit": False  # Don't use the default inheritance, we set colors explicitly
            },
            "smooth": False,
            "font": {"color": "#FFFFFF"},
            "selectionWidth": 3,  # Make selected edges wider
            "hoverWidth": 2,      # Make hovered edges wider
            "arrows": {
                "to": {
                    "enabled": True,
                    "scaleFactor": 0.5
                }
            }
        },
        "physics": {
            "enabled": physics_enabled,
            "stabilization": {
                "enabled": physics_enabled,  # Only stabilize when physics is enabled
                "iterations": 1500 if physics_enabled else 0,
                "updateInterval": 1,
                "fit": physics_enabled  # Only auto-fit when physics is enabled
            },
            "barnesHut": {
                "gravitationalConstant": -15000,
                "centralGravity": 0.3,
                "springLength": 200,
                "springConstant": 0.04,
                "damping": 0.09
            },
            "maxVelocity": 50
        },
        "layout": {"improvedLayout": True, "randomSeed": 42},
        "interaction": {
            "hover": True,
            "tooltipDelay": 200,
            "hideEdgesOnDrag": False,
            "selectable": True,
            "multiselect": False,
            "navigationButtons": True,
            "selectConnectedEdges": True,  # Highlight connected edges when clicking a node
            "hoverConnectedEdges": True,    # Highlight connected edges when hovering over a node
            "dragNodes": True,
            "dragView": True
        }
    }
    
    net.set_options(json.dumps(options))
    return net

def create_filtered_network(graph, filtered_nodes, top10_threshold, top20_threshold, physics_enabled, top50_threshold=None, layout_type="friction_levels"):
    """
    Create a network with only the filtered nodes, now supporting multipartite layouts.
    
    Args:
        graph (nx.DiGraph): The user journey graph
        filtered_nodes (dict): Mapping of page names to WSJF scores that passed the filter
        top10_threshold (float): Threshold for top 10% nodes
        top20_threshold (float): Threshold for top 20% nodes
        physics_enabled (bool): Whether physics simulation is enabled
        top50_threshold (float, optional): Threshold for top 50% nodes
        layout_type (str): Type of multipartite layout to use
        
    Returns:
        Network: PyVis network object
    """
    # Create a pyvis network
    net = Network(height="600px", width="100%", bgcolor="#0B0F19", font_color="#E2E8F0")
    
    # Create filtered subgraph
    filtered_graph = graph.subgraph(filtered_nodes.keys())
    
    # Get positions using multipartite layout (when physics is disabled)
    if not physics_enabled:
        pos = apply_multipartite_layout(filtered_graph, filtered_nodes, layout_type)
    else:
        pos = {node: (0, 0) for node in filtered_graph.nodes()}

    # Create a mapping of node colors for use in highlighting edges
    node_colors = {}
    
    # Add only the filtered nodes
    for node, score in filtered_nodes.items():
        if node in graph.nodes():
            if score >= top10_threshold:
                color = "#F87171"  # Soft red for top 10%
                title = f"{node} | WSJF Score: {score:.3f} Top 10%"
                node_colors[node] = color
            elif score >= top20_threshold:
                color = "#FBBF24"  # Warm amber for top 20%
                title = f"{node} | WSJF Score: {score:.3f} Top 20%"
                node_colors[node] = color
            elif top50_threshold and score >= top50_threshold:
                color = "#A3E635"  # Lime green for top 50%
                title = f"{node} | WSJF Score: {score:.3f} Top 50%"
                node_colors[node] = color
            else:
                color = "#94A3B8"  # Soft steel for low-friction nodes
                title = f"{node} | WSJF Score: {score:.3f}"
                node_colors[node] = color
            
            # Apply position
            x, y = pos.get(node, (0, 0))
            net.add_node(
                node, 
                title=title, 
                color=color, 
                font={"color": "#FFFFFF"},
                x=x, 
                y=y, 
                physics=physics_enabled
            )
    
    # Add edges between filtered nodes
    for source, target, data in graph.edges(data=True):
        if source in filtered_nodes and target in filtered_nodes:
            event = data.get("event", "")
            weight = data.get("weight", 1)
            width = min(1 + (weight / 5), 5)
            
            # Define edge color to match source node's color when highlighted
            source_color = node_colors.get(source, "#FBBF24")  # Default to orange for filtered view
            
            net.add_edge(
                source, 
                target, 
                title=event, 
                width=width, 
                color={
                    "color": "rgba(148, 163, 184, 0.35)",
                    "highlight": source_color,  # Use source node color for highlighting
                    "hover": source_color       # Also use it for hover effects
                },
                smooth={"type": "continuous", "roundness": 0.2}
            )
    
    # Configure network options with more precise color inheritance
    options = {
        "nodes": {
            "font": {"color": "#FFFFFF", "size": 14},
            "color": {
                "highlight": {"border": "#38BDF8", "background": "#38BDF8"}
            },
            "borderWidth": 2,
            "shadow": {
                "enabled": True,
                "color": "rgba(0,0,0,0.5)",
                "size": 10,
                "x": 0,
                "y": 0
            }
        },
        "edges": {
            "color": {
                "color": "rgba(148, 163, 184, 0.35)",
                "inherit": False  # Don't use the default inheritance, we set colors explicitly
            },
            "smooth": False,
            "font": {"color": "#FFFFFF"},
            "selectionWidth": 3,  # Make selected edges wider
            "hoverWidth": 2,      # Make hovered edges wider
            "arrows": {
                "to": {
                    "enabled": True,
                    "scaleFactor": 0.5
                }
            }
        },
        "physics": {
            "enabled": physics_enabled,  # always on
            "stabilization": {
                "enabled": physics_enabled,  # only run stabilization if checkbox is on
                "iterations": 1500 if physics_enabled else 0,
                "fit": physics_enabled
            },
            "barnesHut": {
                "gravitationalConstant": -15000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.04,
                "damping": 0.09
            },
            "maxVelocity": 50
        },
        "layout": {"improvedLayout": True, "randomSeed": 42},
        "interaction": {
            "hover": True,
            "tooltipDelay": 200,
            "hideEdgesOnDrag": False,
            "selectable": True,
            "multiselect": False,
            "navigationButtons": True,
            "selectConnectedEdges": True,  # Highlight connected edges when clicking a node
            "hoverConnectedEdges": True,    # Highlight connected edges when hovering over a node
            "dragNodes": True,
            "dragView": True
        }
    }
    
    net.set_options(json.dumps(options))
    return net

def render_graph_heatmap(graph: nx.DiGraph, score_map: Dict[str, float]):
    """
    Render a graph heatmap with nodes colored by WSJF score, now with multipartite layout support.
    
    Args:
        graph (nx.DiGraph): Directed graph of user journeys.
        score_map (Dict[str, float]): Dictionary mapping page names to WSJF scores.
    """
    st.header("ðŸŒ User Journey Graph", help="Visualizes user journeys with friction nodes and their connecting flows")
    st.write("Visualize user journeys across multiple paths, with high-friction chokepoints (nodes) and connecting flows (edges) highlighted.")
    
    try:
        # Compute thresholds for coloring
        if score_map:
            scores = list(score_map.values())
            top10_threshold = np.percentile(scores, 90)
            top20_threshold = np.percentile(scores, 80)
            top50_threshold = np.percentile(scores, 50)  # Add top 50% threshold
        else:
            top10_threshold = 1.0
            top20_threshold = 0.5
            top50_threshold = 0.2  # Default value for top 50% threshold
        
        # Filter and layout controls - NOW WITH 3 COLUMNS
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Option to show only top friction nodes
            show_options = ["All nodes", "Top 50% friction nodes", "Top 20% friction nodes", "Top 10% friction nodes"]
            show_selection = st.selectbox("Show in graph:", show_options)
        
        with col2:
            # Option to adjust physics
            physics_enabled = st.checkbox("Enable Physics", value=False)
        
        with col3:
            # NEW: Layout type selection
            layout_options = {
                "Friction Levels": "friction_levels",
                "Funnel Stages": "funnel_stages", 
                "Journey Centrality": "betweenness_tiers"
            }
            
            selected_layout_name = st.selectbox(
                "Layout arrangement:",
                list(layout_options.keys()),
                disabled=physics_enabled,
                help="Choose layouts by friction, funnel stages, or journey centrality (only when physics is disabled)"
            )
            layout_type = layout_options[selected_layout_name]
        
        # Display current configuration info
        if not physics_enabled:
            layout_descriptions = {
                "friction_levels": "ðŸ“Š Nodes arranged by friction level (problems at top)",
                "funnel_stages": "ðŸ”„ Nodes arranged by user funnel stages (entry â†’ exit)",
                "betweenness_tiers": "ðŸŽ¯ Nodes arranged by journey centrality (hubs centered)"
            }
            st.info(f"**Active Layout**: {layout_descriptions.get(layout_type, 'Custom arrangement')}")
        
        # Create and configure the network based on filters
        html_content = ""
        if show_selection == "Top 10% friction nodes":
            # Filter to only include top 10% nodes
            filtered_nodes = {node: score for node, score in score_map.items() if score >= top10_threshold}
            if not filtered_nodes:
                st.warning("No nodes meet the top 10% threshold. Try showing all nodes or top 20%.")
                return
            
            net = create_filtered_network(graph, filtered_nodes, top10_threshold, top20_threshold, physics_enabled, top50_threshold, layout_type)
        elif show_selection == "Top 20% friction nodes":
            # Filter to only include top 20% nodes
            filtered_nodes = {node: score for node, score in score_map.items() if score >= top20_threshold}
            if not filtered_nodes:
                st.warning("No nodes meet the top 20% threshold. Try showing all nodes.")
                return
            
            net = create_filtered_network(graph, filtered_nodes, top10_threshold, top20_threshold, physics_enabled, top50_threshold, layout_type)
        elif show_selection == "Top 50% friction nodes":
            # Filter to only include top 50% nodes
            filtered_nodes = {node: score for node, score in score_map.items() if score >= top50_threshold}
            if not filtered_nodes:
                st.warning("No nodes meet the top 50% threshold. Try showing all nodes.")
                return
            
            net = create_filtered_network(graph, filtered_nodes, top10_threshold, top20_threshold, physics_enabled, top50_threshold, layout_type)
        else:
            # Show all nodes
            net = create_full_network(graph, score_map, top10_threshold, top20_threshold, physics_enabled, top50_threshold, layout_type)
        
        # Add export button for the graph
        export_container = st.container()
        with export_container:
            st.markdown(
                '<div class="download-container">Export current graph view using button below</div>',
                unsafe_allow_html=True
            )
        
        # Render the network and get HTML content
        html_content = render_network_graph(net)
        
        # Add export HTML button if we have content
        if html_content:
            filtered_text = "top_10pct" if show_selection == "Top 10% friction nodes" else "top_20pct" if show_selection == "Top 20% friction nodes" else "top_50pct" if show_selection == "Top 50% friction nodes" else "all_nodes"
            filename = f"flow_heatmap_{filtered_text}_{layout_type}"
            
            st.markdown(
                f'<div class="download-container">{get_html_download_link(html_content, filename, "ðŸ“¥ Export Graph as HTML")}</div>',
                unsafe_allow_html=True
            )
        
        # ENHANCED LEGEND - Replace the existing simple legend
        st.markdown("---")  # Separator line
        render_enhanced_legend(layout_type, physics_enabled)
        
    except Exception as e:
        st.error(f"Error in graph heatmap: {str(e)}")
        logger.error(f"Error in graph heatmap: {str(e)}")
        logger.error(traceback.format_exc())

def render_flow_summaries(flow_df: pd.DataFrame):
    """
    Render summaries of fragile flows (paths with multiple high-friction points).
    
    Args:
        flow_df (pd.DataFrame): DataFrame containing fragile flows.
    """
    try:
        if len(flow_df) == 0:
            st.info("No fragile flows detected. A fragile flow contains 2 or more high-friction points.")
            return
        
        # Get unique session IDs
        session_ids = flow_df["session_id"].unique()
        
        # Generate flow length options
        min_flow_length = int(flow_df.groupby("session_id").size().min())
        max_flow_length = int(flow_df.groupby("session_id").size().max())
        flow_length_options = list(range(min_flow_length, max_flow_length + 1))
        
        # Generate chokepoint options
        max_chokepoints = int(flow_df.groupby("session_id")["is_chokepoint"].sum().max())
        max_chokepoints = max(2, max_chokepoints)  # Ensure at least 2
        chokepoint_options = list(range(2, max_chokepoints + 1))
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Dropdown for flow length instead of slider
            selected_length = st.selectbox(
                "Minimum flow length:",
                options=flow_length_options,
                index=0  # Default to minimum
            )
        
        with col2:
            # Dropdown for chokepoints instead of slider
            selected_chokepoints = st.selectbox(
                "Minimum chokepoints:",
                options=chokepoint_options,
                index=0  # Default to 2
            )
            
        with col3:
            # Filter by top friction percentile
            percentile_options = {
                "All flows": 0,
                "Top 25%": 75,
                "Top 10%": 90,
                "Top 5%": 95
            }
            selected_percentile_label = st.selectbox("Show:", list(percentile_options.keys()))
            selected_percentile = percentile_options[selected_percentile_label]
        
        # Apply filters
        filtered_sessions = []
        
        # Calculate total WSJF score for each session for percentile filtering
        session_scores = {}
        for session_id in session_ids:
            session_data = flow_df[flow_df["session_id"] == session_id]
            session_scores[session_id] = session_data["WSJF_Friction_Score"].sum()
        
        # Apply percentile filter if selected
        if selected_percentile > 0:
            scores = list(session_scores.values())
            percentile_threshold = np.percentile(scores, selected_percentile)
            session_ids = [sid for sid, score in session_scores.items() if score >= percentile_threshold]
        
        for session_id in session_ids:
            session_data = flow_df[flow_df["session_id"] == session_id]
            
            # Check flow length
            if len(session_data) < selected_length:
                continue
            
            # Check number of chokepoints
            chokepoint_count = session_data["is_chokepoint"].sum()
            if chokepoint_count < selected_chokepoints:
                continue
            
            filtered_sessions.append(session_id)
        
        if not filtered_sessions:
            st.info("No flows match the selected filters.")
            return
            
        # Create a DataFrame with all filtered session data for export
        filtered_flow_df = flow_df[flow_df["session_id"].isin(filtered_sessions)].copy()
        
        # Add export button for the filtered flows
        export_container = st.container()
        with export_container:
            st.markdown(
                f'<div class="download-container">{get_csv_download_link(filtered_flow_df, "fragile_flows", "ðŸ“¥ Export Filtered Flows")}</div>',
                unsafe_allow_html=True
            )
        
        # Display a summary for each fragile flow
        st.markdown(f"**Found {len(filtered_sessions)} fragile flows matching criteria**")
        
        for i, session_id in enumerate(filtered_sessions[:10]):  # Limit to 10 flows for performance
            session_data = flow_df[flow_df["session_id"] == session_id].sort_values("step_index")
            
            # Calculate total friction score for this flow
            total_friction = session_data["WSJF_Friction_Score"].sum()
            chokepoint_count = session_data["is_chokepoint"].sum()
            
            # Create an expander for this flow with custom styling to ensure consistent text colors
            expander_label = f"Flow {i+1}: {session_id} ({len(session_data)} steps, {chokepoint_count} chokepoints)"
            with st.expander(expander_label):
                # Display the flow path
                st.markdown("### Flow Path")
                
                # Format as page â†’ event â†’ page â†’ event...
                path_parts = []
                for _, row in session_data.iterrows():
                    page = row["page"]
                    event = row["event"]
                    is_chokepoint = row["is_chokepoint"] == 1
                    
                    # Add formatting to highlight chokepoints
                    if is_chokepoint:
                        path_parts.append(f"**{page}** (_{event}_)")
                    else:
                        path_parts.append(f"{page} (_{event}_)")
                
                path_str = " â†’ ".join(path_parts)
                st.markdown(path_str)
                
                # Display metrics for this flow
                st.markdown("### Flow Metrics")
                # Fix: Use 3 columns instead of trying to unpack into col1, col2, col3
                cols = st.columns(3)
                cols[0].metric("Steps", len(session_data))
                cols[1].metric("Chokepoints", chokepoint_count)
                cols[2].metric("Total WSJF Score", f"{total_friction:.6f}")
                
                # Display the flow data in a table
                st.markdown("### Step Details")
                
                # Format DataFrame for display
                display_df = session_data.copy()
                display_df["WSJF_Friction_Score"] = display_df["WSJF_Friction_Score"].apply(lambda x: f"{x:.6f}")
                display_df["is_chokepoint"] = display_df["is_chokepoint"].apply(lambda x: "âœ…" if x == 1 else "")
                
                # Reorder and rename columns
                display_df = display_df[["step_index", "page", "event", "WSJF_Friction_Score", "is_chokepoint"]]
                display_df.columns = ["Step", "Page", "Event", "WSJF Score", "Chokepoint"]
                
                st.dataframe(display_df)
                
                # Add individual flow export option
                st.markdown(
                    f'<div class="download-container">{get_csv_download_link(session_data, f"flow_{session_id}", f"ðŸ“¥ Export Flow {session_id}")}</div>',
                    unsafe_allow_html=True
                )
        
        # Add info about additional flows if there are more than 10
        if len(filtered_sessions) > 10:
            st.info(f"{len(filtered_sessions) - 10} more flows match the criteria but are not shown. Refine filters to see different flows.")
    except Exception as e:
        st.error(f"Error in fragile flows: {str(e)}")
        logger.error(f"Error in fragile flows: {str(e)}")
        logger.error(traceback.format_exc())

def load_advanced_metrics(dataset=None) -> Dict:
    """
    Load advanced metrics data from files.
    
    Args:
        dataset (str, optional): Name of the dataset to load. If None, uses the default dataset.
        
    Returns:
        Dict containing:
        - decision_table: DataFrame with decision table data
        - final_report: DataFrame with network metrics in tabular format
        - network_metrics: Dictionary with additional metrics (if available)
    """
    try:
        # Determine dataset directory
        dataset_dir = Path(f"outputs/{dataset if dataset else 'default'}")
        
        # Check if dataset directory exists
        if not dataset_dir.exists() and dataset:
            st.error(f"Dataset directory not found: {dataset_dir}")
            st.info("Please select a valid dataset from the dropdown.")
            return None
        
        # Check if required files exist
        decision_table_path = dataset_dir / "decision_table.csv"
        final_report_path = dataset_dir / "final_report.csv"
        final_report_json_path = dataset_dir / "final_report.json"
        
        if not decision_table_path.exists():
            st.warning(f"Decision table not found: {decision_table_path}")
            # Create empty DataFrame with expected columns
            decision_table = pd.DataFrame(columns=[
                "node", "D", "alpha", "FB", "percolation_role", 
                "wsjf_score", "ux_label", "suggested_action"
            ])
        else:
            # Load decision table
            decision_table = pd.read_csv(decision_table_path)
        
        if not final_report_path.exists():
            st.warning(f"Final report CSV not found: {final_report_path}")
            # Create empty DataFrame with expected columns
            final_report = pd.DataFrame(columns=["metric", "value"])
        else:
            # Load final report
            final_report = pd.read_csv(final_report_path)
        
        # Load JSON metrics if available for additional data
        network_metrics = {}
        if final_report_json_path.exists():
            try:
                with open(final_report_json_path, 'r') as f:
                    network_metrics = json.load(f)
            except Exception as e:
                logger.error(f"Error loading network metrics JSON: {str(e)}")
                st.warning(f"Error loading network metrics: {str(e)}")
        
        # Try to load recurring patterns if available
        recurring_patterns_path = dataset_dir / "recurring_patterns.json"
        if recurring_patterns_path.exists():
            try:
                with open(recurring_patterns_path, 'r') as f:
                    network_metrics["recurring_patterns"] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading recurring patterns: {str(e)}")
        
        # Count critical nodes (nodes with percolation_role = "critical")
        try:
            network_metrics["critical_nodes_count"] = len(decision_table[decision_table["percolation_role"] == "critical"])
        except Exception as e:
            logger.error(f"Error counting critical nodes: {str(e)}")
            network_metrics["critical_nodes_count"] = 0
        
        return {
            "decision_table": decision_table,
            "final_report": final_report,
            "network_metrics": network_metrics
        }
    
    except Exception as e:
        st.error(f"Error loading advanced metrics: {str(e)}")
        logger.error(f"Error loading advanced metrics: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "decision_table": pd.DataFrame(columns=[
                "node", "D", "alpha", "FB", "percolation_role", 
                "wsjf_score", "ux_label", "suggested_action"
            ]),
            "final_report": pd.DataFrame(columns=["metric", "value"]),
            "network_metrics": {}
        }

def render_top_metrics(metrics_data):
    """
    Render the top metrics panel.
    
    Args:
        metrics_data (Dict): Dictionary containing metrics data.
    """
    st.subheader("ðŸ“Š Network Structure Metrics", help="Key metrics describing the graph structure")
    
    # Extract metrics from the final report
    if "final_report" not in metrics_data or metrics_data["final_report"].empty:
        st.info("No network metrics available for this dataset.")
        return
    
    # Create a mapping of metrics
    metrics = {}
    for _, row in metrics_data["final_report"].iterrows():
        metrics[row['metric']] = row['value']
    
    # Format metrics for display
    try:
        # Get global network metrics
        fractal_dim = float(metrics.get('Fractal Dimension', 0))
        power_law = float(metrics.get('Power Law Alpha', 0))
        percolation = float(metrics.get('Percolation Threshold', 0))
        
        # Display global metrics in columns with gauges
        cols = st.columns(3)
        
        with cols[0]:
            st.metric(
                label="Fractal Dimension (D)", 
                value=f"{fractal_dim:.2f}",
                help="Measures how space-filling the graph is. Higher values indicate more complex user navigation paths."
            )
            
            # Visual indicator
            if fractal_dim > 1.5:
                st.markdown("```diff\n- âš ï¸ System may be overly complex\n```")
            elif fractal_dim < 1.2:
                st.markdown("```diff\n+ â„¹ï¸ Simple, linear structure\n```")
            else:
                st.markdown("```diff\n+ âœ“ Balanced complexity\n```")
        
        with cols[1]:
            st.metric(
                label="Power-Law Alpha (Î±)", 
                value=f"{power_law:.2f}",
                help="Exponent of degree distribution. Lower values indicate stronger hierarchy with hub nodes."
            )
            
            # Visual indicator
            if power_law < 2.5:
                st.markdown("```diff\n- âš ï¸ Vulnerable to single-node failure\n```")
            elif power_law > 3.0:
                st.markdown("```diff\n+ â„¹ï¸ More uniform structure\n```")
            else:
                st.markdown("```diff\n+ âœ“ Balanced scale-free properties\n```")
        
        with cols[2]:
            st.metric(
                label="Percolation Threshold", 
                value=f"{percolation:.2f}",
                help="Fraction of top nodes that can be removed before network collapses."
            )
            
            # Visual indicator
            critical_nodes = metrics_data.get("network_metrics", {}).get("critical_nodes_count", 0)
            if percolation < 0.3:
                st.markdown(f"```diff\n- âš ï¸ Fragile ({critical_nodes} critical nodes)\n```")
            elif percolation > 0.7:
                st.markdown(f"```diff\n+ âœ“ Robust ({critical_nodes} critical nodes)\n```")
            else:
                st.markdown(f"```diff\n+ âœ“ Moderate ({critical_nodes} critical nodes)\n```")
    
    except Exception as e:
        st.error(f"Error displaying metrics: {e}")

def render_decision_table(metrics_data):
    """
    Render the decision table explorer.
    
    Args:
        metrics_data (Dict): Dictionary containing metrics data.
    """
    st.subheader("ðŸ§  Decision Table & Recommendations", help="Explore nodes by role and impact metrics")
    
    decision_table = metrics_data["decision_table"]
    
    if decision_table.empty:
        st.info("No decision table data available for this dataset.")
        return
    
    # Get global metrics for reference (but don't include in node table)
    global_metrics = {}
    for _, row in metrics_data["final_report"].iterrows():
        if row["metric"] == "Fractal Dimension":
            global_metrics["D"] = row["value"]
        elif row["metric"] == "Power Law Alpha":
            global_metrics["alpha"] = row["value"]
    
    # Display global metrics reference
    st.info(f"""
    **Global Network Metrics:** 
    - Fractal Dimension (D): {float(global_metrics.get('D', 0)):.2f}
    - Power-Law Alpha (Î±): {float(global_metrics.get('alpha', 0)):.2f}
    
    These metrics apply to the entire graph and are not node-specific.
    See the Network Structure Metrics panel for details.
    """)
    
    # Check if the node-level metrics are available
    has_node_metrics = "fractal_participation" in decision_table.columns and "hubness" in decision_table.columns
    
    # Add explanation about metrics
    if has_node_metrics:
        with st.expander("â„¹ï¸ Node-Level Metrics Explanation", expanded=False):
            st.markdown("""
            - **Fractal Participation**: How deeply a node participates in nested or self-similar structures. Higher values indicate nodes that are part of complex, redundant, or nested patterns.
            
            - **Hubness**: How important a node is in the scale-free structure of the network. Higher values indicate hub nodes with more influence or centrality.
            
            - **FB (Fractal Betweenness)**: Measures a node's importance in controlling information flow, especially through repetitive patterns.
            
            - **WSJF Score**: Weighted Shortest Job First score - combines frequency of friction with impact.
            """)
    
    # Filter controls
    st.markdown("#### Filter & Explore Nodes")
    
    # Create columns for filters
    cols = st.columns(4)
    
    # Filter by role
    with cols[0]:
        roles = ['All'] + sorted(decision_table['percolation_role'].unique().tolist())
        selected_role = st.selectbox('Role:', roles, help="Filter by node's structural role in the network")
    
    # Filter by UX label
    with cols[1]:
        labels = ['All'] + sorted(decision_table['ux_label'].unique().tolist())
        selected_label = st.selectbox('UX Label:', labels, help="Filter by UX classification")
    
    # Filter by WSJF score range
    with cols[2]:
        min_wsjf = float(decision_table['wsjf_score'].min())
        max_wsjf = float(decision_table['wsjf_score'].max())
        wsjf_range = st.slider('WSJF Score Range:', 
                              min_value=min_wsjf, 
                              max_value=max_wsjf,
                              value=(min_wsjf, max_wsjf),
                              help="Filter by friction impact score")
    
    # Filter by FB (Fractal Betweenness) range
    with cols[3]:
        min_fb = float(decision_table['FB'].min())
        max_fb = float(decision_table['FB'].max())
        fb_range = st.slider('Fractal Betweenness Range:', 
                            min_value=min_fb, 
                            max_value=max_fb,
                            value=(min_fb, max_fb),
                            help="Filter by structural importance score")
    
    # Apply filters
    filtered_table = decision_table.copy()
    
    if selected_role != 'All':
        filtered_table = filtered_table[filtered_table['percolation_role'] == selected_role]
    
    if selected_label != 'All':
        filtered_table = filtered_table[filtered_table['ux_label'] == selected_label]
    
    filtered_table = filtered_table[
        (filtered_table['wsjf_score'] >= wsjf_range[0]) & 
        (filtered_table['wsjf_score'] <= wsjf_range[1]) &
        (filtered_table['FB'] >= fb_range[0]) & 
        (filtered_table['FB'] <= fb_range[1])
    ]
    
    # Sort by UX impact (WSJF + FB)
    filtered_table['impact_score'] = filtered_table['wsjf_score'] + (filtered_table['FB'] / filtered_table['FB'].max())
    filtered_table = filtered_table.sort_values('impact_score', ascending=False)
    
    # Display the table
    st.markdown("#### Decision Table")
    
    # If no results, show a message
    if filtered_table.empty:
        st.warning("No nodes match the selected filters.")
        return
    
    # Enhance display for the table
    display_table = filtered_table.copy()
    
    # Remove global metrics from display table if they exist
    if 'D' in display_table.columns:
        display_table = display_table.drop(columns=['D'])
    if 'alpha' in display_table.columns:
        display_table = display_table.drop(columns=['alpha'])
    
    # Format metrics to be more readable
    if has_node_metrics:
        display_table['fractal_participation'] = display_table['fractal_participation'].round(2)
        display_table['hubness'] = display_table['hubness'].round(2)
    
    display_table['FB'] = display_table['FB'].round(0).astype(int)
    display_table['wsjf_score'] = display_table['wsjf_score'].round(2)
    
    # Select and rename columns for display
    if has_node_metrics:
        columns_to_display = ['node', 'ux_label', 'percolation_role', 'wsjf_score', 'FB', 
                             'fractal_participation', 'hubness', 'suggested_action']
    else:
        columns_to_display = ['node', 'ux_label', 'percolation_role', 'wsjf_score', 'FB', 'suggested_action']
    
    # Filter to include only columns that exist in the dataframe
    columns_to_display = [col for col in columns_to_display if col in display_table.columns]
    display_table = display_table[columns_to_display]
    
    # Rename columns for better display
    column_names = {
        'node': 'Node',
        'ux_label': 'UX Label',
        'percolation_role': 'Role',
        'wsjf_score': 'WSJF',
        'FB': 'Fractal Betweenness',
        'fractal_participation': 'Fractal Participation',
        'hubness': 'Hubness',
        'suggested_action': 'Suggested Action'
    }
    display_table = display_table.rename(columns=column_names)
    
    # Show the table with tooltips for metrics
    st.dataframe(
        display_table,
        column_config={
            "WSJF": st.column_config.NumberColumn(
                "WSJF", 
                help="Weighted Shortest Job First score - combines frequency of friction with impact"
            ),
            "Fractal Betweenness": st.column_config.NumberColumn(
                "Fractal Betweenness", 
                help="Measures a node's importance in controlling information flow"
            ),
            "Fractal Participation": st.column_config.NumberColumn(
                "Fractal Participation", 
                help="How deeply a node participates in nested or self-similar structures"
            ),
            "Hubness": st.column_config.NumberColumn(
                "Hubness", 
                help="How important a node is in the scale-free structure of the network"
            ),
            "Role": st.column_config.TextColumn(
                "Role", 
                help="Node's structural role in the network (critical or standard)"
            ),
            "UX Label": st.column_config.TextColumn(
                "UX Label", 
                help="Classification based on network analysis"
            ),
        },
        use_container_width=True
    )
    
    # Add a rerun metrics button with confirmation
    if st.button("ðŸ”„ Rerun All Metrics"):
        confirmation = st.warning("This will recompute all metrics for this dataset, which may take some time. Continue?")
        if st.button("âœ… Confirm"):
            st.info("Starting metrics computation... This may take a while.")
            st.info("This would trigger a full metrics computation on the backend.")
            
            # Here we would call the backend metrics computation
            # For now, just simulate with a message
            import time
            with st.spinner("Computing metrics..."):
                time.sleep(2)
            st.success("Metrics computation complete! Refresh to see the results.")
    
    # Show detailed actions for critical nodes
    st.markdown("#### Detailed Recommendations")
    for _, row in filtered_table.head(3).iterrows():
        with st.expander(f"ðŸ’¡ {row['node']} - {row['ux_label']}"):
            st.markdown(f"**Suggested Action:** {row['suggested_action']}")
            st.markdown(f"**Why this matters:** This {row['percolation_role']} node has a WSJF score of {row['wsjf_score']:.2f} and Fractal Betweenness of {row['FB']:.0f}, making it a priority for UX improvements.")
            if has_node_metrics:
                st.markdown(f"**Node-level metrics:** Fractal Participation = {row['fractal_participation']:.2f}, Hubness = {row['hubness']:.2f}")
            st.markdown("**Implementation tip:** Consider conducting a dedicated user session to observe how users interact with this part of the application.")

def render_fb_vs_wsjf_chart(metrics_data):
    """
    Render a scatter plot of Fractal Betweenness vs WSJF Score.
    
    Args:
        metrics_data (Dict): Dictionary containing metrics data.
    """
    st.subheader("ðŸ“Š Fractal Betweenness vs WSJF Score", help="Compare structural importance vs friction severity")
    
    decision_table = metrics_data["decision_table"]
    
    if decision_table.empty:
        st.info("No data available for chart.")
        return
    
    # Create a copy for plotting
    plot_data = decision_table.copy()
    
    # Define high priority thresholds
    fb_threshold = plot_data['FB'].quantile(0.75)
    wsjf_threshold = plot_data['wsjf_score'].quantile(0.75)
    
    # Create quadrant labels
    plot_data['priority_quadrant'] = 'Standard'
    plot_data.loc[(plot_data['FB'] > fb_threshold) & (plot_data['wsjf_score'] > wsjf_threshold), 'priority_quadrant'] = 'High Priority'
    plot_data.loc[(plot_data['FB'] > fb_threshold) & (plot_data['wsjf_score'] <= wsjf_threshold), 'priority_quadrant'] = 'High Structural Risk'
    plot_data.loc[(plot_data['FB'] <= fb_threshold) & (plot_data['wsjf_score'] > wsjf_threshold), 'priority_quadrant'] = 'High Friction'
    
    # Define colors and sizes for the plot
    plot_data['color'] = plot_data['priority_quadrant'].map({
        'High Priority': 'red',
        'High Structural Risk': 'orange',
        'High Friction': 'yellow',
        'Standard': 'blue'
    })
    
    plot_data['size'] = 10
    plot_data.loc[plot_data['priority_quadrant'] == 'High Priority', 'size'] = 20
    
    # Use Plotly for better interactivity
    import plotly.express as px
    
    fig = px.scatter(
        plot_data, 
        x='wsjf_score', 
        y='FB',
        color='priority_quadrant',
        color_discrete_map={
            'High Priority': 'red',
            'High Structural Risk': 'orange',
            'High Friction': 'yellow',
            'Standard': 'blue'
        },
        size='size',
        size_max=20,
        hover_name='node',
        hover_data={
            'node': True,
            'wsjf_score': ':.2f',
            'FB': ':.0f',
            'priority_quadrant': True,
            'percolation_role': True,
            'ux_label': True,
            'size': False
        },
        labels={
            'wsjf_score': 'WSJF Score (UX Friction)',
            'FB': 'Fractal Betweenness (Structural Importance)',
            'priority_quadrant': 'Priority Category'
        },
        title='UX Priority Quadrants'
    )
    
    # Add quadrant lines
    fig.add_shape(
        type='line', x0=wsjf_threshold, y0=0, x1=wsjf_threshold, y1=plot_data['FB'].max(),
        line=dict(color='gray', width=1, dash='dash')
    )
    fig.add_shape(
        type='line', x0=0, y0=fb_threshold, x1=plot_data['wsjf_score'].max(), y1=fb_threshold,
        line=dict(color='gray', width=1, dash='dash')
    )
    
    # Add quadrant labels
    fig.add_annotation(
        x=plot_data['wsjf_score'].max() * 0.75, 
        y=plot_data['FB'].max() * 0.75,
        text="HIGH PRIORITY",
        showarrow=False,
        font=dict(color="red", size=14)
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="WSJF Score (UX Friction)",
        yaxis_title="Fractal Betweenness (Structural Importance)",
        legend_title="Priority Category",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation of the chart
    with st.expander("ðŸ“ How to interpret this chart"):
        st.markdown("""
        This chart plots each node by two critical metrics:
        
        - **X-axis: WSJF Score** - Measures user friction and UX priority
        - **Y-axis: Fractal Betweenness** - Measures structural importance in the network
        
        The **quadrants** indicate different priority levels:
        
        - **ðŸ”´ High Priority (top right)**: Both structurally important and causing friction - fix these first!
        - **ðŸŸ  High Structural Risk (bottom right)**: Critical for network structure but not currently causing high friction
        - **ðŸŸ¡ High Friction (top left)**: Causing significant UX issues but not structurally critical
        - **ðŸ”µ Standard (bottom left)**: Lower priority items
        
        Hover over points for detailed information about each node.
        """)

def render_priority_nodes_section(metrics_data):
    """
    Render a section highlighting top priority nodes needing attention.
    
    Args:
        metrics_data (Dict): Dictionary containing metrics data.
    """
    st.subheader("ðŸš© Top Priority Nodes", help="Nodes requiring immediate UX attention")
    
    decision_table = metrics_data["decision_table"]
    
    if decision_table.empty:
        st.info("No priority nodes data available.")
        return
    
    # Calculate combined priority score
    # Normalize FB to 0-1 range for fair comparison
    fb_max = decision_table['FB'].max()
    decision_table['priority_score'] = decision_table['wsjf_score'] + (decision_table['FB'] / fb_max)
    
    # Get top 5 priority nodes
    top_nodes = decision_table.sort_values('priority_score', ascending=False).head(5)
    
    # Create a visually prominent display
    for i, (_, row) in enumerate(top_nodes.iterrows()):
        with st.container():
            cols = st.columns([1, 4])
            
            # Priority number
            with cols[0]:
                st.markdown(f"""
                <div style="background-color:#f63366; color:white; width:40px; height:40px; 
                border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:20px; font-weight:bold;">
                {i+1}
                </div>
                """, unsafe_allow_html=True)
            
            # Node details
            with cols[1]:
                st.markdown(f"### {row['node']}")
                st.markdown(f"**UX Issue:** {row['ux_label']} ({row['percolation_role']} node)")
                st.markdown(f"**Action:** {row['suggested_action']}")
                
                # Progress bars for metrics
                cols2 = st.columns(2)
                with cols2[0]:
                    st.markdown(f"**WSJF Score:** {row['wsjf_score']:.2f}")
                    st.progress(min(1.0, row['wsjf_score']))
                with cols2[1]:
                    st.markdown(f"**FB Score:** {int(row['FB'])}")
                    st.progress(min(1.0, row['FB'] / fb_max))
            
            st.divider()
    
    # Add action button
    st.button("ðŸ“‹ Generate UX Priority Report", help="Generate a detailed report for the UX team")

def render_recurring_patterns(metrics_data):
    """
    Render the recurring patterns analysis.
    
    Args:
        metrics_data (Dict): Dictionary containing metrics data.
    """
    # Check if recurring patterns data is available
    network_metrics = metrics_data["network_metrics"]
    recurring_patterns = network_metrics.get("recurring_patterns", None)
    
    with st.expander("ðŸ”„ Recurring Pattern Analysis", expanded=False):
        if recurring_patterns is None:
            st.info("Recurring patterns data not available for this dataset.")
            return
        
        st.markdown("### Detected Recurring Patterns")
        st.markdown("""
        These are repeating subgraph patterns that may indicate behavioral traps - users bouncing or stuck in loops.
        Redesigning or eliminating these patterns can improve user flow.
        """)
        
        # Display patterns
        for i, pattern in enumerate(recurring_patterns[:5]):  # Show top 5 patterns
            st.markdown(f"**Pattern {i+1}**")
            
            # Format the pattern path
            path_parts = []
            for node in pattern:
                path_parts.append(f"{node}")
            
            path_str = " â†’ ".join(path_parts)
            st.markdown(f"- Path: {path_str}")
            
            # Add a separator between patterns
            if i < len(recurring_patterns[:5]) - 1:
                st.markdown("---")
        
        # If there are more patterns, show a message
        if len(recurring_patterns) > 5:
            st.info(f"{len(recurring_patterns) - 5} more patterns detected but not shown.")

def render_percolation_collapse(metrics_data):
    """
    Render the percolation collapse visualization.
    
    Args:
        metrics_data (Dict): Dictionary containing metrics data.
    """
    with st.expander("ðŸ’¥ Network Stability Analysis", expanded=False):
        network_metrics = metrics_data["network_metrics"]
        
        # Check if we have percolation data in network_metrics
        if "percolation_data" not in network_metrics:
            st.info("Percolation analysis data not available for this dataset.")
            
            # Show simplified info based on threshold only
            threshold = None
            for _, row in metrics_data["final_report"].iterrows():
                if row["metric"] == "Percolation Threshold":
                    threshold = row["value"]
            
            if threshold is not None:
                try:
                    # Convert to float for formatting
                    threshold_float = float(threshold)
                    st.markdown(f"""
                    ### Network Stability Summary
                    
                    This network has a percolation threshold of **{threshold_float:.2f}**.
                    
                    **Interpretation:**
                    - Values closer to 0 indicate a more fragile network structure
                    - Values closer to 1 indicate a more robust network structure
                    
                    **UX Implications:**
                    - Low threshold (<0.3): High risk of user flow breakdowns if key pages have issues
                    - Medium threshold (0.3-0.6): Moderate resilience, but still vulnerable at key nodes
                    - High threshold (>0.6): Robust structure with multiple alternative paths
                    """)
                except (ValueError, TypeError):
                    # Handle case where conversion to float fails
                    st.markdown(f"""
                    ### Network Stability Summary
                    
                    This network has a percolation threshold of **{threshold}**.
                    
                    **Interpretation:**
                    - Values closer to 0 indicate a more fragile network structure
                    - Values closer to 1 indicate a more robust network structure
                    
                    **UX Implications:**
                    - Low threshold (<0.3): High risk of user flow breakdowns if key pages have issues
                    - Medium threshold (0.3-0.6): Moderate resilience, but still vulnerable at key nodes
                    - High threshold (>0.6): Robust structure with multiple alternative paths
                    """)
            return
        
        # If we have detailed percolation data, create a plot
        percolation_data = network_metrics["percolation_data"]
        
        # Create a DataFrame for the plot
        data = pd.DataFrame(percolation_data)
        
        # Plot the percolation collapse
        st.markdown("### Network Percolation Analysis")
        st.markdown("""
        This chart shows how the network collapses as high-importance nodes are removed.
        A rapid drop indicates vulnerability to targeted node failures.
        """)
        
        # Create the chart
        st.line_chart(
            data,
            x="nodes_removed_pct",
            y="largest_component_pct",
            height=300
        )
        
        st.markdown("""
        **Interpretation:**
        - The x-axis shows the percentage of nodes removed (highest importance first)
        - The y-axis shows the size of the largest connected component
        - A steep drop indicates a fragile network structure
        """)

def render_recommendations_panel(metrics_data):
    """
    Render recommendations from the final report.
    
    Args:
        metrics_data (Dict): Dictionary containing metrics data.
    """
    with st.expander("ðŸ“‹ Key Recommendations", expanded=True):
        final_report = metrics_data["final_report"]
        decision_table = metrics_data["decision_table"]
        
        if final_report.empty or decision_table.empty:
            st.info("No recommendation data available for this dataset.")
            return
        
        # Get top metrics for recommendations
        fractal_dimension = None
        power_law_alpha = None
        
        for _, row in final_report.iterrows():
            if row["metric"] == "Fractal Dimension":
                fractal_dimension = row["value"]
            elif row["metric"] == "Power Law Alpha":
                power_law_alpha = row["value"]
        
        # Get top 3 critical nodes
        top_critical_nodes = decision_table[decision_table["percolation_role"] == "critical"].sort_values("wsjf_score", ascending=False).head(3)
        
        # Generate recommendations based on metrics
        st.markdown("### Key Recommendations")
        
        # Structure-based recommendations
        if fractal_dimension is not None:
            st.markdown("#### Structure Recommendations")
            
            # Convert to float if possible
            try:
                fractal_dimension = float(fractal_dimension)
            except (ValueError, TypeError):
                # If conversion fails, use default threshold
                fractal_dimension = 1.5
                
            if fractal_dimension <= 1.2:
                st.markdown("- **Linear Structure Detected**: Users follow a strict sequence with few alternatives.")
                st.markdown("- **Recommendation**: Add shortcuts and alternative paths to increase flexibility.")
            elif fractal_dimension <= 1.7:
                st.markdown("- **Tree-like Structure Detected**: Hierarchical navigation with some branching.")
                st.markdown("- **Recommendation**: Consider connecting related branches to reduce path length.")
            else:
                st.markdown("- **Complex Structure Detected**: Many interconnected paths may confuse users.")
                st.markdown("- **Recommendation**: Simplify navigation by reducing redundant connections.")
        
        # Hierarchy-based recommendations
        if power_law_alpha is not None:
            st.markdown("#### Hub Node Recommendations")
            
            # Convert to float if possible
            try:
                power_law_alpha = float(power_law_alpha)
            except (ValueError, TypeError):
                # If conversion fails, use default threshold
                power_law_alpha = 2.2
                
            if power_law_alpha < 2.0:
                st.markdown("- **Strong Hub Structure Detected**: A few nodes dominate the flow.")
                st.markdown("- **Recommendation**: Strengthen alternative paths to reduce dependency on hub nodes.")
            elif power_law_alpha < 2.5:
                st.markdown("- **Moderate Hub Structure**: Several important nodes with moderate influence.")
                st.markdown("- **Recommendation**: Balance traffic across key pages.")
            else:
                st.markdown("- **Distributed Structure**: Traffic spread across many nodes fairly evenly.")
                st.markdown("- **Recommendation**: Consider creating clearer entry/exit points for major user flows.")
        
        # Node-specific recommendations
        if not top_critical_nodes.empty:
            st.markdown("#### Critical Node Recommendations")
            for i, (_, node) in enumerate(top_critical_nodes.iterrows()):
                st.markdown(f"- **{node['node']}**: {node['suggested_action']}")

def render_glossary_sidebar():
    """
    Render a metrics glossary in the sidebar.
    """
    with st.sidebar.expander("ðŸ“– Metrics Glossary", expanded=False):
        st.markdown("### Global Network Metrics")
        st.markdown("""
        - **Fractal Dimension (D)**: Measures how space-filling the graph is. Higher values (>1.5) indicate more complex user navigation paths with nested or recursive patterns.
        
        - **Power-Law Alpha (Î±)**: Exponent of degree distribution. Lower values (<2.5) indicate stronger hierarchy with vulnerable hub nodes. Higher values (>3.0) suggest more uniform structure.
        
        - **Percolation Threshold**: Fraction of top nodes that can be removed before network collapses. Lower values indicate a more fragile structure dependent on few critical nodes.
        """)
        
        st.markdown("### Node-Level Metrics")
        st.markdown("""
        - **Fractal Participation**: How deeply a node participates in nested or self-similar structures. Higher values indicate nodes that are part of complex, redundant, or nested patterns.
        
        - **Hubness**: How important a node is in the scale-free structure of the network. Higher values indicate hub nodes with more influence or centrality.
        
        - **FB (Fractal Betweenness)**: Measures a node's importance in controlling information flow, especially through repetitive patterns. Higher values indicate critical chokepoints in user flows.
        
        - **WSJF Score**: Weighted Shortest Job First score - combines frequency of friction with impact. Higher values indicate more pressing UX issues to fix.
        """)
        
        st.markdown("### Roles & Labels")
        st.markdown("""
        - **Critical Node**: Removing this node would severely disrupt network connectivity.
        
        - **Standard Node**: Less critical for overall network stability.
        
        - **UX Labels**: Classifications like "redundant bottleneck", "complex hub", etc. that suggest specific UX improvements.
        """)

def advanced_metrics_tab(metrics_data):
    """
    Render the Advanced Metrics tab content.
    
    Args:
        metrics_data (Dict): Dictionary containing metrics data.
    """
    st.title("TeloMesh Advanced Metrics")
    
    # Add tab description
    st.markdown("""
    This dashboard provides advanced network metrics to help identify UX improvement opportunities
    based on structural analysis of user flows and friction points.
    """)
    
    # Render the top metrics cards
    render_top_metrics(metrics_data)
    
    # Add space between sections
    st.markdown("---")
    
    # Render priority nodes section
    render_priority_nodes_section(metrics_data)
    
    # Add space between sections
    st.markdown("---")
    
    # Render the decision table
    render_decision_table(metrics_data)
    
    # Add space between sections
    st.markdown("---")
    
    # Render the FB vs WSJF scatter plot
    render_fb_vs_wsjf_chart(metrics_data)
    
    # Add space between sections
    st.markdown("---")
    
    # Render the recommendations panel
    render_recommendations_panel(metrics_data)
    
    # Render recurring patterns
    render_recurring_patterns(metrics_data)
    
    # Render percolation collapse
    render_percolation_collapse(metrics_data)

def main():
    """
    Main function to run the dashboard.
    """
    # Set page config
    st.set_page_config(
        page_title="TeloMesh Analytics",
        page_icon="ðŸ•¸ï¸",
        layout="wide"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .stProgress > div > div {
        background-color: #f63366;
    }
    .stExpander {
        border-left: 1px solid #f0f2f6;
        padding-left: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create sidebar
    st.sidebar.title("TeloMesh Analytics")
    st.sidebar.markdown("---")
    
    # Add dataset selector to sidebar
    datasets = get_available_datasets()
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        datasets,
        index=0,
        key="dataset_selector"
    )
    
    # Add glossary to sidebar
    render_glossary_sidebar()
    
    # Add sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info(
        "TeloMesh analyzes user flows to identify UX improvement opportunities."
    )
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Overview", "Advanced Metrics", "Settings"])
    
    # Load metrics data
    try:
        metrics_data = load_metrics(selected_dataset)
        
        # Render tab content
        with tab1:
            st.title("Overview")
            st.markdown("Basic overview of the selected dataset.")
            st.info("Switch to the Advanced Metrics tab for detailed UX analytics.")
        
        with tab2:
            advanced_metrics_tab(metrics_data)
        
        with tab3:
            st.title("Settings")
            st.markdown("Configure analysis parameters and visualization options.")
            
            # Add dataset info
            st.subheader("Dataset Information")
            dataset_info = get_dataset_info(selected_dataset)
            if dataset_info:
                st.json(dataset_info)
            else:
                st.info("No dataset information available.")
    
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        st.info("Please check if the selected dataset exists and has valid metric files.")

def get_available_datasets():
    """
    Get a list of available datasets in the outputs directory.
    
    Returns:
        list: List of dataset names
    """
    try:
        outputs_dir = Path("outputs")
        if not outputs_dir.exists():
            return ["default"]
        
        # Get all subdirectories in the outputs directory that contain valid data
        datasets = [d.name for d in outputs_dir.iterdir() if d.is_dir() and is_valid_dataset(d.name)]
        
        # If no datasets found, return default
        if not datasets:
            return ["default"]
        
        # Sort datasets alphabetically
        datasets.sort()
        
        return datasets
    except Exception as e:
        logger.error(f"Error discovering datasets: {str(e)}")
        return ["default"]

def get_dataset_info(dataset_name):
    """
    Get information about a dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        dict: Dictionary with dataset information
    """
    try:
        info_path = Path(f"outputs/{dataset_name}/dataset_info.json")
        if not info_path.exists():
            return {}
        
        with open(info_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading dataset info: {str(e)}")
        return {}

# No replacement needed - we'll use the original is_valid_dataset function at line 262

def load_metrics(dataset=None):
    """
    Load all metrics data for the dashboard.
    
    Args:
        dataset (str, optional): Dataset name. Defaults to None.
        
    Returns:
        Dict: Dictionary containing all metrics data
    """
    try:
        # Create metrics data dictionary
        metrics_data = {}
        
        # Load node metrics data
        node_metrics_path = Path(f"outputs/{dataset}/node_metrics.csv")
        if node_metrics_path.exists():
            metrics_data['node_metrics'] = pd.read_csv(node_metrics_path)
        else:
            logger.warning(f"Node metrics file not found: {node_metrics_path}")
            metrics_data['node_metrics'] = pd.DataFrame()
        
        # Load decision table
        decision_table_path = Path(f"outputs/{dataset}/decision_table.csv")
        if decision_table_path.exists():
            metrics_data['decision_table'] = pd.read_csv(decision_table_path)
        else:
            logger.warning(f"Decision table file not found: {decision_table_path}")
            metrics_data['decision_table'] = pd.DataFrame()
        
        # Load final report
        final_report_path = Path(f"outputs/{dataset}/final_report.csv")
        if final_report_path.exists():
            metrics_data['final_report'] = pd.read_csv(final_report_path)
        else:
            logger.warning(f"Final report file not found: {final_report_path}")
            metrics_data['final_report'] = pd.DataFrame()
        
        # Load recurring patterns
        patterns_path = Path(f"outputs/{dataset}/recurring_patterns.json")
        if patterns_path.exists():
            with open(patterns_path, 'r') as f:
                metrics_data['recurring_patterns'] = json.load(f)
        else:
            logger.warning(f"Recurring patterns file not found: {patterns_path}")
            metrics_data['recurring_patterns'] = {}
        
        return metrics_data
    except Exception as e:
        logger.error(f"Error loading metrics data: {str(e)}")
        return {}

if __name__ == "__main__":
    main()
