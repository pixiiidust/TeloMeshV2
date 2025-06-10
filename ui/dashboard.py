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
from typing import Tuple, Dict
import time
import logging
import traceback
import base64
from pathlib import Path
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# CURSOR RULE: PM-FACING DASHBOARD CONSTRUCTION
# - Build a PM-intuitive, dark-themed Streamlit dashboard
# - Use pre-computed files only (no recomputation in dashboard)
#   - Friction Table: Sortable by WSJF_Friction_Score, filterable by page/event
#   - Flow Summary: Shows fragile flows with ≥2 high-friction points
#   - Graph Heatmap: Dark-themed node-link graph with top 10%/20% highlighting
#   - Click-to-Drilldown: Shows event breakdown, users lost, incoming/outgoing links
#   - Tooltips: Explain metrics in PM-friendly terms

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
# 2. ✅ Tooltip icon (ℹ️) should be visible (white on dark)
# 3. ✅ Tooltip bubble should show with dark background and white text
# 4. ✅ No white-on-white or missing content
# 5. ✅ Tooltip border should be visible and styled (optional accent)
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
            page_icon="🔍",
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
        "WSJF_Friction_Score": "WSJF Friction Score is calculated as exit_rate × betweenness, prioritizing high-impact friction points.",
        "wsjf": "WSJF Friction Score is calculated as exit_rate × betweenness, prioritizing high-impact friction points.",
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
    st.header("🔥 Friction Table")
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
                    f'<div class="download-container">{get_csv_download_link(filtered_df, "friction_points", "📥 Export Filtered Data")}</div>',
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
    
    - funnel_stages: left→right (vertical alignment → x = stage)
    - friction_levels: top→bottom (horizontal alignment → y = friction)
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
            # Subset controls X-axis (left→right)
            scale_x = scale * 0.6
            scale_y = scale * 0.6
        else:
            # Subset controls Y-axis (top→bottom)
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
            print(f"[Layout: {layout_type}] X range: {min(xs):.1f} → {max(xs):.1f}")
            print(f"[Layout: {layout_type}] Y range: {min(ys):.1f} → {max(ys):.1f}")

        return pos

    except Exception as e:
        print(f"⚠️ Layout fallback due to error: {e}")
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
        st.markdown("### 📊 Node Colors")
        st.markdown("""
        - <span style="display:inline-block;width:15px;height:15px;border-radius:50%;background-color:#F87171;"></span> **Red**: Top 10% friction (highest WSJF scores)
        - <span style="display:inline-block;width:15px;height:15px;border-radius:50%;background-color:#FBBF24;"></span> **Yellow**: Top 20% friction
        - <span style="display:inline-block;width:15px;height:15px;border-radius:50%;background-color:#A3E635;"></span> **Green**: Top 50% friction  
        - <span style="display:inline-block;width:15px;height:15px;border-radius:50%;background-color:#94A3B8;"></span> **Gray**: Lower friction
        """, unsafe_allow_html=True)
    
    with col2:
        if not physics_enabled:
            st.markdown("### 📐 Layout Arrangement")
            
            if layout_type == "friction_levels":
                st.markdown("""
                - **Top Layer**: Highest friction (urgent)
                - **Middle Layers**: Medium friction (monitor)  
                - **Lowest Layer**: Low friction (stable)
                    """)
            elif layout_type == "funnel_stages":
                st.markdown("""
                **User Funnel Stages:**
                - 🚪 **Left**: Entry points
                - ⚙️ **Middle**: Conversion steps
                - 🚪 **Right**: Exit points
                    """)
            elif layout_type == "betweenness_tiers":
                st.markdown("""
                **Journey Centrality Structure:**
                - 📍 **Edges**: Peripheral nodes
                - 🎯 **Center**: Hub nodes
                    """)
        else:
            st.markdown("### ℹ️ Physics Mode")
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
    st.header("🌐 User Journey Graph", help="Visualizes user journeys with friction nodes and their connecting flows")
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
                "friction_levels": "📊 Nodes arranged by friction level (problems at top)",
                "funnel_stages": "🔄 Nodes arranged by user funnel stages (entry → exit)",
                "betweenness_tiers": "🎯 Nodes arranged by journey centrality (hubs centered)"
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
                f'<div class="download-container">{get_html_download_link(html_content, filename, "📥 Export Graph as HTML")}</div>',
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
                f'<div class="download-container">{get_csv_download_link(filtered_flow_df, "fragile_flows", "📥 Export Filtered Flows")}</div>',
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
                
                # Format as page → event → page → event...
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
                
                path_str = " → ".join(path_parts)
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
                display_df["is_chokepoint"] = display_df["is_chokepoint"].apply(lambda x: "✅" if x == 1 else "")
                
                # Reorder and rename columns
                display_df = display_df[["step_index", "page", "event", "WSJF_Friction_Score", "is_chokepoint"]]
                display_df.columns = ["Step", "Page", "Event", "WSJF Score", "Chokepoint"]
                
                st.dataframe(display_df)
                
                # Add individual flow export option
                st.markdown(
                    f'<div class="download-container">{get_csv_download_link(session_data, f"flow_{session_id}", f"📥 Export Flow {session_id}")}</div>',
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
        - fractal_dimension: Fractal dimension of the graph
        - power_law_alpha: Power law exponent of the graph
        - percolation_threshold: Percolation threshold of the graph
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
        final_report_json_path = dataset_dir / "final_report.json"
        recurring_patterns_path = dataset_dir / "recurring_patterns.json"
        recurring_exit_paths_path = dataset_dir / "recurring_exit_paths.json"
        
        # Initialize metrics data dictionary
        metrics_data = {}
        
        # Load decision table
        if decision_table_path.exists():
            metrics_data["decision_table"] = pd.read_csv(decision_table_path)
        else:
            st.warning(f"Decision table not found: {decision_table_path}")
            # Create empty DataFrame with expected columns
            metrics_data["decision_table"] = pd.DataFrame(columns=[
                "node", "FB", "percolation_role", 
                "wsjf_score", "ux_label", "suggested_action"
            ])
        
        # Load final report JSON for network metrics
        network_metrics = {}
        if final_report_json_path.exists():
            try:
                with open(final_report_json_path, 'r') as f:
                    network_metrics = json.load(f)
                    
                    # Extract global metrics
                    metrics_data["fractal_dimension"] = network_metrics.get("fractal_dimension", 0)
                    metrics_data["power_law_alpha"] = network_metrics.get("power_law_alpha", 0)
                    metrics_data["percolation_threshold"] = network_metrics.get("percolation_threshold", 0)
                    metrics_data["clustering_coefficient"] = network_metrics.get("clustering_coefficient", 0)
            except Exception as e:
                logger.error(f"Error loading network metrics JSON: {str(e)}")
                st.warning(f"Error loading network metrics: {str(e)}")
                
                # Set default values for global metrics
                metrics_data["fractal_dimension"] = 0
                metrics_data["power_law_alpha"] = 0
                metrics_data["percolation_threshold"] = 0
                metrics_data["clustering_coefficient"] = 0
        else:
            st.warning(f"Final report JSON not found: {final_report_json_path}")
            
            # Set default values for global metrics
            metrics_data["fractal_dimension"] = 0
            metrics_data["power_law_alpha"] = 0
            metrics_data["percolation_threshold"] = 0
            metrics_data["clustering_coefficient"] = 0
        
        # Try to load recurring patterns if available
        if recurring_patterns_path.exists():
            try:
                with open(recurring_patterns_path, 'r') as f:
                    network_metrics["recurring_patterns"] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading recurring patterns: {str(e)}")
        
        # Try to load recurring exit paths if available
        if recurring_exit_paths_path.exists():
            try:
                with open(recurring_exit_paths_path, 'r') as f:
                    network_metrics["recurring_exit_paths"] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading recurring exit paths: {str(e)}")
        
        # Count critical nodes (nodes with percolation_role = "critical")
        try:
            network_metrics["critical_nodes_count"] = len(metrics_data["decision_table"][
                metrics_data["decision_table"]["percolation_role"] == "critical"
            ])
        except Exception as e:
            logger.error(f"Error counting critical nodes: {str(e)}")
            network_metrics["critical_nodes_count"] = 0
        
        # Add network metrics to the data dictionary
        metrics_data["network_metrics"] = network_metrics
        
        return metrics_data
    
    except Exception as e:
        st.error(f"Error loading advanced metrics: {str(e)}")
        logger.error(f"Error loading advanced metrics: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return empty data structures on error
        return {
            "decision_table": pd.DataFrame(columns=[
                "node", "FB", "percolation_role", 
                "wsjf_score", "ux_label", "suggested_action"
            ]),
            "fractal_dimension": 0,
            "power_law_alpha": 0,
            "percolation_threshold": 0,
            "clustering_coefficient": 0,
            "network_metrics": {}
        }

def render_top_metrics(metrics_data):
    """
    Render the top metrics panel showing global network metrics with diagnostic badges.
    
    Args:
        metrics_data (Dict): Dictionary containing metrics data.
    """
    st.subheader("📊 Network Structure Metrics", help="Key metrics describing the graph structure")
    
    # Extract metrics from the data
    if not metrics_data:
        st.info("No network metrics available for this dataset.")
        return
    
    # Get global network metrics
    fractal_dim = metrics_data.get("fractal_dimension", 0)
    power_law = metrics_data.get("power_law_alpha", 0)
    percolation = metrics_data.get("percolation_threshold", 0)
    critical_nodes = metrics_data.get("network_metrics", {}).get("critical_nodes_count", 0)
    
    # Display global metrics in columns with diagnostic badges
    cols = st.columns(3)
    
    with cols[0]:
        st.metric(
            label="Fractal Dimension (D)", 
            value=f"{fractal_dim:.2f}",
            help="Measures how space-filling the graph is. Higher values indicate more complex user navigation paths."
        )
        
        # Visual indicator based on thresholds
        if fractal_dim > 2.6 or fractal_dim <= 0.8:
            st.markdown("❌ Severe Risk: Extreme complexity or oversimplification")
        elif 2.2 < fractal_dim <= 2.6:
            st.markdown("⚠️ Needs Review: Moderately complex structure")
        else:
            st.markdown("✅ Healthy: Balanced complexity")
    
    with cols[1]:
        st.metric(
            label="Power-Law Alpha (α)", 
            value=f"{power_law:.2f}",
            help="Exponent of degree distribution. Lower values indicate stronger hierarchy with hub nodes."
        )
        
        # Visual indicator based on thresholds
        if power_law < 1.5 or power_law > 3.5:
            st.markdown("❌ Severe Risk: Extreme hierarchy or lack of structure")
        elif 1.5 <= power_law < 1.8 or power_law > 2.6:
            st.markdown("⚠️ Needs Review: Moderate structural concern")
        else:
            st.markdown("✅ Healthy: Balanced scale-free properties")
    
    with cols[2]:
        st.metric(
            label="Percolation Threshold", 
            value=f"{percolation:.2f}",
            help="Fraction of top nodes that can be removed before network collapses."
        )
        
        # Visual indicator based on thresholds
        if percolation < 0.3:
            st.markdown(f"❌ Severe Risk: Fragile ({critical_nodes} critical nodes)")
        elif 0.3 <= percolation <= 0.5:
            st.markdown(f"⚠️ Needs Review: Moderate ({critical_nodes} critical nodes)")
        else:
            st.markdown(f"✅ Healthy: Robust ({critical_nodes} critical nodes)")
            
    # Add explanatory text
    st.markdown("""
    <div style="font-size: 0.9em; color: #888;">
    These metrics help identify structural issues in your user flow graph:
    <ul>
        <li><b>Fractal Dimension (D):</b> 1.0-2.2 is optimal. Higher values indicate overly complex paths.</li>
        <li><b>Power-Law Alpha (α):</b> 1.8-2.6 is optimal. Lower values indicate vulnerable hub-dependent structure.</li>
        <li><b>Percolation Threshold:</b> >0.5 is optimal. Lower values indicate network fragility.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def render_decision_table(metrics_data):
    """
    Render the decision table explorer with filters.
    
    Args:
        metrics_data (Dict): Dictionary containing metrics data.
    """
    st.subheader("🧠 Decision Table & Recommendations", help="Explore nodes by role and impact metrics")
    
    if not metrics_data or "decision_table" not in metrics_data:
        st.info("No decision table data available for this dataset.")
        return
    
    decision_table = metrics_data["decision_table"]
    
    if decision_table.empty:
        st.info("Decision table is empty for this dataset.")
        return
    
    # Display global metrics reference
    st.write(f"""
    **Global Network Metrics:** 
    - Fractal Dimension (D): {metrics_data.get("fractal_dimension", 0):.2f}
    - Power-Law Alpha (α): {metrics_data.get("power_law_alpha", 0):.2f}
    - Percolation Threshold: {metrics_data.get("percolation_threshold", 0):.2f}
        """)
    
    # Create filters
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Filter by percolation role
        role_filter = st.selectbox(
            "Filter by role:",
            ["All"] + sorted(decision_table["percolation_role"].unique().tolist()),
            key="role_filter"
        )
    
    with col2:
        # Filter by minimum FB score
        min_fb = decision_table["FB"].min() if not decision_table.empty else 0
        max_fb = decision_table["FB"].max() if not decision_table.empty else 100
        
        fb_threshold = st.slider(
            "Minimum FB score:",
            min_value=float(min_fb),
            max_value=float(max_fb),
            value=float(min_fb),
            key="fb_threshold"
        )
    
    # Apply filters
    filtered_table = decision_table.copy()
    
    # Remove global metrics columns if they exist
    if "D" in filtered_table.columns:
        filtered_table = filtered_table.drop(columns=["D"])
    if "alpha" in filtered_table.columns:
        filtered_table = filtered_table.drop(columns=["alpha"])
    
    # Apply role filter
    if role_filter != "All":
        filtered_table = filtered_table[filtered_table["percolation_role"] == role_filter]
    
    # Apply FB threshold filter
    filtered_table = filtered_table[filtered_table["FB"] >= fb_threshold]
    
    # Format the FB values
    filtered_table["FB"] = filtered_table["FB"].apply(lambda x: f"{x:.3f}")
    
    # Sort by FB score descending
    filtered_table = filtered_table.sort_values("FB", ascending=False)
    
    # Add UX meaning explanation
    st.write("### Node-Level Metrics")
    st.write("""
    This table shows **node-level metrics** with UX recommendations specific to each page.
    The global metrics (D, α) are shown above for reference, but apply to the entire network.
    
    - **FB**: Fractal Betweenness - measures importance in recurring patterns
    - **Role**: Critical nodes can collapse the network if removed
    - **WSJF**: Friction score (exit rate × structural importance)
        """)
    
    # Display the table
    st.dataframe(filtered_table, use_container_width=True)
    
    # Add download button for the table
    st.download_button(
        label="Download Table as CSV",
        data=filtered_table.to_csv(index=False).encode('utf-8'),
        file_name="decision_table.csv",
        mime="text/csv",
    )

def render_fb_vs_wsjf_chart(metrics_data):
    """
    Render a simple interactive FB vs WSJF scatter plot with proper scaling.
    
    Args:
        metrics_data (Dict): Dictionary containing metrics data.
    """
    st.subheader("📊 FB vs WSJF Priority Matrix", help="Identify high-priority nodes based on structural importance and friction")
    
    if not metrics_data or "decision_table" not in metrics_data:
        st.info("No decision table data available for this dataset.")
        return
    
    decision_table = metrics_data["decision_table"]
    
    if decision_table.empty:
        st.info("Decision table is empty for this dataset.")
        return
    
    # Make a copy of the data
    chart_data = decision_table.copy()
    
    # Ensure FB and WSJF are numeric
    chart_data["FB"] = pd.to_numeric(chart_data["FB"], errors="coerce")
    chart_data["wsjf_score"] = pd.to_numeric(chart_data["wsjf_score"], errors="coerce")
    
    # Drop rows with missing values
    chart_data = chart_data.dropna(subset=["FB", "wsjf_score"])
    
    # If still empty after cleaning, return
    if chart_data.empty:
        st.warning("No valid data points for FB vs WSJF chart.")
        return
    
    # Create simpler titles for the axes
    x_axis_title = "Structural Importance (FB)"
    y_axis_title = "User Friction (WSJF)"
    
    # Options for chart customization
    with st.expander("Chart Options", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Let user select between different scaling options
            scale_option = st.radio(
                "Scale Type:",
                ["Actual Range", "Full Range (0-1)", "Normalized"],
                index=2
            )
            
            log_scale_x = st.checkbox("Log scale for FB", value=False)
            log_scale_y = st.checkbox("Log scale for WSJF", value=False)
        
        with col2:
            # Let user select different quadrant calculation methods
            quadrant_method = st.radio(
                "Quadrant Lines:",
                ["Median", "Mean", "Fixed (0.75)"],
                index=0
            )
            
            # Let user choose whether to emphasize critical points
            emphasize_critical = st.checkbox("Emphasize Critical Points", value=True)
            
        with col3:
            # Number of points to label
            num_labels = st.slider("Top Points to Label:", 3, 10, 5)
            
            # Let user select label mode
            label_mode = st.radio(
                "Label:",
                ["Top FB & WSJF", "Critical Only", "All"],
                index=0
            )
    
    # Create a simple interactive scatter plot with Plotly
    import plotly.express as px
    import numpy as np
    
    # Determine the actual ranges of the data
    fb_min = chart_data["FB"].min()
    fb_max = chart_data["FB"].max()
    wsjf_min = chart_data["wsjf_score"].min()
    wsjf_max = chart_data["wsjf_score"].max()
    
    # Calculate thresholds for quadrants based on selected method
    if quadrant_method == "Median":
        fb_threshold = chart_data["FB"].median()
        wsjf_threshold = chart_data["wsjf_score"].median()
    elif quadrant_method == "Mean":
        fb_threshold = chart_data["FB"].mean()
        wsjf_threshold = chart_data["wsjf_score"].mean()
    else:  # Fixed
        fb_threshold = 0.75  # Fixed value that makes more sense for FB
        wsjf_threshold = chart_data["wsjf_score"].median()  # Keep median for WSJF
    
    # Create normalized versions of the metrics for better visualization
    # Min-max scaling to spread the points more evenly
    chart_data["FB_norm"] = (chart_data["FB"] - fb_min) / (fb_max - fb_min)
    chart_data["wsjf_norm"] = (chart_data["wsjf_score"] - wsjf_min) / (wsjf_max - wsjf_min)
    
    # Choose which columns to use based on scale option
    if scale_option == "Normalized":
        x_col = "FB_norm"
        y_col = "wsjf_norm"
        fb_threshold_norm = (fb_threshold - fb_min) / (fb_max - fb_min)
        wsjf_threshold_norm = (wsjf_threshold - wsjf_min) / (wsjf_max - wsjf_min)
    else:
        x_col = "FB"
        y_col = "wsjf_score"
        fb_threshold_norm = fb_threshold
        wsjf_threshold_norm = wsjf_threshold
    
    # Add quadrant labels to the data
    chart_data["quadrant"] = "Low Priority"
    chart_data.loc[(chart_data[x_col] >= fb_threshold_norm) & (chart_data[y_col] >= wsjf_threshold_norm), "quadrant"] = "High Priority"
    chart_data.loc[(chart_data[x_col] >= fb_threshold_norm) & (chart_data[y_col] < wsjf_threshold_norm), "quadrant"] = "Structural Only"
    chart_data.loc[(chart_data[x_col] < fb_threshold_norm) & (chart_data[y_col] >= wsjf_threshold_norm), "quadrant"] = "User Friction Only"
    
    # Make points larger for emphasis
    if emphasize_critical:
        point_sizes = [12 if role == "critical" else 8 for role in chart_data["percolation_role"]]
        point_symbols = ["circle" if role == "critical" else "diamond" for role in chart_data["percolation_role"]]
    else:
        point_sizes = [10] * len(chart_data)
        point_symbols = ["circle"] * len(chart_data)
    
    chart_data["size"] = point_sizes
    chart_data["symbol"] = point_symbols
    
    # Color by quadrant instead of percolation role for better visual separation
    color_map = {
        "High Priority": "#F87171",        # Red
        "User Friction Only": "#FBBF24",   # Yellow
        "Structural Only": "#60A5FA",      # Blue
        "Low Priority": "#94A3B8"          # Gray
    }
    
    # Create the scatter plot
    fig = px.scatter(
        chart_data, 
        x=x_col, 
        y=y_col,
        color="quadrant",
        color_discrete_map=color_map,
        hover_name="node",
        hover_data={
            "FB": ":.3f",
            "wsjf_score": ":.3f",
            "percolation_role": True,
            "ux_label": True
        },
        title="Structural Importance vs User Friction",
        labels={
            x_col: x_axis_title,
            y_col: y_axis_title,
            "quadrant": "Priority"
        },
        height=600,
        symbol="symbol",
        size="size",
        size_max=15
    )
    
    # Determine which points to label
    if label_mode == "Critical Only":
        nodes_to_label = chart_data[chart_data["percolation_role"] == "critical"]
    elif label_mode == "All":
        nodes_to_label = chart_data
    else:  # Top FB & WSJF
        top_fb = chart_data.nlargest(num_labels, "FB")
        top_wsjf = chart_data.nlargest(num_labels, "wsjf_score")
        nodes_to_label = pd.concat([top_fb, top_wsjf]).drop_duplicates()
    
    # Add text labels
    for _, row in nodes_to_label.iterrows():
        fig.add_annotation(
            x=row[x_col],
            y=row[y_col],
            text=row["node"],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="#888",
            font=dict(size=10),
            ax=20,
            ay=-20
        )
    
    # Add quadrant lines
    fig.add_shape(
        type="line",
        x0=fb_threshold_norm,
        y0=0 if scale_option == "Full Range (0-1)" else (0 if y_col == "wsjf_norm" else wsjf_min),
        x1=fb_threshold_norm,
        y1=1 if scale_option == "Full Range (0-1)" else (1 if y_col == "wsjf_norm" else wsjf_max),
        line=dict(color="rgba(100, 100, 100, 0.5)", width=1, dash="dash"),
    )
    
    fig.add_shape(
        type="line",
        x0=0 if scale_option == "Full Range (0-1)" else (0 if x_col == "FB_norm" else fb_min),
        y0=wsjf_threshold_norm,
        x1=1 if scale_option == "Full Range (0-1)" else (1 if x_col == "FB_norm" else fb_max),
        y1=wsjf_threshold_norm,
        line=dict(color="rgba(100, 100, 100, 0.5)", width=1, dash="dash"),
    )
    
    # Calculate optimal positions for quadrant labels
    if scale_option == "Full Range (0-1)":
        x_low = fb_threshold_norm / 2
        x_high = (1 + fb_threshold_norm) / 2
        y_low = wsjf_threshold_norm / 2
        y_high = (1 + wsjf_threshold_norm) / 2
    elif scale_option == "Normalized":
        x_low = fb_threshold_norm / 2
        x_high = (1 + fb_threshold_norm) / 2
        y_low = wsjf_threshold_norm / 2
        y_high = (1 + wsjf_threshold_norm) / 2
    else:
        fb_range = fb_max - fb_min
        wsjf_range = wsjf_max - wsjf_min
        x_low = fb_min + (fb_threshold - fb_min) / 2
        x_high = fb_threshold + (fb_max - fb_threshold) / 2
        y_low = wsjf_min + (wsjf_threshold - wsjf_min) / 2
        y_high = wsjf_threshold + (wsjf_max - wsjf_threshold) / 2
    
    # Add quadrant labels
    fig.add_annotation(
        x=x_low,
        y=y_high,
        text="User Friction Only",
        showarrow=False,
        font=dict(size=12, color="#FBBF24")
    )
    
    fig.add_annotation(
        x=x_high,
        y=y_high,
        text="High Priority",
        showarrow=False,
        font=dict(size=12, color="#F87171")
    )
    
    fig.add_annotation(
        x=x_high,
        y=y_low,
        text="Structural Only",
        showarrow=False,
        font=dict(size=12, color="#60A5FA")
    )
    
    fig.add_annotation(
        x=x_low,
        y=y_low,
        text="Low Priority",
        showarrow=False,
        font=dict(size=12, color="#94A3B8")
    )
    
    # Set axis ranges based on scaling option
    if scale_option == "Full Range (0-1)":
        x_range = [0, 1]
        y_range = [0, 1]
    elif scale_option == "Normalized":
        x_range = [0, 1]
        y_range = [0, 1]
    else:  # Actual Range
        # Add some padding to the ranges
        fb_range = fb_max - fb_min
        wsjf_range = wsjf_max - wsjf_min
        x_range = [max(0, fb_min - fb_range * 0.05), fb_max + fb_range * 0.05]
        y_range = [max(0, wsjf_min - wsjf_range * 0.05), wsjf_max + wsjf_range * 0.05]
    
    # Update layout for better readability
    fig.update_layout(
        xaxis=dict(
            title=x_axis_title,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="rgba(100, 100, 100, 0.5)",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(100, 100, 100, 0.2)",
            range=x_range
        ),
        yaxis=dict(
            title=y_axis_title,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="rgba(100, 100, 100, 0.5)",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(100, 100, 100, 0.2)",
            range=y_range
        ),
        legend=dict(
            title="Priority Level",
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=80, b=120),
        title=dict(
            text="Structural Importance vs User Friction",
            y=0.95,
            x=0.5,
            xanchor="center",
            yanchor="top"
        )
    )
    
    # Add a second legend for the symbol meaning
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=12, symbol="circle"),
            name="Critical Node (Circle)",
            showlegend=True,
            legendgroup="symbols"
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=8, symbol="diamond"),
            name="Standard Node (Diamond)",
            showlegend=True,
            legendgroup="symbols"
        )
    )
    
    # Adjust legend layout to accommodate both legends
    fig.update_layout(
        legend=dict(
            title="Node Types",
            orientation="h",
            yanchor="top",
            y=-0.20,  # Position below the first legend
            xanchor="center",
            x=0.5,
            groupclick="toggleitem"
        )
    )
    
    # Clean up the quadrant annotations to prevent overlap
    for i, annotation in enumerate(fig.layout.annotations[:4]):  # First 4 annotations are quadrant labels
        annotation.font.size = 11
        
        # Adjust positioning to avoid overlapping with points or other elements
        if annotation.text == "High Priority":
            annotation.y = annotation.y - 0.05
            annotation.x = annotation.x - 0.05
        elif annotation.text == "User Friction Only":
            annotation.y = annotation.y - 0.05
            annotation.x = annotation.x + 0.05
        elif annotation.text == "Structural Only":
            annotation.y = annotation.y + 0.05
            annotation.x = annotation.x - 0.05
        elif annotation.text == "Low Priority":
            annotation.y = annotation.y + 0.05
            annotation.x = annotation.x + 0.05
    
    # Apply log scales if selected
    if log_scale_x:
        fig.update_xaxes(type="log")
    if log_scale_y:
        fig.update_yaxes(type="log")
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanatory text with clearer language
    st.write("""
    **Understanding the chart:**
    
    - **High Priority (Red)**: High structural importance AND high user friction - these require immediate attention
    - **User Friction Only (Yellow)**: Pages causing friction but not critical to site structure - improve user experience
    - **Structural Only (Blue)**: Important navigation nodes with low friction - maintain and optimize
    - **Low Priority (Gray)**: Less important pages with minimal friction - monitor but lower priority
    
    **Node Symbols:**
    - **Filled Circles (●)**: Critical nodes whose removal would severely disrupt network connectivity
    - **Diamonds (◆)**: Standard nodes that are less critical to overall network structure
    
    Critical nodes have more impact on network integrity - changes to these pages will affect the overall user flow.
    """)

def render_recurring_patterns(metrics_data):
    """
    Render visualization of recurring patterns in the user flow graph.
    
    Args:
        metrics_data (Dict): Dictionary containing metrics data.
    """
    with st.expander("🔄 Recurring User Flow Patterns", expanded=False):
        st.write("Identify common loops and repeated paths in user journeys")
        
        # Check if metrics data is available
        if not metrics_data or "network_metrics" not in metrics_data:
            st.info("No network metrics data available for this dataset.")
            return
        
        network_metrics = metrics_data["network_metrics"]
        
        # First check if we have exit paths data (newer version)
        if "recurring_exit_paths" in network_metrics:
            exit_paths_data = network_metrics["recurring_exit_paths"]
            if exit_paths_data and "exit_paths" in exit_paths_data and exit_paths_data["exit_paths"]:
                # First, add a tab for regular recurring patterns (if available)
                if "recurring_patterns" in network_metrics and network_metrics["recurring_patterns"]:
                    pattern_tab, exit_tab = st.tabs(["Regular Recurring Patterns", "Exit Path Patterns"])
                    
                    with pattern_tab:
                        st.write("### Regular Recurring Patterns")
                        st.write("These patterns show loops and repeated sequences in user journeys, regardless of whether they lead to exits.")
                        render_regular_patterns(network_metrics["recurring_patterns"])
                    
                    with exit_tab:
                        st.write("### Exit Path Patterns")
                        st.write("These patterns specifically track sequences that lead to users leaving the site (with exit rates).")
                        render_exit_path_patterns(exit_paths_data)
                else:
                    # If only exit paths are available
                    st.write("### Exit Path Patterns")
                    st.write("These patterns specifically track sequences that lead to users leaving the site (with exit rates).")
                    render_exit_path_patterns(exit_paths_data)
                return
        
        # Fall back to older recurring patterns data
        if "recurring_patterns" not in network_metrics or not network_metrics["recurring_patterns"]:
            # Check if we have a separate recurring_exit_paths.json file
            dataset_name = metrics_data.get("dataset_name", "default")
            exit_paths_file = Path(f"outputs/{dataset_name}/recurring_exit_paths.json")
            
            if exit_paths_file.exists():
                try:
                    with open(exit_paths_file, 'r') as f:
                        exit_paths_data = json.load(f)
                        if exit_paths_data and "exit_paths" in exit_paths_data and exit_paths_data["exit_paths"]:
                            st.write("### Exit Path Patterns")
                            st.write("These patterns specifically track sequences that lead to users leaving the site (with exit rates).")
                            render_exit_path_patterns(exit_paths_data)
                            return
                except Exception as e:
                    st.warning(f"Error loading exit paths data: {str(e)}")
            
            st.info("No recurring patterns found in this dataset.")
            return
        
        # If we only have regular patterns
        st.write("### Regular Recurring Patterns")
        st.write("These patterns show loops and repeated sequences in user journeys, regardless of whether they lead to exits.")
        render_regular_patterns(network_metrics["recurring_patterns"])

def render_regular_patterns(patterns_data):
    """
    Render visualization of regular recurring patterns.
    
    Args:
        patterns_data (Dict): Dictionary containing recurring patterns data.
    """
    if not patterns_data or "recurring_patterns" not in patterns_data or not patterns_data["recurring_patterns"]:
        st.info("No recurring patterns found in this dataset.")
        return
    
    # Get the patterns and node counts
    patterns = patterns_data["recurring_patterns"]
    node_counts = patterns_data.get("node_loop_counts", {})
    total_patterns = patterns_data.get("total_patterns", len(patterns))
    
    # Display summary
    st.write(f"Found {total_patterns} recurring patterns in the user flow graph.")
    
    # Create pattern data for display
    pattern_list = []
    for i, pattern in enumerate(patterns[:20]):  # Limit to top 20 patterns
        pattern_str = " → ".join(str(node) for node in pattern)
        pattern_list.append({
            "Pattern ID": i + 1,
            "Path": pattern_str,
            "Length": len(pattern),
            "Nodes": ", ".join(str(node) for node in pattern)
        })
    
    # Create DataFrame for display
    if pattern_list:
        patterns_df = pd.DataFrame(pattern_list)
        
        # Display patterns in a container (not an expander since we're already in an expander)
        st.write(f"#### Top 20 Recurring Patterns (out of {total_patterns})")
        st.dataframe(patterns_df, use_container_width=True)
    
    # Create node participation data
    if node_counts:
        node_data = []
        for node, count in sorted(node_counts.items(), key=lambda x: x[1], reverse=True):
            node_data.append({
                "Node": node,
                "Loop Count": count,
                "Percentage": f"{(count / total_patterns * 100):.1f}%"
            })
        
        # Create DataFrame for display
        nodes_df = pd.DataFrame(node_data)
        
        # Display top 10 nodes by loop participation
        st.write("### Top Nodes in Recurring Patterns")
        st.dataframe(nodes_df.head(10), use_container_width=True)
        
        # Create visualization of node participation
        try:
            import plotly.express as px
            
            # Create bar chart of top 10 nodes by loop participation
            fig = px.bar(
                nodes_df.head(10),
                x="Node",
                y="Loop Count",
                color="Loop Count",
                color_continuous_scale="Viridis",
                title="Top Nodes by Participation in Recurring Patterns",
                labels={"Node": "Node Path", "Loop Count": "Number of Loops"}
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Node Path",
                yaxis_title="Number of Loops",
                coloraxis_showscale=True,
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Error creating visualization: {str(e)}")
    
        # Add explanatory text
        st.write("""
        **What are recurring patterns?**
        
        Recurring patterns are loops or repeated sequences in user journeys. High loop counts may indicate:
        
        - Users getting stuck in circular navigation
        - Back-and-forth between related pages
        - Search-browse-return cycles
        
        Nodes with high loop participation are candidates for UX optimization to reduce unnecessary repetition.
        """)

def render_exit_path_patterns(exit_paths_data):
    """
    Render visualization of recurring exit paths.
    
    Args:
        exit_paths_data (Dict): Dictionary containing exit paths data.
    """
    # Get the exit paths
    exit_paths = exit_paths_data.get("exit_paths", [])
    node_counts = exit_paths_data.get("node_loop_counts", {})
    total_paths = len(exit_paths)
    
    # Display summary
    st.write(f"Found {total_paths} recurring exit paths in the user flow graph.")
    
    # Calculate average exit rate if possible
    exit_rates = [path_data.get("exit_rate", 0) for path_data in exit_paths if "exit_rate" in path_data]
    if exit_rates:
        avg_exit_rate = sum(exit_rates) / len(exit_rates)
        max_exit_rate = max(exit_rates)
        
        # Show exit rate metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Exit Rate", f"{avg_exit_rate:.2%}")
        with col2:
            st.metric("Highest Exit Rate", f"{max_exit_rate:.2%}")
    
    # Create exit path data for display
    exit_path_list = []
    for i, path_data in enumerate(exit_paths[:20]):  # Limit to top 20 paths
        path = path_data.get("path", [])
        count = path_data.get("count", 0)
        exit_rate = path_data.get("exit_rate", 0)
        
        path_str = " → ".join(str(node) for node in path)
        exit_path_list.append({
            "Path ID": i + 1,
            "Path": path_str,
            "Count": count,
            "Exit Rate": f"{exit_rate:.2%}",
            "Length": len(path)
        })
    
    # Create DataFrame for display
    if exit_path_list:
        exit_paths_df = pd.DataFrame(exit_path_list)
        
        # Sort by exit rate (descending)
        exit_paths_df = exit_paths_df.sort_values(by="Exit Rate", ascending=False)
        
        # Display patterns in a container (not an expander since we're already in an expander)
        st.write(f"#### Top 20 Exit Paths (out of {total_paths}) - Ranked by Exit Rate")
        st.dataframe(exit_paths_df, use_container_width=True)
    
    # Create node participation data
    if node_counts:
        node_data = []
        for node, count in sorted(node_counts.items(), key=lambda x: x[1], reverse=True):
            node_data.append({
                "Node": node,
                "Path Count": count,
                "Percentage": f"{(count / sum(node_counts.values()) * 100):.1f}%"
            })
        
        # Create DataFrame for display
        nodes_df = pd.DataFrame(node_data)
        
        # Display top 10 nodes by participation
        st.write("### Top Nodes in Exit Paths")
        st.dataframe(nodes_df.head(10), use_container_width=True)
        
        # Create visualization of exit paths
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Extract top 5 exit paths for visualization
            top_paths = exit_paths[:5]
            
            # Create Sankey diagram data
            source = []
            target = []
            value = []
            path_labels = []
            
            # Assign unique indices to nodes
            node_indices = {}
            node_idx = 0
            
            # Process each path
            for path_data in top_paths:
                path = path_data.get("path", [])
                count = path_data.get("count", 0)
                
                # Skip paths that are too short
                if len(path) < 2:
                    continue
                
                # Add path label
                path_str = " → ".join(path)
                path_labels.append(f"{path_str} ({count})")
                
                # Add nodes and links
                for i in range(len(path) - 1):
                    # Add source node if not seen before
                    if path[i] not in node_indices:
                        node_indices[path[i]] = node_idx
                        node_idx += 1
                    
                    # Add target node if not seen before
                    if path[i+1] not in node_indices:
                        node_indices[path[i+1]] = node_idx
                        node_idx += 1
                    
                    # Add link
                    source.append(node_indices[path[i]])
                    target.append(node_indices[path[i+1]])
                    value.append(count)
            
            # Create node labels
            node_labels = [""] * len(node_indices)
            for node, idx in node_indices.items():
                node_labels[idx] = str(node)
            
            # Create Sankey diagram if we have data
            if source and target and value:
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=node_labels,
                        color="blue"
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                        hoverlabel=dict(bgcolor="white", font_size=16),
                        color="rgba(100, 100, 200, 0.4)"
                    )
                )])
                
                # Update layout
                fig.update_layout(
                    title="Top 5 Exit Paths",
                    font=dict(size=12),
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Add path labels
                st.markdown("**Top Exit Paths:**")
                for i, label in enumerate(path_labels):
                    st.markdown(f"**{i+1}.** {label}")
            
            # Create bar chart of top 10 nodes by participation
            bar_fig = px.bar(
                nodes_df.head(10),
                x="Node",
                y="Path Count",
                color="Path Count",
                color_continuous_scale="Viridis",
                title="Top Nodes by Participation in Exit Paths",
                labels={"Node": "Node Path", "Path Count": "Number of Exit Paths"}
            )
            
            # Update layout
            bar_fig.update_layout(
                xaxis_title="Node Path",
                yaxis_title="Number of Exit Paths",
                coloraxis_showscale=True,
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            
            # Display chart
            st.plotly_chart(bar_fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Error creating visualization: {str(e)}")
    
    # Add explanatory text
    st.write("""
    **What are recurring exit paths?**
    
    Recurring exit paths are common sequences in user journeys that lead to exits or abandonments. High exit rates may indicate:
    
    - Pain points that cause users to leave
    - Confusing workflows where users get stuck
    - Natural exit points that may need optimization
    
    Analyzing these paths helps identify where to focus UX improvements for better user retention.
    
    **Exit Rate:** The percentage of sessions where users leave the site after following this specific path.
    Higher exit rates indicate potential friction points or abandonment issues.
    """)

def render_percolation_collapse(metrics_data):
    """
    Render visualization of percolation collapse simulation results.
    
    Args:
        metrics_data (Dict): Dictionary containing metrics data.
    """
    with st.expander("🔍 Network Stability Analysis", expanded=False):
        st.write("Understand how the removal of key nodes affects the user flow network")
        
        if not metrics_data:
            st.info("No metrics data available for this dataset.")
            return
        
        # Get percolation threshold and critical nodes count
        percolation_threshold = metrics_data.get("percolation_threshold", 0)
        critical_nodes_count = metrics_data.get("network_metrics", {}).get("critical_nodes_count", 0)
        
        # Display percolation metrics
        st.metric(
            label="Percolation Threshold", 
            value=f"{percolation_threshold:.2f}",
            help="Fraction of top nodes that can be removed before network collapses."
        )
        
        # Create a simple visualization of percolation collapse
        import plotly.graph_objects as go
        import numpy as np
        
        # Simulate percolation collapse curve
        # This is a visualization based on the threshold, not actual simulation results
        x = np.linspace(0, 1, 100)
        
        # Using a sigmoid function to model collapse
        # Centered at percolation_threshold
        midpoint = max(0.01, min(0.99, percolation_threshold))
        steepness = 15  # Controls how sharp the collapse is
        
        # Generate y values (fraction of largest connected component remaining)
        y = 1 / (1 + np.exp(steepness * (x - midpoint)))
        
        # Create figure
        fig = go.Figure()
        
        # Add collapse curve
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="Network Integrity",
                line=dict(color="#3B82F6", width=3),
            )
        )
        
        # Add vertical line at percolation threshold
        fig.add_shape(
            type="line",
            x0=percolation_threshold,
            y0=0,
            x1=percolation_threshold,
            y1=1,
            line=dict(color="red", width=2, dash="dash"),
        )
        
        # Add annotation for percolation threshold
        fig.add_annotation(
            x=percolation_threshold,
            y=0.9,
            text=f"Threshold: {percolation_threshold:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="#FF0000",
            font=dict(size=12, color="#FF0000"),
            ax=40,
            ay=0,
        )
        
        # Update layout
        fig.update_layout(
            title="Network Collapse Simulation",
            xaxis_title="Fraction of Nodes Removed (highest centrality first)",
            yaxis_title="Fraction of Network Remaining Connected",
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(211, 211, 211, 0.3)",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="rgba(211, 211, 211, 0.5)",
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(211, 211, 211, 0.3)",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="rgba(211, 211, 211, 0.5)",
            ),
            plot_bgcolor="rgba(255, 255, 255, 0)",
            paper_bgcolor="rgba(255, 255, 255, 0)",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display critical nodes information
        st.write(f"**Critical Nodes:** {critical_nodes_count} nodes identified as critical for network stability.")
        
        # Add interpretation based on threshold
        if percolation_threshold < 0.3:
            st.error("""
            **High Risk Network:** Your user flow network is highly dependent on a small number of critical nodes.
            If these key pages/events are disrupted, users may be unable to complete their journeys.
            Consider adding alternative paths and reducing bottlenecks.
                """)
        elif percolation_threshold < 0.5:
            st.warning("""
            **Moderate Risk Network:** Your user flow network has some resilience but still depends on key nodes.
            Consider strengthening alternative paths through critical sections of the journey.
                """)
        else:
            st.success("""
            **Robust Network:** Your user flow network demonstrates good resilience to disruption.
            Users have multiple paths to accomplish their goals, which is ideal for complex applications.
                """)
            
        # Add explanatory text
        st.write("""
        **What is Percolation Analysis?**
        
        Percolation analysis models how the network breaks apart as key nodes are progressively removed.
        The percolation threshold is the fraction of nodes that can be removed before the network collapses.
        
        - **Low threshold (<0.3):** Fragile network dependent on few critical nodes
        - **Medium threshold (0.3-0.5):** Moderately resilient network
        - **High threshold (>0.5):** Robust network with good redundancy
        
        This helps identify over-reliance on specific pages or events in your user experience.
            """)

def render_glossary_sidebar():
    """
    Render a metrics glossary in the sidebar.
    """
    with st.sidebar.expander("📘 Advanced Metrics Glossary", expanded=False):
        st.markdown("### Global Network Metrics")
        st.markdown("""
        - **Fractal Dimension (D)**: Measures how space-filling the graph is. Higher values (>2.2) indicate more complex user navigation paths with nested or recursive patterns.
        
        - **Power-Law Alpha (α)**: Exponent of degree distribution. Lower values (<1.8) indicate stronger hierarchy with vulnerable hub nodes. Higher values (>2.6) suggest more uniform structure.
        
        - **Percolation Threshold**: Fraction of top nodes that can be removed before network collapses. Lower values (<0.3) indicate a more fragile structure dependent on few critical nodes.
            """)
        
        st.markdown("### Node-Level Metrics")
        st.markdown("""
        - **Fractal Betweenness (FB)**: Measures a node's structural importance in controlling information flow. Values typically range from 0.6-1.0, with higher values indicating critical navigation points.
        
        - **Percolation Role**: How important a node is for maintaining network connectivity:
          • **Critical**: Removing this node would severely disrupt network connectivity
          • **Standard**: Less critical for overall network stability
        
        - **WSJF Score**: Weighted Shortest Job First score - combines frequency of friction with impact. Higher values (typically 0.2-0.4) indicate more pressing UX issues to fix.
            """)
        
        st.markdown("### Priority Quadrants")
        st.markdown("""
        - **High Priority (Top Right)**: High structural importance AND high user friction - requires immediate attention. These pages are both critical to site structure and causing significant user friction.
        
        - **User Friction Only (Top Left)**: Pages causing friction but not critical to site structure - focus on improving user experience but less urgent structurally.
        
        - **Structural Only (Bottom Right)**: Important navigation nodes with low friction - maintain and optimize these core pathways that are working well.
        
        - **Low Priority (Bottom Left)**: Less important pages with minimal friction - monitor but lower priority for changes.
            """)
        
        st.markdown("### Graph Statistics")
        st.markdown("""
        - **Node Count**: Total number of unique pages or states in the user journey.
        
        - **Edge Count**: Total number of transitions between pages.
        
        - **Edge/Node Ratio**: Measure of graph density - higher values indicate more complex navigation options.
        
        - **Connected Components**: Number of isolated subgraphs - ideally should be 1 for a fully connected experience.
            """)
        
        st.markdown("### UX Interpretations")
        st.markdown("""
        - **Redundant Bottleneck**: Node with high FB and high WSJF - users frequently encounter friction at this structurally important point.
        
        - **Complex Hub**: Node with high centrality in multiple recurring patterns - may be confusing users with too many options.
        
        - **Exit Paths**: Sequences of pages that frequently lead to users leaving the site - high exit rates indicate potential friction or completion points.
        
        - **Recurring Patterns**: Loops or repeated sequences in user journeys - may indicate users getting stuck in circular navigation.
            """)

def render_developer_controls(dataset):
    """
    Render developer controls for rerunning metrics.
    
    Args:
        dataset (str): Current dataset name
    """
    with st.expander("Developer Controls", expanded=False):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            rerun = st.button("🔁 Rerun All Metrics")
            if rerun:
                st.warning("Rerunning metrics may take some time...")
                try:
                    # Import required modules
                    from analysis.event_chokepoints import main as run_chokepoints
                    
                    # Run the analysis
                    st.info(f"Rerunning metrics for dataset {dataset}...")
                    
                    # Show spinner during processing
                    with st.spinner("Running analysis..."):
                        # Construct input paths
                        input_flows = f"outputs/{dataset}/session_flows.csv"
                        input_graph = f"outputs/{dataset}/user_graph.gpickle"
                        input_graph_multi = f"outputs/{dataset}/user_graph_multi.gpickle"
                        output_dir = f"outputs/{dataset}"
                        
                        # Check if files exist
                        if not os.path.exists(input_flows):
                            st.error(f"Session flows file not found: {input_flows}")
                            return
                        
                        if not os.path.exists(input_graph):
                            st.error(f"Graph file not found: {input_graph}")
                            return
                        
                        # Run the analysis
                        run_chokepoints(input_flows, input_graph, input_graph_multi, output_dir, fast=True)
                        
                        st.success("Metrics recomputed successfully! Refresh the page to see updated results.")
                        
                except Exception as e:
                    st.error(f"Error running metrics: {str(e)}")
                    logger.error(f"Error running metrics: {str(e)}")
                    logger.error(traceback.format_exc())
        
        with col2:
            show_debug = st.checkbox("Show intermediate logs", value=False)
            if show_debug:
                try:
                    # Display session stats log if available
                    log_file = f"outputs/{dataset}/session_stats.log"
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            st.code(f.read(), language="text")
                    else:
                        st.info(f"No log file found at {log_file}")
                        
                    # Display metrics JSON if available
                    metrics_file = f"outputs/{dataset}/metrics.json"
                    if os.path.exists(metrics_file):
                        with open(metrics_file, 'r') as f:
                            st.json(json.load(f))
                    else:
                        st.info(f"No metrics file found at {metrics_file}")
                except Exception as e:
                    st.error(f"Error loading debug information: {str(e)}")
        
        # Add timestamp of last analysis
        try:
            decision_table_path = f"outputs/{dataset}/decision_table.csv"
            if os.path.exists(decision_table_path):
                mod_time = os.path.getmtime(decision_table_path)
                mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                st.info(f"Last analysis: {mod_time_str}")
            else:
                st.info("No previous analysis found.")
        except Exception as e:
            st.error(f"Error checking last analysis time: {str(e)}")
            
def render_advanced_metrics_tab(metrics_data):
    """
    Render the Advanced Metrics tab content.
    
    Args:
        metrics_data (Dict): Dictionary containing metrics data.
    """
    st.title("Advanced Metrics")
    
    # Add tab description
    st.markdown("""
    This dashboard provides advanced network metrics to help identify UX improvement opportunities
    based on structural analysis of user flows and friction points.
        """)
    
    # Render the top metrics cards
    render_top_metrics(metrics_data)
    
    # Add space between sections
    st.markdown("---")
    
    # Render the decision table
    render_decision_table(metrics_data)
    
    # Add space between sections
    st.markdown("---")
    
    # Render the FB vs WSJF chart
    render_fb_vs_wsjf_chart(metrics_data)
    
    # Add space between sections
    st.markdown("---")
    
    # Render recurring patterns
    render_recurring_patterns(metrics_data)
    
    # Add space between sections
    st.markdown("---")
    
    # Render percolation collapse
    render_percolation_collapse(metrics_data)
    
    # Add developer controls
    st.markdown("---")
    render_developer_controls(metrics_data.get("dataset_name", "default"))

def main():
    """Main function to run the dashboard."""
    configure_dark_theme()
    
    # Display header with logo
    logo_path = "logos/telomesh logo.png"
    if os.path.exists(logo_path):
        logo_html = f'<img src="data:image/png;base64,{load_logo_base64(logo_path)}" class="telomesh-logo">'
        st.markdown(
            f'<div class="telomesh-header" style="flex-direction: column; align-items: center;">{logo_html}<h1>User Journey Intelligence</h1></div>',
            unsafe_allow_html=True
        )
    else:
        st.title("User Journey Intelligence")
    
    # Sidebar setup
    st.sidebar.header("Dashboard Controls")
    
    # Dataset selection
    st.sidebar.subheader("Dataset Selection")
    datasets, most_recent = discover_datasets()
    
    if not datasets:
        st.sidebar.warning("No datasets found. Please run the TeloMesh pipeline first.")
        st.error("No datasets found in the outputs directory.")
        st.info("To generate a dataset, run: `python main.py --dataset <n> --users <count> --events <count>`")
        return
    
    # Filter to only valid datasets
    valid_datasets = [d for d in datasets if is_valid_dataset(d)]
    
    if not valid_datasets:
        st.sidebar.warning("No valid datasets found. Please run the TeloMesh pipeline first.")
        st.error("No valid datasets found in the outputs directory.")
        st.info("To generate a dataset, run: `python main.py --dataset <n> --users <count> --events <count>`")
        return
    
    # Dataset info
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset in /outputs",
        valid_datasets,
        index=valid_datasets.index(most_recent) if most_recent in valid_datasets else 0
    )
    
    # Display dataset info if available
    dataset_info_path = Path(f"outputs/{selected_dataset}/dataset_info.json")
    if dataset_info_path.exists():
        try:
            with open(dataset_info_path, 'r') as f:
                dataset_info = json.load(f)
            
            # Format the timestamp if it exists
            if "creation_timestamp" in dataset_info:
                try:
                    timestamp = datetime.fromisoformat(dataset_info["creation_timestamp"])
                    dataset_info["creation_timestamp"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass  # Keep the original timestamp if parsing fails
            
            # Show dataset info in an expander
            with st.sidebar.expander("Dataset Info", expanded=False):
                for key, value in dataset_info.items():
                    if key == "dataset_name":
                        continue  # Skip displaying dataset name since it's already shown in the selectbox
                    key_display = key.replace("_", " ").title()
                    st.write(f"**{key_display}:** {value}")
        except Exception as e:
            logger.error(f"Error loading dataset info: {str(e)}")
    
    # Add the new glossary to sidebar
    render_glossary_sidebar()
    
    # Add sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info(
        "TeloMesh analyzes user flows to identify UX improvement opportunities."
    )
    
    # Load data based on selected dataset
    friction_df, flow_df, node_map, G = load_friction_data(selected_dataset)
    
    if friction_df is None or flow_df is None or node_map is None or G is None:
        st.error("Failed to load data. Please check the logs for details.")
        return
    
    # Add node count to sidebar dataset info
    if G is not None:
        node_count = len(G.nodes())
        edge_count = len(G.edges())
        
        # Display in sidebar
        with st.sidebar.expander("Graph Statistics", expanded=False):
            st.write(f"**Nodes:** {node_count}")
            st.write(f"**Edges:** {edge_count}")
            st.write(f"**Edge/Node Ratio:** {edge_count/node_count:.1f}")
            
            # Display connected components info
            if nx.is_directed(G):
                num_components = nx.number_weakly_connected_components(G)
                st.write(f"**Connected Components:** {num_components}")
            else:
                num_components = nx.number_connected_components(G)
                st.write(f"**Connected Components:** {num_components}")
    
    # Load data for advanced metrics
    metrics_data = load_advanced_metrics(selected_dataset)
    if metrics_data:
        # Add dataset name to metrics data for reference
        metrics_data["dataset_name"] = selected_dataset
    
    # Navigation tabs - add Advanced Metrics tab
    tab1, tab2, tab3, tab4 = st.tabs([
        "Friction Analysis", 
        "Flow Analysis", 
        "User Journey Graph",
        "Advanced Metrics"
    ])
    
    # Compute thresholds
    top10_threshold = friction_df['WSJF_Friction_Score'].quantile(0.9)
    top20_threshold = friction_df['WSJF_Friction_Score'].quantile(0.8)
    
    with tab1:
        st.header("🔥 Friction Analysis Overview")
        st.write("Select to display top 3 pages ranked by users lost, exit rate or WSJF score")
        
        # Render friction table
        render_friction_table(friction_df)
        
    with tab2:
        st.header("🔁 User Flow Analysis", help="User journeys with multiple friction points")
        st.write("""
        * This analysis ranks user journeys by total WSJF scores along the path. 
        * High scores indicate multiple high-friction chokepoints that can disrupt user flows and lead to increased drop-offs. 
        * Flow length is defined as the number of page/event steps along a user journey path.
            """)
        
        # Render flow summaries without an additional header
        render_flow_summaries(flow_df)
        
    with tab3:
        # The render_graph_heatmap function already has its own header with tooltip
        render_graph_heatmap(G, node_map)
    
    with tab4:
        # Check if metrics data is available
        if metrics_data:
            render_advanced_metrics_tab(metrics_data)
        else:
            st.title("Advanced Metrics")
            st.warning("Could not load advanced metrics data for this dataset.")
            st.info("Please check if the dataset contains the required files: decision_table.csv, final_report.json, etc.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "TeloMesh User Flow Intelligence Dashboard | "
        "Built with Streamlit, NetworkX, and PyVis"
    )

# Run the dashboard when this script is executed directly
if __name__ == "__main__":
    main() 