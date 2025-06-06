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

def main():
    """Main function to run the dashboard."""
    # Configure dark theme
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
        st.info("To generate a dataset, run: `python main.py --dataset <name> --users <count> --events <count>`")
        return
    
    # Filter to only valid datasets
    valid_datasets = [d for d in datasets if is_valid_dataset(d)]
    
    if not valid_datasets:
        st.sidebar.warning("No valid datasets found. Please run the TeloMesh pipeline first.")
        st.error("No valid datasets found in the outputs directory.")
        st.info("To generate a dataset, run: `python main.py --dataset <name> --users <count> --events <count>`")
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
    
    # Load data based on selected dataset
    friction_df, flow_df, node_map, G = load_friction_data(selected_dataset)
    
    if friction_df is None or flow_df is None or node_map is None or G is None:
        st.error("Failed to load data. Please check the logs for details.")
        return
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["Friction Analysis", "Flow Analysis", "User Journey Graph"])
    
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
    
    # Footer
    st.markdown("---")
    st.markdown(
        "TeloMesh User Flow Intelligence Dashboard | "
        "Built with Streamlit, NetworkX, and PyVis"
    )

# Run the dashboard when this script is executed directly
if __name__ == "__main__":
    main() 