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
            align-items: center !important;
            margin-bottom: 1.5rem !important;
        }
        
        .telomesh-logo {
            height: 3rem !important;
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

def load_friction_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict, nx.DiGraph]:
    """
    Load the friction data from pre-computed files.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict, nx.DiGraph]: Tuple containing:
            - friction_df: DataFrame with event chokepoints
            - flow_df: DataFrame with fragile flows
            - node_map: Dictionary mapping page names to WSJF scores
            - graph: NetworkX DiGraph of user journeys
    """
    try:
        # Load event chokepoints
        friction_df = pd.read_csv("outputs/event_chokepoints.csv")
        
        # Load fragile flows
        flow_df = pd.read_csv("outputs/high_friction_flows.csv")
        
        # Load node map
        with open("outputs/friction_node_map.json", "r") as f:
            node_map = json.load(f)
        
        # Load the graph
        with open("outputs/user_graph.gpickle", "rb") as f:
            graph = pickle.load(f)
        
        return friction_df, flow_df, node_map, graph
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(traceback.format_exc())
        raise

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
        "WSJF_Friction_Score": "Frustration × importance. Prioritizes fixing pain points that are both high in exit rate and structurally critical.",
        "is_chokepoint": "Identifies high-friction points (top 10% by WSJF score)."
    }
    
    return tooltips.get(metric, "Hover for more information about this metric.")

def render_friction_table(df: pd.DataFrame):
    """
    Render a sortable, filterable table of friction points.
    
    Args:
        df (pd.DataFrame): DataFrame with event chokepoints.
    """
    st.header("🔥 Friction Points", help="Areas where users get stuck or exit")
    
    try:
        # Add summary metrics at the top
        total_users_lost = df["users_lost"].sum()
        avg_exit_rate = df["exit_rate"].mean()
        avg_wsjf = df["WSJF_Friction_Score"].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Users Lost", f"{total_users_lost:,}")
        col2.metric("Avg. Exit Rate", f"{avg_exit_rate:.2%}")
        col3.metric("Avg. WSJF Score", f"{avg_wsjf:.6f}")
        
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
        components.html(html_content, height=height)
        
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

def render_graph_heatmap(graph: nx.DiGraph, score_map: Dict[str, float]):
    """
    Render a graph heatmap with nodes colored by WSJF score.
    
    Args:
        graph (nx.DiGraph): Directed graph of user journeys.
        score_map (Dict[str, float]): Dictionary mapping page names to WSJF scores.
    """
    st.header("🌐 User Flow Heatmap", help="Visual map of user journeys with friction points highlighted")
    
    try:
        # Compute thresholds for coloring
        if score_map:
            scores = list(score_map.values())
            top10_threshold = np.percentile(scores, 90)
            top20_threshold = np.percentile(scores, 80)
        else:
            top10_threshold = 1.0
            top20_threshold = 0.5
        
        # Filter controls
        col1, col2 = st.columns(2)
        
        with col1:
            # Option to show only top friction nodes
            show_options = ["All nodes", "Top 20% friction nodes", "Top 10% friction nodes"]
            show_selection = st.selectbox("Show in graph:", show_options)
        
        with col2:
            # Option to adjust physics
            physics_enabled = st.checkbox("Enable physics for movable nodes\n(toggle to re-center graph)", value=False)
        
        # Create and configure the network based on filters
        html_content = ""
        if show_selection == "Top 10% friction nodes":
            # Filter to only include top 10% nodes
            filtered_nodes = {node: score for node, score in score_map.items() if score >= top10_threshold}
            if not filtered_nodes:
                st.warning("No nodes meet the top 10% threshold. Try showing all nodes or top 20%.")
                return
            
            net = create_filtered_network(graph, filtered_nodes, top10_threshold, top20_threshold, physics_enabled)
        elif show_selection == "Top 20% friction nodes":
            # Filter to only include top 20% nodes
            filtered_nodes = {node: score for node, score in score_map.items() if score >= top20_threshold}
            if not filtered_nodes:
                st.warning("No nodes meet the top 20% threshold. Try showing all nodes.")
                return
            
            net = create_filtered_network(graph, filtered_nodes, top10_threshold, top20_threshold, physics_enabled)
        else:
            # Show all nodes
            net = create_full_network(graph, score_map, top10_threshold, top20_threshold, physics_enabled)
        
        # Add export button for the graph
        export_container = st.container()
        with export_container:
            st.markdown(
                '<div class="download-container">Use the button below to export the current graph view</div>',
                unsafe_allow_html=True
            )
        
        # Render the network and get HTML content
        html_content = render_network_graph(net)
        
        # Add export HTML button if we have content
        if html_content:
            filtered_text = "top_10pct" if show_selection == "Top 10% friction nodes" else "top_20pct" if show_selection == "Top 20% friction nodes" else "all_nodes"
            filename = f"flow_heatmap_{filtered_text}"
            
            st.markdown(
                f'<div class="download-container">{get_html_download_link(html_content, filename, "📥 Export Graph as HTML")}</div>',
                unsafe_allow_html=True
            )
        
        # Add a legend
        st.markdown("""
        **Legend:**
        - 🔴 **Red nodes**: Top 10% friction points (highest WSJF scores)
        - 🟠 **Orange nodes**: Top 20% friction points
        - ⚪ **Gray nodes**: Lower friction points
        - Hover over nodes and edges for more details
        - Click nodes to see their connections
        """)
    except Exception as e:
        st.error(f"Error in graph heatmap: {str(e)}")
        logger.error(f"Error in graph heatmap: {str(e)}")
        logger.error(traceback.format_exc())

def create_full_network(graph, score_map, top10_threshold, top20_threshold, physics_enabled):
    """
    Create a network with all nodes.
    
    Args:
        graph (nx.DiGraph): The user journey graph
        score_map (dict): Mapping of page names to WSJF scores
        top10_threshold (float): Threshold for top 10% nodes
        top20_threshold (float): Threshold for top 20% nodes
        physics_enabled (bool): Whether physics simulation is enabled
        
    Returns:
        Network: PyVis network object
    """
    # Create a pyvis network
    net = Network(height="600px", width="100%", bgcolor="#0B0F19", font_color="#E2E8F0")
    
    # Create a mapping of node colors for use in highlighting edges
    node_colors = {}
    
    # Add nodes with colors based on WSJF score
    for node in graph.nodes():
        score = score_map.get(node, 0)
        
        if score >= top10_threshold:
            color = "#F87171"  # Soft red for top 10%
            title = f"{node}<br>WSJF Score: {score:.6f}<br>Top 10% friction point"
            node_colors[node] = color
        elif score >= top20_threshold:
            color = "#FBBF24"  # Warm amber for top 20%
            title = f"{node}<br>WSJF Score: {score:.6f}<br>Top 20% friction point"
            node_colors[node] = color
        else:
            color = "#94A3B8"  # Soft steel for low-friction nodes
            title = f"{node}<br>WSJF Score: {score:.6f}"
            node_colors[node] = color
        
        net.add_node(node, title=title, color=color, font={"color": "#FFFFFF"})
    
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
            }
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
            "enabled": physics_enabled,
            "barnesHut": {
                "gravitationalConstant": -10000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.04
            },
            "maxVelocity": 50
        },
        "layout": {"improvedLayout": True},
        "interaction": {
            "hover": True,
            "tooltipDelay": 200,
            "hideEdgesOnDrag": False,
            "selectable": True,
            "multiselect": False,
            "navigationButtons": True,
            "selectConnectedEdges": True,  # Highlight connected edges when clicking a node
            "hoverConnectedEdges": True    # Highlight connected edges when hovering over a node
        }
    }
    
    net.set_options(json.dumps(options))
    return net

def create_filtered_network(graph, filtered_nodes, top10_threshold, top20_threshold, physics_enabled):
    """
    Create a network with only the filtered nodes.
    
    Args:
        graph (nx.DiGraph): The user journey graph
        filtered_nodes (dict): Mapping of page names to WSJF scores that passed the filter
        top10_threshold (float): Threshold for top 10% nodes
        top20_threshold (float): Threshold for top 20% nodes
        physics_enabled (bool): Whether physics simulation is enabled
        
    Returns:
        Network: PyVis network object
    """
    # Create a pyvis network
    net = Network(height="600px", width="100%", bgcolor="#0B0F19", font_color="#E2E8F0")
    
    # Create a mapping of node colors for use in highlighting edges
    node_colors = {}
    
    # Add only the filtered nodes
    for node, score in filtered_nodes.items():
        if node in graph.nodes():
            if score >= top10_threshold:
                color = "#F87171"  # Soft red for top 10%
                title = f"{node}<br>WSJF Score: {score:.6f}<br>Top 10% friction point"
                node_colors[node] = color
            else:  # Must be top 20% if it's in filtered_nodes
                color = "#FBBF24"  # Warm amber for top 20%
                title = f"{node}<br>WSJF Score: {score:.6f}<br>Top 20% friction point"
                node_colors[node] = color
            
            net.add_node(node, title=title, color=color, font={"color": "#FFFFFF"})
    
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
                }
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
            "enabled": physics_enabled,
            "barnesHut": {
                "gravitationalConstant": -10000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.04
            },
            "maxVelocity": 50
        },
        "layout": {"improvedLayout": True},
        "interaction": {
            "hover": True,
            "tooltipDelay": 200,
            "hideEdgesOnDrag": False,
            "selectable": True,
            "multiselect": False,
            "navigationButtons": True,
            "selectConnectedEdges": True,  # Highlight connected edges when clicking a node
            "hoverConnectedEdges": True    # Highlight connected edges when hovering over a node
        }
    }
    
    net.set_options(json.dumps(options))
    return net

def render_flow_summaries(flow_df: pd.DataFrame):
    """
    Render summaries of fragile flows (paths with multiple high-friction points).
    
    Args:
        flow_df (pd.DataFrame): DataFrame containing fragile flows.
    """
    st.header("🔁 Fragile Flows", help="User journeys with multiple friction points")
    
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
        col1, col2 = st.columns(2)
        
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
        
        # Apply filters
        filtered_sessions = []
        
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
    """Main entry point for the dashboard."""
    # Configure dark theme
    configure_dark_theme()
    
    # Add TeloMesh logo and title with better vertical layout
    try:
        # Try to load the logo
        logo_path = "logos/telomesh logo.png"
        encoded_logo = load_logo_base64(logo_path)
        
        if encoded_logo:
            # Use flex column for vertical stacking of logo and title
            st.markdown(f'''
            <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{encoded_logo}" width="80" style="margin-bottom: 10px;" />
                <h1 style="margin: 0; color: #F8FAFC; text-align: center;">TeloMesh User Flow Intelligence</h1>
            </div>
            ''', unsafe_allow_html=True)
        else:
            # If logo not found, use icon with text
            st.markdown(f'''
            <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 20px;">
                <div style="font-size: 48px; margin-bottom: 10px;">🔍</div>
                <h1 style="margin: 0; color: #F8FAFC; text-align: center;">TeloMesh User Flow Intelligence</h1>
            </div>
            ''', unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error displaying header: {str(e)}")
        # Fallback to simple title
        st.title("TeloMesh User Flow Intelligence")
    
    st.markdown("""
    This dashboard helps product managers identify and prioritize UX improvements by analyzing user journeys and detecting friction points.
    
    - **Friction Points**: Individual (page, event) pairs ranked by WSJF Friction Score
    - **User Flow Heatmap**: Visual representation of user journeys with friction points highlighted
    - **Fragile Flows**: User paths containing multiple high-friction points
    """)
    
    # Load data
    try:
        event_df, flow_df, node_map, graph = load_friction_data()
        
        if event_df is not None and flow_df is not None and graph is not None:
            # Create tabs
            tabs = st.tabs(["Friction Points", "User Flow Heatmap", "Fragile Flows"])
            
            with tabs[0]:
                render_friction_table(event_df)
                
            with tabs[1]:
                render_graph_heatmap(graph, node_map)
                
            with tabs[2]:
                render_flow_summaries(flow_df)
                
        else:
            st.error("Failed to load data. Please check the data files.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Dashboard error: {str(e)}")
        logger.error(traceback.format_exc())

# Run the dashboard when this script is executed directly
if __name__ == "__main__":
    main() 