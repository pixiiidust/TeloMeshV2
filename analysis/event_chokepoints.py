"""
TeloMesh Event Chokepoints Analysis - LOVABLE Stage

This module identifies UX weak points by analyzing:
- Event-level exit rates: Probability session ends after (page, event)
- Page betweenness: How structurally critical a page is in session paths
- Fragile flows: Flows where ≥2 high-friction (page, event) steps occur

The WSJF_Friction_Score is calculated as exit_rate × betweenness, which helps
prioritize fixes for high-impact friction points.
"""

import os
import argparse
import pandas as pd
import networkx as nx
import pickle
import json
from collections import defaultdict

# CURSOR RULE: CHOKEPOINT METRIC COMPUTATION
# - Input: session_flows.csv + user_graph.gpickle
# - Output: event_chokepoints.csv with columns:
#     ['page', 'event', 'exit_rate', 'betweenness', 'users_lost', 'WSJF_Friction_Score']
# - WSJF_Friction_Score = exit_rate × betweenness
# - Identify top 10% WSJF scores to define "chokepoints"
# - Output: high_friction_flows.csv → sessions containing ≥2 chokepoints
# - Output: friction_node_map.json → {page: WSJF score of highest event at that page}
# - Ensure that session order is preserved when detecting fragile flows

def compute_exit_rates(df):
    """
    Compute exit rates for each (page, event) pair in the session flows.
    
    Exit rate is defined as the probability that a session ends after a particular (page, event) pair.
    
    Args:
        df (pd.DataFrame): DataFrame containing session flows.
        
    Returns:
        pd.DataFrame: DataFrame with exit rates for each (page, event) pair.
    """
    print("Computing exit rates...")
    
    # Group by session_id
    session_groups = df.groupby('session_id')
    
    # Count total occurrences of each (page, event) pair
    page_event_counts = df.groupby(['page', 'event']).size().reset_index(name='total_count')
    
    # Initialize dictionary to count exit events
    exit_counts = defaultdict(int)
    
    # Count exits for each (page, event) pair
    for session_id, session_df in session_groups:
        # Sort by step_index to ensure chronological order
        session_df = session_df.sort_values('step_index')
        
        # Get the last step in the session
        last_step = session_df.iloc[-1]
        
        # Increment the exit count for this (page, event) pair
        exit_counts[(last_step['page'], last_step['event'])] += 1
    
    # Create a DataFrame with exit counts
    exit_data = []
    for (page, event), count in exit_counts.items():
        exit_data.append({
            'page': page,
            'event': event,
            'exit_count': count
        })
    
    exit_df = pd.DataFrame(exit_data)
    
    # Merge with total counts to compute exit rates
    result = pd.merge(exit_df, page_event_counts, on=['page', 'event'], how='right')
    
    # Fill NaN exit_count with 0 (for page-event pairs that never occur as exits)
    result['exit_count'] = result['exit_count'].fillna(0).astype(int)
    
    # Compute exit rate and users lost
    result['exit_rate'] = result['exit_count'] / result['total_count']
    result['users_lost'] = result['exit_count']
    
    # Fill NaN exit_rate with 0
    result['exit_rate'] = result['exit_rate'].fillna(0)
    
    # Select and rename columns
    result = result[['page', 'event', 'exit_rate', 'users_lost']]
    
    print(f"Exit rates computed for {len(result)} (page, event) pairs.")
    return result

def compute_betweenness(g):
    """
    Compute betweenness centrality for all nodes in the graph.
    
    Betweenness centrality measures how often a node appears on shortest paths between other nodes.
    
    Args:
        g (nx.DiGraph): Directed graph of user journeys.
        
    Returns:
        dict: Dictionary mapping node names to betweenness centrality values.
    """
    print("Computing betweenness centrality...")
    
    # Compute betweenness centrality
    centrality = nx.betweenness_centrality(g, weight='weight')
    
    # Normalize to 0-1 range if not already
    if centrality and max(centrality.values()) > 0:
        max_centrality = max(centrality.values())
        centrality = {node: value / max_centrality for node, value in centrality.items()}
    
    print(f"Betweenness centrality computed for {len(centrality)} nodes.")
    return centrality

def compute_wsjf_friction(exit_df, centrality_dict):
    """
    Compute WSJF Friction Score for each (page, event) pair.
    
    WSJF (Weighted Shortest Job First) Friction Score is calculated as exit_rate × betweenness,
    which prioritizes fixing pain points that are both high in exit rate and structurally critical.
    
    Args:
        exit_df (pd.DataFrame): DataFrame with exit rates.
        centrality_dict (dict): Dictionary of betweenness centrality values.
        
    Returns:
        pd.DataFrame: DataFrame with WSJF Friction Scores.
    """
    print("Computing WSJF Friction Scores...")
    
    # Add betweenness to exit_df
    exit_df['betweenness'] = exit_df['page'].map(centrality_dict)
    
    # Fill NaN betweenness with 0
    exit_df['betweenness'] = exit_df['betweenness'].fillna(0)
    
    # Compute WSJF Friction Score
    exit_df['WSJF_Friction_Score'] = exit_df['exit_rate'] * exit_df['betweenness']
    
    # Sort by WSJF Friction Score
    result = exit_df.sort_values('WSJF_Friction_Score', ascending=False)
    
    print(f"WSJF Friction Scores computed for {len(result)} (page, event) pairs.")
    return result

def detect_fragile_flows(session_df, chokepoints_df):
    """
    Detect fragile flows, which are user paths containing ≥2 high-friction (page, event) pairs.
    
    Args:
        session_df (pd.DataFrame): DataFrame containing session flows.
        chokepoints_df (pd.DataFrame): DataFrame with WSJF Friction Scores.
        
    Returns:
        pd.DataFrame: DataFrame containing fragile flows.
    """
    print("Detecting fragile flows...")
    
    # Define chokepoints as the top 10% of (page, event) pairs by WSJF Friction Score
    chokepoint_threshold = chokepoints_df['WSJF_Friction_Score'].quantile(0.9)
    chokepoints = set(tuple(x) for x in chokepoints_df[
        chokepoints_df['WSJF_Friction_Score'] >= chokepoint_threshold
    ][['page', 'event']].values)
    
    print(f"Using threshold {chokepoint_threshold:.6f} for chokepoints (top 10%).")
    print(f"Identified {len(chokepoints)} chokepoint (page, event) pairs.")
    
    # Group sessions
    session_groups = session_df.groupby('session_id')
    
    # Initialize list to store fragile flows
    fragile_flows = []
    
    # Identify sessions with ≥2 chokepoints
    for session_id, session_data in session_groups:
        # Sort by step_index to ensure chronological order
        session_data = session_data.sort_values('step_index')
        
        # Count chokepoints in this session
        session_chokepoints = []
        
        for _, row in session_data.iterrows():
            page_event_pair = (row['page'], row['event'])
            
            if page_event_pair in chokepoints:
                session_chokepoints.append({
                    'session_id': session_id,
                    'step_index': row['step_index'],
                    'page': row['page'],
                    'event': row['event'],
                    'user_id': row['user_id'] if 'user_id' in row else None
                })
        
        # If this session has ≥2 chokepoints, add all its steps to fragile_flows
        if len(session_chokepoints) >= 2:
            for _, row in session_data.iterrows():
                # Look up the WSJF score for this (page, event) pair
                wsjf_score = chokepoints_df[
                    (chokepoints_df['page'] == row['page']) & 
                    (chokepoints_df['event'] == row['event'])
                ]['WSJF_Friction_Score'].values
                
                # Use 0 if not found
                wsjf_score = wsjf_score[0] if len(wsjf_score) > 0 else 0
                
                fragile_flows.append({
                    'session_id': session_id,
                    'step_index': row['step_index'],
                    'page': row['page'],
                    'event': row['event'],
                    'WSJF_Friction_Score': wsjf_score,
                    'is_chokepoint': 1 if (row['page'], row['event']) in chokepoints else 0,
                    'user_id': row['user_id'] if 'user_id' in row else None
                })
    
    # Convert to DataFrame
    if fragile_flows:
        fragile_flows_df = pd.DataFrame(fragile_flows)
        
        # Sort by session_id and step_index
        fragile_flows_df = fragile_flows_df.sort_values(['session_id', 'step_index'])
        
        print(f"Detected {fragile_flows_df['session_id'].nunique()} fragile flows with >=2 chokepoints.")
        return fragile_flows_df
    else:
        print("No fragile flows detected.")
        # Return empty DataFrame with the correct columns
        return pd.DataFrame(columns=[
            'session_id', 'step_index', 'page', 'event', 
            'WSJF_Friction_Score', 'is_chokepoint', 'user_id'
        ])

def export_chokepoints(df, flow_df, node_map, output_dir="outputs"):
    """
    Export chokepoints data to CSV and JSON files.
    
    Args:
        df (pd.DataFrame): DataFrame with WSJF Friction Scores.
        flow_df (pd.DataFrame): DataFrame containing fragile flows.
        node_map (dict): Dictionary mapping page names to WSJF scores.
        output_dir (str): Directory to save the output files.
        
    Returns:
        None
    """
    print("Exporting results...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Export event_chokepoints.csv
    df.to_csv(f"{output_dir}/event_chokepoints.csv", index=False)
    print(f"Exported event_chokepoints.csv with {len(df)} rows.")
    
    # Export high_friction_flows.csv
    flow_df.to_csv(f"{output_dir}/high_friction_flows.csv", index=False)
    print(f"Exported high_friction_flows.csv with {len(flow_df)} rows.")
    
    # Export friction_node_map.json
    with open(f"{output_dir}/friction_node_map.json", 'w') as f:
        json.dump(node_map, f, indent=2)
    print(f"Exported friction_node_map.json with {len(node_map)} nodes.")

def create_node_map(df):
    """
    Create a node map with the highest WSJF Friction Score for each page.
    
    Args:
        df (pd.DataFrame): DataFrame with WSJF Friction Scores.
        
    Returns:
        dict: Dictionary mapping page names to their highest WSJF Friction Score.
    """
    # Group by page and find the maximum WSJF Friction Score
    node_map = {}
    
    for page, group in df.groupby('page'):
        # Get the maximum WSJF Friction Score for this page
        max_score = group['WSJF_Friction_Score'].max()
        node_map[page] = max_score
    
    return node_map

def main(input_flows="outputs/session_flows.csv", 
         input_graph="outputs/user_graph.gpickle",
         output_dir="outputs",
         fast=False):
    """
    Main function to run the event chokepoints analysis.
    
    Args:
        input_flows (str): Path to the input session flows CSV.
        input_graph (str): Path to the input graph pickle file.
        output_dir (str): Directory to save the output files.
        fast (bool): Whether to run in fast mode (skip certain validations).
        
    Returns:
        None
    """
    print("\n[ANALYSIS] TeloMesh Event Chokepoints Analysis")
    print("=" * 50)
    
    # Load the session flows data
    print(f"Loading session flows from {input_flows}...")
    session_df = pd.read_csv(input_flows)
    print(f"Loaded {len(session_df)} steps from {session_df['session_id'].nunique()} sessions.")
    
    # Load the graph data
    print(f"Loading graph from {input_graph}...")
    with open(input_graph, 'rb') as f:
        G = pickle.load(f)
    print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    
    # Compute exit rates
    exit_df = compute_exit_rates(session_df)
    
    # Compute betweenness centrality
    centrality_dict = compute_betweenness(G)
    
    # Compute WSJF Friction Score
    friction_df = compute_wsjf_friction(exit_df, centrality_dict)
    
    # Create node map for visualization
    node_map = create_node_map(friction_df)
    
    # Detect fragile flows
    fragile_flows_df = detect_fragile_flows(session_df, friction_df)
    
    # Export results
    export_chokepoints(friction_df, fragile_flows_df, node_map, output_dir)
    
    print("\n[OK] Event chokepoints analysis complete!")
    print("=" * 50)
    
    if not fast:
        # Print some statistics
        total_pages = len(friction_df['page'].unique())
        total_events = len(friction_df['event'].unique())
        total_page_events = len(friction_df)
        top_friction_threshold = friction_df['WSJF_Friction_Score'].quantile(0.9)
        top_friction_count = sum(friction_df['WSJF_Friction_Score'] >= top_friction_threshold)
        
        print("\nSummary Statistics:")
        print(f"Total pages: {total_pages}")
        print(f"Total event types: {total_events}")
        print(f"Total (page, event) pairs: {total_page_events}")
        print(f"Top 10% friction threshold: {top_friction_threshold:.6f}")
        print(f"Number of high-friction points: {top_friction_count}")
        print(f"Number of fragile flows: {fragile_flows_df['session_id'].nunique()}")
        
        # Print top 5 friction points
        print("\nTop 5 Friction Points:")
        top5 = friction_df.head(5)
        for _, row in top5.iterrows():
            print(f"  {row['page']} ({row['event']}): {row['WSJF_Friction_Score']:.6f}")
    
    return friction_df, fragile_flows_df, node_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TeloMesh Event Chokepoints Analysis")
    parser.add_argument("--n_users", type=int, default=1000, help="Number of unique users")
    parser.add_argument("--events_per_user", type=int, default=6, help="Average events per user")
    parser.add_argument("--input_flows", type=str, default="outputs/session_flows.csv", help="Path to session flows CSV")
    parser.add_argument("--input_graph", type=str, default="outputs/user_graph.gpickle", help="Path to graph pickle")
    parser.add_argument("--output_path", type=str, default="outputs", help="Output directory")
    parser.add_argument("--fast", action="store_true", help="Skip detailed statistics output")
    
    args = parser.parse_args()
    
    # Run the main function
    main(args.input_flows, args.input_graph, args.output_path, args.fast)