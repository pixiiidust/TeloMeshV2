#!/usr/bin/env python3
"""
TeloMesh - User Journey Analysis Pipeline
"""

import argparse
import os
import sys
import time
import shutil
import json
from datetime import datetime
from pathlib import Path

from data.synthetic_event_generator import generate_synthetic_events
from ingest.parse_sessions import parse_sessions
from ingest.build_graph import build_graph
from ingest.flow_metrics import run_metrics
from analysis.event_chokepoints import main as run_chokepoints

def create_dataset_directories(dataset_name):
    """
    Create dataset-specific directories in data/ and outputs/.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        tuple: Paths to the dataset directories (data_dir, output_dir)
    """
    data_dir = f"data/{dataset_name}"
    output_dir = f"outputs/{dataset_name}"
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    return data_dir, output_dir

def generate_dataset_metadata(dataset_name, data_dir, output_dir, users_count=0, events_count=0, sessions_count=0):
    """
    Generate metadata for the dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        data_dir (str): Path to the data directory
        output_dir (str): Path to the output directory
        users_count (int): Number of users in the dataset
        events_count (int): Number of events in the dataset
        sessions_count (int): Number of sessions in the dataset
        
    Returns:
        dict: Dataset metadata
    """
    metadata = {
        "dataset_name": dataset_name,
        "creation_timestamp": datetime.now().isoformat(),
        "num_users": users_count,
        "num_events": events_count,
        "num_sessions": sessions_count
    }
    
    # Write metadata to file
    metadata_path = f"{output_dir}/dataset_info.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def run_all_stages(dataset_name="default", users=100, events_per_user=50, input_file=None):
    """
    Run all stages of the TeloMesh pipeline.
    
    Args:
        dataset_name (str): Name of the dataset to create
        users (int): Number of users to generate in synthetic data.
        events_per_user (int): Average number of events per user.
        input_file (str): Path to input events file (instead of generating synthetic data)
    """
    print("\n[SIMPLE] TeloMesh Pipeline - SIMPLE Stage")
    print("=" * 50)
    
    # Create dataset directories
    data_dir, output_dir = create_dataset_directories(dataset_name)
    
    # Stage 1: Data acquisition (synthetic or existing)
    print("\n[Stage 1] Data acquisition...")
    start_time = time.time()
    
    events_path = f"{data_dir}/events.csv"
    
    if input_file:
        # Use existing data
        print(f"Using existing data from {input_file}")
        # Copy input file to dataset directory
        shutil.copy(input_file, events_path)
        # Get basic stats from the file
        import pandas as pd
        df = pd.read_csv(events_path)
        total_events = len(df)
        unique_users = df["user_id"].nunique() if "user_id" in df.columns else 0
        print(f"[OK] Copied {total_events} events for {unique_users} users.")
    else:
        # Generate synthetic data
        print(f"Generating synthetic events for {users} users with ~{events_per_user} events each...")
        df = generate_synthetic_events(users, events_per_user)
        # Save to dataset directory
        df.to_csv(events_path, index=False)
        total_events = len(df)
        unique_users = df["user_id"].nunique()
        print(f"[OK] Generated {total_events} events for {unique_users} users.")
    
    print(f"[OK] Events saved to {events_path}")
    print(f"[TIME] Time taken: {time.time() - start_time:.2f} seconds")
    
    # Stage 2: Parse sessions
    print("\n[Stage 2] Parsing sessions...")
    start_time = time.time()
    session_flows_path = f"{output_dir}/session_flows.csv"
    df = parse_sessions(events_path, session_flows_path)
    unique_users = df["user_id"].nunique()
    unique_sessions = df["session_id"].nunique()
    total_events = len(df)
    print(f"[OK] Parsed {total_events} events from {unique_sessions} sessions for {unique_users} users.")
    print(f"[OK] Output saved to {session_flows_path}")
    print(f"[TIME] Time taken: {time.time() - start_time:.2f} seconds")
    
    # Stage 3: Build graph
    print("\n[Stage 3] Building user journey graph...")
    start_time = time.time()
    graph_path = f"{output_dir}/user_graph.gpickle"
    G = build_graph(session_flows_path, graph_path)
    num_nodes = len(G.nodes)
    num_edges = len(G.edges)
    print(f"[OK] Graph built with {num_nodes} nodes and {num_edges} edges.")
    print(f"[OK] Output saved to {graph_path}")
    print(f"[TIME] Time taken: {time.time() - start_time:.2f} seconds")
    
    # Stage 4: Flow metrics validation
    print("\n[Stage 4] Validating flow metrics...")
    start_time = time.time()
    metrics_path = f"{output_dir}/metrics.json"
    stats_path = f"{output_dir}/session_stats.log"
    # Redirect metrics to dataset directory
    run_metrics(session_flows_path, graph_path, metrics_path, stats_path)
    print(f"[OK] Flow metrics validation complete.")
    print(f"[OK] Output saved to {stats_path}")
    print(f"[TIME] Time taken: {time.time() - start_time:.2f} seconds")
    
    # LOVABLE Stage components
    print("\n[LOVABLE] TeloMesh Pipeline - LOVABLE Stage")
    print("=" * 50)
    
    # Stage 5: Event chokepoints analysis
    print("\n[Stage 5] Analyzing event chokepoints...")
    start_time = time.time()
    friction_df, fragile_flows_df, node_map = run_chokepoints(session_flows_path, graph_path, output_dir, fast=True)
    print(f"[OK] Event chokepoints analysis complete.")
    print(f"[OK] Outputs saved to:")
    print(f"  - {output_dir}/event_chokepoints.csv")
    print(f"  - {output_dir}/high_friction_flows.csv")
    print(f"  - {output_dir}/friction_node_map.json")
    print(f"[TIME] Time taken: {time.time() - start_time:.2f} seconds")
    
    # Generate dataset metadata
    generate_dataset_metadata(
        dataset_name, 
        data_dir, 
        output_dir, 
        unique_users, 
        total_events, 
        unique_sessions
    )
    
    # Stage 6: Dashboard creation
    print("\n[Stage 6] Dashboard creation...")
    print(f"[OK] Dashboard UI ready.")
    print(f"[OK] To view the dashboard, run: streamlit run ui/dashboard.py")
    
    # Summary
    print("\n[DONE] All stages completed successfully!")
    print("=" * 50)
    print("[FILES] Output files:")
    print(f"  - {events_path}")
    print(f"  - {session_flows_path}")
    print(f"  - {graph_path}")
    print(f"  - {stats_path}")
    print(f"  - {output_dir}/event_chokepoints.csv")
    print(f"  - {output_dir}/high_friction_flows.csv")
    print(f"  - {output_dir}/friction_node_map.json")
    print(f"  - {output_dir}/dataset_info.json")
    print("\n[DASHBOARD] To view the dashboard:")
    print("  streamlit run ui/dashboard.py")

def main():
    """Main function to run the TeloMesh pipeline."""
    parser = argparse.ArgumentParser(description="TeloMesh User Journey Analysis Pipeline")
    parser.add_argument("--stage", type=str, choices=["all", "synthetic", "parse", "graph", "metrics", "chokepoints", "dashboard"], 
                        default="all", help="Which stage to run")
    parser.add_argument("--users", type=int, default=100, help="Number of users to generate")
    parser.add_argument("--events", type=int, default=50, help="Average number of events per user")
    parser.add_argument("--fast", action="store_true", help="Skip slow scaling tests and detailed output")
    parser.add_argument("--dataset", type=str, default="default", help="Name of the dataset to create")
    parser.add_argument("--input", type=str, help="Path to input events file (instead of generating synthetic data)")
    
    args = parser.parse_args()
    
    # Ensure output directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create dataset directories
    data_dir, output_dir = create_dataset_directories(args.dataset)
    
    if args.stage == "all":
        run_all_stages(args.dataset, args.users, args.events, args.input)
    elif args.stage == "synthetic":
        print("[STAGE] Generating synthetic events...")
        # Create dataset directories
        data_dir, output_dir = create_dataset_directories(args.dataset)
        # Generate synthetic data
        events_path = f"{data_dir}/events.csv"
        df = generate_synthetic_events(args.users, args.events)
        df.to_csv(events_path, index=False)
        print(f"[OK] Generated {len(df)} events -> {events_path}")
    elif args.stage == "parse":
        print("[STAGE] Parsing sessions...")
        # Create dataset directories
        data_dir, output_dir = create_dataset_directories(args.dataset)
        # Define input and output paths
        events_path = f"{data_dir}/events.csv"
        session_flows_path = f"{output_dir}/session_flows.csv"
        # Check if input file exists, otherwise use default path
        if not os.path.exists(events_path) and args.input:
            # Copy input file to dataset directory if provided
            shutil.copy(args.input, events_path)
        # Parse sessions
        df = parse_sessions(events_path, session_flows_path)
        print(f"[OK] Parsed {len(df)} events into structured flows -> {session_flows_path}")
    elif args.stage == "graph":
        print("[STAGE] Building user journey graph...")
        # Create dataset directories
        data_dir, output_dir = create_dataset_directories(args.dataset)
        # Define input and output paths
        session_flows_path = f"{output_dir}/session_flows.csv"
        graph_path = f"{output_dir}/user_graph.gpickle"
        # Build graph
        G = build_graph(session_flows_path, graph_path)
        print(f"[OK] Graph built with {len(G.nodes)} nodes and {len(G.edges)} edges -> {graph_path}")
    elif args.stage == "metrics":
        print("[STAGE] Validating flow metrics...")
        # Create dataset directories
        data_dir, output_dir = create_dataset_directories(args.dataset)
        # Define input and output paths
        session_flows_path = f"{output_dir}/session_flows.csv"
        graph_path = f"{output_dir}/user_graph.gpickle"
        metrics_path = f"{output_dir}/metrics.json"
        stats_path = f"{output_dir}/session_stats.log"
        # Run metrics
        run_metrics(session_flows_path, graph_path, metrics_path, stats_path)
        print(f"[OK] Flow metrics validation complete -> {stats_path}")
    elif args.stage == "chokepoints":
        print("[STAGE] Analyzing event chokepoints...")
        # Create dataset directories
        data_dir, output_dir = create_dataset_directories(args.dataset)
        # Define input and output paths
        session_flows_path = f"{output_dir}/session_flows.csv"
        graph_path = f"{output_dir}/user_graph.gpickle"
        # Run chokepoints analysis
        run_chokepoints(session_flows_path, graph_path, output_dir, fast=args.fast)
        print(f"[OK] Event chokepoints analysis complete -> {output_dir}/event_chokepoints.csv")
    elif args.stage == "dashboard":
        print("[DASHBOARD] To run the dashboard:")
        print("streamlit run ui/dashboard.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 