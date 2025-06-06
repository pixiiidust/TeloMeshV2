#!/usr/bin/env python3
"""
TeloMesh - User Journey Analysis Pipeline
"""

import argparse
import os
import sys
import time

from data.synthetic_event_generator import generate_synthetic_events
from ingest.parse_sessions import parse_sessions
from ingest.build_graph import build_graph
from ingest.flow_metrics import run_metrics
from analysis.event_chokepoints import main as run_chokepoints

def run_all_stages(users=100, events_per_user=50):
    """
    Run all stages of the TeloMesh pipeline.
    
    Args:
        users (int): Number of users to generate in synthetic data.
        events_per_user (int): Average number of events per user.
    """
    print("\n📊 TeloMesh Pipeline - SIMPLE Stage")
    print("=" * 50)
    
    # Stage 1: Generate synthetic events
    print("\n🔹 Stage 1: Generating synthetic events...")
    start_time = time.time()
    df = generate_synthetic_events(users, events_per_user)
    total_events = len(df)
    unique_users = df["user_id"].nunique()
    print(f"✅ Generated {total_events} events for {unique_users} users.")
    print(f"✅ Output saved to data/events.csv")
    print(f"⏱️  Time taken: {time.time() - start_time:.2f} seconds")
    
    # Stage 2: Parse sessions
    print("\n🔹 Stage 2: Parsing sessions...")
    start_time = time.time()
    df = parse_sessions("data/events.csv", "outputs/session_flows.csv")
    unique_users = df["user_id"].nunique()
    unique_sessions = df["session_id"].nunique()
    total_events = len(df)
    print(f"✅ Parsed {total_events} events from {unique_sessions} sessions for {unique_users} users.")
    print(f"✅ Output saved to outputs/session_flows.csv")
    print(f"⏱️  Time taken: {time.time() - start_time:.2f} seconds")
    
    # Stage 3: Build graph
    print("\n🔹 Stage 3: Building user journey graph...")
    start_time = time.time()
    G = build_graph("outputs/session_flows.csv", "outputs/user_graph.gpickle")
    num_nodes = len(G.nodes)
    num_edges = len(G.edges)
    print(f"✅ Graph built with {num_nodes} nodes and {num_edges} edges.")
    print(f"✅ Output saved to outputs/user_graph.gpickle")
    print(f"⏱️  Time taken: {time.time() - start_time:.2f} seconds")
    
    # Stage 4: Flow metrics validation
    print("\n🔹 Stage 4: Validating flow metrics...")
    start_time = time.time()
    run_metrics()
    print(f"✅ Flow metrics validation complete.")
    print(f"✅ Output saved to logs/session_stats.log")
    print(f"⏱️  Time taken: {time.time() - start_time:.2f} seconds")
    
    # LOVABLE Stage components
    print("\n📊 TeloMesh Pipeline - LOVABLE Stage")
    print("=" * 50)
    
    # Stage 5: Event chokepoints analysis
    print("\n🔹 Stage 5: Analyzing event chokepoints...")
    start_time = time.time()
    friction_df, fragile_flows_df, node_map = run_chokepoints("outputs/session_flows.csv", "outputs/user_graph.gpickle", "outputs", fast=True)
    print(f"✅ Event chokepoints analysis complete.")
    print(f"✅ Outputs saved to:")
    print(f"  - outputs/event_chokepoints.csv")
    print(f"  - outputs/high_friction_flows.csv")
    print(f"  - outputs/friction_node_map.json")
    print(f"⏱️  Time taken: {time.time() - start_time:.2f} seconds")
    
    # Stage 6: Dashboard creation
    print("\n🔹 Stage 6: Dashboard creation...")
    print(f"✅ Dashboard UI ready.")
    print(f"✅ To view the dashboard, run: streamlit run ui/dashboard.py")
    
    # Summary
    print("\n🎉 All stages completed successfully!")
    print("=" * 50)
    print("📂 Output files:")
    print("  - data/events.csv")
    print("  - outputs/session_flows.csv")
    print("  - outputs/user_graph.gpickle")
    print("  - logs/session_stats.log")
    print("  - outputs/event_chokepoints.csv")
    print("  - outputs/high_friction_flows.csv")
    print("  - outputs/friction_node_map.json")
    print("\n📊 To view the dashboard:")
    print("  streamlit run ui/dashboard.py")

def main():
    """Main function to run the TeloMesh pipeline."""
    parser = argparse.ArgumentParser(description="TeloMesh User Journey Analysis Pipeline")
    parser.add_argument("--stage", type=str, choices=["all", "synthetic", "parse", "graph", "metrics", "chokepoints", "dashboard"], 
                        default="all", help="Which stage to run")
    parser.add_argument("--users", type=int, default=100, help="Number of users to generate")
    parser.add_argument("--events", type=int, default=50, help="Average number of events per user")
    parser.add_argument("--fast", action="store_true", help="Skip slow scaling tests and detailed output")
    
    args = parser.parse_args()
    
    # Ensure output directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("analysis", exist_ok=True)
    os.makedirs("ui", exist_ok=True)
    
    if args.stage == "all":
        run_all_stages(args.users, args.events)
    elif args.stage == "synthetic":
        print("📊 Generating synthetic events...")
        df = generate_synthetic_events(args.users, args.events)
        print(f"✅ Generated {len(df)} events.")
    elif args.stage == "parse":
        print("📊 Parsing sessions...")
        df = parse_sessions("data/events.csv", "outputs/session_flows.csv")
        print(f"✅ Parsed {len(df)} events into structured flows.")
    elif args.stage == "graph":
        print("📊 Building user journey graph...")
        G = build_graph("outputs/session_flows.csv", "outputs/user_graph.gpickle")
        print(f"✅ Graph built with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    elif args.stage == "metrics":
        print("📊 Validating flow metrics...")
        run_metrics()
        print(f"✅ Flow metrics validation complete.")
    elif args.stage == "chokepoints":
        print("📊 Analyzing event chokepoints...")
        run_chokepoints("outputs/session_flows.csv", "outputs/user_graph.gpickle", "outputs", fast=args.fast)
        print(f"✅ Event chokepoints analysis complete.")
    elif args.stage == "dashboard":
        print("📊 To run the dashboard:")
        print("streamlit run ui/dashboard.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 