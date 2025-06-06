import os
import json
import logging
import pandas as pd
import networkx as nx
import pickle
from collections import Counter

# CURSOR RULE: BASIC SESSION + GRAPH VALIDATION
# - Compute:
#     - sessions_per_user ≥ 1
#     - average flow length ≥ 3
#     - most common event type (e.g., "Clicked CTA")
# - Output test logs to logs/session_stats.log
# - Raise warning if total sessions < 50 or nodes < 10

def setup_logging():
    """Set up logging to file and console."""
    # Ensure the logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/session_stats.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

def validate_sessions(session_flows_csv="outputs/session_flows.csv"):
    """
    Validate session data for quality metrics.
    
    Args:
        session_flows_csv (str): Path to the session flows CSV file.
        
    Returns:
        dict: Dictionary of session statistics.
    """
    logging.info("Validating session flows...")
    
    # Read session flows
    try:
        df = pd.read_csv(session_flows_csv)
    except Exception as e:
        logging.error(f"Error reading session flows: {e}")
        return {}
    
    # Calculate statistics
    stats = {}
    
    # Total events
    stats["total_events"] = len(df)
    logging.info(f"Total events: {stats['total_events']}")
    
    # Unique users
    unique_users = df["user_id"].nunique()
    stats["unique_users"] = unique_users
    logging.info(f"Unique users: {stats['unique_users']}")
    
    # Total sessions
    total_sessions = df["session_id"].nunique()
    stats["total_sessions"] = total_sessions
    logging.info(f"Total sessions: {stats['total_sessions']}")
    
    # Sessions per user
    stats["sessions_per_user"] = total_sessions / unique_users
    logging.info(f"Sessions per user: {stats['sessions_per_user']:.2f}")
    
    # Check if sessions per user >= 1
    if stats["sessions_per_user"] < 1:
        logging.warning("Sessions per user is less than 1")
    
    # Average flow length (steps per session)
    session_steps = df.groupby("session_id").size()
    stats["avg_flow_length"] = session_steps.mean()
    logging.info(f"Average flow length: {stats['avg_flow_length']:.2f} steps")
    
    # Check if average flow length >= 3
    if stats["avg_flow_length"] < 3:
        logging.warning("Average flow length is less than 3 steps")
    
    # Most common event type
    event_counts = df["event"].value_counts()
    most_common_event = event_counts.index[0]
    stats["most_common_event"] = most_common_event
    stats["most_common_event_count"] = event_counts[most_common_event]
    logging.info(f"Most common event: {most_common_event} ({stats['most_common_event_count']} occurrences)")
    
    # Check if total sessions < 50
    if total_sessions < 50:
        logging.warning(f"Low session volume: {total_sessions} sessions (less than 50)")
    
    return stats

def validate_graph(graph_pickle="outputs/user_graph.gpickle"):
    """
    Validate graph data for quality metrics.
    
    Args:
        graph_pickle (str): Path to the graph pickle file.
        
    Returns:
        dict: Dictionary of graph statistics.
    """
    logging.info("Validating user flow graph...")
    
    # Read graph
    try:
        with open(graph_pickle, 'rb') as f:
            G = pickle.load(f)
    except Exception as e:
        logging.error(f"Error reading graph: {e}")
        return {}
    
    # Calculate statistics
    stats = {}
    
    # Node count
    stats["node_count"] = len(G.nodes)
    logging.info(f"Node count: {stats['node_count']}")
    
    # Check if node count < 10
    if stats["node_count"] < 10:
        logging.warning(f"Insufficient graph complexity: {stats['node_count']} nodes (less than 10)")
    
    # Edge count
    stats["edge_count"] = len(G.edges)
    logging.info(f"Edge count: {stats['edge_count']}")
    
    # Unique events
    edge_events = [data.get("event") for _, _, data in G.edges(data=True)]
    event_counter = Counter(edge_events)
    stats["unique_edge_events"] = len(event_counter)
    logging.info(f"Unique edge events: {stats['unique_edge_events']}")
    
    # Check if there are at least 2 unique event types
    if stats["unique_edge_events"] < 2:
        logging.warning("Less than 2 unique edge event types")
    
    # Most common edge event
    most_common_edge_event = event_counter.most_common(1)[0][0]
    stats["most_common_edge_event"] = most_common_edge_event
    stats["most_common_edge_event_count"] = event_counter[most_common_edge_event]
    logging.info(f"Most common edge event: {most_common_edge_event} ({stats['most_common_edge_event_count']} occurrences)")
    
    # Connected components
    if nx.is_directed(G):
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    
    stats["connected_components"] = len(components)
    logging.info(f"Connected components: {stats['connected_components']}")
    
    # Ensure at least 1 connected component
    if stats["connected_components"] < 1:
        logging.warning("Graph has no connected components")
    
    return stats

def run_metrics():
    """Run all validation metrics and save to file."""
    setup_logging()
    
    logging.info("=" * 50)
    logging.info("TeloMesh Flow Metrics Validation")
    logging.info("=" * 50)
    
    # Validate sessions
    session_stats = validate_sessions()
    
    # Validate graph
    graph_stats = validate_graph()
    
    # Combine all stats
    all_stats = {**session_stats, **graph_stats}
    
    # Convert NumPy types to Python native types for JSON serialization
    for key, value in all_stats.items():
        if hasattr(value, 'item'):  # Check if it's a NumPy type
            all_stats[key] = value.item()  # Convert to native Python type
    
    # Save stats as JSON
    try:
        with open("logs/metrics.json", "w", encoding="utf-8") as f:
            json.dump(all_stats, f, indent=2)
        logging.info("Metrics saved to logs/metrics.json")
    except Exception as e:
        logging.error(f"Error saving metrics to JSON: {e}")
    
    # Final summary
    logging.info("\nValidation Summary:")
    logging.info(f"Sessions per user: {all_stats.get('sessions_per_user', 0):.2f} (target: >= 1)")
    logging.info(f"Average flow length: {all_stats.get('avg_flow_length', 0):.2f} steps (target: >= 3)")
    logging.info(f"Most common event: {all_stats.get('most_common_event', 'N/A')}")
    logging.info(f"Node count: {all_stats.get('node_count', 0)} (target: >= 10)")
    logging.info(f"Edge count: {all_stats.get('edge_count', 0)}")
    
    # Report overall validation result
    if (all_stats.get('sessions_per_user', 0) >= 1 and 
        all_stats.get('avg_flow_length', 0) >= 3 and 
        all_stats.get('node_count', 0) >= 10 and
        all_stats.get('total_sessions', 0) >= 50 and
        all_stats.get('unique_edge_events', 0) >= 2):
        logging.info("PASS: All validation criteria met!")
    else:
        logging.warning("WARNING: Some validation criteria not met. See warnings above.")

def main():
    """Main function to run flow metrics validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate session and graph data")
    parser.add_argument("--sessions", type=str, default="outputs/session_flows.csv", 
                        help="Path to the session flows CSV file")
    parser.add_argument("--graph", type=str, default="outputs/user_graph.gpickle", 
                        help="Path to the graph pickle file")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Validate sessions
    session_stats = validate_sessions(args.sessions)
    
    # Validate graph
    graph_stats = validate_graph(args.graph)
    
    # Final message
    print("Flow metrics validation complete. See logs/session_stats.log for details.")

if __name__ == "__main__":
    main() 