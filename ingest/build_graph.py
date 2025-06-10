import os
import pandas as pd
import networkx as nx
import pickle

def build_graph(input_csv="outputs/session_flows.csv", output_graph="outputs/user_graph.gpickle", multi_graph=False):
    """
    Build a directed graph representing user journeys from session flow data.
    
    Args:
        input_csv (str): Path to the input CSV file containing session flows.
        output_graph (str): Path to save the output NetworkX graph.
        multi_graph (bool): If True, create a MultiDiGraph that preserves multiple edges between nodes.
        
    Returns:
        nx.DiGraph or nx.MultiDiGraph: The directed graph representing user journeys.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_graph), exist_ok=True)
    
    # Read the session flows data
    df = pd.read_csv(input_csv)
    
    # Create a directed graph or multi-directed graph based on the parameter
    G = nx.MultiDiGraph() if multi_graph else nx.DiGraph()
    
    # Process each session to build the graph
    for session_id, group in df.groupby("session_id"):
        # Sort by step_index to ensure chronological order
        session_steps = group.sort_values("step_index")
        
        # Add edges for each transition
        for i in range(len(session_steps) - 1):
            from_step = session_steps.iloc[i]
            to_step = session_steps.iloc[i+1]
            
            from_page = from_step["page"]
            to_page = to_step["page"]
            event = to_step["event"]
            
            # Add nodes if they don't exist
            if not G.has_node(from_page):
                G.add_node(from_page)
            
            if not G.has_node(to_page):
                G.add_node(to_page)
            
            if multi_graph:
                # For MultiDiGraph, add a new edge for each transition
                G.add_edge(from_page, to_page, event=event, weight=1)
            else:
                # For DiGraph, update existing edges or add new ones
                if G.has_edge(from_page, to_page):
                    # Check if it's the same event type
                    edge_data = G.get_edge_data(from_page, to_page)
                    if edge_data["event"] == event:
                        # Increment the weight
                        G[from_page][to_page]["weight"] += 1
                    else:
                        # It's a different event type - we'll use a MultiDiGraph in a more advanced version
                        # For now, use the most common event type or update based on count
                        # This is a simplification for the SIMPLE stage
                        if edge_data["weight"] < 1:
                            # Replace with the new event if the old one is less frequent
                            G[from_page][to_page]["event"] = event
                            G[from_page][to_page]["weight"] = 1
                else:
                    # Add a new edge
                    G.add_edge(from_page, to_page, event=event, weight=1)
    
    # Save the graph using pickle
    with open(output_graph, 'wb') as f:
        pickle.dump(G, f)
    
    return G

def main():
    """Main function to build the user journey graph."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build a user journey graph from session flows")
    parser.add_argument("--input", type=str, default="outputs/session_flows.csv", 
                        help="Path to the input CSV file")
    parser.add_argument("--output", type=str, default="outputs/user_graph.gpickle", 
                        help="Path to save the output graph")
    parser.add_argument("--multi", action="store_true", 
                        help="Create a MultiDiGraph that preserves multiple edges between nodes")
    
    args = parser.parse_args()
    
    print(f"Building graph from {args.input}...")
    G = build_graph(args.input, args.output, args.multi)
    
    # Print some stats
    num_nodes = len(G.nodes)
    num_edges = len(G.edges) if not isinstance(G, nx.MultiDiGraph) else G.number_of_edges()
    
    print(f"Graph built with {num_nodes} nodes and {num_edges} edges.")
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Build a user journey graph from session flows.")
    parser.add_argument("--n_users", type=int, default=1000, help="Number of unique users")
    parser.add_argument("--events_per_user", type=int, default=6, help="Average events per user")
    parser.add_argument("--output_path", type=str, default="outputs/user_graph.gpickle", help="Graph output path")
    parser.add_argument("--fast", action="store_true", help="Skip slow scaling tests during development")
    parser.add_argument("--input_path", type=str, default="outputs/session_flows.csv", help="CSV input path")
    parser.add_argument("--multi", action="store_true", help="Create a MultiDiGraph that preserves multiple edges between nodes")

    args = parser.parse_args()

    # Build the graph
    print(f"Building graph from {args.input_path}...")
    G = build_graph(args.input_path, args.output_path, args.multi)
    
    # Print graph statistics
    num_nodes = len(G.nodes)
    num_edges = len(G.edges) if not isinstance(G, nx.MultiDiGraph) else G.number_of_edges()
    
    graph_type = "MultiDiGraph" if isinstance(G, nx.MultiDiGraph) else "DiGraph"
    print(f"✅ {graph_type} built with {num_nodes} nodes and {num_edges} edges → {args.output_path}")

    if not args.fast:
        # Run graph quality checks
        if num_nodes > 0 and num_edges > 0:
            # Check connected components
            if nx.is_directed(G):
                components = list(nx.weakly_connected_components(G))
            else:
                components = list(nx.connected_components(G))
            
            print(f"✅ Connected components: {len(components)}")
            
            if len(components) < 1:
                print("⚠️ Warning: No connected components found")
            
            # Check edge event types
            edge_events = set()
            for u, v, data in G.edges(data=True):
                edge_events.add(data.get("event"))
            
            print(f"✅ Unique edge events: {len(edge_events)}")
            
            if len(edge_events) < 2:
                print("⚠️ Warning: Less than 2 unique edge event types")
            
            # Check node count
            if num_nodes < 10:
                print(f"⚠️ Warning: Insufficient graph complexity: {num_nodes} nodes (less than 10)")
        else:
            print("⚠️ Warning: Empty graph created")
    else:
        print("⚡ FAST_MODE enabled: graph quality checks bypassed.") 