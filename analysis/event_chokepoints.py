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
        output_dir (str): Directory to save output files.
        
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export WSJF Friction Scores
    chokepoints_path = os.path.join(output_dir, "event_chokepoints.csv")
    df.to_csv(chokepoints_path, index=False)
    print(f"Exported {len(df)} chokepoints to {chokepoints_path}")
    
    # Export fragile flows
    flows_path = os.path.join(output_dir, "high_friction_flows.csv")
    if flow_df is not None and len(flow_df) > 0:
        flow_df.to_csv(flows_path, index=False)
        print(f"Exported {flow_df['session_id'].nunique()} fragile flows to {flows_path}")
    else:
        # Create an empty file with column headers to ensure file exists
        empty_df = pd.DataFrame(columns=[
            'session_id', 'step_index', 'page', 'event', 
            'WSJF_Friction_Score', 'is_chokepoint', 'user_id'
        ])
        empty_df.to_csv(flows_path, index=False)
        print(f"Exported empty fragile flows file to {flows_path} (no fragile flows detected)")
    
    # Export node map
    node_map_path = os.path.join(output_dir, "friction_node_map.json")
    with open(node_map_path, 'w') as f:
        json.dump(node_map, f, indent=2)
    print(f"Exported node map with {len(node_map)} pages to {node_map_path}")

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

def convert_to_digraph(G):
    """
    Convert a MultiDiGraph to a DiGraph by collapsing multiple edges.
    
    Args:
        G: nx.Graph or nx.DiGraph or nx.MultiDiGraph
        
    Returns:
        nx.DiGraph: Collapsed graph with summed weights
    """
    if isinstance(G, nx.MultiDiGraph):
        DG = nx.DiGraph()
        
        # Copy nodes and their attributes
        for node, data in G.nodes(data=True):
            DG.add_node(node, **data)
        
        # Collapse edges, summing weights
        for u, v in G.edges():
            # Get all edges between u and v
            all_edges = G.get_edge_data(u, v)
            
            # Sum weights across all edges
            total_weight = sum(data.get('weight', 1) for data in all_edges.values())
            
            # Add edge with summed weight
            DG.add_edge(u, v, weight=total_weight)
            
        return DG
    elif isinstance(G, nx.DiGraph):
        return G
    else:
        # If it's an undirected graph, convert to directed
        return nx.DiGraph(G)

def compute_fractal_dimension(G, max_radius=10):
    """
    Compute the fractal dimension of a graph using box-counting.
    
    Args:
        G: nx.Graph or nx.DiGraph or nx.MultiDiGraph
        max_radius: Maximum radius for box-counting
        
    Returns:
        float: Estimated fractal dimension D, typically in [1.0, 2.0]
    """
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Convert to DiGraph if needed
    G = convert_to_digraph(G)
    
    # Check if graph is too small for reliable calculation
    if len(G.nodes()) < 10:
        logger.warning(f"Graph too small ({len(G.nodes())} nodes) for reliable fractal dimension. Using default D=1.0")
        return 1.0
    
    # Try to use pyunicorn if available
    try:
        from pyunicorn import Network
        # Convert NetworkX graph to pyunicorn Network
        adj_matrix = nx.to_numpy_array(G)
        network = Network(adjacency=adj_matrix)
        D = float(network.fractal_dimension())
        # Ensure value is positive and in reasonable range
        if D <= 0 or not np.isfinite(D):
            logger.warning(f"Invalid fractal dimension {D} from pyunicorn. Using fallback method.")
            raise ValueError("Invalid fractal dimension")
        return D
    except (ImportError, ValueError) as e:
        if isinstance(e, ImportError):
            logger.warning("pyunicorn not available, using fallback box-counting method")
        
        # Fallback: Custom box-counting implementation
        radii = np.logspace(0, np.log10(max_radius), 10)
        box_counts = []
        
        # Get distance matrix
        try:
            dist_matrix = dict(nx.all_pairs_shortest_path_length(G))
        except Exception as e:
            logger.warning(f"Error computing distance matrix: {e}. Using default D=1.0")
            return 1.0
        
        for radius in radii:
            # Count minimum number of boxes needed to cover the graph
            covered = set()
            boxes = 0
            
            nodes = list(G.nodes())
            while len(covered) < len(nodes):
                # Find the node that covers the most uncovered nodes
                best_node = None
                best_coverage = 0
                best_coverage_set = set()
                
                for node in nodes:
                    if node in covered:
                        continue
                    
                    # Count nodes within radius of this node
                    coverage = set()
                    for other in nodes:
                        if other in covered:
                            continue
                        try:
                            if dist_matrix[node][other] <= radius:
                                coverage.add(other)
                        except KeyError:
                            # If there's no path, ignore
                            pass
                    
                    if len(coverage) > best_coverage:
                        best_node = node
                        best_coverage = len(coverage)
                        best_coverage_set = coverage
                
                # Add the best box
                if best_node is not None:
                    covered.update(best_coverage_set)
                    boxes += 1
                else:
                    # If we can't cover more nodes, break
                    break
            
            box_counts.append(boxes)
        
        # Linear regression on log-log plot
        log_radii = np.log(radii)
        log_counts = np.log(box_counts)
        
        # Use only valid points (non-zero counts)
        valid = np.isfinite(log_counts)
        if np.sum(valid) < 2:
            logger.warning("Not enough valid points for fractal dimension calculation. Using default D=1.0")
            return 1.0
        
        try:
            # Try np.polyfit which returns a polynomial coefficients array
            coefficients = np.polyfit(log_radii[valid], log_counts[valid], 1)
            slope = coefficients[0]  # First coefficient is the slope
            D = float(-slope)
            
            # Validate result and ensure it's in the typical range
            if D <= 0 or not np.isfinite(D):
                logger.warning(f"Invalid fractal dimension D={D}. Using default D=1.0")
                return 1.0
                
            # Clamp to reasonable range
            if D > 3.0:  # No real-world network should exceed D=3
                logger.warning(f"Unusually high fractal dimension D={D}. Clamping to D=3.0")
                return 3.0
                
            return D
        except Exception as e:
            logger.warning(f"Error in polyfit: {e}")
            # Fallback to a simpler calculation if polyfit fails
            if len(log_radii[valid]) > 0:
                # Use first and last point to estimate slope
                first_idx = np.where(valid)[0][0]
                last_idx = np.where(valid)[0][-1]
                slope = (log_counts[last_idx] - log_counts[first_idx]) / (log_radii[last_idx] - log_radii[first_idx])
                D = float(-slope)
                
                # Validate result
                if D <= 0 or not np.isfinite(D):
                    logger.warning(f"Invalid fractal dimension D={D} from fallback. Using default D=1.0")
                    return 1.0
                    
                # Clamp to reasonable range
                if D > 3.0:
                    logger.warning(f"Unusually high fractal dimension D={D}. Clamping to D=3.0")
                    return 3.0
                    
                return D
            else:
                logger.warning("No valid points for fractal dimension calculation. Using default D=1.0")
                return 1.0  # Default value if all else fails

def compute_power_law_alpha(G):
    """
    Compute the power-law exponent alpha for the degree distribution.
    
    Args:
        G: nx.Graph or nx.DiGraph or nx.MultiDiGraph
        
    Returns:
        float: Estimated power-law exponent alpha, typically in range [1.5, 3.0]
    """
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Convert to DiGraph if needed
    G = convert_to_digraph(G)
    
    # Check if graph is too small for reliable calculation
    if len(G.nodes()) < 10:
        logger.warning(f"Graph too small ({len(G.nodes())} nodes) for reliable power-law fit. Using default α=2.5")
        return 2.5
    
    # Get degrees
    degrees = [d for _, d in G.degree()]
    
    # Check if we have enough unique degree values
    unique_degrees = set(degrees)
    if len(unique_degrees) < 3:
        logger.warning(f"Not enough unique degree values ({len(unique_degrees)}) for power-law fit. Using default α=2.5")
        return 2.5
    
    # Try to use powerlaw package
    try:
        import powerlaw
        try:
            fit = powerlaw.Fit(degrees)
            alpha = float(fit.power_law.alpha)
            
            # Validate result
            if not np.isfinite(alpha) or alpha <= 1.0:
                logger.warning(f"Invalid power-law exponent α={alpha}. Using default α=2.5")
                return 2.5
                
            # Clamp to reasonable range
            if alpha > 5.0:  # Extremely rare to see alpha > 5 in real networks
                logger.warning(f"Unusually high power-law exponent α={alpha}. Clamping to α=5.0")
                return 5.0
                
            return alpha
        except Exception as e:
            logger.warning(f"Error in powerlaw fit: {e}. Using fallback method.")
            # Continue to fallback method
    except ImportError:
        logger.warning("powerlaw package not available, using fallback method")
    
    # Fallback: Use linear regression on log-log plot
    from scipy import stats
    
    # Count degree frequencies
    degree_counts = {}
    for d in degrees:
        if d not in degree_counts:
            degree_counts[d] = 0
        degree_counts[d] += 1
    
    # Calculate P(k)
    n = len(degrees)
    pk = {k: count/n for k, count in degree_counts.items()}
    
    # Get log-log data points
    ks = sorted(pk.keys())
    # Filter out zeros to avoid log(0)
    ks = [k for k in ks if k > 0 and pk[k] > 0]
    
    # Check if we have enough data points after filtering
    if len(ks) < 2:
        logger.warning(f"Not enough data points for power-law fit after filtering. Using default α=2.5")
        return 2.5
        
    log_ks = np.log(ks)
    log_pks = np.log([pk[k] for k in ks])
    
    # Linear regression
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_ks, log_pks)
        
        # Alpha is negative slope + 1
        alpha = float(-slope + 1)
        
        # Validate result
        if not np.isfinite(alpha) or alpha <= 1.0:
            logger.warning(f"Invalid power-law exponent α={alpha} from regression. Using default α=2.5")
            return 2.5
            
        # Clamp to reasonable range
        if alpha > 5.0:
            logger.warning(f"Unusually high power-law exponent α={alpha}. Clamping to α=5.0")
            return 5.0
            
        return alpha
    except Exception as e:
        logger.warning(f"Error in linear regression: {e}. Using default α=2.5")
        return 2.5

def detect_repeating_subgraphs(G, max_length=4):
    """
    Detect recurring subgraphs in the graph, focusing on cycles and repeated paths.
    
    Args:
        G: nx.Graph or nx.DiGraph or nx.MultiDiGraph
        max_length: Maximum subgraph size to detect
        
    Returns:
        List[Tuple]: List of node tuples that form recurring subgraphs
    """
    # Convert to DiGraph if needed
    G = convert_to_digraph(G)
    
    repeating_subgraphs = []
    
    # Detect cycles of length 2 (back-and-forth patterns)
    for u, v in G.edges():
        if G.has_edge(v, u):
            repeating_subgraphs.append((u, v))
    
    # Detect cycles of length 3
    if max_length >= 3:
        for u in G.nodes():
            for v in G.neighbors(u):
                for w in G.neighbors(v):
                    if G.has_edge(w, u):
                        repeating_subgraphs.append((u, v, w))
    
    # Detect cycles of length 4
    if max_length >= 4:
        for u in G.nodes():
            for v in G.neighbors(u):
                for w in G.neighbors(v):
                    if w == u:
                        continue  # Skip 2-cycles
                    for x in G.neighbors(w):
                        if x == v or x == u:
                            continue  # Skip 2-cycles and 3-cycles
                        if G.has_edge(x, u):
                            repeating_subgraphs.append((u, v, w, x))
    
    return repeating_subgraphs

def simulate_percolation(G, ranked_nodes=None, threshold=0.5, fast=False):
    """
    Simulate percolation by removing nodes in order and finding when the graph collapses.
    
    Args:
        G: nx.Graph or nx.DiGraph or nx.MultiDiGraph
        ranked_nodes: List of nodes in order of removal (default: by betweenness)
        threshold: Collapse threshold as fraction of original size
        fast: If True, use sampling for faster approximation
        
    Returns:
        float: Percolation threshold (fraction of nodes removed when collapse occurs)
    """
    # Convert to DiGraph if needed
    G = convert_to_digraph(G)
    
    # If no ranked_nodes provided, rank by betweenness
    if ranked_nodes is None:
        betweenness = nx.betweenness_centrality(G, weight='weight')
        ranked_nodes = sorted(betweenness.keys(), key=lambda n: betweenness[n], reverse=True)
    
    # Get original largest component size
    original_size = len(max(nx.weakly_connected_components(G), key=len))
    target_size = original_size * threshold
    
    # Create working copy of the graph
    G_copy = G.copy()
    
    # Track progress
    step_size = 1
    if fast and len(ranked_nodes) > 100:
        step_size = len(ranked_nodes) // 20  # Check only 20 points
    
    nodes_removed = 0
    for i in range(0, len(ranked_nodes), step_size):
        nodes_to_remove = ranked_nodes[i:i+step_size]
        G_copy.remove_nodes_from(nodes_to_remove)
        nodes_removed += len(nodes_to_remove)
        
        # Check if graph has collapsed
        if len(G_copy) == 0:
            return 1.0  # All nodes removed
        
        components = list(nx.weakly_connected_components(G_copy))
        if not components:
            return 1.0  # No components left
            
        largest_component = max(components, key=len)
        if len(largest_component) <= target_size:
            return nodes_removed / len(G)
    
    return 1.0  # Graph never collapsed

def compute_fractal_betweenness(G, repeating_subgraphs=None, centrality=None):
    """
    Compute fractal betweenness by combining betweenness centrality with subgraph membership.
    
    Args:
        G: nx.Graph or nx.DiGraph or nx.MultiDiGraph
        repeating_subgraphs: List of recurring subgraphs (if None, will be computed)
        centrality: Dict of betweenness centrality values (if None, will be computed)
        
    Returns:
        Dict: Fractal betweenness scores for each node
    """
    # Convert to DiGraph if needed
    G = convert_to_digraph(G)
    
    # Compute repeating subgraphs if not provided
    if repeating_subgraphs is None:
        repeating_subgraphs = detect_repeating_subgraphs(G)
    
    # Compute betweenness centrality if not provided
    if centrality is None:
        centrality = nx.betweenness_centrality(G, weight='weight')
    
    # Count subgraph membership for each node
    subgraph_counts = {node: 0 for node in G.nodes()}
    for subgraph in repeating_subgraphs:
        for node in subgraph:
            subgraph_counts[node] += 1
    
    # Combine with betweenness centrality
    fractal_betweenness = {}
    for node in G.nodes():
        # FB(n) = betweenness(n) × subgraph_count(n)
        fb = centrality.get(node, 0) * (1 + subgraph_counts.get(node, 0))
        fractal_betweenness[node] = fb
    
    return fractal_betweenness

def build_decision_table(G, D, alpha, FB, threshold, chokepoints, cc=None):
    """
    Build a decision table with UX recommendations based on graph metrics.
    
    Args:
        G: nx.Graph or nx.DiGraph or nx.MultiDiGraph
        D: float - Fractal dimension
        alpha: float - Power-law exponent
        FB: dict - Fractal betweenness scores
        threshold: float - Percolation threshold
        chokepoints: pd.DataFrame - DataFrame with WSJF Friction Scores
        cc: float - Clustering coefficient (optional)
        
    Returns:
        pd.DataFrame: Decision table with UX recommendations
    """
    # Create a new DataFrame
    table = []
    
    # Get node-level WSJF scores (max per page)
    node_scores = {}
    for page in chokepoints['page'].unique():
        node_scores[page] = chokepoints[chokepoints['page'] == page]['WSJF_Friction_Score'].max()
    
    # Add all nodes from the graph
    for node in G.nodes():
        # Skip non-string nodes (should be rare in real data)
        if not isinstance(node, str):
            continue
            
        # Get metrics for this node
        wsjf_score = node_scores.get(node, 0)
        fb_score = FB.get(node, 0)
        
        # Determine if node is critical based on percolation
        # Nodes with high betweenness are typically critical
        is_critical = fb_score > (sum(FB.values()) / len(FB)) if FB else False
        
        # Determine UX label based on network properties
        ux_label = "standard"
        suggested_action = "No action needed"
        
        # Use fractal dimension to determine flow type
        if D <= 1.2:
            # Linear/simple paths
            ux_label = "linear bottleneck" if wsjf_score > 0.1 else "linear path"
            suggested_action = "Redesign with high priority" if wsjf_score > 0.1 else "Monitor usage patterns"
        elif D <= 1.7:
            # Tree-like structure
            ux_label = "tree bottleneck" if wsjf_score > 0.1 else "tree branch"
            suggested_action = "Add shortcuts" if wsjf_score > 0.1 else "Consider simplifying"
        else:
            # Complex structure
            ux_label = "complex hub" if wsjf_score > 0.1 else "complex connector"
            suggested_action = "Simplify navigation" if wsjf_score > 0.1 else "Consider restructuring"
        
        # Adjust based on clustering coefficient if provided
        if cc is not None:
            if cc < 0.2:
                # Low clustering means disjointed experience
                if wsjf_score > 0.1:
                    ux_label = "disjointed bottleneck"
                    suggested_action = "Improve connections between related features"
            elif cc > 0.6:
                # High clustering means potentially redundant paths
                if wsjf_score > 0.1:
                    ux_label = "redundant bottleneck"
                    suggested_action = "Consolidate duplicate functionality"
        
        # Create table row
        table.append({
            "node": node,
            "D": D,
            "alpha": alpha,
            "FB": fb_score,
            "percolation_role": "critical" if is_critical else "standard",
            "wsjf_score": wsjf_score,
            "ux_label": ux_label,
            "suggested_action": suggested_action
        })
    
    # Convert to DataFrame
    result = pd.DataFrame(table)
    
    # If empty, return an empty DataFrame with correct columns
    if result.empty:
        return pd.DataFrame(columns=[
            "node", "D", "alpha", "FB", "percolation_role", 
            "wsjf_score", "ux_label", "suggested_action"
        ])
    
    # Sort by WSJF score
    result = result.sort_values("wsjf_score", ascending=False)
    
    return result

def compute_clustering_coefficient(G):
    """
    Compute the average clustering coefficient of a graph.
    
    The clustering coefficient measures how interconnected the graph is,
    based on the ratio of triangles to connected triples.
    
    Args:
        G: nx.Graph or nx.DiGraph or nx.MultiDiGraph
        
    Returns:
        float: Average clustering coefficient, typically in range [0, 1]
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Convert to DiGraph if needed
    G = convert_to_digraph(G)
    
    # Check if graph is too small
    if len(G.nodes()) < 3:
        logger.warning(f"Graph too small ({len(G.nodes())} nodes) for clustering coefficient. Using default 0.0")
        return 0.0
    
    try:
        # For directed graphs, consider the underlying undirected graph
        undirected = G.to_undirected()
        
        # Calculate the clustering coefficient
        clustering = nx.average_clustering(undirected)
        
        # Check if valid
        if clustering < 0 or clustering > 1:
            logger.warning(f"Invalid clustering coefficient {clustering}. Using default 0.0")
            return 0.0
            
        return float(clustering)
    except Exception as e:
        logger.warning(f"Error calculating clustering coefficient: {e}. Using default 0.0")
        return 0.0

def main(input_flows="outputs/session_flows.csv", 
         input_graph="outputs/user_graph.gpickle",
         input_graph_multi="outputs/user_graph_multi.gpickle",
         output_dir="outputs",
         fast=False):
    """
    Main function to run the event chokepoints analysis.
    
    Args:
        input_flows (str): Path to the session flows CSV.
        input_graph (str): Path to the user graph pickle.
        input_graph_multi (str): Path to the multi-graph pickle (for enhanced analysis).
        output_dir (str): Directory to save output files.
        fast (bool): Whether to use fast mode for performance optimization.
        
    Returns:
        Tuple: (friction_df, fragile_flows_df, node_map)
    """
    print("\n[ANALYSIS] TeloMesh Event Chokepoints Analysis")
    print("==================================================")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    import logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(lineno)d %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Load session flows
    print(f"Loading session flows from {input_flows}...")
    session_df = pd.read_csv(input_flows)
    print(f"Loaded {len(session_df)} steps from {session_df['session_id'].nunique()} sessions.")
    
    # Load user graph
    print(f"Loading graph from {input_graph}...")
    with open(input_graph, 'rb') as f:
        G = pickle.load(f)
    print(f"Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges.")
    
    # Load multi-graph if available (for enhanced analysis)
    multi_G = None
    if os.path.exists(input_graph_multi):
        print(f"Loading multi-graph from {input_graph_multi}...")
        with open(input_graph_multi, 'rb') as f:
            multi_G = pickle.load(f)
        print(f"Loaded multi-graph with {len(multi_G.nodes())} nodes and {len(multi_G.edges())} edges.")
    
    # Compute exit rates
    print("Computing exit rates...")
    exit_df = compute_exit_rates(session_df)
    
    # Compute betweenness centrality
    print("Computing betweenness centrality...")
    centrality = compute_betweenness(G)
    
    # Compute WSJF Friction Score
    print("Computing WSJF Friction Scores...")
    chokepoints_df = compute_wsjf_friction(exit_df, centrality)
    
    # Detect fragile flows
    print("Detecting fragile flows...")
    fragile_flows_df = detect_fragile_flows(session_df, chokepoints_df)
    
    # Create node map
    node_map = create_node_map(chokepoints_df)
    
    # Advanced network analysis
    # Compute fractal dimension
    print("Computing fractal dimension...")
    D = compute_fractal_dimension(G)
    print(f"Fractal dimension D = {D:.3f}")
    
    # Compute power law exponent
    print("Computing power-law exponent...")
    alpha = compute_power_law_alpha(G)
    # Use 'a' instead of the Greek alpha character to avoid encoding issues
    print(f"Power-law exponent a = {alpha:.3f}")
    
    # Compute clustering coefficient
    print("Computing clustering coefficient...")
    cc = compute_clustering_coefficient(G)
    print(f"Clustering coefficient = {cc:.3f}")
    
    # Detect repeating subgraphs
    print("Detecting repeating subgraphs...")
    repeating_subgraphs = detect_repeating_subgraphs(G, max_length=4)
    print(f"Detected {len(repeating_subgraphs)} repeating subgraphs.")
    
    # Simulate percolation
    print("Simulating percolation...")
    threshold = simulate_percolation(G, threshold=0.5, fast=fast)
    print(f"Percolation threshold = {threshold:.3f}")
    
    # Compute fractal betweenness
    print("Computing fractal betweenness...")
    FB = compute_fractal_betweenness(G, repeating_subgraphs, centrality)
    print(f"Fractal betweenness computed for {len(FB)} nodes.")
    
    # Build decision table
    print("Building decision table...")
    decision_table = build_decision_table(G, D, alpha, FB, threshold, chokepoints_df, cc)
    print(f"Decision table built with {len(decision_table)} rows.")
    
    # Export results
    print("Exporting results...")
    
    # Make sure output directory exists for this dataset
    os.makedirs(output_dir, exist_ok=True)
    
    # Export chokepoints
    chokepoints_df.to_csv(os.path.join(output_dir, "event_chokepoints.csv"), index=False)
    
    # Export fragile flows
    if not fragile_flows_df.empty:
        fragile_flows_df.to_csv(os.path.join(output_dir, "high_friction_flows.csv"), index=False)
    else:
        # Create an empty file to indicate analysis was run
        with open(os.path.join(output_dir, "high_friction_flows.csv"), 'w') as f:
            f.write("session_id,step_index,page,event,WSJF_Friction_Score,is_chokepoint,user_id\n")
    
    # Export node map
    with open(os.path.join(output_dir, "friction_node_map.json"), 'w') as f:
        json.dump(node_map, f, indent=2)
    
    # Export decision table
    decision_table.to_csv(os.path.join(output_dir, "decision_table.csv"), index=False)
    
    # Create final report with key metrics
    final_report = {
        "fractal_dimension": D,
        "power_law_alpha": alpha,
        "clustering_coefficient": cc,
        "percolation_threshold": threshold,
        "top_fb_nodes": [(node, score) for node, score in sorted(FB.items(), key=lambda x: x[1], reverse=True)[:5]],
        "top_chokepoints": chokepoints_df.iloc[:5][['page', 'event', 'WSJF_Friction_Score']].values.tolist()
    }
    
    # Export final report as JSON
    with open(os.path.join(output_dir, "final_report.json"), 'w') as f:
        json.dump(final_report, f, indent=2)
    
    # Export final report as CSV for easier reading
    report_df = pd.DataFrame([
        {"metric": "Fractal Dimension", "value": D},
        {"metric": "Power Law Alpha", "value": alpha},
        {"metric": "Clustering Coefficient", "value": cc},
        {"metric": "Percolation Threshold", "value": threshold},
        {"metric": "Top FB Node", "value": final_report["top_fb_nodes"][0][0] if final_report["top_fb_nodes"] else "N/A"},
        {"metric": "Top Chokepoint", "value": f"{final_report['top_chokepoints'][0][0]}:{final_report['top_chokepoints'][0][1]}" if final_report["top_chokepoints"] else "N/A"}
    ])
    report_df.to_csv(os.path.join(output_dir, "final_report.csv"), index=False)
    
    print(f"Exported {len(chokepoints_df)} chokepoints to {output_dir}\\event_chokepoints.csv")
    print(f"Exported {'no' if fragile_flows_df.empty else fragile_flows_df['session_id'].nunique()} fragile flows to {output_dir}\\high_friction_flows.csv")
    print(f"Exported node map with {len(node_map)} pages to {output_dir}\\friction_node_map.json")
    print(f"Exported decision table to {output_dir}\\decision_table.csv")
    print(f"Exported final report to {output_dir}\\final_report.json and {output_dir}\\final_report.csv")
    print("\n")
    
    return chokepoints_df, fragile_flows_df, node_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TeloMesh Event Chokepoints Analysis")
    parser.add_argument("--n_users", type=int, default=1000, help="Number of unique users")
    parser.add_argument("--events_per_user", type=int, default=6, help="Average events per user")
    parser.add_argument("--input_flows", type=str, default="outputs/session_flows.csv", help="Path to session flows CSV")
    parser.add_argument("--input_graph", type=str, default="outputs/user_graph.gpickle", help="Path to graph pickle")
    parser.add_argument("--input_graph_multi", type=str, default="outputs/user_graph_multi.gpickle", help="Path to multi-graph pickle")
    parser.add_argument("--output_path", type=str, default="outputs", help="Output directory")
    parser.add_argument("--fast", action="store_true", help="Skip detailed statistics output")
    
    args = parser.parse_args()
    
    # Run the main function
    main(args.input_flows, args.input_graph, args.input_graph_multi, args.output_path, args.fast)