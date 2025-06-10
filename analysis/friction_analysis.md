# Analysis Components

This folder contains the core analytical components of TeloMesh that identify friction points in user journeys and provide advanced network analytics.

## Key Components
- Event chokepoint detection algorithms
- WSJF (Weighted Shortest Job First) friction scoring system
- Betweenness centrality calculations for page importance
- Exit rate analysis for identifying abandonment points
- User flow analysis for multi-point journey failures
- Dataset-specific analysis outputs

## Advanced Network Analysis
The system includes sophisticated network analysis metrics:

- **Fractal Dimension (D)**: Measures the complexity of user navigation patterns (typically 2.0-3.0)
- **Power-law Alpha (α)**: Quantifies the degree distribution characteristics of the network (typically 1.8-2.6)
- **Percolation Threshold**: Identifies the critical point at which the network collapses (typically 0.2-0.5)
- **Fractal Betweenness (FB)**: Enhanced centrality measure that considers network structure (typically 0.6-1.0)
- **Recurring Patterns Detection**: Identifies common navigation loops and repetitive behaviors in user journeys
  - **Node Participation Analysis**: Identifies and ranks which nodes (pages) appear most frequently in recurring patterns
  - **Loop Count Metrics**: Tracks how often each page appears in detected loops and patterns
  - **Participation Percentage**: Shows what percentage of all recurring patterns include each node
- **Exit Path Analysis**: Detects sequences that frequently lead to users leaving the site with associated exit rates
  - **Exit Path Detection**: Identifies common sequences that lead to site abandonment
  - **Exit Rate Tracking**: Calculates the percentage of users who exit after following specific paths

## Priority Matrix
The system now employs a quadrant-based prioritization system using FB vs WSJF visualization:

- **High Priority (Top Right)**: Nodes with high structural importance AND high user friction - require immediate attention
- **User Friction Only (Top Left)**: Pages causing friction but not critical to site structure - focus on user experience improvements
- **Structural Only (Bottom Right)**: Important navigation nodes with low friction - maintain and optimize these core pathways
- **Low Priority (Bottom Left)**: Less important pages with minimal friction - monitor but lower priority for changes

## Decision Table & Recommendations
The system generates actionable UX recommendations based on network analytics:

- Identifies UX patterns (linear paths, tree structures, complex hubs)
- Provides specific improvement suggestions based on network characteristics
- Classifies nodes as standard or critical based on percolation role
- Detects redundant pathways and suggests consolidation when appropriate
- Analyzes network stability and identifies critical structural dependencies

## Performance Optimization
The analysis now supports a `--fast` mode for processing large datasets efficiently:
- Optimized algorithms for subgraph detection and percolation simulation
- Sampling techniques for large networks
- Parallel processing options for computationally intensive tasks

## Graph Statistics
The analysis provides key graph statistics for better network understanding:
- **Node Count**: Total number of unique pages or states in the user journey
- **Edge Count**: Total number of transitions between pages
- **Edge/Node Ratio**: Measure of graph density - higher values indicate more complex navigation options
- **Connected Components**: Number of isolated subgraphs - ideally should be 1 for a fully connected experience

## Dataset Integration
The analysis components support dataset organization, with all outputs stored in dataset-specific directories:
```
outputs/
└── [dataset_name]/
    ├── event_chokepoints.csv
    ├── high_friction_flows.csv
    ├── friction_node_map.json
    ├── user_graph.gpickle
    ├── user_graph_multi.gpickle
    ├── decision_table.csv
    ├── metrics.json
    ├── recurring_patterns.json
    ├── recurring_exit_paths.json
    ├── session_flows.csv
    └── dataset_info.json
```

Each dataset maintains its own analysis results, allowing for comparison between different user groups or before/after UX changes.

## Usage
Run the analysis with advanced network metrics:
```bash
python main.py --dataset myproject --fast
```

The `--fast` flag enables optimized processing for large datasets. 

To view the analysis results with the interactive dashboard:
```bash
streamlit run ui/dashboard.py
``` 