# Analysis Components

This folder contains the core analytical components of TeloMesh that identify friction points in user journeys and provide advanced network analytics.

## Key Components
- Event chokepoint detection algorithms
- WSJF (Weighted Shortest Job First) friction scoring system
- Betweenness centrality calculations for page importance
- Exit rate analysis for identifying abandonment points
- User flow analysis for multi-point journey failures
- Transition pairs analysis for page-to-page navigation patterns
- Dataset-specific analysis outputs
- Robust WSJF threshold calculation for handling zero-inflated distributions
- Performance benchmarks for different dataset sizes

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
- Robust threshold calculation for handling datasets of all sizes (10K to 100K+ users)

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

## Advanced Flow Analysis
The system now provides two complementary approaches to analyzing user flows:

1. **Flow Sequences**: 
   - Identifies multi-step user journey sessions containing 2+ chokepoints
   - Shows complete paths where users encounter multiple friction points
   - Filterable by path length steps and total WSJF score ranking

2. **Transition Pairs**:
   - Extracts common page-to-page transitions (A→B) across sessions
   - Identifies specific navigation patterns with chokepoints
   - Calculates transition frequency, session count, and total WSJF score
   - Helps pinpoint problematic navigation patterns between pages
   - Distinguishes between "from" chokepoints and "to" chokepoints in transitions

These complementary views help product managers understand both the complete user journeys (Flow Sequences) and the specific problematic transitions between pages (Transition Pairs).

## Event Flow Analysis

### Weighted Shortest Job First (WSJF) Friction Score

The WSJF Friction Score is calculated to identify high-friction (page, event) pairs in the user journey:

```
WSJF_Friction_Score = exit_rate × betweenness
```

Where:
- **exit_rate**: The percentage of sessions that end after this (page, event)
- **betweenness**: A graph-based centrality measure indicating how crucial this node is for connecting different parts of the user journey

### Robust WSJF Threshold Calculation

The system identifies chokepoints by comparing each (page, event) pair's WSJF_Friction_Score against a threshold. Originally, this used a simple 90th percentile approach:

```python
# Original approach (has issues with zero-inflated data)
chokepoint_threshold = chokepoints_df['WSJF_Friction_Score'].quantile(0.9)
```

However, with large datasets (100K+ users), the distribution of WSJF scores becomes heavily zero-inflated (90%+ zeros), causing the 90th percentile to be 0.0. This resulted in all non-zero scores being classified as chokepoints.

The new approach uses robust statistics and adaptive thresholding:

1. **Filter to non-zero scores**: Exclude zeros before calculating statistics
2. **Use Median + MAD**: Calculate threshold as `median + (mad_multiplier * MAD)`
   - Median: More robust to outliers than mean
   - MAD (Median Absolute Deviation): Better scale measure for skewed distributions
3. **Handle edge cases**: Special handling for all-zero, few non-zero, or very low MAD cases
4. **Adaptive adjustment**: Automatically adjust threshold to ensure a reasonable number of chokepoints (5-15%)

This approach works reliably across all dataset sizes, from small test sets to 100K+ user production data.

### Fragile Flows

Fragile flows are user journeys that contain multiple high-friction points (chokepoints). These represent particularly problematic user experiences where users encounter several points of difficulty in a single session.

### Large-Scale Performance

The enhanced WSJF threshold calculation has been extensively tested with datasets ranging from 10K to 100K users. For detailed performance benchmarks, see [performance_benchmarks.md](performance_benchmarks.md).

Key findings:
- Processing time scales linearly with dataset size
- The threshold calculation adapts to zero-inflation in larger datasets
- For 100K user datasets, chokepoint detection completes in ~4 minutes

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