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
The system now includes sophisticated network analysis metrics:

- **Fractal Dimension (D)**: Measures the complexity of user navigation patterns (typically 1.0-3.0)
- **Power-law Alpha (α)**: Quantifies the degree distribution characteristics of the network
- **Clustering Coefficient**: Measures how interconnected pages are in the user journey
- **Percolation Threshold**: Identifies the critical point at which the network collapses
- **Fractal Betweenness**: Enhanced centrality measure that considers repeating subgraph patterns
- **Repeating Subgraph Detection**: Identifies common navigation patterns in user journeys

## Decision Table & Recommendations
The system now generates actionable UX recommendations based on network analytics:

- Identifies UX patterns (linear paths, tree structures, complex hubs)
- Provides specific improvement suggestions based on network characteristics
- Classifies nodes as standard, bottlenecks, or critical junctures
- Detects redundant pathways and suggests consolidation when appropriate

## Performance Optimization
The analysis now supports a `--fast` mode for processing large datasets efficiently:
- Optimized algorithms for subgraph detection and percolation simulation
- Sampling techniques for large networks
- Parallel processing options for computationally intensive tasks

## Dataset Integration
The analysis components support dataset organization, with all outputs stored in dataset-specific directories:
```
outputs/
└── [dataset_name]/
    ├── event_chokepoints.csv
    ├── high_friction_flows.csv
    ├── friction_node_map.json
    ├── decision_table.csv
    ├── final_report.json
    ├── final_report.csv
    └── dataset_info.json
```

Each dataset maintains its own analysis results, allowing for comparison between different user groups or before/after UX changes.

## Usage
Run the analysis with advanced network metrics:
```bash
python main.py --dataset myproject --fast
```

The `--fast` flag enables optimized processing for large datasets. 