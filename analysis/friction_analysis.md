# Analysis Components

This folder contains the core analytical components of TeloMesh that identify friction points in user journeys.

## Key Components
- Event chokepoint detection algorithms
- WSJF (Weighted Shortest Job First) friction scoring system
- Betweenness centrality calculations for page importance
- Exit rate analysis for identifying abandonment points
- User flow analysis for multi-point journey failures
- Dataset-specific analysis outputs

The main file `event_chokepoints.py` processes user session data to generate ranked lists of friction points, high-friction flows, and graph visualization data.

## Dataset Integration
The analysis components now support dataset organization, with all outputs stored in dataset-specific directories:
```
outputs/
└── [dataset_name]/
    ├── event_chokepoints.csv
    ├── high_friction_flows.csv
    ├── friction_node_map.json
    └── dataset_info.json
```

Each dataset maintains its own analysis results, allowing for comparison between different user groups or before/after UX changes. 