# Data Ingestion Pipeline

This folder contains the components responsible for transforming raw event data into structured user flows and network graphs.

## Key Components
- Session parsing and event sequence extraction
- User flow graph construction with NetworkX
- Flow metrics calculation and validation
- Data transformation for analytical processing
- Dataset-specific processing and outputs

## Pipeline Steps
The pipeline consists of four main steps:
1. `parse_sessions.py`: Converts raw CSV data into structured user flows
2. `build_graph.py`: Transforms user flows into directed graph representations:
   - Standard DiGraph for basic analysis
   - MultiDiGraph for advanced network analysis
3. `flow_metrics.py`: Validates and generates metrics about the session data quality
4. Advanced network analysis (in the analysis module)

## Multi-Graph Support
The pipeline now generates two graph representations:
- **DiGraph**: Simplified graph where edges represent aggregated transitions between pages
- **MultiDiGraph**: Detailed graph preserving all individual transitions for enhanced analysis
  - Enables more accurate fractal dimension calculation
  - Provides better subgraph detection
  - Preserves exact user journey paths

## Dataset Organization
The ingestion pipeline supports dataset-specific processing:

```
data/
└── [dataset_name]/
    └── events.csv

outputs/
└── [dataset_name]/
    ├── session_flows.csv
    ├── user_graph.gpickle
    ├── user_graph_multi.gpickle
    ├── metrics.json
    └── session_stats.log
```

## Performance Considerations
- For large datasets (>1000 users), processing time scales linearly
- Memory usage increases with the number of unique transitions
- For very large datasets, use the `--fast` flag to optimize processing

## Usage
```bash
# Process a specific dataset
python main.py --dataset myproject

# Use fast mode for large datasets
python main.py --dataset large_project --fast

# Process with custom parameters
python main.py --dataset myproject --users 500 --events 20
```

Use the `--dataset` parameter with main.py to process specific datasets:
```bash
python main.py --stage parse --dataset myproject
python main.py --stage graph --dataset myproject
python main.py --stage metrics --dataset myproject 