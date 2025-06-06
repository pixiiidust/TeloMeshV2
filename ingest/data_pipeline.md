# Data Ingestion Pipeline

This folder contains the components responsible for transforming raw event data into structured user flows and network graphs.

## Key Components
- Session parsing and event sequence extraction
- User flow graph construction with NetworkX
- Flow metrics calculation and validation
- Data transformation for analytical processing
- Dataset-specific processing and outputs

The pipeline consists of three main steps:
1. `parse_sessions.py`: Converts raw CSV data into structured user flows
2. `build_graph.py`: Transforms user flows into a directed graph representation
3. `flow_metrics.py`: Validates and generates metrics about the session data quality

## Dataset Organization
The ingestion pipeline now supports dataset-specific processing:

```
data/
└── [dataset_name]/
    └── events.csv

outputs/
└── [dataset_name]/
    ├── session_flows.csv
    ├── user_graph.gpickle
    ├── metrics.json
    └── session_stats.log
```

Use the `--dataset` parameter with main.py to process specific datasets:
```bash
python main.py --stage parse --dataset myproject
python main.py --stage graph --dataset myproject
python main.py --stage metrics --dataset myproject 