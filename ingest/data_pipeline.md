# Data Ingestion Pipeline

This folder contains the components responsible for transforming raw event data into structured user flows and network graphs.

## Key Components
- Session parsing and event sequence extraction
- User flow graph construction with NetworkX
- Flow metrics calculation and validation
- Data transformation for analytical processing

The pipeline consists of three main steps:
1. `parse_sessions.py`: Converts raw CSV data into structured user flows
2. `build_graph.py`: Transforms user flows into a directed graph representation
3. `flow_metrics.py`: Validates and generates metrics about the session data quality 