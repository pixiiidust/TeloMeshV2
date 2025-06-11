# TeloMesh Data Pipeline

This folder contains modules for data ingestion, session parsing, and graph construction.

## Pipeline Components

### 1. parse_sessions.py
- Processes raw event data into structured session flows
- Automatically divides user activities into sessions based on time gaps
- Handles timestamp parsing for both formats (with and without milliseconds)
- Chronologically organizes user journeys with sequential step indices
- Input: Raw event data CSV
- Output: Structured session flows CSV

### 2. build_graph.py
- Constructs a user interaction graph from session flow data
- Creates node and edge relationships based on user navigation patterns
- Supports both directed and multi-edge graph construction
- Calculates basic graph metrics and connectivity information
- Input: Session flows CSV
- Output: NetworkX graph in .gpickle format (both standard and multigraph)

### 3. flow_metrics.py
- Computes advanced flow-based metrics for session data
- Identifies session transition patterns and common pathways
- Calculates session diversity and engagement metrics
- Provides journey complexity measures across sessions
- Input: Session flows CSV
- Output: Flow metrics JSON and CSV

## Data Format Requirements

Input data should be a CSV file with the following columns:
- `user_id`: Unique identifier for the user
- `page_url`: URL or identifier of the page visited
- `event_name`: Type of event (click, view, submit, etc.)
- `timestamp`: ISO format timestamp (now supports both formats with and without milliseconds)

Example:
```
user_id,page_url,event_name,timestamp
user123,/home,page_view,2023-05-01T09:15:32.123Z
user123,/products,page_view,2023-05-01T09:16:45.678Z
user123,/products/item,click,2023-05-01T09:17:20.456Z
```

The pipeline also accepts timestamp formats without milliseconds:
```
user_id,page_url,event_name,timestamp
user123,/home,page_view,2023-05-01T09:15:32Z
user123,/products,page_view,2023-05-01T09:16:45Z
user123,/products/item,click,2023-05-01T09:17:20Z
```

## Output Formats

### Session Flows (outputs/session_flows.csv)
Standardized format for user journeys with chronological steps:
```
user_id,session_id,step_index,page,event,timestamp
user123,session_user123_00,0,/home,page_view,2023-05-01 09:15:32.123
user123,session_user123_00,1,/products,page_view,2023-05-01 09:16:45.678
user123,session_user123_00,2,/products/item,click,2023-05-01 09:17:20.456
```

### User Graph (outputs/user_graph.gpickle)
NetworkX graph structure with:
- Nodes: Individual pages/states
- Edges: Transitions between pages
- Attributes: Transition counts, timing data, etc.

### Flow Metrics (outputs/flow_metrics.json)
```json
{
  "transition_patterns": {
    "/home→/products": 156,
    "/products→/cart": 89,
    ...
  },
  "common_paths": [
    {
      "path": ["/home", "/products", "/cart", "/checkout"],
      "count": 42
    },
    ...
  ],
  "page_metrics": {
    "/home": {
      "entry_count": 203,
      "exit_count": 18,
      ...
    },
    ...
  }
}
```

## Usage

### Basic Pipeline Execution
```bash
# Step 1: Parse sessions from raw events
python -m ingest.parse_sessions --input data/events.csv --output outputs/session_flows.csv --session-gap 30

# Step 2: Build user interaction graph
python -m ingest.build_graph --input outputs/session_flows.csv --output outputs/user_graph.gpickle

# Step 3: Compute flow metrics
python -m ingest.flow_metrics --input outputs/session_flows.csv --output outputs/flow_metrics.json
```

### Integrated Pipeline
For end-to-end processing:
```bash
python main.py --dataset my_project
```

This will run the entire pipeline with default parameters and store outputs in `outputs/my_project/`.

## Error Handling and Validation

The pipeline includes:
- Session quality checks to ensure reasonable session counts and steps per session
- Timestamp parsing validation with error reporting for invalid formats
- Fast mode option for processing large datasets with reduced validation
