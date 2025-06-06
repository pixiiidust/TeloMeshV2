# TeloMesh

TeloMesh is a user journey analysis pipeline that processes event data to build a directed graph of user flows. This tool helps product managers and UX designers identify patterns, chokepoints, and optimization opportunities in user journeys.

## New in Version 2
- Enhanced dashboard with export features
- Friction Intelligence visualization
- Dark theme with improved UX

## Project Structure

The project is organized into the following components:

```
TeloMesh/
├── data/
│   ├── events.csv                   # Synthetic event log
│   └── synthetic_event_generator.py # Generates test data
├── ingest/
│   ├── parse_sessions.py            # CSV → sessions
│   ├── build_graph.py               # sessions → graph
│   └── flow_metrics.py              # Validates sessions and graph
├── analysis/
│   └── event_chokepoints.py         # Compute friction metrics
├── ui/
│   └── dashboard.py                 # Streamlit dashboard
├── outputs/
│   ├── session_flows.csv            # Flat session log
│   ├── user_graph.gpickle           # Directed, typed, weighted graph
│   ├── event_chokepoints.csv        # Friction points analysis
│   ├── high_friction_flows.csv      # Flows with multiple chokepoints
│   └── friction_node_map.json       # WSJF scores for heatmap
├── logs/
│   ├── session_stats.log            # Session validation logs
│   └── metrics.json                 # Metrics summary
├── tests/
│   ├── test_synthetic_events.py     # Tests for event generation
│   ├── test_parse_sessions.py       # Tests for session parsing
│   ├── test_build_graph.py          # Tests for graph building
│   ├── test_flow_metrics.py         # Tests for metrics validation
│   ├── test_event_chokepoints.py    # Tests for friction analysis
│   └── test_dashboard_ui.py         # Tests for dashboard
└── main.py                          # Pipeline entry point
```

## Features

- **Synthetic Event Generation**: Generate realistic synthetic user event data
- **Session Parsing**: Extract structured session paths from event data
- **Graph Building**: Create a directed, event-typed graph of user flows
- **Flow Metrics**: Validate session and graph quality metrics
- **Friction Analysis**: Identify chokepoints and fragile flows in user journeys
- **Dashboard**: Interactive visualization of friction points and user flows

## Usage

### Running the Full Pipeline

```bash
python main.py --users 100 --events 50
```

This will:
1. Generate synthetic events
2. Parse events into sessions
3. Build a user journey graph
4. Validate flow metrics
5. Analyze friction points
6. Prepare data for dashboard

### Running the Dashboard

```bash
streamlit run ui/dashboard.py
```

This will launch the TeloMesh Friction Intelligence Dashboard with the following features:
- Friction Points table with export to CSV
- User Flow Heatmap with export to HTML
- Fragile Flows visualization with export to CSV

### Running Individual Stages

You can run individual stages of the pipeline:

```bash
# Generate synthetic events
python main.py --stage synthetic --users 100 --events 50

# Parse events into sessions
python main.py --stage parse

# Build user journey graph
python main.py --stage graph

# Validate flow metrics
python main.py --stage metrics

# Analyze friction points
python main.py --stage analysis
```

## Using Your Own Data

TeloMesh can analyze user journey data from analytics platforms like Mixpanel, Amplitude, or Google Analytics. Follow these steps to use your own data:

### Data Format Requirements

Place your exported data in `data/events.csv` with the following required columns:

| Column          | Type     | Description                              | Example                     |
|-----------------|----------|------------------------------------------|----------------------------|
| `user_id`       | string   | Unique identifier for each user          | "user_123", "5678abcd"     |
| `timestamp`     | datetime | When the event occurred                  | "2023-04-15T14:32:01.000Z" |
| `page`          | string   | URL path or screen name                  | "/checkout", "ProductPage" |
| `event`         | string   | User action that occurred                | "click_button", "scroll"   |
| `session_id`    | string   | Unique session identifier                | "session_456", "abc123"    |

#### Optional columns:
| Column          | Type     | Description                              | Example                     |
|-----------------|----------|------------------------------------------|----------------------------|
| `event_properties` | JSON/string | Additional event details           | `{"button_id": "submit"}` |
| `user_properties`  | JSON/string | User profile information           | `{"plan": "premium"}`     |

### Exporting from Analytics Platforms

#### Mixpanel
1. Go to **Insights** or **Events** view in Mixpanel
2. Set your desired time range and filters
3. Click **Export** → **CSV**
4. Use the following column mapping:
   - `distinct_id` → `user_id`
   - `time` → `timestamp`
   - `$current_url` or `screen` → `page`
   - `event` remains as `event`
   - `$insert_id` → `session_id` (or generate one with your session definition)
   - `properties` → `event_properties`

#### Amplitude
1. Go to **Events** or **User Streams** in Amplitude
2. Set your desired date range and filters
3. Click **Export** → **CSV/JSON**
4. Use the following column mapping:
   - `user_id` remains as `user_id`
   - `event_time` → `timestamp`
   - `page_url` or `page_title` → `page`
   - `event_type` → `event`
   - `session_id` remains as `session_id`
   - `event_properties` remains as `event_properties`

#### Google Analytics 4
1. Export your events using the GA4 API or BigQuery
2. Map the following columns:
   - `user_pseudo_id` → `user_id`
   - `event_timestamp` → `timestamp`
   - `page_location` or `page_title` → `page`
   - `event_name` → `event`
   - A combination of `user_pseudo_id` and `session_id` → `session_id`

### Session Definition

If your data doesn't include proper session IDs, you can generate them during import by:
1. Sorting events by `user_id` and `timestamp`
2. Creating a new session whenever there's a gap of 30+ minutes between events for the same user

### Processing Your Custom Data

After placing your properly formatted data in `data/events.csv`, run:

```bash
# Skip synthetic data generation and start from parsing
python main.py --stage parse

# Or if you need to transform your data first
python main.py --input-file path/to/your/exported/data.csv --format mixpanel
```

The pipeline will:
1. Parse your events into sessions
2. Build the user journey graph
3. Calculate metrics and friction points
4. Prepare data for the dashboard

### Custom Data Processing Script

If your data requires significant preprocessing, you can use this template:

```python
import pandas as pd

# Load your raw data
raw_data = pd.read_csv('path/to/your/data.csv')

# Transform to required format
transformed_data = pd.DataFrame({
    'user_id': raw_data['your_user_id_column'],
    'timestamp': pd.to_datetime(raw_data['your_timestamp_column']),
    'page': raw_data['your_page_column'],
    'event': raw_data['your_event_column'],
    'session_id': raw_data['your_session_id_column']
})

# Sort by user and timestamp
transformed_data = transformed_data.sort_values(['user_id', 'timestamp'])

# Save in the required location
transformed_data.to_csv('data/events.csv', index=False)
```

### Example Custom Data

Here's a sample of correctly formatted data:

```csv
user_id,timestamp,page,event,session_id
user_123,2023-04-15T14:30:00.000Z,/home,page_view,session_abc
user_123,2023-04-15T14:31:25.000Z,/products,page_view,session_abc
user_123,2023-04-15T14:32:45.000Z,/product/123,click_button,session_abc
user_123,2023-04-15T14:33:10.000Z,/cart,page_view,session_abc
user_456,2023-04-15T15:10:00.000Z,/home,page_view,session_def
user_456,2023-04-15T15:12:30.000Z,/search,search_query,session_def
```

## Development

### Prerequisites

- Python 3.6+
- Required packages: pandas, networkx, pytest, streamlit, pyvis

### Installation

```bash
# Clone the repository
git clone https://github.com/pixiiidust/TeloMeshV2.git
cd TeloMeshV2

# Install dependencies
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_synthetic_events.py
```

## Graph Structure

The user journey graph is structured as follows:

- **Nodes**: Represent pages in the user journey
- **Edges**: Represent transitions between pages
- **Edge Attributes**:
  - `event`: The event that triggered the transition
  - `weight`: The number of times this transition occurred

## Metrics

The flow metrics validation checks:

- **Sessions per user**: Should be ≥ 1
- **Average flow length**: Should be ≥ 3 steps
- **Node count**: Should be ≥ 10
- **Total sessions**: Should be ≥ 50
- **Unique edge events**: Should be ≥ 2

## License

MIT
