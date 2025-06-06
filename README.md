# TeloMesh

TeloMesh is a user journey analysis pipeline that processes event data to build a directed graph of user flows. This tool helps product managers and UX designers identify patterns, chokepoints, and optimization opportunities in user journeys.

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
├── outputs/
│   ├── session_flows.csv            # Flat session log
│   └── user_graph.gpickle           # Directed, typed, weighted graph
├── logs/
│   ├── session_stats.log            # Session validation logs
│   └── metrics.json                 # Metrics summary
├── tests/
│   ├── test_synthetic_events.py     # Tests for event generation
│   ├── test_parse_sessions.py       # Tests for session parsing
│   ├── test_build_graph.py          # Tests for graph building
│   └── test_flow_metrics.py         # Tests for metrics validation
└── main.py                          # Pipeline entry point
```

## Features

- **Synthetic Event Generation**: Generate realistic synthetic user event data
- **Session Parsing**: Extract structured session paths from event data
- **Graph Building**: Create a directed, event-typed graph of user flows
- **Flow Metrics**: Validate session and graph quality metrics

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
```

## Development

### Prerequisites

- Python 3.6+
- Required packages: pandas, networkx, pytest

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/telomesh.git
cd telomesh

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