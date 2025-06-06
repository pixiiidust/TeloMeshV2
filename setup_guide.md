# TeloMesh Setup Guide

## Introduction

TeloMesh is a user journey analysis tool designed to help product managers identify friction points in user flows. This guide will walk you through the setup process to get TeloMesh running on your system.

## Prerequisites

- Python 3.9+ installed
- Git (for cloning the repository)
- pip (Python package manager)

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/TeloMesh.git
cd TeloMesh
```

2. **Create and activate a virtual environment (recommended)**

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Features

- **Synthetic Event Generation**: Generate realistic synthetic user event data
- **Session Parsing**: Extract structured session paths from event data
- **Graph Building**: Create a directed, event-typed graph of user flows
- **Flow Metrics**: Validate session and graph quality metrics
- **Friction Analysis**: Identify chokepoints and fragile flows in user journeys
- **Dashboard**: Interactive visualization of friction points and user flows
- **Analytics Import**: Convert data from Mixpanel, Amplitude, and Google Analytics 4

## Repository Structure

The repository is organized into the following components:

### Core Directories

#### `/data`
Contains input data and data generation tools:
- `events.csv` - Raw event log containing user interactions (synthetic or imported)
- `synthetic_event_generator.py` - Python script that generates realistic test data with configurable parameters

#### `/ingest`
Pipeline components for transforming raw events into structured session data:
- `parse_sessions.py` - Converts raw CSV events into structured user sessions
- `build_graph.py` - Creates a directed graph representation of user journeys
- `flow_metrics.py` - Validates sessions and graph quality metrics

#### `/analysis`
Advanced analysis components for discovering insights:
- `event_chokepoints.py` - Identifies friction points by computing exit rates and betweenness centrality

#### `/ui`
User interface components:
- `dashboard.py` - Streamlit dashboard with interactive visualizations, filtering, and data export capabilities

#### `/utils`
Utility scripts and tools:
- `analytics_converter.py` - Tool for converting data from Mixpanel, Amplitude, and GA4 to TeloMesh format

#### `/outputs`
Generated data files:
- `session_flows.csv` - Processed session data in tabular format
- `user_graph.gpickle` - Serialized NetworkX graph with user journey information
- `event_chokepoints.csv` - Identified friction points with WSJF scores
- `high_friction_flows.csv` - User journeys that contain multiple high-friction points
- `friction_node_map.json` - Mapping of pages to WSJF scores for visualization

#### `/logs`
Logging and monitoring data:
- `session_stats.log` - Log file with session validation details
- `metrics.json` - Summary metrics for quality assessment

#### `/tests`
Test suites for each component:
- `test_synthetic_events.py` - Tests for data generation
- `test_parse_sessions.py` - Tests for session parsing
- `test_build_graph.py` - Tests for graph construction
- `test_flow_metrics.py` - Tests for metrics calculation
- `test_event_chokepoints.py` - Tests for friction point identification
- `test_dashboard_ui.py` - Tests for UI components

#### `/logos`
Brand assets and visual elements:
- `Telomesh logo.png` - Primary logo for application header
- `telomesh logo white.png` - White version of logo for dark backgrounds

### Key Files

- `main.py` - Entry point for the pipeline with command-line arguments
- `requirements.txt` - Package dependencies for easy installation
- `directory_structure.lua` - Canonical definition of project structure
- `README.md` - Project documentation
- `.streamlit/config.toml` - Streamlit configuration for dark theme and UI settings
- `.gitignore` - Configuration for Git to exclude temporary files

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

This will launch the TeloMesh User Flow Intelligence Dashboard with the following features:
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

### Using the Analytics Converter

TeloMesh includes a powerful utility to convert data from popular analytics platforms:

```bash
# Convert data and make it ready for TeloMesh in one step
python utils/analytics_converter.py --input your_analytics_export.csv --output data/events.csv --format [platform] --telomesh-format

# Where [platform] is one of: mixpanel, amplitude, ga4
```

#### Supported Platforms:

- **Mixpanel**: Convert event exports with user, timestamp, and page information
- **Amplitude**: Convert Amplitude CSV exports with event data
- **Google Analytics 4**: Convert GA4 event exports with page and user data

#### Generate Sample Data:

If you don't have real analytics data, you can generate realistic sample data:

```bash
# Generate sample data ready for TeloMesh
python utils/analytics_converter.py --generate-sample --format amplitude --output data/events.csv --telomesh-format
```

For detailed documentation on analytics conversion, see the [Analytics Converter Guide](utils/GUIDE.md).

## WSJF Friction Scoring

TeloMesh uses a Weighted Shortest Job First (WSJF) approach to prioritize friction points:

`WSJF_Friction_Score = exit_rate × betweenness`

Where:
- `exit_rate`: Probability of session ending after the event
- `betweenness`: Page's structural importance in user flows

## Troubleshooting

- **Missing dependencies**: Ensure all packages in `requirements.txt` are installed
- **File not found errors**: Check that all input files exist in the specified paths
- **Graph visualization issues**: Try adjusting node count filters in the dashboard
- **Streamlit theme errors**: If you encounter theme-related errors, make sure your Streamlit version is compatible with the theme options used or update `.streamlit/config.toml`

## Next Steps

After identifying friction points, consider:
1. Prioritizing improvements based on WSJF scores
2. Running A/B tests on identified problematic flows
3. Monitoring metrics over time to track progress

## Getting Help

For more information, check the documentation in `/docs` or open an issue on the GitHub repository. 