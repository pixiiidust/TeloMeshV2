# TeloMesh

TeloMesh is a user journey analysis pipeline that processes event data to build a directed graph of user flows. This tool helps product managers and UX designers identify patterns, chokepoints, and optimization opportunities in user journeys.

## Theory & Methodology

TeloMesh uses graph theory and network analysis to identify friction points in user journeys:

1. **Graph Construction**: User sessions are converted into directed graphs where:
   - Nodes represent pages/screens
   - Edges represent user actions (events)
   - Edge weights correspond to transition frequency

2. **Friction Detection Algorithm**: TeloMesh identifies problem areas using a composite scoring system:
   - **Exit Rate**: Percentage of users who abandon their journey at a specific (page, event) pair
   - **Betweenness Centrality**: Measures how critical a node is to the overall flow structure
   - **WSJF Friction Score**: Calculated as `exit_rate × betweenness`, prioritizing high-impact friction points

3. **Percolation Analysis**: Inspired by network percolation theory, TeloMesh identifies:
   - Critical junctions where small improvements yield large UX gains
   - Cascading failure patterns where multiple friction points compound
   - Fragile flows where users encounter multiple high-WSJF obstacles

## Key Features

### 1. Friction Points Analysis
- Identifies and ranks individual (page, event) pairs by WSJF Friction Score
- Calculates exit probability and lost user volume
- Highlights structural weaknesses in the user journey

### 2. User Flow Heatmap
- Visual representation of user journeys with friction highlighted
- Color-coded nodes based on friction percentile (top 10%, top 20%)
- Interactive graph visualization with tooltips and event details

### 3. Fragile Flows Detection
- Identifies user paths containing multiple high-friction points
- Highlights sequences where users encounter cascading obstacles
- Prioritizes flow improvements based on cumulative friction

## For Product Managers

TeloMesh helps product managers by:
- Quantifying UX friction points for prioritization
- Visualizing critical user journey bottlenecks
- Identifying high-impact improvement opportunities
- Providing data-driven insights for roadmap planning
- Enabling before/after comparisons of UX changes

By combining exit rates with structural importance, the WSJF scoring system ensures that improvements focus on areas with the highest impact on overall user experience.

## New in Version 2
- Enhanced dashboard with export features
- Friction Intelligence visualization
- Dark theme with improved UX

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
- `README.md` - Project documentation (this file)
- `.streamlit/config.toml` - Streamlit configuration for dark theme and UI settings
- `.gitignore` - Configuration for Git to exclude temporary files

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