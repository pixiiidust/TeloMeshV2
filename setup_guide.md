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
- **Dataset Management**: Create, organize, and switch between multiple datasets

## Repository Structure

The repository is organized into the following components:

### Core Directories

#### `/data`
Contains input data and data generation tools:
- `synthetic_event_generator.py` - Python script that generates realistic test data with configurable parameters
- Dataset-specific subdirectories (e.g., `data/myproject/`) - Isolated input data for each dataset
- `synthetic_data.md` - Documentation about the data generation process

#### `/ingest`
Pipeline components for transforming raw events into structured session data:
- `parse_sessions.py` - Converts raw CSV events into structured user sessions
- `build_graph.py` - Creates a directed graph representation of user journeys
- `flow_metrics.py` - Validates sessions and graph quality metrics
- `data_pipeline.md` - Documentation about the data ingestion process

#### `/analysis`
Advanced analysis components for discovering insights:
- `event_chokepoints.py` - Identifies friction points by computing exit rates and betweenness centrality
- `friction_analysis.md` - Documentation about the friction analysis methodology

#### `/ui`
User interface components:
- `dashboard.py` - Streamlit dashboard with interactive visualizations, filtering, and data export capabilities
- `dashboard_components.md` - Documentation about the dashboard components

#### `/utils`
Utility scripts and tools:
- `analytics_converter.py` - Tool for converting data from Mixpanel, Amplitude, and GA4 to TeloMesh format
- `README.md` - Overview of the utilities available
- `GUIDE.md` - Detailed guide for using the Analytics Converter

#### `/outputs`
Generated data files organized by dataset:
- Dataset-specific subdirectories (e.g., `outputs/myproject/`) containing:
  - `session_flows.csv` - Processed session data in tabular format
  - `user_graph.gpickle` - Serialized NetworkX graph with user journey information
  - `event_chokepoints.csv` - Identified friction points with WSJF scores
  - `high_friction_flows.csv` - User journeys that contain multiple high-friction points
  - `friction_node_map.json` - Mapping of pages to WSJF scores for visualization
  - `dataset_info.json` - Metadata about the dataset (creation timestamp, users, events, sessions)

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
- `telomesh logo.png` - Primary logo for application header
- `telomesh logo white.png` - White version of logo for dark backgrounds

### Key Files

- `main.py` - Entry point for the pipeline with command-line arguments
- `requirements.txt` - Package dependencies for easy installation
- `directory_structure.lua` - Canonical definition of project structure
- `README.md` - Project documentation
- `.streamlit/config.toml` - Streamlit configuration for dark theme and UI settings
- `.gitignore` - Configuration for Git to exclude temporary files

## Usage

### Running the Full Pipeline with Dataset Management

```bash
# Generate a new dataset with synthetic data
python main.py --dataset myproject --users 100 --events 50

# Process existing data into a new dataset
python main.py --dataset client_data --input path/to/events.csv
```

This will:
1. Create dataset directories in `data/` and `outputs/`
2. Generate synthetic events or use existing data
3. Parse events into sessions
4. Build a user journey graph
5. Validate flow metrics
6. Analyze friction points
7. Prepare data for dashboard
8. Generate dataset metadata

### Running the Dashboard

```bash
streamlit run ui/dashboard.py
```

This will launch the TeloMesh User Flow Intelligence Dashboard with the following features:
- Dataset selection dropdown
- Friction Analysis table with export to CSV
- User Flow Analysis with flow summaries
- User Journey Graph with interactive visualization
- Dark theme optimized for data visualization

### Running Individual Stages

You can run individual stages of the pipeline with dataset organization:

```bash
# Generate synthetic events for a specific dataset
python main.py --stage synthetic --dataset myproject --users 100 --events 50

# Parse events into sessions for a specific dataset
python main.py --stage parse --dataset myproject

# Build user journey graph for a specific dataset
python main.py --stage graph --dataset myproject

# Validate flow metrics for a specific dataset
python main.py --stage metrics --dataset myproject

# Analyze friction points for a specific dataset
python main.py --stage analysis --dataset myproject
```

### Using the Analytics Converter

TeloMesh includes a powerful utility to convert data from popular analytics platforms:

```bash
# Convert data and make it ready for TeloMesh in one step
python utils/analytics_converter.py --input your_analytics_export.csv --output data/myproject/events.csv --format [platform] --telomesh-format

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
python utils/analytics_converter.py --generate-sample --format amplitude --output data/myproject/events.csv --telomesh-format
```

For detailed documentation on analytics conversion, see the [Analytics Converter Guide](utils/GUIDE.md).

## WSJF Friction Scoring

TeloMesh uses a Weighted Shortest Job First (WSJF) approach to prioritize friction points:

`WSJF_Friction_Score = exit_rate × betweenness`

Where:
- `exit_rate`: Probability of session ending after the event
- `betweenness`: Page's structural importance in user flows

## Dashboard UI Overview

The TeloMesh dashboard consists of three main tabs:

1. **Friction Analysis**: Table of friction points ranked by WSJF score
2. **User Flow Analysis**: Summaries of user journeys with multiple friction points
3. **User Journey Graph**: Interactive visualization of user flows with friction highlighted

Each section includes detailed tooltips explaining the metrics and interactive elements for exploring the data.

## Multi-Dataset Management

TeloMesh now supports managing multiple datasets:

1. **Creation**: Use the `--dataset` parameter to specify a dataset name
2. **Organization**: Data is stored in dataset-specific directories
3. **Discovery**: The dashboard automatically finds available datasets
4. **Selection**: Switch between datasets using the sidebar dropdown
5. **Metadata**: View dataset information including creation date and size

## Troubleshooting

- **Missing dependencies**: Ensure all packages in `requirements.txt` are installed
- **File not found errors**: Check that all input files exist in the specified paths
- **Graph visualization issues**: Try adjusting node count filters in the dashboard
- **Streamlit theme errors**: If you encounter theme-related errors, make sure your Streamlit version is compatible with the theme options used or update `.streamlit/config.toml`
- **Dataset not found**: Ensure the dataset exists in the `outputs/` directory with the required files

## Next Steps

After identifying friction points, consider:
1. Prioritizing improvements based on WSJF scores
2. Running A/B tests on identified problematic flows
3. Creating multiple datasets to compare before/after changes
4. Monitoring metrics over time to track progress

## Getting Help

For more information, check the documentation in the repository or open an issue on the GitHub repository. 