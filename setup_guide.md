# TeloMesh Setup Guide

## Introduction

TeloMesh is a user journey analysis tool designed to help product managers identify friction points in user flows. It combines traditional user journey analytics with advanced network science techniques to provide deeper insights into user behavior patterns and actionable UX recommendations. This guide will walk you through the setup process to get TeloMesh running on your system.

## Quick Start

For those who want to get started immediately:

```bash
# Clone the repository
git clone https://github.com/yourusername/TeloMesh.git
cd TeloMesh

# Install dependencies
pip install -r requirements.txt

# Generate a test dataset and run analysis
python main.py --dataset quickstart --users 100 --events 50

# Launch the dashboard
streamlit run ui/dashboard.py
```

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
- **Graph Building**: Create directed graphs (DiGraph and MultiDiGraph) of user flows
- **Flow Metrics**: Validate session and graph quality metrics
- **Friction Analysis**: Identify chokepoints and fragile flows in user journeys
- **Advanced Network Analysis**: Calculate metrics like fractal dimension, power-law alpha, and more
- **UX Recommendations Engine**: Generate actionable insights based on network characteristics
- **Dashboard**: Interactive visualization of friction points and user flows
- **Analytics Import**: Convert data from Mixpanel, Amplitude, and Google Analytics 4
- **Dataset Management**: Create, organize, and switch between multiple datasets
- **Performance Optimization**: Fast mode for processing large datasets efficiently

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
- `build_graph.py` - Creates directed graphs (DiGraph and MultiDiGraph) of user journeys
- `flow_metrics.py` - Validates sessions and graph quality metrics
- `data_pipeline.md` - Documentation about the data ingestion process

#### `/analysis`
Advanced analysis components for discovering insights:
- `event_chokepoints.py` - Identifies friction points and performs advanced network analysis
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
  - `user_graph_multi.gpickle` - MultiDiGraph for advanced network analysis
  - `event_chokepoints.csv` - Identified friction points with WSJF scores
  - `high_friction_flows.csv` - User journeys that contain multiple high-friction points
  - `friction_node_map.json` - Mapping of pages to WSJF scores for visualization
  - `decision_table.csv` - UX recommendations based on network analysis
  - `final_report.json` - Summary of key network metrics
  - `final_report.csv` - Tabular format of network metrics
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
- `test_event_chokepoints.py` - Tests for friction point identification and network analysis
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

# Use fast mode for large datasets
python main.py --dataset large_project --users 1000 --events 50 --fast

# Create a MultiDiGraph instead of standard DiGraph
python main.py --dataset multi_edge_project --users 100 --events 50 --multi
```

This will:
1. Create dataset directories in `data/` and `outputs/`
2. Generate synthetic events or use existing data
3. Parse events into sessions
4. Build user journey graphs (standard DiGraph and MultiDiGraph for advanced analysis)
5. Validate flow metrics
6. Analyze friction points and perform advanced network analysis
7. Generate UX recommendations based on network characteristics
8. Prepare data for dashboard
9. Generate dataset metadata

### Running the Dashboard

```bash
streamlit run ui/dashboard.py
```

This will launch the TeloMesh User Flow Intelligence Dashboard with the following features:
- Dataset selection dropdown
- Friction Analysis table with export to CSV
- User Flow Analysis with flow summaries
- User Journey Graph with interactive visualization
- UX Recommendations based on network analysis
- Dark theme optimized for data visualization

### Running Individual Stages

You can run individual stages of the pipeline with dataset organization:

```bash
# Generate synthetic events for a specific dataset
python main.py --stage synthetic --dataset myproject --users 100 --events 50

# Parse events into sessions for a specific dataset
python main.py --stage parse --dataset myproject

# Build user journey graph for a specific dataset
python main.py --stage graph --dataset myproject --multi

# Validate flow metrics for a specific dataset
python main.py --stage metrics --dataset myproject

# Analyze friction points for a specific dataset
python main.py --stage chokepoints --dataset myproject --fast
```

### Using the Analytics Converter

TeloMesh includes a QOL utility to convert data from popular analytics platforms:

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

## Analysis Methodology

### WSJF Friction Scoring

TeloMesh uses a Weighted Shortest Job First (WSJF) approach to prioritize friction points:

`WSJF_Friction_Score = exit_rate × betweenness`

Where:
- `exit_rate`: Probability of session ending after the event
- `betweenness`: Page's structural importance in user flows

### Advanced Network Analysis

TeloMesh includes sophisticated network analysis metrics:

- **Fractal Dimension (D)**: Measures the complexity of user navigation patterns (1.0-3.0)
  - Lower values indicate simpler, more linear user flows
  - Higher values suggest complex, interconnected navigation patterns
  
- **Power-law Alpha (α)**: Quantifies the degree distribution characteristics (typically 2.0-5.0)
  - Higher values indicate more centralized hub-and-spoke structures
  - Lower values suggest more distributed connection patterns
  
- **Clustering Coefficient**: Measures how interconnected pages are (0.0-1.0)
  - Higher values indicate groups of pages that are highly interconnected
  - Lower values suggest more tree-like or linear structures
  
- **Percolation Threshold**: Identifies the critical point at which the network collapses (0.0-1.0)
  - Lower values indicate fragile networks vulnerable to disruption
  - Higher values suggest more robust user journey structures
  
- **Fractal Betweenness**: Enhanced centrality measure that considers repeating subgraph patterns
  - Identifies critical junction points in the user journey
  - Highlights pages that serve as connectors between different functional areas
  
- **Repeating Subgraph Detection**: Identifies common navigation patterns in user journeys
  - Reveals frequently traversed paths and user behavior patterns
  - Helps identify optimizable sequences and workflows

### UX Recommendations Engine

Based on network analysis, TeloMesh generates a decision table with actionable UX recommendations:

- **Structural Role Classification**: Automatically categorizes pages into roles:
  - Standard pages: Normal content or functional pages
  - Bottlenecks: Pages with high friction that impede user flow
  - Critical junctions: Pages that connect multiple user paths
  - Hub pages: Central navigation points with multiple connections
  
- **Pattern Recognition**: Identifies common UX patterns requiring attention:
  - Linear bottlenecks: Sequential paths with high friction
  - Hub-and-spoke issues: Central pages causing navigation problems
  - Tree hierarchy friction: Navigational branches with varying friction
  - Complex mesh simplification: Overlapping page networks needing streamlining
  
- **Actionable Recommendations**: Provides specific suggestions tailored to page characteristics:
  - Design improvements for high-friction pages
  - Content optimizations for exit-prone pages
  - Navigation enhancements for critical junction points
  - Structural changes for inefficient user flows
  
- **Decision Priority**: Ranks improvement opportunities by potential impact:
  - Based on combined metrics (WSJF, network position, user volume)
  - Considers both page-level and flow-level optimizations

## Output Files

TeloMesh generates several key output files for analysis:

### Core Friction Analysis Files
- **event_chokepoints.csv**: Comprehensive list of all page-event pairs with their friction metrics
- **high_friction_flows.csv**: User sessions containing multiple high-friction points
- **friction_node_map.json**: Page-to-WSJF score mapping for visualization

### Advanced Analysis Files
- **decision_table.csv**: UX recommendations table with the following columns:
  - `node`: Page identifier
  - `D`: Fractal dimension
  - `alpha`: Power-law exponent
  - `FB`: Fractal betweenness
  - `percolation_role`: Critical/Standard classification
  - `wsjf_score`: Weighted friction score
  - `ux_label`: Pattern classification (e.g., "linear bottleneck")
  - `suggested_action`: Specific recommendation

- **final_report.json**: JSON summary of key network metrics, including:
  - `fractal_dimension`: Overall complexity measure
  - `power_law_alpha`: Degree distribution exponent
  - `clustering_coefficient`: Network interconnectedness
  - `percolation_threshold`: Network robustness measure
  - `top_fb_nodes`: Highest fractal betweenness pages
  - `top_chokepoints`: Highest WSJF friction pages

- **final_report.csv**: Tabular version of the same metrics for easy import into spreadsheets or dashboards

## Dashboard UI Overview

The TeloMesh dashboard consists of three main tabs:

1. **Friction Analysis**: Table of user event friction points ranked by user dropoffs, exit rate and WSJF score
2. **User Flow Analysis**: Summaries of user journeys with multiple friction points across path steps
3. **User Journey Graph**: Interactive visualization of user flow terrain with friction highlighted across connecting events
   - Color-coded nodes by friction percentile (🔴 Top 10%, 🟠 Top 20%, 🟢 Top 50%, ⚪ Lower friction)
   - Three layout options when physics is disabled:
     - Friction Levels: Nodes arranged by friction severity (problems at top)
     - Funnel Stages: Nodes arranged by user journey stage (entry → exit)
     - Journey Centrality: Nodes arranged by betweenness centrality (hubs centered)
   - Configurable physics settings with optimized parameters

Each section includes detailed tooltips explaining the metrics and interactive elements for exploring the data.

## Performance Optimization

For large datasets, TeloMesh provides a `--fast` mode:

- Optimized algorithms for subgraph detection and percolation simulation
- Sampling techniques for large networks
- Skips detailed statistics output and visualization preprocessing
- Significant performance improvements for datasets with 1000+ users
- Maintains accuracy of core metrics while reducing processing time

## Multi-Dataset Management

TeloMesh supports managing multiple datasets:

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
- **Performance issues**: For large datasets, use the `--fast` flag to optimize processing

## Next Steps

After identifying friction points, consider:
1. Reviewing the decision table and UX recommendations for actionable insights
2. Prioritizing UX improvements based on WSJF scores and network metrics
3. Running A/B tests on identified problematic flows
4. Creating multiple datasets to compare before/after changes
5. Monitoring metrics over time to track progress

## Getting Help

For more information, check the documentation in the repository or open an issue on the GitHub repository. 
