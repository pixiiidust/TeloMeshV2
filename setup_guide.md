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
- **Advanced Network Analysis**: Calculate metrics like fractal dimension, power-law alpha, and fractal betweenness
- **Interactive Priority Matrix**: Visualize nodes by structural importance and user friction with customizable scaling
- **Recurring Patterns Analysis**: Detect repeating sequences and exit paths in user journeys
- **UX Recommendations Engine**: Generate actionable insights based on network characteristics
- **Interactive Dashboard**: Visualize friction points, user flows, and advanced network metrics
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
- `dashboard_v2.py` - Backup of the latest dashboard version with advanced metrics features
- `dashboard_components.md` - Documentation about the dashboard components

#### `/utils`
Utility scripts and tools:
- `analytics_converter.py` - Tool for converting data from Mixpanel, Amplitude, and GA4 to TeloMesh format
- `README.md` - Comprehensive documentation for the Analytics Converter

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
  - `metrics.json` - Summary of key network metrics and analysis results
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
- `test_advanced_metrics_tab.py` - Tests for advanced metrics visualization
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

# Control the number of unique pages/nodes in the generated graph
python main.py --dataset custom_nodes --users 100 --events 50 --pages 32
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
- Flow Analysis with two views:
  - Flow Sequences tab for analyzing multi-step user journey sessions with chokepoints
  - Transition Pairs tab for analyzing common page-to-page transitions across sessions
- User Journey Graph with interactive visualization
- Advanced Metrics with UX recommendations based on network analysis
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

For detailed documentation on analytics conversion, see the [Analytics Converter Guide](utils/README.md).

For generating large datasets; these recently tested runtimes for 10k, 20k, and 100k users, 20 events, 30 page nodes:
<img src="https://github.com/user-attachments/assets/bb028836-7b6e-47d5-af1d-1b32320dde53" width="100%"/>
<br>

## Analysis Methodology

### WSJF Friction Scoring

TeloMesh uses a Weighted Shortest Job First (WSJF) approach to prioritize friction points:

`WSJF_Friction_Score = exit_rate × betweenness`

Where:
- `exit_rate`: Probability of session ending after the event
- `betweenness`: Page's structural importance in user flows

### Advanced Network Analysis

TeloMesh includes sophisticated network analysis metrics:

- **Fractal Dimension (D)**: Measures the complexity of user navigation patterns (typically 2.0-3.0)
  - Lower values indicate simpler, more linear user flows
  - Higher values suggest complex, interconnected navigation patterns
  
- **Power-law Alpha (α)**: Quantifies the degree distribution characteristics (typically 1.8-2.6)
  - Lower values indicate stronger hierarchy with vulnerable hub nodes
  - Higher values suggest more uniform connection distribution
  
- **Percolation Threshold**: Identifies the critical point at which the network collapses (typically 0.2-0.5)
  - Lower values indicate fragile networks vulnerable to disruption
  - Higher values suggest more robust user journey structures
  
- **Fractal Betweenness (FB)**: Enhanced centrality measure for structural importance (typically 0.6-1.0)
  - Identifies critical junction points in the user journey
  - Highlights pages that serve as connectors between different functional areas
  
- **Recurring Patterns Detection**: Identifies common navigation loops and repetitive behaviors
  - Reveals frequently traversed paths and user behavior patterns
  - Helps identify optimizable sequences and workflows
  
- **Exit Path Analysis**: Detects sequences that frequently lead to users leaving the site
  - Tracks exit rates for common departure sequences
  - Reveals problematic paths that lead to user drop-off

### UX Recommendations Engine

Based on network analysis, TeloMesh generates a decision table with actionable UX recommendations:

- **Priority Matrix Classification**: Automatically categorizes pages into quadrants:
  - High Priority: High structural importance AND high user friction
  - User Friction Only: Pages causing friction but not critical to site structure
  - Structural Only: Important navigation nodes with low friction
  - Low Priority: Less important pages with minimal friction
  
- **Percolation Role**: Classifies nodes based on network resilience impact:
  - Critical nodes: Removing these would severely disrupt network connectivity
  - Standard nodes: Less critical for overall network stability
  
- **Actionable Recommendations**: Provides specific suggestions tailored to page characteristics:
  - Design improvements for high-friction pages
  - Content optimizations for exit-prone pages
  - Navigation enhancements for critical junction points
  - Structural changes for inefficient user flows
  
- **Decision Priority**: Ranks improvement opportunities by potential impact:
  - Based on combined metrics (WSJF, FB, percolation role)
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
  - `FB`: Fractal betweenness (structural importance)
  - `percolation_role`: Critical/Standard classification
  - `wsjf_score`: Weighted friction score
  - `ux_label`: Pattern classification (e.g., "redundant bottleneck")
  - `suggested_action`: Specific recommendation

- **metrics.json**: JSON summary of key network metrics, including:
  - `fractal_dimension`: Overall complexity measure
  - `power_law_alpha`: Degree distribution exponent
  - `percolation_threshold`: Network robustness measure
  - `recurring_patterns`: Repeating sequence information
  - `exit_paths`: Common exit sequences with exit rates
  - `network_metrics`: Graph statistics and structural properties

## Dashboard UI Overview

The TeloMesh dashboard consists of four main tabs:

1. **Friction Analysis**: Table of friction points ranked by WSJF score, exit rate, and users lost
2. **Flow Analysis**: Analysis of user journeys with two complementary views:
   - **Flow Sequences**: Multi-step user journey sessions containing 2+ chokepoints, filterable by path length steps and total WSJF score
   - **Transition Pairs**: Common page-to-page transitions across sessions, showing specific navigation patterns that cause friction
3. **User Journey Graph**: Interactive visualization of user flows with friction highlighted across connecting events
4. **Advanced Metrics**: Network analysis visualizations including FB vs WSJF Priority Matrix and Recurring Patterns analysis

### User Journey Graph Features
- Color-coded nodes by friction percentile (🔴 Top 10%, 🟠 Top 20%, 🟢 Top 50%, ⚪ Lower friction)
- Three layout options when physics is disabled:
  - Friction Levels: Nodes arranged by friction severity (problems at top)
  - Funnel Stages: Nodes arranged by user journey stage (entry → exit)
  - Journey Centrality: Nodes arranged by betweenness centrality (hubs centered)
- Configurable physics settings with optimized parameters

### Advanced Metrics Features
- **FB vs WSJF Priority Matrix**: Interactive scatter plot visualizing nodes by structural importance and user friction
  - Customizable scaling options (actual range, normalized, full range)
  - Quadrant-based prioritization (High Priority, User Friction Only, Structural Only, Low Priority)
  - Log scale options for better visualization of skewed data
  - Node labeling options with customizable highlighting
  
- **Recurring User Flow Patterns**: Analysis of repeating sequences in user journeys
  - Identification of common loops and navigation patterns
  - Exit path analysis showing sequences that lead to users leaving the site
  - Visualization of pattern frequency and participation rates
  
- **Network Stability Analysis**: Percolation threshold visualization
  - Simulation of network resilience to node removal
  - Critical node identification for maintaining network integrity
  
- **Network Structure Metrics**: Global metrics with diagnostic indicators
  - Fractal Dimension (D): Measures navigation complexity
  - Power-Law Alpha (α): Quantifies degree distribution characteristics
  - Percolation Threshold: Identifies network fragility

### Sidebar Components
- **Advanced Metrics Glossary**: Detailed explanations of all metrics and their interpretations
- **Graph Statistics**: Key network statistics including node count, edge count, and edge/node ratio
- **Dataset Selection**: Dropdown menu for selecting different datasets
- **Dataset Info**: Metadata about the currently selected dataset

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

### Session Parsing Options

```bash
# Parse sessions with default settings
python main.py --stage parse --dataset myproject

# Parse sessions with custom session gap (45 minutes instead of default 30)
python main.py --stage parse --dataset myproject --session-gap 45
```

The session gap parameter controls how TeloMesh identifies distinct user sessions:
- Default: 30 minutes (industry standard)
- Shorter gaps (e.g., 15 minutes): Creates more granular sessions, useful for fast-paced applications
- Longer gaps (e.g., 60 minutes): Better for content-heavy sites where users spend more time between interactions

## All Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | `default` | Name of the dataset to create |
| `--users` | `100` | Number of users to generate in synthetic data |
| `--events` | `50` | Average number of events per user |
| `--pages` | `16` | Number of unique pages (nodes) to generate |
| `--session-gap` | `30` | Minutes of inactivity to consider as a new session boundary |
| `--fast` | `False` | Skip slow scaling tests and detailed output |
| `--multi` | `False` | Create MultiDiGraph that preserves multiple edge types |
| `--stage` | `all` | Which stage to run (all, synthetic, parse, graph, metrics, chokepoints, dashboard) |
| `--input` | `None` | Path to input events file (instead of generating synthetic data) |

## Pipeline Stages

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
