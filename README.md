# TeloMesh - UX Journey Intelligence (Alpha)

> Graph-based UX analysis for prioritizing high-impact improvements in user journeys

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Dashboard Screenshots](#dashboard-screenshots)
- [Theory & Methodology](#theory--methodology)
- [For Product Teams](#for-product-teams)
- [What's New](#whats-new)
- [Future Versions](#future-versions)
- [Developing with TeloMesh](#developing-with-telomesh)
- [Repository Structure](#repository-structure)
- [Recent Improvements](#recent-improvements)

## Overview

TeloMesh is a user-journey analysis pipeline that transforms session and event data into a directed graph of user flows. It provides:

- **Actionable decision logic** for product and UX teams to identify patterns, triage chokepoints, and prioritize improvements  
- **Efficient opportunity detection** using advanced network-analysis metrics  
- **Scalable performance** for large datasets, with tailored UX recommendations  

By turning exported session data from Mixpanel, Amplitude, or GA4 into graph-based diagnostics, TeloMesh reveals where friction clusters, why users drop off, and which fixes matter most.

**[📖 View comparison and ML potential with other analytics tools](comparison.md)**

## Key Features

### 1. Friction Analysis Dashboard
- Rank UX pain points by WSJF score (weighted combination of exit rate and structural importance)
- Identify high-impact abandonment points with quantified user loss metrics
- Export prioritized opportunities for roadmap planning and team alignment

### 2. Interactive Journey Visualization
- Color-coded journey graph with friction hotspots (red for highest WSJF scores)
- Multiple view options: Friction Levels, Funnel Stages, or Journey Centrality
- Interactive exploration with tooltips, physics controls, and multi-graph support

### 3. User Flow Analysis
- Flow Sequences tab shows multi-step user journeys containing multiple chokepoints
- Transition Pairs tab reveals common page-to-page navigation patterns causing friction
- Filter and sort by path length, chokepoint count, and friction severity

### 4. Advanced Network Metrics & Recommendations (WIP)
- Network structure analysis with fractal dimension, power-law alpha, and percolation threshold
- Decision table with quadrant-based prioritization and page-specific UX recommendations
- Pattern detection for recurring navigation loops and critical structural chokepoints

### 5. Enterprise Features
- **Dataset Organization**: Create and manage multiple datasets with dashboard selection
- **Performance Optimization**: Fast mode processing with 70% speed improvement for large datasets
- **Analytics Integration**: Convert data from Mixpanel, Amplitude, and Google Analytics 4

## Getting Started

### Quick Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TeloMesh.git
   cd TeloMesh
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Generate a dataset or import your own analytics data:
   ```bash
   # Generate synthetic data with different options (PowerShell compatible)
   # Standard analysis pipeline
   python main.py --dataset my_project --users 100 --events 30 --pages 30
   
   # For larger datasets with performance optimization (less granular results)
   python main.py --dataset large_project --users 10000 --events 50 --pages 50 --fast
   
   # For multi-graph analysis preserving individual transitions
   python main.py --dataset detailed_analysis --users 10000 --events 30 --pages 30 --multi
   ```
   
   **Parameters:**
   - `--users`: Number of simulated users
   - `--events`: Number of actions per user
   - `--pages`: Number of screen nodes in the journey
   - `--dataset`: Name of the output folder where results will be stored
   
   ```bash
   # Convert and import analytics data from various platforms
   # Import own analytics data with different source format options
   python utils/analytics_converter.py --format mixpanel --input your_data.json --output mydata --telomesh-format
   python utils/analytics_converter.py --format ga4 --input your_data.csv --output mydata --telomesh-format
   python utils/analytics_converter.py --format amplitude --input your_data.json --output mydata --telomesh-format
   
   # Then run analysis on the imported data
   python main.py --dataset mydata
   ```

4. Run the dashboard to analyze your data:
   ```bash
   streamlit run ui/dashboard.py
   ```

**[📖 View the complete setup guide](setup_guide.md)** for detailed installation and usage instructions.

## Dashboard Screenshots

### Identify and rank UX pain points by WSJF score, exit rate, and user abandonment:

<img src="https://github.com/user-attachments/assets/f0ab3ebb-2e4a-416f-acd1-73c43a6eaa86" width="100%"/>
<br>

### Triage critical chokepoints that lead to user dropoffs across multi-step journey sequences:

<img src="https://github.com/user-attachments/assets/89b9ec34-8b80-48a5-b74d-33cb7321fc3a" width="100%"/>
<br>
<img src="https://github.com/user-attachments/assets/b6a5893e-fc6d-45ac-a98a-05c0ffbbef36" width="100%"/>

### Visualize high-impact UX optimization targets with WSJF-prioritized journey graphs:

<img src="https://github.com/user-attachments/assets/e7a6d71b-0c52-4a19-8100-6fcbd2747204" width="100%"/>
<br>
<img src="https://github.com/user-attachments/assets/386f3115-5a70-401f-bcc9-fba2a70b18c5" width="100%"/>
<img src="https://github.com/user-attachments/assets/ca4226bc-6f68-4410-b08c-2eeafddcd068" width="100%"/>

### Dashboard Sidebar (switch between datasets / view dataset details / glossary for network terminology):
<img src="https://github.com/user-attachments/assets/6682fa2b-edfc-4ab6-b7be-93ba17c540f1" width="100%"/>
<br>

### Preview of advanced metrics tab (UI & Testing WIP): 
Advanced Network Metrics leverages network science techniques to provide deeper structural insights into complex user flows:

* **Network Structure Metrics:** Analyze user journeys that branch and repeat at multiple scales (Fractal Dimension), hierarchical structure (Power-Law Alpha), and system stability (Percolation Threshold) with visual risk indicators

* **Decision Table & Recommendations:** Receive specific UX improvement suggestions for each page based on its structural importance and friction score, sorted by priority

* **Priority Matrix Visualization:** Plot pages by Fractal Betweenness (structural importance) vs. WSJF (user friction) to identify high-impact optimization opportunities

* **Recurring Pattern Detection:** Uncover common navigation loops and repeated sequences that may indicate users getting stuck in circular journeys

* **Network Stability Analysis:** Evaluate system resilience and identify critical pages that could disrupt the entire user experience if removed or modified

<img src="https://github.com/user-attachments/assets/7f7c5666-77f5-4ca9-a8be-70671f1d755b" width="100%"/>
<br>
<img src="https://github.com/user-attachments/assets/85e4729a-eb33-4b9e-9825-c92dcce76664" width="100%"/>

## For Product Teams

TeloMesh helps product teams by:
- Quantifying UX friction points for prioritization
- Visualizing critical user journey bottlenecks
- Identifying high-impact improvement opportunities
- Providing data-driven insights for roadmap planning
- Enabling before/after comparisons of UX changes
- Generating specific UX recommendations based on network metrics
- Optimizing performance for large datasets with fast mode

By combining exit rates with structural importance, the WSJF scoring system ensures that improvements focus on areas with the highest impact on overall user experience.

## Theory & Methodology

* TeloMesh applies network science and graph theory to user experience analysis, creating a systematic approach to prioritizing UX improvements.
* Approach provides scaffold to integrate GNN based machine learning agents for UX optimizations at scale:

### Core Framework: WSJF Prioritization
**WSJF** (Weighted Shortest Job First) is an Agile prioritization method that ranks work by value-to-effort ratio, helping teams focus on high-impact, low-effort improvements. TeloMesh extends this concept to UX optimization:

### 1. Journey Mapping & Analysis
- **Graph Construction**: Convert user sessions into directed graphs where:
  - Nodes represent pages/screens
  - Edges represent user actions (events)
  - Edge weights correspond to transition frequency
  - MultiDiGraph option preserves individual transitions

- **Friction Detection**: Identify problem areas using a composite scoring system:
  - **Exit Rate**: Percentage of users abandoning at specific (page, event) pairs
  - **Betweenness Centrality**: Measures node importance in the overall flow structure
  - **WSJF Friction Score**: Calculated as `exit_rate × betweenness`, highlighting high-impact friction points

### 2. Pattern Recognition & Structural Analysis
- **UX Pattern Recognition**: Identify common interaction patterns:
  - Linear bottlenecks: Sequential paths with high friction
  - Hub-and-spoke structures: Central pages with multiple connections
  - Tree hierarchies: Navigational branches with varying friction
  - Complex meshes: Interconnected page networks requiring simplification

- **Percolation Analysis**: Identify critical network components:
  - Critical junctions where small improvements yield large UX gains
  - Cascading failure patterns where multiple friction points compound
  - Fragile flows where users encounter multiple high-WSJF obstacles

### 3. Advanced Network Metrics
- **Fractal Dimension** (1.0-3.0): Measures the complexity of user navigation patterns
- **Power-law Alpha** (2.0-5.0): Quantifies degree distribution characteristics
- **Clustering Coefficient** (0.0-1.0): Measures how interconnected pages are
- **Percolation Threshold** (0.0-1.0): Identifies critical points for network collapse
- **Fractal Betweenness**: Enhanced centrality measure that considers repeating subgraph patterns

## What's New

### Version 2.2
- Advanced network analysis with fractal dimension and power-law metrics
- Multi-graph support for detailed user journey analysis
- UX recommendations engine with decision table
- Enhanced Flow Analysis with separate Flow Sequences and Transition Pairs tabs
- Performance optimization with fast mode for large datasets
- Enhanced output with comprehensive network metrics reports
- Improved UI labels and descriptions for better clarity

### Version 2.1
- Dataset organization and discovery
- Enhanced dashboard with improved UI labels and descriptions
- Improved visualization for dark theme
- Multi-dataset support for A/B testing comparison
- Updated terminology for improved clarity

## Future Versions

The TeloMesh roadmap is organized by complexity and priority:

### 1. Core Enhancements (Near-term)
- **Customizable Scoring:** User-defined metrics (revenue impact, conversion rates) and weighted page importance
- **Advanced Analytics:** Journey comparison, segmentation analysis, and custom North Star Metrics 
- **Visualization Improvements:** Interactive simulations, executive dashboards, and custom themes

### 2. Enterprise Features (Mid-term)
- **Scale & Performance:** Support for millions of sessions, distributed processing, and cloud deployment
- **Integration:** Native connections to analytics platforms, data warehouses, and PM tools (JIRA, Aha!)
- **Team Collaboration:** Multi-user support with commenting and sharing capabilities

### 3. AI & Advanced Capabilities (Long-term)
- **Machine Learning:** GNN based prediction via learned weights for UX pattern detection, and semantic clustering
- **Generative AI:** Natural language explanations and AI-powered UX recommendations
- **Advanced Visualization:** 3D journey mapping, animation of user flow changes, and forecast simulations

For more details on contributing to these features, see our [contribution guidelines](contributing.md).

## Developing with TeloMesh

For developers looking to contribute or customize:

1. Set up your development environment:
   ```bash
   # Clone the repo with development branch
   git clone -b dev https://github.com/yourusername/TeloMesh.git
   cd TeloMesh
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. Run tests to verify your setup:
   ```bash
   python -m pytest tests/
   ```

3. Common development workflows:
   ```bash
   # Generate test dataset and run in debug mode
   python main.py --dataset test_dev --users 20 --events 15
   
   # Run the dashboard with hot reloading
   streamlit run ui/dashboard.py --server.runOnSave=true
   ```

See [contributing.md](contributing.md) for code style guidelines and pull request process.

## Repository Structure

### Core Directories
- `data/` - Data generation tools and input data storage
- `ingest/` - Data ingestion and session parsing
- `analysis/` - Analysis of user flows and friction points
- `ui/` - User interface dashboards
- `utils/` - Utility scripts including the Analytics Converter
- `outputs/` - Generated output files, organized by dataset
- `tests/` - Test files for the project
- `logs/` - Logging and monitoring data

### Key Files
- `main.py` - Entry point for the pipeline with command-line arguments
- `requirements.txt` - Package dependencies for easy installation
- `directory_structure.lua` - Canonical definition of project structure
- `README.md` - Project documentation (this file)
- `.streamlit/config.toml` - Streamlit configuration for dark theme and UI settings
- `.gitignore` - Configuration for Git to exclude temporary files
- `utils/analytics_converter.py` - Tool for converting data from analytics platforms
- `utils/README.md` - Comprehensive documentation for the Analytics Converter

## Recent Improvements

### Robust WSJF Threshold Calculation

The TeloMesh dashboard has been updated to handle large datasets (100K+ users) by implementing a more robust threshold calculation for WSJF Friction Scores. This fixes an issue where the 90th percentile would become 0.0 with zero-inflated data, causing all non-zero scores to be incorrectly identified as chokepoints.

Key enhancements:
- Uses median and MAD (Median Absolute Deviation) for robustness against outliers
- Filters zero scores before statistical calculations
- Implements adaptive thresholding to ensure a reasonable number of chokepoints
- Handles edge cases elegantly (all zeros, few non-zeros, low variability)

This improvement was developed using Test-Driven Development (TDD) with a comprehensive test suite that includes regression testing for the specific 100K user issue.
