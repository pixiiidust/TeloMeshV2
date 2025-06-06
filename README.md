# TeloMesh (Early Access Prototype)

* TeloMesh is a user journey analysis pipeline that processes session and event data to build a directed graph of user flows. 
* Provides actionable decision logic for product and UX managers to identify UX patterns, triage chokepoints, and prioritize high signal metrics.
* Efficiently outputs optimization opportunities from user flow data.

**[📖 View the complete setup guide](setup_guide.md)** for detailed installation and usage instructions.

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TeloMesh.git
   cd TeloMesh
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Generate a dataset and run the pipeline:
   ```bash
   python main.py --dataset myproject --users 100 --events 50
   ```

4. Run the dashboard to analyze your data:
   ```bash
   streamlit run ui/dashboard.py
   ```

For more detailed setup instructions, see the [Setup Guide](setup_guide.md).

## Key Features

### 1. Friction Points Analysis
- Identifies and ranks individual (page, event) pairs by WSJF Friction Score
- Calculates exit probability and lost user volume
- Highlights structural weaknesses in the user journey

### 2. User Journey Graph
- Visual representation of user journeys with friction highlighted
- Color-coded nodes based on friction percentile:
  - Red: Top 10% friction (highest WSJF scores)
  - Yellow: Top 20% friction 
  - Green: Top 50% friction
  - Gray: Lower friction nodes
- Multiple layout options when physics is disabled:
  - Friction Levels: Nodes arranged by friction severity
  - Funnel Stages: Nodes arranged in entry-to-exit order
  - Journey Centrality: Nodes arranged by betweenness centrality
- Interactive graph visualization with tooltips and event details
- Configurable physics settings with optimized gravitational constant

### 3. User Flow Analysis
- Identifies user paths containing multiple high-friction points
- Highlights sequences where users encounter cascading obstacles
- Prioritizes flow improvements based on cumulative friction

### 4. Dataset Organization
- Create and manage multiple datasets with `--dataset` parameter
- Dataset metadata tracking (users, events, sessions, timestamp)
- Dataset discovery and selection in the dashboard UI
- Isolated outputs for multiple projects/experiments

### 5. Analytics Converter Utility
- Convert data from Mixpanel, Amplitude, and Google Analytics 4
- Automated column mapping for seamless integration
- Sample data generation for testing

## Features

- **Synthetic Event Generation**: Generate realistic synthetic user event data
- **Session Parsing**: Extract structured session paths from event data
- **Graph Building**: Create a directed, event-typed graph of user flows
- **Flow Metrics**: Validate session and graph quality metrics
- **Friction Analysis**: Identify chokepoints and fragile flows in user journeys
- **Dashboard**: Interactive visualization of friction points and user flows
- **Analytics Import**: Convert data from Mixpanel, Amplitude, and Google Analytics 4
- **Dataset Management**: Create, discover, and switch between multiple datasets

## Theory & Methodology

TeloMesh uses graph theory and network analysis to identify friction points in user journeys and prioritizes them using WSJF. 

WSJF (Weighted Shortest Job First) is a prioritization method used in Agile and Lean development (especially the Scaled Agile Framework - SAFe) that ranks work items by dividing their value by their effort, so teams tackle the highest-value, lowest-effort tasks first:

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

## For Product Managers

TeloMesh helps product managers by:
- Quantifying UX friction points for prioritization
- Visualizing critical user journey bottlenecks
- Identifying high-impact improvement opportunities
- Providing data-driven insights for roadmap planning
- Enabling before/after comparisons of UX changes

By combining exit rates with structural importance, the WSJF scoring system ensures that improvements focus on areas with the highest impact on overall user experience.

## Dashboard Screenshots
Discover UX pain points:

<img src="https://github.com/user-attachments/assets/fb7321df-d86f-4506-857e-60daae9e92c6" width="600"/>
<br>

Triage pain points along user flows:

<img src="https://github.com/user-attachments/assets/4eec77aa-8490-42c9-8a9e-63dbd8b31418" width="600"/>
<br>

Map priorities with user journey graphs: 

<img src="https://github.com/user-attachments/assets/0e591fae-775d-490e-aeda-76ca0c1e8227" width="600"/>
<br>
<img src="https://github.com/user-attachments/assets/d25e7827-298e-46b5-bd4d-af98c00c3c7c" width="600"/>

## New in Version 2.1
- Dataset organization and discovery
- Enhanced dashboard with improved UI labels and descriptions
- Improved visualization for dark theme
- Multi-dataset support for A/B testing comparison
- Updated terminology for improved clarity

## Future Versions

The TeloMesh roadmap includes several enhancements planned for upcoming versions:

### Customizable WSJF Friction Scoring Framework
- **Custom metric inputs**, including revenue loss per exit, flow volume, touchpoint ROI, and continuity risk across the funnel.
- **Detect UX friction with sensitivity across multiple touchpoints** e.g., user traffic combined with downstream conversion probabilities, and funnel-weighted structural importance
- **Highlight structurally critical nodes (high betweenness)**, adaptively weighting early to mid-funnel stages to surface UX breakpoints: high-variance metrics like end funnel revenue often overlook small wins completely.
- **Enable small, leveraged UX interventions**, e.g. 2–5% improvements in cohort flows can drive incremental, outsized gains in NSMs / OKRs
- **Quantify precise signals for positive/negative user flow cascades**, verified empircally by percolation analysis + graph data 

### Performance & Scale
- **Large Session Testing**: Optimize the pipeline for analyzing datasets with millions of sessions
- **Distributed Processing**: Support for processing large datasets across multiple nodes
- **Performance Benchmarks**: Standard tests to evaluate processing time across different dataset sizes

### Optimizing UX via Machine Learning
- **Mapping as graph native input** to machine learning agents via vector embeddings and knowledge graphs (eg.cognee).
- **Structural user behavior analysis + AI reasoning** = Actionable optimization intelligence that scales across products and learns from every UX implementation.​​​​​​​​​​​​​​​​

### Enhanced User Interface
- **Web-Based Upload Interface**: A frontend interface allowing users to directly upload analytics files
- **Drag-and-Drop Data Import**: Intuitive interface for importing and mapping data
- **Visualization Exports**: Additional export formats for reports and presentations

### Advanced Analytics
- **Custom NSM Settings**: Allow Global North Star Metrics to be defined for specific pages/events
- **Journey Comparison**: Compare before/after journeys to measure A/B testing improvement impact
- **Segmentation Analysis**: Filter friction analysis by user segments and cohorts
- **Predictive Flow Modeling**: Predict potential friction points in proposed new user journeys

### Integration
- **Direct API Connections**: Native integrations with major analytics platforms
- **Data Warehousing**: Connect directly to data warehouses like Snowflake or BigQuery
- **CI/CD Integration**: Automate friction analysis as part of continuous integration pipelines

If you're interested in contributing to any of these future features, please see our [contribution guidelines](contributing.md).

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
- `utils/GUIDE.md` - Detailed guide for using the Analytics Converter
