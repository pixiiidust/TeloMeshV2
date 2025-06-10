# TeloMesh (Early Access Prototype)

* TeloMesh is a user journey analysis pipeline that processes session and event data to build a directed graph of user flows. 
* Provides actionable decision logic for product and UX managers to identify UX patterns, triage chokepoints, and prioritize high signal metrics.
* Efficiently outputs optimization opportunities from user flow data with advanced network analysis metrics.
* Features performance optimization for large datasets and specific UX recommendations based on network characteristics.

## Why use TeloMesh? 

TeloMesh enhances existing analytics platforms such as Mixpanel / Amplitude / GA4 by turning exported session data into graph-based UX diagnostics—revealing where friction clusters, why users drop off, and which fixes matter most.

**[📖View comparison.md to find out more](comparison.md).**

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

3. Generate a dataset and run the pipeline:
   ```bash
   # Standard analysis
   python main.py --dataset myproject --users 100 --events 50
   
   # For large datasets with performance optimization
   python main.py --dataset large_project --users 1000 --events 50 --fast
   
   # For multi-graph analysis
   python main.py --dataset detailed_analysis --users 100 --events 50 --multi
   
   # Control the number of unique pages (nodes) in the graph
   python main.py --dataset node_control --users 100 --events 50 --pages 32
   ```

4. Run the dashboard to analyze your data:
   ```bash
   streamlit run ui/dashboard.py
   ```

**[📖 View the complete setup guide](setup_guide.md)** for detailed installation and usage instructions.

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
- Multi-graph support for preserving detailed user journey paths

### 3. User Flow Analysis
- Identifies user paths containing multiple high-friction points
- Flow Sequences tab shows multi-step user journey sessions with 2+ chokepoints
- Transition Pairs tab reveals common page-to-page transitions across sessions
- Highlights sequences where users encounter cascading obstacles
- Prioritizes flow improvements based on cumulative friction
- Filters by path length steps and chokepoint count

### 4. Advanced Network Analysis & UX Recommendations
- **Network Structure Metrics:** Fractal dimension (journey complexity), Power-law alpha (hierarchy), and Percolation threshold (stability)
- **Decision Table:** Actionable UX insights with quadrant-based prioritization (High Priority, User Friction Only, Structural Only, Low Priority)
- **Priority Matrix:** Interactive visualization of structural importance vs user friction
- **Pattern Detection:** Identify recurring navigation loops and common exit paths
- **Critical Pages Analysis:** Automatic classification of pages into critical vs standard roles with specific improvement suggestions
- **Network Statistics:** Measure system connectivity, resilience, and node participation metrics
- **Comprehensive Metrics Glossary:** Detailed explanations for advanced users

### 5. Dataset Organization
- Create and manage multiple datasets with `--dataset` parameter
- Dataset metadata tracking (users, events, sessions, timestamp)
- Dataset discovery and selection in the dashboard UI
- Isolated outputs for multiple projects/experiments

### 6. Performance Optimization
- Fast mode for processing large datasets efficiently
- Optimized algorithms for pattern detection
- Reduced computational complexity for network analysis
- Processing time improvements of up to 70% for datasets with 1000+ users
- Memory usage optimization for complex graphs

### 7. Analytics Converter Utility
- Convert data from Mixpanel, Amplitude, and Google Analytics 4
- Automated column mapping for seamless integration
- Sample data generation for testing

## Theory & Methodology

TeloMesh uses graph theory and network analysis to identify friction points in user journeys and prioritizes them using WSJF. 

WSJF (Weighted Shortest Job First) is a prioritization method used in Agile and Lean development (especially the Scaled Agile Framework - SAFe) that ranks work items by dividing their value by their effort, so teams tackle the highest-value, lowest-effort tasks first:

1. **Graph Construction**: User sessions are converted into directed graphs where:
   - Nodes represent pages/screens
   - Edges represent user actions (events)
   - Edge weights correspond to transition frequency
   - MultiDiGraph option preserves all individual transitions

2. **Friction Detection Algorithm**: TeloMesh identifies problem areas using a composite scoring system:
   - **Exit Rate**: Percentage of users who abandon their journey at a specific (page, event) pair
   - **Betweenness Centrality**: Measures how critical a node is to the overall flow structure
   - **WSJF Friction Score**: Calculated as `exit_rate × betweenness`, prioritizing high-impact friction points

3. **Advanced Network Analysis**: TeloMesh employs sophisticated network science techniques:
   - **Fractal Dimension**: Measures the complexity of user navigation patterns (1.0-3.0)
   - **Power-law Alpha**: Quantifies the degree distribution characteristics (typically 2.0-5.0)
   - **Clustering Coefficient**: Measures how interconnected pages are (0.0-1.0)
   - **Percolation Threshold**: Identifies the critical point at which the network collapses (0.0-1.0)
   - **Fractal Betweenness**: Enhanced centrality measure that considers repeating subgraph patterns

4. **UX Pattern Recognition**: The system identifies common UI/UX patterns:
   - Linear bottlenecks: Sequential paths with high friction
   - Hub-and-spoke structures: Central pages with multiple connections
   - Tree hierarchies: Navigational branches with varying friction
   - Complex meshes: Interconnected page networks requiring simplification

5. **Percolation Analysis**: Inspired by network percolation theory, TeloMesh identifies:
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
- Generating specific UX recommendations based on network metrics
- Optimizing performance for large datasets with fast mode

By combining exit rates with structural importance, the WSJF scoring system ensures that improvements focus on areas with the highest impact on overall user experience.

## Dashboard Screenshots
### Discover UX pain points:

<img src="https://github.com/user-attachments/assets/f0ab3ebb-2e4a-416f-acd1-73c43a6eaa86" width="100%"/>
<br>

### Triage pain points along user session flows:

<img src="https://github.com/user-attachments/assets/89b9ec34-8b80-48a5-b74d-33cb7321fc3a" width="100%"/>
<br>
<img src="https://github.com/user-attachments/assets/b6a5893e-fc6d-45ac-a98a-05c0ffbbef36" width="100%"/>

### Map priorities with user journey graphs: 

<img src="https://github.com/user-attachments/assets/7b5da3a3-6cb3-4cfc-9052-1ee6972a198c" width="100%"/>
<br>
<img src="https://github.com/user-attachments/assets/d592a20f-75e3-4f3e-b792-c8035031cdc9" width="100%"/>
<img src="https://github.com/user-attachments/assets/ebb8c6d1-e694-4f20-8bd0-d8e331661d3c" width="100%"/>

### WIP Preview of Advanced Metrics dashboard (Section 4): 
Advanced Network Metrics leverages network science techniques to provide deeper structural insights into complex user flows:

* **Network Structure Metrics:** Analyze user journeys that branch and repeat at multiple scales (Fractal Dimension), hierarchical structure (Power-Law Alpha), and system stability (Percolation Threshold) with visual risk indicators

* **Decision Table & Recommendations:** Receive specific UX improvement suggestions for each page based on its structural importance and friction score, sorted by priority

* **Priority Matrix Visualization:** Plot pages by Fractal Betweenness (structural importance) vs. WSJF (user friction) to identify high-impact optimization opportunities

* **Recurring Pattern Detection:** Uncover common navigation loops and repeated sequences that may indicate users getting stuck in circular journeys

* **Network Stability Analysis:** Evaluate system resilience and identify critical pages that could disrupt the entire user experience if removed or modified

<img src="https://github.com/user-attachments/assets/7f7c5666-77f5-4ca9-a8be-70671f1d755b" width="100%"/>
<br>
<img src="https://github.com/user-attachments/assets/85e4729a-eb33-4b9e-9825-c92dcce76664" width="100%"/>



## New in Version 2.2
- Advanced network analysis with fractal dimension and power-law metrics
- Multi-graph support for detailed user journey analysis
- UX recommendations engine with decision table
- Enhanced Flow Analysis with separate Flow Sequences and Transition Pairs tabs
- Performance optimization with fast mode for large datasets
- Enhanced output with comprehensive network metrics reports
- Improved UI labels and descriptions for better clarity

## New in Version 2.1
- Dataset organization and discovery
- Enhanced dashboard with improved UI labels and descriptions
- Improved visualization for dark theme
- Multi-dataset support for A/B testing comparison
- Updated terminology for improved clarity

## Future Versions

The TeloMesh roadmap includes several enhancements planned for upcoming versions:

### Customizable WSJF Friction Scoring Framework
- **Custom user-defined metrics**, including revenue loss per exit, conversion rate impact, and custom business metrics
- **Multi-factor scoring models** that can be tuned for different business priorities
- **Weighted page importance** based on business value or revenue impact
- **Integration with A/B testing frameworks** for measuring before/after improvement impact
- **Custom threshold settings** for friction classification and alerting

### Enterprise Scale
- **Massive Session Testing**: Scale to analyzing datasets with millions of sessions
- **Distributed Processing**: Support for processing huge datasets across multiple nodes
- **Cloud Deployment**: Containerized deployment for cloud environments
- **Real-time Analysis**: Stream processing of incoming user data
- **Long-term Trend Analysis**: Track UX improvements over time with historical comparisons

### AI-Powered UX Optimization
- **Machine learning-based friction prediction** to identify potential issues before they occur
- **Automatic pattern detection** for complex user behavior
- **Semantic clustering** of user journeys and intent mapping
- **Natural language explanations** of complex network patterns
- **Generative AI recommendations** for UX improvements

### Extended Visualization
- **3D graph visualization** for complex user journey spaces
- **Interactive simulation** of UX changes and their potential impact
- **Executive dashboards** with high-level KPIs and metrics
- **Custom visualization themes** for branded reports
- **Animation of user flow changes** over time or between versions

### Advanced Analytics
- **Custom NSM Settings**: Allow Global North Star Metrics to be defined for specific pages/events
- **Journey Comparison**: Compare before/after journeys to measure A/B testing improvement impact
- **Segmentation Analysis**: Filter friction analysis by user segments and cohorts
- **Predictive Flow Modeling**: Predict potential friction points in proposed new user journeys
- **Funnel Analysis**: Advanced conversion funnel visualization and optimization

### Enterprise Integration
- **Direct API Connections**: Native integrations with major analytics platforms
- **Data Warehousing**: Connect directly to data warehouses like Snowflake or BigQuery
- **CI/CD Integration**: Automate friction analysis as part of continuous integration pipelines
- **Team Collaboration**: Multi-user support with commenting and sharing
- **Export to Product Management Tools**: Integration with JIRA, Aha!, and other PM tools

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

### Developing with TeloMesh

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
   # Standard analysis
   python main.py --dataset myproject --users 100 --events 50
   
   # For large datasets with performance optimization
   python main.py --dataset large_project --users 1000 --events 50 --fast
   
   # For multi-graph analysis
   python main.py --dataset detailed_analysis --users 100 --events 50 --multi
   
   # Control the number of unique pages (nodes) in the graph
   python main.py --dataset node_control --users 100 --events 50 --pages 32
   ```

See `setup_guide.md` for detailed parameter documentation and advanced usage examples.
