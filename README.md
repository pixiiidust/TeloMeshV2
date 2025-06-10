# TeloMesh (Early Access Prototype)

* TeloMesh is a user journey analysis pipeline that processes session and event data to build a directed graph of user flows. 
* Provides actionable decision logic for product and UX managers to identify UX patterns, triage chokepoints, and prioritize high signal metrics.
* Efficiently outputs optimization opportunities from user flow data with advanced network analysis metrics.
* Features performance optimization for large datasets and specific UX recommendations based on network characteristics.

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
   # Standard analysis
   python main.py --dataset myproject --users 100 --events 50
   
   # For large datasets with performance optimization
   python main.py --dataset large_project --users 1000 --events 50 --fast
   
   # For multi-graph analysis
   python main.py --dataset detailed_analysis --users 100 --events 50 --multi
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
- Multi-graph support for preserving detailed user journey paths

### 3. User Flow Analysis
- Identifies user paths containing multiple high-friction points
- Highlights sequences where users encounter cascading obstacles
- Prioritizes flow improvements based on cumulative friction

### 4. Advanced Network Analysis
- Fractal dimension calculation to measure user journey complexity
- Power-law alpha exponent to quantify network degree distribution
- Clustering coefficient analysis to evaluate page interconnectedness
- Percolation threshold simulation to assess network robustness
- Fractal betweenness centrality combining structural importance with repeating patterns
- Repeating subgraph detection to identify common navigation paths

### 5. UX Recommendations Engine
- Decision table with actionable UX insights based on network metrics
- Automatic classification of pages into structural role categories
- Specific improvement suggestions tailored to page characteristics
- Network pattern recognition for identifying UI/UX optimization opportunities
- Comprehensive final report with key metrics and top issues

### 6. Dataset Organization
- Create and manage multiple datasets with `--dataset` parameter
- Dataset metadata tracking (users, events, sessions, timestamp)
- Dataset discovery and selection in the dashboard UI
- Isolated outputs for multiple projects/experiments

### 7. Performance Optimization
- Fast mode for processing large datasets efficiently
- Optimized algorithms for subgraph detection
- Reduced computational complexity for network analysis
- Processing time improvements of up to 70% for datasets with 1000+ users
- Memory usage optimization for complex graphs

### 8. Analytics Converter Utility
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

## New in Version 2.2
- Advanced network analysis with fractal dimension and power-law metrics
- Multi-graph support for detailed user journey analysis
- UX recommendations engine with decision table
- Performance optimization with fast mode for large datasets
- Enhanced output with comprehensive network metrics reports

## New in Version 2.1
- Dataset organization and discovery
- Enhanced dashboard with improved UI labels and descriptions
- Customizable dark theme with improved visualization
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

If you're interested in contributing to any of these future features, please see our [contribution guidelines](CONTRIBUTING.md).

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
