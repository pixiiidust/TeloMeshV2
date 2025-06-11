# UI Components

This folder contains the TeloMesh User Flow Intelligence Dashboard, which visualizes user journey analysis through interactive components.

## Table of Contents
- [Key Features](#key-features)
- [Dashboard Tabs](#dashboard-tabs)
- [User Journey Graph Features](#user-journey-graph-features)
- [Advanced Metrics Features](#advanced-metrics-features)
- [Developer Controls Details](#developer-controls-details)
- [Flow Analysis Features](#flow-analysis-features)
- [Sidebar Components](#sidebar-components)
- [Export Functionality](#export-functionality)
- [Dataset Management](#dataset-management)

## Key Features
- Dataset selection dropdown for managing multiple projects
- Interactive Friction Analysis with sortable metrics
- User Flow Analysis with path visualization
- User Journey Graph with network visualization and multiple layout options
- Advanced Metrics dashboard with network analysis visualizations
- Export functionality for all dashboard sections (CSV and HTML)
- Dark theme optimized for data visualization
- Tooltips for all metrics and analyses

The main file `dashboard.py` provides a comprehensive Streamlit interface for product managers to identify and analyze user experience friction points.

## Dashboard Tabs
- **Friction Analysis**: Table of friction points ranked by WSJF score
- **Flow Analysis**: Analysis of user journeys with multiple friction points and transition patterns
- **User Journey Graph**: Interactive visualization of user flows with friction highlighted
- **Advanced Metrics**: Network analysis metrics with interactive visualizations

## User Journey Graph Features
- **Node Color Coding**:
  - Red: Top 10% friction (highest WSJF scores)
  - Yellow: Top 20% friction
  - Green: Top 50% friction
  - Gray: Lower friction nodes
  
- **Layout Arrangements** (when physics is disabled):
  - **Friction Levels**: Nodes arranged by friction level with high-friction pages at top, low-friction at bottom, creating a vertical heat map of friction intensity
  - **Funnel Stages**: Nodes arranged by user journey position from entry pages (left) to exit pages (right), showing natural flow progression
  - **Journey Centrality**: Nodes arranged by betweenness centrality with most central nodes in the middle, revealing structural hubs
  - Each layout can be toggled with a single click to provide different analytical perspectives

- **Enhanced Visualization**:
  - Larger container size (1000px height) for better visibility of complex graphs
  - Reduced edge opacity (0.25) for less visual clutter
  - Curved edges with reduced roundness (0.15) for cleaner path visualization
  - Improved physics settings with stronger repulsion and node spacing
  - Node selection feature to highlight specific nodes regardless of filters

- **Physics Settings**:
  - Toggle physics simulation on/off with immediate visual feedback
  - Gravitational constant (-15000) optimized for clear node separation
  - Spring length (200px) for appropriate spacing between connected nodes
  - Interactive node dragging with position maintenance when physics disabled

- **Filtering Options**:
  - Show all nodes or filter to top 10%, 20%, or 50% friction nodes
  - Enable/disable physics for different visualization styles
  - Node selection dropdown to always display specific nodes of interest
  - Export current graph view as interactive HTML with all interactivity preserved

- **Export Functionality**:
  - Export the current graph view as interactive HTML with a download link
  - Exported HTML preserves all interactivity, node positions, and physics settings
  - Standalone file that can be shared with stakeholders without TeloMesh installation

## Advanced Metrics Features
- **FB vs WSJF Priority Matrix**: Interactive scatter plot visualizing nodes by structural importance and user friction
  - Customizable scaling options (actual range, normalized, full range)
  - Quadrant-based prioritization (High Priority, User Friction Only, Structural Only, Low Priority)
  - **Node Symbol Meaning**:
    - **Solid Circles (●)**: Critical nodes whose removal would severely disrupt network connectivity
    - **Diamonds (◆)**: Standard nodes that are less critical to overall network structure
  - Size emphasis option to highlight critical nodes with larger markers
  - Detailed explanations of metrics and prioritization logic
  
- **Recurring User Flow Patterns**: Analysis of repeating sequences in user journeys
  - Identification of common loops and navigation patterns
  - Exit path analysis showing sequences that lead to users leaving the site
  - Visualization of pattern frequency and participation rates
  - **Node Participation Analysis**: Identification of nodes most frequently involved in recurring patterns
    - Top nodes ranked by loop count participation 
    - Bar chart visualization of node participation frequency
    - Percentage breakdown of node involvement in recurring patterns
    - Separate analysis for regular recurring patterns and exit paths
  
- **Network Stability Analysis**: Percolation threshold visualization
  - Simulation of network resilience to node removal
  - Critical node identification for maintaining network integrity
  - Line chart showing network collapse as critical nodes are removed
  
- **Network Structure Metrics**: Global metrics with diagnostic indicators
  - Fractal Dimension (D): Measures navigation complexity
  - Power-Law Alpha (α): Quantifies degree distribution characteristics
  - Percolation Threshold: Identifies network fragility

- **Developer Controls** (hidden section):
  - Analysis refresh option to regenerate metrics without rerunning the pipeline
  - Last analysis timestamp display for tracking updates
  - Diagnostic information for troubleshooting
  - Access to raw metrics files for advanced analysis

## Developer Controls Details
The Developer Controls section is accessible from the Advanced Metrics tab and provides:

- **Metrics Regeneration**: 
  - One-click recomputation of all network metrics from existing graph files
  - Fast mode enabled by default for quick updates
  - Success/error status messages with detailed logging
  
- **Debug Information**:
  - Session statistics log display with detailed session quality metrics
  - Raw metrics.json file viewer showing all computed network metrics
  - Last analysis timestamp to track when metrics were last updated
  
- **Usage Scenarios**:
  - After modifying network analysis algorithms to see updated results
  - When debugging unexpected metric values
  - To refresh metrics if files were modified externally
  - For comparing algorithm versions without regenerating synthetic data

## Flow Analysis Features
- **Flow Sequences tab**:
  - Filter flows by minimum flow length steps
  - Filter by minimum number of chokepoints (friction points)
  - Focus on top percentage of flows by total WSJF score
  - Visualize multi-step user journey sessions with highlighted chokepoints
  - Export filtered flows to CSV with detailed metrics
- **Transition Pairs tab**:
  - Analyze common page-to-page transitions across sessions
  - Filter by "from" page and "to" page
  - Filter by transitions with chokepoints (from, to, or both)
  - Sort by frequency, session count, or total WSJF score
  - Export transition data to CSV

## Sidebar Components
- **Advanced Metrics Glossary**: Detailed explanations of all metrics and their interpretations
- **Graph Statistics**: Key network statistics including:
  - Node count: Total number of unique pages in the journey
  - Edge count: Total number of transitions between pages
  - Edge/Node Ratio: Measure of graph density (higher values indicate more complex navigation)
  - Connected Components: Number of isolated subgraphs (ideally should be 1)
- **Dataset Selection**: Dropdown menu for selecting different datasets
- **Dataset Info**: Metadata about the currently selected dataset including:
  - Creation timestamp
  - Number of users
  - Number of events
  - Number of sessions

## Export Functionality
The dashboard provides comprehensive export capabilities:
- **CSV Export**: For tabular data including:
  - Friction Analysis table (full or filtered)
  - User Flow Analysis results
  - Decision Table recommendations
  - Recurring patterns data
- **HTML Export**: For interactive visualizations:
  - User Journey Graph with all interactivity preserved
  - FB vs WSJF Priority Matrix
  - Network visualizations

Each export is provided with a clear download link and includes a timestamp to track when the export was generated.

## Dataset Management
The dashboard automatically discovers datasets in the `outputs/` directory and provides a dropdown for selection. For each dataset, the dashboard displays:
- Dataset name
- Creation timestamp
- Number of users
- Number of events
- Number of sessions

If a dataset is not found or has missing files, appropriate error messages are displayed to guide the user. 