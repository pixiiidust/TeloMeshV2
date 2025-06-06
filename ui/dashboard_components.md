# UI Components

This folder contains the TeloMesh User Flow Intelligence Dashboard, which visualizes user journey analysis through interactive components.

## Key Features
- Dataset selection dropdown for managing multiple projects
- Interactive Friction Analysis with sortable metrics
- User Flow Analysis with path visualization
- User Journey Graph with network visualization and multiple layout options
- Export functionality for all dashboard sections (CSV and HTML)
- Dark theme optimized for data visualization
- Tooltips for all metrics and analyses

The main file `dashboard.py` provides a comprehensive Streamlit interface for product managers to identify and analyze user experience friction points.

## Dashboard Tabs
- **Friction Analysis**: Table of friction points ranked by WSJF score
- **User Flow Analysis**: Summaries of user journeys with multiple friction points
- **User Journey Graph**: Interactive visualization of user flows with friction highlighted

## User Journey Graph Features
- **Node Color Coding**:
  - Red: Top 10% friction (highest WSJF scores)
  - Yellow: Top 20% friction
  - Green: Top 50% friction
  - Gray: Lower friction nodes
  
- **Layout Arrangements** (when physics is disabled):
  - **Friction Levels**: Nodes arranged by friction level (problems at top)
  - **Funnel Stages**: Nodes arranged by user funnel stages (entry → exit)
  - **Journey Centrality**: Nodes arranged by betweenness centrality (hubs centered)

- **Physics Settings**:
  - Toggle physics simulation on/off
  - Gravitational constant (-15000) optimized for visualizing node relationships
  - Spring length (200px) and other force parameters tuned for clarity

- **Filtering Options**:
  - Show all nodes or filter to top 10%, 20%, or 50% friction nodes
  - Enable/disable physics for different visualization styles
  - Export current graph view as interactive HTML

## Flow Analysis Features
- Filter flows by minimum length
- Filter by minimum number of chokepoints (friction points)
- Focus on top percentage of flows by friction score
- Visualize complete user journeys with highlighted friction points

## Dataset Management
The dashboard automatically discovers datasets in the `outputs/` directory and provides a dropdown for selection. For each dataset, the dashboard displays:
- Dataset name
- Creation timestamp
- Number of users
- Number of events
- Number of sessions

If a dataset is not found or has missing files, appropriate error messages are displayed to guide the user. 