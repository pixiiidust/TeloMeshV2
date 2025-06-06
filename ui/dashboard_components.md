# UI Components

This folder contains the TeloMesh User Flow Intelligence Dashboard, which visualizes user journey analysis through interactive components.

## Key Features
- Dataset selection dropdown for managing multiple projects
- Interactive Friction Analysis with sortable metrics
- User Flow Analysis with path visualization
- User Journey Graph with network visualization
- Export functionality for all dashboard sections
- Dark theme optimized for data visualization
- Tooltips for all metrics and analyses

The main file `dashboard.py` provides a comprehensive Streamlit interface for product managers to identify and analyze user experience friction points.

## Dashboard Tabs
- **Friction Analysis**: Table of friction points ranked by WSJF score
- **User Flow Analysis**: Summaries of user journeys with multiple friction points
- **User Journey Graph**: Interactive visualization of user flows with friction highlighted

## Dataset Management
The dashboard automatically discovers datasets in the `outputs/` directory and provides a dropdown for selection. For each dataset, the dashboard displays:
- Dataset name
- Creation timestamp
- Number of users
- Number of events
- Number of sessions

If a dataset is not found or has missing files, appropriate error messages are displayed to guide the user. 