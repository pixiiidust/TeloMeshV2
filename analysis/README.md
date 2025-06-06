# Analysis Components

This folder contains the core analytical components of TeloMesh that identify friction points in user journeys.

## Key Components
- Event chokepoint detection algorithms
- WSJF (Weighted Shortest Job First) friction scoring system
- Betweenness centrality calculations for page importance
- Exit rate analysis for identifying abandonment points
- Fragile flow detection for multi-point journey failures

The main file `event_chokepoints.py` processes user session data to generate ranked lists of friction points, high-friction flows, and graph visualization data. 