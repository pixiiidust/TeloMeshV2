# Data Generation

This folder contains tools for generating synthetic user journey data for testing and demonstration purposes, as well as dataset-specific input data.

## Components
- Synthetic event generator for creating realistic user paths
- Session simulators with configurable friction points
- Parameter controls for adjusting user volumes and event frequencies
- Configurable abandonment patterns to test detection algorithms
- Dataset organization for managing multiple projects

## Enhanced Data Generation
The synthetic data generator now supports:
- Scale testing with up to thousands of users and events
- Control over the number of unique pages/nodes in the journey graph
- Generation of complex journey patterns for network analysis
- Controlled randomness for realistic event timing and navigation

## Page/Node Generation Control
The `--pages` parameter provides precise control over network structure:
- Default: 16 nodes (basic website structure)
- Lower values (8-12): Creates simpler, more linear journeys suitable for basic flow testing
- Higher values (20-50): Creates more complex networks with multiple navigation options
- Very high values (50+): Simulates large applications or e-commerce sites with extensive product catalogs

The page count significantly affects network analysis metrics:
- Higher page counts typically increase network complexity (higher fractal dimension)
- Lower page counts create more concentrated traffic patterns (higher centrality values)
- Page count directly impacts edge/node ratio and percolation characteristics

## Network Structure Testing
The generator creates data with specific network properties:
- Linear paths for simple journeys
- Tree structures for hierarchical navigation
- Complex hub-and-spoke patterns
- Clustered subgraphs for detecting related functionality
- Varied edge weights to simulate popularity differences
- Customizable number of unique nodes/pages with the `--pages` parameter

## Performance Testing
The generator can create datasets of various sizes:
- Small datasets (~100 users) for quick testing
- Medium datasets (~500-1000 users) for standard analysis
- Large datasets (1000+ users) for performance testing
- Support for the `--fast` flag for efficient processing

## Dataset Organization
Each dataset is stored in a dedicated subdirectory (e.g., `data/myproject/`) containing:
- `events.csv` - Raw event data for the dataset
- Additional dataset-specific input files as needed

## Usage Examples
```bash
# Generate default synthetic data (16 pages)
python main.py --dataset default_pages --users 100 --events 50

# Generate data with fewer pages for simple flows
python main.py --dataset simple_site --users 100 --events 50 --pages 8

# Generate data with more pages for complex site simulation
python main.py --dataset complex_site --users 200 --events 50 --pages 32

# Generate large e-commerce simulation
python main.py --dataset ecommerce --users 500 --events 50 --pages 64
```

## Generated Data Structure
The synthetic data generator creates events with the following format:
```csv
user_id,timestamp,page_url,event_name,session_id
user_001,2023-01-01T12:00:00,/home,Page View,session_user_001_00
user_001,2023-01-01T12:01:30,/products,Click,session_user_001_00
...
```

These events are then processed through the TeloMesh pipeline to create session flows, user journey graphs, and friction analysis. 