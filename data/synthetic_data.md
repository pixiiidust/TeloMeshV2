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
- Generation of complex journey patterns for network analysis
- Controlled clustering coefficient in user paths
- Configurable subgraph patterns
- Variable path lengths and branch factors

## Network Structure Testing
The generator creates data with specific network properties:
- Linear paths for simple journeys
- Tree structures for hierarchical navigation
- Complex hub-and-spoke patterns
- Clustered subgraphs for detecting related functionality
- Varied edge weights to simulate popularity differences

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
# Generate a small dataset
python main.py --dataset test_small --users 100 --events 10

# Generate a medium dataset
python main.py --dataset test_medium --users 500 --events 20

# Generate a large dataset with fast processing
python main.py --dataset test_large --users 1500 --events 10 --fast
``` 