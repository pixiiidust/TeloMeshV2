# Data Generation

This folder contains tools for generating synthetic user journey data for testing and demonstration purposes, as well as dataset-specific input data.

## Components
- Synthetic event generator for creating realistic user paths
- Session simulators with configurable friction points
- Parameter controls for adjusting user volumes and event frequencies
- Configurable abandonment patterns to test detection algorithms
- Dataset organization for managing multiple projects

The main file `synthetic_event_generator.py` creates test datasets that mimic real-world user behavior patterns with predefined friction points.

## Dataset Organization
Each dataset is stored in a dedicated subdirectory (e.g., `data/myproject/`) containing:
- `events.csv` - Raw event data for the dataset
- Additional dataset-specific input files as needed

Use the `--dataset` parameter with `main.py` to generate and manage datasets:
```bash
python main.py --dataset myproject --users 100 --events 50
``` 