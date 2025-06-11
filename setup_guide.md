# TeloMesh Setup Guide

This guide provides instructions for setting up and running the TeloMesh application.

## Prerequisites

- Python 3.9+ 
- Pip package manager
- Git (for cloning the repository)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/telomesh.git
   cd telomesh
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage

### Processing a Dataset

1. Place your event data in CSV format in the `data/` directory:
   ```
   data/
   └── my_project/
       └── events.csv
   ```

2. Run the processing pipeline:
   ```bash
   python main.py --dataset my_project
   ```

3. View the results in the interactive dashboard:
   ```bash
   streamlit run ui/dashboard.py
   ```

4. In the dashboard, select your dataset from the dropdown menu to view the analysis results.

## Using the Dashboard

The TeloMesh dashboard provides several interactive visualizations:

1. **Friction Points**: View high-friction (page, event) pairs identified by WSJF scoring
2. **User Flow Network**: Interactive network visualization of user navigation paths
3. **Advanced Metrics**: Network analysis metrics including fractal dimension, power-law alpha, etc.
4. **Fragile Flows**: Multi-point friction areas where users encounter multiple chokepoints
5. **Decision Table**: Actionable recommendations based on network characteristics

## Input Data Format

Your events.csv file should have the following columns:
- `user_id`: Unique identifier for the user
- `page_url`: URL or identifier of the page visited
- `event_name`: Type of event (click, view, submit, etc.)
- `timestamp`: ISO format timestamp (supports both formats with and without milliseconds)

Example:
```
user_id,page_url,event_name,timestamp
user123,/home,page_view,2023-05-01T09:15:32.123Z
user123,/products,page_view,2023-05-01T09:16:45.678Z
user123,/products/item,click,2023-05-01T09:17:20.456Z
```

## Performance Considerations

For different dataset sizes:

- **Small datasets (<50K users)**: Standard processing is recommended
  ```bash
  python main.py --dataset small_project
  ```

- **Medium datasets (50K-200K users)**: Use fast mode for improved performance
  ```bash
  python main.py --dataset medium_project --fast
  ```

- **Large datasets (>200K users)**: Use fast mode and consider data sampling
  ```bash
  python main.py --dataset large_project --fast --sample 0.5
  ```

For performance benchmarks with different dataset sizes, see [analysis/performance_benchmarks.md](analysis/performance_benchmarks.md).

## Advanced Configuration

### Session Gap Definition

Customize the time gap (in minutes) that defines a new session:
```bash
python main.py --dataset my_project --session-gap 45
```

### Custom Output Directory

Specify a custom output directory:
```bash
python main.py --dataset my_project --output-dir ./custom_outputs
```

### Using Synthetic Data

Generate and analyze synthetic data for testing:
```bash
python main.py --synthetic --n_users 1000 --events_per_user 10
```

## Troubleshooting

### Zero-Inflation in Large Datasets

If you notice that most WSJF scores are zero in large datasets, this is expected. TeloMesh now uses a robust threshold calculation method that properly handles zero-inflated distributions.

### Memory Usage

For very large datasets, you might encounter memory issues. Try these solutions:
- Use the `--fast` flag to optimize memory usage
- Process a sample of your data with the `--sample` parameter
- Increase available memory or use a machine with more RAM

### Dashboard Loading Time

For large datasets, the dashboard might take longer to load. Patience is recommended, or you can optimize by:
- Using the `--fast` flag during data processing
- Pre-filtering your dataset to focus on specific user segments
- Using the command-line interface for initial analysis

## For Developers

### Running Tests

Run all tests:
```bash
pytest
```

Run specific test files:
```bash
pytest tests/test_event_chokepoints.py
```

### Code Structure

```
TeloMesh/
├── analysis/             # Analysis algorithms and models
├── data/                 # Input data folders by dataset
├── ingest/               # Data ingestion and transformation
├── outputs/              # Analysis outputs by dataset
├── tests/                # Test suite
├── ui/                   # Dashboard and visualization
├── utils/                # Utility functions
├── main.py               # Main application entry point
└── setup_guide.md        # This setup guide
```

For detailed module documentation, see the README.md in each directory.
