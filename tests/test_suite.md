# TeloMesh Test Suite

This folder contains all tests for the TeloMesh application.

## Test Categories

- **Feature Tests**: Validate core functionality components
- **Integration Tests**: Test interactions between components
- **UI Tests**: Verify dashboard functionality
- **Performance Tests**: Ensure scalability with different dataset sizes
- **Regression Tests**: Protect against specific previously-fixed issues

## Key Test Files

- **test_event_chokepoints.py**: Tests for chokepoint detection algorithms
- **test_build_graph.py**: Tests for graph construction from session data
- **test_parse_sessions.py**: Tests for event parsing and session extraction
- **test_dashboard_ui.py**: Tests for UI components and visualizations
- **test_node_level_metrics.py**: Tests for node-level network metrics
- **test_advanced_metrics_tab.py**: Tests for advanced network analytics
- **test_dataset_organization.py**: Tests for multi-dataset support
- **test_flow_metrics.py**: Tests for session flow metrics calculations
- **test_wsjf_threshold.py**: Tests for robust WSJF threshold calculation

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_event_chokepoints.py

# Run tests with verbosity
pytest -v

# Run tests with code coverage reporting
pytest --cov=.
```

## Test Focus Areas

1. **Data Ingestion**: Timestamp parsing, session identification, flow construction
2. **Graph Analysis**: Correct graph construction, metrics calculation
3. **Friction Detection**: Accurate chokepoint identification, WSJF scoring
4. **Advanced Metrics**: Fractal dimension, betweenness, power-law scaling
5. **Dashboard UI**: Proper visualization, correct metrics display
6. **Scalability**: Performance with varying dataset sizes
7. **Robustness**: Proper handling of edge cases and zero-inflated distributions 