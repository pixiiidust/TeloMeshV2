# Test Suite

This folder contains the automated test suite for TeloMesh components, following a test-driven development approach.

## Test Categories
- Unit tests for individual component functions
- Integration tests for pipeline connections
- Validation tests for output data quality
- UI component tests for dashboard functionality

The test files follow a one-to-one mapping with implementation files:
- `test_synthetic_events.py` → Tests for data generation
- `test_parse_sessions.py` → Tests for session parsing
- `test_build_graph.py` → Tests for graph construction
- `test_flow_metrics.py` → Tests for flow validation
- `test_event_chokepoints.py` → Tests for friction analysis
- `test_dashboard_ui.py` → Tests for UI components 