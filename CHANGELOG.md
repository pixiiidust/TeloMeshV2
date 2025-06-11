# Changelog

## [Version 2.3.0] - 2025-06-10

### Fixed
- WSJF threshold calculation for large datasets (100K+ users) using robust statistics (Median + MAD) instead of percentiles
- Edge case handling for zero-inflated distributions in chokepoint detection
- Adaptive threshold adjustment to ensure reasonable chokepoint identification (5-15%)
- Timestamp parsing in parse_sessions.py to handle both formats (with and without milliseconds)

### Added
- New robust WSJF threshold calculation method (`calculate_robust_wsjf_threshold`)
- Comprehensive test suite for WSJF threshold calculation (`test_wsjf_threshold.py`) with TDD approach
- Regression test for 100K user case 
- Improved debug output in event_chokepoints.py
- Performance benchmarks for different dataset sizes (analysis/performance_benchmarks.md)
- Enhanced documentation of threshold calculation in analysis.md
- Updated data pipeline documentation to include timestamp format handling

### Changed
- Reorganized documentation for better organization
  - Created analysis/analysis.md with comprehensive documentation
  - Moved performance benchmarks to analysis/performance_benchmarks.md
  - Moved setup_guide.md to root level from docs folder
  - Updated test_suite.md with information about new tests
- Directory structure updated to reflect new files and functions
- Improved error handling for timestamp parsing with informative messages

### Removed
- Deprecated pattern_matching.py and associated tests
- Redundant documentation files in docs/ folder (merged into respective module folders) 