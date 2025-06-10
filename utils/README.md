# TeloMesh Analytics Converter

This utility converts analytics data exports from common platforms (Mixpanel, Amplitude, Google Analytics 4) to the format required by TeloMesh for user journey analysis.

## Key Features

- Convert data from Mixpanel, Amplitude, and Google Analytics 4
- Generate sample data for testing and demonstration
- Automatically adapt column names for direct TeloMesh compatibility
- Intelligent session identification and generation
- Customizable sample data generation

## Quick Start

The most common usage pattern for the TeloMesh Analytics Converter is:

```bash
# Convert data and make it ready for TeloMesh in one step
python utils/analytics_converter.py --input your_analytics_export.csv --output data/events.csv --source [platform] --telomesh-format

# Generate sample data ready for TeloMesh in one step
python utils/analytics_converter.py --generate-sample --source [platform] --output data/events.csv --telomesh-format

# Where [platform] is one of: mixpanel, amplitude, ga4
```

With the `--telomesh-format` flag, the converter will automatically rename columns to match what TeloMesh expects (`page_url` and `event_name`), allowing you to use the output file directly with the main pipeline.

## Usage Options

### Converting Analytics Data

```bash
# Convert Mixpanel data
python utils/analytics_converter.py --input mixpanel_export.csv --output converted_data.csv --source mixpanel --telomesh-format

# Convert Amplitude data
python utils/analytics_converter.py --input amplitude_export.csv --output converted_data.csv --source amplitude --telomesh-format

# Convert Google Analytics 4 data
python utils/analytics_converter.py --input ga4_export.csv --output converted_data.csv --source ga4 --telomesh-format

# Convert with custom session gap (time between events to consider a new session)
python utils/analytics_converter.py --input mixpanel_export.csv --output data/events.csv --source mixpanel --session-gap 45 --telomesh-format
```

### Generating Sample Data

If you don't have real data to work with, you can generate sample data in any of the supported formats:

```bash
# Generate sample Mixpanel data
python utils/analytics_converter.py --generate-sample --source mixpanel --output sample_data.csv --telomesh-format

# Generate sample Amplitude data
python utils/analytics_converter.py --generate-sample --source amplitude --output sample_data.csv --telomesh-format

# Generate sample GA4 data
python utils/analytics_converter.py --generate-sample --source ga4 --output sample_data.csv --telomesh-format

# Customize sample data volume
python utils/analytics_converter.py --generate-sample --source mixpanel --users 20 --events-per-user 30 --output sample_data.csv --telomesh-format
```

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--input` | Path to the input analytics export file |
| `--output` | Path where the converted file will be saved |
| `--source` | Analytics platform format: mixpanel, amplitude, or ga4 (alias for --format) |
| `--telomesh-format` | Automatically adapt column names for TeloMesh compatibility |
| `--generate-sample` | Generate sample data instead of converting |
| `--users` | Number of users for sample data (default: 10) |
| `--events-per-user` | Events per user for sample data (default: 6) |
| `--session-gap` | Minutes of inactivity to consider a new session (default: 30) |

## Data Format

### TeloMesh-Compatible Format

When using the `--telomesh-format` flag (recommended), columns are correctly formatted for TeloMesh:

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Unique identifier for each user |
| `timestamp` | datetime | When the event occurred |
| `page_url` | string | URL path or screen name |
| `event_name` | string | User action that occurred |
| `session_id` | string | Unique session identifier |
| `event_properties` | JSON/string | Additional event details (optional) |

### Standard Converter Output Format

When converting without the `--telomesh-format` flag (not recommended for direct use with TeloMesh):

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Unique identifier for each user |
| `timestamp` | datetime | When the event occurred |
| `page` | string | URL path or screen name |
| `event` | string | User action that occurred |
| `session_id` | string | Unique session identifier |
| `event_properties` | JSON/string | Additional event details (optional) |

## Platform-Specific Mappings

### Mixpanel
- `distinct_id` → `user_id`
- `time` → `timestamp`
- `$current_url` or `screen` → `page` (becomes `page_url` with `--telomesh-format`)
- `event` remains as `event` (becomes `event_name` with `--telomesh-format`)
- `$insert_id` → `session_id` (or generated)
- `properties` → `event_properties`

### Amplitude
- `user_id` remains as `user_id`
- `event_time` → `timestamp`
- `page_url` or `page_title` → `page` (becomes `page_url` with `--telomesh-format`)
- `event_type` → `event` (becomes `event_name` with `--telomesh-format`)
- `session_id` remains as `session_id`
- `event_properties` remains as `event_properties`

### Google Analytics 4
- `user_pseudo_id` → `user_id`
- `event_timestamp` → `timestamp` (converted from microseconds since epoch)
- `page_location` → `page` (becomes `page_url` with `--telomesh-format`)
- `event_name` → `event` (becomes `event_name` with `--telomesh-format`)
- `ga_session_id` → `session_id` (prefixed with user_id)
- Event parameters (columns starting with `event_param_`) → `event_properties`

## Session Identification

If your data doesn't include proper session IDs, the converter will automatically generate them based on:
- User ID
- Timestamp
- Session gap (default: 30 minutes) - Any gap larger than this between events for the same user will start a new session

The session gap parameter is particularly important for accurate user journey analysis:
- Too short (< 15 min): May fragment single sessions into multiple ones, creating artificial breaks
- Too long (> 60 min): May combine separate visits into one session, blurring distinct user journeys
- Default (30 min): Industry standard that works well for most websites and applications

You can adjust this with the `--session-gap` parameter based on your specific application's user behavior patterns.

## Working with Custom Data

If your analytics data is from a platform not directly supported, you can:

1. Pre-process your data to match one of the supported formats
2. Convert using the most appropriate platform format
3. Post-process if needed

Alternatively, you can structure your CSV file to match the TeloMesh input format directly:
- Include columns: `user_id`, `timestamp`, `page_url`, `event_name`, `session_id`

## End-to-End Examples

### Example 1: Converting Mixpanel Data for TeloMesh

1. Export your data from Mixpanel as CSV or JSON
2. Run the converter with the TeloMesh format flag:
   ```bash
   python utils/analytics_converter.py --input mixpanel_export.csv --output data/events.csv --source mixpanel --telomesh-format
   ```
3. Run the TeloMesh pipeline:
   ```bash
   python main.py --dataset myproject
   ```

### Example 2: Working with Sample Data

If you don't have analytics data yet but want to try TeloMesh:

1. Generate sample data directly in TeloMesh format:
   ```bash
   python utils/analytics_converter.py --generate-sample --source amplitude --output data/events.csv --telomesh-format
   ```
2. Run the TeloMesh pipeline:
   ```bash
   python main.py --dataset myproject
   ```

### Example 3: Converting GA4 Data

1. Export data from Google Analytics 4
2. Convert and prepare for TeloMesh:
   ```bash
   python utils/analytics_converter.py --input ga4_export.csv --output data/events.csv --source ga4 --telomesh-format
   ```
3. Run the TeloMesh pipeline:
   ```bash
   python main.py --dataset myproject
   ``` 