# TeloMesh Analytics Converter

This utility converts analytics data exports from common platforms (Mixpanel, Amplitude, Google Analytics 4) to the format required by TeloMesh for user journey analysis.

## Usage

### Converting Analytics Data

```bash
# Convert Mixpanel data
python analytics_converter.py --input mixpanel_export.csv --output telomesh_data.csv --format mixpanel

# Convert Amplitude data
python analytics_converter.py --input amplitude_export.csv --output telomesh_data.csv --format amplitude

# Convert Google Analytics 4 data
python analytics_converter.py --input ga4_export.csv --output telomesh_data.csv --format ga4

# Custom session timeout (default is 30 minutes)
python analytics_converter.py --input mixpanel_export.csv --output telomesh_data.csv --format mixpanel --session-gap 15
```

### Generating Sample Data

If you don't have real data to work with, you can generate sample data in any of the supported formats:

```bash
# Generate sample Mixpanel data
python analytics_converter.py --generate-sample --format mixpanel --output sample_mixpanel.csv

# Generate sample Amplitude data
python analytics_converter.py --generate-sample --format amplitude --output sample_amplitude.csv

# Generate sample GA4 data
python analytics_converter.py --generate-sample --format ga4 --output sample_ga4.csv

# Customize sample data volume
python analytics_converter.py --generate-sample --format mixpanel --users 20 --events 30 --output sample_mixpanel.csv
```

## Data Format

TeloMesh requires data in the following format:

| Column          | Type     | Description                              |
|-----------------|----------|------------------------------------------|
| `user_id`       | string   | Unique identifier for each user          |
| `timestamp`     | datetime | When the event occurred                  |
| `page`          | string   | URL path or screen name                  |
| `event`         | string   | User action that occurred                |
| `session_id`    | string   | Unique session identifier                |
| `event_properties` | JSON/string | Additional event details (optional) |

## Platform-Specific Mappings

### Mixpanel
- `distinct_id` → `user_id`
- `time` → `timestamp`
- `$current_url` or `screen` → `page`
- `event` remains as `event`
- `$insert_id` → `session_id` (or generated)
- `properties` → `event_properties`

### Amplitude
- `user_id` remains as `user_id`
- `event_time` → `timestamp`
- `page_url` or `page_title` → `page`
- `event_type` → `event`
- `session_id` remains as `session_id`
- `event_properties` remains as `event_properties`

### Google Analytics 4
- `user_pseudo_id` → `user_id`
- `event_timestamp` → `timestamp`
- `page_location` or `page_title` → `page`
- `event_name` → `event`
- A combination of `user_pseudo_id` and `ga_session_id` → `session_id`
- Event parameters → `event_properties`

## Session Identification

If your data doesn't include proper session IDs, the converter will automatically generate them based on:
- User ID
- Timestamp
- Session gap (default: 30 minutes) - Any gap larger than this between events for the same user will start a new session

## Examples

### Example 1: Converting Mixpanel Data

1. Export your data from Mixpanel as CSV
2. Run the converter:
   ```
   python analytics_converter.py --input mixpanel_export.csv --output telomesh_data.csv --format mixpanel
   ```
3. Use the converted data with TeloMesh:
   ```
   python main.py --input-file telomesh_data.csv
   ```

### Example 2: Working with Sample Data

1. Generate sample data:
   ```
   python analytics_converter.py --generate-sample --format amplitude --output sample_amplitude.csv
   ```
2. Convert the sample data:
   ```
   python analytics_converter.py --input sample_amplitude.csv --output telomesh_data.csv --format amplitude
   ```
3. Use with TeloMesh:
   ```
   python main.py --input-file telomesh_data.csv
   ``` 