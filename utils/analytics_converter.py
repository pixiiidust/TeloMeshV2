#!/usr/bin/env python
"""
TeloMesh Analytics Converter Utility

This script converts analytics data exports from common platforms (Mixpanel, Amplitude, Google Analytics 4)
to the format required by TeloMesh for user journey analysis.

Usage:
    python analytics_converter.py --input input_file.csv --output output_file.csv --format [mixpanel|amplitude|ga4] [--telomesh-format]
    
    # Generate sample data:
    python analytics_converter.py --generate-sample --format [mixpanel|amplitude|ga4] --output sample_data.csv
    
Requirements:
    - pandas
    - dateutil
"""

import os
import sys
import argparse
import json
from datetime import datetime, timedelta
import pandas as pd
from dateutil import parser
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TeloMesh Analytics Converter")

def generate_session_id(user_id, timestamp, gap_minutes=30):
    """
    Generate a session ID based on user ID and timestamp.
    Used when no session ID is provided or for platforms that don't have proper session IDs.
    
    Args:
        user_id (str): User identifier
        timestamp (datetime): Event timestamp
        gap_minutes (int): Session timeout in minutes
        
    Returns:
        str: Generated session ID
    """
    # Simple implementation - production code might use more robust methods
    return f"session_{user_id}_{timestamp.strftime('%Y%m%d_%H%M')}"

def parse_json_column(value):
    """
    Parse a JSON string into a Python dictionary.
    
    Args:
        value: The JSON string or already parsed object
        
    Returns:
        dict: Parsed dictionary or empty dict if parsing fails
    """
    if pd.isna(value):
        return {}
        
    if isinstance(value, dict):
        return value
        
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}

def convert_mixpanel(input_file, output_file, session_gap_minutes=30):
    """
    Convert Mixpanel CSV export to TeloMesh format.
    
    Mixpanel mapping:
    - distinct_id → user_id
    - time → timestamp
    - $current_url or screen → page
    - event remains as event
    - $insert_id → session_id (or generate one)
    - properties → event_properties
    
    Args:
        input_file (str): Path to input Mixpanel CSV
        output_file (str): Path to output CSV
        session_gap_minutes (int): Gap in minutes to define a new session
    """
    logger.info(f"Converting Mixpanel data from {input_file} to {output_file}")
    
    try:
        # Read input file
        df = pd.read_csv(input_file)
        
        # Check required columns
        required_cols = ['distinct_id', 'time', 'event']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {', '.join(missing_cols)}")
            return False
            
        # Create output dataframe with TeloMesh format
        output_df = pd.DataFrame()
        
        # Map columns
        output_df['user_id'] = df['distinct_id']
        
        # Convert time to timestamp
        output_df['timestamp'] = pd.to_datetime(df['time'])
        
        # Determine page column
        if '$current_url' in df.columns:
            output_df['page'] = df['$current_url']
        elif 'screen' in df.columns:
            output_df['page'] = df['screen']
        else:
            # Default to event if no page info is available
            output_df['page'] = df['event'].apply(lambda x: f"/{x.lower().replace(' ', '_')}")
            logger.warning("No page column found, using derived page from event name")
        
        # Map event
        output_df['event'] = df['event']
        
        # Session ID
        if '$insert_id' in df.columns:
            output_df['session_id'] = df['$insert_id']
        else:
            # Generate session IDs
            logger.info("No session ID column found, generating session IDs")
            
            # Sort by user and time
            output_df = output_df.sort_values(['user_id', 'timestamp'])
            
            # Initialize session ID column
            output_df['session_id'] = None
            
            # Track current session info
            current_user = None
            current_session = None
            last_timestamp = None
            
            # Generate session IDs
            for i, row in output_df.iterrows():
                if (current_user != row['user_id'] or
                    last_timestamp is None or
                    (row['timestamp'] - last_timestamp).total_seconds() / 60 > session_gap_minutes):
                    # New session
                    current_user = row['user_id']
                    current_session = generate_session_id(row['user_id'], row['timestamp'])
                
                output_df.at[i, 'session_id'] = current_session
                last_timestamp = row['timestamp']
        
        # Event properties
        if 'properties' in df.columns:
            output_df['event_properties'] = df['properties'].apply(parse_json_column)
        else:
            output_df['event_properties'] = [{}] * len(output_df)
        
        # Save to output file
        output_df.to_csv(output_file, index=False)
        logger.info(f"Conversion completed: {len(output_df)} rows written to {output_file}")
        return output_df
        
    except Exception as e:
        logger.error(f"Error converting Mixpanel data: {str(e)}")
        return None

def convert_amplitude(input_file, output_file, session_gap_minutes=30):
    """
    Convert Amplitude CSV/JSON export to TeloMesh format.
    
    Amplitude mapping:
    - user_id remains as user_id
    - event_time → timestamp
    - page_url or page_title → page
    - event_type → event
    - session_id remains as session_id
    - event_properties remains as event_properties
    
    Args:
        input_file (str): Path to input Amplitude CSV/JSON
        output_file (str): Path to output CSV
        session_gap_minutes (int): Gap in minutes to define a new session
    """
    logger.info(f"Converting Amplitude data from {input_file} to {output_file}")
    
    try:
        # Determine file type and read
        if input_file.endswith('.json'):
            with open(input_file, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            df = pd.read_csv(input_file)
        
        # Check required columns
        required_cols = ['user_id', 'event_time', 'event_type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {', '.join(missing_cols)}")
            return False
            
        # Create output dataframe with TeloMesh format
        output_df = pd.DataFrame()
        
        # Map columns
        output_df['user_id'] = df['user_id']
        
        # Convert time to timestamp
        output_df['timestamp'] = pd.to_datetime(df['event_time'])
        
        # Determine page column
        if 'page_url' in df.columns:
            output_df['page'] = df['page_url']
        elif 'page_title' in df.columns:
            output_df['page'] = df['page_title']
        else:
            # Default to event if no page info is available
            output_df['page'] = df['event_type'].apply(lambda x: f"/{x.lower().replace(' ', '_')}")
            logger.warning("No page column found, using derived page from event name")
        
        # Map event
        output_df['event'] = df['event_type']
        
        # Session ID
        if 'session_id' in df.columns:
            output_df['session_id'] = df['session_id']
        else:
            # Generate session IDs
            logger.info("No session ID column found, generating session IDs")
            
            # Sort by user and time
            output_df = output_df.sort_values(['user_id', 'timestamp'])
            
            # Initialize session ID column
            output_df['session_id'] = None
            
            # Track current session info
            current_user = None
            current_session = None
            last_timestamp = None
            
            # Generate session IDs
            for i, row in output_df.iterrows():
                if (current_user != row['user_id'] or
                    last_timestamp is None or
                    (row['timestamp'] - last_timestamp).total_seconds() / 60 > session_gap_minutes):
                    # New session
                    current_user = row['user_id']
                    current_session = generate_session_id(row['user_id'], row['timestamp'])
                
                output_df.at[i, 'session_id'] = current_session
                last_timestamp = row['timestamp']
        
        # Event properties
        if 'event_properties' in df.columns:
            output_df['event_properties'] = df['event_properties'].apply(parse_json_column)
        else:
            output_df['event_properties'] = [{}] * len(output_df)
        
        # Save to output file
        output_df.to_csv(output_file, index=False)
        logger.info(f"Conversion completed: {len(output_df)} rows written to {output_file}")
        return output_df
        
    except Exception as e:
        logger.error(f"Error converting Amplitude data: {str(e)}")
        return None

def convert_ga4(input_file, output_file, session_gap_minutes=30):
    """
    Convert Google Analytics 4 (GA4) CSV export to TeloMesh format.
    
    GA4 mapping:
    - user_pseudo_id → user_id
    - event_timestamp → timestamp (converted from microseconds since epoch)
    - page_location → page
    - event_name → event
    - ga_session_id → session_id (prefixed with user_id)
    - event_params → event_properties
    
    Args:
        input_file (str): Path to input GA4 CSV
        output_file (str): Path to output CSV
        session_gap_minutes (int): Gap in minutes to define a new session
    """
    logger.info(f"Converting GA4 data from {input_file} to {output_file}")
    
    try:
        # Read input file
        df = pd.read_csv(input_file)
        
        # Check required columns
        required_cols = ['user_pseudo_id', 'event_timestamp', 'event_name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {', '.join(missing_cols)}")
            return False
            
        # Create output dataframe with TeloMesh format
        output_df = pd.DataFrame()
        
        # Map columns
        output_df['user_id'] = df['user_pseudo_id']
        
        # Convert time to timestamp - GA4 uses microseconds since epoch
        if df['event_timestamp'].dtype == 'int64':
            # Convert from microseconds since epoch
            output_df['timestamp'] = pd.to_datetime(df['event_timestamp'], unit='us')
        else:
            # Try to parse as string
            output_df['timestamp'] = pd.to_datetime(df['event_timestamp'])
        
        # Determine page column
        if 'page_location' in df.columns:
            output_df['page'] = df['page_location']
        else:
            # Default to event if no page info is available
            output_df['page'] = df['event_name'].apply(lambda x: f"/{x.lower().replace(' ', '_')}")
            logger.warning("No page column found, using derived page from event name")
        
        # Map event
        output_df['event'] = df['event_name']
        
        # Session ID
        if 'ga_session_id' in df.columns:
            # Prefix with user_id to ensure uniqueness
            output_df['session_id'] = df.apply(
                lambda row: f"{row['user_pseudo_id']}_{row['ga_session_id']}", axis=1
            )
        else:
            # Generate session IDs
            logger.info("No session ID column found, generating session IDs")
            
            # Sort by user and time
            output_df = output_df.sort_values(['user_id', 'timestamp'])
            
            # Initialize session ID column
            output_df['session_id'] = None
            
            # Track current session info
            current_user = None
            current_session = None
            last_timestamp = None
            
            # Generate session IDs
            for i, row in output_df.iterrows():
                if (current_user != row['user_id'] or
                    last_timestamp is None or
                    (row['timestamp'] - last_timestamp).total_seconds() / 60 > session_gap_minutes):
                    # New session
                    current_user = row['user_id']
                    current_session = generate_session_id(row['user_id'], row['timestamp'])
                
                output_df.at[i, 'session_id'] = current_session
                last_timestamp = row['timestamp']
        
        # Event properties - combine all event_param_* columns
        event_param_cols = [col for col in df.columns if col.startswith('event_param_')]
        if event_param_cols:
            # Gather event parameters into a single dictionary
            event_properties = []
            for _, row in df.iterrows():
                props = {}
                for col in event_param_cols:
                    param_name = col.replace('event_param_', '')
                    if not pd.isna(row[col]):
                        props[param_name] = row[col]
                event_properties.append(props)
            
            output_df['event_properties'] = event_properties
        else:
            output_df['event_properties'] = [{}] * len(output_df)
        
        # Save to output file
        output_df.to_csv(output_file, index=False)
        logger.info(f"Conversion completed: {len(output_df)} rows written to {output_file}")
        return output_df
        
    except Exception as e:
        logger.error(f"Error converting GA4 data: {str(e)}")
        return None

def generate_sample_data(platform, output_file, num_users=10, events_per_user=20):
    """
    Generate sample data for the specified analytics platform.
    
    Args:
        platform (str): One of 'mixpanel', 'amplitude', 'ga4'
        output_file (str): Path to output file
        num_users (int): Number of users to generate
        events_per_user (int): Average number of events per user
    
    Returns:
        pd.DataFrame: Generated sample data
    """
    logger.info(f"Generating sample {platform} data with {num_users} users")
    
    # Define event types
    event_types = [
        'page_view', 
        'click_button', 
        'add_to_cart', 
        'begin_checkout',
        'purchase',
        'user_sign_in'
    ]
    
    # Define pages
    pages = [
        '/home',
        '/products',
        '/product/1',
        '/cart',
        '/checkout',
        '/account'
    ]
    
    # Generate synthetic data
    rows = []
    
    # Base timestamp
    base_timestamp = datetime.now()
    
    for user_id in range(1, num_users + 1):
        # Each user gets 3 sessions
        for session_num in range(3):
            # Session timestamp starts 1 day apart per user, sessions 2 hours apart
            session_timestamp = base_timestamp + timedelta(
                days=user_id - 1,
                hours=session_num * 2
            )
            
            # Generate 6 events per session
            for event_idx in range(6):
                # Each event is 1-3 minutes apart
                event_timestamp = session_timestamp + timedelta(
                    minutes=event_idx * 2 + 1
                )
                
                # Select page and event type
                page = pages[min(event_idx, len(pages) - 1)]
                event = event_types[min(event_idx, len(event_types) - 1)]
                
                # Create event data based on platform
                if platform == 'mixpanel':
                    row = {
                        'user_id': f"user_{user_id}",
                        'distinct_id': f"user_{user_id}",
                        'time': event_timestamp,
                        '$current_url': page,
                        'event': event,
                        '$insert_id': f"session_user_{user_id}_{session_num}",
                        'properties': json.dumps({
                            'source': 'sample_data',
                            'session_number': session_num + 1
                        })
                    }
                elif platform == 'amplitude':
                    row = {
                        'user_id': f"user_{user_id}",
                        'event_time': event_timestamp,
                        'page_url': page,
                        'event_type': event,
                        'session_id': f"session_user_{user_id}_{session_num}",
                        'event_properties': json.dumps({
                            'source': 'sample_data',
                            'session_number': session_num + 1
                        })
                    }
                elif platform == 'ga4':
                    # For GA4, timestamps are in microseconds since epoch
                    timestamp_us = int(event_timestamp.timestamp() * 1000000)
                    
                    row = {
                        'user_id': f"user_{user_id}",
                        'user_pseudo_id': f"user_{user_id}",
                        'event_timestamp': timestamp_us,
                        'timestamp': event_timestamp,
                        'page_location': page,
                        'event_name': event,
                        'ga_session_id': f"session_{session_num}",
                        'event_param_source': 'sample_data',
                        'event_param_session_number': session_num + 1
                    }
                else:
                    raise ValueError(f"Unknown platform: {platform}")
                
                rows.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Ensure output directory exists if the output file has a directory path
    dir_name = os.path.dirname(output_file)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Generated {len(df)} sample events for {platform} saved to {output_file}")
    
    return df

def adapt_for_telomesh(df, output_file):
    """
    Adapt a DataFrame to TeloMesh expected format.
    
    Args:
        df (pd.DataFrame): DataFrame in standardized format
        output_file (str): Output file path
        
    Returns:
        pd.DataFrame: Adapted DataFrame
    """
    try:
        # Check if it has the required columns
        required_cols = ['user_id', 'timestamp', 'page', 'event', 'session_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns for TeloMesh format: {', '.join(missing_cols)}")
            return None
        
        # Make a copy of the DataFrame to avoid modifying the original
        adapted_df = df.copy()
        
        # Rename columns to match what TeloMesh expects
        column_mapping = {
            'page': 'page_url',
            'event': 'event_name'
        }
        
        adapted_df = adapted_df.rename(columns=column_mapping)
        
        # Ensure the output directory exists if the output file has a directory path
        dir_name = os.path.dirname(output_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        # Save to output file
        adapted_df.to_csv(output_file, index=False)
        
        logger.info(f"Adapted data for TeloMesh and saved to {output_file}")
        logger.info(f"Converted {len(adapted_df)} events with columns: {list(adapted_df.columns)}")
        
        return adapted_df
    
    except Exception as e:
        logger.error(f"Error adapting data for TeloMesh: {str(e)}")
        return None

def main():
    """Main function for the script."""
    parser = argparse.ArgumentParser(
        description='Convert analytics data to TeloMesh format',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument(
        '--format', 
        type=str, 
        choices=['mixpanel', 'amplitude', 'ga4'],
        help='Input data format'
    )
    parser.add_argument(
        '--telomesh-format', 
        action='store_true',
        help='Output in format directly usable by TeloMesh (renames columns to page_url and event_name)'
    )
    parser.add_argument(
        '--generate-sample', 
        action='store_true',
        help='Generate sample data instead of converting'
    )
    parser.add_argument(
        '--users', 
        type=int, 
        default=10,
        help='Number of users for sample data (default: 10)'
    )
    parser.add_argument(
        '--events-per-user', 
        type=int, 
        default=6,
        help='Events per user for sample data (default: 6)'
    )
    
    args = parser.parse_args()
    
    # Check for required arguments
    if args.generate_sample:
        if not args.format:
            logger.error("--format is required with --generate-sample")
            return 1
        
        if not args.output:
            logger.error("--output is required with --generate-sample")
            return 1
        
        # Generate sample data to a temporary file
        temp_output = args.output + ".temp"
        result = generate_sample_data(
            args.format, 
            temp_output, 
            args.users, 
            args.events_per_user
        )
        
        if result is None:
            return 1
        
        # Standardize column names for sample data
        if args.format == 'mixpanel':
            # Rename columns to standardized format
            result.rename(columns={
                'distinct_id': 'user_id',
                'time': 'timestamp', 
                '$current_url': 'page',
                'event': 'event'
            }, inplace=True)
        elif args.format == 'amplitude':
            # Rename columns to standardized format
            result.rename(columns={
                'event_time': 'timestamp',
                'page_url': 'page',
                'event_type': 'event'
            }, inplace=True)
        elif args.format == 'ga4':
            # Rename columns to standardized format
            result.rename(columns={
                'user_pseudo_id': 'user_id',
                'page_location': 'page',
                'event_name': 'event'
            }, inplace=True)
            
            # Convert timestamp from microseconds if needed
            if 'timestamp' not in result.columns and 'event_timestamp' in result.columns:
                if result['event_timestamp'].dtype == 'int64':
                    result['timestamp'] = pd.to_datetime(result['event_timestamp'], unit='us')
        
        # For telomesh format, adapt the result
        if args.telomesh_format:
            # Use adapt_for_telomesh to create final output
            adapt_for_telomesh(result, args.output)
        else:
            # Just copy the temp file to the final output
            result.to_csv(args.output, index=False)
            logger.info(f"Sample data saved to {args.output}")
        
        # Remove temp file
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except:
                pass
        
        return 0
    else:
        if not args.input or not args.output or not args.format:
            logger.error("--input, --output, and --format are required for conversion")
            return 1
        
        # Check if input file exists
        if not os.path.exists(args.input):
            logger.error(f"Input file does not exist: {args.input}")
            return 1
        
        # Convert to a temporary file
        temp_output = args.output + ".temp"
        
        # Convert based on format
        if args.format == 'mixpanel':
            result = convert_mixpanel(args.input, temp_output)
        elif args.format == 'amplitude':
            result = convert_amplitude(args.input, temp_output)
        elif args.format == 'ga4':
            result = convert_ga4(args.input, temp_output)
        else:
            logger.error(f"Unknown format: {args.format}")
            return 1
        
        if result is None:
            return 1
            
        # For telomesh format, adapt the result
        if args.telomesh_format:
            # Use adapt_for_telomesh to create final output
            adapt_for_telomesh(result, args.output)
        else:
            # Just copy the temp file to the final output
            result.to_csv(args.output, index=False)
            logger.info(f"Converted data saved to {args.output}")
        
        # Remove temp file
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except:
                pass
        
        return 0

if __name__ == "__main__":
    sys.exit(main()) 