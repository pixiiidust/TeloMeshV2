#!/usr/bin/env python
"""
TeloMesh Analytics Converter Utility

This script converts analytics data exports from common platforms (Mixpanel, Amplitude, Google Analytics 4)
to the format required by TeloMesh for user journey analysis.

Usage:
    python analytics_converter.py --input input_file.csv --output output_file.csv --format [mixpanel|amplitude|ga4]
    
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
        return True
        
    except Exception as e:
        logger.error(f"Error converting Mixpanel data: {str(e)}")
        return False

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
        return True
        
    except Exception as e:
        logger.error(f"Error converting Amplitude data: {str(e)}")
        return False

def convert_ga4(input_file, output_file, session_gap_minutes=30):
    """
    Convert Google Analytics 4 export to TeloMesh format.
    
    GA4 mapping:
    - user_pseudo_id → user_id
    - event_timestamp → timestamp
    - page_location or page_title → page
    - event_name → event
    - A combination of user_pseudo_id and session_id → session_id
    
    Args:
        input_file (str): Path to input GA4 CSV/JSON
        output_file (str): Path to output CSV
        session_gap_minutes (int): Gap in minutes to define a new session
    """
    logger.info(f"Converting GA4 data from {input_file} to {output_file}")
    
    try:
        # Determine file type and read
        if input_file.endswith('.json'):
            with open(input_file, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
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
        
        # Convert timestamp
        # GA4 timestamps can be in microseconds since epoch
        if df['event_timestamp'].dtype == 'int64':
            # Check if it's in microseconds (standard GA4 export)
            if df['event_timestamp'].iloc[0] > 1600000000000000:  # Around year 2020 in microseconds
                output_df['timestamp'] = pd.to_datetime(df['event_timestamp'], unit='us')
            else:
                output_df['timestamp'] = pd.to_datetime(df['event_timestamp'], unit='s')
        else:
            # Try to parse as string
            output_df['timestamp'] = pd.to_datetime(df['event_timestamp'])
        
        # Determine page column
        if 'page_location' in df.columns:
            output_df['page'] = df['page_location']
        elif 'page_title' in df.columns:
            output_df['page'] = df['page_title']
        else:
            # Default to event if no page info is available
            output_df['page'] = df['event_name'].apply(lambda x: f"/{x.lower().replace(' ', '_')}")
            logger.warning("No page column found, using derived page from event name")
        
        # Map event
        output_df['event'] = df['event_name']
        
        # Session ID
        if 'ga_session_id' in df.columns:
            # Combine user and GA session for uniqueness
            output_df['session_id'] = df.apply(
                lambda row: f"{row['user_pseudo_id']}_{row['ga_session_id']}", 
                axis=1
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
        
        # Event properties - combine available parameters
        if 'event_params' in df.columns:
            output_df['event_properties'] = df['event_params'].apply(parse_json_column)
        else:
            # Collect all columns that might be parameters
            param_cols = [col for col in df.columns if col.startswith('event_param_')]
            
            if param_cols:
                # Combine parameter columns into properties
                output_df['event_properties'] = df[param_cols].apply(
                    lambda row: {col.replace('event_param_', ''): row[col] for col in param_cols if pd.notna(row[col])},
                    axis=1
                )
            else:
                output_df['event_properties'] = [{}] * len(output_df)
        
        # Save to output file
        output_df.to_csv(output_file, index=False)
        logger.info(f"Conversion completed: {len(output_df)} rows written to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting GA4 data: {str(e)}")
        return False

def generate_sample_data(platform, output_file, num_users=10, events_per_user=20):
    """
    Generate sample data in the format of a specific platform.
    
    Args:
        platform (str): Platform to generate data for ('mixpanel', 'amplitude', 'ga4')
        output_file (str): Path to output file
        num_users (int): Number of unique users to generate
        events_per_user (int): Average number of events per user
    """
    logger.info(f"Generating sample {platform} data with {num_users} users")
    
    # Define common parameters
    users = [f"user_{i}" for i in range(1, num_users + 1)]
    
    # Common pages and events
    pages = [
        "/home", 
        "/products", 
        "/product/1", 
        "/product/2", 
        "/cart", 
        "/checkout", 
        "/payment",
        "/confirmation",
        "/account",
        "/search"
    ]
    
    events = [
        "page_view",
        "click_button",
        "add_to_cart",
        "remove_from_cart",
        "begin_checkout",
        "purchase",
        "search",
        "login",
        "signup",
        "view_item"
    ]
    
    # Generate random data based on platform
    data = []
    now = datetime.now()
    
    for user_id in users:
        # Random session count (1-3 sessions per user)
        session_count = min(3, max(1, int(events_per_user / 5)))
        
        for session in range(session_count):
            # Session start time (random in the last 7 days)
            session_start = now - timedelta(
                days=session,  # Each session on a different day
                hours=session * 2,  # Space out sessions
                minutes=int(30 * session)  # Additional spacing
            )
            
            # Events in this session
            event_count = max(3, min(15, int(events_per_user / session_count)))
            
            # Session events
            current_time = session_start
            current_page_idx = 0  # Start at home
            
            for e in range(event_count):
                # Basic event
                event_data = {
                    "user_id": user_id,
                    "page": pages[current_page_idx],
                    "event": events[min(e % len(events), len(events) - 1)],
                    "timestamp": current_time
                }
                
                # Add platform-specific fields
                if platform == 'mixpanel':
                    event_data["distinct_id"] = user_id
                    event_data["time"] = current_time.isoformat()
                    event_data["$current_url"] = pages[current_page_idx]
                    event_data["$insert_id"] = f"session_{user_id}_{session}"
                    event_data["properties"] = json.dumps({
                        "source": "sample_data",
                        "session_number": session + 1
                    })
                    
                elif platform == 'amplitude':
                    event_data["user_id"] = user_id
                    event_data["event_time"] = current_time.isoformat()
                    event_data["event_type"] = events[min(e % len(events), len(events) - 1)]
                    event_data["page_url"] = pages[current_page_idx]
                    event_data["session_id"] = f"session_{user_id}_{session}"
                    event_data["event_properties"] = json.dumps({
                        "source": "sample_data",
                        "session_number": session + 1
                    })
                    
                elif platform == 'ga4':
                    event_data["user_pseudo_id"] = user_id
                    # GA4 uses microseconds since epoch
                    event_data["event_timestamp"] = int(current_time.timestamp() * 1000000)
                    event_data["event_name"] = events[min(e % len(events), len(events) - 1)]
                    event_data["page_location"] = pages[current_page_idx]
                    event_data["ga_session_id"] = f"session_{session}"
                    # GA4 has structured params
                    event_data["event_param_source"] = "sample_data"
                    event_data["event_param_session_number"] = session + 1
                
                data.append(event_data)
                
                # Advance time by 1-5 minutes
                current_time += timedelta(minutes=1 + e % 4)
                
                # Maybe advance to next page
                if e > 0 and e % 2 == 0 and current_page_idx < len(pages) - 1:
                    current_page_idx += 1
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    logger.info(f"Generated {len(df)} sample events for {platform} saved to {output_file}")
    return True

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Convert analytics data to TeloMesh format')
    
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--format', type=str, choices=['mixpanel', 'amplitude', 'ga4'], 
                        help='Input file format (analytics platform)')
    parser.add_argument('--session-gap', type=int, default=30,
                        help='Gap in minutes to define a new session (default: 30)')
    parser.add_argument('--generate-sample', action='store_true',
                        help='Generate sample data instead of converting')
    parser.add_argument('--users', type=int, default=10,
                        help='Number of users for sample data (default: 10)')
    parser.add_argument('--events', type=int, default=20,
                        help='Events per user for sample data (default: 20)')
    
    args = parser.parse_args()
    
    # Generate sample data if requested
    if args.generate_sample:
        if not args.format:
            logger.error("Please specify a format for sample data generation")
            return 1
            
        if not args.output:
            args.output = f"sample_{args.format}_data.csv"
            
        result = generate_sample_data(args.format, args.output, args.users, args.events)
        return 0 if result else 1
    
    # Otherwise, validate input parameters
    if not args.input or not args.output or not args.format:
        logger.error("Please provide input file, output file, and format")
        parser.print_help()
        return 1
    
    # Check that input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file does not exist: {args.input}")
        return 1
        
    # Convert based on format
    if args.format == 'mixpanel':
        result = convert_mixpanel(args.input, args.output, args.session_gap)
    elif args.format == 'amplitude':
        result = convert_amplitude(args.input, args.output, args.session_gap)
    elif args.format == 'ga4':
        result = convert_ga4(args.input, args.output, args.session_gap)
    else:
        logger.error(f"Unsupported format: {args.format}")
        return 1
        
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main()) 