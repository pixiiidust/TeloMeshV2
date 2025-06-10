import json
import os
import pandas as pd
from datetime import datetime, timedelta

def parse_sessions(input_csv="data/events.csv", output_csv="outputs/session_flows.csv", session_gap_minutes=30):
    """
    Parse event data from CSV format to a flattened session flows CSV format.
    
    Args:
        input_csv (str): Path to the input CSV file containing event data.
        output_csv (str): Path to the output CSV file.
        session_gap_minutes (int): Minutes of inactivity to consider as a new session boundary. Default is 30.
    
    Returns:
        pd.DataFrame: The parsed session data as a DataFrame.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Convert timestamps to datetime for sorting
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by user_id and timestamp
    df = df.sort_values(['user_id', 'timestamp'])
    
    # If session_id is not already in the data, split sessions based on session_gap_minutes idle gap
    if 'session_id' not in df.columns:
        # Group by user_id
        user_groups = df.groupby('user_id')
        
        # Initialize session_ids
        df['session_id'] = None
        
        for user_id, user_df in user_groups:
            # Reset the index for easier iteration
            user_df = user_df.reset_index(drop=False)
            
            # Initialize session counter for this user
            session_counter = 0
            
            # Assign the first event to the first session
            session_id = f"session_{user_id}_{session_counter:02d}"
            df.loc[user_df.iloc[0]['index'], 'session_id'] = session_id
            
            # Iterate through events to find session boundaries
            last_timestamp = user_df.iloc[0]['timestamp']
            
            for i in range(1, len(user_df)):
                current_timestamp = user_df.iloc[i]['timestamp']
                
                # Check if idle gap is > session_gap_minutes
                if (current_timestamp - last_timestamp) > timedelta(minutes=session_gap_minutes):
                    # Start a new session
                    session_counter += 1
                    session_id = f"session_{user_id}_{session_counter:02d}"
                
                # Assign session_id to this event
                df.loc[user_df.iloc[i]['index'], 'session_id'] = session_id
                
                # Update last timestamp
                last_timestamp = current_timestamp
    
    # Initialize an empty list to store the flattened data
    flattened_data = []
    
    # Process each session
    for session_id, session_df in df.groupby('session_id'):
        # Sort by timestamp to ensure chronological order
        session_df = session_df.sort_values('timestamp')
        
        # Process each event in the session
        for step_index, (_, event) in enumerate(session_df.iterrows()):
            # Extract event details
            user_id = event['user_id']
            page = event['page_url']
            event_name = event['event_name']
            timestamp = event['timestamp']
            
            # Create a row for this event
            row = {
                "user_id": user_id,
                "session_id": session_id,
                "step_index": step_index,
                "page": page,
                "event": event_name,
                "timestamp": timestamp
            }
            
            flattened_data.append(row)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(flattened_data)
    
    # Save to CSV
    result_df.to_csv(output_csv, index=False)
    
    return result_df

def main():
    """Main function to parse sessions data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse event data to session flows CSV format")
    parser.add_argument("--input", type=str, default="data/events.csv", 
                        help="Path to the input CSV file")
    parser.add_argument("--output", type=str, default="outputs/session_flows.csv", 
                        help="Path to the output CSV file")
    parser.add_argument("--session-gap", type=int, default=30,
                        help="Minutes of inactivity to consider as a new session boundary")
    
    args = parser.parse_args()
    
    print(f"Parsing events from {args.input} to {args.output} with {args.session_gap} minute session gap...")
    df = parse_sessions(args.input, args.output, args.session_gap)
    
    # Print some stats
    unique_users = df["user_id"].nunique()
    unique_sessions = df["session_id"].nunique()
    total_events = len(df)
    
    print(f"Parsed {total_events} events from {unique_sessions} sessions for {unique_users} users.")
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Parse event data to session flows.")
    parser.add_argument("--n_users", type=int, default=1000, help="Number of unique users")
    parser.add_argument("--events_per_user", type=int, default=6, help="Average events per user")
    parser.add_argument("--output_path", type=str, default="outputs/session_flows.csv", help="CSV output path")
    parser.add_argument("--fast", action="store_true", help="Skip slow scaling tests during development")
    parser.add_argument("--input_path", type=str, default="data/events.csv", help="CSV input path")
    parser.add_argument("--session-gap", type=int, default=30,
                        help="Minutes of inactivity to consider as a new session boundary")

    args = parser.parse_args()

    # Parse the sessions
    print(f"Parsing events from {args.input_path} to {args.output_path} with {args.session_gap} minute session gap...")
    df = parse_sessions(args.input_path, args.output_path, args.session_gap)
    
    # Print stats
    unique_users = df["user_id"].nunique()
    unique_sessions = df["session_id"].nunique()
    total_events = len(df)
    
    print(f"✅ Parsed {total_events} events from {unique_sessions} sessions for {unique_users} users → {args.output_path}")

    if not args.fast:
        # Run session quality check
        if unique_users > 0:
            sessions_per_user = unique_sessions / unique_users
            print(f"✅ Sessions per user: {sessions_per_user:.2f}")
            
            if sessions_per_user < 1:
                print(f"⚠️ Warning: Sessions per user is less than 1 ({sessions_per_user:.2f})")
            
            # Average steps per session
            avg_steps = total_events / unique_sessions
            print(f"✅ Average steps per session: {avg_steps:.2f}")
            
            if avg_steps < 3:
                print(f"⚠️ Warning: Average steps per session is less than 3 ({avg_steps:.2f})")
        else:
            print("⚠️ Warning: No users found in the data")
    else:
        print("⚡ FAST_MODE enabled: session quality checks bypassed.") 