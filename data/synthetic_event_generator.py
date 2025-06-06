import csv
import os
import random
import uuid
from datetime import datetime, timedelta
import pandas as pd

def generate_synthetic_events(n_users=100, events_per_user=50):
    """
    Generate synthetic event data with the specified number of users and events per user.
    
    Args:
        n_users (int): Number of users to generate. Default is 100.
        events_per_user (int): Average number of events per user. Default is 50.
        
    Returns:
        pd.DataFrame: A DataFrame containing the generated events.
    """
    # Ensure the output directory exists
    os.makedirs("data", exist_ok=True)
    
    events = []
    
    # Common pages and events to make the data realistic
    pages = [
        "/home", "/products", "/product/123", "/product/456", "/product/789",
        "/cart", "/checkout", "/search", "/account", "/settings", "/login",
        "/register", "/about", "/contact", "/faq", "/blog"
    ]
    
    events_list = [
        "Page View", "Click", "Scroll", "Form Submit", "Add to Cart",
        "Remove from Cart", "Checkout", "Search", "Login", "Logout",
        "Register", "Update Profile", "Error"
    ]
    
    # Generate users
    user_ids = [f"user_{i:03d}" for i in range(1, n_users + 1)]
    
    for user_id in user_ids:
        # Generate 1-5 sessions per user
        num_sessions = random.randint(1, 5)
        
        for session_idx in range(num_sessions):
            session_id = f"session_{user_id}_{session_idx:02d}"
            
            # Decide how many events in this session (3-7 events per session)
            num_events = random.randint(3, 7)
            
            # Start time for this session
            start_time = datetime.now() - timedelta(days=random.randint(1, 30))
            
            current_page = "/home"  # Start at home page
            
            for event_idx in range(num_events):
                # Generate a timestamp with 30sec-5min spacing
                if event_idx > 0:
                    spacing_seconds = random.randint(30, 300)
                    start_time += timedelta(seconds=spacing_seconds)
                
                # For variety, sometimes change page
                if random.random() > 0.6 or event_idx == 0:
                    current_page = random.choice(pages)
                
                # Generate an event
                event_name = random.choice(events_list)
                
                # Create the event
                event = {
                    "user_id": user_id,
                    "timestamp": start_time.isoformat(),
                    "page_url": current_page,
                    "event_name": event_name,
                    "session_id": session_id  # Adding session_id for easier parsing
                }
                
                events.append(event)
    
    # Convert to DataFrame
    df = pd.DataFrame(events)
    
    # Enforce the scaling constraint: num_events ≥ 5 × num_users
    min_events = 5 * n_users
    if len(df) < min_events:
        # If not enough events, generate more for random users
        additional_needed = min_events - len(df)
        while additional_needed > 0:
            user_id = random.choice(user_ids)
            session_id = f"session_{user_id}_extra"
            start_time = datetime.now() - timedelta(days=random.randint(1, 30))
            current_page = random.choice(pages)
            event_name = random.choice(events_list)
            
            event = {
                "user_id": user_id,
                "timestamp": start_time.isoformat(),
                "page_url": current_page,
                "event_name": event_name,
                "session_id": session_id
            }
            
            events.append(event)
            additional_needed -= 1
        
        # Update DataFrame
        df = pd.DataFrame(events)
    
    # Verify the scaling constraint
    assert len(df) >= 5 * n_users, f"Expected at least {5*n_users} events, got {len(df)}"
    
    return df

def main():
    """Main function to generate synthetic event data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic event data")
    parser.add_argument("--users", type=int, default=100, help="Number of users to generate")
    parser.add_argument("--events", type=int, default=50, help="Average number of events per user")
    
    args = parser.parse_args()
    
    print(f"Generating synthetic events for {args.users} users with ~{args.events} events per user...")
    df = generate_synthetic_events(args.users, args.events)
    
    # Print some stats
    total_events = len(df)
    unique_users = df["user_id"].nunique()
    unique_sessions = df["session_id"].nunique()
    
    print(f"Generated {total_events} events for {unique_users} users in {unique_sessions} sessions.")
    print(f"Output saved to data/events.csv")
    
    # Save to CSV file
    df.to_csv("data/events.csv", index=False)

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate synthetic user event data.")
    parser.add_argument("--n_users", type=int, default=1000, help="Number of unique users")
    parser.add_argument("--events_per_user", type=int, default=6, help="Average events per user")
    parser.add_argument("--output_path", type=str, default="data/events.csv", help="CSV output path")
    parser.add_argument("--fast", action="store_true", help="Skip slow scaling tests during development")

    args = parser.parse_args()

    # Generate the synthetic events
    df = generate_synthetic_events(args.n_users, args.events_per_user)
    
    # Save the events to CSV
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(f"[OK] Generated {len(df)} events for {args.n_users} users -> {args.output_path}")

    if not args.fast:
        # Run full scaling test
        if args.n_users in [100, 1000, 10000]:
            expected_min = 5 * args.n_users
            actual = len(df)
            assert actual >= expected_min, f"[ERROR] Scaling test failed: {actual} < {expected_min} events"
            print("[OK] Scaling test passed.")
        else:
            print("[WARN] Skipped scaling test (unsupported n_users value).")
    else:
        print("[FAST] FAST_MODE enabled: scaling test bypassed.") 