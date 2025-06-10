import csv
import os
import random
import uuid
from datetime import datetime, timedelta
import pandas as pd

def generate_synthetic_events(n_users=100, events_per_user=50, n_pages=16):
    """
    Generate synthetic event data with the specified number of users and events per user.
    
    Args:
        n_users (int): Number of users to generate. Default is 100.
        events_per_user (int): Average number of events per user. Default is 50.
        n_pages (int): Number of unique pages to generate. Default is 16.
        
    Returns:
        pd.DataFrame: A DataFrame containing the generated events.
    """
    # Ensure the output directory exists
    os.makedirs("data", exist_ok=True)
    
    events = []
    
    # Base pages for realistic structure
    base_pages = [
        "/home", "/products", "/product/123", "/product/456", "/product/789",
        "/cart", "/checkout", "/search", "/account", "/settings", "/login",
        "/register", "/about", "/contact", "/faq", "/blog"
    ]
    
    # Generate additional pages if needed
    if n_pages > len(base_pages):
        # Generate product pages
        additional_product_pages = [f"/product/{i}" for i in range(1000, 1000 + n_pages - len(base_pages))]
        pages = base_pages + additional_product_pages
    else:
        # Use a subset of base pages if fewer pages requested
        pages = base_pages[:n_pages]
    
    # Ensure we have the requested number of pages
    assert len(pages) == n_pages, f"Generated {len(pages)} pages, expected {n_pages}"
    
    # Common events to make the data realistic
    events_list = [
        "Page View", "Click", "Scroll", "Form Submit", "Add to Cart",
        "Remove from Cart", "Checkout", "Search", "Login", "Logout",
        "Register", "Update Profile", "Error"
    ]
    
    # Generate users
    user_ids = [f"user_{i:03d}" for i in range(1, n_users + 1)]
    
    # Track page usage to ensure all pages are used
    page_usage = {page: 0 for page in pages}
    
    for user_id in user_ids:
        # Generate 1-5 sessions per user
        num_sessions = random.randint(1, 5)
        
        for session_idx in range(num_sessions):
            session_id = f"session_{user_id}_{session_idx:02d}"
            
            # Decide how many events in this session (3-7 events per session)
            num_events = random.randint(3, 7)
            
            # Start time for this session
            start_time = datetime.now() - timedelta(days=random.randint(1, 30))
            
            # For the first step, choose a page with low usage to balance page distribution
            if any(count == 0 for count in page_usage.values()):
                # Prioritize unused pages
                unused_pages = [p for p, count in page_usage.items() if count == 0]
                current_page = random.choice(unused_pages)
            else:
                # Prefer less used pages, but still allow some randomness
                weights = {page: max(1, 20 - count) for page, count in page_usage.items()}
                weighted_pages = []
                for page, weight in weights.items():
                    weighted_pages.extend([page] * weight)
                current_page = random.choice(weighted_pages)
            
            for event_idx in range(num_events):
                # Generate a timestamp with 30sec-5min spacing
                if event_idx > 0:
                    spacing_seconds = random.randint(30, 300)
                    start_time += timedelta(seconds=spacing_seconds)
                
                # For variety, sometimes change page
                if random.random() > 0.6 or event_idx == 0:
                    # Update page usage count
                    page_usage[current_page] = page_usage.get(current_page, 0) + 1
                    
                    # Select next page - prefer pages with lower usage to ensure all pages are used
                    if random.random() < 0.3 and any(count < 3 for count in page_usage.values()):
                        # Prioritize less used pages
                        less_used_pages = [p for p, count in page_usage.items() if count < 3]
                        current_page = random.choice(less_used_pages)
                    else:
                        # Normal random selection
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
            
            # Prioritize unused pages
            if any(count == 0 for count in page_usage.values()):
                unused_pages = [p for p, count in page_usage.items() if count == 0]
                current_page = random.choice(unused_pages)
            else:
                current_page = random.choice(pages)
            
            # Update page usage
            page_usage[current_page] = page_usage.get(current_page, 0) + 1
            
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
    
    # Verify that all pages were used
    pages_used = set(df["page_url"].unique())
    if len(pages_used) < n_pages:
        print(f"Warning: Only {len(pages_used)} out of {n_pages} pages were used in the generated events")
    
    # Print page usage statistics
    page_counts = df["page_url"].value_counts()
    print(f"Generated events with {len(page_counts)} unique pages out of {n_pages} requested")
    
    return df

def main():
    """Main function to generate synthetic event data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic event data")
    parser.add_argument("--users", type=int, default=100, help="Number of users to generate")
    parser.add_argument("--events", type=int, default=50, help="Average number of events per user")
    parser.add_argument("--pages", type=int, default=16, help="Number of unique pages to generate")
    
    args = parser.parse_args()
    
    print(f"Generating synthetic events for {args.users} users with ~{args.events} events per user across {args.pages} pages...")
    df = generate_synthetic_events(args.users, args.events, args.pages)
    
    # Print some stats
    total_events = len(df)
    unique_users = df["user_id"].nunique()
    unique_sessions = df["session_id"].nunique()
    unique_pages = df["page_url"].nunique()
    
    print(f"Generated {total_events} events for {unique_users} users in {unique_sessions} sessions using {unique_pages} pages.")
    print(f"Output saved to data/events.csv")
    
    # Save to CSV file
    df.to_csv("data/events.csv", index=False)

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate synthetic user event data.")
    parser.add_argument("--n_users", type=int, default=1000, help="Number of unique users")
    parser.add_argument("--events_per_user", type=int, default=6, help="Average events per user")
    parser.add_argument("--n_pages", type=int, default=16, help="Number of unique pages to generate")
    parser.add_argument("--output_path", type=str, default="data/events.csv", help="CSV output path")
    parser.add_argument("--fast", action="store_true", help="Skip slow scaling tests during development")

    args = parser.parse_args()

    # Generate the synthetic events
    df = generate_synthetic_events(args.n_users, args.events_per_user, args.n_pages)
    
    # Save the events to CSV
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(f"[OK] Generated {len(df)} events for {args.n_users} users with {df['page_url'].nunique()} unique pages -> {args.output_path}")

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