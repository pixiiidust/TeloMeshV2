import os
import pandas as pd
import pytest
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.synthetic_event_generator import generate_synthetic_events

def test_events_csv_exists():
    """Test that the output events.csv file is created."""
    assert os.path.exists("data/events.csv")

def test_minimum_users():
    """Test that the minimum number of users (100) are created."""
    df = pd.read_csv("data/events.csv")
    user_ids = df["user_id"].unique()
    assert len(user_ids) >= 100, f"Expected at least 100 users, got {len(user_ids)}"

def test_minimum_rows_per_user():
    """Test that there are at least 5 events per user on average."""
    df = pd.read_csv("data/events.csv")
    num_users = df["user_id"].nunique()
    assert len(df) >= 5 * num_users, f"Expected at least {5*num_users} events, got {len(df)}"

def test_required_columns_exist():
    """Test that all required columns exist."""
    df = pd.read_csv("data/events.csv")
    required_columns = {"user_id", "timestamp", "page_url", "event_name", "session_id"}
    assert required_columns.issubset(set(df.columns))

def test_timestamps_parse():
    """Test that all timestamps are valid ISO8601 format."""
    df = pd.read_csv("data/events.csv")
    # This will raise an exception if any timestamp is not valid
    pd.to_datetime(df["timestamp"])

def test_minimum_user_count():
    """Test that there are at least 30 users."""
    df = pd.read_csv("data/events.csv")
    assert df["user_id"].nunique() >= 30, f"Expected at least 30 users, got {df['user_id'].nunique()}"

@pytest.mark.parametrize("n_users", [100, 1000, 10000])
def test_scaling_tiers(n_users):
    """Test that the generator scales correctly for different user counts."""
    # Generate events in memory
    df = generate_synthetic_events(n_users=n_users, events_per_user=6)
    # Verify the scaling constraint
    assert len(df) >= 5 * n_users, f"Expected at least {5*n_users} events, got {len(df)}" 