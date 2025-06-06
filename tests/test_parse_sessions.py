import os
import pandas as pd
import pytest

def test_session_flows_csv_exists():
    """Test that the output session_flows.csv file is created."""
    assert os.path.exists("outputs/session_flows.csv")

def test_session_flows_columns():
    """Test that the required columns exist in the output CSV."""
    df = pd.read_csv("outputs/session_flows.csv")
    required_columns = {"user_id", "session_id", "step_index", "page", "event"}
    assert required_columns.issubset(set(df.columns))

def test_minimum_sessions():
    """Test that there are a minimum number of sessions."""
    df = pd.read_csv("outputs/session_flows.csv")
    assert df["session_id"].nunique() >= 30, f"Expected at least 30 sessions, got {df['session_id'].nunique()}"

def test_chronological_steps():
    """Test that steps within a session are in chronological order."""
    df = pd.read_csv("outputs/session_flows.csv")
    # Group by session_id and check that step_index is monotonically increasing
    for session_id, group in df.groupby("session_id"):
        assert group["step_index"].is_monotonic_increasing, f"Steps not in order for session {session_id}"

def test_steps_match_events():
    """Test that the total number of steps matches the total number of events in the input."""
    # Load the original events.csv to count events
    events_df = pd.read_csv("data/events.csv")
    total_events = len(events_df)
    
    # Now count rows in the parsed CSV
    flows_df = pd.read_csv("outputs/session_flows.csv")
    assert len(flows_df) == total_events, f"Expected {total_events} rows, got {len(flows_df)}" 