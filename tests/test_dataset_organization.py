import os
import shutil
import json
import time
import pytest
from pathlib import Path
import subprocess
import pandas as pd

# Test constants
TEST_DATASET_NAME = "test_dataset"
TEST_USERS = 10
TEST_EVENTS = 5

@pytest.fixture(scope="function")
def cleanup_test_dataset():
    """Clean up test dataset directories before and after tests."""
    # Clean up before test
    data_dir = Path(f"data/{TEST_DATASET_NAME}")
    output_dir = Path(f"outputs/{TEST_DATASET_NAME}")
    
    if data_dir.exists():
        shutil.rmtree(data_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    yield
    
    # Clean up after test
    if data_dir.exists():
        shutil.rmtree(data_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

def test_synthetic_data_flow(cleanup_test_dataset):
    """Test Flow 1: Verify synthetic data generation creates proper dataset structure."""
    # Run main.py with test dataset
    subprocess.run([
        "python", "main.py", 
        "--dataset", TEST_DATASET_NAME,
        "--users", str(TEST_USERS),
        "--events", str(TEST_EVENTS)
    ], check=True)
    
    # Check that directories exist
    assert Path(f"data/{TEST_DATASET_NAME}").exists(), "Dataset directory in data/ was not created"
    assert Path(f"outputs/{TEST_DATASET_NAME}").exists(), "Dataset directory in outputs/ was not created"
    
    # Check expected output files
    expected_files = [
        f"data/{TEST_DATASET_NAME}/events.csv",
        f"outputs/{TEST_DATASET_NAME}/session_flows.csv",
        f"outputs/{TEST_DATASET_NAME}/user_graph.gpickle",
        f"outputs/{TEST_DATASET_NAME}/event_chokepoints.csv",
        f"outputs/{TEST_DATASET_NAME}/high_friction_flows.csv",
        f"outputs/{TEST_DATASET_NAME}/friction_node_map.json",
        f"outputs/{TEST_DATASET_NAME}/dataset_info.json"
    ]
    
    for file_path in expected_files:
        assert Path(file_path).exists(), f"Expected file {file_path} was not created"

def test_existing_data_flow(cleanup_test_dataset):
    """Test Flow 2: Verify existing data processing creates proper dataset structure."""
    # First, generate some sample data to use as existing data
    sample_data_path = "data/sample_input.csv"
    subprocess.run([
        "python", "data/synthetic_event_generator.py",
        "--n_users", str(TEST_USERS),
        "--events_per_user", str(TEST_EVENTS),
        "--output_path", sample_data_path
    ], check=True)
    
    # Run main.py with input file
    subprocess.run([
        "python", "main.py", 
        "--dataset", TEST_DATASET_NAME,
        "--input", sample_data_path
    ], check=True)
    
    # Check that directories exist
    assert Path(f"data/{TEST_DATASET_NAME}").exists(), "Dataset directory in data/ was not created"
    assert Path(f"outputs/{TEST_DATASET_NAME}").exists(), "Dataset directory in outputs/ was not created"
    
    # Check expected output files
    expected_files = [
        f"data/{TEST_DATASET_NAME}/events.csv",
        f"outputs/{TEST_DATASET_NAME}/session_flows.csv",
        f"outputs/{TEST_DATASET_NAME}/user_graph.gpickle",
        f"outputs/{TEST_DATASET_NAME}/event_chokepoints.csv",
        f"outputs/{TEST_DATASET_NAME}/high_friction_flows.csv",
        f"outputs/{TEST_DATASET_NAME}/friction_node_map.json",
        f"outputs/{TEST_DATASET_NAME}/dataset_info.json"
    ]
    
    for file_path in expected_files:
        assert Path(file_path).exists(), f"Expected file {file_path} was not created"
    
    # Clean up sample input file
    if Path(sample_data_path).exists():
        os.remove(sample_data_path)

def test_dataset_info_json_metadata(cleanup_test_dataset):
    """Test that dataset_info.json is created with correct metadata."""
    # Run main.py with test dataset
    subprocess.run([
        "python", "main.py", 
        "--dataset", TEST_DATASET_NAME,
        "--users", str(TEST_USERS),
        "--events", str(TEST_EVENTS)
    ], check=True)
    
    # Check that dataset_info.json exists
    info_path = Path(f"outputs/{TEST_DATASET_NAME}/dataset_info.json")
    assert info_path.exists(), "dataset_info.json was not created"
    
    # Check metadata content
    with open(info_path, 'r') as f:
        metadata = json.load(f)
    
    assert "dataset_name" in metadata, "dataset_name not in metadata"
    assert "creation_timestamp" in metadata, "creation_timestamp not in metadata"
    assert "num_users" in metadata, "num_users not in metadata"
    assert "num_events" in metadata, "num_events not in metadata"
    assert "num_sessions" in metadata, "num_sessions not in metadata"
    
    assert metadata["dataset_name"] == TEST_DATASET_NAME, "Incorrect dataset name"
    assert metadata["num_users"] > 0, "Number of users should be greater than 0"
    assert metadata["num_events"] > 0, "Number of events should be greater than 0"
    assert metadata["num_sessions"] > 0, "Number of sessions should be greater than 0"

def test_multiple_datasets(cleanup_test_dataset):
    """Test creating multiple datasets doesn't interfere with each other."""
    test_dataset2 = f"{TEST_DATASET_NAME}_2"
    
    # Create first dataset
    subprocess.run([
        "python", "main.py", 
        "--dataset", TEST_DATASET_NAME,
        "--users", str(TEST_USERS),
        "--events", str(TEST_EVENTS)
    ], check=True)
    
    # Create second dataset
    subprocess.run([
        "python", "main.py", 
        "--dataset", test_dataset2,
        "--users", str(TEST_USERS),
        "--events", str(TEST_EVENTS)
    ], check=True)
    
    # Check both datasets exist
    assert Path(f"data/{TEST_DATASET_NAME}").exists(), "First dataset directory not found"
    assert Path(f"data/{test_dataset2}").exists(), "Second dataset directory not found"
    assert Path(f"outputs/{TEST_DATASET_NAME}").exists(), "First output directory not found"
    assert Path(f"outputs/{test_dataset2}").exists(), "Second output directory not found"
    
    # Clean up second dataset
    shutil.rmtree(f"data/{test_dataset2}")
    shutil.rmtree(f"outputs/{test_dataset2}")

def test_default_dataset_name():
    """Test that when no dataset name is provided, 'default' is used."""
    # First clean up any existing default dataset
    data_dir = Path("data/default")
    output_dir = Path("outputs/default")
    
    if data_dir.exists():
        shutil.rmtree(data_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Run main.py without specifying a dataset name
    subprocess.run([
        "python", "main.py", 
        "--users", str(TEST_USERS),
        "--events", str(TEST_EVENTS)
    ], check=True)
    
    # Check that directories exist
    assert Path("data/default").exists(), "Default dataset directory in data/ was not created"
    assert Path("outputs/default").exists(), "Default dataset directory in outputs/ was not created"
    
    # Clean up
    if data_dir.exists():
        shutil.rmtree(data_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir) 