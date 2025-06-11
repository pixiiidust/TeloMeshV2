import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import from analysis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.event_chokepoints import calculate_robust_wsjf_threshold

def create_zero_inflated_test_data():
    """Reproduces the 100K dataset problem: 90% zeros"""
    # Using exactly 90% zeros and 10% with a range of values
    zeros = [0.0] * 351  # 90% zeros
    
    # Mix of small and some larger values (to properly test threshold behavior)
    small_values = [0.0000001] * 30  # Very small values
    larger_values = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5]  # Some higher values to be detected
    
    return pd.DataFrame({
        'page': [f'page_{i}' for i in range(390)],  # Actual count from 100K test
        'event': [f'event_{i}' for i in range(390)],
        'WSJF_Friction_Score': zeros + small_values + larger_values
    })

def create_normal_test_data():
    """Simulates working 10K/20K datasets"""
    return pd.DataFrame({
        'page': [f'page_{i}' for i in range(100)],
        'event': [f'event_{i}' for i in range(100)],
        'WSJF_Friction_Score': [0.0] * 20 + np.random.uniform(0.01, 0.3, 80).tolist()
    })

def create_100k_regression_test_data():
    """Creates a dataset that closely matches the 100K user test case that originally failed"""
    # Create a large dataframe with 95% zeros and 5% with very small values
    n_rows = 2000  # Larger size to better simulate the real dataset
    zeros = [0.0] * int(n_rows * 0.95)  # 95% zeros
    
    # The remaining values follow a highly skewed distribution
    np.random.seed(42)  # For reproducibility
    small_values = np.random.exponential(0.01, int(n_rows * 0.04)).tolist()  # 4% very small values
    large_values = np.random.exponential(0.1, int(n_rows * 0.01)).tolist()  # 1% larger values
    
    return pd.DataFrame({
        'page': [f'page_{i//10}' for i in range(n_rows)],
        'event': [f'event_{i%10}' for i in range(n_rows)],
        'WSJF_Friction_Score': zeros + small_values + large_values
    })

def test_zero_inflated_distribution():
    """Test that the function handles zero-inflated distributions correctly (100K user case)"""
    # Given: A dataset with 90% zeros (like the 100K dataset)
    df = create_zero_inflated_test_data()
    
    # Verify the problem exists with traditional percentile
    old_threshold = df['WSJF_Friction_Score'].quantile(0.9)
    # Check if it's very close to zero (allowing for floating point precision)
    assert old_threshold < 1e-6, f"Test data doesn't reproduce the zero-threshold problem, got {old_threshold}"
    
    # When: Using the new function
    new_threshold = calculate_robust_wsjf_threshold(df)
    
    # Then: Should produce a positive threshold
    assert new_threshold > 0, "Should calculate a positive threshold"
    
    # And: Should identify a reasonable number of chokepoints (not all 390)
    chokepoints = df[df['WSJF_Friction_Score'] >= new_threshold]
    percentage = (len(chokepoints) / len(df)) * 100
    
    # Should be between 1% and 25% (reasonable range)
    assert 1 <= percentage <= 25, f"Should identify reasonable chokepoints, got {percentage:.1f}%"

def test_normal_distribution():
    """Test that the function works correctly on normal distributions (10K/20K user case)"""
    # Given: A dataset with normal distribution (like 10K/20K datasets)
    df = create_normal_test_data()
    
    # When: Using the new function
    threshold = calculate_robust_wsjf_threshold(df)
    
    # Then: Should produce a positive threshold
    assert threshold > 0, "Should calculate a positive threshold"
    
    # And: Should identify a reasonable number of chokepoints
    chokepoints = df[df['WSJF_Friction_Score'] >= threshold]
    percentage = (len(chokepoints) / len(df)) * 100
    
    # Should be between 5% and 30% (reasonable range)
    assert 5 <= percentage <= 30, f"Should identify reasonable chokepoints, got {percentage:.1f}%"

def test_all_zero_scores():
    """Test edge case of all zero WSJF scores"""
    # Given: A dataset with all zeros
    df = pd.DataFrame({
        'page': ['A', 'B', 'C'],
        'event': ['click', 'view', 'exit'],
        'WSJF_Friction_Score': [0.0, 0.0, 0.0]
    })
    
    # When: Using the new function
    threshold = calculate_robust_wsjf_threshold(df)
    
    # Then: Should use fallback and not return zero
    assert threshold > 0, "Should use min_floor fallback for all-zero case"

def test_few_non_zero_scores():
    """Test edge case of few non-zero WSJF scores"""
    # Given: A dataset with just a few non-zeros
    df = pd.DataFrame({
        'page': ['A', 'B', 'C', 'D', 'E', 'F'],
        'event': ['click'] * 6,
        'WSJF_Friction_Score': [0.0, 0.0, 0.0, 0.0, 0.1, 0.2]
    })
    
    # When: Using the new function
    threshold = calculate_robust_wsjf_threshold(df)
    
    # Then: Should handle the case correctly
    assert threshold > 0, "Should handle few non-zero case"
    
    # And: Should identify reasonable chokepoints
    chokepoints = df[df['WSJF_Friction_Score'] >= threshold]
    assert len(chokepoints) > 0, "Should identify at least one chokepoint"

def test_mathematical_correctness():
    """Test that the MAD calculation is mathematically correct"""
    # Given: A dataset with known values
    df = pd.DataFrame({
        'page': ['A', 'B', 'C', 'D', 'E'],
        'event': ['click'] * 5,
        'WSJF_Friction_Score': [0.1, 0.2, 0.3, 0.4, 0.5]  # Median = 0.3, MAD = 0.1
    })
    
    # When: Using the new function with mad_multiplier=2
    threshold = calculate_robust_wsjf_threshold(df, mad_multiplier=2.0)
    
    # Then: Should calculate correct threshold: median + 2*mad = 0.3 + 2*0.1 = 0.5
    assert abs(threshold - 0.5) < 1e-6, f"Should calculate correct threshold, got {threshold}"

def test_debug_output(capsys):
    """Test that debug output contains useful information"""
    # Given: A test dataset
    df = create_zero_inflated_test_data()
    
    # When: Using the function with verbose=True
    calculate_robust_wsjf_threshold(df, verbose=True)
    
    # Then: Output should contain useful debug information
    captured = capsys.readouterr()
    assert "non-zero" in captured.out, "Debug output should mention non-zero count"
    assert "Median:" in captured.out, "Debug output should show median"
    assert "MAD:" in captured.out, "Debug output should show MAD"
    assert "Threshold:" in captured.out, "Debug output should show threshold"

def test_100k_regression():
    """Regression test specifically for the 100K user case that originally failed"""
    # Given: A dataset that closely resembles the 100K user production data
    df = create_100k_regression_test_data()
    
    # Verify we have the same problem with the original method
    old_threshold = df['WSJF_Friction_Score'].quantile(0.9)
    # The 90th percentile should be very close to zero
    assert old_threshold < 1e-6, f"Regression test data doesn't match original problem, got {old_threshold}"
    
    # When: Using the new function
    new_threshold = calculate_robust_wsjf_threshold(df, verbose=True)
    
    # Then: Should produce a positive threshold
    assert new_threshold > 0, "Should calculate a positive threshold"
    
    # And: Should identify a reasonable number of chokepoints (not all rows)
    chokepoints = df[df['WSJF_Friction_Score'] >= new_threshold]
    percentage = (len(chokepoints) / len(df)) * 100
    
    # The key test: we should not be identifying ALL non-zero values as chokepoints
    # (which was the original bug - all non-zeros were chokepoints)
    non_zeros = df[df['WSJF_Friction_Score'] > 0]
    chokepoint_percentage = (len(chokepoints) / len(non_zeros)) * 100
    
    # Should identify a subset of non-zero values, not all of them
    assert chokepoint_percentage < 90, f"Should not identify all non-zeros as chokepoints ({chokepoint_percentage:.1f}%)"
    
    # And should identify more than a trivial number
    assert len(chokepoints) > 0, "Should identify at least some chokepoints" 