#!/usr/bin/env python3
"""
Script to analyze optimization results and generate summary statistics.

Usage: 
    Method 1: python analyze_results.py <optimizer_name> <benchmark_name>
    Method 2: python analyze_results.py <csv_file_path>

Example: 
    python analyze_results.py LLM autotrans
    python analyze_results.py nn_all_specs/results_LLMGB.csv
"""

import os
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_results_from_file(input_file):
    """
    Analyze results CSV file and generate summary statistics.
    
    Args:
        input_file (str): Path to the results CSV file
    
    Returns:
        tuple: (pd.DataFrame of summary statistics, optimizer_name, benchmark_name, output_dir)
    """
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found")
    
    # Extract optimizer and benchmark names from the file path
    input_path = Path(input_file)
    
    # Try to extract optimizer name from filename (e.g., results_LLMGB.csv -> LLMGB)
    filename = input_path.stem  # e.g., 'results_LLMGB'
    match = re.match(r'results_(\w+)', filename)
    if match:
        optimizer_name = match.group(1)
    else:
        optimizer_name = "Unknown"
    
    # Try to extract benchmark name from parent directory (e.g., nn_all_specs -> nn)
    parent_dir = input_path.parent.name
    match = re.match(r'(\w+)_all_specs', parent_dir)
    if match:
        benchmark_name = match.group(1)
    else:
        benchmark_name = parent_dir
    
    # Output files go to current working directory (same as script location)
    output_dir = Path.cwd()
    
    # Read the CSV file
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    return df, optimizer_name, benchmark_name, output_dir


def analyze_results(optimizer_name, benchmark_name):
    """
    Analyze results CSV file and generate summary statistics.
    
    Args:
        optimizer_name (str): Name of the optimizer (e.g., 'LLM', 'DA', 'UR')
        benchmark_name (str): Name of the benchmark (e.g., 'autotrans', 'cc')
    
    Returns:
        tuple: (pd.DataFrame from CSV, optimizer_name, benchmark_name, output_dir)
    """
    
    # Construct input file path
    input_file = f"{benchmark_name}_all_specs/results_{optimizer_name}.csv"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found")
    
    output_dir = Path(f"{benchmark_name}_all_specs")
    
    # Read the CSV file
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    return df, optimizer_name, benchmark_name, output_dir


def compute_summary_statistics(df):
    """
    Compute summary statistics from results dataframe.
    
    Args:
        df (pd.DataFrame): Results dataframe
    
    Returns:
        pd.DataFrame: Summary statistics for each specification
    """
    
    # Group by specification
    grouped = df.groupby('specification')
    
    results = []
    
    for spec, group in grouped:
        total_runs = len(group)
        falsified_runs = group[group['Falsified'] == True]
        num_falsified = len(falsified_runs)
        
        # Calculate falsification ratio
        falsification_ratio = num_falsified / total_runs
        
        # Calculate robustness statistics
        robustness_values = group['robustness']
        robustness_min = robustness_values.min()
        robustness_max = robustness_values.max()
        robustness_avg = robustness_values.mean()
        
        # Calculate nfev statistics for falsified runs
        if num_falsified > 0:
            falsified_nfev = falsified_runs['nfev']
            nfev_avg_falsified = falsified_nfev.mean()
            nfev_median_falsified = falsified_nfev.median()
        else:
            nfev_avg_falsified = np.nan
            nfev_median_falsified = np.nan
        
        results.append({
            'specification': spec,
            'total_runs': total_runs,
            'falsified_runs': num_falsified,
            'falsification_ratio': falsification_ratio,
            'robustness_min': robustness_min,
            'robustness_max': robustness_max,
            'robustness_avg': robustness_avg,
            'nfev_avg_falsified': nfev_avg_falsified,
            'nfev_median_falsified': nfev_median_falsified
        })
    
    # Custom sort: length of specification first, then alphabetical
    # This ensures AT6abc (len 6) comes after AT6c (len 4)
    results.sort(key=lambda x: (len(x['specification']), x['specification']))

    return pd.DataFrame(results)


def main():
    # Check number of arguments
    if len(sys.argv) not in [2, 3]:
        print("Usage:")
        print("  Method 1: python analyze_results.py <optimizer_name> <benchmark_name>")
        print("  Method 2: python analyze_results.py <csv_file_path>")
        print("\nExamples:")
        print("  python analyze_results.py LLM autotrans")
        print("  python analyze_results.py nn_all_specs/results_LLMGB.csv")
        sys.exit(1)
    
    try:
        # Determine which method to use based on number of arguments
        if len(sys.argv) == 2:
            # Method 2: CSV file path provided
            csv_file_path = sys.argv[1]
            df, optimizer_name, benchmark_name, output_dir = analyze_results_from_file(csv_file_path)
        else:
            # Method 1: optimizer_name and benchmark_name provided
            optimizer_name = sys.argv[1]
            benchmark_name = sys.argv[2]
            df, optimizer_name, benchmark_name, output_dir = analyze_results(optimizer_name, benchmark_name)
        
        # Compute summary statistics
        summary_df = compute_summary_statistics(df)
        
        # Create output file paths
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_long = output_dir / f"summary_long_{optimizer_name}.csv"
        output_file_short = output_dir / f"summary_short_{optimizer_name}.csv"
        
        # Save long version (all columns)
        summary_df.to_csv(output_file_long, index=False, float_format='%.6f')
        print(f"Long summary saved to {output_file_long}")
        
        # Create short version with selected columns only
        short_df = summary_df[['specification', 'falsification_ratio', 'nfev_avg_falsified', 'nfev_median_falsified']].copy()
        short_df.to_csv(output_file_short, index=False, float_format='%.6f')
        print(f"Short summary saved to {output_file_short}")
        
        # Display results
        print(f"\nSummary for {optimizer_name} on {benchmark_name}:")
        print("="*80)
        
        # Format display with better column names
        display_df = summary_df.copy()
        display_df.columns = [
            'Spec', 'Total', 'Falsified', 'Fals_Ratio', 'Rob_Min', 
            'Rob_Max', 'Rob_Avg', 'NFev_Avg_Fals', 'NFev_Med_Fals'
        ]
        
        # Round numerical columns for display
        numeric_cols = ['Fals_Ratio', 'Rob_Min', 'Rob_Max', 
                       'Rob_Avg', 'NFev_Avg_Fals', 'NFev_Med_Fals']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)
        
        print(display_df.to_string(index=False))
        
        # Summary statistics
        print("\nOverall Statistics:")
        print(f"Total specifications: {len(summary_df)}")
        print(f"Specifications with at least one falsification: {(summary_df['falsification_ratio'] > 0).sum()}")
        print(f"Average falsification ratio: {summary_df['falsification_ratio'].mean():.4f}")
        print(f"Average robustness: {summary_df['robustness_avg'].mean():.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
