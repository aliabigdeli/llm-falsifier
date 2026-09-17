import csv
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import sys

import numpy as np
from .critical_time_finder import find_critical_time

# Suppress matplotlib's verbose debug logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def analyze_violation(specification, trace, robustness_value: float, output_file: Optional[str] = None):
    """
    Generalized violation analysis for any specification.
    
    Args:
        specification: The specification object (RTAMTDense/RTAMTDiscrete)
        trace: The simulation trace from best_result
        robustness_value: The computed robustness value
        output_file: Optional file path to write analysis to
    """
    times = list(trace.times)
    states = trace.states
    
    def write_output(text: str, file_handle=None):
        """Helper function to write to both console and file if provided"""
        print(text)
        if file_handle:
            file_handle.write(text + '\n')
    
    # Open file if provided
    file_handle = None
    if output_file:
        file_handle = Path(output_file).open('a')
    
    try:
        write_output(f"\n{'='*50}", file_handle)
        write_output("ROBUSTNESS ANALYSIS", file_handle)
        write_output(f"{'='*50}", file_handle)
        write_output(f"Robustness value: {robustness_value:.6f}", file_handle)
        
        if robustness_value < 0:
            write_output("🚨 SPECIFICATION VIOLATED! Counterexample found.", file_handle)
            write_output(f"Violation severity: {abs(robustness_value):.6f}", file_handle)
            
            write_output("\nViolation analysis:", file_handle)
            write_output(f"Specification: {specification.phi}", file_handle)
            write_output(f"Variable mappings: {specification.column_map}", file_handle)
            
            # Extract variable values from trace based on column mapping
            variable_traces = {}
            for var_name, column_idx in specification.column_map.items():
                variable_traces[var_name] = [state[column_idx] for state in states]
            
            # Sample some time points to show variable values
            write_output("\nTrace analysis around potential violation points:", file_handle)
            sample_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
            for i in sample_indices:
                if i >= len(times):
                    continue
                t = times[i]
                trace_line = f"  t={t:.2f}s:"
                for var_name, values in variable_traces.items():
                    if i < len(values):
                        trace_line += f" {var_name}={values[i]:.2f}"
                    else:
                        trace_line += f" {var_name}=N/A"
                write_output(trace_line, file_handle)
            
            # Show min/max values for each variable
            write_output("\nVariable ranges during simulation:", file_handle)
            for var_name, values in variable_traces.items():
                if values:  # Check if list is not empty
                    min_val, max_val = min(values), max(values)
                    min_idx = values.index(min_val)
                    max_idx = values.index(max_val)
                    write_output(f"  {var_name}: min={min_val:.2f} at t={times[min_idx]:.2f}s, max={max_val:.2f} at t={times[max_idx]:.2f}s", file_handle)
        else:
            write_output("✅ Specification satisfied", file_handle)
            write_output(f"Safety margin: {robustness_value:.6f}", file_handle)
    
    finally:
        if file_handle:
            file_handle.close()


def find_critical_time_from_spec_trace(
    spec: 'RTAMTDense',
    trace: 'Trace',
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Find critical time from an RTAMTDense specification and Staliro Trace object or trace dict.

    This function extracts the formula and signal mapping directly from the 
    RTAMTDense specification object, making it convenient to use with the
    staliro framework.

    Args:
        spec: RTAMTDense specification object from staliro.specifications
        trace: Staliro Trace object or a dict with keys "times" and "states"
        verbose: If True, print detailed information during analysis

    Returns:
        Dictionary with critical time information including:
            - critical_time: The critical time point
            - critical_index: Index in the trace
            - robustness: Overall robustness value
            - path: List of formulas in the path to critical time
            - details: Additional details
            - formula: The STL formula string
            - signal_mapping: The signal to column mapping

    Example:
        >>> from staliro.specifications import RTAMTDense
        >>> from staliro.core.model import Trace
        >>> import numpy as np
        >>> 
        >>> # Create specification
        >>> spec = RTAMTDense("G[0,20] (speed <= 120)", {"speed": 0})
        >>> 
        >>> # Create trace
        >>> times = np.linspace(0, 20, 100)
        >>> signals = np.column_stack([speed_values, rpm_values])
        >>> trace = Trace(times, signals)
        >>> 
        >>> # Find critical time
        >>> result = find_critical_time_from_spec_trace(spec, trace)
        >>> print(f"Critical time: {result['critical_time']}")
        >>> print(f"Robustness: {result['robustness']}")
    """

    # Extract formula from the RTAMTDense specification
    if hasattr(spec, 'phi'):
        formula = spec.phi
    else:
        raise AttributeError(
            "Cannot extract formula from RTAMTDense specification. "
            "Expected 'phi' attribute."
        )

    # Extract signal mapping from the RTAMTDense specification
    if hasattr(spec, 'column_map'):
        signal_mapping = spec.column_map
    else:
        raise AttributeError(
            "Cannot extract signal mapping from RTAMTDense specification. "
            "Expected 'column_map' attribute."
        )

    if not isinstance(signal_mapping, dict):
        raise TypeError(
            f"Signal mapping must be a dictionary, got {type(signal_mapping)}"
        )

    # Support trace as a dict with "times" and "states", or as Trace object
    if isinstance(trace, dict) and "times" in trace and "states" in trace:
        times_obj = trace["times"]
        states_obj = trace["states"]
    else:
        times_obj = getattr(trace, "times", None)
        states_obj = getattr(trace, "states", None)

    if times_obj is None or states_obj is None:
        raise ValueError("Trace object or dict must provide 'times' and 'states'.")

    # Convert times to list
    if isinstance(times_obj, np.ndarray):
        time_points = times_obj.tolist()
    elif isinstance(times_obj, list):
        time_points = times_obj
    else:
        time_points = list(times_obj)

    trace_data = {'time': time_points}

    # Map signal names to trace columns
    for signal_name, col_idx in signal_mapping.items():
        # np.ndarray (states) case
        if isinstance(states_obj, np.ndarray):
            if states_obj.ndim == 1:
                # Single column signal as flat vector
                if col_idx == 0:
                    trace_data[signal_name] = states_obj.tolist()
                else:
                    raise ValueError(
                        f"Column index {col_idx} out of bounds for signal '{signal_name}'. "
                        f"Trace 'states' has shape {states_obj.shape}."
                    )
            else:
                if col_idx < states_obj.shape[1]:
                    trace_data[signal_name] = states_obj[:, col_idx].tolist()
                else:
                    raise ValueError(
                        f"Column index {col_idx} out of bounds for signal '{signal_name}'. "
                        f"Trace has {states_obj.shape[1]} columns."
                    )
        elif isinstance(states_obj, list):
            if len(states_obj) > 0:
                # Each state is a row: list[tuple|list|array]
                if isinstance(states_obj[0], (list, tuple, np.ndarray)):
                    if col_idx < len(states_obj[0]):
                        trace_data[signal_name] = [state[col_idx] for state in states_obj]
                    else:
                        raise ValueError(
                            f"Column index {col_idx} out of bounds for signal '{signal_name}'. "
                            f"Each state has {len(states_obj[0])} columns."
                        )
                else:
                    raise TypeError(f"Unexpected state format: {type(states_obj[0])}")
            else:
                raise ValueError("Trace states are empty")
        else:
            raise TypeError(f"Unexpected trace.states type: {type(states_obj)}")

    result = find_critical_time(formula, signal_mapping, trace_data, verbose=verbose)
    result['formula'] = formula
    result['signal_mapping'] = signal_mapping

    return result


def plot_trace_variables(specification, trace, filename: str = "plot.jpeg", include_critical_time: bool = True, save_trace_to_csv: bool = True):
    """
    Generalized plotting for any variables in the specification.
    
    Args:
        specification: The specification object (RTAMTDense/RTAMTDiscrete)
        trace: The simulation trace from best_result
        filename: Output filename for the plot
        include_critical_time: Whether to include the critical time in the plot
    """
    times = list(trace.times)
    states = trace.states

    # Extract variable traces based on specification mapping
    variable_traces = {}
    for var_name, column_idx in specification.column_map.items():
        variable_traces[var_name] = [state[column_idx] for state in states]

    if save_trace_to_csv:
        # Save trace values to CSV file
        csv_filename = filename[:-5] + ".csv" if filename.endswith('.jpeg') else filename + ".csv"
        csv_columns = ["time"] + list(variable_traces.keys())
        rows = []
        for i in range(len(times)):
            row = [times[i]]
            for var_name in variable_traces:
                row.append(variable_traces[var_name][i] if i < len(variable_traces[var_name]) else "")
            rows.append(row)
        # Write to CSV
        with open(csv_filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_columns)
            writer.writerows(rows)
        print(f"Trace values saved to CSV: {csv_filename}")

    # Create subplots for each variable
    num_vars = len(variable_traces)
    if num_vars > 0:
        fig, axes = plt.subplots(num_vars, 1, figsize=(12, 8), sharex=True)
        
        # If only one variable, axes is not a list, so make it a list for consistency
        if num_vars == 1:
            axes = [axes]

        for i, (var_name, values) in enumerate(variable_traces.items()):
            axes[i].plot(times, values, label=var_name, linewidth=2)
            axes[i].set_ylabel(var_name.capitalize(), fontsize=10)
            axes[i].grid(True, which='major', alpha=0.6, linestyle='-', linewidth=0.8)
            axes[i].grid(True, which='minor', alpha=0.5, linestyle=':', linewidth=0.7)
            axes[i].minorticks_on()
            axes[i].legend(loc='best')

        # Find and plot critical time if requested
        critical_time = None
        if include_critical_time:
            try:
                critical_time_result = find_critical_time_from_spec_trace(specification, trace, verbose=True)
                critical_time = critical_time_result['critical_time']
                
                # Add vertical line at critical time for each subplot
                for i in range(num_vars):
                    axes[i].axvline(x=critical_time, color='red', linestyle='--', linewidth=2, 
                                   label=f'Critical Time: {critical_time:.3f}s, ρ={critical_time_result["robustness"]:.3f}')
                    axes[i].legend(loc='best')
                
                print(f"\nCritical time found at t={critical_time:.3f}s")
                print(f"Robustness: {critical_time_result['robustness']:.4f}")
                
            except Exception as e:
                print(f"\nWarning: Could not find critical time: {e}")

        # Add a title to the plot
        plot_title = f"Trace Vars for Spec: {getattr(specification, 'phi', 'N/A')}"
        fig.suptitle(plot_title, fontsize=10, fontweight='bold')
        
        # Set x-axis label on the bottom subplot
        axes[-1].set_xlabel('Time (s)', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\nPlot saved as: {filename}")


def save_results_to_csv(csv_filename: str, spec_name: str, seed: int, robustness: float, nfev: int, execution_time: float):
    """Save results to CSV file with proper headers."""
    is_falsified = robustness < 0
    
    # Check if CSV file exists and write header if it doesn't
    file_exists = os.path.exists(csv_filename)
    with Path(csv_filename).open('a', newline='') as csvfile:
        fieldnames = ['specification', 'seed', 'robustness', 'Falsified', 'nfev', 'exec_time', 'time_per_iteration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the current result
        writer.writerow({
            'specification': spec_name,
            'seed': seed,
            'robustness': robustness,
            'Falsified': is_falsified,
            'nfev': nfev,
            'exec_time': round(execution_time, 2),
            'time_per_iteration': round(execution_time / nfev, 2)
        })


def write_analysis_report(txt_filename: str, spec_name: str, optimizer, args, options, 
                         result, best_sample, robustness: float, system_name: str, execution_time: float):
    """Write analysis report to text file."""
    with Path(txt_filename).open('w') as f:
        f.write(f"{system_name.upper()} SPECIFICATION ANALYSIS REPORT\n")
        f.write(f"{'='*50}\n\n")
        f.write("Command-line arguments used to run this script:\n")
        f.write(" ".join(sys.argv) + "\n\n")
        f.write(f"Specification name: {spec_name}\n")
        f.write(f"Optimizer: {type(optimizer).__name__}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"Simulation time interval: {options.interval}\n")
        f.write(f"Max number of iterations (max budget), equal to max number of allowed function evaluations: {options.iterations}\n")
        f.write(f"Number of Function Evaluations (nfev), equal to the number of model simulations/simulink runs: {result.runs[0].result.nfev}\n")
        f.write(f"Total runtime execution time: {execution_time:.2f} seconds\n\n")
        
        # Show counterexample sample if violation occurred
        if robustness < 0:
            f.write("COUNTEREXAMPLE INPUT SAMPLE:\n")
            f.write(f"{best_sample}\n\n")
        else:
            f.write("BEST SAMPLE FOUND:\n")
            f.write(f"{best_sample}\n\n")
