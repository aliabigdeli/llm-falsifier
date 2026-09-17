"""
Test script for the Critical Time Finder

This script demonstrates how to use the critical_time_finder module with various
STL specifications.
"""

import numpy as np
import matplotlib.pyplot as plt
from llm_falsifier.critical_time_finder import find_critical_time, fill_equal_robustness
import os
import rtamt


# Create output directory for plots
PLOT_DIR = "test_critical_time_plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def plot_test_results(trace_data, result, formula, test_name, signal_names=None, thresholds=None, signal_mapping=None, mark_crit_t_rob_in_rob_plot=False):
    """
    Create a visualization of the test results showing signals, critical times, and robustness.
    
    Args:
        trace_data: Dictionary with 'time' and signal data
        result: Result dictionary from find_critical_time
        formula: STL formula string
        test_name: Name for the plot file
        signal_names: List of signal names to plot (default: all except 'time')
        thresholds: Dictionary of signal_name -> threshold_value for threshold lines
        signal_mapping: Dictionary mapping signal names to indices (required for robustness computation)
    """
    if signal_names is None:
        signal_names = [k for k in trace_data.keys() if k != 'time']
    
    time_points = trace_data['time']
    critical_time = result['critical_time']
    critical_index = result['critical_index']
    robustness = result['robustness']
    
    # Evaluate formula to get robustness trace
    robustness_traces = {}
    if signal_mapping is not None:
        try:
            # Prepare traces for rtamt (used by all formula evaluations)
            traces = []
            for sig_name in signal_mapping.keys():
                trace_data_for_sig = [[time_points[i], trace_data[sig_name][i]] 
                                     for i in range(len(time_points))]
                traces.append((sig_name, trace_data_for_sig))
            
            # Evaluate main formula
            spec = rtamt.StlDenseTimeSpecification()
            for sig_name in signal_mapping.keys():
                spec.declare_var(sig_name, 'float')
            spec.spec = formula
            spec.parse()
            rob_trace = spec.evaluate(*traces)
            robustness_traces['main'] = {
                'times': time_points,
                'values': fill_equal_robustness(time_points, [t for t, v in rob_trace], [v for t, v in rob_trace]),
                'label': 'Main Formula'
            }
            
            # Evaluate subformulas from the critical path
            # Skip the last element (atomic formula) and the first (same as main)
            path = result.get('path', [])
            if len(path) > 2:
                for i, subformula in enumerate(path[1:-1], start=1):  # Skip first and last
                    try:
                        spec_sub = rtamt.StlDenseTimeSpecification()
                        for sig_name in signal_mapping.keys():
                            spec_sub.declare_var(sig_name, 'float')
                        spec_sub.spec = subformula
                        spec_sub.parse()
                        rob_trace_sub = spec_sub.evaluate(*traces)
                        robustness_traces[f'sub_{i}'] = {
                            'times': time_points,
                            'values': fill_equal_robustness(time_points, 
                                                           [t for t, v in rob_trace_sub], 
                                                           [v for t, v in rob_trace_sub]),
                            'label': f'Subformula {i}: {subformula[:50]}{"..." if len(subformula) > 50 else ""}'
                        }
                    except Exception as e_sub:
                        print(f"Warning: Could not evaluate subformula {i}: {e_sub}")
        except Exception as e:
            print(f"Warning: Could not evaluate robustness trace: {e}")
    
    # Determine number of subplots needed (signals + robustness)
    num_signals = len(signal_names)
    num_subplots = num_signals + 1  # Add 1 for robustness subplot
    fig, axes = plt.subplots(num_subplots, 1, figsize=(14, 4 * num_subplots))
    
    # Make axes always iterable
    if num_subplots == 1:
        axes = [axes]
    
    # Plot each signal
    for idx, signal_name in enumerate(signal_names):
        ax = axes[idx]
        signal_values = trace_data[signal_name]
        
        # Plot signal
        ax.plot(time_points, signal_values, 'b-o', linewidth=2, markersize=6, label=f'{signal_name}')
        
        # Mark critical time
        ax.axvline(x=critical_time, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Critical Time t* = {critical_time:.2f}s')
        
        # Mark signal value at critical time
        signal_at_critical = signal_values[critical_index]
        ax.plot(critical_time, signal_at_critical, 'ro', markersize=12,
                label=f'{signal_name} at t* = {signal_at_critical:.2f}')
        
        # Add threshold line if provided
        if thresholds and signal_name in thresholds:
            threshold = thresholds[signal_name]
            ax.axhline(y=threshold, color='green', linestyle='-', linewidth=1.5, alpha=0.5,
                       label=f'Threshold = {threshold}')
        
        # Styling
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(f'{signal_name}', fontsize=12)
        ax.set_title(f'{signal_name} vs Time with Critical Time Highlighted', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        # Add statistics text box
        stats_text = f'Min: {min(signal_values):.2f}\n'
        stats_text += f'Max: {max(signal_values):.2f}\n'
        stats_text += f'Mean: {np.mean(signal_values):.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ===== ROBUSTNESS SUBPLOT =====
    if robustness_traces:
        ax_rob = axes[num_signals]  # Last subplot for robustness
        
        # Define colors for different traces
        colors = ['green', 'blue', 'purple', 'orange', 'cyan', 'magenta']
        color_idx = 0
        
        # Plot main formula robustness
        if 'main' in robustness_traces:
            trace = robustness_traces['main']
            ax_rob.plot(trace['times'], trace['values'], 
                       color=colors[color_idx % len(colors)], linewidth=2, 
                       marker='o', markersize=5, label=trace['label'])
            color_idx += 1
        
        # Plot sub-formula robustness traces (if available)
        for key, trace in robustness_traces.items():
            if key != 'main':
                ax_rob.plot(trace['times'], trace['values'], 
                           color=colors[color_idx % len(colors)], linewidth=1.5,
                           marker='s', markersize=4, alpha=0.7, label=trace['label'])
                color_idx += 1
        
        # Mark critical time
        ax_rob.axvline(x=critical_time, color='red', linestyle='--', linewidth=2, alpha=0.7,
                      label=f'Critical Time t* = {critical_time:.2f}s')
        
        # Mark robustness at critical time for main formula
        if mark_crit_t_rob_in_rob_plot:
            if 'main' in robustness_traces:
                trace = robustness_traces['main']
                # Find robustness value at critical time
                rob_times = np.array(trace['times'])
                rob_values = np.array(trace['values'])
                # Find closest time point
                idx = np.argmin(np.abs(rob_times - critical_time))
                rob_at_critical = rob_values[idx]
                ax_rob.plot(critical_time, rob_at_critical, 'ro', markersize=12,
                        label=f'Main ρ at t* = {rob_at_critical:.4f}')
        
        # Mark zero robustness line
        ax_rob.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3,
                      label='ρ = 0 (boundary)')
        
        # Styling
        ax_rob.set_xlabel('Time (s)', fontsize=12)
        ax_rob.set_ylabel('Robustness (ρ)', fontsize=12)
        ax_rob.set_title('Robustness Traces vs Time', fontsize=14, fontweight='bold')
        ax_rob.grid(True, alpha=0.3)
        ax_rob.legend(loc='best', fontsize=9)
        
        # Add statistics
        if 'main' in robustness_traces:
            trace = robustness_traces['main']
            rob_values = trace['values']
            stats_text = f'Min ρ: {min(rob_values):.4f}\n'
            stats_text += f'Max ρ: {max(rob_values):.4f}\n'
            stats_text += f'Mean ρ: {np.mean(rob_values):.4f}'
            ax_rob.text(0.02, 0.98, stats_text, transform=ax_rob.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Add overall information at the top
    fig.suptitle(f'{test_name}\nFormula: {formula}\nOverall Robustness: {robustness:.4f}',
                 fontsize=12, fontweight='bold')
    
    # Add path information at the bottom
    path_text = "Critical Path:\n" + "\n".join([f"  {i+1}. {step}" for i, step in enumerate(result['path'])])
    fig.text(0.02, 0.01, path_text, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    
    # Save plot
    plot_path = os.path.join(PLOT_DIR, f'{test_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()


def test_simple_always():
    """Test with a simple always specification"""
    print("=" * 80)
    print("TEST 1: Simple Always Specification")
    print("=" * 80)
    
    formula = "always[0,5] (signal > 10)"
    signal_mapping = {"signal": 0}
    
    # Create a trace where minimum occurs at t=3.0
    trace_data = {
        'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'signal': [15.0, 20.0, 12.0, 10.5, 11.0, 18.0, 22.0]
    }
    
    result = find_critical_time(formula, signal_mapping, trace_data, verbose=True)
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"Critical Time: {result['critical_time']:.2f}")
    print(f"Critical Index: {result['critical_index']}")
    print(f"Overall Robustness: {result['robustness']:.4f}")
    print(f"Expected robustness at t=3.0: {trace_data['signal'][3] - 10:.4f}")
    print(f"\nPath to critical time:")
    for i, step in enumerate(result['path'], 1):
        print(f"  {i}. {step}")
    print("=" * 80)
    
    # Generate plot
    plot_test_results(trace_data, result, formula, 
                     test_name='test1_simple_always',
                     thresholds={'signal': 10},
                     signal_mapping=signal_mapping)
    print()


def test_nested_eventually_always():
    """Test with nested temporal operators"""
    print("=" * 80)
    print("TEST 2: Nested Temporal Operators")
    print("=" * 80)
    
    formula = "always[0,5] (eventually[0,4] (signal > 10))"
    signal_mapping = {"signal": 0}
    
    trace_data = {
        'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'signal': [20, 18, 17, 11, 10.5, 12, 10.8, 15, 16, 14, 13]
    }
    
    result = find_critical_time(formula, signal_mapping, trace_data, verbose=True)
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"Critical Time: {result['critical_time']:.2f}")
    print(f"Overall Robustness: {result['robustness']:.4f}")
    print(f"\nPath to critical time:")
    for i, step in enumerate(result['path'], 1):
        print(f"  {i}. {step}")
    print("=" * 80)
    
    # Generate plot
    plot_test_results(trace_data, result, formula, 
                     test_name='test2_nested_eventually_always',
                     thresholds={'signal': 10},
                     signal_mapping=signal_mapping)
    print()


def test_not_nested_eventually_always():
    """Test with nested temporal operators"""
    print("=" * 80)
    print("TEST 3: Not Nested Temporal Operators")
    print("=" * 80)
    
    formula = "not (always[0,5] (eventually[0,4] (signal > 10)))"
    signal_mapping = {"signal": 0}
    
    trace_data = {
        'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'signal': [20, 18, 17, 11, 10.5, 12, 10.8, 15, 16, 14, 13]
    }
    
    result = find_critical_time(formula, signal_mapping, trace_data, verbose=True)
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"Critical Time: {result['critical_time']:.2f}")
    print(f"Overall Robustness: {result['robustness']:.4f}")
    print(f"\nPath to critical time:")
    for i, step in enumerate(result['path'], 1):
        print(f"  {i}. {step}")
    print("=" * 80)
    
    # Generate plot
    plot_test_results(trace_data, result, formula, 
                     test_name='test3_not_nested_eventually_always',
                     thresholds={'signal': 10},
                     signal_mapping=signal_mapping)
    print()


def test_and_operator():
    """Test with AND operator"""
    print("=" * 80)
    print("TEST 4: AND Operator")
    print("=" * 80)
    
    formula = "always[0, 6] ((eventually[0,2] (signal1 >= 15)) and (always[0,3] (signal2 <= 5)))"
    signal_mapping = {"signal1": 0, "signal2": 1}
    
    trace_data = {
        'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        'signal1': [20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
        'signal2': [0, 0, 0, 0, 2, 3, 9, 4, 0, 0]
    }
    
    result = find_critical_time(formula, signal_mapping, trace_data, verbose=True)
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"Critical Time: {result['critical_time']:.2f}")
    print(f"Overall Robustness: {result['robustness']:.4f}")
    print(f"\nPath to critical time:")
    for i, step in enumerate(result['path'], 1):
        print(f"  {i}. {step}")
    print("=" * 80)
    
    # Generate plot
    plot_test_results(trace_data, result, formula, 
                     test_name='test4_and_operator',
                     thresholds={'signal1': 15, 'signal2': 5},
                     signal_mapping=signal_mapping)
    print()


def test_or_operator():
    """Test with OR operator"""
    print("=" * 80)
    print("TEST 5: OR Operator")
    print("=" * 80)
    
    formula = "always[0, 6] ((eventually[0,2] (signal1 >= 15)) or (always[0,3] (signal2 <= 5)))"
    signal_mapping = {"signal1": 0, "signal2": 1}
    
    trace_data = {
        'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        'signal1': [10, 11, 12, 10, 13, 14, 11, 20, 20, 20],
        'signal2': [3, 3, 4, 4, 6, 6.9, 8, 2, 1, 1]
    }
    
    result = find_critical_time(formula, signal_mapping, trace_data, verbose=True)
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"Critical Time: {result['critical_time']:.2f}")
    print(f"Overall Robustness: {result['robustness']:.4f}")
    print(f"\nPath to critical time:")
    for i, step in enumerate(result['path'], 1):
        print(f"  {i}. {step}")
    print("=" * 80)
    
    # Generate plot
    plot_test_results(trace_data, result, formula, 
                     test_name='test5_or_operator',
                     thresholds={'signal1': 15, 'signal2': 5},
                     signal_mapping=signal_mapping)
    print()


def test_implication():
    """Test with implication operator"""
    print("=" * 80)
    print("TEST 6: Implication Operator")
    print("=" * 80)
    
    formula = "always[0, 6] ((eventually[0,2] (signal1 >= 15)) -> (always[0,3] (signal2 <= 5)))"
    signal_mapping = {"signal1": 0, "signal2": 1}
    
    trace_data = {
        'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        'signal1': [16, 17, 18, 20, 22, 19, 18, 15, 14, 13],
        'signal2': [3, 4, 4, 4, 7, 8, 9, 2, 1, 1]
    }
    
    result = find_critical_time(formula, signal_mapping, trace_data, verbose=True)
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"Critical Time: {result['critical_time']:.2f}")
    print(f"Overall Robustness: {result['robustness']:.4f}")
    print(f"\nPath to critical time:")
    for i, step in enumerate(result['path'], 1):
        print(f"  {i}. {step}")
    print("=" * 80)
    
    # Generate plot
    plot_test_results(trace_data, result, formula, 
                     test_name='test6_implication',
                     thresholds={'signal1': 15, 'signal2': 5},
                     signal_mapping=signal_mapping)
    print()


def test_autotrans_at1():
    """Test with Autotrans AT1 specification"""
    print("=" * 80)
    print("TEST 7: Autotrans AT1 - G[0, 20] (speed <= 120)")
    print("=" * 80)
    
    formula = "G[0, 20] (speed <= 120)"
    signal_mapping = {"speed": 0}
    
    # Simulated trace data
    time_points = np.linspace(0, 20, 50)
    # Speed increases and peaks at t=12
    speed_values = 80 + 35 * np.sin(time_points * np.pi / 20) + 10 * np.sin(time_points * 2)
    speed_values[24] = 125  # Violation at t=12
    
    trace_data = {
        'time': time_points.tolist(),
        'speed': speed_values.tolist()
    }
    
    result = find_critical_time(formula, signal_mapping, trace_data, verbose=True)
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"Critical Time: {result['critical_time']:.2f}")
    print(f"Speed at critical time: {trace_data['speed'][result['critical_index']]:.2f}")
    print(f"Overall Robustness: {result['robustness']:.4f}")
    print(f"\nPath to critical time:")
    for i, step in enumerate(result['path'], 1):
        print(f"  {i}. {step}")
    print("=" * 80)
    
    # Generate plot
    plot_test_results(trace_data, result, formula, 
                     test_name='test7_autotrans_at1',
                     thresholds={'speed': 120},
                     signal_mapping=signal_mapping)
    print()


def test_cc_cc1():
    """Test with Chasing cars CC1 specification"""
    print("=" * 80)
    print("TEST 8: Chasing cars CC1 - G[0, 100] (y54 <= 40)")
    print("=" * 80)
    
    formula = "G[0, 100] (y54 <= 40)"
    signal_mapping = {"y54": 0}
    
    # Simulated trace data
    time_points = np.linspace(0, 100, 200)
    # Distance varies, peaks at t=60
    y54_values = 30 + 8 * np.sin(time_points * np.pi / 100)
    y54_values[120] = 42  # Violation at t=60
    
    trace_data = {
        'time': time_points.tolist(),
        'y54': y54_values.tolist()
    }
    
    result = find_critical_time(formula, signal_mapping, trace_data, verbose=False)
    
    print(f"\nRESULTS:")
    print(f"Critical Time: {result['critical_time']:.2f}")
    print(f"y54 at critical time: {trace_data['y54'][result['critical_index']]:.2f}")
    print(f"Overall Robustness: {result['robustness']:.4f}")
    print(f"\nPath to critical time:")
    for i, step in enumerate(result['path'], 1):
        print(f"  {i}. {step}")
    print("=" * 80)
    
    # Generate plot
    plot_test_results(trace_data, result, formula, 
                     test_name='test8_cc_cc1',
                     thresholds={'y54': 40},
                     signal_mapping=signal_mapping)
    print()


def test_complex_nested():
    """Test with a complex nested specification"""
    print("=" * 80)
    print("TEST 9: Complex Nested Specification")
    print("=" * 80)
    
    # CC5 specification
    formula = "G[0,72] (F[0,8] ((G[0,5] (y21 >= 9)) -> (G[5,20] (y54 >= 9))))"
    signal_mapping = {"y21": 0, "y54": 1}
    
    # Simulated trace data
    time_points = np.linspace(0, 80, 160)
    y21_values = 10 + 2 * np.sin(time_points * np.pi / 40)
    y54_values = 10 + 3 * np.sin(time_points * np.pi / 50 + 1)
    y54_values[80:100] = 8  # Create a violation region
    
    trace_data = {
        'time': time_points.tolist(),
        'y21': y21_values.tolist(),
        'y54': y54_values.tolist()
    }
    
    result = find_critical_time(formula, signal_mapping, trace_data, verbose=True)
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"Critical Time: {result['critical_time']:.2f}")
    print(f"y21 at critical time: {trace_data['y21'][result['critical_index']]:.2f}")
    print(f"y54 at critical time: {trace_data['y54'][result['critical_index']]:.2f}")
    print(f"Overall Robustness: {result['robustness']:.4f}")
    print(f"\nPath to critical time:")
    for i, step in enumerate(result['path'], 1):
        print(f"  {i}. {step}")
    print("=" * 80)
    
    # Generate plot
    plot_test_results(trace_data, result, formula, 
                     test_name='test9_complex_nested',
                     thresholds={'y21': 9, 'y54': 9},
                     signal_mapping=signal_mapping)
    print()


def test_triple_and():
    """Test with triple AND - multiple children"""
    print("=" * 80)
    print("TEST 10: Triple AND (Multiple Children)")
    print("=" * 80)
    
    # Test formula: (A) and (B) and (C) where A is the limiting constraint
    formula = "(always[0,20] (speed <= 35)) and (always[0,20] (speed <= 50)) and (always[0,20] (speed <= 65))"
    signal_mapping = {"speed": 0}
    
    # Create trace where speed increases and violates constraints
    times = np.linspace(0, 30, 301)
    speeds = np.linspace(0, 70, 301)  # Speed increases from 0 to 70
    
    trace_data = {
        'time': times.tolist(),
        'speed': speeds.tolist()
    }
    
    result = find_critical_time(formula, signal_mapping, trace_data, verbose=False)
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"Critical Time: {result['critical_time']:.2f}")
    print(f"Overall Robustness: {result['robustness']:.4f}")
    print(f"Speed at critical time: {speeds[result['critical_index']]:.2f}")
    print(f"\nPath to critical time:")
    for i, step in enumerate(result['path'], 1):
        print(f"  {i}. {step}")
    
    # The tightest constraint is speed <= 35, most violated at t=20
    # Speed at t=20 is about 46.67, so robustness = 35 - 46.67 = -11.67
    print(f"\nExpected: Critical time at t=20.00 where speed first exceeds 35")
    print(f"          (the tightest constraint in the AND)")
    print("=" * 80)
    
    # Generate plot
    plot_test_results(trace_data, result, formula, 
                     test_name='test10_triple_and',
                     thresholds={'speed': 35},
                     signal_mapping=signal_mapping)
    print()


def test_triple_or():
    """Test with triple OR - multiple children"""
    print("=" * 80)
    print("TEST 11: Triple OR (Multiple Children)")
    print("=" * 80)
    
    # Test formula: (A) or (B) or (C) where we need the max
    formula = "(always[0,10] (x <= 10)) or (always[0,10] (x <= 20)) or (always[0,10] (x <= 30))"
    signal_mapping = {"x": 0}
    
    # Create trace where x increases and violates all constraints
    times = np.linspace(0, 15, 151)
    x_values = np.linspace(0, 50, 151)  # x increases from 0 to 50
    
    trace_data = {
        'time': times.tolist(),
        'x': x_values.tolist()
    }
    
    result = find_critical_time(formula, signal_mapping, trace_data, verbose=False)
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"Critical Time: {result['critical_time']:.2f}")
    print(f"Overall Robustness: {result['robustness']:.4f}")
    print(f"\nPath to critical time:")
    for i, step in enumerate(result['path'], 1):
        print(f"  {i}. {step}")
    
    # For OR, the limiting constraint is the loosest one (x <= 30)
    # which gives the highest (least negative) robustness
    print(f"\nExpected: Critical path goes through the loosest constraint (x <= 30)")
    print(f"          which has the maximum robustness among the three")
    print("=" * 80)
    
    # Generate plot
    plot_test_results(trace_data, result, formula, 
                     test_name='test11_triple_or',
                     thresholds={'x': 30},
                     signal_mapping=signal_mapping)
    print()


if __name__ == "__main__":
    # Run all tests
    tests = [
        test_simple_always,
        test_nested_eventually_always,
        test_not_nested_eventually_always,
        test_and_operator,
        test_or_operator,
        test_implication,
        test_autotrans_at1,
        test_cc_cc1,
        test_complex_nested,
        test_triple_and,
        test_triple_or,
    ]
    
    for i, test_func in enumerate(tests, 1):
        try:
            print(f"\n{'#'*80}")
            print(f"# Running Test {i}/{len(tests)}")
            print(f"{'#'*80}\n")
            test_func()
        except Exception as e:
            print(f"\n{'!'*80}")
            print(f"ERROR in {test_func.__name__}: {str(e)}")
            print(f"{'!'*80}\n")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

