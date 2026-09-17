import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np

from staliro.core.result import worst_eval, worst_run
from staliro.options import Options
from staliro.staliro import simulate_model, staliro

from .optimizer_factory import create_optimizer
from .utils import (
    analyze_violation,
    plot_trace_variables,
    save_results_to_csv,
    write_analysis_report,
)


class SpecificationRunner:
    def __init__(self, system_name: str, model, spec_dict: Dict[str, Any], 
                 signals: List[Any], interval: tuple, static_parameters: Optional[List] = None):
        self.system_name = system_name
        self.model = model
        self.spec_dict = spec_dict
        self.signals = signals
        self.interval = interval
        self.static_parameters = static_parameters
        
    def run_analysis(self, args, dimension_descriptions: Optional[List[str]] = None,
                    output_descriptions: Optional[Dict[str, str]] = None):
        """Run the complete analysis pipeline."""
        # Set random seed for reproducible results
        random.seed(args.seed)
        np.random.seed(args.seed)
        
        logging.basicConfig(level=logging.DEBUG)

        # Select specification from command line argument
        spec_name = args.spec
        specification = self.spec_dict[spec_name]
        
        print(f"Analyzing specification: {spec_name}")
        print(f"Formula: {specification.phi}")
        print(f"Variables: {specification.column_map}")
        print(f"Random seed: {args.seed}")
        print("-" * 50)

        # Create optimizer
        optimizer, prompts_filename = create_optimizer(
            args.optimizer, args, spec_name, self.system_name,
            dimension_descriptions, output_descriptions, specification
        )
        
        # Setup options
        options_kwargs = {
            'runs': 1, 
            'iterations': args.max_budget, 
            'interval': self.interval, 
            'signals': self.signals, 
            'seed': args.seed
        }
        
        # Add static_parameters only if provided
        if self.static_parameters is not None:
            options_kwargs['static_parameters'] = self.static_parameters
        
        options = Options(**options_kwargs)
        
        # Start timing measurement (for non-resumable optimizers)
        start_time = time.time()
        
        # Run staliro
        result = staliro(self.model, specification, optimizer, options)

        # Get best sample (lowest robustness value in falsification)
        best_sample = worst_eval(worst_run(result)).sample
        best_result = simulate_model(self.model, options, best_sample)

        # Evaluate robustness
        robustness = specification.evaluate(best_result.trace.states, best_result.trace.times)
        
        # Get execution time - use optimizer's cumulative time if available (for resumable optimizers)
        if hasattr(result.runs[0].result, 'elapsed_time') and result.runs[0].result.elapsed_time > 0:
            # Use the optimizer's cumulative time (for resumable LLM optimizers)
            execution_time = result.runs[0].result.elapsed_time
        else:
            # Fall back to measured time (for non-resumable optimizers)
            execution_time = time.time() - start_time
        
        print(f"Analysis execution time: {execution_time:.2f} seconds")
        
        # Create output directories and save results
        self._save_results(args, spec_name, robustness, result, best_sample, best_result, optimizer, options, prompts_filename, execution_time)
        
        return robustness, best_result
    
    def _save_results(self, args, spec_name: str, robustness: float, result, best_sample, 
                     best_result, optimizer, options, prompts_filename: Optional[str], execution_time: float):
        """Save all results to files."""
        # Check if output folder exists, if not create it
        output_dir = f"./{self.system_name}_all_specs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created {output_dir} folder")
        
        # Save results to CSV file
        csv_filename = f"{output_dir}/results_{args.optimizer}.csv"
        save_results_to_csv(csv_filename, spec_name, args.seed, robustness, result.runs[0].result.nfev, execution_time)
        print(f"Results saved to CSV: {csv_filename}")
        
        # Create optimizer-specific directory
        dir_path = f"{output_dir}/{args.optimizer}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        filename = f"{dir_path}/{spec_name}_{args.optimizer}_seed{args.seed}_report"
        
        # Write analysis report
        txt_filename = filename + ".txt"
        write_analysis_report(txt_filename, spec_name, optimizer, args, options, 
                            result, best_sample, robustness, self.system_name, execution_time)
        
        # Use generalized analysis function (appends to the file)
        analyze_violation(self.spec_dict[spec_name], best_result.trace, robustness, txt_filename)
        
        # Show counterexample sample if violation occurred (console output)
        if robustness < 0:
            print(f"\nCounterexample input sample: {best_sample}")
        
        print(f"\nAnalysis report saved as: {txt_filename}")
        
        # Print prompts file location if using LLM optimizers
        if prompts_filename:
            print(f"LLM prompts saved as: {prompts_filename}")
            
        # Use generalized plotting function
        plot_trace_variables(self.spec_dict[spec_name], best_result.trace, filename + ".jpeg")
