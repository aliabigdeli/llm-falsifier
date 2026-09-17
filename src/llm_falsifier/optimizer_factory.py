from typing import Dict, List, Optional, Tuple, Union


def create_optimizer(
    optimizer_type: str, 
    args, 
    spec_name: str, 
    system_name: str,
    dimension_descriptions: Optional[List[str]] = None,
    output_descriptions: Optional[Dict[str, str]] = None,
    specification = None
) -> Tuple[Union["DualAnnealing", "LLMOptimizer", "LLMGrayBoxOpt", "DifferentialEvolution", "PSO", "BasinHopping", "CMAES", "UniformRandom"], Optional[str]]:
    """Factory function to create optimizers based on type."""
    # Import here to avoid circular import
    from .optimizers import (
        DualAnnealing,
        UniformRandom,
        CMAES,
        PSO,
        BasinHopping,
        DifferentialEvolution,
        LLMGrayBoxOpt,
        LLMOptimizer,
    )
    
    if optimizer_type == "LLM":
        # Create prompts filename based on specification and seed
        prompts_filename = f"./{system_name}_all_specs/{optimizer_type}/{spec_name}_LLM_seed{args.seed}_prompts.txt"
        state_filename = None # f"./{system_name}_all_specs/{optimizer_type}/{spec_name}_LLM_seed{args.seed}_state.json"
        
        optimizer = LLMOptimizer(
            model_name=args.llm_model,
            max_history=args.llm_history,
            save_prompts=True,
            prompt_file=prompts_filename,
            reasoning_effort=args.reasoning_effort,
            exemplars_order=args.exemplars_order,
            state_file=state_filename,
        )
        return optimizer, prompts_filename
        
    elif optimizer_type == "LLMGB":
        if not dimension_descriptions or not specification:
            raise ValueError("LLMGB optimizer requires dimension_descriptions and specification")
        
        # Create prompts filename based on specification and seed
        prompts_filename = f"./{system_name}_all_specs/{optimizer_type}/{spec_name}_LLMGB_seed{args.seed}_prompts.txt"
        state_filename = None # f"./{system_name}_all_specs/{optimizer_type}/{spec_name}_LLMGB_seed{args.seed}_state.json"
        
        optimizer = LLMGrayBoxOpt(
            dimension_descriptions=dimension_descriptions,
            specification=specification,
            output_descriptions=output_descriptions,
            model_name=args.llm_model,
            max_history=args.llm_history,
            temperature=1.0,
            save_prompts=True,
            prompt_file=prompts_filename,
            output_time_selection=args.output_time_selection,
            reasoning_effort=args.reasoning_effort,
            include_stl_translation=args.include_stl_trans,
            include_critical_time=args.include_critical_time,
            exemplars_order=args.exemplars_order,
            state_file=state_filename
        )
        return optimizer, prompts_filename
        
    elif optimizer_type == "DE":
        optimizer = DifferentialEvolution(
            strategy='best1bin',    # Good balanced strategy
            popsize=20,             # Slightly larger population for better exploration
            mutation=(0.5, 1.2),    # Slightly wider mutation range
            recombination=0.7,      # Standard crossover rate
            polish=True             # Local refinement for better solutions
        )
        return optimizer, None
        
    elif optimizer_type == "PSO":
        optimizer = PSO(
            swarm_size=30,          # Good balance of exploration and computational cost
            inertia=0.9,            # High inertia for exploration
            cognitive=2.0,          # Standard cognitive weight
            social=2.0,             # Standard social weight
            adaptive_inertia=True   # Reduces inertia over time for better convergence
        )
        return optimizer, None
        
    elif optimizer_type == "BH":
        optimizer = BasinHopping(
            niter=100,              # Number of basin hopping iterations
            T=1.0,                  # Temperature for accepting jumps
            stepsize=0.5            # Step size for random jumps
        )
        return optimizer, None
        
    elif optimizer_type == "CMAES":
        optimizer = CMAES(
            sigma0=0.3,             # Initial standard deviation
            popsize=None,           # Use default CMA-ES population sizing
            maxiter_factor=50       # Conservative iteration factor
        )
        return optimizer, None
        
    elif optimizer_type == "UR":
        optimizer = UniformRandom()
        return optimizer, None
        
    else:  # Default to DualAnnealing
        optimizer = DualAnnealing()
        return optimizer, None
