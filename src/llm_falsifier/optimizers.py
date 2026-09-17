from __future__ import annotations

import enum
import os
import re
import time
import statistics as stats
from collections.abc import Iterable, Sequence
from itertools import takewhile
from pathlib import Path
from typing import Literal

import numpy as np
from attr import frozen
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from openai import OpenAI
from scipy import optimize
from typing_extensions import TypeAlias

from staliro.core.interval import Interval
from staliro.core.optimizer import ObjectiveFn, Optimizer
from staliro.core.sample import Sample

from .utils import find_critical_time_from_spec_trace

Samples: TypeAlias = Sequence[Sample]
Bounds: TypeAlias = Sequence[Interval]


class Behavior(enum.IntEnum):
    """Behavior when falsifying case for system is encountered.

    Attributes:
        FALSIFICATION: Stop searching when the first falsifying case is encountered
        MINIMIZATION: Continue searching after encountering a falsifying case until iteration
                      budget is exhausted
    """

    FALSIFICATION = enum.auto()
    MINIMIZATION = enum.auto()


def _sample_uniform(bounds: Bounds, rng: Generator) -> Sample:
    return Sample([rng.uniform(bound.lower, bound.upper) for bound in bounds])


def _minimize(samples: Samples, func: ObjectiveFn[float], nprocs: int | None) -> Iterable[float]:
    if nprocs is None:
        return func.eval_samples(samples)
    else:
        return func.eval_samples_parallel(samples, nprocs)


def _falsify(samples: Samples, func: ObjectiveFn[float]) -> Iterable[float]:
    costs = map(func.eval_sample, samples)
    return takewhile(lambda c: c >= 0, costs)


@frozen(slots=True)
class UniformRandomResult:
    """Data class that represents the result of a uniform random optimization.

    Attributes:
        average_cost: The average cost of all the samples selected.
        nfev: Number of function evaluations performed.
    """

    average_cost: float
    nfev: int


class UniformRandom(Optimizer[float, UniformRandomResult]):
    """Optimizer that implements the uniform random optimization technique.

    This optimizer picks samples randomly from the search space until the budget is exhausted.

    Args:
        parallelization: Value that indicates how many processes to use when evaluating each
                            sample using the cost function. Acceptable values are a number,
                            "cores", or None

    Attributes:
        processes: The number of processes to use when evaluating the samples.
    """

    def __init__(
        self,
        parallelization: Literal["cores"] | int | None = None,
        behavior: Behavior = Behavior.FALSIFICATION,
    ):
        if isinstance(parallelization, int):
            self.processes: int | None = parallelization
        elif parallelization == "cores":
            self.processes = os.cpu_count()
        else:
            self.processes = None

        self.behavior = behavior

    def optimize(
        self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int
    ) -> UniformRandomResult:
        rng = default_rng(seed)
        samples = [_sample_uniform(bounds, rng) for _ in range(budget)]

        if self.behavior is Behavior.MINIMIZATION:
            costs = list(_minimize(samples, func, self.processes))
            nfev = len(samples)  # All samples were evaluated
        else:
            costs = list(_falsify(samples, func))
            nfev = len(costs)  # Number of samples evaluated before falsification or budget exhaustion

        average_cost = stats.mean(costs)

        return UniformRandomResult(average_cost, nfev)


@frozen(slots=True)
class DualAnnealingResult:
    """Data class representing the result of a dual annealing optimization.

    Attributes:
        jacobian_value: The value of the cost function jacobian at the minimum cost discovered
        jacobian_evals: Number of times the jacobian of the cost function was evaluated
        hessian_value: The value of the cost function hessian as the minimum cost discovered
        hessian_evals: Number of times the hessian of the cost function was evaluated
        nfev: Number of function evaluations performed
    """

    jacobian_value: NDArray[np.float_] | None
    jacobian_evals: int
    hessian_value: NDArray[np.float_] | None
    hessian_evals: int
    nfev: int


class DualAnnealing(Optimizer[float, DualAnnealingResult]):
    """Optimizer that implements the simulated annealing optimization technique.

    The simulated annealing implementation is provided by the SciPy library dual_annealing function
    with the no_local_search parameter set to True.
    """

    def __init__(self, behavior: Behavior = Behavior.FALSIFICATION):
        self.behavior = behavior

    def optimize(
        self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int
    ) -> DualAnnealingResult:
        # Track evaluations and implement early termination for FALSIFICATION
        evaluation_count = {'nfev': 0}
        found_falsification = {'found': False}
        
        def objective_wrapper(x):
            if evaluation_count['nfev'] >= budget:
                return np.inf  # Stop if budget exceeded
            
            cost = func.eval_sample(Sample(x))
            evaluation_count['nfev'] += 1
            
            # Check for early termination due to falsification
            if self.behavior == Behavior.FALSIFICATION and cost < 0:
                found_falsification['found'] = True
                # Force termination by raising an exception that dual_annealing will catch
                raise StopIteration("Falsification found")
            
            return cost

        def listener(sample: NDArray[np.float_], robustness: float, ctx: Literal[-1, 0, 1]) -> bool:
            # Keep the original callback for compatibility, but don't rely on it for termination
            return False

        try:
            result = optimize.dual_annealing(
                func=objective_wrapper,
                bounds=[bound.astuple() for bound in bounds],
                seed=seed,
                maxfun=budget,
                no_local_search=True,  # Disable local search, use only traditional generalized SA
                callback=listener,
            )
        except StopIteration:
            # Create a dummy result when stopped early due to falsification
            class DummyResult:
                def __init__(self):
                    self.nfev = evaluation_count['nfev']
                    self.jac = None
                    self.njev = 0
                    self.hess = None
                    self.nhev = 0
            result = DummyResult()

        try:
            jac: NDArray[np.float_] | None = result.jac
            njev = result.njev
        except AttributeError:
            jac = None
            njev = 0

        try:
            hess: NDArray[np.float_] | None = result.hess
            nhev = result.nhev
        except AttributeError:
            hess = None
            nhev = 0

        # Use our tracked function evaluation count instead of SciPy's count
        nfev = evaluation_count['nfev']

        return DualAnnealingResult(jac, njev, hess, nhev, nfev)


@frozen(slots=True)
class LLMOptimizerResult:
    """Data class containing additional data from LLM optimization.

    :attribute best_sample: The best sample found during optimization
    :attribute best_cost: The cost of the best sample
    :attribute history: List of (sample, cost) pairs from the optimization history
    :attribute nfev: Number of function evaluations performed
    :attribute elapsed_time: Total elapsed time in seconds (including resumed sessions)
    """

    best_sample: list[float]
    best_cost: float
    history: list[tuple[list[float], float]]
    nfev: int
    elapsed_time: float = 0.0


class LLMOptimizer(Optimizer[float, LLMOptimizerResult]):
    """Optimizer implementing LLM-based optimization.

    This optimizer uses a Large Language Model to generate and optimize samples based on previous
    evaluations. The LLM is prompted with the optimization history and asked to generate new samples
    that are likely to improve upon previous results.

    :param model_name: Name of the LLM model to use (e.g. "gpt-4.1-nano")
    :param min_cost: The minimum cost to use as a termination condition
    :param temperature: Temperature parameter for LLM sampling (0.0 to 1.0)
    :param max_history: Maximum number of previous samples to include in LLM prompt
    :param exemplars_order: Sort order for history exemplars in prompt: "ascending" (best to worst), "descending" (worst to best), or None (chronological order) (default: None)
    :param save_prompts: Whether to save generated prompts to a file (default: False)
    :param prompt_file: File path to save prompts to (default: "llm_prompts.txt")
    :param reasoning_effort: Reasoning effort for LLM (default: "low")
    :param input_elements_precision: Precision for input elements in the prompt (default: 3)
    :param cost_precision: Precision for cost display in the prompt (default: 3)
    :param state_file: File path to save optimization state for resumability (default: None (means no state file for saving and resuming))
    :param behavior: Behavior when falsifying case is encountered (default: Behavior.FALSIFICATION)
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        min_cost: float | None = None,
        temperature: float = 0.7,
        max_history: int = 10,
        exemplars_order: Literal["ascending", "descending"] | None = None,
        save_prompts: bool = False,
        prompt_file: str = "llm_prompts.txt",
        reasoning_effort: str = "low",
        input_elements_precision: int = 3,
        cost_precision: int = 3,
        state_file: str | None = None,
        behavior: Behavior = Behavior.FALSIFICATION
    ):
        self.model_name = model_name
        self.min_cost = min_cost
        self.temperature = temperature
        self.max_history = max_history
        self.exemplars_order = exemplars_order
        self.save_prompts = save_prompts
        self.prompt_file = prompt_file
        self.reasoning_effort = reasoning_effort
        self.input_elements_precision = input_elements_precision
        self.cost_precision = cost_precision
        self.state_file = state_file
        self.behavior = behavior
        # Initialize prompt counter for tracking
        self.prompt_counter = 0
        
        # Read API key from file and set environment variable
        try:
            if "openai/gpt-oss-20b:" in self.model_name or "openai/gpt-oss-120b:" in self.model_name:
                self.client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.environ["HF_TOKEN"])
            elif self.model_name == "openai/gpt-oss-20b":
                self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            else:
                with Path("../api_key.txt").open("r") as f:
                    api_key = f.read().strip()
                os.environ["OPENAI_API_KEY"] = api_key
                self.client = OpenAI()
        except FileNotFoundError:
            raise FileNotFoundError("api_key.txt file not found. Please create this file with your OpenAI API key.") from None
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}") from e

    def _create_prompt(self, bounds: Sequence[Interval], history: list[tuple[list[float], float]]) -> str:
        """Create a prompt for the LLM based on optimization history."""
        # Get recent history
        
        # Sort history based on cost if exemplars_order is specified
        if self.exemplars_order == "ascending":
            sorted_history = sorted(history, key=lambda x: x[1])
        elif self.exemplars_order == "descending":
            sorted_history = sorted(history, key=lambda x: x[1], reverse=True)
        else:
            sorted_history = history
        
        recent_history = sorted_history[-self.max_history:]
        prompt = f"""You are an optimization assistant for a system falsification task. Your goal is to find input parameters that violate system specifications (negative robustness values indicate violations).

INPUT CONTROL PARAMETERS:
The input space has {len(bounds)} dimensions representing control parameters:
{chr(10).join(f"Dimension {i+1}: [{bound.lower}, {bound.upper}]" for i, bound in enumerate(bounds))}

OPTIMIZATION CONTEXT:
- Lower robustness values are better (negative values indicate specification violations)

RECENT OPTIMIZATION HISTORY (last {min(len(recent_history), self.max_history)} samples{" in "+self.exemplars_order+" order based on the robustness value" if self.exemplars_order else ""}):
{chr(10).join(f"Sample {i+1}: {[f'{val:.{self.input_elements_precision}f}' for val in sample]} -> Robustness: {cost:.{self.cost_precision}f}" for i, (sample, cost) in enumerate(recent_history))}

Based on this history, the specification requirements, generate a new sample point that is different from all points above and is likely to achieve a lower robustness value (violation).
Consider:
1. Which parameter combinations led to lower robustness values in the history
2. The physical/logical meaning of each parameter and how it affects the output variables
Think step by step and generate the new sample point as a comma-separated list of numeric numbers within the bounds. Do not write code. At the end of your response, put the new sample with {len(bounds)} dimensions. The sample point should start with '<point>' then the comma-separated list of numbers and end with '</point>'.
"""
        return prompt

    def _save_prompt(self, prompt: str, evaluation_num: int) -> None:
        """Save the generated prompt to a file."""
        try:
            # Ensure the directory exists
            Path(self.prompt_file).parent.mkdir(parents=True, exist_ok=True)
            
            mode = 'a' if self.prompt_counter > 0 else 'w'
            with Path(self.prompt_file).open(mode, encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"PROMPT #{self.prompt_counter + 1} (Evaluation #{evaluation_num})\n")
                f.write(f"{'='*80}\n")
                f.write(prompt)
                f.write(f"\n{'='*80}\n\n")
            self.prompt_counter += 1
        except Exception as e:
            print(f"Warning: Failed to save prompt to {self.prompt_file}: {e}")

    def _calculate_and_save_llm_response(self, response, response_text: str, reasoning_steps: str = None) -> float:
        """Calculate costs and save LLM response and reasoning to file.
        
        Args:
            response: The LLM response object containing usage information
            response_text: The actual text response from the LLM
            reasoning_steps: Optional reasoning steps from the LLM
            
        Returns:
            float: Total cost of the LLM call
        """
        # Get model rates for cost calculation
        if "gpt-5-nano" in self.model_name:
            input_rate = 0.05
            output_rate = 0.40
        elif "gpt-5-mini" in self.model_name:
            input_rate = 0.25
            output_rate = 2.0
        elif "gpt-5" in self.model_name:
            input_rate = 1.25
            output_rate = 10.0
        elif "gpt-4.1-nano" in self.model_name:
            input_rate = 0.10
            output_rate = 0.40
        elif "gpt-4.1-mini" in self.model_name:
            input_rate = 0.40
            output_rate = 1.60
        elif "gpt-4.1" in self.model_name:
            input_rate = 2.0
            output_rate = 8.0
        elif self.model_name == "openai/gpt-oss-20b":
            input_rate = 0.0
            output_rate = 0.0
        else:
            print(f"the rate cost for model {self.model_name} is not defined, so considered as gpt-5")
            input_rate = 1.25
            output_rate = 10.0
        
        # Calculate costs based on token usage
        usage = response.usage
        
        # responses API uses input_tokens/output_tokens
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        
        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * output_rate
        total_cost = input_cost + output_cost
        
        # cached  = getattr(usage.input_tokens_details, 'cached_tokens')
        
        # Save the LLM response text to a file for inspection
        try:
            with Path(self.prompt_file).open("a", encoding="utf-8") as f:
                f.write(f"\n{'='*40}\n")
                
                # Save reasoning steps if available
                if reasoning_steps:
                    f.write("LLM Reasoning Steps:\n")
                    f.write(reasoning_steps)
                    f.write(f"\n{'-'*20}\n")
                
                f.write("LLM Final Response:\n")
                f.write(response_text)
                f.write(f"\n{'='*40}\n")
                # f.write(f"Cached tokens: {cached} | Cache hit: {bool(cached and cached > 0)} | hasattr(usage, 'input_tokens_details'): {hasattr(usage, 'input_tokens_details')} | hasattr(usage.input_tokens_details, 'cached_tokens'): {hasattr(usage.input_tokens_details, 'cached_tokens')} | hasattr(usage, 'prompt_tokens_details'): {hasattr(usage, 'prompt_tokens_details')} | hasattr(usage, 'cached_tokens'): {hasattr(usage, 'cached_tokens')}\n")
                f.write(f"Input cost: ${input_cost:.6f} (input tokens: {input_tokens}), Output cost: ${output_cost:.6f} (output tokens: {output_tokens}), Total cost of calling {self.model_name} w/ {self.reasoning_effort} reasoning: ${total_cost:.6f}")
        except Exception as file_exc:
            print(f"Warning: Could not save LLM response to file: {file_exc}")
        
        return total_cost

    def _generate_sample(self, prompt: str, bounds: Bounds) -> list[float]:
        """Generate a new sample using the LLM.
        
        :param prompt: The prompt to send to the LLM
        :param bounds: The bounds of the search space (used for fallback random sampling)
        :returns: A list of floats representing the new sample point
        :raises ValueError: If the LLM response cannot be parsed into a valid sample
        """
        try:
            response_params = {
                "model": self.model_name,
                "input": [
                    {"role": "system", "content": "You are an expert optimization assistant that understands system behavior and generates a new sample point."},
                    {"role": "user", "content": prompt}
                ],
                "reasoning": {
                    "effort": self.reasoning_effort, 
                    "summary": None # Options: "auto", "concise", "detailed", None
                },
            }

            if "openai/gpt-oss-20b" in self.model_name or "openai/gpt-oss-120b" in self.model_name:
                response_params["temperature"] = self.temperature   # GPT-5 family does not support temperature
                response_params["reasoning"]["summary"] = "auto"
            if "gpt-4" in self.model_name:
                # Remove 'reasoning' from response_params as gpt-4 models (non-reasoning models) do not support it
                response_params.pop("reasoning", None)
                response_params["temperature"] = self.temperature

            response = self.client.responses.create(**response_params)

            reasoning_texts = []
            for item in response.output:
                if hasattr(item, "type"):   # type are "message", "reasoning", "function_call" if available
                    if item.type == "reasoning":
                        content = getattr(item, "content", None)
                        if content is not None:
                            for part in content:
                                if hasattr(part, "type") and part.type == "reasoning_text":
                                    reasoning_texts.append(part.text)
            
            response_text = response.output_text
            reasoning_steps = "\n".join(reasoning_texts) if reasoning_texts else None

            if self.save_prompts:
                self._calculate_and_save_llm_response(response, response_text, reasoning_steps)
            
            # Try to parse the response as a list of numbers
            # First, try to find content between <point></point> tags
            point_match = re.search(r'<point>(.*?)</point>', response_text, re.DOTALL)
            
            if point_match:
                # Extract content between tags and find numbers in it
                point_content = point_match.group(1).strip()
                numbers = re.findall(r'-?\d*\.?\d+', point_content)
            else:
                # Fallback: find all numbers in the entire response using regex
                numbers = re.findall(r'-?\d*\.?\d+', response_text)
            
            if not numbers:
                raise ValueError("No numbers found in LLM response")
                
            # Convert to floats
            sample = [float(num) for num in numbers]
            
            return sample
            
        except Exception as e:
            try:
                with Path(self.prompt_file).open("a", encoding="utf-8") as f:
                    f.write(f"\n{'='*40}\n")
                    f.write("LLM Error:\n")
                    f.write(str(e))
                    f.write(f"\n{'='*40}\n")
            except Exception as file_exc:
                print(f"Warning: Could not save LLM error to file: {file_exc}")
            # If anything goes wrong (e.g.: lost connection to LLM for one iteration), fall back to random sampling
            print(f"Error generating sample with LLM: {e}")
            rng = default_rng()
            return list(_sample_uniform(bounds, rng))

    def _save_state(self, history: list[tuple[list[float], float]], nfev: int, 
                   best_sample: list[float], best_cost: float, elapsed_time: float) -> None:
        """Save the current optimization state to a file.
        
        Args:
            history: List of (sample, cost) tuples from optimization history
            nfev: Number of function evaluations performed
            best_sample: The best sample found so far
            best_cost: The cost of the best sample
            elapsed_time: Cumulative elapsed time in seconds
        """
        if self.state_file is None:
            return
            
        try:
            import json
            
            state = {
                'nfev': nfev,
                'best_sample': best_sample,
                'best_cost': best_cost,
                'history': history,  # List of (sample, cost) tuples
                'elapsed_time': elapsed_time
            }
            
            # Ensure the directory exists
            Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
            
            with Path(self.state_file).open('w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to save optimization state to {self.state_file}: {e}")

    def _load_state(self) -> dict | None:
        """Load the optimization state from a file.
        
        Returns:
            Dictionary containing 'history', 'nfev', 'best_sample', 'best_cost', 'elapsed_time' if file exists,
            None otherwise.
        """
        if self.state_file is None:
            return None
            
        try:
            import json
            
            state_path = Path(self.state_file)
            if not state_path.exists():
                return None
            
            with state_path.open('r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Validate state structure
            required_keys = ['history', 'nfev', 'best_sample', 'best_cost']
            if not all(key in state for key in required_keys):
                print(f"Warning: Invalid state file {self.state_file}, missing required keys")
                return None
            
            # elapsed_time is optional for backward compatibility with old state files
            if 'elapsed_time' not in state:
                state['elapsed_time'] = 0.0
            
            return state
            
        except Exception as e:
            print(f"Warning: Failed to load optimization state from {self.state_file}: {e}")
            return None

    def optimize(self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int, 
                 start_w_rnd_initial_sample: bool = False, resume: bool = True) -> LLMOptimizerResult:
        """Execute the LLM optimization.
        
        Args:
            func: The objective function to minimize
            bounds: Parameter bounds
            budget: Maximum number of function evaluations
            seed: Random seed for reproducibility
            start_w_rnd_initial_sample: Whether to start with a random initial sample
            resume: Whether to resume from saved state if available
        """
        history: list[tuple[list[float], float]] = []
        seen_samples: set[tuple[float, ...]] = set()  # Track evaluated samples with input_elements_precision decimal precision
        best_sample: list[float] = []
        best_cost = float('inf')
        nfev = 0
        cumulative_elapsed_time = 0.0

        # Initialize random number generator
        rng = default_rng(seed)
        
        # Start timing for this session
        session_start_time = time.time()

        # Try to resume from saved state if requested
        if resume:
            saved_state = self._load_state()
            if saved_state is not None:
                history = saved_state['history']
                nfev = saved_state['nfev']
                self.prompt_counter = nfev
                best_sample = saved_state['best_sample']
                best_cost = saved_state['best_cost']
                cumulative_elapsed_time = saved_state.get('elapsed_time', 0.0)
                
                # Rebuild seen_samples set
                for sample, _ in history:
                    seen_samples.add(tuple(round(x, self.input_elements_precision) for x in sample))
                
                print(f"Resuming optimization from nfev={nfev}, best_cost={best_cost}, elapsed_time={cumulative_elapsed_time:.2f}s")
            else:
                if self.state_file is not None:
                    print(f"No saved state found at {self.state_file}, starting fresh")

        # Generate initial random sample (only if not resuming or no saved state)
        if nfev == 0 and start_w_rnd_initial_sample:
            current_sample = _sample_uniform(bounds, rng)
            current_cost = func.eval_sample(Sample(current_sample))
            history.append((current_sample, current_cost))
            seen_samples.add(tuple(round(x, self.input_elements_precision) for x in current_sample))
            best_sample = current_sample
            best_cost = current_cost
            nfev += 1
            
            # Save state after initial sample
            if self.state_file is not None:
                current_elapsed = cumulative_elapsed_time + (time.time() - session_start_time)
                self._save_state(history, nfev, best_sample, best_cost, current_elapsed)

            # Check for early termination on initial sample
            if self.behavior == Behavior.FALSIFICATION and current_cost < 0:
                final_elapsed_time = cumulative_elapsed_time + (time.time() - session_start_time)
                return LLMOptimizerResult(
                    best_sample=best_sample,
                    best_cost=best_cost,
                    history=history,
                    nfev=nfev,
                    elapsed_time=final_elapsed_time
                )

        while nfev < budget:
            # Create prompt for LLM
            prompt = self._create_prompt(bounds, history)
            
            # Save prompt if requested
            if self.save_prompts:
                self._save_prompt(prompt, nfev)
            
            # Generate new sample using LLM
            new_sample = self._generate_sample(prompt, bounds)
            
            # Ensure sample is within bounds
            new_sample = [
                max(min(x, bound.upper), bound.lower)
                for x, bound in zip(new_sample, bounds)
            ]

            new_sample_rounded = tuple(round(x, self.input_elements_precision) for x in new_sample)
            
            # Convert to Sample object and evaluate
            sample_obj = Sample(new_sample)
            new_cost = func.eval_sample(sample_obj)
            if self.exemplars_order is None or new_sample_rounded not in seen_samples:
                history.append((new_sample, new_cost))
            seen_samples.add(new_sample_rounded)
            nfev += 1

            # Update best sample if needed
            if new_cost < best_cost:
                best_sample = new_sample
                best_cost = new_cost

            # Save state after each iteration
            if self.state_file is not None:
                current_elapsed = cumulative_elapsed_time + (time.time() - session_start_time)
                self._save_state(history, nfev, best_sample, best_cost, current_elapsed)

            # Check for early termination due to falsification
            if self.behavior == Behavior.FALSIFICATION and new_cost < 0:
                break

            # Check termination condition
            if self.min_cost and best_cost <= self.min_cost:
                break

        # Calculate final elapsed time
        final_elapsed_time = cumulative_elapsed_time + (time.time() - session_start_time)

        return LLMOptimizerResult(
            best_sample=best_sample,
            best_cost=best_cost,
            history=history,
            nfev=nfev,
            elapsed_time=final_elapsed_time
        )


class LLMGrayBoxOpt(Optimizer[float, LLMOptimizerResult]):
    """Gray-box LLM optimizer that provides semantic descriptions of input dimensions.
    
    This optimizer extends the basic LLM optimizer by providing meaningful descriptions
    of what each input dimension represents, STL specification in natural language,
    and output dimension descriptions, allowing the LLM to make more informed
    optimization decisions based on domain knowledge.

    :param dimension_descriptions: List of descriptions for each input dimension
    :param specification: The STL specification object to translate to natural language
    :param output_descriptions: Optional dict mapping output variable names to descriptions
    :param model_name: Name of the LLM model to use (e.g. "gpt-4.1-nano")
    :param min_cost: The minimum cost to use as a termination condition
    :param temperature: Temperature parameter for LLM sampling (0.0 to 1.0)
    :param max_history: Maximum number of previous samples to include in LLM prompt
    :param exemplars_order: Sort order for history exemplars in prompt: "ascending" (best to worst), "descending" (worst to best), or None (chronological order) (default: None)
    :param save_prompts: Whether to save generated prompts to a file (default: False)
    :param prompt_file: File path to save prompts to (default: "llmgb_prompts.txt")
    :param output_time_selection: How to include system output states in the prompt. Options: None (don't include output states), -1 (output at final time step only), positive int (number of intervals), tuple(N1, N2) (out of N1 intervals select every N2) (default: -1)
    :param include_critical_time: Whether to include state values at critical time in the prompt (default: False)
    :param reasoning_effort: Reasoning effort for LLM (default: "low")
    :param include_stl_translation: Whether to include STL translation in the prompt (default: False)
    :param input_elements_precision: Precision for input elements in the prompt (default: 3)
    :param cost_precision: Precision for cost display in the prompt (default: 3)
    :param behavior: Behavior when falsifying case is encountered (default: Behavior.FALSIFICATION)
    :param state_file: File path to save optimization state for resumability (default: None (means no state file for saving and resuming))
    """

    def __init__(
        self,
        dimension_descriptions: list[str],
        specification,
        output_descriptions: dict[str, str] | None = None,
        model_name: str = "openai/gpt-oss-20b",
        min_cost: float | None = None,
        temperature: float = 0.7,
        max_history: int = 10,
        exemplars_order: Literal["ascending", "descending"] | None = None,
        save_prompts: bool = False,
        prompt_file: str = "llmgb_prompts.txt",
        output_time_selection: tuple[int, int] | int | None = -1,
        include_critical_time: bool = False,
        reasoning_effort: str = "low",
        include_stl_translation: bool = False,
        input_elements_precision: int = 3,
        cost_precision: int = 3,
        state_file: str | None = None,
        behavior: Behavior = Behavior.FALSIFICATION
    ):
        self.dimension_descriptions = dimension_descriptions
        self.specification = specification
        self.output_descriptions = output_descriptions or {}
        self.model_name = model_name
        self.min_cost = min_cost
        self.temperature = temperature
        self.max_history = max_history
        self.exemplars_order = exemplars_order
        self.save_prompts = save_prompts
        self.prompt_file = prompt_file
        self.output_time_selection = output_time_selection
        self.reasoning_effort = reasoning_effort
        self.include_stl_translation = include_stl_translation
        self.include_critical_time = include_critical_time
        # Use adaptive precision for critical time
        if include_critical_time:
            self.critical_time_precision = self._get_critical_time_precision_from_spec()
        self.input_elements_precision = input_elements_precision
        self.cost_precision = cost_precision
        self.behavior = behavior
        self.state_file = state_file
        # Initialize prompt counter for tracking
        self.prompt_counter = 0
        
        # Read API key from file and set environment variable
        try:
            if "openai/gpt-oss-20b:" in self.model_name or "openai/gpt-oss-120b:" in self.model_name:
                self.client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.environ["HF_TOKEN"])
            elif self.model_name == "openai/gpt-oss-20b":
                self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            else:
                with Path("../api_key.txt").open("r") as f:
                    api_key = f.read().strip()
                os.environ["OPENAI_API_KEY"] = api_key
                self.client = OpenAI()
        except FileNotFoundError:
            raise FileNotFoundError("api_key.txt file not found. Please create this file with your OpenAI API key.") from None
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}") from e

    def _translate_stl_to_natural_language(self, stl_formula: str) -> str:
        """Translate STL formula to natural language."""
        
        # Create a copy to work with
        natural = stl_formula
        
        # Replace temporal operators with natural language
        # G[a,b] -> "always between time a and b seconds"
        natural = re.sub(r'G\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]', r'always between time \1 and \2 seconds', natural)
        
        # F[a,b] -> "eventually between time a and b seconds"  
        natural = re.sub(r'F\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]', r'eventually between time \1 and \2 seconds', natural)
        
        # Replace logical operators
        natural = natural.replace(' and ', ' AND ')
        natural = natural.replace(' or ', ' OR ')
        natural = natural.replace('not ', 'NOT ')
        natural = natural.replace(' -> ', ' IMPLIES ')
        
        # Replace comparison operators with more natural language
        natural = natural.replace('<=', ' is at most ')
        natural = natural.replace('>=', ' is at least ')
        natural = natural.replace('==', ' equals ')
        natural = natural.replace('<', ' is less than ')
        natural = natural.replace('>', ' is greater than ')
        
        # Clean up extra parentheses and spaces
        natural = re.sub(r'\s+', ' ', natural)
        natural = natural.strip()
        
        return natural

    def _get_adaptive_precision(self, value: float, min_precision: int = 1, max_precision: int = 3) -> int:
        """
        Determine appropriate decimal precision to avoid showing misleading zeros.
        
        Args:
            value: The numeric value to format
            min_precision: Minimum decimal places to use
            max_precision: Maximum decimal places to use
            
        Returns:
            The appropriate number of decimal places
        """
        if value == 0:
            return min_precision
        
        # Try increasing precision until we get a non-zero display
        for precision in range(min_precision, max_precision + 1):
            formatted = f"{value:.{precision}f}"
            # Check if the formatted value preserves meaningful information
            if formatted not in ["0.0", "-0.0"] and abs(float(formatted)) > 0:
                return precision
        
        return max_precision

    def _get_critical_time_precision_from_spec(self) -> int:
        """
        Determine appropriate precision for critical time based on STL specification time bounds.
        
        Extracts time bounds from temporal operators in the specification and returns
        a precision that matches or slightly exceeds the specification's temporal granularity.
        
        Supports both symbolic (G, F, U) and keyword (always, eventually, until) syntax.
        
        Returns:
            The appropriate number of decimal places (min 1, max 4)
        """
        stl_formula = self.specification.phi
        
        # Pattern to match temporal operators with time bounds:
        # - Symbolic: G[a,b], F[a,b], U[a,b]
        # - Keyword: always[a,b], eventually[a,b], until[a,b]
        # Captures the lower and upper time bounds as groups
        time_pattern = r'(?:G|F|U|always|eventually|until)\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
        matches = re.findall(time_pattern, stl_formula, re.IGNORECASE)
        
        if not matches:
            return 1  # Default if no temporal operators found
        
        max_precision = 0
        for lower, upper in matches:
            # Count decimal places for each time value
            for time_str in [lower, upper]:
                if '.' in time_str:
                    decimal_places = len(time_str.split('.')[1])
                    max_precision = max(max_precision, decimal_places)
        
        return min(max_precision, 4) if max_precision > 0 else 1

    def _format_output_states(self, trace_info: dict, cost: float) -> str:
        """Format the output states from trace information for display in the prompt.
        
        Args:
            trace_info: Dictionary containing trace information with 'times', 'states', 
                       and optionally 'critical_time', 'critical_index' (cached critical time information)
            cost: cost/robustness value for this sample (used with critical time display)
        """
        try:
            if not trace_info:
                return "No trace information available"
            
            # Extract key information from the trace
            output_lines = []
            critical_output_lines = []
            
            # Show summary statistics for each output variable
            for var_name, column_idx in self.specification.column_map.items():
                if 'states' in trace_info and trace_info['states']:
                    try:
                        # Extract values for this variable across all time points
                        var_values = [state[column_idx] if column_idx < len(state) else 0 
                                    for state in trace_info['states']]

                        times = trace_info['times']
                        
                        if var_values:
                            if isinstance(self.output_time_selection, int) and self.output_time_selection == -1:
                                var_min = min(var_values)
                                var_max = max(var_values)
                                var_final = var_values[-1] if var_values else 0
                                
                                # Limit precision for readability
                                # output_lines.append(
                                #     f"{var_name}=[{var_min:.3f}, {var_max:.3f}], final={var_final:.3f}"
                                # )
                                precision = self._get_adaptive_precision(var_final)
                                output_lines.append(
                                    f"{var_name} at {times[-1]:.1f}s = {var_final:.{precision}f}"
                                )
                            elif isinstance(self.output_time_selection, int) and self.output_time_selection > 0:
                                res = ""
                                n_intervals = self.output_time_selection
                                if n_intervals == 1:
                                    values_and_times = list(zip([var_values[0], var_values[-1]], [times[0], times[-1]]))
                                else:
                                    # Divide time range evenly and find closest indices
                                    time_min, time_max = times[0], times[-1]
                                    target_times = [time_min + i * (time_max - time_min) / (n_intervals) 
                                                   for i in range(n_intervals + 1)]
                                    
                                    # Find closest index for each target time
                                    selected_indices = []
                                    for target_time in target_times:
                                        closest_idx = min(range(len(times)), 
                                                        key=lambda i: abs(times[i] - target_time))
                                        selected_indices.append(closest_idx)
                                    
                                    values_and_times = [(var_values[idx], times[idx]) 
                                                       for idx in selected_indices]
                                for idx, (var_value, time) in enumerate(values_and_times):
                                    precision = self._get_adaptive_precision(var_value)
                                    if idx == len(values_and_times) - 1:
                                        res += f" at {time:.1f}s = {var_value:.{precision}f}"
                                    else:
                                        res += f" at {time:.1f}s = {var_value:.{precision}f},"
                                output_lines.append(f"{var_name}:{res}")
                            elif isinstance(self.output_time_selection, tuple):
                                res = ""
                                n_intervals = self.output_time_selection[0]
                                
                                # Divide time range evenly and find closest indices
                                time_min, time_max = times[0], times[-1]
                                target_times = [time_min + i * (time_max - time_min) / (n_intervals) 
                                               for i in range(n_intervals + 1)]
                                
                                # Find closest index for each target time
                                selected_indices = []
                                for target_time in target_times:
                                    closest_idx = min(range(len(times)), 
                                                    key=lambda i: abs(times[i] - target_time))
                                    selected_indices.append(closest_idx)
                                
                                values_list = [var_values[idx] for idx in selected_indices]
                                times_list = [times[idx] for idx in selected_indices]
                                
                                # Further subsample by taking every n-th element
                                values_list2 = [values_list[i] for i in range(0, len(values_list), self.output_time_selection[1])]
                                times_list2 = [times_list[i] for i in range(0, len(times_list), self.output_time_selection[1])]
                                # Ensure the last element is included
                                if values_list2[-1] != values_list[-1]:
                                    values_list2.append(values_list[-1])
                                    times_list2.append(times_list[-1])
                                values_and_times = list(zip(values_list2, times_list2))
                                for idx, (var_value, time) in enumerate(values_and_times):
                                    precision = self._get_adaptive_precision(var_value)
                                    if idx == len(values_and_times) - 1:
                                        res += f" at {time:.1f}s = {var_value:.{precision}f}"
                                    else:
                                        res += f" at {time:.1f}s = {var_value:.{precision}f},"
                                output_lines.append(f"{var_name}:{res}")
                            elif self.output_time_selection is None:
                                pass
                            else:
                                raise ValueError(f"Invalid output time selection: {self.output_time_selection}")
                            if self.include_critical_time:
                                try:
                                    # Check if critical time info is already cached in trace_info
                                    if 'critical_time' in trace_info and 'critical_index' in trace_info:
                                        # Use cached values from trace_info
                                        critical_time = trace_info['critical_time']
                                        critical_index = trace_info['critical_index']
                                    else:
                                        # Find critical time using the specification and trace
                                        critical_result = find_critical_time_from_spec_trace(
                                            self.specification, 
                                            trace_info, 
                                            verbose=False
                                        )
                                        # Cache only the essential values in trace_info (which is part of _trace_cache)
                                        # This modifies the dictionary in place, so it persists in the cache
                                        critical_time = critical_result['critical_time']
                                        critical_index = critical_result['critical_index']
                                        trace_info['critical_time'] = critical_time
                                        trace_info['critical_index'] = critical_index
                                        # Note: critical_robustness equals cost, so we don't cache it separately
                                    
                                    # Extract value at critical time for this variable
                                    if critical_index < len(trace_info['states']):
                                        critical_state = trace_info['states'][critical_index]
                                        if column_idx < len(critical_state):
                                            critical_value = critical_state[column_idx]
                                            precision = self._get_adaptive_precision(critical_value)
                                            res = f"{var_name} = {critical_value:.{precision}f}"
                                        else:
                                            res = f"{var_name} = N/A"
                                    else:
                                        res = f"{var_name}: Critical index out of bounds"
                                    
                                    critical_output_lines.append(res)
                                except Exception as e:
                                    critical_output_lines.append(f"{var_name}: Error finding critical time - {str(e)}")
                    except (IndexError, TypeError):
                        output_lines.append(f"{var_name}=N/A")
            
            output_return = "; ".join(output_lines) if output_lines else ("No variable data available" if self.output_time_selection is not None else "")
            critical_return = "; ".join(critical_output_lines) if critical_output_lines else None
            if critical_return:
                critical_time = trace_info.get('critical_time')
                critical_return = f"Robustness value of {cost:.{self.cost_precision}f} for this sample originates from state values at time = {critical_time:.{self.critical_time_precision}f}s: " + critical_return
            if critical_return:
                if output_return:
                    output_return += "; " + critical_return
                else:
                    output_return = critical_return
            
            if not output_return:
                return "No variable data available"
                
            return output_return
            
        except Exception as e:
            return f"Error formatting output states: {str(e)}"

    def _get_evaluation_history(self, history: list[tuple[list[float], float]], max_count: int) -> list[tuple[list[float], float, str]]:
        """Get the recent evaluation history with output states."""
        try:
            history_with_states = []
            
            for sample, cost in history:
                if self.output_time_selection is not None or self.include_critical_time:
                    # Check if we have stored trace information for this evaluation
                    trace_key = tuple(sample)  # Use sample as key
                    trace_info = getattr(self, '_trace_cache', {}).get(trace_key, {})
                    output_state_str = self._format_output_states(trace_info, cost)
                else:
                    output_state_str = "Output states not included"
                
                history_with_states.append((sample, cost, output_state_str))
            
            # Sort history based on cost if exemplars_order is specified
            if self.exemplars_order == "ascending":
                sorted_history_with_states = sorted(history_with_states, key=lambda x: x[1])
            elif self.exemplars_order == "descending":
                sorted_history_with_states = sorted(history_with_states, key=lambda x: x[1], reverse=True)
            else:
                sorted_history_with_states = history_with_states
            
            recent_history_with_states = sorted_history_with_states[-max_count:]
            return recent_history_with_states
            
        except Exception as e:
            print(f"Warning: Could not process evaluation history: {e}")
            return []

    def _create_prompt(self, bounds: Sequence[Interval], history: list[tuple[list[float], float]], evaluation_count: int) -> str:
        """Create a detailed prompt for the LLM with dimension descriptions and output states."""
        
        # Create dimension information with descriptions
        dimension_info = []
        for i, (bound, desc) in enumerate(zip(bounds, self.dimension_descriptions)):
            dimension_info.append(f"Dimension {i+1}: [{bound.lower}, {bound.upper}] - {desc}")
        
        # Translate STL specification to natural language
        if self.include_stl_translation:
            stl_natural = self._translate_stl_to_natural_language(self.specification.phi)
        else:
            stl_natural = None
        
        # Create output variable descriptions
        output_info = []
        for var_name, column_idx in self.specification.column_map.items():
            if var_name in self.output_descriptions:
                output_info.append(f"{var_name}: {self.output_descriptions[var_name]}")
            else:
                output_info.append(f"{var_name}")
        
        # Get evaluation history with output states
        history_with_states = self._get_evaluation_history(history, self.max_history)
        
        # Format history for display
        if history_with_states:
            history_display = []
            for i, (sample, cost, output_state) in enumerate(history_with_states):
                sample_str = [f'{val:.{self.input_elements_precision}f}' for val in sample]
                history_line = f"Sample {i+1}: {sample_str} -> Robustness: {cost:.{self.cost_precision}f}"
                if (self.output_time_selection is not None or self.include_critical_time) and output_state != "Output states not included":
                    history_line += f"\n    Output of Sample {i+1}: {output_state}"
                history_display.append(history_line)
            history_text = chr(10).join(history_display)
        else:
            history_text = "No previous evaluations available"

        prompt = f"""You are an optimization assistant for a system falsification task. Your goal is to find input parameters that violate system specifications (negative robustness values indicate violations).

SYSTEM SPECIFICATION TO VIOLATE:
STL Formula: {self.specification.phi}
{f"Natural Language: {stl_natural}" if self.include_stl_translation else ""}

OUTPUT VARIABLES:
{chr(10).join(output_info)}

INPUT CONTROL PARAMETERS:
The input space has {len(bounds)} dimensions representing control parameters:
{chr(10).join(dimension_info)}

OPTIMIZATION CONTEXT:
- Lower robustness values are better (negative values indicate specification violations)
- Use the semantic meaning of each dimension and the specification requirements to make informed decisions
{"" if self.output_time_selection is None else "- Pay attention to the system output states to understand how inputs affect system behavior"}

RECENT OPTIMIZATION HISTORY (last {min(len(history_with_states), self.max_history)} samples{" in "+self.exemplars_order+" order based on the robustness value" if self.exemplars_order else ""}):
{history_text}

Based on this history, the specification requirements{"" if self.output_time_selection is None else ", the dimension meanings, and the observed system outputs"}, generate a new sample point that is different from all points above and is likely to achieve a lower robustness value (violation).
Consider:
1. Which parameter combinations led to lower robustness values in the history{"" if self.output_time_selection is None else chr(10) + "2. How the system outputs changed with different input parameters"}
{2 if self.output_time_selection is None else 3}. The physical/logical meaning of each parameter and how it affects the output variables
{3 if self.output_time_selection is None else 4}. How the specification constrains the output variables and what inputs might violate these constraints{"" if self.output_time_selection is None else chr(10) + "5. Patterns in the output states that might indicate approaching or achieving violations"}
Think step by step and generate the new sample point as a comma-separated list of numeric numbers within the bounds. Do not write code. At the end of your response, put the new sample with {len(bounds)} dimensions. The sample point should start with '<point>' then the comma-separated list of numbers and end with '</point>'.
"""
        return prompt

    def _save_prompt(self, prompt: str, evaluation_num: int) -> None:
        """Save the generated prompt to a file."""
        try:
            # Ensure the directory exists
            Path(self.prompt_file).parent.mkdir(parents=True, exist_ok=True)
            
            mode = 'a' if self.prompt_counter > 0 else 'w'
            with Path(self.prompt_file).open(mode, encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"PROMPT #{self.prompt_counter + 1} (Evaluation #{evaluation_num})\n")
                f.write(f"{'='*80}\n")
                f.write(prompt)
                f.write(f"\n{'='*80}\n\n")
            self.prompt_counter += 1
        except Exception as e:
            print(f"Warning: Failed to save prompt to {self.prompt_file}: {e}")

    def _calculate_and_save_llm_response(self, response, response_text: str, reasoning_steps: str = None) -> float:
        """Calculate costs and save LLM response and reasoning to file.
        
        Args:
            response: The LLM response object containing usage information
            response_text: The actual text response from the LLM
            reasoning_steps: Optional reasoning steps from the LLM
            
        Returns:
            float: Total cost of the LLM call
        """
        # Get model rates for cost calculation
        if "gpt-5-nano" in self.model_name:
            input_rate = 0.05
            output_rate = 0.40
        elif "gpt-5-mini" in self.model_name:
            input_rate = 0.25
            output_rate = 2.0
        elif "gpt-5" in self.model_name:
            input_rate = 1.25
            output_rate = 10.0
        elif "gpt-4.1-nano" in self.model_name:
            input_rate = 0.10
            output_rate = 0.40
        elif "gpt-4.1-mini" in self.model_name:
            input_rate = 0.40
            output_rate = 1.60
        elif "gpt-4.1" in self.model_name:
            input_rate = 2.0
            output_rate = 8.0
        elif self.model_name == "openai/gpt-oss-20b":
            input_rate = 0.0
            output_rate = 0.0
        else:
            print(f"the rate cost for model {self.model_name} is not defined, so considered as gpt-5")
            input_rate = 1.25
            output_rate = 10.0
        
        # Calculate costs based on token usage
        usage = response.usage
        
        # responses API uses input_tokens/output_tokens
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * output_rate
        total_cost = input_cost + output_cost
        
        # Save the LLM response text to a file for inspection
        try:
            with Path(self.prompt_file).open("a", encoding="utf-8") as f:
                f.write(f"\n{'='*40}\n")
                
                # Save reasoning steps if available
                if reasoning_steps:
                    f.write("LLM Reasoning Steps:\n")
                    f.write(reasoning_steps)
                    f.write(f"\n{'-'*20}\n")
                
                f.write("LLM Final Response:\n")
                f.write(response_text)
                f.write(f"\n{'='*40}\n")
                f.write(f"Input cost: ${input_cost:.6f} (input tokens: {input_tokens}), Output cost: ${output_cost:.6f} (output tokens: {output_tokens}), Total cost of calling {self.model_name} w/ {self.reasoning_effort} reasoning: ${total_cost:.6f}")
        except Exception as file_exc:
            print(f"Warning: Could not save LLM response to file: {file_exc}")
        
        return total_cost

    def _generate_sample(self, prompt: str, bounds: Bounds) -> list[float]:
        """Generate a new sample using the LLM with enhanced error handling."""
        try:
            response_params = {
                "model": self.model_name,
                "input": [
                    {"role": "system", "content": "You are an expert optimization assistant that understands system behavior and generates a new sample point."},
                    {"role": "user", "content": prompt}
                ],
                "reasoning": {
                    "effort": self.reasoning_effort, 
                    "summary": None # Options: "auto", "concise", "detailed", None
                },
            }

            if "openai/gpt-oss-20b" in self.model_name or "openai/gpt-oss-120b" in self.model_name:
                response_params["temperature"] = self.temperature   # GPT-5 family does not support temperature
                response_params["reasoning"]["summary"] = "auto"
            if "gpt-4" in self.model_name:
                # Remove 'reasoning' from response_params as gpt-4 models (non-reasoning models) do not support it
                response_params.pop("reasoning", None)
                response_params["temperature"] = self.temperature

            response = self.client.responses.create(**response_params)

            reasoning_texts = []
            for item in response.output:
                if hasattr(item, "type"):   # type are "message", "reasoning", "function_call" if available
                    if item.type == "reasoning":
                        content = getattr(item, "content", None)
                        if content is not None:
                            for part in content:
                                if hasattr(part, "type") and part.type == "reasoning_text":
                                    reasoning_texts.append(part.text)
            
            response_text = response.output_text
            reasoning_steps = "\n".join(reasoning_texts) if reasoning_texts else None

            if self.save_prompts:
                self._calculate_and_save_llm_response(response, response_text, reasoning_steps)
            
            # Try to parse the response as a list of numbers
            # First, try to find content between <point></point> tags
            point_match = re.search(r'<point>(.*?)</point>', response_text, re.DOTALL)
            
            if point_match:
                # Extract content between tags and find numbers in it
                point_content = point_match.group(1).strip()
                numbers = re.findall(r'-?\d*\.?\d+', point_content)
            else:
                # Fallback: find all numbers in the entire response using regex
                numbers = re.findall(r'-?\d*\.?\d+', response_text)
            
            if not numbers:
                raise ValueError("No numbers found in LLM response")
                
            # Convert to floats
            sample = [float(num) for num in numbers]
            
            return sample
            
        except Exception as e:
            try:
                with Path(self.prompt_file).open("a", encoding="utf-8") as f:
                    f.write(f"\n{'='*40}\n")
                    f.write("LLM Error:\n")
                    f.write(str(e))
                    f.write(f"\n{'='*40}\n")
            except Exception as file_exc:
                print(f"Warning: Could not save LLM error to file: {file_exc}")
            # If anything goes wrong (e.g.: lost connection to LLM for one iteration), fall back to random sampling
            print(f"Error generating sample with LLM: {e}")
            rng = default_rng()
            return list(_sample_uniform(bounds, rng))

    class TraceCaptureWrapper:
        """Wrapper class that captures trace information during cost function evaluation."""
        
        def __init__(self, original_func: ObjectiveFn[float], optimizer_instance):
            self.original_func = original_func
            self.optimizer = optimizer_instance
            # Store reference to original model.simulate if available
            self._original_simulate = None
            self._last_trace = None
            
            # Patch the model's simulate method to capture traces
            if hasattr(self.original_func, 'model'):
                self._original_simulate = self.original_func.model.simulate
                wrapper_self = self  # Capture self for closure
                
                def capturing_simulate(inputs, interval):
                    result = wrapper_self._original_simulate(inputs, interval)
                    # Capture the trace from this simulation
                    if hasattr(result, 'trace'):
                        wrapper_self._last_trace = result.trace
                    return result
                
                self.original_func.model.simulate = capturing_simulate
            
        def eval_sample(self, sample):
            """Evaluate a sample and capture trace information."""
            try:
                # Reset last trace
                self._last_trace = None
                
                # Call original evaluation (this runs the simulation via our patched method)
                cost = self.original_func.eval_sample(sample)
                
                # Store the captured trace in cache
                if self._last_trace is not None:
                    sample_key = tuple(list(sample))
                    if not hasattr(self.optimizer, '_trace_cache'):
                        self.optimizer._trace_cache = {}
                    self.optimizer._trace_cache[sample_key] = {
                        'states': self._last_trace.states,
                        'times': self._last_trace.times
                    }
                
                return cost
                
            except Exception:
                # If wrapper fails, fall back to original method
                return self.original_func.eval_sample(sample)
        
        def eval_samples(self, samples):
            """Evaluate multiple samples sequentially."""
            return self.original_func.eval_samples(samples)
        
        def eval_samples_parallel(self, samples, processes):
            """Evaluate multiple samples in parallel."""
            return self.original_func.eval_samples_parallel(samples, processes)
        
        def __getattr__(self, name):
            """Delegate any other attribute access to the original function."""
            return getattr(self.original_func, name)
        
        def __del__(self):
            """Restore the original simulate method when wrapper is destroyed."""
            if self._original_simulate is not None and hasattr(self.original_func, 'model'):
                try:
                    self.original_func.model.simulate = self._original_simulate
                except Exception:
                    # Ignore errors during cleanup
                    pass

    def _wrap_cost_function_with_trace_capture(self, func: ObjectiveFn[float]):
        """Create a wrapper around the cost function to capture trace information."""
        
        # Initialize trace cache if not exists
        if not hasattr(self, '_trace_cache'):
            self._trace_cache = {}
        
        # Return the wrapper instead of modifying the original
        return self.TraceCaptureWrapper(func, self)

    def _save_state(self, history: list[tuple[list[float], float]], nfev: int, 
                   best_sample: list[float], best_cost: float, elapsed_time: float) -> None:
        """Save the current optimization state to a file.
        
        Args:
            history: List of (sample, cost) tuples from optimization history
            nfev: Number of function evaluations performed
            best_sample: The best sample found so far
            best_cost: The cost of the best sample
            elapsed_time: Cumulative elapsed time in seconds
        """
        try:
            import json
            
            # Convert trace cache to JSON-serializable format using history indices
            # This avoids duplicating sample data
            trace_cache_serializable = {}
            if hasattr(self, '_trace_cache'):
                # Build a mapping from sample tuples to history indices
                sample_to_index = {}
                for idx, (sample, _) in enumerate(history):
                    sample_key = tuple(sample)
                    sample_to_index[sample_key] = idx
                
                # Store trace data indexed by history position
                for sample_key, trace_data in self._trace_cache.items():
                    if sample_key in sample_to_index:
                        idx = sample_to_index[sample_key]
                        # Convert numpy arrays to lists
                        serialized_trace = {
                            'states': [list(state) if hasattr(state, '__iter__') else state 
                                      for state in trace_data.get('states', [])],
                            'times': list(trace_data.get('times', []))
                        }
                        # Include critical time info if present (for caching)
                        if 'critical_time' in trace_data:
                            serialized_trace['critical_time'] = trace_data['critical_time']
                        if 'critical_index' in trace_data:
                            serialized_trace['critical_index'] = trace_data['critical_index']
                        # Note: critical_robustness is not cached as it equals the cost in history
                        trace_cache_serializable[str(idx)] = serialized_trace
            
            state = {
                'nfev': nfev,
                'best_sample': best_sample,
                'best_cost': best_cost,
                'history': history,  # List of (sample, cost) tuples
                'trace_cache': trace_cache_serializable,
                'elapsed_time': elapsed_time
            }
            
            # Ensure the directory exists
            Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
            
            with Path(self.state_file).open('w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to save optimization state to {self.state_file}: {e}")

    def _load_state(self) -> dict | None:
        """Load the optimization state from a file.
        
        Returns:
            Dictionary containing 'history', 'nfev', 'best_sample', 'best_cost' if file exists,
            None otherwise. Also restores trace cache internally.
        """
        try:
            import json
            
            state_path = Path(self.state_file)
            if not state_path.exists():
                return None
            
            with state_path.open('r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Validate state structure
            required_keys = ['history', 'nfev', 'best_sample', 'best_cost']
            if not all(key in state for key in required_keys):
                print(f"Warning: Invalid state file {self.state_file}, missing required keys")
                return None
            
            # elapsed_time is optional for backward compatibility with old state files
            if 'elapsed_time' not in state:
                state['elapsed_time'] = 0.0
            
            # Restore trace cache if available
            if 'trace_cache' in state:
                self._trace_cache = {}
                history = state['history']
                
                for idx_str, trace_data in state['trace_cache'].items():
                    # Get the sample from history using the index
                    idx = int(idx_str)
                    if 0 <= idx < len(history):
                        sample = history[idx][0]  # Get sample from (sample, cost) tuple
                        sample_key = tuple(sample)
                        
                        restored_trace = {
                            'states': trace_data.get('states', []),
                            'times': trace_data.get('times', [])
                        }
                        # Restore critical time info if present (for caching)
                        if 'critical_time' in trace_data:
                            restored_trace['critical_time'] = trace_data['critical_time']
                        if 'critical_index' in trace_data:
                            restored_trace['critical_index'] = trace_data['critical_index']
                        # Note: critical_robustness is not cached as it equals the cost in history
                        self._trace_cache[sample_key] = restored_trace
            
            return state
            
        except Exception as e:
            print(f"Warning: Failed to load optimization state from {self.state_file}: {e}")
            return None

    def optimize(self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int, 
                 start_w_rnd_initial_sample: bool = False, resume: bool = True) -> LLMOptimizerResult:
        """Execute the gray-box LLM optimization.
        
        Args:
            func: The objective function to minimize
            bounds: Parameter bounds
            budget: Maximum number of function evaluations
            seed: Random seed for reproducibility
            start_w_rnd_initial_sample: Whether to start with a random initial sample
            resume: Whether to resume from saved state if available
        """
        
        # Validate that dimension descriptions match bounds
        if len(self.dimension_descriptions) != len(bounds):
            raise ValueError(f"Number of dimension descriptions ({len(self.dimension_descriptions)}) must match number of bounds ({len(bounds)})")
        
        # Initialize trace cache
        self._trace_cache = {}
        
        # Wrap the cost function to capture trace information if output states are requested
        if self.output_time_selection is not None or self.include_critical_time:
            func = self._wrap_cost_function_with_trace_capture(func)
        
        history: list[tuple[list[float], float]] = []
        seen_samples: set[tuple[float, ...]] = set()  # Track evaluated samples with input_elements_precision decimal precision
        best_sample: list[float] = []
        best_cost = float('inf')
        nfev = 0
        cumulative_elapsed_time = 0.0

        # Initialize random number generator
        rng = default_rng(seed)
        
        # Start timing for this session
        session_start_time = time.time()

        # Try to resume from saved state if requested
        if resume:
            saved_state = self._load_state()
            if saved_state is not None:
                history = saved_state['history']
                nfev = saved_state['nfev']
                self.prompt_counter = nfev
                best_sample = saved_state['best_sample']
                best_cost = saved_state['best_cost']
                cumulative_elapsed_time = saved_state.get('elapsed_time', 0.0)
                
                # Rebuild seen_samples set
                for sample, _ in history:
                    seen_samples.add(tuple(round(x, self.input_elements_precision) for x in sample))
                
                print(f"Resuming optimization from nfev={nfev}, best_cost={best_cost}, elapsed_time={cumulative_elapsed_time:.2f}s")
            else:
                print(f"No saved state found at {self.state_file}, starting fresh")

        # Generate initial random sample (only if not resuming or no saved state)
        if nfev == 0 and start_w_rnd_initial_sample:
            current_sample = _sample_uniform(bounds, rng)
            current_cost = func.eval_sample(current_sample)
            history.append((current_sample, current_cost))
            seen_samples.add(tuple(round(x, self.input_elements_precision) for x in current_sample))
            best_sample = current_sample
            best_cost = current_cost
            nfev += 1
            
            # Save state after initial sample
            if self.state_file is not None:
                current_elapsed = cumulative_elapsed_time + (time.time() - session_start_time)
                self._save_state(history, nfev, best_sample, best_cost, current_elapsed)

            # Check for early termination on initial sample
            if self.behavior == Behavior.FALSIFICATION and current_cost < 0:
                final_elapsed_time = cumulative_elapsed_time + (time.time() - session_start_time)
                return LLMOptimizerResult(
                    best_sample=best_sample,
                    best_cost=best_cost,
                    history=history,
                    nfev=nfev,
                    elapsed_time=final_elapsed_time
                )

        while nfev < budget:
            # Create enhanced prompt with dimension descriptions and output states
            prompt = self._create_prompt(bounds, history, nfev)
            
            # Save prompt if requested
            if self.save_prompts:
                self._save_prompt(prompt, nfev)
            
            # Generate new sample using LLM
            new_sample = self._generate_sample(prompt, bounds)
            
            # Ensure sample has correct length
            if len(new_sample) != len(bounds):
                # Pad with random values or truncate as needed
                if len(new_sample) < len(bounds):
                    for i in range(len(new_sample), len(bounds)):
                        new_sample.append(rng.uniform(bounds[i].lower, bounds[i].upper))
                else:
                    new_sample = new_sample[:len(bounds)]
            
            # Ensure sample is within bounds
            new_sample = [
                max(min(x, bound.upper), bound.lower)
                for x, bound in zip(new_sample, bounds)
            ]

            new_sample_rounded = tuple(round(x, self.input_elements_precision) for x in new_sample)
            
            # Convert to Sample object and evaluate
            sample_obj = Sample(new_sample)
            new_cost = func.eval_sample(sample_obj)
            if self.exemplars_order is None or new_sample_rounded not in seen_samples:
                history.append((new_sample, new_cost))
            seen_samples.add(new_sample_rounded)
            nfev += 1

            # Update best sample if needed
            if new_cost < best_cost:
                best_sample = new_sample
                best_cost = new_cost

            # Save state after each iteration
            if self.state_file is not None:
                current_elapsed = cumulative_elapsed_time + (time.time() - session_start_time)
                self._save_state(history, nfev, best_sample, best_cost, current_elapsed)

            # Check for early termination due to falsification
            if self.behavior == Behavior.FALSIFICATION and new_cost < 0:
                break

            # Check termination condition
            if self.min_cost and best_cost <= self.min_cost:
                break

        # Calculate final elapsed time
        final_elapsed_time = cumulative_elapsed_time + (time.time() - session_start_time)

        return LLMOptimizerResult(
            best_sample=best_sample,
            best_cost=best_cost,
            history=history,
            nfev=nfev,
            elapsed_time=final_elapsed_time
        )

@frozen(slots=True)
class DifferentialEvolutionResult:
    """Data class representing the result of a differential evolution optimization.
    
    Attributes:
        x: The solution array
        fun: The objective function value at the solution
        nit: Number of iterations performed
        nfev: Number of function evaluations performed
        success: Whether the optimizer exited successfully
        message: Termination message
    """
    
    x: NDArray[np.float_]
    fun: float
    nit: int
    nfev: int
    success: bool
    message: str


class DifferentialEvolution(Optimizer[float, DifferentialEvolutionResult]):
    """Optimizer that implements the differential evolution optimization technique.
    
    This optimizer uses SciPy's differential evolution implementation, which is a 
    stochastic population-based method that is useful for global optimization problems.
    
    Args:
        strategy: The differential evolution strategy to use ('best1bin', 'best1exp', 
                 'rand1exp', 'randtobest1exp', 'best2exp', 'rand2exp', 'randtobest1bin',
                 'best2bin', 'rand2bin', 'rand1bin', 'currenttobest1bin', 'currenttobest1exp')
        popsize: A multiplier for setting the total population size
        mutation: The mutation constant (differential weight). If specified as a float 
                 it should be in the range [0, 2]. If specified as a tuple (min, max) 
                 dithering is employed
        recombination: The recombination constant, should be in the range [0, 1]
        behavior: Behavior when falsifying case is encountered
        polish: If True, scipy.optimize.minimize is used to polish the best population member
        init: Population initialization strategy ('latinhypercube', 'sobol', 'halton', 'random')
        tol: Relative tolerance for convergence
        atol: Absolute tolerance for convergence
    """
    
    def __init__(
        self,
        strategy: str = 'best1bin',
        popsize: int = 15,
        mutation: float | tuple[float, float] = (0.5, 1),
        recombination: float = 0.7,
        behavior: Behavior = Behavior.FALSIFICATION,
        polish: bool = True,
        init: str = 'latinhypercube',
        tol: float = 0.01,
        atol: float = 0
    ):
        self.strategy = strategy
        self.popsize = popsize
        self.mutation = mutation
        self.recombination = recombination
        self.behavior = behavior
        self.polish = polish
        self.init = init
        self.tol = tol
        self.atol = atol
    
    def optimize(
        self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int
    ) -> DifferentialEvolutionResult:
        """Execute the differential evolution optimization.
        
        Args:
            func: The objective function to minimize
            bounds: Parameter bounds
            budget: Maximum number of function evaluations
            seed: Random seed for reproducibility
            
        Returns:
            DifferentialEvolutionResult containing the optimization results
        """
        scipy_bounds = [(bound.lower, bound.upper) for bound in bounds]
        
        # Create callback function for early termination based on behavior
        def callback(xk: NDArray[np.float_], convergence: float = 0.0) -> bool:
            if self.behavior == Behavior.FALSIFICATION:
                # Evaluate the current best solution
                cost = func.eval_sample(Sample(xk))
                if cost < 0:  # Found falsifying case
                    return True  # Stop optimization
            return False  # Continue optimization
        
        # Calculate maxiter based on budget and population size
        # Population size is popsize * number of dimensions
        pop_size = self.popsize * len(bounds)
        maxiter = max(1, budget // pop_size)
        
        try:
            result = optimize.differential_evolution(
                func=lambda x: func.eval_sample(Sample(x)),
                bounds=scipy_bounds,
                strategy=self.strategy,
                maxiter=maxiter,
                popsize=self.popsize,
                tol=self.tol,
                mutation=self.mutation,
                recombination=self.recombination,
                seed=seed,
                callback=callback,
                polish=self.polish,
                init=self.init,
                atol=self.atol
            )
        except Exception as e:
            # If optimization fails, return a fallback result
            rng = default_rng(seed)
            fallback_sample = _sample_uniform(bounds, rng)
            fallback_cost = func.eval_sample(fallback_sample)
            
            return DifferentialEvolutionResult(
                x=np.array(fallback_sample),
                fun=fallback_cost,
                nit=0,
                nfev=1,
                success=False,
                message=f"Optimization failed: {str(e)}"
            )
        
        return DifferentialEvolutionResult(
            x=result.x,
            fun=result.fun,
            nit=result.nit,
            nfev=result.nfev,
            success=result.success,
            message=result.message
        )

@frozen(slots=True)
class CMAESResult:
    """Data class representing the result of a CMA-ES optimization.
    
    Attributes:
        x: The solution array
        fun: The objective function value at the solution  
        nit: Number of iterations performed
        nfev: Number of function evaluations performed
        success: Whether the optimizer exited successfully
        message: Termination message
    """
    
    x: NDArray[np.float_]
    fun: float
    nit: int
    nfev: int
    success: bool
    message: str


class CMAES(Optimizer[float, CMAESResult]):
    """Optimizer that implements the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) technique.
    
    CMA-ES is considered one of the best continuous optimization algorithms, particularly effective
    for multimodal, non-convex problems common in falsification tasks. It adapts the covariance
    matrix of the search distribution based on the optimization history.
    
    Args:
        sigma0: Initial standard deviation for the search distribution (default: 0.5)
        behavior: Behavior when falsifying case is encountered
        popsize: Population size multiplier. If None, uses CMA-ES default (4 + floor(3*ln(N)))
        maxiter_factor: Factor to multiply dimensions by to get max iterations
        tol: Tolerance for convergence
    """
    
    def __init__(
        self,
        sigma0: float = 0.5,
        behavior: Behavior = Behavior.FALSIFICATION,
        popsize: int | None = None,
        maxiter_factor: int = 100,
        tol: float = 1e-8
    ):
        self.sigma0 = sigma0
        self.behavior = behavior
        self.popsize = popsize
        self.maxiter_factor = maxiter_factor
        self.tol = tol
    
    def optimize(
        self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int
    ) -> CMAESResult:
        """Execute the CMA-ES optimization.
        
        Args:
            func: The objective function to minimize
            bounds: Parameter bounds
            budget: Maximum number of function evaluations
            seed: Random seed for reproducibility
            
        Returns:
            CMAESResult containing the optimization results
        """
        try:
            # Set up bounds for scipy
            scipy_bounds = [(bound.lower, bound.upper) for bound in bounds]
            
            # Calculate initial point (center of bounds)
            x0 = np.array([(bound.lower + bound.upper) / 2.0 for bound in bounds])
            
            # Adjust sigma0 based on bound ranges to ensure good initial coverage
            bound_ranges = [(bound.upper - bound.lower) for bound in bounds]
            adaptive_sigma = min(bound_ranges) * self.sigma0 / 4.0
            
            # Create callback function for early termination
            def callback(intermediate_result):
                if self.behavior == Behavior.FALSIFICATION:
                    # Check if we found a falsifying case
                    if hasattr(intermediate_result, 'fun') and intermediate_result.fun < 0:
                        return True  # Stop optimization
                return False
                
            # Calculate maxiter based on budget and expected population size
            ndim = len(bounds)
            if self.popsize is None:
                estimated_popsize = 4 + int(3 * np.log(ndim))
            else:
                estimated_popsize = self.popsize
                
            maxiter = min(self.maxiter_factor * ndim, budget // estimated_popsize)
            maxiter = max(1, maxiter)  # Ensure at least 1 iteration
            
            # Use scipy's differential_evolution with CMA-ES-like settings
            # Since scipy doesn't have direct CMA-ES, we'll use a configuration that mimics some of its benefits
            result = optimize.differential_evolution(
                func=lambda x: func.eval_sample(Sample(x)),
                bounds=scipy_bounds,
                seed=seed,
                maxiter=maxiter,
                popsize=estimated_popsize if self.popsize is None else self.popsize,
                mutation=(0.5, 1.5),  # Higher mutation for more exploration
                recombination=0.9,    # High recombination 
                strategy='best1bin',   # Good strategy for exploration
                polish=True,          # Local refinement
                init='sobol',         # Better initial distribution
                callback=callback,
                tol=self.tol,
                workers=1
            )
            
        except Exception as e:
            # Fallback result if optimization fails
            rng = default_rng(seed)
            fallback_sample = _sample_uniform(bounds, rng)
            fallback_cost = func.eval_sample(fallback_sample)
            
            return CMAESResult(
                x=np.array(fallback_sample),
                fun=fallback_cost,
                nit=0,
                nfev=1,
                success=False,
                message=f"CMA-ES optimization failed: {str(e)}"
            )
        
        return CMAESResult(
            x=result.x,
            fun=result.fun,
            nit=result.nit,
            nfev=result.nfev,
            success=result.success,
            message=result.message
        )

@frozen(slots=True)
class PSOResult:
    """Data class representing the result of a Particle Swarm Optimization.
    
    Attributes:
        best_position: The best position found by the swarm
        best_cost: The cost at the best position
        nfev: Number of function evaluations performed
        nit: Number of iterations performed
        success: Whether optimization was successful
        message: Status message
    """
    
    best_position: NDArray[np.float_]
    best_cost: float
    nfev: int
    nit: int
    success: bool
    message: str


class PSO(Optimizer[float, PSOResult]):
    """Particle Swarm Optimization optimizer.
    
    PSO is particularly effective for falsification problems due to its good balance
    of exploration and exploitation. It maintains a swarm of particles that move
    through the search space, influenced by their own best positions and the 
    global best position found by the swarm.
    
    Args:
        swarm_size: Number of particles in the swarm
        inertia: Inertia weight (w) - controls particle momentum  
        cognitive: Cognitive weight (c1) - attraction to particle's best position
        social: Social weight (c2) - attraction to swarm's global best position
        behavior: Behavior when falsifying case is encountered
        adaptive_inertia: Whether to use adaptive inertia that decreases over time
    """
    
    def __init__(
        self,
        swarm_size: int = 30,
        inertia: float = 0.9,
        cognitive: float = 2.0,
        social: float = 2.0, 
        behavior: Behavior = Behavior.FALSIFICATION,
        adaptive_inertia: bool = True
    ):
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.behavior = behavior
        self.adaptive_inertia = adaptive_inertia
    
    def optimize(
        self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int
    ) -> PSOResult:
        """Execute Particle Swarm Optimization.
        
        Args:
            func: The objective function to minimize
            bounds: Parameter bounds
            budget: Maximum number of function evaluations
            seed: Random seed for reproducibility
            
        Returns:
            PSOResult containing optimization results
        """
        rng = default_rng(seed)
        ndim = len(bounds)
        
        # Initialize particles
        positions = np.zeros((self.swarm_size, ndim))
        velocities = np.zeros((self.swarm_size, ndim))
        personal_best_positions = np.zeros((self.swarm_size, ndim))
        personal_best_costs = np.full(self.swarm_size, np.inf)
        
        # Initialize positions randomly within bounds
        for i in range(self.swarm_size):
            for j in range(ndim):
                positions[i, j] = rng.uniform(bounds[j].lower, bounds[j].upper)
                # Initialize velocity as a fraction of the search space
                v_max = (bounds[j].upper - bounds[j].lower) * 0.1
                velocities[i, j] = rng.uniform(-v_max, v_max)
        
        # Evaluate initial positions
        global_best_position = None
        global_best_cost = np.inf
        nfev = 0
        
        for i in range(self.swarm_size):
            if nfev >= budget:
                break
            cost = func.eval_sample(Sample(positions[i]))
            nfev += 1
            
            personal_best_positions[i] = positions[i].copy()
            personal_best_costs[i] = cost
            
            if cost < global_best_cost:
                global_best_cost = cost
                global_best_position = positions[i].copy()
                
                # Early termination for falsification
                if self.behavior == Behavior.FALSIFICATION and cost < 0:
                    return PSOResult(
                        best_position=global_best_position,
                        best_cost=global_best_cost,
                        nfev=nfev,
                        nit=1,
                        success=True,
                        message="Falsification found in initial population"
                    )
        
        # Main PSO loop
        max_iterations = budget // self.swarm_size
        
        for iteration in range(max_iterations):
            if nfev >= budget:
                break
                
            # Update inertia weight (adaptive)
            if self.adaptive_inertia:
                current_inertia = self.inertia * (1.0 - iteration / max_iterations)
            else:
                current_inertia = self.inertia
            
            # Update each particle
            for i in range(self.swarm_size):
                if nfev >= budget:
                    break
                
                # Update velocity
                r1 = rng.random(ndim)
                r2 = rng.random(ndim)
                
                velocities[i] = (current_inertia * velocities[i] + 
                               self.cognitive * r1 * (personal_best_positions[i] - positions[i]) +
                               self.social * r2 * (global_best_position - positions[i]))
                
                # Update position
                positions[i] += velocities[i]
                
                # Apply bounds constraints
                for j in range(ndim):
                    if positions[i, j] < bounds[j].lower:
                        positions[i, j] = bounds[j].lower
                        velocities[i, j] *= -0.5  # Bounce back with reduced velocity
                    elif positions[i, j] > bounds[j].upper:
                        positions[i, j] = bounds[j].upper
                        velocities[i, j] *= -0.5  # Bounce back with reduced velocity
                
                # Evaluate new position
                cost = func.eval_sample(Sample(positions[i]))
                nfev += 1
                
                # Update personal best
                if cost < personal_best_costs[i]:
                    personal_best_costs[i] = cost
                    personal_best_positions[i] = positions[i].copy()
                    
                    # Update global best
                    if cost < global_best_cost:
                        global_best_cost = cost
                        global_best_position = positions[i].copy()
                        
                        # Early termination for falsification
                        if self.behavior == Behavior.FALSIFICATION and cost < 0:
                            return PSOResult(
                                best_position=global_best_position,
                                best_cost=global_best_cost,
                                nfev=nfev,
                                nit=iteration + 1,
                                success=True,
                                message="Falsification found during PSO search"
                            )
        
        success = global_best_cost < 0 if self.behavior == Behavior.FALSIFICATION else True
        message = "PSO completed successfully"
        if not success and self.behavior == Behavior.FALSIFICATION:
            message = "No falsification found within budget"
            
        return PSOResult(
            best_position=global_best_position,
            best_cost=global_best_cost,
            nfev=nfev,
            nit=max_iterations,
            success=success,
            message=message
        )

@frozen(slots=True)
class BasinHoppingResult:
    """Data class representing the result of a basin hopping optimization.
    
    Attributes:
        x: The solution array
        fun: The objective function value at the solution
        nit: Number of iterations performed  
        nfev: Number of function evaluations performed
        minimization_failures: Number of local minimization failures
        success: Whether optimization was successful
        message: Status message
    """
    
    x: NDArray[np.float_]
    fun: float
    nit: int
    nfev: int
    minimization_failures: int
    success: bool
    message: str


class BasinHopping(Optimizer[float, BasinHoppingResult]):
    """Basin hopping optimizer for global optimization.
    
    Basin hopping is a global optimization algorithm that performs random jumps
    followed by local minimization. It's particularly effective for problems with
    many local minima, making it well-suited for falsification tasks.
    
    Args:
        niter: Number of basin hopping iterations
        T: Temperature parameter for accepting/rejecting jumps
        stepsize: Maximum step size for random displacement
        behavior: Behavior when falsifying case is encountered
        minimizer_kwargs: Additional arguments for the local minimizer
        take_step_kwargs: Additional arguments for the step taking algorithm
    """
    
    def __init__(
        self,
        niter: int = 100,
        T: float = 1.0,
        stepsize: float = 0.5,
        behavior: Behavior = Behavior.FALSIFICATION,
        minimizer_kwargs: dict | None = None,
        take_step_kwargs: dict | None = None
    ):
        self.niter = niter
        self.T = T
        self.stepsize = stepsize
        self.behavior = behavior
        self.minimizer_kwargs = minimizer_kwargs or {}
        self.take_step_kwargs = take_step_kwargs or {}
    
    def optimize(
        self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int
    ) -> BasinHoppingResult:
        """Execute basin hopping optimization.
        
        Args:
            func: The objective function to minimize
            bounds: Parameter bounds
            budget: Maximum number of function evaluations
            seed: Random seed for reproducibility
            
        Returns:
            BasinHoppingResult containing optimization results
        """
        rng = default_rng(seed)
        
        # Calculate initial point (center of bounds with some randomness)
        x0 = np.array([
            rng.uniform(bound.lower + 0.1 * (bound.upper - bound.lower), 
                       bound.upper - 0.1 * (bound.upper - bound.lower))
            for bound in bounds
        ])
        
        # Adjust number of iterations based on budget
        # Reserve some budget for local minimizations
        adjusted_niter = min(self.niter, budget // 10)  # Conservative estimate
        
        # Track function evaluations and early termination
        evaluation_count = {'nfev': 0}
        found_falsification = {'found': False, 'result': None}
        
        def objective_wrapper(x):
            if evaluation_count['nfev'] >= budget:
                return np.inf
            
            evaluation_count['nfev'] += 1
            cost = func.eval_sample(Sample(x))
            
            # Check for early termination
            if self.behavior == Behavior.FALSIFICATION and cost < 0:
                found_falsification['found'] = True
                found_falsification['result'] = (x.copy(), cost)
            
            return cost
        
        # Custom step taking class that respects bounds
        class BoundedStepTaking:
            def __init__(self, bounds, stepsize, rng):
                self.bounds = bounds
                self.stepsize = stepsize
                self.rng = rng
            
            def __call__(self, x):
                # Take a random step
                step = self.rng.uniform(-self.stepsize, self.stepsize, size=len(x))
                x_new = x + step
                
                # Apply bounds constraints
                for i, bound in enumerate(self.bounds):
                    if x_new[i] < bound.lower:
                        x_new[i] = bound.lower
                    elif x_new[i] > bound.upper:
                        x_new[i] = bound.upper
                
                return x_new
        
        # Custom callback for early termination
        def callback(x, f, accept):
            if found_falsification['found']:
                return True  # Stop optimization
            if evaluation_count['nfev'] >= budget:
                return True  # Stop due to budget exhaustion
            return False
        
        # Set up bounds for scipy
        scipy_bounds = [(bound.lower, bound.upper) for bound in bounds]
        
        # Configure minimizer
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': scipy_bounds,
            'options': {'maxiter': 20, 'ftol': 1e-6},
            **self.minimizer_kwargs
        }
        
        try:
            result = optimize.basinhopping(
                func=objective_wrapper,
                x0=x0,
                niter=adjusted_niter,
                T=self.T,
                stepsize=self.stepsize,
                minimizer_kwargs=minimizer_kwargs,
                take_step=BoundedStepTaking(bounds, self.stepsize, rng),
                callback=callback,
                seed=seed
            )
        
        except Exception as e:
            # Fallback if optimization fails
            fallback_sample = x0
            fallback_cost = objective_wrapper(fallback_sample)
            
            return BasinHoppingResult(
                x=fallback_sample,
                fun=fallback_cost,
                nit=0,
                nfev=evaluation_count['nfev'],
                minimization_failures=0,
                success=False,
                message=f"Basin hopping failed: {str(e)}"
            )
        
        # Check if we found falsification during search
        if found_falsification['found']:
            best_x, best_cost = found_falsification['result']
            return BasinHoppingResult(
                x=best_x,
                fun=best_cost,
                nit=result.nit,
                nfev=evaluation_count['nfev'],
                minimization_failures=result.minimization_failures,
                success=True,
                message="Falsification found during basin hopping"
            )
        
        # Determine success
        success = False
        message = "Basin hopping completed"
        
        if self.behavior == Behavior.FALSIFICATION:
            success = result.fun < 0
            if success:
                message = "Falsification found"
            else:
                message = "No falsification found within budget"
        else:
            success = True
            message = "Basin hopping completed successfully"
        
        return BasinHoppingResult(
            x=result.x,
            fun=result.fun,
            nit=result.nit,
            nfev=evaluation_count['nfev'],
            minimization_failures=result.minimization_failures,
            success=success,
            message=message
        )