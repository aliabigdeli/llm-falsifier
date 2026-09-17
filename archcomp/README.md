#  ARCH-Comp benchmark

This directory contains ARCH-Comp Falsification benchmarks. Each benchmark demonstrates a different system type and corresponding specifications for temporal logic falsification. Model definitions and signal creation follow those used in [Psy-Taliro](https://github.com/cpslab-asu/ARCH-Comp-2024-Repeatability).

## Available Benchmarks

- **autotrans_all_specs.py** - Automotive transmission benchmark
- **cc_all_specs.py** - Chasing cars benchmark
- **nn_all_specs.py** - Neural network controller system benchmark
- **f16_all_specs.py** - Aircraft Ground Collision Avoidance System (F16) benchmark
- **sc_all_specs.py** - Steam condenser with Recurrent Neural Network Controller  benchmark

## Running Individual Benchmark Files

Each benchmark file can be run independently with the following command structure:

```bash
uv run <benchmark_file> [options]
```

### Command Line Options

All benchmark files support the following common options:

#### Core Options:
- `-s, --spec SPEC`: Specification to analyze (each benchmark has its own set of specifications)
- `-o, --optimizer OPTIMIZER`: Optimizer to use (default: DA)
  - Available: `DA`, `LLM`, `LLMGB`, `DE`, `PSO`, `BH`, `CMAES`, `UR`
#### Optimization Settings:
- `--seed SEED`: Random seed for reproducible results (default: 42). For LLM-based optimizers, this does not guarantee reproducibility, as LLMs are inherently nondeterministic.
- `-m, --max-budget BUDGET`: Maximum number of iterations/budget (default: 100)

#### LLM-Specific Options:
- `-l, --llm-model MODEL`: LLM model to use (default: openai/gpt-oss-20b)
  - Three sources for models are currently accepted:
    1. Local LLMs via [LM-Studio](https://lmstudio.ai/) (version 0.3.29 or higher) — use names like "openai/gpt-oss-20b" (you need to have the model downloaded in LM-Studio, load the model, and have a running local server).
    2. OpenAI models via [API](https://platform.openai.com/docs/models), such as "gpt-5-nano", "gpt-5-mini", etc.
    3. Hugging Face inference providers in the format "openai/gpt-oss-20b:`PROVIDER`", such as "openai/gpt-oss-20b:groq". A full list of providers can be found at [this link](https://huggingface.co/inference/models).
- `-r, --reasoning-effort EFFORT`: Reasoning effort level (default: low)
  - Available: `minimal`, `low`, `medium`, `high` (Non-reasoning models like gpt-4.1 or gpt-4o do not support reasoning. The `minimal` option is only available for the GPT-5 series.)
- `--llm-history HISTORY`: Max number of exemplars in a prompt for LLM optimizers (default: 10)

#### LLMGB-Specific Options:
LLMGB stands for LLMGrayBox, so it includes all the LLM-Specific Options as well as the following additional options:
- `--input-descriptions TYPE`: How input descriptions are generated (default: 1)
  - `1`: Short auto-generated (input names extracted from the Simulink model)
  - `2`: Long auto-generated (input names extracted from the Simulink model)
  - `3`: Long manual - the manual descriptions are available in the benchmark files listed in the [Available Benchmarks](#available-benchmarks) section above.
- `--has-output-descriptions`: Use manual output descriptions (flag) – the manual descriptions are available in the benchmark files listed in the [Available Benchmarks](#available-benchmarks) section above.
- `--include-stl-trans`: Include STL translation in prompt (flag)
- `--output-time-selection SELECTION`: Output time selection strategy
  - `'none'`: Don't include output states in the prompt
  - `-1`: Include only the output state at the final time step (default)
  - Integer (e.g., `5` means the output state will be included from 5 evenly spaced intervals across the total time)
  - Tuple (e.g., `"8,2"` or `"(8,2)"`: first number is the number of intervals, second number is the step for selecting outputs from those intervals, so `"8,2"` means take every 2nd output from 8 evenly spaced intervals)
- `--include-critical-time`: Include state values at critical time in the prompt (flag, default: False)

### Examples

```bash
# Run with LLM optimizer
uv run autotrans_all_specs.py -s AT1 -o LLM -l gpt-5-nano -r low

# Run with LLMGB optimizer with different seed and output-time-selection
uv run autotrans_all_specs.py -s AT2 -o LLMGB --seed 1 -r low -l gpt-5-nano --output-time-selection 6

# Run with LLMGB optimizer without output states
uv run autotrans_all_specs.py -s AT1 -o LLMGB -l gpt-5-nano --output-time-selection none
```

### Available Specifications by Benchmark:

#### Autotrans (autotrans_all_specs.py):
- `AT1`, `AT2`, `AT51`, `AT52`, `AT53`, `AT54`, `AT6a`, `AT6b`, `AT6c`, `AT6abc`

#### CC - Chasing cars (cc_all_specs.py):
- `CC1`, `CC2`, `CC3`, `CC4`, `CC5`, `CCx`

#### Neural Network (nn_all_specs.py):
- `NN1`, `NN2`, `NNx`

#### F-16 (f16_all_specs.py):
- `F16a`

#### Steam Condenser (sc_all_specs.py):
- `SCa`

## Test all specifications

The `all_exp_script.sh` script allows you to systematically test all specifications.