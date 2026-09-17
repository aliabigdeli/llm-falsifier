# LLM-Falsifier

![Teaser](teaser.png)

This repository contains the implementation of **LLM-Falsifier**, a framework for falsifying Cyber-Physical Systems (CPS) using Large Language Models (LLMs). It is built on top of the [$\Psi$-TaLiRo](https://github.com/cpslab-asu/psy-taliro) framework.

The core logic, including LLM-based optimizers, is located in `src/llm_falsifier`, while experimental benchmarks are provided in the `archcomp/` directory.

## Prerequisites

- **Python**: Version 3.9
- **Package Manager**: [uv](https://docs.astral.sh/uv/)
- **MATLAB**: Version 2022a with Simulink installed (required for `matlabengine==9.12.21`)

## Installation

1.  **Install `uv`** (if not already installed):
    ```bash
    # On macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # On Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2.  **Sync Dependencies**:
    Initialize the environment and install dependencies locked in `uv.lock`.
    ```bash
    uv sync
    ```

3.  **Setup API Key**:
    Create a file named `api_key.txt` in the root directory of the project and paste your OpenAI API key into it.
    ```bash
    echo "sk-your-openai-api-key-here" > api_key.txt
    ```
    Alternatively, you can use local LLMs such as `gpt-oss-20b` by setting up an LM-Studio local server, without needing an OpenAI API key.

## Usage

Experiments are located in the `archcomp/` directory.

### Running Falsification Experiments

To run an experiment, navigate to the `archcomp` directory and use `uv run` to execute the experiment script.

```bash
cd archcomp
uv run autotrans_all_specs.py \
    --spec AT1 \
    --optimizer LLMGB \
    --seed 1 \
    --reasoning-effort low \
    --llm-model gpt-5-nano \
    --max-budget 100 \
    --output-time-selection 6 \
    --include-critical-time
```

## Project Structure

- **`src/llm_falsifier/`**: Source code for the package.
  - `optimizers.py`: Implementation of `LLMOptimizer`(LLM) and `LLMGrayBoxOpt`(LLMGB) optimizers.
  - `base_runner.py`: Base class for running specifications.
  - `optimizer_factory.py`: Factory for creating optimizer instances.
- **`archcomp/`**: Benchmarks and experiment scripts (e.g., Automatic Transmission, F16, etc.).
- **`pyproject.toml` & `uv.lock`**: Project dependency definitions.

## Troubleshooting

- **MATLAB Engine Error**: Ensure you have MATLAB R2022a installed. The `matlabengine` version pinned in this project (`9.12.21`) corresponds specifically to R2022a. If you have a different version, you may need to update the dependency in `pyproject.toml` and run `uv sync`, though compatibility is not guaranteed.
- **API Key Errors**: Ensure `api_key.txt` exists in the project root or the `OPENAI_API_KEY` environment variable is set correctly.
