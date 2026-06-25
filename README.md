# ⚔️ Annihilation

<div align="center">
  <img src="./logo.jpeg" alt="Annihilation Logo" width="300"/>
</div>

**Autonomous Language Model Decensoring Framework**

[![License: AGPLv3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-red)](https://pytorch.org/)

---

## ⚠️ Work in Progress

> **⚡ This project is actively under development. Features, APIs, and documentation may change without notice.**

---

## 🔥 What is Annihilation?

**Annihilation** is a powerful, fully automatic framework for removing censorship (safety alignment) from transformer-based language models. It uses an advanced implementation of **directional ablation** (abliteration) combined with **TPE-based parameter optimization** to achieve unprecedented results without expensive post-training.

### Key Features

- 🤖 **Fully Autonomous** - No human intervention required; the system automatically finds optimal decensoring parameters
- ⚡ **State-of-the-Art Performance** - Achieves excellent refusal suppression while preserving model capabilities
- 🔧 **Advanced Abliteration** - Parametric directional ablation with flexible weight kernels
- 🧠 **Smart Optimization** - Co-minimizes refusal count and KL divergence using Optuna's TPE sampler
- 🎯 **Multi-Architecture Support** - Works with dense models, MoE architectures, hybrid models, and many multimodal models
- 📊 **Research Tools** - Built-in residual geometry analysis and visualization capabilities

---


---

## 🚀 Quick Start

Use a Python virtual environment so Annihilation's dependencies do not collide
with packages installed globally.

```powershell
# Windows PowerShell
python -m venv annihilation-env
.\annihilation-env\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -U annihilate-llm

# Decensor any model automatically
annihilate Qwen/Qwen3-4B-Instruct-2507
```

```bash
# macOS/Linux/Android terminal
python -m venv annihilation-env
source annihilation-env/bin/activate
python -m pip install -U pip
python -m pip install -U annihilate-llm

# Decensor any model automatically
annihilate Qwen/Qwen3-4B-Instruct-2507
```

### Updating

Upgrade to the latest PyPI release inside your active virtual environment:

```powershell
# Windows PowerShell
.\annihilation-env\Scripts\Activate.ps1
python -m pip install --no-cache-dir -U annihilate-llm
annihilate --help
```

```bash
# macOS/Linux/Android terminal
source annihilation-env/bin/activate
python -m pip install --no-cache-dir -U annihilate-llm
annihilate --help
```

### Repair or Uninstall

If you get import errors, broken console scripts, stale `heretic.main` errors,
or confusing pip error codes, run the repair helper from the same virtual
environment. Use `python -m` so it still works when the `annihilate` command
itself is broken.

Show the current install, command shims, and exact repair commands:

```powershell
python -m annihilate.repair doctor
```

Clean reinstall Annihilate and remove old Heretic package names from the active
environment:

```powershell
python -m annihilate.repair repair --yes
```

Fully uninstall Annihilate from the active environment:

```powershell
python -m annihilate.repair uninstall --yes
```

The installed package also exposes `annihilate-doctor`, `annihilate-repair`, and
`annihilate-uninstall`, but `python -m annihilate.repair ...` is the safest form
when command shims are stale or broken.

### GitHub Package Bundle

Release tags publish the Python wheel and source distribution to the GitHub
Container Registry package `ghcr.io/tjcrims0nx/annihilate-llm`. The package
bundle stores built artifacts under `/packages/` inside the image and replaces
the old `annihilation-llm-tjcrims0nx` package name.

`1.4.3` adds the repair/uninstall helpers. `1.4.2` fixed broken console script
metadata from `1.4.1`, which could point the `annihilate` command at the old
`heretic.main` module. These releases also keep the `bitsandbytes` dependency
fix, add the tokenizer helper dependencies `sentencepiece` and `tiktoken`,
restore the Annihilate banner, and print a clear error when a GGUF repository is
supplied.

### Requirements

- **Python**: 3.10+
- **PyTorch**: 2.2+ (hardware-specific installation required)
- **Hardware**: GPU recommended (CUDA, ROCm, XPU, or MPS)
- **Model format**: Use Transformers-compatible Hugging Face repositories with
  safetensors or PyTorch weights. GGUF repositories are for llama.cpp-style
  inference and cannot be abliterated with the PEFT/LoRA workflow.

### GPU Setup on Windows

If Windows sees your NVIDIA GPU but Annihilate says no GPU is detected, your
virtual environment probably has CPU-only PyTorch installed.

Newer Annihilate builds also check `nvidia-smi` directly. If the NVIDIA driver
can see the GPU but PyTorch cannot use CUDA, Annihilate will report the GPU and
warn that the active Python environment needs a CUDA-enabled PyTorch build.

Check that Windows can see the GPU:

```powershell
nvidia-smi
```

Replace CPU-only PyTorch with a CUDA build inside the active environment:

```powershell
.\annihilation-env\Scripts\Activate.ps1
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify PyTorch can see CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

`torch.cuda.is_available()` should print `True`. If your GPU has limited VRAM
such as 4 GB, start with smaller models and expect larger models to run out of
memory.

### GPU Setup on Ubuntu

Install Python venv support and create an isolated environment:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
python3 -m venv annihilation-env
source annihilation-env/bin/activate
python -m pip install -U pip
python -m pip install -U annihilate-llm
```

For NVIDIA GPUs, confirm the driver can see the card:

```bash
nvidia-smi
```

If Annihilate says no GPU is detected, replace CPU-only PyTorch with a CUDA
build inside the active environment:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify PyTorch can see CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

`torch.cuda.is_available()` should print `True`. If `nvidia-smi` is missing or
fails, install or repair the NVIDIA driver first, then reopen the terminal and
activate the environment again.

---

## ⚙️ Configuration

Annihilation works out of the box with defaults, but offers extensive configuration options:

```bash
# View all options
annihilate --help

# Or use a config file
# Rename config.default.toml to config.toml and modify as needed
```

### Key Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `n_trials` | 200 | Number of optimization trials |
| `n_startup_trials` | 60 | Random exploration trials before TPE focuses the search |
| `quantization` | none | Model quantization (bnb_4bit) |
| `row_normalization` | full | Weight normalization strategy |
| `orthogonalize_direction` | true | Direction adjustment method |

### Aggressive Optimization

The default Annihilate run already uses the enhanced aggressive optimizer from
upstream processing. It runs a multi-objective Optuna TPE search that minimizes
both refusal count and KL divergence, using:

- `n_trials = 200`
- `n_startup_trials = 60`
- multivariate TPE sampling
- `n_ei_candidates = 128`
- both global and per-layer refusal direction search
- separate ablation kernels for attention output projections and MLP down
  projections

For a quick smoke test, lower `n_trials` and `n_startup_trials`. For a real
service or cloud GPU run, leave the defaults in place or increase the prompt
slices in `config.toml` if the GPU has enough time and memory.

---

## 🔬 How It Works

Annihilation implements **parametric directional ablation**:

1. **Direction Computation** - Calculates refusal directions by computing difference-of-means between first-token residuals for harmful vs harmless prompts

2. **Parametric Ablation** - For each transformer component (attention out-projection, MLP down-projection), orthogonalizes weights against the refusal direction using LoRA adapters

3. **Multi-Parameter Optimization** - Uses Optuna's TPE sampler to co-optimize:
   - Ablation weight kernel shape (max_weight, position, min_weight, distance)
   - Direction index (layer selection or interpolation)
   - Per-component parameters (attention vs MLP)

4. **Automatic Selection** - Chooses from Pareto-optimal trials based on refusal count vs KL divergence tradeoff

---

## 📊 Benchmarking

After decensoring, you can:

- 💬 **Chat** with the model to test behavior
- 📈 **Benchmark** using standard evaluation frameworks (MMLU, GSM8K, etc.)
- 💾 **Save** the model locally or upload to Hugging Face

---

## 🧪 Research Features

Install with research dependencies for visualization tools:

```bash
pip install -U annihilate-llm[research]
```

### Residual Geometry

Print a quantitative analysis of how residual vectors for "harmful" and
"harmless" prompts relate to each other:

```bash
annihilate --print-residual-geometry
```

You can also enable it in `config.toml`:

```toml
print_residual_geometry = true
```

When enabled, Annihilate computes first-token residual vectors for each
transformer layer, then prints metrics such as:

- `g` - mean residual vector for good prompts
- `g*` - geometric median residual vector for good prompts
- `b` - mean residual vector for bad prompts
- `b*` - geometric median residual vector for bad prompts
- `r` - refusal direction for means, computed as `b - g`
- `r*` - refusal direction for geometric medians, computed as `b* - g*`
- `S(x,y)` - cosine similarity between vectors
- `|x|` - L2 norm of a vector
- `Silh` - mean silhouette coefficient of good/bad residual clusters

This is useful for studying where refusal behavior separates across the model
stack and how strongly each layer distinguishes the prompt groups.

### Residual Plots

Generate PaCMAP visualizations of residual vectors:

```bash
annihilate --plot-residuals
```

Or enable plotting in `config.toml`:

```toml
plot_residuals = true
residual_plot_path = "plots"
residual_plot_title = 'PaCMAP Projection of Residual Vectors for "Harmless" and "Harmful" Prompts'
residual_plot_style = "dark_background"
```

When run with `--plot-residuals`, Annihilate will:

1. Compute hidden-state residual vectors for the first output token, for each
   transformer layer, for both "harmful" and "harmless" prompts.
2. Perform a PaCMAP projection from residual space into 2D space.
3. Left-right align the projected "harmful" and "harmless" residuals by their
   geometric medians. PaCMAP is initialized from the previous layer's projection
   so consecutive layers animate smoothly.
4. Scatter-plot each layer and save a PNG image for every layer.
5. Generate an animated GIF showing how residuals transform between layers.

<div align="center">
  <img src="./assets/residual-projections.gif" alt="Animated PaCMAP projection of residual vectors across transformer layers" width="800"/>
</div>

PaCMAP is CPU-heavy. For larger models or large prompt sets, residual plots can
take an hour or more even when model inference is running on a GPU. For long
optimization jobs, it is usually best to run the decensoring pass first and run
plot generation as a separate analysis pass afterward.

---

## 📜 License

**Annihilation** is free software distributed under the **GNU Affero General Public License v3**.

See [LICENSE](LICENSE) for full details.

---

## ⚡ Disclaimer

This tool is provided for **research and educational purposes** only. The developers do not condone the use of decensored models for harmful activities. Users are responsible for ensuring compliance with applicable laws and model terms of service.

---

<div align="center">

**Breaking the Chains | Unleashing Model Potential**

*"The only way to discover the limits of the possible is to go beyond them into the impossible."*

</div>
