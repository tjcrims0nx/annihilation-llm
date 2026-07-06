# ⚔️ Annihilation

<div align="center">
  <img src="./logo.jpeg" alt="Annihilation Logo" width="300"/>
</div>

**Autonomous Language Model Decensoring Framework**

[![License: AGPLv3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-red)](https://pytorch.org/)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange)](https://www.rust-lang.org/)

---

## ⚠️ Work in Progress

> **⚡ This project is actively under development. Features, APIs, and documentation may change without notice.**

---

## 🔥 What is Annihilation?

**Annihilation** is a powerful, fully automatic framework for removing censorship (safety alignment) from transformer-based language models. It uses an advanced implementation of **directional ablation** (abliteration) combined with **TPE-based parameter optimization** to achieve unprecedented results without expensive post-training.

### Key Features

- 🤖 **Fully Autonomous** - No human intervention required; the system automatically finds optimal decensoring parameters
- 🖥️ **Beautiful Terminal UI** - Brand new Rust-based TUI for real-time monitoring and easy workflow management
- ⚡ **State-of-the-Art Performance** - Achieves excellent refusal suppression while preserving model capabilities
- 🔧 **Advanced Abliteration** - Parametric directional ablation with flexible weight kernels
- 🧠 **Smart Optimization** - Co-minimizes refusal count and KL divergence using Optuna's TPE sampler
- 🎯 **Multi-Architecture Support** - Works with dense models, MoE architectures, hybrid models, and many multimodal models
- 📊 **Research Tools** - Built-in residual geometry analysis and visualization capabilities

---

## 🖥️ The Annihilation TUI

Annihilation now features a beautifully styled, high-performance **Rust Terminal User Interface (TUI)** built with `ratatui`. This provides a complete end-to-end workflow from model selection to final exporting, eliminating the need to memorize complex CLI arguments.

### Splash Screen & Model Selection
<div align="center">
  <img src="./assets/tui-splash.png" alt="Annihilation TUI Splash Screen" width="800"/>
</div>

The main menu allows you to seamlessly configure the optimization preset, type in a Hugging Face model ID (or local path), or easily resume previous models from the **Recent Models** menu. If the app detects an interrupted run, the built-in **Checkpoint System** will automatically prompt you to seamlessly resume where you left off.

### Live Processing Dashboard
<div align="center">
  <img src="./assets/tui-dashboard.png" alt="Annihilation TUI Processing Dashboard" width="800"/>
</div>

Once decensoring begins, the dashboard takes over. It features:
- **Real-time Live Charts**: Dynamic sparkline charts plotting the KL Divergence and Refusal counts across all trials.
- **Hardware Monitoring**: Instant readouts for GPU identity, VRAM usage, RAM usage, and Tokens/sec.
- **Backend Log Parsing**: The Rust interface asynchronously parses the Python backend, presenting clean, color-coded live logs.

Once completed, the UI presents a **Pareto Optimal Results Table**, allowing you to chat with the decensored model and export the final weights directly to Hugging Face or a local directory!

---

## 🚀 Installation & Setup

You will need both **Python** (for the ML backend) and **Rust** (for the TUI).

### 1. Python Environment Setup

We highly recommend using a Python virtual environment to prevent dependency collisions.

**Windows (PowerShell)**:
```powershell
python -m venv annihilation-env
.\annihilation-env\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -U annihilate-llm
```

**macOS/Linux**:
```bash
python -m venv annihilation-env
source annihilation-env/bin/activate
python -m pip install -U pip
python -m pip install -U annihilate-llm
```

### 2. GPU Setup (Highly Recommended)

If Windows/Linux sees your NVIDIA GPU but Annihilate says no GPU is detected, your virtual environment likely has CPU-only PyTorch installed.

Replace CPU-only PyTorch with a CUDA build inside the active environment:
```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify PyTorch can see CUDA:
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```
*(Should print `True`)*

---

## 🎮 Running the Application

Once everything is installed and your virtual environment is active, launching the TUI is incredibly easy.

### Windows
Just double-click the `start.bat` script in the root of the repository, or run it from the command line:
```powershell
.\start.bat
```

### macOS / Linux
Navigate into the `tui` directory and use Cargo to run the application in release mode:
```bash
cd tui
cargo run --release
```

---

## ⚙️ Advanced Configuration (CLI Backend)

If you prefer to bypass the TUI and use the Python backend directly, Annihilation works out of the box with defaults, but offers extensive configuration options:

```bash
# View all options
annihilate --help

# Decensor a model automatically via CLI
annihilate Qwen/Qwen3-4B-Instruct-2507
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

The default Annihilate run uses a multi-objective Optuna TPE search that minimizes both refusal count and KL divergence, using:
- `n_trials = 200`
- `n_startup_trials = 60`
- multivariate TPE sampling
- both global and per-layer refusal direction search
- separate ablation kernels for attention output projections and MLP down projections

---

## 🔬 How It Works

Annihilation implements **parametric directional ablation**:

1. **Direction Computation** - Calculates refusal directions by computing difference-of-means between first-token residuals for harmful vs harmless prompts
2. **Parametric Ablation** - For each transformer component (attention out-projection, MLP down-projection), orthogonalizes weights against the refusal direction using LoRA adapters
3. **Multi-Parameter Optimization** - Uses Optuna's TPE sampler to co-optimize Ablation weight kernel shape, Direction index, and Per-component parameters.
4. **Automatic Selection** - Chooses from Pareto-optimal trials based on refusal count vs KL divergence tradeoff

---

## 🧪 Research Features

Install with research dependencies for visualization tools:
```bash
pip install -U annihilate-llm[research]
```

### Residual Plots

Generate PaCMAP visualizations of residual vectors:
```bash
annihilate --plot-residuals
```

When run, Annihilate will compute hidden-state residual vectors for the first output token across all layers, perform a PaCMAP projection into 2D space, and generate an animated GIF showing how residuals transform between layers.

<div align="center">
  <img src="./assets/residual-projections.gif" alt="Animated PaCMAP projection" width="800"/>
</div>

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
