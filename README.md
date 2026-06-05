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

### Requirements

- **Python**: 3.10+
- **PyTorch**: 2.2+ (hardware-specific installation required)
- **Hardware**: GPU recommended (CUDA, ROCm, XPU, or MPS)
- **Optional**: Install `annihilate-llm[bnb]` only on platforms
  that support bitsandbytes if you want `bnb_4bit` quantization.

### GPU Setup on Windows

If Windows sees your NVIDIA GPU but Annihilate says no GPU is detected, your
virtual environment probably has CPU-only PyTorch installed.

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
| `quantization` | none | Model quantization (bnb_4bit) |
| `row_normalization` | full | Weight normalization strategy |
| `orthogonalize_direction` | true | Direction adjustment method |

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

Features:
- `--plot-residuals` - Generate PaCMAP projections of residual vectors
- `--print-residual-geometry` - Detailed residual analysis metrics

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
