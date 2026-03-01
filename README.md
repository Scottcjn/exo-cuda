<div align="center">

# exo-cuda: Distributed NVIDIA CUDA Inference

[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-GPL--3.0-yellow)](LICENSE)
[![Tesla](https://img.shields.io/badge/Tesla-V100%2FM40-orange)](https://github.com/Scottcjn/exo-cuda)

[![BCOS Certified](https://img.shields.io/badge/BCOS-Certified-brightgreen?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xMiAxTDMgNXY2YzAgNS41NSAzLjg0IDEwLjc0IDkgMTIgNS4xNi0xLjI2IDktNi40NSA5LTEyVjVsLTktNHptLTIgMTZsLTQtNCA1LjQxLTUuNDEgMS40MSAxLjQxTDEwIDE0bDYtNiAxLjQxIDEuNDFMMTAgMTd6Ii8+PC9zdmc+)](BCOS.md)
**First verified working NVIDIA CUDA distributed inference for exo**

*Run large language models across multiple NVIDIA GPUs with automatic node discovery*

[Quick Start](#-quick-start) • [Verified Hardware](#-verified-hardware) • [Multi-Node Setup](#-multi-node-cluster) • [Troubleshooting](#-troubleshooting)

</div>

---

## 🎯 What This Fork Adds

The original [exo](https://github.com/exo-explore/exo) focuses on Apple Silicon (MLX). This fork restores **full NVIDIA CUDA support** via tinygrad:

| Feature | Original exo | exo-cuda |
|---------|-------------|----------|
| Apple Silicon (MLX) | ✅ | ✅ |
| NVIDIA CUDA | ❌ Broken | ✅ **Working** |
| Tesla V100/M40 | ❌ | ✅ **Tested** |
| Multi-GPU cluster | ⚠️ MLX only | ✅ **CUDA cluster** |
| Distributed inference | ✅ | ✅ |

## ⚡ Quick Start

```bash
# Clone this repo
git clone https://github.com/Scottcjn/exo-cuda.git
cd exo-cuda

# Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Upgrade tinygrad to latest (fixes CUDA issues)
pip install --upgrade git+https://github.com/tinygrad/tinygrad.git

# Start with CUDA backend
exo --inference-engine tinygrad --chatgpt-api-port 8001 --disable-tui
```

## 📋 Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Ubuntu 22.04/24.04, Debian 12+ |
| **Python** | 3.10+ (3.12 recommended) |
| **NVIDIA Driver** | 525+ (`nvidia-smi` to verify) |
| **CUDA Toolkit** | 12.0+ (`nvcc --version` to verify) |
| **GPU Memory** | 8GB+ per node |

### Install CUDA Toolkit
```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Verify
nvcc --version
nvidia-smi
```

## ✅ Verified Hardware

Tested December 2024 - January 2025:

| Server | GPU | VRAM | Status |
|--------|-----|------|--------|
| Dell PowerEdge C4130 | Tesla V100-SXM2 | 16GB | ✅ Working |
| Dell PowerEdge C4130 | Tesla M40 | 24GB | ✅ Working |
| Custom Build | RTX 3090 | 24GB | ✅ Working |
| Multi-node cluster | V100 + M40 | 40GB total | ✅ Working |

## 🖥️ Multi-Node Cluster

### Node 1 (Primary + API)
```bash
exo --inference-engine tinygrad --chatgpt-api-port 8001 --disable-tui
```

### Node 2+ (Workers)
```bash
exo --inference-engine tinygrad --disable-tui
```

**That's it!** Nodes auto-discover via UDP broadcast. No manual configuration.

### Manual Peer Configuration (Optional)
```bash
# Create peers.json
echo '{"peers": ["192.168.1.100:5678", "192.168.1.101:5678"]}' > peers.json

# Start with manual discovery
exo --inference-engine tinygrad --discovery-module manual \
    --discovery-config-path peers.json
```


## Minimal Smoke Test (NVIDIA path)

Run this exact sequence to confirm CUDA inference is actually healthy:

```bash
nvidia-smi
nvcc --version
python3 -c "from tinygrad import Device; print(Device.DEFAULT)"
exo --inference-engine tinygrad --chatgpt-api-port 8001 --disable-tui
```

In another terminal:

```bash
curl -sS http://localhost:8001/v1/models
curl -sS http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.2-1b","messages":[{"role":"user","content":"ping"}],"max_tokens":16}'
```

If this works, your CUDA stack is ready for multi-node expansion.

## 🔌 API Usage

exo provides a **ChatGPT-compatible API**:

```bash
# Chat completion
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-3b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# List models
curl http://localhost:8001/v1/models
```

### Supported Models

All tinygrad-compatible models work:

| Model | Parameters | Min VRAM |
|-------|------------|----------|
| Llama 3.2 1B | 1B | 4GB |
| Llama 3.2 3B | 3B | 8GB |
| Llama 3.1 8B | 8B | 16GB |
| Llama 3.1 70B | 70B | 140GB (cluster) |
| DeepSeek Coder | Various | Varies |
| Qwen 2.5 | 0.5B-72B | Varies |
| Mistral 7B | 7B | 14GB |

## 🔧 Environment Variables

```bash
# Debug logging (0-9, higher = more verbose)
DEBUG=2 exo --inference-engine tinygrad

# Tinygrad-specific debug (1-6)
TINYGRAD_DEBUG=2 exo --inference-engine tinygrad

# Limit GPU visibility
CUDA_VISIBLE_DEVICES=0,1 exo --inference-engine tinygrad
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `nvcc not found` | `sudo apt install nvidia-cuda-toolkit` |
| `OpenCL exp2 error` | `pip install --upgrade git+https://github.com/tinygrad/tinygrad.git` |
| `No GPU detected` | Check `nvidia-smi` and `nvcc --version` |
| `Out of memory` | Use smaller model or add more nodes |
| `Connection refused` | Check firewall allows UDP broadcast |

### Common Fixes

```bash
# Fix tinygrad CUDA issues
pip install --upgrade git+https://github.com/tinygrad/tinygrad.git

# Verify CUDA is working
python3 -c "from tinygrad import Device; print(Device.DEFAULT)"
# Should print: CUDA

# Test GPU memory
nvidia-smi --query-gpu=memory.free --format=csv
```

## 📊 Performance Tips

1. **Use SXM2 GPUs** - NVLink provides faster inter-GPU communication
2. **Match GPU types** - Heterogeneous clusters work but homogeneous is faster
3. **10GbE+ networking** - For multi-node clusters, network is the bottleneck
4. **Disable TUI** - `--disable-tui` reduces overhead

## 🔗 Related Projects

| Project | Description |
|---------|-------------|
| [nvidia-power8-patches](https://github.com/Scottcjn/nvidia-power8-patches) | NVIDIA drivers for IBM POWER8 |
| [cuda-power8-patches](https://github.com/Scottcjn/cuda-power8-patches) | CUDA toolkit for POWER8 |
| [llama-cpp-power8](https://github.com/Scottcjn/llama-cpp-power8) | llama.cpp on POWER8 |

## 🙏 Credits

- Original [exo](https://github.com/exo-explore/exo) by [exo labs](https://x.com/exolabs)
- [tinygrad](https://github.com/tinygrad/tinygrad) for the CUDA backend
- NVIDIA for CUDA toolkit

## 📜 License

GPL-3.0 (same as original exo)

---

<div align="center">

**Maintained by [Elyan Labs](https://elyanlabs.ai)**

*Distributed NVIDIA inference that actually works*

[Report Issues](https://github.com/Scottcjn/exo-cuda/issues) • [Original exo](https://github.com/exo-explore/exo)

</div>
