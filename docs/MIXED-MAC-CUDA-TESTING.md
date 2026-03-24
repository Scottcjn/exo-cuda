# Mixed Mac and CUDA Environment Testing Guide

This guide documents testing exo-cuda in mixed environments with both Apple Silicon (MLX) and NVIDIA CUDA nodes.

## 🎯 Issue Reference

Addresses: [Issue #1 - Mixed Mac and CUDA environment?](https://github.com/Scottcjn/exo-cuda/issues/1)

## 📋 Test Environment

### Network Topology
```
┌─────────────────────────────────────────────────────────────┐
│                    Local Network (LAN)                       │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ RTX 3090     │    │ Mac Mini M4  │    │ Mac Mini M1  │  │
│  │ Windows/WSL  │    │ macOS        │    │ macOS        │  │
│  │ CUDA 12.x    │    │ MLX          │    │ MLX          │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │          │
│         └───────────────────┼───────────────────┘          │
│                             │                               │
│                    ┌────────▼────────┐                     │
│                    │  UDP Broadcast  │                     │
│                    │  Auto-Discovery │                     │
│                    └─────────────────┘                     │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐                       │
│  │ iMac M1      │    │ M3/M4 Laptop │                       │
│  │ macOS        │    │ (Mobile)     │                       │
│  │ MLX          │    │ MLX          │                       │
│  └──────────────┘    └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### Hardware Configuration

| Device | GPU | Backend | VRAM | Role |
|--------|-----|---------|------|------|
| RTX 3090 (Windows/WSL) | NVIDIA CUDA | tinygrad | 24GB | Primary Node |
| Mac Mini M4 | Apple Silicon | MLX | 16GB | Worker Node |
| Mac Mini M1 | Apple Silicon | MLX | 8GB | Worker Node |
| iMac M1 | Apple Silicon | MLX | 8GB | Worker Node |
| M3/M4 Laptops | Apple Silicon | MLX | 8-16GB | Mobile Nodes |

## 🔧 Setup Instructions

### 1. CUDA Node (Windows/WSL with RTX 3090)

```bash
# In WSL2 (Ubuntu)
# Install CUDA toolkit
sudo apt update
sudo apt install -y nvidia-cuda-toolkit

# Verify CUDA
nvcc --version
nvidia-smi

# Clone and install exo-cuda
git clone https://github.com/Scottcjn/exo-cuda.git
cd exo-cuda
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Upgrade tinygrad for latest CUDA fixes
pip install --upgrade git+https://github.com/tinygrad/tinygrad.git

# Start as primary node
exo --inference-engine tinygrad --chatgpt-api-port 8001 --disable-tui
```

### 2. Mac Nodes (M1/M2/M3/M4)

```bash
# On each Mac
git clone https://github.com/exo-explore/exo.git
cd exo
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Start as worker node
exo --inference-engine mlx --disable-tui
```

## 🌐 Network Configuration

### Firewall Rules

**Windows (RTX 3090 node)**:
```powershell
# Allow UDP broadcast discovery (port 5678)
New-NetFirewallRule -DisplayName "exo Discovery" -Direction Inbound -Protocol UDP -LocalPort 5678 -Action Allow

# Allow TCP for node communication
New-NetFirewallRule -DisplayName "exo Node" -Direction Inbound -Protocol TCP -LocalPort 5678 -Action Allow
```

**macOS (Mac nodes)**:
```bash
# System Preferences → Security & Privacy → Firewall
# Allow Python to accept incoming connections
```

### Verify Connectivity

```bash
# From CUDA node, ping Mac nodes
ping <mac-mini-m4-ip>
ping <mac-mini-m1-ip>

# Test UDP broadcast discovery
# All nodes should see each other automatically
```

## 🧪 Testing Procedure

### Step 1: Start Primary Node (CUDA)

```bash
# On RTX 3090 (Windows/WSL)
cd exo-cuda
source .venv/bin/activate
exo --inference-engine tinygrad --chatgpt-api-port 8001 --disable-tui
```

**Expected Output**:
```
🟢 Listening on 0.0.0.0:5678
🔍 Discovering peers...
📦 Available models: llama-3.2-1b, llama-3.2-3b, ...
🌐 API: http://0.0.0.0:8001/v1/chat/completions
```

### Step 2: Start Mac Worker Nodes

```bash
# On each Mac (M4, M1, iMac, laptops)
cd exo
source .venv/bin/activate
exo --inference-engine mlx --disable-tui
```

**Expected Output**:
```
🟢 Listening on 0.0.0.0:5678
🔍 Discovering peers...
🔗 Connected to: 192.168.1.100:5678 (RTX 3090)
📦 Ready for inference
```

### Step 3: Verify Cluster Formation

```bash
# Check connected nodes from any node
curl http://localhost:8001/v1/models

# Or check logs for peer discovery
# Should see all nodes listed
```

### Step 4: Run Inference Test

```bash
# Test with a small model first
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-1b",
    "messages": [{"role": "user", "content": "Hello from mixed environment!"}]
  }'
```

## 📊 Performance Expectations

| Model | CUDA Only | Mixed (CUDA+MLX) | Notes |
|-------|-----------|------------------|-------|
| llama-3.2-1b | ~50 t/s | ~35 t/s | Cross-backend overhead |
| llama-3.2-3b | ~30 t/s | ~20 t/s | Good for testing |
| llama-3.1-8b | ~15 t/s | ~10 t/s | Requires multiple nodes |

## ⚠️ Known Issues & Solutions

### Issue 1: Nodes Don't Discover Each Other

**Symptoms**: Mac nodes don't see CUDA node

**Solution**:
```bash
# Manual peer configuration
# Create peers.json on each node
echo '{"peers": ["192.168.1.100:5678", "192.168.1.101:5678"]}' > peers.json

# Restart nodes
```

### Issue 2: WSL2 Network Isolation

**Symptoms**: WSL2 can't communicate with host network

**Solution**:
```bash
# In WSL2, add route to local network
sudo ip route add 192.168.1.0/24 dev eth0

# Or use WSL2 mirror mode (Windows 11)
# .wslconfig configuration
```

### Issue 3: Model Compatibility

**Symptoms**: Models fail to load across different backends

**Solution**:
- Use models supported by both tinygrad and MLX
- Start with llama-3.2-1b or llama-3.2-3b
- Ensure all nodes have same model versions

## 📝 Test Results Template

```markdown
## Test Run: YYYY-MM-DD

**Environment**:
- CUDA Node: RTX 3090, Windows 11 + WSL2, CUDA 12.x
- Mac Nodes: M4 (16GB), M1 (8GB), iMac M1 (8GB)
- Network: Gigabit LAN

**Results**:
- ✅ Node discovery: Automatic (UDP broadcast)
- ✅ Cluster formation: 4/4 nodes connected
- ✅ Inference test: llama-3.2-1b @ ~35 t/s
- ✅ Cross-backend: Working (tinygrad + MLX)

**Issues Encountered**:
- None / [describe]

**Notes**:
- [any observations]
```

## 🚀 Advanced: Dynamic Node Joining/Leaving

For M3/M4 laptops entering/leaving the network:

1. **Auto-discovery**: Nodes automatically discover via UDP broadcast
2. **Graceful leave**: No special handling needed, cluster adapts
3. **Rejoin**: Simply restart `exo` on the laptop

```bash
# On laptop (when returning to network)
exo --inference-engine mlx --disable-tui
# Automatically rejoins existing cluster
```

## 📚 References

- [exo-cuda README](../README.md)
- [Multi-Node Setup](../README.md#-multi-node-cluster)
- [Original exo (MLX)](https://github.com/exo-explore/exo)
- [tinygrad CUDA Support](https://github.com/tinygrad/tinygrad)

---

**Contributed by**: [@Dlove123](https://github.com/Dlove123)
**Date**: 2026-03-24
**Issue**: [#1](https://github.com/Scottcjn/exo-cuda/issues/1)
