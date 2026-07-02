# Interweave Protocol

Backend-agnostic distributed inference protocol for heterogeneous compute clusters.

## Overview

Interweave enables different hardware (CUDA GPUs, Power8 CPUs, Apple Silicon) to participate in the same inference pipeline by treating tensor operations as abstract math that can be serialized, transferred, and reinterpreted on any backend.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      INTERWEAVE PROTOCOL                            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Node A     │    │   Node B     │    │   Node C     │          │
│  │  (V100 CUDA) │◄──►│  (Power8 CPU)│◄──►│  (M4 MLX)    │          │
│  │  tinygrad    │    │  llama.cpp   │    │  mlx         │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                             │                                       │
│                    ┌────────▼────────┐                              │
│                    │ Universal Tensor │                              │
│                    │     Format       │                              │
│                    └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from exo.interweave import (
    InterweaveRouter,
    InterweaveShard,
    UniversalTensor,
    create_model_shards,
)

# Create router with auto-detected backend
router = InterweaveRouter(node_id="my-node")

# Add remote peers
await router.add_peer("power8", "192.168.0.50:50051")

# Create shards for split inference
shards = create_model_shards(
    model_id='llama-3.1-70b',
    n_layers=80,
    splits=[
        (0, 'tinygrad_cuda'),   # Layers 0-39 on GPU
        (40, 'llama_cpp'),       # Layers 40-79 on CPU
    ],
)

# Route computation (auto-selects optimal backend)
output, state, backend = await router.route_forward(
    shards[0],
    input_tensor,
    state
)
```

## Components

### UniversalTensor

Backend-agnostic tensor format that can be converted to/from any backend.

```python
from exo.interweave import UniversalTensor, DType
import numpy as np

# Create from numpy
arr = np.random.randn(1, 4096).astype(np.float16)
tensor = UniversalTensor.from_numpy(arr)

# Convert to different backends
numpy_arr = tensor.to_numpy()
# tinygrad_tensor = tensor.to_tinygrad()  # if tinygrad available
# mlx_array = tensor.to_mlx()  # if mlx available

# Serialize for network transfer
data = tensor.serialize()
restored = UniversalTensor.deserialize(data)

# Convert dtype
tensor_f32 = tensor.convert_dtype(DType.F32)
```

**Supported dtypes:**
- `f32` - 32-bit float
- `f16` - 16-bit float
- `bf16` - bfloat16
- `i8` - 8-bit integer (quantized)
- `i4` - 4-bit integer (quantized)
- `i32` - 32-bit integer (for token IDs)

### InterweaveShard

Extended shard definition with backend preferences and resource estimates.

```python
from exo.interweave import InterweaveShard

shard = InterweaveShard(
    model_id='llama-3.1-70b',
    start_layer=0,
    end_layer=39,
    n_layers=80,
    preferred_backends=('tinygrad_cuda', 'llama_cpp'),
    required_dtype='f16',  # Optional: force specific dtype
    memory_estimate=20 * 1024**3,  # 20GB
    is_embedding=True,
    is_output=False,
)

# Split shard at layer boundary
first, second = shard.split(at_layer=20)

# Check backend compatibility
from exo.interweave import BackendRegistry
backend = BackendRegistry.create_backend('tinygrad_cuda')
if shard.compatible_with(backend):
    print("Backend can handle this shard")
```

### InterweaveState

Synchronized inference state for KV-cache management across backends.

```python
from exo.interweave import InterweaveState, UniversalTensor
import numpy as np

# Create state
state = InterweaveState(request_id="inference-001")
state.sequence_position = 100

# Add KV-cache for layers
for layer in range(40):
    k = UniversalTensor.from_numpy(np.zeros((1, 64, 100, 128), dtype=np.float16))
    v = UniversalTensor.from_numpy(np.zeros((1, 64, 100, 128), dtype=np.float16))
    state.set_cache_for_layer(layer, k, v)

# Serialize for transfer (with compression)
data = state.serialize(compress=True)
print(f"State size: {len(data) / 1024**2:.1f} MB")

# Deserialize on receiving node
restored = InterweaveState.deserialize(data)
```

### BackendRegistry

Runtime backend discovery and selection.

```python
from exo.interweave import BackendRegistry

# Detect available backends
available = BackendRegistry.detect_available()
print(f"Available: {available}")
# Output: ['tinygrad_cuda', 'tinygrad_cpu'] or ['llama_cpp', 'tinygrad_cpu']

# Create backend instance
backend = BackendRegistry.create_backend('tinygrad_cuda')

# Select optimal backend for a shard
optimal = BackendRegistry.select_optimal(shard, available)
```

### InterweaveRouter

Routes computation to optimal backends across the cluster.

```python
from exo.interweave import InterweaveRouter

# Create router (auto-detects local backend)
router = InterweaveRouter(
    node_id="v100-node",
    prefer_local=True,  # Prefer local execution when possible
)

# Add peer nodes
await router.add_peer(
    node_id="power8",
    address="192.168.0.50:50051",
)

# Find candidates for a shard
candidates = await router.find_candidates(shard)
for c in candidates:
    print(f"{c.node_id}: {c.backend_name}, score={c.score}")

# Route forward pass
output, new_state, backend_used = await router.route_forward(
    shard, input_tensor, state
)

# Get cluster status
status = await router.get_cluster_status()
```

## Available Backends

### tinygrad_cuda

NVIDIA GPU inference via tinygrad.

```python
from exo.interweave.backends import TinygradCudaBackend

backend = TinygradCudaBackend(device='CUDA')
# Properties:
#   name: 'tinygrad_cuda'
#   device_type: 'cuda'
#   supported_dtypes: ['f32', 'f16', 'bf16']
#   preferred_dtype: 'f16'
```

### tinygrad_cpu

CPU fallback using tinygrad.

```python
from exo.interweave.backends import TinygradCpuBackend

backend = TinygradCpuBackend()
# Properties:
#   name: 'tinygrad_cpu'
#   device_type: 'cpu'
```

### llama_cpp

llama.cpp for quantized CPU inference (ideal for high-memory systems like Power8).

```python
from exo.interweave.backends import LlamaCppBackend

backend = LlamaCppBackend(
    server_url="http://localhost:8080",
    n_threads=128,  # Power8 has many cores
    n_ctx=8192,
)
# Properties:
#   name: 'llama_cpp'
#   device_type: 'cpu'
#   supported_dtypes: ['f32', 'f16', 'i8', 'i4']
#   preferred_dtype: 'i4'  # Q4_K_M quantization
```

## Wire Protocol

The Interweave protocol uses gRPC for node communication.

### Proto Definition

See `exo/interweave/proto/interweave.proto` for the full definition.

Key messages:
- `UniversalTensorProto` - Serialized tensor with shape, dtype, data
- `InterweaveShardProto` - Shard definition with backend preferences
- `InterweaveStateProto` - KV-cache state for transfer
- `ForwardRequest/Response` - Forward pass RPC

### Service Methods

```protobuf
service InterweaveService {
    rpc Forward(ForwardRequest) returns (ForwardResponse);
    rpc StreamForward(ForwardRequest) returns (stream StreamForwardResponse);
    rpc Generate(GenerateRequest) returns (GenerateResponse);
    rpc HealthCheck(InterweaveHealthRequest) returns (InterweaveHealthResponse);
    rpc GetTopology(GetTopologyRequest) returns (GetTopologyResponse);
}
```

## Target Architecture: V100 + Power8

The primary use case is split inference between:
- **V100 GPU** (16GB VRAM): Compute-intensive layers
- **Power8 CPU** (576GB RAM): Memory-overflow layers with quantization

```
┌─────────────────────────┐     ┌─────────────────────────┐
│    192.168.0.161        │     │    192.168.0.50         │
│    Dell C4130           │     │    IBM Power8           │
│    V100 16GB            │◄───►│    576GB RAM            │
│    tinygrad CUDA        │     │    llama.cpp CPU        │
│    Layers 0-39 (f16)    │     │    Layers 40-79 (i4)    │
└─────────────────────────┘     └─────────────────────────┘
```

### Performance Characteristics

| Metric | Value |
|--------|-------|
| Hidden state transfer | 16 KB per token |
| Serialization time | 0.03 ms |
| Network transfer @10Gbps | 0.01 ms |
| Deserialization time | 0.03 ms |
| **Total overhead** | **~0.07 ms/token** |
| Overhead at 50 tok/s | 0.4% |

### Memory Split Example (Llama 70B)

```python
from exo.interweave import create_model_shards

shards = create_model_shards(
    model_id='llama-3.1-70b',
    n_layers=80,
    splits=[
        (0, 'tinygrad_cuda'),   # V100: 72.5 GB @ f16
        (40, 'llama_cpp'),       # Power8: 72.5 GB @ f16 (or ~18 GB @ i4)
    ],
)
```

## Matmul Offload Servers

Besides full shard routing, Interweave includes a lighter offload path:
the model stays on the big-memory host (POWER8, 512GB RAM) and only the
matrix multiplications are shipped to a GPU box over a binary TCP protocol.

Two servers implement this:

### matmul_server.py (reference, tested)

One canonical Python server that replaced the ~20 prototypes this path was
developed through. Dtype dispatch for F32, F16, Q8_0, Q4_K, and MXFP4 with
correct ggml block sizes, input validation, per-recv socket timeouts, a GPU
serialization lock, a localhost-bound health/metrics HTTP sidecar, and a
CPU/NumPy fallback so it runs anywhere.

```bash
# Start the server (CUDA if available, CPU fallback otherwise)
python exo/interweave/matmul_server.py --port 8096

# Health check
curl http://127.0.0.1:8097/health
```

### native/gpu_server_ms.cu (production)

Multi-stream native CUDA server: per-connection cuBLAS lanes so requests
overlap, plus a refcounted Q4_K weight cache. This is what serves the
POWER8 to C4130 offload in production. See `native/README.md` for build
and run instructions.

## Testing

Run the test suite:

```bash
cd /path/to/exo_cuda
source .venv/bin/activate

# Matmul offload server suite (CPU-runnable, no GPU required)
cd exo/interweave && python -m pytest test_matmul_server.py test_gpu_cpu_parity.py

# Run all protocol tests
python -m exo.interweave.test_interweave

# Simple component test
python -m exo.interweave.simple_test

# Live inference test (requires model)
python -m exo.interweave.live_inference_test
```

## Files

```
exo/interweave/
├── __init__.py           # Package exports
├── tensor_format.py      # UniversalTensor class
├── shard.py              # InterweaveShard dataclass
├── state.py              # InterweaveState (KV-cache)
├── backend.py            # Backend ABC + Registry
├── router.py             # InterweaveRouter
├── proto/
│   ├── interweave.proto  # Wire protocol definition
│   ├── interweave_pb2.py # Generated protobuf
│   └── interweave_pb2_grpc.py  # Generated gRPC
├── backends/
│   ├── __init__.py
│   ├── llamacpp.py       # llama.cpp backend
│   └── tinygrad_cuda.py  # TinyGrad CUDA/CPU backends
├── matmul_server.py      # Unified matmul offload server (reference)
├── native/
│   ├── gpu_server_ms.cu  # Production multi-stream CUDA server
│   └── README.md         # Build and run instructions
├── test_matmul_server.py # Matmul server suite (CPU-runnable)
├── test_gpu_cpu_parity.py # GPU vs CPU numerical parity
├── test_interweave.py    # Test suite
├── simple_test.py        # Quick component test
└── live_inference_test.py # Full inference test
```

## Future Work

- [ ] MLX Metal backend for Apple Silicon
- [ ] ROCm backend for AMD GPUs
- [ ] Automatic shard placement optimization
- [ ] Pipeline parallelism (overlapping computation/transfer)
- [ ] Tensor parallelism support
- [ ] Dynamic load balancing

## License

Same as exo - GPL-3.0
