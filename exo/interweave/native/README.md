# Native CUDA Matmul Offload Server

`gpu_server_ms.cu` is the production matmul offload server that runs on the
Dell C4130 (V100 + M40) and serves GEMM requests from the POWER8 llama.cpp
client over the 40GbE link. It is the native successor to the Python
reference server at `../matmul_server.py`.

## Why a native server

The Python servers process one request at a time. This one gives every
connection its own CUDA lane (cuBLAS handle + stream + scratch buffers), so
copy-in, GEMM, and copy-out overlap across clients. A shared Q4_K weight
cache is refcounted, so LRU eviction can never free a weight an in-flight
GEMM is still reading.

Key properties:

- Per-connection CUDA lanes for overlapped execution
- Refcounted, LRU-evicted Q4_K weight cache (budget via env)
- On-GPU Q4_K dequant kernel (144-byte blocks, 256 values per block)
- FP16 tensor-op math on compute capability 7.0+ (V100); falls back to
  default math on older cards like the M40 (CC 5.2)
- Wire protocol v4, magic `0x47505534` ("GPU4")

## Build

```bash
nvcc -O3 -std=c++17 gpu_server_ms.cu -o gpu_server_ms -lcublas -lpthread
```

Verified with CUDA 12.0 on Ubuntu. The nvlink warning about a static
libpthread is harmless.

## Run

```bash
# argv1 = port (default 8097)
# GPU_DEVICE selects the CUDA device index
# GPU_CACHE_BUDGET_MB caps the resident Q4_K weight cache
GPU_DEVICE=0 GPU_CACHE_BUDGET_MB=1500 ./gpu_server_ms 8099
```

For multi-GPU boxes, run one instance per GPU on its own port and shard
clients across them.

## Relationship to the Python server

`../matmul_server.py` is the readable reference implementation with the
CPU/NumPy fallback and the CPU-runnable test suite. The wire contract for
single requests is shared lineage (magic-discriminated binary TCP), but this
server speaks the newer v4 protocol with weight caching. Use the Python
server to understand or test the protocol; use this one in production.
