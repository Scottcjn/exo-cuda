#!/usr/bin/env python3
"""
Interweave GPU Matmul Offload Server — unified, canonical edition.

This single module is the canonical successor to the ~20 prototype matmul_*.py
servers that accumulated in exo/interweave/ during development. Deleting those
prototypes is a deliberate, separate cutover step (see DEPLOY notes) — they are
left in place until every client is confirmed on this server, so this file
landing does not by itself remove them. It is the one production server for the
Interweave heterogeneous-offload path: a POWER8 (or any CPU)
client streams weight tensors over TCP, a CUDA GPU here dequantizes
+ multiplies them, and results stream back (F32 by default — see WIRE_ORDER).

Design goals (why this file exists):
  * ONE source of truth per quant kernel — no more 148-vs-144 byte drift.
  * Correct ggml block sizes:  Q4_K=144B/256, Q8_0=34B/32, MXFP4=17B/32.
  * Row-major (WIRE_ORDER="C") + F32 result, matching the real POWER8 client
    (ggml-gpu-offload-local.h); the abandoned FP16 experiments are still
    selectable via RESULT_DTYPE but are not the default.
  * Dtype dispatch via a registry, not if/elif ladders — trivially extensible.
  * Input validation: a bad M/N/K header is rejected, never blindly allocated
    (the old servers would OOM the GPU on a malformed request — a DoS on a
    public-facing port).
  * Graceful CPU/NumPy fallback when cupy/CUDA is unavailable, so the server
    still serves correct (if slower) results and CI can test it GPU-free.
  * Backward wire compatibility: the existing single-op GPU3 protocol AND a
    batched protocol are both served on the same port, discriminated by magic.
  * A /health + /metrics HTTP sidecar so operators and contributors can see
    state without reading the binary stream.

NUMERICAL CONTRACT NOTE (read before "fixing" the kernels):
  The Q4_K decode below mirrors the prior working POWER8 path verbatim. It uses
  one 6-bit super-block value as both scale and min multiplier. Stock ggml
  get_scale_min_k4 packs 8 *separate* scales and mins. These differ. This file
  deliberately preserves the existing behavior so the live path is not broken on
  an untested change; bringing the decode to full stock-ggml parity is a tracked
  follow-up that must be validated end-to-end against the llama.cpp client with a
  live GPU. Do not change the dequant math here without that validation.
"""

import argparse
import json
import socket
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# --------------------------------------------------------------------------- #
# ggml type ids (must match the client's GGML_TYPE_* constants)
# --------------------------------------------------------------------------- #
TYPE_F32 = 0
TYPE_F16 = 1
TYPE_Q8_0 = 8
TYPE_Q4_K = 12
TYPE_MXFP4 = 39

TYPE_NAMES = {
    TYPE_F32: "F32",
    TYPE_F16: "F16",
    TYPE_Q8_0: "Q8_0",
    TYPE_Q4_K: "Q4_K",
    TYPE_MXFP4: "MXFP4",
}

# (elements_per_block, bytes_per_block) for the quantized formats.
BLOCK_SPEC = {
    TYPE_Q4_K: (256, 144),
    TYPE_Q8_0: (32, 34),
    TYPE_MXFP4: (32, 17),
}

# Wire protocol magics.
MAGIC_GPU3 = 0x47505533   # "GPU3" — single-op request (legacy, kept compatible)
MAGIC_BATCH = 0x42415443  # "BATC" — batched request

# Safety caps. A single dimension above this, or a per-operand byte size above
# MAX_OPERAND_BYTES, is rejected before any allocation.
MAX_DIM = 1 << 17          # 131072
MAX_OPERAND_BYTES = 4 << 30  # 4 GiB — bounds a single wire read
MAX_ELEMS = MAX_OPERAND_BYTES // 4  # bounds each F32 working matrix (A, B, C)

# Wire layout contract, pinned to the real POWER8 client (ggml-gpu-offload-local.h):
#   * request:  24-byte header {magic, M, N, K, A_type, B_type} then A then B bytes
#   * operands and result are ROW-MAJOR (C-order) — raw ggml tensor buffers
#   * response: 16-byte header {magic, status, M, N} THEN the result payload
#     (the client reads resp[4] first, checks status, then recv(C, M*N*elem))
#   * the result dtype defaults to F32 (client does recv(C, M*N*sizeof(float)))
# RESULT_DTYPE may be set to "f16" to drive the legacy bandwidth-halving
# experiments, but "f32" is the default because it matches the live client.
# NOTE: a client built for f16 results must agree out-of-band — the 16-byte
# response header carries no element-size field.
WIRE_ORDER = "C"
RESULT_DTYPE = "f32"  # "f32" | "f16"

# Per-connection socket timeout (seconds). This is a PER-recv idle timeout (it
# resets whenever recv() returns bytes), NOT a total-transfer deadline — an
# actively streaming multi-GiB operand never trips it; only a recv that blocks
# this long with zero bytes does. Without it, a client that connects and sends a
# partial request (or nothing) pins a daemon thread forever (tri-brain DoS).
CLIENT_TIMEOUT = 300

# Dequant byte-unpacking (and the CUDA kernels) assume a little-endian host.
# Every host that runs this — x86_64 and ppc64le — is LE. A big-endian host
# would silently produce wrong scales, so fail loudly at import time instead.
if sys.byteorder != "little":
    raise RuntimeError(
        f"matmul_server requires a little-endian host; got {sys.byteorder}-endian. "
        "The dequant kernels read fp16 scales as LE bytes."
    )

# --------------------------------------------------------------------------- #
# GPU backend (cupy) — optional. CUDA kernels are ported verbatim from the
# reconciled mxfp4 prototype (the most complete/correct of the variants).
# --------------------------------------------------------------------------- #
HAS_GPU = False
cp = None
_GPU_LOCK = threading.Lock()  # serializes GPU compute across client threads

_Q4K_KERNEL = None
_Q8_0_KERNEL = None
_MXFP4_KERNEL = None

# Known GPU/CPU parity edge: the CUDA kernels decode fp16 block scales manually
# and flush fp16 subnormals (exp==0) to zero, whereas the NumPy CPU path uses
# native np.float16 which preserves them. Quant scales are effectively never
# subnormal (|d| < 2^-14), and any divergence stays within the parity test's
# 1e-3 tolerance, but exact bit-parity is therefore not claimed for subnormal
# scales. Bringing the kernels to native-fp16 decode is part of the same
# full-ggml-Q4_K follow-up that needs a live GPU + client to validate.


def _init_gpu():
    """Try to bring up cupy + compile kernels. Returns True on success."""
    global HAS_GPU, cp, _Q4K_KERNEL, _Q8_0_KERNEL, _MXFP4_KERNEL
    try:
        import cupy as _cp
        # Touch the runtime so a present-but-broken driver fails here, not mid-request.
        _cp.cuda.runtime.getDeviceCount()
    except Exception as exc:  # ImportError, CUDARuntimeError, ...
        print(f"[matmul] GPU unavailable ({exc.__class__.__name__}: {exc}); using CPU fallback")
        HAS_GPU = False
        return False

    cp = _cp

    _Q4K_KERNEL = cp.RawKernel(r"""
    #define FLOAT_INF 1e30f
    extern "C" __global__ void dequant_q4k(
        const unsigned char* data, float* output, const int n_blocks
    ) {
        const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (block_idx >= n_blocks) return;
        const unsigned char* block = data + block_idx * 144;
        float* out = output + block_idx * 256;
        unsigned short d_bits = block[0] | (block[1] << 8);
        unsigned short dmin_bits = block[2] | (block[3] << 8);
        unsigned int d_sign = (d_bits >> 15) & 1;
        unsigned int d_exp = (d_bits >> 10) & 0x1F;
        unsigned int d_mant = d_bits & 0x3FF;
        float d;
        if (d_exp == 0) d = 0.0f;
        else if (d_exp == 31) d = d_sign ? -FLOAT_INF : FLOAT_INF;
        else { d_exp = d_exp - 15 + 127; unsigned int f = (d_sign << 31) | (d_exp << 23) | (d_mant << 13); d = __int_as_float(f); }
        unsigned int dmin_sign = (dmin_bits >> 15) & 1;
        unsigned int dmin_exp = (dmin_bits >> 10) & 0x1F;
        unsigned int dmin_mant = dmin_bits & 0x3FF;
        float dmin;
        if (dmin_exp == 0) dmin = 0.0f;
        else if (dmin_exp == 31) dmin = dmin_sign ? -FLOAT_INF : FLOAT_INF;
        else { dmin_exp = dmin_exp - 15 + 127; unsigned int f = (dmin_sign << 31) | (dmin_exp << 23) | (dmin_mant << 13); dmin = __int_as_float(f); }
        const unsigned char* scales = block + 4;
        const unsigned char* qs = block + 16;
        for (int j = 0; j < 8; j++) {
            int byte_idx = (j * 6) / 8;
            int bit_offset = (j * 6) % 8;
            unsigned int sc = scales[byte_idx] >> bit_offset;
            if (bit_offset > 2 && byte_idx < 11) sc |= scales[byte_idx + 1] << (8 - bit_offset);
            sc &= 0x3F;
            float scale = d * (float)sc;
            float min_val = dmin * (float)sc;
            for (int k = 0; k < 16; k++) {
                unsigned char q = qs[j * 16 + k];
                out[j * 32 + k * 2]     = scale * (float)(q & 0xF) - min_val;
                out[j * 32 + k * 2 + 1] = scale * (float)(q >> 4) - min_val;
            }
        }
    }
    """, "dequant_q4k")

    _Q8_0_KERNEL = cp.RawKernel(r"""
    extern "C" __global__ void dequant_q8_0(const unsigned char* data, float* output, int n_blocks) {
        int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (block_idx >= n_blocks) return;
        const unsigned char* block = data + block_idx * 34;
        float* out = output + block_idx * 32;
        unsigned short d_raw = block[0] | (block[1] << 8);
        unsigned int sign = (d_raw >> 15) & 1;
        unsigned int exp = (d_raw >> 10) & 0x1F;
        unsigned int mant = d_raw & 0x3FF;
        float d;
        if (exp == 0) d = 0.0f;
        else if (exp == 31) d = (sign ? -1e30f : 1e30f);
        else { exp = exp - 15 + 127; unsigned int f = (sign << 31) | (exp << 23) | (mant << 13); d = __int_as_float(f); }
        for (int i = 0; i < 32; i++) { signed char q = (signed char)block[2 + i]; out[i] = d * (float)q; }
    }
    """, "dequant_q8_0")

    _MXFP4_KERNEL = cp.RawKernel(r"""
    extern "C" __global__ void dequant_mxfp4(const unsigned char* data, float* output, int n_blocks) {
        __shared__ signed char kvalues[16];
        if (threadIdx.x < 16) {
            const signed char table[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
            kvalues[threadIdx.x] = table[threadIdx.x];
        }
        __syncthreads();
        int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (block_idx >= n_blocks) return;
        const unsigned char* block = data + block_idx * 17;
        float* out = output + block_idx * 32;
        unsigned char e = block[0];
        float scale;
        if (e < 2) { unsigned int bits = 0x00200000u << e; scale = __int_as_float(bits); }
        else { unsigned int bits = (unsigned int)(e - 1) << 23; scale = __int_as_float(bits); }
        for (int j = 0; j < 16; j++) {
            unsigned char qs = block[1 + j];
            out[j]      = (float)kvalues[qs & 0x0F] * scale;
            out[j + 16] = (float)kvalues[qs >> 4]   * scale;
        }
    }
    """, "dequant_mxfp4")

    HAS_GPU = True
    print(f"[matmul] GPU ready: cupy {cp.__version__}, kernels compiled")
    return True


# --------------------------------------------------------------------------- #
# Size accounting + validation
# --------------------------------------------------------------------------- #
def operand_bytes(dtype: int, rows: int, cols: int) -> int:
    """Expected wire byte size for a (rows x cols) operand of the given dtype."""
    n = rows * cols
    if dtype in BLOCK_SPEC:
        elems, blk_bytes = BLOCK_SPEC[dtype]
        n_blocks = (n + elems - 1) // elems
        return n_blocks * blk_bytes
    if dtype == TYPE_F16:
        return n * 2
    if dtype == TYPE_F32:
        return n * 4
    raise ValueError(f"unsupported dtype {dtype}")


def validate_op(M: int, N: int, K: int, a_type: int, b_type: int) -> str:
    """Return '' if the op is acceptable, else a human-readable rejection reason."""
    for name, v in (("M", M), ("N", N), ("K", K)):
        if v <= 0:
            return f"{name}={v} must be positive"
        if v > MAX_DIM:
            return f"{name}={v} exceeds MAX_DIM={MAX_DIM}"
    for label, t in (("A", a_type), ("B", b_type)):
        if t not in TYPE_NAMES:
            return f"operand {label} has unsupported dtype {t}"
    # Quantized operands MUST be block-aligned along their ROW WIDTH (the inner
    # dimension), because ggml quant blocks are row-local: a (rows x cols)
    # quantized tensor stores each row as cols/elems independent blocks. Checking
    # rows*cols would wrongly accept e.g. Q4_K M=2,K=128 (256 total elems == one
    # block) even though each 128-wide row is only half a block — the dequantizer
    # would then read blocks across row boundaries and return garbage.
    # A is (M x K) -> row width K;  B is (K x N) -> row width N. (tri-brain L2)
    for label, t, width in (("A", a_type, K), ("B", b_type, N)):
        if t in BLOCK_SPEC:
            elems = BLOCK_SPEC[t][0]
            if width % elems != 0:
                return (f"operand {label} row width {width} not a multiple of "
                        f"{TYPE_NAMES[t]} block size {elems}")
    try:
        sa = operand_bytes(a_type, M, K)
        sb = operand_bytes(b_type, K, N)
    except ValueError as exc:
        return str(exc)
    if sa > MAX_OPERAND_BYTES:
        return f"A wire size {sa} exceeds MAX_OPERAND_BYTES={MAX_OPERAND_BYTES}"
    if sb > MAX_OPERAND_BYTES:
        return f"B wire size {sb} exceeds MAX_OPERAND_BYTES={MAX_OPERAND_BYTES}"
    # Bound the F32 WORKING SET, not just the wire size. Every operand is
    # dequantized to F32 and the result C is computed in F32 before any cast, so
    # a small-wire request (quantized, or small-K) can still force huge F32
    # allocations for A, B, or C and OOM the box. result_bytes_per_elem() on the
    # wire is irrelevant here — compute is always F32. (tri-brain B2 + L2)
    for label, r, c in (("A(f32)", M, K), ("B(f32)", K, N), ("result", M, N)):
        if r * c > MAX_ELEMS:
            return f"{label} {r}x{c} exceeds element cap {MAX_ELEMS}"
    return ""


# --------------------------------------------------------------------------- #
# CPU (NumPy) dequantizers — mirror the CUDA kernels bit-for-bit so GPU and CPU
# paths produce identical results. Host is assumed little-endian (x86/POWER8 LE).
# --------------------------------------------------------------------------- #
_MXFP4_TABLE = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
                        dtype=np.float32)


def _cpu_q8_0(data: bytes, rows: int, cols: int) -> np.ndarray:
    n = rows * cols
    nb = (n + 31) // 32
    arr = np.frombuffer(data[:nb * 34], dtype=np.uint8).reshape(nb, 34)
    d = arr[:, :2].copy().view(np.float16).reshape(nb, 1).astype(np.float32)
    q = arr[:, 2:].astype(np.int8).astype(np.float32)  # uint8->int8 reinterprets sign
    out = (d * q).reshape(-1)[:n]
    return out.reshape((rows, cols), order=WIRE_ORDER)


def _cpu_mxfp4(data: bytes, rows: int, cols: int) -> np.ndarray:
    n = rows * cols
    nb = (n + 31) // 32
    arr = np.frombuffer(data[:nb * 17], dtype=np.uint8).reshape(nb, 17)
    e = arr[:, 0].astype(np.uint32)
    bits = np.where(e < 2, (np.uint32(0x00200000) << e), ((e - 1) << 23)).astype(np.uint32)
    scale = bits.view(np.float32).reshape(nb, 1)
    packed = arr[:, 1:]  # (nb, 16)
    low = _MXFP4_TABLE[packed & 0x0F]
    high = _MXFP4_TABLE[packed >> 4]
    out = np.empty((nb, 32), dtype=np.float32)
    out[:, :16] = low * scale
    out[:, 16:] = high * scale
    return out.reshape(-1)[:n].reshape((rows, cols), order=WIRE_ORDER)


def _cpu_q4k(data: bytes, rows: int, cols: int) -> np.ndarray:
    n = rows * cols
    nb = (n + 255) // 256  # ceil — matches operand_bytes; never drops a partial block
    if nb == 0:
        return np.zeros((rows, cols), dtype=np.float32)
    arr = np.frombuffer(data[:nb * 144], dtype=np.uint8).reshape(nb, 144)
    d = arr[:, 0:2].copy().view(np.float16).reshape(nb, 1).astype(np.float32)
    dmin = arr[:, 2:4].copy().view(np.float16).reshape(nb, 1).astype(np.float32)
    scales = arr[:, 4:16].astype(np.uint32)
    qs = arr[:, 16:144]
    sc = np.empty((nb, 8), dtype=np.float32)
    for j in range(8):
        byte_idx = (j * 6) // 8
        bit = (j * 6) % 8
        v = scales[:, byte_idx] >> bit
        if bit > 2 and byte_idx < 11:
            v = v | (scales[:, byte_idx + 1] << (8 - bit))
        sc[:, j] = (v & 0x3F).astype(np.float32)
    scale = d * sc      # (nb, 8)
    minv = dmin * sc    # (nb, 8)
    qs2 = qs.reshape(nb, 8, 16)
    low = (qs2 & 0xF).astype(np.float32)
    high = (qs2 >> 4).astype(np.float32)
    out = np.empty((nb, 8, 32), dtype=np.float32)
    out[:, :, 0::2] = scale[:, :, None] * low - minv[:, :, None]
    out[:, :, 1::2] = scale[:, :, None] * high - minv[:, :, None]
    return out.reshape(-1)[:n].reshape((rows, cols), order=WIRE_ORDER)


def _cpu_plain(data: bytes, rows: int, cols: int, dtype: int) -> np.ndarray:
    np_dtype = np.float16 if dtype == TYPE_F16 else np.float32
    return np.frombuffer(data, dtype=np_dtype).reshape((rows, cols), order=WIRE_ORDER).astype(np.float32)


_CPU_DEQUANT = {
    TYPE_Q8_0: _cpu_q8_0,
    TYPE_Q4_K: _cpu_q4k,
    TYPE_MXFP4: _cpu_mxfp4,
}


# --------------------------------------------------------------------------- #
# GPU dequantizers
# --------------------------------------------------------------------------- #
def _gpu_q8_0(data: bytes, rows: int, cols: int):
    n = rows * cols
    nb = (n + 31) // 32
    g = cp.asarray(np.frombuffer(data[:nb * 34], dtype=np.uint8))
    o = cp.zeros(nb * 32, dtype=cp.float32)
    _Q8_0_KERNEL(((nb + 255) // 256,), (256,), (g, o, nb))
    return o[:n].reshape((rows, cols), order=WIRE_ORDER)


def _gpu_q4k(data: bytes, rows: int, cols: int):
    n = rows * cols
    nb = (n + 255) // 256  # ceil — matches operand_bytes; never drops a partial block
    if nb == 0:
        return cp.zeros((rows, cols), dtype=cp.float32)
    g = cp.asarray(np.frombuffer(data[:nb * 144], dtype=np.uint8))
    o = cp.zeros(nb * 256, dtype=cp.float32)
    _Q4K_KERNEL(((nb + 255) // 256,), (256,), (g, o, nb))
    return o[:n].reshape((rows, cols), order=WIRE_ORDER)


def _gpu_mxfp4(data: bytes, rows: int, cols: int):
    n = rows * cols
    nb = (n + 31) // 32
    g = cp.asarray(np.frombuffer(data[:nb * 17], dtype=np.uint8))
    o = cp.zeros(nb * 32, dtype=cp.float32)
    _MXFP4_KERNEL(((nb + 255) // 256,), (256,), (g, o, nb))
    return o[:n].reshape((rows, cols), order=WIRE_ORDER)


def _gpu_dequant(dtype: int, data: bytes, rows: int, cols: int):
    if dtype == TYPE_Q8_0:
        return _gpu_q8_0(data, rows, cols)
    if dtype == TYPE_Q4_K:
        return _gpu_q4k(data, rows, cols)
    if dtype == TYPE_MXFP4:
        return _gpu_mxfp4(data, rows, cols)
    np_dtype = np.float16 if dtype == TYPE_F16 else np.float32
    host = np.frombuffer(data, dtype=np_dtype).reshape((rows, cols), order=WIRE_ORDER).astype(np.float32)
    return cp.asarray(host)


# --------------------------------------------------------------------------- #
# Core compute: dequant A and B, multiply, serialize the result.
# Result is row-major (WIRE_ORDER) and F32 by default, matching the live client.
# --------------------------------------------------------------------------- #
def _serialize_result(C: np.ndarray) -> bytes:
    out_dtype = np.float16 if RESULT_DTYPE == "f16" else np.float32
    if WIRE_ORDER == "C":
        C = np.ascontiguousarray(C, dtype=out_dtype)
        return C.tobytes(order="C")
    C = np.asfortranarray(C.astype(out_dtype))
    return C.tobytes(order="F")


def result_bytes_per_elem() -> int:
    return 2 if RESULT_DTYPE == "f16" else 4


def compute_matmul(M, N, K, a_type, b_type, data_a, data_b) -> bytes:
    if HAS_GPU:
        # Serialize GPU work: handle_client spawns a thread per connection, but
        # there is one device and all threads share cupy's default stream/pool.
        # Concurrent matmuls would race on memory and risk OOM; the lock keeps
        # GPU memory bounded to one op's working set at a time. (tri-brain L2)
        with _GPU_LOCK:
            A = _gpu_dequant(a_type, data_a, M, K)
            B = _gpu_dequant(b_type, data_b, K, N)
            C = cp.asnumpy(cp.matmul(A, B))
    else:
        A = _CPU_DEQUANT[a_type](data_a, M, K) if a_type in _CPU_DEQUANT \
            else _cpu_plain(data_a, M, K, a_type)
        B = _CPU_DEQUANT[b_type](data_b, K, N) if b_type in _CPU_DEQUANT \
            else _cpu_plain(data_b, K, N, b_type)
        C = np.matmul(A, B)
    return _serialize_result(C)


# --------------------------------------------------------------------------- #
# Shared metrics
# --------------------------------------------------------------------------- #
class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.start = time.time()
        self.requests = 0
        self.rejected = 0
        self.errors = 0
        self.bytes_in = 0
        self.bytes_out = 0
        self.connections = 0
        self.active = 0
        self.by_dtype = {}
        self.last_error = ""

    def record(self, a_type, b_type, nbytes_in, nbytes_out):
        with self.lock:
            self.requests += 1
            self.bytes_in += nbytes_in
            self.bytes_out += nbytes_out
            for t in (a_type, b_type):
                name = TYPE_NAMES.get(t, f"T{t}")
                self.by_dtype[name] = self.by_dtype.get(name, 0) + 1

    def reject(self, reason):
        with self.lock:
            self.rejected += 1
            self.last_error = reason

    def error(self, reason):
        with self.lock:
            self.errors += 1
            self.last_error = reason

    def snapshot(self):
        with self.lock:
            uptime = max(time.time() - self.start, 1e-9)
            gb_in = self.bytes_in / 1e9
            return {
                "status": "ok",
                "gpu": HAS_GPU,
                "uptime_s": round(uptime, 1),
                "connections": self.connections,
                "active_connections": self.active,
                "requests": self.requests,
                "rejected": self.rejected,
                "errors": self.errors,
                "gb_in": round(gb_in, 3),
                "gb_out": round(self.bytes_out / 1e9, 3),
                "throughput_gbps_in": round(gb_in / uptime, 4),
                "by_dtype": dict(self.by_dtype),
                "last_error": self.last_error,
            }


METRICS = Metrics()


# --------------------------------------------------------------------------- #
# TCP framing helpers
# --------------------------------------------------------------------------- #
def recv_exact(conn, n):
    chunks = []
    got = 0
    while got < n:
        chunk = conn.recv(min(n - got, 4 * 1024 * 1024))
        if not chunk:
            return None
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def _read_single_op(conn):
    """Read the remaining 20 bytes of a GPU3 single-op header. Returns tuple or None."""
    rest = recv_exact(conn, 20)
    if not rest:
        return None
    return struct.unpack("<IIIII", rest)  # M, N, K, A_type, B_type


def _serve_single(conn):
    """Handle one GPU3 single-op request. Returns False to close the connection."""
    op = _read_single_op(conn)
    if op is None:
        return False
    M, N, K, a_type, b_type = op
    reason = validate_op(M, N, K, a_type, b_type)
    if reason:
        METRICS.reject(reason)
        print(f"[matmul] reject single-op: {reason}")
        # Well-formed error response so the client doesn't hang: status=1, M=N=0.
        conn.sendall(struct.pack("<IIII", MAGIC_GPU3, 1, 0, 0))
        return False  # framing is now ambiguous (we didn't drain data) -> close
    size_a = operand_bytes(a_type, M, K)
    size_b = operand_bytes(b_type, K, N)
    data_a = recv_exact(conn, size_a)
    data_b = recv_exact(conn, size_b)
    if data_a is None or data_b is None:
        return False
    t0 = time.time()
    result = compute_matmul(M, N, K, a_type, b_type, data_a, data_b)
    conn.sendall(struct.pack("<IIII", MAGIC_GPU3, 0, M, N) + result)
    METRICS.record(a_type, b_type, size_a + size_b, len(result))
    dt = (time.time() - t0) * 1000
    tf = 2 * M * K * N / 1e12 / (dt / 1000) if dt > 0 else 0
    print(f"[matmul] {M}x{K}({TYPE_NAMES.get(a_type)}) @ {K}x{N}({TYPE_NAMES.get(b_type)}) "
          f"-> {M}x{N} in {dt:.1f}ms ({tf:.2f} TFLOPS)")
    return True


def _serve_batch(conn):
    """Handle one batched request (count ops). Returns False to close."""
    hdr = recv_exact(conn, 4)
    if not hdr:
        return False
    (count,) = struct.unpack("<I", hdr)
    if count <= 0 or count > 4096:
        METRICS.reject(f"bad batch count {count}")
        conn.sendall(struct.pack("<II", 1, 0))  # status frame so client sees a rejection, not a bare EOF
        return False
    for _ in range(count):
        op_hdr = recv_exact(conn, 20)
        if not op_hdr:
            return False
        M, N, K, a_type, b_type = struct.unpack("<IIIII", op_hdr)
        reason = validate_op(M, N, K, a_type, b_type)
        if reason:
            METRICS.reject(reason)
            conn.sendall(struct.pack("<II", 1, 0))  # status=1, size=0
            return False
        size_a = operand_bytes(a_type, M, K)
        size_b = operand_bytes(b_type, K, N)
        data_a = recv_exact(conn, size_a)
        data_b = recv_exact(conn, size_b)
        if data_a is None or data_b is None:
            return False
        result = compute_matmul(M, N, K, a_type, b_type, data_a, data_b)
        conn.sendall(struct.pack("<II", 0, len(result)) + result)
        METRICS.record(a_type, b_type, size_a + size_b, len(result))
    return True


def handle_client(conn, addr):
    conn.settimeout(CLIENT_TIMEOUT)  # partial/idle clients can't pin a thread forever
    with METRICS.lock:
        METRICS.connections += 1
        METRICS.active += 1
    print(f"[matmul] client connected: {addr}")
    try:
        while True:
            magic_bytes = recv_exact(conn, 4)
            if not magic_bytes:
                break
            (magic,) = struct.unpack("<I", magic_bytes)
            if magic == MAGIC_GPU3:
                if not _serve_single(conn):
                    break
            elif magic == MAGIC_BATCH:
                if not _serve_batch(conn):
                    break
            else:
                print(f"[matmul] bad magic {hex(magic)} from {addr}; closing")
                break
    except Exception as exc:  # noqa: BLE001 — one bad client must not kill the server
        import traceback
        METRICS.error(f"{exc.__class__.__name__}: {exc}")
        traceback.print_exc()
    finally:
        conn.close()
        with METRICS.lock:
            METRICS.active -= 1
        print(f"[matmul] client disconnected: {addr}")


# --------------------------------------------------------------------------- #
# Health / metrics HTTP sidecar
# --------------------------------------------------------------------------- #
class _HealthHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence default stderr logging
        pass

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/health", "/metrics"):
            body = json.dumps(METRICS.snapshot()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def start_health_server(host, port):
    # Best-effort: the matmul service must not fail to start just because the
    # metrics port is busy. A failed health bind logs and is skipped, not fatal.
    # (tri-brain L3)
    try:
        httpd = ThreadingHTTPServer((host, port), _HealthHandler)
    except OSError as exc:
        print(f"[matmul] health/metrics disabled ({host}:{port}: {exc})")
        return None
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    print(f"[matmul] health/metrics on http://{host}:{port}/health")
    return httpd


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def serve(host="0.0.0.0", port=8096, health_port=8097, health_host="127.0.0.1",
          init_gpu=True):
    if init_gpu:
        _init_gpu()
    # Bind the health/metrics sidecar to localhost by DEFAULT, independent of the
    # matmul host. The metrics surface (request counts, dtype mix, last_error)
    # is observability, not data — it should not be world-readable just because
    # the matmul port listens on 0.0.0.0. (tri-brain L3) Override with
    # --health-host 0.0.0.0 for an intentionally remote dashboard.
    start_health_server(health_host, health_port)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    sock.bind((host, port))
    sock.listen(16)
    print(f"[matmul] Interweave matmul server listening on {host}:{port} (GPU={HAS_GPU})")
    try:
        while True:
            conn, addr = sock.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        print("\n[matmul] shutting down")
    finally:
        sock.close()


def main():
    ap = argparse.ArgumentParser(description="Interweave unified GPU matmul offload server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--health-port", type=int, default=8097)
    ap.add_argument("--health-host", default="127.0.0.1",
                    help="bind host for /health+/metrics (default localhost; "
                         "set 0.0.0.0 to expose the dashboard remotely)")
    ap.add_argument("--no-gpu", action="store_true", help="force CPU/NumPy fallback")
    args = ap.parse_args()
    serve(host=args.host, port=args.port, health_port=args.health_port,
          health_host=args.health_host, init_gpu=not args.no_gpu)


if __name__ == "__main__":
    main()
