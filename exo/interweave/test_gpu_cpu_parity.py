#!/usr/bin/env python3
"""
GPU<->CPU parity test for the unified Interweave matmul server.

Run this on any box with a CUDA GPU + cupy (e.g. the C4130 V100 once its driver
is back, or any RTX box). It confirms the compiled CUDA dequant kernels produce
bit-identical-within-tolerance results to the NumPy CPU path that the main
test suite already validated against hand-computed golden values.

    python3 test_gpu_cpu_parity.py

Exit 0 = GPU matches CPU. Skips (exit 0 + message) if no GPU is present.
"""

import sys
import numpy as np

import matmul_server as ms


def build_q8_0(flat):
    n = flat.size
    out = bytearray()
    for b in range(n // 32):
        blk = flat[b * 32:(b + 1) * 32].astype(np.float32)
        amax = float(np.max(np.abs(blk)))
        d = amax / 127.0 if amax > 0 else 0.0
        q = np.clip(np.round(blk / d), -127, 127).astype(np.int8) if d > 0 else np.zeros(32, np.int8)
        out += np.float16(d).tobytes() + q.tobytes()
    return bytes(out)


def build_mxfp4(n_blocks, rng):
    # Random E8M0 exponent in a sane range + random nibbles.
    out = bytearray()
    for _ in range(n_blocks):
        out += bytes([int(rng.integers(120, 136))])
        out += bytes(rng.integers(0, 256, size=16, dtype=np.uint8).tolist())
    return bytes(out)


def build_q4k(n_blocks, rng):
    out = bytearray()
    for _ in range(n_blocks):
        out += np.float16(rng.standard_normal()).tobytes()      # d
        out += np.float16(abs(rng.standard_normal())).tobytes()  # dmin
        out += bytes(rng.integers(0, 256, size=12, dtype=np.uint8).tolist())   # scales
        out += bytes(rng.integers(0, 256, size=128, dtype=np.uint8).tolist())  # qs
    return bytes(out)


def main():
    if not ms._init_gpu():
        print("No GPU/cupy available — skipping parity test (this is fine on CPU-only boxes).")
        sys.exit(0)

    rng = np.random.default_rng(7)
    failures = 0

    cases = [
        ("Q8_0", ms.TYPE_Q8_0, ms._gpu_q8_0, ms._cpu_q8_0, 4, 64, build_q8_0(rng.standard_normal(4 * 64).astype(np.float32))),
        ("MXFP4", ms.TYPE_MXFP4, ms._gpu_mxfp4, ms._cpu_mxfp4, 4, 64, build_mxfp4(4 * 64 // 32, rng)),
        ("Q4_K", ms.TYPE_Q4_K, ms._gpu_q4k, ms._cpu_q4k, 4, 256, build_q4k(4 * 256 // 256, rng)),
    ]
    for name, _t, gpu_fn, cpu_fn, rows, cols, data in cases:
        g = ms.cp.asnumpy(gpu_fn(data, rows, cols))
        c = cpu_fn(data, rows, cols)
        max_err = float(np.max(np.abs(g - c)))
        ok = np.allclose(g, c, atol=1e-3, rtol=1e-3)
        print(f"  {'PASS' if ok else 'FAIL'}  {name} dequant GPU==CPU  (max abs err {max_err:.2e})")
        failures += 0 if ok else 1

    # Full matmul parity (F32 + Q8_0 A operand)
    A = rng.standard_normal((8, 64)).astype(np.float32)
    B = rng.standard_normal((64, 5)).astype(np.float32)
    a_bytes = np.ascontiguousarray(A).tobytes()
    b_bytes = np.ascontiguousarray(B).tobytes()

    ms.HAS_GPU = True
    gpu_f32 = np.frombuffer(ms.compute_matmul(8, 5, 64, ms.TYPE_F32, ms.TYPE_F32, a_bytes, b_bytes),
                            dtype=np.float32).reshape((8, 5))
    ms.HAS_GPU = False
    cpu_f32 = np.frombuffer(ms.compute_matmul(8, 5, 64, ms.TYPE_F32, ms.TYPE_F32, a_bytes, b_bytes),
                            dtype=np.float32).reshape((8, 5))
    ok = np.allclose(gpu_f32, cpu_f32, atol=1e-2)
    print(f"  {'PASS' if ok else 'FAIL'}  F32 matmul GPU==CPU  (max abs err {np.max(np.abs(gpu_f32 - cpu_f32)):.2e})")
    failures += 0 if ok else 1

    print(f"\n{'ALL PARITY PASS' if failures == 0 else f'{failures} PARITY FAILURES'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
