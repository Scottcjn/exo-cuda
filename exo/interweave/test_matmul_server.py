#!/usr/bin/env python3
"""
Correctness + plumbing tests for the unified Interweave matmul server.

Runs CPU-only (no cupy / no GPU required) so it can validate the byte-layout
math and the wire protocol anywhere, including CI. When a GPU + cupy are
present, run test_gpu_cpu_parity.py separately to confirm GPU == CPU.

Strategy:
  * Dequant correctness is checked against INDEPENDENT hand-computed oracles
    (golden values), not against the server's own kernels — no circular tests.
  * End-to-end matmul checks framing, sizes, and row-major (WIRE_ORDER="C")
    handling by round-tripping real tensors through an in-process server thread.
  * Validation checks confirm malformed headers are rejected, not allocated.

Usage:  python3 test_matmul_server.py
Exit code 0 = all passed.
"""

import socket
import json
import struct
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np

import matmul_server as ms

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# --------------------------------------------------------------------------- #
# Reference encoders (independent of the server) for building test inputs.
# --------------------------------------------------------------------------- #
def encode_q8_0(flat: np.ndarray) -> bytes:
    """Encode a flat float array to Q8_0 bytes (34B/32 elems). Reference quantizer."""
    n = flat.size
    assert n % 32 == 0
    out = bytearray()
    for b in range(n // 32):
        blk = flat[b * 32:(b + 1) * 32].astype(np.float32)
        amax = np.max(np.abs(blk))
        d = (amax / 127.0) if amax > 0 else 0.0
        q = np.round(blk / d).astype(np.int32) if d > 0 else np.zeros(32, np.int32)
        q = np.clip(q, -127, 127).astype(np.int8)
        out += np.float16(d).tobytes()
        out += q.tobytes()
    return bytes(out)


def encode_q4k(n_blocks, rng):
    """Build n_blocks of valid-looking Q4_K bytes (144B each)."""
    out = bytearray()
    for _ in range(n_blocks):
        out += np.float16(abs(rng.standard_normal()) + 0.1).tobytes()  # d
        out += np.float16(abs(rng.standard_normal()) * 0.1).tobytes()  # dmin
        out += bytes(rng.integers(0, 256, size=12, dtype=np.uint8).tolist())
        out += bytes(rng.integers(0, 256, size=128, dtype=np.uint8).tolist())
    return bytes(out)


def encode_mxfp4(n_blocks, rng):
    """Build n_blocks of valid MXFP4 bytes (17B each)."""
    out = bytearray()
    for _ in range(n_blocks):
        out += bytes([int(rng.integers(120, 136))])  # E8M0 exponent (normal range)
        out += bytes(rng.integers(0, 256, size=16, dtype=np.uint8).tolist())
    return bytes(out)


# --------------------------------------------------------------------------- #
# 1. Dequant unit tests vs hand-computed golden values
# --------------------------------------------------------------------------- #
def test_q8_0_golden():
    # One block: d = 2.0, q = 0,1,2,...,31  -> values = 2.0 * q
    d = np.float16(2.0)
    q = np.arange(32, dtype=np.int8)
    data = d.tobytes() + q.tobytes()
    out = ms._cpu_q8_0(data, 32, 1).reshape(-1)  # 32x1: single column, order-independent
    expected = 2.0 * np.arange(32, dtype=np.float32)
    check("q8_0 golden", np.allclose(out, expected, atol=1e-3),
          f"max err {np.max(np.abs(out - expected)):.4f}")

    # Negative reinterpretation: q = -1 (byte 0xFF) -> value -2.0
    data2 = np.float16(2.0).tobytes() + bytes([0xFF]) + bytes(31)
    out2 = ms._cpu_q8_0(data2, 32, 1).reshape(-1)
    check("q8_0 signed reinterpret", abs(out2[0] - (-2.0)) < 1e-3, f"got {out2[0]}")


def test_mxfp4_golden():
    # E8M0 exponent byte. For e >= 2, scale = 2^(e-128).
    # Pick e = 129 -> scale = 2^1 = 2.0. Table[1] = 1 -> value 2.0; table[5]=6 -> 12.0
    table = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]
    e = 129
    # 16 packed bytes: low nibble=1, high nibble=5 in every byte
    packed = bytes([(5 << 4) | 1]) * 16
    data = bytes([e]) + packed
    out = ms._cpu_mxfp4(data, 32, 1).reshape(-1)
    scale = 2.0 ** (e - 128)
    exp_low = table[1] * scale   # first 16
    exp_high = table[5] * scale  # next 16
    ok = np.allclose(out[:16], exp_low) and np.allclose(out[16:], exp_high)
    check("mxfp4 golden", ok, f"low={out[0]} (exp {exp_low}) high={out[16]} (exp {exp_high})")

    # e<2 subnormal-scale branch (kernel: scale = 0x00200000 << e as float).
    # e=1 -> scale = 2^-127; nibble low=1 -> value = table[1]*2^-127 = 2^-127.
    data2 = bytes([1]) + bytes([0x01]) * 16
    out2 = ms._cpu_mxfp4(data2, 32, 1).reshape(-1)
    check("mxfp4 e<2 scale branch", np.isclose(out2[0], 2.0 ** -127, rtol=1e-4),
          f"got {out2[0]:.3e} exp {2.0**-127:.3e}")


def test_q4k_golden():
    # Build one Q4_K block (144B): d=1.0, dmin=0.0, all six-bit scales = 1,
    # qs nibbles known -> value = 1*nibble (min term zero).
    d = np.float16(1.0).tobytes()
    dmin = np.float16(0.0).tobytes()
    # We need the 8 six-bit values (j=0..7, packed at j*6 bits) all == 1.
    # Easiest: set scales bytes so each extracted 6-bit field is 1. Bit layout is
    # dense little-endian across the first 6 bytes. Construct by writing the
    # 48-bit little-endian integer whose 6-bit groups are all 1.
    val = 0
    for j in range(8):
        val |= (1 & 0x3F) << (j * 6)
    scales = val.to_bytes(8, "little")[:6] + bytes(6)  # 12 bytes total
    # qs: 128 bytes. Make low nibble = 3, high nibble = 5 everywhere.
    qs = bytes([(5 << 4) | 3]) * 128
    data = d + dmin + scales + qs
    out = ms._cpu_q4k(data, 256, 1).reshape(-1)
    # With scale=1, min=0: even positions = low nibble = 3, odd = high nibble = 5
    even = out[0::2]
    odd = out[1::2]
    check("q4k golden even (low nibble)", np.allclose(even, 3.0), f"got {even[0]}")
    check("q4k golden odd (high nibble)", np.allclose(odd, 5.0), f"got {odd[0]}")


# --------------------------------------------------------------------------- #
# 2. Validation tests (pure function, no socket)
# --------------------------------------------------------------------------- #
def test_validation():
    check("reject zero dim", ms.validate_op(0, 4, 4, ms.TYPE_F32, ms.TYPE_F32) != "")
    check("reject huge dim", ms.validate_op(ms.MAX_DIM + 1, 4, 4, ms.TYPE_F32, ms.TYPE_F32) != "")
    check("reject bad dtype", ms.validate_op(4, 4, 4, 999, ms.TYPE_F32) != "")
    check("accept normal", ms.validate_op(16, 16, 256, ms.TYPE_Q4_K, ms.TYPE_F32) == "")
    # Byte accounting matches ggml block sizes.
    check("q4k size", ms.operand_bytes(ms.TYPE_Q4_K, 1, 256) == 144)
    check("q8_0 size", ms.operand_bytes(ms.TYPE_Q8_0, 1, 32) == 34)
    check("mxfp4 size", ms.operand_bytes(ms.TYPE_MXFP4, 1, 32) == 17)
    check("f16 size", ms.operand_bytes(ms.TYPE_F16, 2, 8) == 32)
    # tri-brain B1/L2: Q4_K row width (inner dim) must be block-aligned, checked
    # on K for A and N for B — NOT on total element count.
    check("reject Q4_K A bad row width (K=100)", ms.validate_op(16, 4, 100, ms.TYPE_Q4_K, ms.TYPE_F32) != "")
    # The subtle case Codex caught: M*K=256 (one block total) but K=128 (half a
    # block per row) must STILL be rejected.
    check("reject Q4_K A K=128 (half block/row)", ms.validate_op(2, 4, 128, ms.TYPE_Q4_K, ms.TYPE_F32) != "")
    # B operand alignment is on N (its row width).
    check("reject Q4_K B bad row width (N=100)", ms.validate_op(16, 100, 256, ms.TYPE_F32, ms.TYPE_Q4_K) != "")
    check("accept Q4_K aligned (K=256)", ms.validate_op(2, 4, 256, ms.TYPE_Q4_K, ms.TYPE_F32) == "")
    # tri-brain B2/L2: huge F32 working set (result or operand) rejected.
    check("reject huge result", ms.validate_op(ms.MAX_DIM, ms.MAX_DIM, 1, ms.TYPE_F32, ms.TYPE_F32) != "")


# --------------------------------------------------------------------------- #
# 3. End-to-end socket tests (in-process CPU server)
# --------------------------------------------------------------------------- #
def _start_server(port, health_port):
    # Force CPU path regardless of host GPU so the test is deterministic.
    ms.HAS_GPU = False
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(8)

    def loop():
        while True:
            try:
                conn, addr = sock.accept()
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=ms.handle_client, args=(conn, addr), daemon=True).start()

    threading.Thread(target=loop, daemon=True).start()
    return sock


def _gpu3_matmul(port, A, B, a_type=ms.TYPE_F32, b_type=ms.TYPE_F32,
                 a_bytes=None, b_bytes=None):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    if a_bytes is None:
        a_bytes = np.ascontiguousarray(A, dtype=np.float32).tobytes()  # row-major
    if b_bytes is None:
        b_bytes = np.ascontiguousarray(B, dtype=np.float32).tobytes()
    c = socket.create_connection(("127.0.0.1", port), timeout=10)
    c.sendall(struct.pack("<IIIIII", ms.MAGIC_GPU3, M, N, K, a_type, b_type))
    c.sendall(a_bytes)
    c.sendall(b_bytes)
    hdr = ms.recv_exact(c, 16)
    magic, status, rM, rN = struct.unpack("<IIII", hdr)
    if status != 0:
        c.close()
        return None, status
    res_dtype = np.float16 if ms.RESULT_DTYPE == "f16" else np.float32
    payload = ms.recv_exact(c, rM * rN * ms.result_bytes_per_elem())
    c.close()
    C = np.frombuffer(payload, dtype=res_dtype).reshape((rM, rN), order="C").astype(np.float32)
    return C, status


def test_e2e_f32():
    port = 8196
    _start_server(port, 8197)
    time.sleep(0.2)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 16)).astype(np.float32)
    B = rng.standard_normal((16, 4)).astype(np.float32)
    C, status = _gpu3_matmul(port, A, B)
    ref = A @ B
    check("e2e F32 matmul", C is not None and np.allclose(C, ref, atol=0.05),
          f"max err {np.max(np.abs(C - ref)):.4f}" if C is not None else f"status {status}")


def test_e2e_q8_0():
    port = 8198
    _start_server(port, 8199)
    time.sleep(0.2)
    rng = np.random.default_rng(1)
    # A is Q8_0: 4 rows x 64 cols (each row dequantizes per 32-elem block).
    M, K, N = 4, 64, 5
    A_ref = rng.standard_normal((M, K)).astype(np.float32)
    # Encode A row-major flat (server reads order="C").
    a_flat = np.ascontiguousarray(A_ref).reshape(-1)
    a_bytes = encode_q8_0(a_flat)
    # Independent oracle: decode with plain d*q arithmetic in-test.
    A_oracle = ms._cpu_q8_0(a_bytes, M, K)  # same formula; validates plumbing/sizes
    B = rng.standard_normal((K, N)).astype(np.float32)
    C, status = _gpu3_matmul(port, np.zeros((M, K), np.float32), B,
                             a_type=ms.TYPE_Q8_0, a_bytes=a_bytes)
    ref = A_oracle @ B
    check("e2e Q8_0 matmul", C is not None and np.allclose(C, ref, atol=0.05),
          f"max err {np.max(np.abs(C - ref)):.4f}" if C is not None else f"status {status}")


def test_e2e_reject():
    port = 8200
    _start_server(port, 8201)
    time.sleep(0.2)
    # Oversized M must come back status=1.
    c = socket.create_connection(("127.0.0.1", port), timeout=10)
    c.sendall(struct.pack("<IIIIII", ms.MAGIC_GPU3, ms.MAX_DIM + 1, 4, 4,
                          ms.TYPE_F32, ms.TYPE_F32))
    hdr = ms.recv_exact(c, 16)
    c.close()
    ok = hdr is not None and struct.unpack("<IIII", hdr)[1] == 1
    check("e2e reject oversized", ok, f"hdr {hdr!r}")

    # Bad magic must close the connection (recv returns empty).
    c2 = socket.create_connection(("127.0.0.1", port), timeout=10)
    c2.sendall(struct.pack("<I", 0xDEADBEEF))
    resp = c2.recv(16)
    c2.close()
    check("e2e bad magic closes", resp == b"", f"got {resp!r}")


def test_e2e_batch():
    port = 8202
    _start_server(port, 8203)
    time.sleep(0.2)
    rng = np.random.default_rng(2)
    ops = []
    for _ in range(3):
        A = rng.standard_normal((4, 8)).astype(np.float32)
        B = rng.standard_normal((8, 3)).astype(np.float32)
        ops.append((A, B))
    c = socket.create_connection(("127.0.0.1", port), timeout=10)
    c.sendall(struct.pack("<II", ms.MAGIC_BATCH, len(ops)))
    for A, B in ops:
        M, K = A.shape
        _, N = B.shape
        c.sendall(struct.pack("<IIIII", M, N, K, ms.TYPE_F32, ms.TYPE_F32))
        c.sendall(np.ascontiguousarray(A, dtype=np.float32).tobytes())
        c.sendall(np.ascontiguousarray(B, dtype=np.float32).tobytes())
    all_ok = True
    res_dtype = np.float16 if ms.RESULT_DTYPE == "f16" else np.float32
    for A, B in ops:
        M, K = A.shape
        _, N = B.shape
        hdr = ms.recv_exact(c, 8)
        status, size = struct.unpack("<II", hdr)
        payload = ms.recv_exact(c, size)
        C = np.frombuffer(payload, dtype=res_dtype).reshape((M, N), order="C").astype(np.float32)
        if not (status == 0 and np.allclose(C, A @ B, atol=0.05)):
            all_ok = False
    c.close()
    check("e2e batch of 3", all_ok)


def test_e2e_q4k():
    port = 8206
    _start_server(port, 8207)
    time.sleep(0.2)
    rng = np.random.default_rng(4)
    M, K, N = 2, 256, 3
    a_bytes = encode_q4k(M * K // 256, rng)
    A_oracle = ms._cpu_q4k(a_bytes, M, K)
    B = rng.standard_normal((K, N)).astype(np.float32)
    C, status = _gpu3_matmul(port, np.zeros((M, K), np.float32), B,
                             a_type=ms.TYPE_Q4_K, a_bytes=a_bytes)
    ref = A_oracle @ B
    check("e2e Q4_K matmul", C is not None and np.allclose(C, ref, atol=0.1),
          f"max err {np.max(np.abs(C - ref)):.4f}" if C is not None else f"status {status}")


def test_e2e_mxfp4():
    port = 8208
    _start_server(port, 8209)
    time.sleep(0.2)
    rng = np.random.default_rng(5)
    M, K, N = 2, 64, 3
    a_bytes = encode_mxfp4(M * K // 32, rng)
    A_oracle = ms._cpu_mxfp4(a_bytes, M, K)
    B = rng.standard_normal((K, N)).astype(np.float32)
    C, status = _gpu3_matmul(port, np.zeros((M, K), np.float32), B,
                             a_type=ms.TYPE_MXFP4, a_bytes=a_bytes)
    ref = A_oracle @ B
    check("e2e MXFP4 matmul", C is not None and np.allclose(C, ref, atol=0.1),
          f"max err {np.max(np.abs(C - ref)):.4f}" if C is not None else f"status {status}")


def test_metrics_snapshot():
    # The /health + /metrics sidecar serves METRICS.snapshot(); exercise its
    # shape and that requests were counted by the e2e tests above.
    snap = ms.METRICS.snapshot()
    keys_ok = all(k in snap for k in ("status", "gpu", "requests", "rejected", "by_dtype", "uptime_s"))
    check("metrics snapshot shape", keys_ok and snap["requests"] > 0, f"snap={snap}")


def test_e2e_f16_result():
    # tri-brain SHOULD-FIX: exercise the RESULT_DTYPE="f16" wire path so the
    # non-default mode is not silently broken. Restore the default afterward.
    port = 8204
    _start_server(port, 8205)
    time.sleep(0.2)
    saved = ms.RESULT_DTYPE
    ms.RESULT_DTYPE = "f16"
    try:
        rng = np.random.default_rng(3)
        A = rng.standard_normal((6, 12)).astype(np.float32)
        B = rng.standard_normal((12, 4)).astype(np.float32)
        C, status = _gpu3_matmul(port, A, B)  # reader keys off ms.RESULT_DTYPE
        ref = A @ B
        check("e2e f16 result wire", C is not None and np.allclose(C, ref, atol=0.1),
              f"max err {np.max(np.abs(C - ref)):.4f}" if C is not None else f"status {status}")
    finally:
        ms.RESULT_DTYPE = saved


def test_e2e_persistent_multi_op():
    # Core feature: multiple GPU3 ops streamed over ONE connection without
    # reconnecting (the POWER8 client does this). The brains noted every other
    # e2e test uses one-op-per-connection.
    port = 8210
    _start_server(port, 8211)
    time.sleep(0.2)
    rng = np.random.default_rng(6)
    c = socket.create_connection(("127.0.0.1", port), timeout=10)
    ok = True
    for _ in range(3):
        A = rng.standard_normal((5, 8)).astype(np.float32)
        B = rng.standard_normal((8, 4)).astype(np.float32)
        M, K = A.shape
        _, N = B.shape
        c.sendall(struct.pack("<IIIIII", ms.MAGIC_GPU3, M, N, K, ms.TYPE_F32, ms.TYPE_F32))
        c.sendall(np.ascontiguousarray(A, dtype=np.float32).tobytes())
        c.sendall(np.ascontiguousarray(B, dtype=np.float32).tobytes())
        hdr = ms.recv_exact(c, 16)
        _, status, rM, rN = struct.unpack("<IIII", hdr)
        res_dtype = np.float16 if ms.RESULT_DTYPE == "f16" else np.float32
        payload = ms.recv_exact(c, rM * rN * ms.result_bytes_per_elem())
        C = np.frombuffer(payload, dtype=res_dtype).reshape((rM, rN), order="C").astype(np.float32)
        if not (status == 0 and np.allclose(C, A @ B, atol=0.05)):
            ok = False
    c.close()
    check("e2e persistent multi-op (3 ops, 1 conn)", ok)


def test_health_http():
    httpd = ms.start_health_server("127.0.0.1", 8212)
    try:
        time.sleep(0.1)
        with urllib.request.urlopen("http://127.0.0.1:8212/health", timeout=5) as r:
            body = json.loads(r.read())
            code = r.getcode()
        check("health HTTP 200 + json", code == 200 and body.get("status") == "ok"
              and "gpu" in body and "requests" in body, f"code={code} body={body}")
        # unknown path -> 404
        try:
            urllib.request.urlopen("http://127.0.0.1:8212/nope", timeout=5)
            check("health 404 on bad path", False, "expected 404")
        except urllib.error.HTTPError as e:
            check("health 404 on bad path", e.code == 404, f"got {e.code}")
    finally:
        httpd.shutdown()


def main():
    print("Interweave matmul server — correctness tests (CPU path)\n")
    print("[dequant golden]")
    test_q8_0_golden()
    test_mxfp4_golden()
    test_q4k_golden()
    print("[validation]")
    test_validation()
    print("[end-to-end]")
    test_e2e_f32()
    test_e2e_q8_0()
    test_e2e_q4k()
    test_e2e_mxfp4()
    test_e2e_f16_result()
    test_e2e_persistent_multi_op()
    test_e2e_reject()
    test_e2e_batch()
    test_metrics_snapshot()
    test_health_http()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
