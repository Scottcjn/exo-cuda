"""
Regression tests for UniversalTensor int8/int4 quantization round-trip.

convert_dtype() packs quantized levels as *unsigned* zero-point offsets
(0..255 for i8, 0..15 for i4 nibbles). to_numpy() must unpack them the same
way, otherwise any level >= 128 (i8) or >= 8 (i4) gets reinterpreted as a
negative signed byte/nibble and the whole tensor comes back corrupted -
exactly the kind of thing that silently wrecks a distributed KV-cache
transfer without ever raising an exception.
"""

import numpy as np

from exo.interweave.tensor_format import DType, UniversalTensor


def _max_abs_err(orig: np.ndarray, recon: np.ndarray) -> float:
    return float(np.max(np.abs(orig.astype(np.float64) - recon.astype(np.float64))))


def test_i8_roundtrip_stays_within_one_quantization_step():
    arr = np.linspace(-2.0, 2.0, 256).astype(np.float32)
    tensor = UniversalTensor.from_numpy(arr)

    quantized = tensor.convert_dtype(DType.I8)
    recon = quantized.convert_dtype(DType.F32).to_numpy()

    # Max quantization error for 255 levels over the value range should
    # never exceed half a step. Before the fix, high-magnitude values that
    # quantize to level >= 128 wrapped through signed int8 and blew this
    # bound apart (observed error ~4.0 on a range of 4.0).
    step = 4.0 / 255.0
    assert _max_abs_err(arr, recon) <= step / 2 + 1e-4


def test_i4_roundtrip_stays_within_one_quantization_step():
    arr = np.linspace(-2.0, 2.0, 16).astype(np.float32)
    tensor = UniversalTensor.from_numpy(arr)

    quantized = tensor.convert_dtype(DType.I4)
    recon = quantized.convert_dtype(DType.F32).to_numpy()

    step = 4.0 / 15.0
    assert _max_abs_err(arr, recon) <= step / 2 + 1e-4


def test_i8_high_level_values_do_not_wrap_negative():
    # Values near max_val quantize to levels close to 255, which is exactly
    # the region that got corrupted by the signed int8 reinterpretation.
    arr = np.array([-10.0, -5.0, 0.0, 5.0, 9.5, 10.0], dtype=np.float32)
    tensor = UniversalTensor.from_numpy(arr)

    quantized = tensor.convert_dtype(DType.I8)
    raw_levels = quantized.to_numpy()
    # Unsigned levels must never be reported as negative.
    assert raw_levels.min() >= 0
    assert raw_levels.max() <= 255

    recon = quantized.convert_dtype(DType.F32).to_numpy()
    step = 20.0 / 255.0
    assert _max_abs_err(arr, recon) <= step / 2 + 1e-3


def test_i4_high_nibble_values_do_not_wrap_negative():
    arr = np.array([-10.0, -5.0, 0.0, 5.0, 9.0, 10.0], dtype=np.float32)
    tensor = UniversalTensor.from_numpy(arr)

    quantized = tensor.convert_dtype(DType.I4)
    raw_levels = quantized.to_numpy()
    assert raw_levels.min() >= 0
    assert raw_levels.max() <= 15

    recon = quantized.convert_dtype(DType.F32).to_numpy()
    step = 20.0 / 15.0
    assert _max_abs_err(arr, recon) <= step / 2 + 1e-3


def test_i8_random_kv_cache_like_data_roundtrips():
    rng = np.random.default_rng(0)
    arr = (rng.standard_normal(4096).astype(np.float32) * 3.0)

    tensor = UniversalTensor.from_numpy(arr)
    quantized = tensor.convert_dtype(DType.I8)
    recon = quantized.convert_dtype(DType.F32).to_numpy()

    step = (float(arr.max()) - float(arr.min())) / 255.0
    assert _max_abs_err(arr, recon) <= step / 2 + 1e-3


def test_constant_tensor_quantization_does_not_produce_nan():
    # A constant block (e.g. a zero-padded KV slot) has zero dynamic range,
    # which used to divide by zero and poison the tensor with NaN.
    arr = np.full(64, 5.37, dtype=np.float32)
    tensor = UniversalTensor.from_numpy(arr)

    quantized = tensor.convert_dtype(DType.I8)
    assert np.isfinite(quantized.scale)

    recon = quantized.convert_dtype(DType.F32).to_numpy()
    assert np.all(np.isfinite(recon))


def test_all_zero_tensor_quantization_roundtrips_exactly():
    arr = np.zeros(32, dtype=np.float32)
    tensor = UniversalTensor.from_numpy(arr)

    quantized = tensor.convert_dtype(DType.I8)
    recon = quantized.convert_dtype(DType.F32).to_numpy()

    assert np.allclose(recon, 0.0)


def test_i8_serialize_deserialize_matches_direct_dequantize():
    arr = np.linspace(-8.0, 8.0, 512).astype(np.float32)
    tensor = UniversalTensor.from_numpy(arr)
    quantized = tensor.convert_dtype(DType.I8)

    wire = quantized.serialize()
    restored = UniversalTensor.deserialize(wire)

    direct = quantized.convert_dtype(DType.F32).to_numpy()
    via_wire = restored.convert_dtype(DType.F32).to_numpy()
    assert np.array_equal(direct, via_wire)
