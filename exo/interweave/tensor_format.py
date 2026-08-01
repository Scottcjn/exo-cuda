"""
Universal Tensor Format (UTF-T) for Interweave Protocol

Provides backend-agnostic tensor representation with automatic conversion
between numpy, tinygrad, mlx, and torch formats.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, Tuple, TYPE_CHECKING
import numpy as np
import struct

if TYPE_CHECKING:
    try:
        from tinygrad import Tensor as TinyTensor
    except ImportError:
        TinyTensor = None
    try:
        import mlx.core as mx
    except ImportError:
        mx = None
    try:
        import torch
    except ImportError:
        torch = None


class DType(str, Enum):
    """Canonical dtype identifiers for cross-backend compatibility"""
    F32 = 'f32'
    F16 = 'f16'
    BF16 = 'bf16'
    I32 = 'i32'
    I16 = 'i16'
    I8 = 'i8'
    I4 = 'i4'  # Packed 4-bit quantized
    U8 = 'u8'

    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> 'DType':
        """Convert numpy dtype to canonical DType"""
        mapping = {
            np.float32: cls.F32,
            np.float16: cls.F16,
            np.int32: cls.I32,
            np.int16: cls.I16,
            np.int8: cls.I8,
            np.uint8: cls.U8,
        }
        # Handle dtype objects
        if hasattr(dtype, 'type'):
            dtype = dtype.type
        return mapping.get(dtype, cls.F32)

    def to_numpy(self) -> np.dtype:
        """Convert canonical DType to numpy dtype"""
        mapping = {
            DType.F32: np.float32,
            DType.F16: np.float16,
            DType.BF16: np.float32,  # BF16 stored as F32 in numpy
            DType.I32: np.int32,
            DType.I16: np.int16,
            DType.I8: np.int8,
            DType.I4: np.int8,  # Packed format
            DType.U8: np.uint8,
        }
        return np.dtype(mapping[self])

    @property
    def itemsize(self) -> int:
        """Bytes per element"""
        sizes = {
            DType.F32: 4, DType.F16: 2, DType.BF16: 2,
            DType.I32: 4, DType.I16: 2, DType.I8: 1,
            DType.I4: 1, DType.U8: 1,  # I4 is packed 2 per byte
        }
        return sizes[self]


@dataclass
class UniversalTensor:
    """
    Backend-agnostic tensor representation for cross-platform inference.

    Supports automatic conversion between:
    - numpy arrays (universal interchange)
    - tinygrad Tensors (CUDA/CPU)
    - mlx arrays (Apple Silicon)
    - torch Tensors (optional)

    Attributes:
        data: Raw tensor bytes in row-major order
        shape: Tensor dimensions
        dtype: Canonical dtype (f32, f16, bf16, i8, i4, etc.)
        layout: Memory layout ('row_major' or 'col_major')
        device_hint: Preferred device ('any', 'cpu', 'cuda', 'metal')
        scale: Quantization scale (for i8/i4 types)
        zero_point: Quantization zero point (for i8/i4 types)
    """
    data: bytes
    shape: Tuple[int, ...]
    dtype: DType
    layout: str = 'row_major'
    device_hint: str = 'any'

    # Quantization metadata
    scale: Optional[float] = None
    zero_point: Optional[int] = None

    def __post_init__(self):
        """Validate tensor data"""
        if isinstance(self.dtype, str):
            self.dtype = DType(self.dtype)
        if isinstance(self.shape, list):
            self.shape = tuple(self.shape)

    @property
    def numel(self) -> int:
        """Total number of elements"""
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def nbytes(self) -> int:
        """Expected byte size"""
        if self.dtype == DType.I4:
            # Packed 4-bit: 2 elements per byte
            return (self.numel + 1) // 2
        return self.numel * self.dtype.itemsize

    # ==================== FROM CONVERSIONS ====================

    @classmethod
    def from_numpy(cls, arr: np.ndarray, device_hint: str = 'any') -> 'UniversalTensor':
        """Create UniversalTensor from numpy array"""
        # Ensure contiguous row-major
        arr = np.ascontiguousarray(arr)

        return cls(
            data=arr.tobytes(),
            shape=tuple(arr.shape),
            dtype=DType.from_numpy(arr.dtype),
            layout='row_major',
            device_hint=device_hint,
        )

    @classmethod
    def from_tinygrad(cls, tensor: 'TinyTensor', device_hint: str = 'any') -> 'UniversalTensor':
        """Create UniversalTensor from tinygrad Tensor"""
        # Realize and convert to numpy
        arr = tensor.numpy()
        ut = cls.from_numpy(arr, device_hint)

        # Preserve device hint from original
        if 'cuda' in str(tensor.device).lower():
            ut.device_hint = 'cuda'
        elif 'gpu' in str(tensor.device).lower():
            ut.device_hint = 'cuda'

        return ut

    @classmethod
    def from_mlx(cls, arr: 'mx.array', device_hint: str = 'metal') -> 'UniversalTensor':
        """Create UniversalTensor from MLX array"""
        import mlx.core as mx

        # Convert to numpy
        np_arr = np.array(arr)
        ut = cls.from_numpy(np_arr, device_hint='metal')

        return ut

    @classmethod
    def from_torch(cls, tensor: 'torch.Tensor', device_hint: str = 'any') -> 'UniversalTensor':
        """Create UniversalTensor from PyTorch tensor"""
        import torch

        # Move to CPU and convert
        arr = tensor.detach().cpu().numpy()
        ut = cls.from_numpy(arr)

        # Preserve device hint
        if tensor.is_cuda:
            ut.device_hint = 'cuda'
        elif hasattr(tensor, 'is_mps') and tensor.is_mps:
            ut.device_hint = 'metal'

        return ut

    # ==================== TO CONVERSIONS ====================

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        # Quantized i8 stores unsigned zero-point levels (0..255, see
        # convert_dtype). Reading them back through the signed int8 view
        # would wrap anything >= 128 to a negative number and corrupt every
        # dequantization that depends on it.
        if self.dtype == DType.I8 and self.scale is not None:
            arr = np.frombuffer(self.data, dtype=np.uint8)
            return arr.reshape(self.shape)

        np_dtype = self.dtype.to_numpy()
        arr = np.frombuffer(self.data, dtype=np_dtype)

        # Handle packed i4
        if self.dtype == DType.I4:
            # Unpack 4-bit values. Nibbles hold unsigned zero-point levels
            # (0..15, see convert_dtype) - matching how they were packed.
            raw = arr.view(np.uint8)
            unpacked = np.zeros(self.numel, dtype=np.uint8)
            for i, byte in enumerate(raw):
                if i * 2 < self.numel:
                    unpacked[i * 2] = byte & 0x0F
                if i * 2 + 1 < self.numel:
                    unpacked[i * 2 + 1] = (byte >> 4) & 0x0F
            arr = unpacked

        return arr.reshape(self.shape)

    def to_tinygrad(self) -> 'TinyTensor':
        """Convert to tinygrad Tensor"""
        from tinygrad import Tensor, Device

        arr = self.to_numpy()

        # Create tensor on appropriate device
        if self.device_hint == 'cuda':
            return Tensor(arr, device='CUDA')
        elif self.device_hint == 'cpu':
            return Tensor(arr, device='CPU')
        else:
            return Tensor(arr)  # Default device

    def to_mlx(self) -> 'mx.array':
        """Convert to MLX array"""
        import mlx.core as mx

        arr = self.to_numpy()
        return mx.array(arr)

    def to_torch(self, device: str = 'cpu') -> 'torch.Tensor':
        """Convert to PyTorch tensor"""
        import torch

        arr = self.to_numpy()
        tensor = torch.from_numpy(arr.copy())

        if device != 'cpu':
            tensor = tensor.to(device)

        return tensor

    # ==================== DTYPE CONVERSION ====================

    def convert_dtype(self, target_dtype: DType) -> 'UniversalTensor':
        """
        Convert tensor to a different dtype.

        Handles:
        - Upcasting (f16 → f32)
        - Downcasting (f32 → f16)
        - Quantization (f32 → i8)
        - Dequantization (i8 → f32)
        """
        if self.dtype == target_dtype:
            return self

        arr = self.to_numpy()

        # Handle quantization
        if target_dtype in (DType.I8, DType.I4) and self.dtype in (DType.F32, DType.F16, DType.BF16):
            # Quantize: compute scale and zero point
            min_val, max_val = float(arr.min()), float(arr.max())
            value_range = max_val - min_val
            levels = 255.0 if target_dtype == DType.I8 else 15.0
            if value_range == 0.0:
                # Constant block (e.g. zero-padded KV slot): there's no
                # dynamic range to spread across levels, so fall back to a
                # safe unit scale instead of dividing by zero and poisoning
                # the whole tensor with NaN.
                scale = 1.0
                zero_point = round(-min_val)
            else:
                scale = value_range / levels
                zero_point = round(-min_val / scale)
            # Levels are unsigned zero-point offsets (0..255 / 0..15), matching
            # how to_numpy() unpacks them back out below.
            quantized_levels = np.round(arr / scale + zero_point).clip(0, levels).astype(np.uint8)

            if target_dtype == DType.I8:
                quantized = quantized_levels
            else:  # I4: pack two unsigned nibbles per byte
                flat = quantized_levels.flatten()
                packed = np.zeros((flat.size + 1) // 2, dtype=np.uint8)
                for i in range(0, flat.size, 2):
                    low = flat[i] & 0x0F
                    high = (flat[i + 1] & 0x0F) if i + 1 < flat.size else 0
                    packed[i // 2] = low | (high << 4)
                quantized = packed

            return UniversalTensor(
                data=quantized.tobytes(),
                shape=self.shape,
                dtype=target_dtype,
                layout=self.layout,
                device_hint=self.device_hint,
                scale=float(scale),
                zero_point=int(zero_point),
            )

        # Handle dequantization
        if self.dtype in (DType.I8, DType.I4) and target_dtype in (DType.F32, DType.F16):
            if self.scale is None:
                raise ValueError("Cannot dequantize without scale metadata")

            dequantized = (arr.astype(np.float32) - (self.zero_point or 0)) * self.scale

            if target_dtype == DType.F16:
                dequantized = dequantized.astype(np.float16)

            return UniversalTensor.from_numpy(dequantized, self.device_hint)

        # Simple type conversion
        target_np = target_dtype.to_numpy()
        converted = arr.astype(target_np)

        return UniversalTensor.from_numpy(converted, self.device_hint)

    # ==================== SERIALIZATION ====================

    def serialize(self) -> bytes:
        """
        Serialize to bytes for wire transfer.

        Format:
        - 4 bytes: magic number (0x494E5457 = "INTW")
        - 1 byte: version
        - 1 byte: dtype enum
        - 1 byte: layout (0=row_major, 1=col_major)
        - 1 byte: flags (has_scale, has_zero_point)
        - 4 bytes: ndim
        - ndim * 8 bytes: shape (int64 each)
        - [optional] 8 bytes: scale (float64)
        - [optional] 4 bytes: zero_point (int32)
        - remaining: tensor data
        """
        parts = []

        # Header
        parts.append(struct.pack('<I', 0x494E5457))  # Magic
        parts.append(struct.pack('<B', 1))  # Version
        parts.append(struct.pack('<B', list(DType).index(self.dtype)))  # Dtype
        parts.append(struct.pack('<B', 0 if self.layout == 'row_major' else 1))

        # Flags
        flags = 0
        if self.scale is not None:
            flags |= 1
        if self.zero_point is not None:
            flags |= 2
        parts.append(struct.pack('<B', flags))

        # Shape
        parts.append(struct.pack('<I', len(self.shape)))
        for dim in self.shape:
            parts.append(struct.pack('<q', dim))

        # Optional quantization metadata
        if self.scale is not None:
            parts.append(struct.pack('<d', self.scale))
        if self.zero_point is not None:
            parts.append(struct.pack('<i', self.zero_point))

        # Data
        parts.append(self.data)

        return b''.join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> 'UniversalTensor':
        """Deserialize from bytes"""
        offset = 0

        # Magic
        magic, = struct.unpack_from('<I', data, offset)
        offset += 4
        if magic != 0x494E5457:
            raise ValueError(f"Invalid magic number: {hex(magic)}")

        # Version
        version, = struct.unpack_from('<B', data, offset)
        offset += 1
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")

        # Dtype
        dtype_idx, = struct.unpack_from('<B', data, offset)
        offset += 1
        dtype = list(DType)[dtype_idx]

        # Layout
        layout_idx, = struct.unpack_from('<B', data, offset)
        offset += 1
        layout = 'row_major' if layout_idx == 0 else 'col_major'

        # Flags
        flags, = struct.unpack_from('<B', data, offset)
        offset += 1
        has_scale = bool(flags & 1)
        has_zero_point = bool(flags & 2)

        # Shape
        ndim, = struct.unpack_from('<I', data, offset)
        offset += 4
        shape = []
        for _ in range(ndim):
            dim, = struct.unpack_from('<q', data, offset)
            offset += 8
            shape.append(dim)
        shape = tuple(shape)

        # Optional metadata
        scale = None
        zero_point = None
        if has_scale:
            scale, = struct.unpack_from('<d', data, offset)
            offset += 8
        if has_zero_point:
            zero_point, = struct.unpack_from('<i', data, offset)
            offset += 4

        # Data
        tensor_data = data[offset:]

        return cls(
            data=tensor_data,
            shape=shape,
            dtype=dtype,
            layout=layout,
            scale=scale,
            zero_point=zero_point,
        )

    def __repr__(self) -> str:
        return f"UniversalTensor(shape={self.shape}, dtype={self.dtype.value}, device_hint={self.device_hint})"
