#
# Copyright (C) 2025 Roberto L. Castro (Roberto.LopezCastro@ist.ac.at). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import qutlass._CUDA
from qutlass.utils import get_padded_shape_mx, get_padded_shape_nv, pad_to_block
from typing import Literal

import warnings

try:
    from flashinfer import mm_fp4
    _HAS_FLASHINFER = True

    # Try to import backend enum for FlashInfer (if available)
    # FlashInfer 0.5.x doesn't have this, so we use string fallback
    try:
        from flashinfer.jit import Backend as FlashInferBackend
        _FLASHINFER_BACKEND = FlashInferBackend.cudnn
    except (ImportError, AttributeError):
        # Fallback to string (works with FlashInfer 0.5.x)
        _FLASHINFER_BACKEND = "cudnn"
except Exception:
    _HAS_FLASHINFER = False
    _FLASHINFER_BACKEND = None


def matmul_mxf4_bf16_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    backend: Literal["cutlass", "flashinfer"] = "cutlass",
) -> torch.Tensor:
    if backend == "cutlass":
        return qutlass._CUDA.matmul_mxf4_bf16_tn(a, b, a_sf, b_sf, alpha)
    elif backend == "flashinfer":
        if not _HAS_FLASHINFER:
            raise ImportError(
                "flashinfer backend requested but not installed. "
                "Install with:\n"
                "  git clone https://github.com/flashinfer-ai/flashinfer.git --recursive\n"
                "  cd flashinfer && python -m pip install -v ."
            )

        m, packed_k = a.shape
        k = packed_k * 2
        n = b.shape[0]
        BLOCK = 32
        out = torch.empty([m, n], device=a.device, dtype=torch.bfloat16)

        mm_fp4(
            a,
            b.T,
            a_sf.view(torch.uint8).view(-1, k // BLOCK),
            b_sf.view(torch.uint8).view(-1, k // BLOCK).T,
            alpha,
            torch.bfloat16,
            out,
            block_size=BLOCK,
            use_8x4_sf_layout=False,
            backend=_FLASHINFER_BACKEND,
            use_nvfp4=False,
        )

        return out

    else:
        raise ValueError(f"invalid backend {backend!r}; use 'cutlass' or 'flashinfer'")


def matmul_ada_mxf4_bf16_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return qutlass._CUDA.matmul_ada_mxf4_bf16_tn(a, b, a_sf, b_sf, alpha)


def matmul_nvf4_bf16_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    backend: Literal["cutlass", "flashinfer"] = "cutlass",
) -> torch.Tensor:
    if backend == "cutlass":
        return qutlass._CUDA.matmul_nvf4_bf16_tn(a, b, a_sf, b_sf, alpha)
    elif backend == "flashinfer":
        if not _HAS_FLASHINFER:
            raise ImportError(
                "flashinfer backend requested but not installed. "
                "Install with:\n"
                "  git clone https://github.com/flashinfer-ai/flashinfer.git --recursive\n"
                "  cd flashinfer && python -m pip install -v ."
            )

        m, packed_k = a.shape
        k = packed_k * 2
        n = b.shape[0]
        BLOCK = 16
        out = torch.empty([m, n], device=a.device, dtype=torch.bfloat16)

        mm_fp4(
            a,
            b.T,
            a_sf.view(-1, k // BLOCK),
            b_sf.view(-1, k // BLOCK).T,
            alpha,
            torch.bfloat16,
            out,
            block_size=BLOCK,
            use_8x4_sf_layout=False,
            backend=_FLASHINFER_BACKEND,
            use_nvfp4=True,
        )

        return out

    else:
        raise ValueError(f"invalid backend {backend!r}; use 'cutlass' or 'flashinfer'")


def matmul_mxf8_bf16_tn(a: torch.Tensor,
                        b: torch.Tensor,
                        block_scale_a: torch.Tensor,
                        block_scale_b: torch.Tensor,
                        alpha: torch.Tensor) -> torch.Tensor:
    return qutlass._CUDA.matmul_mxf8_bf16_tn(a, b, block_scale_a, block_scale_b, alpha)

def matmul_mxf8_bf16_nn(a: torch.Tensor,
                        b: torch.Tensor,
                        block_scale_a: torch.Tensor,
                        block_scale_b: torch.Tensor,
                        alpha: torch.Tensor) -> torch.Tensor:
    return qutlass._CUDA.matmul_mxf8_bf16_nn(a, b, block_scale_a, block_scale_b, alpha)


def fusedQuantizeMx(
    a: torch.Tensor,
    b: torch.Tensor,
    # TODO: add global_scale for consistency?
    *,
    method: Literal["quest", "abs_max"] = "quest",
    return_mask: bool = False,
    use_v2: bool = None,
):
    """
    Fused Hadamard rotation + MX quantization.

    Automatically dispatches to v2 API for K > 256 (CUTLASS 2.x with arbitrary K support).
    Falls back to legacy API for K ≤ 256 for backwards compatibility.

    GPU Architecture Support:
        - v2 API (K > 256): SM80+, SM90+, SM120 (Blackwell RTX Pro)
        - v2 API NOT supported: SM103 (B300) - no backward compat with SM80 kernels
        - Legacy API (K ≤ 256): All architectures

    Args:
        a: Input tensor [M, K] or [B, M, K] in BF16
        b: Hadamard rotation matrix [K, K] in BF16
        method: Quantization method - "quest" (variance-based) or "abs_max"
        return_mask: If True, return clipping mask (only for method='quest', v1 only)
        use_v2: Force v2 API (True), legacy API (False), or auto-detect (None)

    Returns:
        tuple of (quantized_values, scale_factors) or (quantized_values, scale_factors, mask)

    Raises:
        RuntimeError: If K > 256 on non-Blackwell GPU - v2 API requires SM100 or SM120+
    """
    K = a.size(-1)

    # Check GPU architecture compatibility
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability()
        # v2 API supports arbitrary K on:
        # - SM100 (B200) - Tested, Uses existing fused_quantize_mx_sm100.cu with arbitrary K
        # - SM120+ (RTX Pro 6000, RTX 5090) - Untested, Uses CUTLASS 2.x SM80 via backward compatibility
        sm_version = compute_capability[0] * 10 + compute_capability[1]
        v2_supported = sm_version in [100, 103] or sm_version >= 120
    else:
        v2_supported = False

    # auto-detect: use v2 for K > 256 AND architecture supports it, OR if explicitly requested
    if use_v2 is None:
        use_v2 = K > 256 and v2_supported
    elif use_v2 and not v2_supported:
        # User explicitly requested v2 but architecture doesn't support it
        if sm_version == 100:
            raise RuntimeError(
                f"v2 API not supported on B200/B300 (SM100). "
                f"CUTLASS SM80 kernels lack backward compatibility to SM100. "
                f"CUTLASS 3.x/4.x Blackwell kernels only support FP8/FP6/FP4, not BF16. "
                f"Please use K ≤ 256 (legacy API automatically used)."
            )
        else:
            raise RuntimeError(
                f"v2 API requested but not supported on this GPU (sm_{sm_version}). "
                f"v2 API requires SM120+ (Blackwell RTX Pro 6000, RTX 5090). "
                f"Detected GPU: sm_{sm_version}. "
                f"Use use_v2=False to force legacy API (K ≤ 256 only)."
            )

    if use_v2:
        # V2 API: CUTLASS 4.x with arbitrary K support
        if return_mask:
            raise ValueError(
                "return_mask is not supported with v2 API. "
                "Use use_v2=False to force legacy API for K ≤ 256."
            )

        # Handle batched inputs [B, M, K] by reshaping to 2D
        original_shape = a.shape
        if a.dim() > 2:
            a_2d = a.reshape(-1, K)  # [B*M, K]
        else:
            a_2d = a

        # Call v2 API
        xh_e2m1, xh_e8m0_uint8 = fusedQuantizeMx_v2(a_2d, b, method=method)

        # Reshape outputs back to original batch dimensions
        if len(original_shape) > 2:
            xh_e2m1 = xh_e2m1.view(*original_shape[:-1], K // 2)
            xh_e8m0_uint8 = xh_e8m0_uint8.view(*original_shape[:-1], K // 32)

        # Convert uint8 to float8_e8m0fnu for compatibility with existing code
        xh_e8m0 = xh_e8m0_uint8.view(torch.float8_e8m0fnu)

        return xh_e2m1, xh_e8m0

    else:
        # Legacy API: Original CUTLASS implementation (K ≤ 256)
        if K > 256:
            if torch.cuda.is_available():
                compute_capability = torch.cuda.get_device_capability()
                sm_version = compute_capability[0] * 10 + compute_capability[1]
                if sm_version == 103:
                    raise RuntimeError(
                        f"K={K} exceeds legacy API limit of 256, and v2 API is not supported on SM103 (B300). "
                        f"v2 API uses SM80 CUTLASS 2.x kernels which don't run on SM103. "
                        f"Supported GPUs for K > 256: SM80+ (Ampere/Ada/Hopper), SM120 (Blackwell RTX Pro). "
                        f"B300 (SM103) currently limited to K ≤ 256."
                    )
            raise ValueError(
                f"K={K} exceeds legacy API limit of 256. "
                f"Use use_v2=True or set use_v2=None for auto-detection."
            )

        # use actual dimensions, not padded (padding is only for to_blocked())
        actual_rows = a.numel() // a.size(-1) 
        actual_cols = a.size(-1) // 32

        xh_e2m1 = torch.empty(
            *a.shape[:-1], a.size(-1) // 2, dtype=torch.uint8, device=a.device
        )
        xh_e8m0 = torch.empty(
            actual_rows, actual_cols, dtype=torch.float8_e8m0fnu, device=a.device
        )

        if method == "quest":
            if return_mask:
                clip_mask = torch.empty(
                    *a.shape[:-1], a.size(-1) // 8, dtype=torch.uint8, device=a.device
                )
                return qutlass._CUDA.fusedQuantizeMxQuestWithMask(
                    a, b, xh_e2m1, xh_e8m0, clip_mask
                )
            else:
                return qutlass._CUDA.fusedQuantizeMxQuest(a, b, xh_e2m1, xh_e8m0)
        elif method == "abs_max":
            if return_mask:
                raise ValueError("return_mask is only supported for method 'quest'")
            return qutlass._CUDA.fusedQuantizeMxAbsMax(a, b, xh_e2m1, xh_e8m0)
        else:
            raise ValueError(f"invalid method {method!r}, must be 'quest' or 'abs_max'")

def fusedQuantizeMx_v2(
    a: torch.Tensor,
    h: torch.Tensor,
    *,
    method: Literal["quest", "abs_max"] = "quest",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CUTLASS 4.x Fused Quantization with arbitrary K support.

    Computes: D_fp4, D_scale = Quantize(A @ H^T)

    This v2 API supports arbitrary K dimensions (32, 64, 128, 256, 512, 1024, 2048, 4096, ...)
    unlike the original API which is limited to K <= 256.

    Args:
        a: Input tensor [M, K] in BF16
        h: Hadamard rotation matrix [K, K] in BF16
        method: Quantization method - "quest" (variance-based) or "abs_max"

    Returns:
        tuple of (quantized_values, scale_factors):
            - quantized_values: Packed E2M1 values [M, K/2] as uint8
            - scale_factors: E8M0 scale factors [M, K/32] as uint8 (one per 32-element group)

    Requirements:
        - K must be at least 32
        - K must be divisible by 32
        - Requires Blackwell GPU (SM100 or SM120)
    """
    if a.dim() != 2:
        raise ValueError(f"Input tensor must be 2D [M, K], got {a.dim()}D")

    M, K = a.shape

    if K < 32:
        raise ValueError(f"K must be at least 32, got {K}")
    if K % 32 != 0:
        raise ValueError(f"K must be divisible by 32, got {K}")
    if h.shape != (K, K):
        raise ValueError(f"Hadamard matrix must be [K, K] = [{K}, {K}], got {list(h.shape)}")

    # allocate output tensors
    # - packed E2M1: each pair of 4-bit values packed into 1 byte -> [M, K/2]
    # - scale factors: one e8m0 per 32-element group -> [M, K/32]
    xh_e2m1 = torch.empty(M, K // 2, dtype=torch.uint8, device=a.device)
    xh_e8m0 = torch.empty(M, K // 32, dtype=torch.uint8, device=a.device)

    if method == "quest":
        qutlass._CUDA.fusedQuantizeMxQuest_v2(a, h, xh_e2m1, xh_e8m0)
    elif method == "abs_max":
        qutlass._CUDA.fusedQuantizeMxAbsMax_v2(a, h, xh_e2m1, xh_e8m0)
    else:
        raise ValueError(f"invalid method {method!r}, must be 'quest' or 'abs_max'")

    return xh_e2m1, xh_e8m0


def fusedQuantizeNv(
    a: torch.Tensor,
    b: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    method: Literal["quest", "abs_max"] = "abs_max",
) -> tuple[torch.Tensor, torch.Tensor]:
    padded_rows, padded_cols = get_padded_shape_nv(a)
    xh_e2m1 = torch.empty(
        *a.shape[:-1], a.size(-1) // 2, dtype=torch.uint8, device=a.device
    )
    xh_e4m3 = torch.empty(
        padded_rows, padded_cols, dtype=torch.float8_e4m3fn, device=a.device
    )

    if method == "quest":
        return qutlass._CUDA.fusedQuantizeNvQuest(a, b, xh_e2m1, xh_e4m3, global_scale)
    elif method == "abs_max":
        return qutlass._CUDA.fusedQuantizeNvAbsMax(a, b, xh_e2m1, xh_e4m3, global_scale)
    else:
        raise ValueError(f"invalid method {method!r}, must be 'quest' or 'abs_max'")


def backward_t_bf16(
    x: torch.Tensor,
    h: torch.Tensor,
    xh_e2m1: torch.Tensor = None,
    xh_e8m0: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if xh_e2m1 is None:
        xh_e2m1 = torch.empty(
            *x.shape[:-2],
            x.size(-1),
            x.size(-2) // 2,
            dtype=torch.float4_e2m1fn_x2,
            device=h.device,
        )
    if xh_e8m0 is None:
        xh_e8m0 = torch.empty(
            *x.shape[:-2],
            x.size(-1),
            x.size(-2) // 32,
            dtype=torch.float8_e8m0fnu,
            device=h.device,
        )

    assert (
        x.dtype == h.dtype == torch.bfloat16
        and xh_e2m1.dtype == torch.float4_e2m1fn_x2
        and xh_e8m0.dtype == torch.float8_e8m0fnu
    )
    assert (
        x.is_contiguous()
        and h.is_contiguous()
        and xh_e2m1.is_contiguous()
        and xh_e8m0.is_contiguous()
    )

    qutlass._CUDA.backward_t_bf16(x, h, xh_e2m1, xh_e8m0)

    return xh_e2m1, xh_e8m0


def backward_qt_bf16(
    x_e2m1: torch.Tensor,
    x_e8m0: torch.Tensor,
    h: torch.Tensor,
    alpha: torch.Tensor,
    xh_e2m1: torch.Tensor = None,
    xh_e8m0: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if xh_e2m1 is None:
        xh_e2m1 = torch.empty(
            *x_e2m1.shape[:-2],
            x_e2m1.size(-1) * 2,
            x_e2m1.size(-2) // 2,
            dtype=torch.float4_e2m1fn_x2,
            device=h.device,
        )
    if xh_e8m0 is None:
        xh_e8m0 = torch.empty(
            *x_e8m0.shape[:-2],
            x_e8m0.size(-1) * 32,
            x_e8m0.size(-2) // 32,
            dtype=torch.float8_e8m0fnu,
            device=h.device,
        )

    # assert h.dtype == torch.bfloat16 and x_e2m1.dtype == xh_e2m1.dtype == torch.float4_e2m1fn_x2 and x_e8m0.dtype == xh_e8m0.dtype == torch.float8_e8m0fnu
    assert (
        x_e2m1.is_contiguous()
        and x_e8m0.is_contiguous()
        and h.is_contiguous()
        and xh_e2m1.is_contiguous()
        and xh_e8m0.is_contiguous()
    )

    qutlass._CUDA.backward_qt_bf16(x_e2m1, x_e8m0, h, alpha, xh_e2m1, xh_e8m0)

    return xh_e2m1, xh_e8m0

def backward_bf16_square_double_mxfp8(x_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x_bf16.size(0) % 128 != 0:
        x_bf16 = pad_to_block(x_bf16, [0], 128)
    x_fp8 = torch.empty_like(x_bf16, dtype=torch.float8_e4m3fn)
    row_scales = torch.empty(x_bf16.shape[0], x_bf16.shape[1] // 32, device=x_bf16.device, dtype=torch.float8_e8m0fnu)
    column_scales = torch.empty(x_bf16.shape[1], x_bf16.shape[0] // 32, device=x_bf16.device, dtype=torch.float8_e8m0fnu)

    qutlass._CUDA.backward_bf16_square_double_mxfp8(x_bf16, x_fp8, row_scales, column_scales)

    return x_fp8, row_scales, column_scales

def mxfp4_transpose_mxfp8(x_fp4: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: padding in kernel
    # >>>>
    if x_fp4.size(0) % 256 != 0:
        m = x_fp4.shape[0]
        m_up128 = ((m - 1) // 256) * 256 + 256
        x_fp4 = pad_to_block(x_fp4, [0], 256)
        scales[m:m_up128] = 1.0
    # <<<<

    x_fp8 = torch.empty(x_fp4.shape[1] * 2, x_fp4.shape[0], device=x_fp4.device, dtype=torch.float8_e4m3fn)
    shared_exps = torch.empty(x_fp4.shape[1] * 2, x_fp4.shape[0] // 32, device=x_fp4.device, dtype=torch.float8_e8m0fnu)

    qutlass._CUDA.mxfp4_transpose_mxfp8(x_fp4, scales, x_fp8, shared_exps)

    return x_fp8, shared_exps