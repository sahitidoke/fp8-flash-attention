import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_e4m3(x, FP8_MAX: tl.constexpr, eps: tl.constexpr):
    """Given a 2D tile `x` (fp32), return (x_fp8, scale) where
    x_fp8 = round_to_e4m3(x / scale), scale = amax(x) / FP8_MAX.

    scale is a scalar (per-tile, not per-row/per-channel) here for
    simplicity -- see the README for the per-row variant, which trades a
    slightly more expensive reduction for better accuracy on tensors with
    per-token dynamic range (e.g. the P tile after softmax).
    """
    amax = tl.max(tl.abs(x))
    scale = tl.maximum(amax, eps) / FP8_MAX
    x_scaled = x / scale
    x_fp8 = x_scaled.to(tl.float8e4nv)
    return x_fp8, scale


@triton.jit
def _quantize_e4m3_rowwise(x, FP8_MAX: tl.constexpr, eps: tl.constexpr):
    """Per-row dynamic quantization: one scale per row of the tile.
    Used for the P tile, where per-row range varies a lot more than
    per-tile range (each row is an independent softmax distribution).
    Returns (x_fp8, scale_col) where scale_col has shape (BLOCK_M, 1) and
    broadcasts back out on dequant.
    """
    amax = tl.max(tl.abs(x), axis=1)  # (BLOCK_M,)
    scale = tl.maximum(amax, eps) / FP8_MAX
    x_scaled = x / scale[:, None]
    x_fp8 = x_scaled.to(tl.float8e4nv)
    return x_fp8, scale

@triton.jit
def _fp8_dq_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    seq_len, head_dim,
    sm_scale,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    Q_ptr += pid_bh * stride_qh
    K_ptr += pid_bh * stride_kh
    V_ptr += pid_bh * stride_vh
    O_ptr += pid_bh * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    q_fp8, q_scale = _quantize_e4m3(q, FP8_MAX, EPS)

    m_i = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    n_end = (pid_m + 1) * BLOCK_M if CAUSAL else seq_len

    for n0 in range(0, n_end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        v_ptrs = V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        k_fp8, k_scale = _quantize_e4m3(k, FP8_MAX, EPS)
        s = tl.dot(q_fp8, tl.trans(k_fp8)).to(tl.float32)
        s = s * (q_scale * k_scale) * sm_scale

        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))
        s = tl.where(offs_n[None, :] < seq_len, s, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)

        l_i = l_i * alpha + tl.sum(p, axis=1)

        p_fp8, p_scale = _quantize_e4m3_rowwise(p, FP8_MAX, EPS)
        v_fp8, v_scale = _quantize_e4m3(v, FP8_MAX, EPS)

        pv = tl.dot(p_fp8, v_fp8).to(tl.float32)
        pv = pv * p_scale[:, None] * v_scale

        acc = acc * alpha[:, None] + pv
        m_i = m_ij

    acc = acc / l_i[:, None]

    o_ptrs = O_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=o_mask)


def fp8_double_quant_attention(q, k, v, causal=False, block_m=64, block_n=64):
    assert q.is_cuda
    B, H, N, D = q.shape
    assert k.shape == (B, H, N, D) and v.shape == (B, H, N, D)
    assert D <= 128

    o = torch.empty_like(q)
    sm_scale = 1.0 / (D ** 0.5)

    BLOCK_D = triton.next_power_of_2(D)
    grid = (triton.cdiv(N, block_m), B * H)

    _fp8_dq_attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        N, D,
        sm_scale,
        CAUSAL=causal,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=BLOCK_D,
        FP8_MAX=448.0, EPS=1e-8,
    )
    return o
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU visible. This kernel needs an Ada/Hopper/Blackwell "
            "GPU (fp8 tensor cores) to run. Use reference_numpy.py to study "
            "the algorithm/accuracy on CPU instead."
        )
    torch.manual_seed(0)
    B, H, N, D = 2, 8, 1024, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)

    for causal in (False, True):
        out = fp8_double_quant_attention(q, k, v, causal=causal)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), is_causal=causal
        )
        rel_err = (out.float() - ref).norm() / ref.norm()
        print(f"causal={causal}  rel_err={rel_err.item():.4e}")
