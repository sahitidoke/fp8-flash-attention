import numpy as np

FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN_NORMAL = 2.0 ** -6
MANTISSA_BITS = 3
def quantize_e4m3_simulated(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    sign = np.sign(x)
    ax = np.abs(x)

    out = np.zeros_like(x)
    nonzero = ax > 0
    if not np.any(nonzero):
        return out

    exp = np.floor(np.log2(np.maximum(ax, 1e-30)))
    exp = np.clip(exp, np.log2(FP8_E4M3_MIN_NORMAL), np.floor(np.log2(FP8_E4M3_MAX)))
    step = 2.0 ** (exp - MANTISSA_BITS)
    q = np.round(ax / step) * step
    out = sign * q
    out = np.clip(out, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    return out

def quantize_tile_dynamic(x: np.ndarray, axis=None, eps=1e-8):
    amax = np.max(np.abs(x), axis=axis, keepdims=True)
    scale = np.maximum(amax, eps) / FP8_E4M3_MAX
    x_scaled = x / scale
    x_q = quantize_e4m3_simulated(x_scaled)
    x_dq = x_q * scale
    return x_dq, scale
def attention_fp64_reference(Q, K, V, causal=False):
    Q64, K64, V64 = Q.astype(np.float64), K.astype(np.float64), V.astype(np.float64)
    d = Q.shape[-1]
    S = (Q64 @ K64.transpose(0, 1, 3, 2)) / np.sqrt(d)
    if causal:
        n = S.shape[-1]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        S = np.where(mask, -np.inf, S)
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    P = P / P.sum(axis=-1, keepdims=True)
    O = P @ V64
    return O.astype(np.float32)
def attention_fp8_naive(Q, K, V, causal=False):
    d = Q.shape[-1]
    Q_dq, _ = quantize_tile_dynamic(Q, axis=None)  
    K_dq, _ = quantize_tile_dynamic(K, axis=None)
    V_dq, _ = quantize_tile_dynamic(V, axis=None)

    S = (Q_dq @ K_dq.transpose(0, 1, 3, 2)) / np.sqrt(d)
    if causal:
        n = S.shape[-1]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        S = np.where(mask, -np.inf, S)
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    P = P / P.sum(axis=-1, keepdims=True)
    P_dq, _ = quantize_tile_dynamic(P, axis=None)
    O = P_dq @ V_dq
    return O.astype(np.float32)
def attention_fp8_online_dq(Q, K, V, block_m=64, block_n=64, causal=False):
    B, H, N, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    O = np.zeros((B, H, N, d), dtype=np.float32)
    for b in range(B):
        for h in range(H):
            Qb, Kb, Vb = Q[b, h], K[b, h], V[b, h]
            for m0 in range(0, N, block_m):
                m1 = min(m0 + block_m, N)
                Qi = Qb[m0:m1]  # (bm, d)
                # per-Q-tile dynamic quantization
                Qi_dq, _ = quantize_tile_dynamic(Qi, axis=None)

                bm = m1 - m0
                m_i = np.full((bm, 1), -np.inf, dtype=np.float32)  
                l_i = np.zeros((bm, 1), dtype=np.float32)       
                acc = np.zeros((bm, d), dtype=np.float32)           

                n_end = m1 if causal else N
                for n0 in range(0, n_end, block_n):
                    n1 = min(n0 + block_n, n_end)
                    Kj = Kb[n0:n1]  # (bn, d)
                    Vj = Vb[n0:n1]  # (bn, d)
                    Kj_dq, _ = quantize_tile_dynamic(Kj, axis=None)
                    Sij = (Qi_dq @ Kj_dq.T) * scale 
                    if causal and n1 > m0:
                        row_idx = np.arange(m0, m1)[:, None]
                        col_idx = np.arange(n0, n1)[None, :]
                        mask = col_idx > row_idx
                        Sij = np.where(mask, -np.inf, Sij)
                    m_ij = np.maximum(m_i, Sij.max(axis=-1, keepdims=True))
                    p_ij = np.exp(Sij - m_ij)
                    alpha = np.exp(m_i - m_ij)  
                    l_i = l_i * alpha + p_ij.sum(axis=-1, keepdims=True)
                    p_ij_dq, _ = quantize_tile_dynamic(p_ij, axis=None)
                    Vj_dq, _ = quantize_tile_dynamic(Vj, axis=None)
                    pv = p_ij_dq @ Vj_dq  
                    acc = acc * alpha + pv
                    m_i = m_ij
                O[b, h, m0:m1] = acc / l_i
    return O
def rel_error(x, ref):
    num = np.linalg.norm((x - ref).reshape(-1))
    den = np.linalg.norm(ref.reshape(-1)) + 1e-12
    return num / den

def max_abs_error(x, ref):
    return np.max(np.abs(x - ref))

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B, H, N, d = 2, 4, 256, 64
    Q = rng.standard_normal((B, H, N, d)).astype(np.float32)
    K = rng.standard_normal((B, H, N, d)).astype(np.float32)
    V = rng.standard_normal((B, H, N, d)).astype(np.float32)
    outlier_channels = rng.choice(d, size=3, replace=False)
    Q[..., outlier_channels] *= 15.0
    K[..., outlier_channels] *= 15.0
    for causal in (False, True):
        print(f"\n=== causal={causal} ===")
        ref = attention_fp64_reference(Q, K, V, causal=causal)

        naive = attention_fp8_naive(Q, K, V, causal=causal)
        dq = attention_fp8_online_dq(Q, K, V, block_m=64, block_n=64, causal=causal)

        print(f"naive single-scale fp8   : rel_err={rel_error(naive, ref):.4e}  "
              f"max_abs_err={max_abs_error(naive, ref):.4e}")
        print(f"online double-quant fp8  : rel_err={rel_error(dq, ref):.4e}  "
              f"max_abs_err={max_abs_error(dq, ref):.4e}")