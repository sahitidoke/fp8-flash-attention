This is a hardware independent, numpy-only implementation of Flash Attention in fp8. Usually, Flash Attention runs its two matmuls in fp16/bf16. However, fp8 tensor cores have 2x throughput which makes it an attractive option.

The standard attention implementation in fp8 computes a single scale for all of Q, a single scale for all of K, etc, and reuses that scale for the entire sequence. Moreover, it also runs only the first matmul in fp8 (QK^T) and PV is done in fp16/bf16.  However, we cannot simply cast a tensor to fp8 naively with one global scale factor and a single outlier value as it wrecks the precision available to everything else (fp8 only has about 2-3 decimal digits of precision and a hard ceiling on the largest representable value (448)).

In this implementation, I adopt double quantization where the scale for Q and K are recomputed as each tile streams through the flash-attention loop and also run the PV matmul in fp8. 

This is tested on simulated data so the script builds a synthetic Q/K/V with injected outlier channels. The program compares rel_error(naive fp8, f64 true flash attention) and rel_error(double quantization fp8, f64 true flash attention). The proposed double quantization outperforms by 20%.

DISCLAIMER: THIS IS HARDWARE INDEPENDENT AND DOES NOT EXPLOIT GPU ARCHITECTURE IT IS JUST VALIDATING THE ALGORITHM VIA ROUNDING APPROXIMATION

