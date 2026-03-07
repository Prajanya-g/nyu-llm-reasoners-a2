"""FlashAttention-2 backward pass using recomputation (PyTorch, no torch.compile).

Shared by both the pure PyTorch and Triton autograd.Function implementations.
Equations 13–19: recompute P from S = Q@K^T*scale and L, then dP, D, dS, dQ, dK, dV.
S and P are freed as soon as no longer needed to avoid OOM. Not using torch.compile
so backward runs correctly on CPU (e.g. Gradescope).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def flash_attention_backward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    dO: torch.Tensor,
    L: torch.Tensor,
    is_causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward for FlashAttention-2. Takes Q, K, V, O, dO, L; returns dQ, dK, dV.
    Uses recomputation (no stored softmax). Frees S and P early to reduce memory.
    """
    scale = 1.0 / math.sqrt(Q.shape[-1])

    # S = Q @ K^T * scale  (B, Nq, Nk)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if is_causal:
        n_queries = Q.shape[1]
        n_keys = K.shape[1]
        q_idx = torch.arange(n_queries, device=Q.device, dtype=torch.long)
        k_idx = torch.arange(n_keys, device=K.device, dtype=torch.long)
        causal_mask = q_idx.view(1, -1, 1) >= k_idx.view(1, 1, -1)
        S = torch.where(causal_mask, S, torch.full_like(S, -1e6))

    # P = exp(S - L); free S immediately after
    P = torch.exp(S - L.unsqueeze(-1))
    del S

    # Eq: dP = dO @ V^T
    dP = torch.matmul(dO, V.transpose(-2, -1))

    # D vector: D_i = sum_j P_ij * dP_ij  (B, Nq)
    D = (P * dP).sum(dim=-1)

    # dS = P * (dP - D); then free dP
    dS = P * (dP - D.unsqueeze(-1))
    del dP

    # dQ, dK from dS; then free dS
    dQ = scale * torch.matmul(dS, K)
    dK = scale * torch.matmul(dS.transpose(-2, -1), Q)
    del dS

    # dV = P^T @ dO; then free P
    dV = torch.matmul(P.transpose(-2, -1), dO)
    del P

    return dQ, dK, dV
