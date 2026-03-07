"""FlashAttention-2 backward pass using recomputation (PyTorch + torch.compile).

Shared by both the pure PyTorch and Triton autograd.Function implementations.
Equations 13–19: recompute P from S = Q@K^T*scale and L, then dP, D, dS, dQ, dK, dV.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def _flash_backward_impl(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    dO: torch.Tensor,
    L: torch.Tensor,
    is_causal: bool,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass: inputs Q, K, V, O, dO, L; returns dQ, dK, dV. Uses D vector."""
    # S = Q @ K^T * scale  (B, Nq, Nk)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if is_causal:
        n_queries = Q.shape[1]
        n_keys = K.shape[1]
        q_idx = torch.arange(n_queries, device=Q.device, dtype=torch.long)
        k_idx = torch.arange(n_keys, device=K.device, dtype=torch.long)
        causal_mask = q_idx.view(1, -1, 1) >= k_idx.view(1, 1, -1)
        S = torch.where(causal_mask, S, torch.full_like(S, -1e6))

    # P = exp(S - L); L is logsumexp(S) so P = softmax(S)
    P = torch.exp(S - L.unsqueeze(-1))

    # Eq: dP = dO @ V^T
    dP = torch.matmul(dO, V.transpose(-2, -1))

    # D vector: D_i = sum_j P_ij * dP_ij  (B, Nq)
    D = (P * dP).sum(dim=-1)

    # dS = P * (dP - D)
    dS = P * (dP - D.unsqueeze(-1))

    # dQ = scale * (dS @ K), dK = scale * (dS^T @ Q), dV = P^T @ dO
    dQ = scale * torch.matmul(dS, K)
    dK = scale * torch.matmul(dS.transpose(-2, -1), Q)
    dV = torch.matmul(P.transpose(-2, -1), dO)

    return dQ, dK, dV


# Compiled backward for use by both PyTorch and Triton autograd backwards
_flash_backward_compiled = torch.compile(_flash_backward_impl)


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
    Uses recomputation (no stored softmax). Call torch.compile'd implementation.
    """
    scale = 1.0 / math.sqrt(Q.shape[-1])
    return _flash_backward_compiled(Q, K, V, O, dO, L, is_causal, scale)
