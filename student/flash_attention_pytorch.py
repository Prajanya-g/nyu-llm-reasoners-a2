"""Pure PyTorch (no Triton) FlashAttention-2 forward pass as autograd.Function.

Tiled implementation with online softmax; saves O, L, Q, K, V for backward.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.autograd import Function

# Tile sizes >= 16; dimensions are powers of 2 and >= 16 in tests.
BR = 16  # query block size
BC = 16  # key block size


class FlashAttentionPyTorch(Function):
    """FlashAttention-2 forward in pure PyTorch (tiled, batched). Returns O; saves L, Q, K, V, O."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # Q, K, V: (batch, seq_q, d), (batch, seq_k, d), (batch, seq_k, d)
        B, Nq, D = Q.shape
        Nk = K.shape[1]
        scale = 1.0 / math.sqrt(D)

        O = torch.zeros(B, Nq, D, dtype=Q.dtype, device=Q.device)
        L = torch.zeros(B, Nq, dtype=Q.dtype, device=Q.device)

        for i in range(0, Nq, BR):
            bq = min(BR, Nq - i)
            q_i = Q[:, i : i + bq, :]  # (B, bq, D)
            o_i = torch.zeros(B, bq, D, dtype=Q.dtype, device=Q.device)
            m_i = torch.full(
                (B, bq),
                float("-inf"),
                dtype=Q.dtype,
                device=Q.device,
            )
            l_i = torch.zeros(B, bq, dtype=Q.dtype, device=Q.device)

            for j in range(0, Nk, BC):
                bk = min(BC, Nk - j)
                k_j = K[:, j : j + bk, :]  # (B, bk, D)
                v_j = V[:, j : j + bk, :]  # (B, bk, D)

                # S_ij = (q_i @ k_j^T) * scale  (B, bq, bk)
                S_ij = torch.matmul(q_i, k_j.transpose(-2, -1)) * scale

                if is_causal:
                    # Causal: query position i+bq-1 cannot attend to key positions >= i+bq
                    # So for query block [i, i+bq), keys [j, j+bk) are masked if j+bk > i+qi for qi in block
                    q_idx = torch.arange(i, i + bq, device=Q.device)
                    k_idx = torch.arange(j, j + bk, device=Q.device)
                    mask = q_idx.view(1, -1, 1) >= k_idx.view(1, 1, -1)
                    S_ij = torch.where(mask, S_ij, float("-inf"))

                # Online softmax update
                m_ij = S_ij.max(dim=-1, keepdim=True).values  # (B, bq, 1)
                m_new = torch.maximum(m_i, m_ij.squeeze(-1))  # (B, bq)

                exp_S = torch.exp(S_ij - m_new.unsqueeze(-1))  # (B, bq, bk)
                sum_exp = exp_S.sum(dim=-1)  # (B, bq)

                alpha = torch.exp(m_i - m_new)  # (B, bq)
                l_new = alpha * l_i + sum_exp  # (B, bq)

                # o_i = (o_i * alpha * l_i + exp_S @ v_j) / l_new
                o_i = (
                    o_i * (alpha * l_i).unsqueeze(-1)
                    + torch.matmul(exp_S, v_j)
                ) / l_new.unsqueeze(-1)

                m_i = m_new
                l_i = l_new

            L[:, i : i + bq] = m_i + torch.log(l_i.clamp(min=1e-12))
            O[:, i : i + bq, :] = o_i

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], ...]:
        raise NotImplementedError
