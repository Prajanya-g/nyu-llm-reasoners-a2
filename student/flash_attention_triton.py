"""FlashAttention-2 forward pass using a Triton kernel (Algorithm 1).

Single loop over key tiles; on-chip buffers in float32; cast P to V dtype for matmul,
and O to output dtype when writing.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
from torch.autograd import Function
from triton import cdiv
from triton import language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices: one program per (query_tile, batch)
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Block pointers for this batch and query tile
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load Q block once (this program handles one query tile)
    Q_i = tl.load(Q_block_ptr)

    # On-chip accumulators in float32 (Algorithm 1)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    m_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) + float("-inf")
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for _ in range(Tk):
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)

        # S_ij = Q_i @ K_j^T * scale  (Q_TILE_SIZE x K_TILE_SIZE)
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale

        # Online softmax: row max and new m
        m_ij = tl.max(S_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)

        # P_ij = exp(S_ij - m_new); sum over keys
        P_ij = tl.exp(S_ij - m_new[:, None])
        sum_P = tl.sum(P_ij, axis=1)
        l_new = alpha * l_i + sum_P

        # Cast P to V dtype before matmul; accumulate in float32
        P_ij_v = P_ij.to(V_j.dtype)
        contrib = tl.dot(P_ij_v, V_j)

        # O_i = (O_i * alpha * l_i + P_ij @ V_j) / l_new
        O_i = (O_i * (alpha * l_i)[:, None] + contrib) / l_new[:, None]

        m_i = m_new
        l_i = l_new

        # Advance K and V block pointers to next key tile
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # L = logsumexp = m_i + log(l_i)
    L_i = m_i + tl.log(tl.maximum(l_i, 1e-12))

    # Cast O_i to output dtype before writing
    out_dtype = O_block_ptr.type.element_ty
    O_i_out = O_i.to(out_dtype)
    tl.store(O_block_ptr, O_i_out)
    tl.store(L_block_ptr, L_i.to(L_block_ptr.type.element_ty))


# Tile sizes (Bq, Bk) >= 16
Q_TILE_SIZE_CONST = 16
K_TILE_SIZE_CONST = 16


class FlashAttentionTriton(Function):
    """FlashAttention-2 forward via Triton kernel. Saves Q, K, V, O, L for backward."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, Nq, D = Q.shape
        Nk = K.shape[1]
        scale = 1.0 / math.sqrt(D)

        O = torch.empty_like(Q)
        L = torch.empty((B, Nq), dtype=Q.dtype, device=Q.device)

        Tq = cdiv(Nq, Q_TILE_SIZE_CONST)
        grid = (Tq, B)

        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            O,
            L,
            stride_qb=Q.stride(0),
            stride_qq=Q.stride(1),
            stride_qd=Q.stride(2),
            stride_kb=K.stride(0),
            stride_kk=K.stride(1),
            stride_kd=K.stride(2),
            stride_vb=V.stride(0),
            stride_vk=V.stride(1),
            stride_vd=V.stride(2),
            stride_ob=O.stride(0),
            stride_oq=O.stride(1),
            stride_od=O.stride(2),
            stride_lb=L.stride(0),
            stride_lq=L.stride(1),
            N_QUERIES=Nq,
            N_KEYS=Nk,
            scale=scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE_CONST,
            K_TILE_SIZE=K_TILE_SIZE_CONST,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], ...]:
        raise NotImplementedError
