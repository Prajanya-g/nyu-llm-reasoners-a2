from __future__ import annotations

from typing import Type

import torch

try:
    from student.flash_attention_pytorch import FlashAttentionPyTorch
except ImportError:
    FlashAttentionPyTorch = None  # type: ignore[misc, assignment]

try:
    from student.flash_attention_triton import FlashAttentionTriton
except ImportError:
    FlashAttentionTriton = None  # type: ignore[misc, assignment]


def get_flashattention_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    if FlashAttentionPyTorch is None:
        raise NotImplementedError(
            "student.flash_attention_pytorch.FlashAttentionPyTorch not found"
        )
    return FlashAttentionPyTorch


def get_flashattention_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and (optionally) backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    if FlashAttentionTriton is None:
        raise NotImplementedError(
            "student.flash_attention_triton.FlashAttentionTriton not found"
        )
    return FlashAttentionTriton


