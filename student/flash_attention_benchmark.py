#!/usr/bin/env python3
"""Benchmark FlashAttention-2 (Triton) vs regular PyTorch attention.

Uses triton.testing.do_bench. Batch size 1, causal masking. Sweeps sequence
lengths (powers of 2 from 128 to 65536), embedding dims (16, 32, 64, 128),
and dtypes (bfloat16, float32). Reports forward, backward, and end-to-end
latencies for both implementations. Run on a single GPU.
"""

from __future__ import annotations

import argparse
import itertools
import math
from typing import Any

import torch
from student.flash_attention_triton import FlashAttentionTriton
from student.table_utils import format_markdown
from student.table_utils import table_from_records

try:
    import triton
    from triton.testing import do_bench
except ImportError:
    do_bench = None
    triton = None

BATCH_SIZE = 1
IS_CAUSAL = True
# Sequence lengths: powers of 2 from 128 to 65536
SEQ_LEN_LIST = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
# Embedding dimensions: powers of 2 from 16 to 128
D_LIST = [16, 32, 64, 128]
DTYPE_LIST = [torch.float32, torch.bfloat16]
WARMUP_MS = 50
REP_MS = 100


def _bench_ms(
    fn: Any,
    warmup: int = WARMUP_MS,
    rep: int = REP_MS,
    grad_to_none: list[torch.Tensor] | None = None,
) -> float:
    """Run do_bench and return median time in ms as a float."""
    out = do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        grad_to_none=grad_to_none,
        return_mode="median",
    )
    return float(out) if not isinstance(out, (list, tuple)) else float(out[0])


def pytorch_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Standard PyTorch causal attention: O = softmax(causal(QK^T/sqrt(d))) @ V."""
    n_queries = q.shape[1]
    n_keys = k.shape[1]
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(q, k.transpose(-2, -1)) * scale
    causal = torch.arange(n_queries, device=q.device)[:, None] >= torch.arange(
        n_keys, device=q.device
    )[None, :]
    S = torch.where(causal, S, torch.full_like(S, -1e6))
    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, v)
    return O


def run_one_config(
    seq_len: int,
    d: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 42,
) -> dict[str, Any]:
    """Benchmark PyTorch and Triton for one (seq_len, d, dtype). Returns one table row."""
    torch.manual_seed(seed)
    row: dict[str, Any] = {
        "seq_len": seq_len,
        "d": d,
        "dtype": str(dtype).replace("torch.", ""),
    }

    # Generate inputs once before benchmarking (assignment requirement)
    q = torch.randn(
        BATCH_SIZE, seq_len, d, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        BATCH_SIZE, seq_len, d, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        BATCH_SIZE, seq_len, d, device=device, dtype=dtype, requires_grad=True
    )
    do = torch.randn(BATCH_SIZE, seq_len, d, device=device, dtype=dtype)

    def _pytorch_forward_fn() -> torch.Tensor:
        return pytorch_causal_attention(q, k, v)

    def _pytorch_fwd_bwd_fn() -> None:
        o = pytorch_causal_attention(q, k, v)
        o.backward(do)

    def _triton_forward_fn() -> torch.Tensor:
        return FlashAttentionTriton.apply(q, k, v, IS_CAUSAL)

    def _triton_fwd_bwd_fn() -> None:
        o = FlashAttentionTriton.apply(q, k, v, IS_CAUSAL)
        o.backward(do)

    try:
        # PyTorch: forward then forward+backward; grad_to_none clears grads between reps
        torch.cuda.synchronize()
        pytorch_fwd_ms = _bench_ms(_pytorch_forward_fn)
        torch.cuda.synchronize()
        pytorch_total_ms = _bench_ms(_pytorch_fwd_bwd_fn, grad_to_none=[q, k, v])
        pytorch_bwd_ms = max(0.0, pytorch_total_ms - pytorch_fwd_ms)

        row["pytorch_fwd_ms"] = round(pytorch_fwd_ms, 3)
        row["pytorch_bwd_ms"] = round(pytorch_bwd_ms, 3)
        row["pytorch_total_ms"] = round(pytorch_total_ms, 3)
    except Exception as e:
        row["pytorch_fwd_ms"] = (
            "OOM" if "out of memory" in str(e).lower() else str(e)[:20]
        )
        row["pytorch_bwd_ms"] = row["pytorch_fwd_ms"]
        row["pytorch_total_ms"] = row["pytorch_fwd_ms"]

    try:
        torch.cuda.empty_cache()
        # Re-create tensors for Triton (same config, fresh grads)
        q = torch.randn(
            BATCH_SIZE, seq_len, d, device=device, dtype=dtype, requires_grad=True
        )
        k = torch.randn(
            BATCH_SIZE, seq_len, d, device=device, dtype=dtype, requires_grad=True
        )
        v = torch.randn(
            BATCH_SIZE, seq_len, d, device=device, dtype=dtype, requires_grad=True
        )
        do = torch.randn(BATCH_SIZE, seq_len, d, device=device, dtype=dtype)

        triton_fwd_ms = _bench_ms(_triton_forward_fn)
        torch.cuda.synchronize()
        triton_total_ms = _bench_ms(_triton_fwd_bwd_fn, grad_to_none=[q, k, v])
        triton_bwd_ms = max(0.0, triton_total_ms - triton_fwd_ms)

        row["triton_fwd_ms"] = round(triton_fwd_ms, 3)
        row["triton_bwd_ms"] = round(triton_bwd_ms, 3)
        row["triton_total_ms"] = round(triton_total_ms, 3)
    except Exception as e:
        row["triton_fwd_ms"] = (
            "OOM" if "out of memory" in str(e).lower() else str(e)[:20]
        )
        row["triton_bwd_ms"] = row["triton_fwd_ms"]
        row["triton_total_ms"] = row["triton_fwd_ms"]

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="FlashAttention-2 vs PyTorch benchmark")
    parser.add_argument(
        "--seq-len",
        type=int,
        nargs="+",
        default=SEQ_LEN_LIST,
        help="Sequence lengths to sweep",
    )
    parser.add_argument(
        "--d",
        type=int,
        nargs="+",
        default=D_LIST,
        help="Embedding dimensions to sweep",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        default=["float32", "bfloat16"],
        choices=["float32", "bfloat16"],
        help="Dtypes to sweep",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_MS,
        help="do_bench warmup (ms)",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=REP_MS,
        help="do_bench rep (ms)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write table to this path (markdown)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a single GPU (CUDA).")
    if do_bench is None:
        raise RuntimeError("triton.testing.do_bench is required; install triton.")

    device = torch.device("cuda")
    global WARMUP_MS, REP_MS
    WARMUP_MS = args.warmup
    REP_MS = args.rep

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    dtypes = [dtype_map[s] for s in args.dtypes]

    configs = list(
        itertools.product(
            args.seq_len,
            args.d,
            dtypes,
        )
    )
    records: list[dict[str, Any]] = []
    for i, (seq_len, d, dtype) in enumerate(configs):
        print(f"[{i + 1}/{len(configs)}] seq_len={seq_len}, d={d}, dtype={dtype}")
        row = run_one_config(seq_len, d, dtype, device)
        records.append(row)
        torch.cuda.empty_cache()

    df = table_from_records(records)
    table_md = format_markdown(df)
    print("\n" + "=" * 80)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Batch size:", BATCH_SIZE, "| Causal:", IS_CAUSAL)
    print("=" * 80)
    print(table_md)

    if args.output:
        from pathlib import Path
        Path(args.output).write_text(
            f"# FlashAttention-2 vs PyTorch benchmark\n\n"
            f"GPU: {torch.cuda.get_device_name(0)}\n"
            f"Batch size: {BATCH_SIZE}, Causal: {IS_CAUSAL}\n\n"
            + table_md,
            encoding="utf-8",
        )
        print(f"\nWrote table to {args.output}")


if __name__ == "__main__":
    main()
