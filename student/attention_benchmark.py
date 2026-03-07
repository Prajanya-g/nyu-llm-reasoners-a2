#!/usr/bin/env python3
"""Benchmark attention at different scales: batch=8, single-head (no head dim).

Iterates d_model in [16, 32, 64, 128] and seq_len in [256, 1024, 4096, 8192, 16384].
Times 100 forward and 100 backward passes; warmup and torch.cuda.synchronize().
Reports memory before backward and catches OOM. With --compile, also runs
torch.compile(attention) and reports comparison table (uncompiled vs compiled).
"""

from __future__ import annotations

import argparse
import itertools
import timeit
from typing import Any

import torch

from a1_basics.model import scaled_dot_product_attention

BATCH_SIZE = 8
D_MODEL_LIST = [16, 32, 64, 128]
SEQ_LEN_LIST = [256, 1024, 4096, 8192, 16384]
NUM_WARMUP = 5
NUM_FORWARD = 100
NUM_BACKWARD = 100
DEVICE = "cuda"


def make_causal_mask(
    batch: int, seq_len: int, device: torch.device
) -> torch.Tensor:
    """Causal mask: (batch, seq_len, seq_len), True where query >= key."""
    q = torch.arange(seq_len, device=device).view(1, seq_len, 1)
    k = torch.arange(seq_len, device=device).view(1, 1, seq_len)
    mask = (q >= k).expand(batch, -1, -1)
    return mask


def _run_one_config_with_attn(
    d_model: int,
    seq_len: int,
    attn_fn: Any,
    label: str,
) -> dict[str, Any]:
    """Run forward and backward timing using attn_fn(Q, K, V, mask=mask)."""
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        Q = torch.randn(
            BATCH_SIZE, seq_len, d_model,
            device=DEVICE, dtype=torch.float32, requires_grad=True,
        )
        K = torch.randn(
            BATCH_SIZE, seq_len, d_model,
            device=DEVICE, dtype=torch.float32, requires_grad=True,
        )
        V = torch.randn(
            BATCH_SIZE, seq_len, d_model,
            device=DEVICE, dtype=torch.float32, requires_grad=True,
        )
        mask = make_causal_mask(BATCH_SIZE, seq_len, Q.device)

        for _ in range(NUM_WARMUP):
            out = attn_fn(Q, K, V, mask=mask)
            torch.cuda.synchronize()

        timer = timeit.default_timer
        fwd_times = []
        for _ in range(NUM_FORWARD):
            torch.cuda.synchronize()
            t0 = timer()
            out = attn_fn(Q, K, V, mask=mask)
            torch.cuda.synchronize()
            fwd_times.append(timer() - t0)

        mean_fwd_ms = (sum(fwd_times) / len(fwd_times)) * 1000
        memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

        for _ in range(NUM_WARMUP):
            out = attn_fn(Q, K, V, mask=mask)
            loss = out.sum()
            loss.backward()
            Q.grad = None
            K.grad = None
            V.grad = None
            torch.cuda.synchronize()

        bwd_times = []
        for _ in range(NUM_BACKWARD):
            torch.cuda.synchronize()
            t0 = timer()
            out = attn_fn(Q, K, V, mask=mask)
            loss = out.sum()
            loss.backward()
            Q.grad = None
            K.grad = None
            V.grad = None
            torch.cuda.synchronize()
            bwd_times.append(timer() - t0)

        mean_bwd_ms = (sum(bwd_times) / len(bwd_times)) * 1000
        return {
            f"forward_{label}_ms": round(mean_fwd_ms, 2),
            f"backward_{label}_ms": round(mean_bwd_ms, 2),
            "memory_before_backward_mb": round(memory_mb, 1),
            f"status_{label}": "ok",
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            f"forward_{label}_ms": None,
            f"backward_{label}_ms": None,
            "memory_before_backward_mb": None,
            f"status_{label}": "OOM",
        }


def run_one_config(
    d_model: int,
    seq_len: int,
    include_compiled: bool = False,
) -> dict[str, Any]:
    """Run timing for (d_model, seq_len); optionally include torch.compile comparison."""
    row: dict[str, Any] = {"d_model": d_model, "seq_len": seq_len}

    uncompiled = _run_one_config_with_attn(
        d_model, seq_len, scaled_dot_product_attention, "uncompiled",
    )
    row.update(uncompiled)

    if include_compiled:
        compiled_sdpa = torch.compile(scaled_dot_product_attention)
        compiled = _run_one_config_with_attn(
            d_model, seq_len, compiled_sdpa, "compiled",
        )
        row.update(compiled)

    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark attention (batch=8, single-head).")
    p.add_argument(
        "--compile",
        action="store_true",
        help="Also run torch.compile(attention) and add compiled columns to table",
    )
    p.add_argument("--output", type=str, default=None, help="Write table CSV path")
    p.add_argument("--latex", action="store_true")
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this benchmark requires GPU.")
        return

    rows: list[dict[str, Any]] = []
    configs = list(itertools.product(D_MODEL_LIST, SEQ_LEN_LIST))
    for i, (d_model, seq_len) in enumerate(configs):
        print(
            f"[{i+1}/{len(configs)}] d_model={d_model} seq_len={seq_len} ...",
            flush=True,
        )
        row = run_one_config(
            d_model, seq_len, include_compiled=args.compile,
        )
        rows.append(row)

    try:
        from student.table_utils import (
            format_latex,
            format_markdown,
            table_from_records,
            write_table,
        )
    except ImportError:
        from table_utils import (
            format_latex,
            format_markdown,
            table_from_records,
            write_table,
        )

    df = table_from_records(rows)
    title = "Attention (batch=8, single-head, 100 fwd / 100 bwd)"
    if args.compile:
        title += " — uncompiled vs torch.compile"
    print(f"\n{title}:\n")
    print(df.to_string(index=False))

    if args.latex:
        print("\n--- LaTeX ---\n" + format_latex(df))
    if args.markdown:
        print("\n--- Markdown ---\n" + format_markdown(df))
    if args.output:
        write_table(
            df,
            path=args.output,
            latex=args.latex,
            markdown=args.markdown,
            latex_kwargs={
                "caption": "Attention timings and memory",
                "label": "tab:attention",
            },
        )
        suffix = " (+ .tex, .md)" if (args.latex or args.markdown) else ""
        print(f"\nWrote {args.output}{suffix}")


if __name__ == "__main__":
    main()
