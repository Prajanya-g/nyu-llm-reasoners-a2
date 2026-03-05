#!/usr/bin/env python3
"""End-to-end benchmark for forward and backward passes.

Given hyperparameters, initializes a model, generates a random batch, runs warm-up
steps, then times n steps (forward-only or forward+backward) using timeit.default_timer()
and torch.cuda.synchronize() after each step.

Warmup comparison (for writeup): run with --compare_warmup to get a table for
warmup=0, 1, 2, 5. Example:
  uv run python -m student.benchmark --compare_warmup --size small
  uv run python -m student.benchmark --compare_warmup --output warmup.csv --latex
"""

from __future__ import annotations

import argparse
import math
import timeit
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from a1_basics.model import BasicsTransformerLM
from a1_basics.data import get_batch

# §1.1.2 Table 1: vocab_size=10_000, batch_size=4 for all; context_length varies.
MODEL_SIZES: dict[str, dict[str, Any]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}
VOCAB_SIZE_REF = 10_000
BATCH_SIZE_REF = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark forward and/or backward passes of BasicsTransformerLM."
    )
    # Model hyperparameters
    p.add_argument(
        "--size",
        type=str,
        choices=list(MODEL_SIZES),
        default=None,
        help="§1.1.2 Table 1 preset (uses vocab_size=10k, batch_size=4)",
    )
    p.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    p.add_argument("--context_length", type=int, default=128, help="Context length")
    p.add_argument("--d_model", type=int, default=768, help="Model dimension")
    p.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    p.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    p.add_argument("--d_ff", type=int, default=3072, help="FFN inner dimension")
    p.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta")
    # Batch and timing (§1.1.2: 5 warmup, 10 measurement steps)
    p.add_argument("--batch_size", type=int, default=8, help="Batch size")
    p.add_argument("--warmup", type=int, default=5, help="Warm-up steps (w)")
    p.add_argument("--steps", type=int, default=10, help="Timed steps (n)")
    p.add_argument(
        "--mode",
        type=str,
        choices=["forward", "forward_backward"],
        default="forward_backward",
        help="Whether to time forward only or forward + backward",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    # §1.1.2: run all Table 1 sizes and output table for writeup
    p.add_argument(
        "--all_sizes",
        action="store_true",
        help="Run forward and forward_backward for each Table 1 size; output table",
    )
    p.add_argument(
        "--sizes",
        type=str,
        nargs="*",
        default=None,
        help="With --all_sizes: only run these sizes e.g. small medium (default: all)",
    )
    p.add_argument("--output", type=str, default=None, help="Write table to path (with --all_sizes)")
    p.add_argument("--latex", action="store_true", help="Print/write LaTeX table (with --all_sizes)")
    p.add_argument("--markdown", action="store_true", help="Print/write Markdown table (with --all_sizes)")
    # Warmup comparison for writeup: effect of 0, 1, 2, 5 warmup steps
    p.add_argument(
        "--compare_warmup",
        action="store_true",
        help="Run same config with warmup 0, 1, 2, 5; output comparison table",
    )
    return p.parse_args()


def run_benchmark(
    *,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    batch_size: int,
    mode: str,
    device: str,
    warmup: int = 5,
    steps: int = 10,
    rope_theta: float = 10000.0,
) -> tuple[float, float]:
    """Run benchmark; return (mean_ms, std_ms) over steps. Uses timeit.default_timer and cuda sync."""
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(device)

    rng = np.random.default_rng(42)
    dataset_size = max(
        context_length + 1, batch_size * (context_length + 1)
    )
    dataset = rng.integers(0, vocab_size, size=dataset_size).astype(np.int64)
    x, y = get_batch(
        dataset, batch_size=batch_size, context_length=context_length, device=device
    )

    do_backward = mode == "forward_backward"
    if do_backward:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    else:
        model.eval()

    for _ in range(warmup):
        logits = model(x)
        if do_backward:
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), y.view(-1), ignore_index=-100
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    timer = timeit.default_timer
    step_times_s: list[float] = []
    for _ in range(steps):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = timer()
        logits = model(x)
        if do_backward:
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), y.view(-1), ignore_index=-100
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        step_times_s.append(timer() - start)

    step_times_ms = np.array(step_times_s) * 1000
    return float(np.mean(step_times_ms)), float(np.std(step_times_ms))


def main() -> None:
    args = parse_args()
    device = args.device

    if args.all_sizes:
        _run_all_sizes(args)
        return

    if args.compare_warmup:
        _run_compare_warmup(args)
        return

    # Apply §1.1.2 preset if --size set
    if args.size is not None:
        preset = MODEL_SIZES[args.size]
        args.vocab_size = VOCAB_SIZE_REF
        args.batch_size = BATCH_SIZE_REF
        args.d_model = preset["d_model"]
        args.d_ff = preset["d_ff"]
        args.num_layers = preset["num_layers"]
        args.num_heads = preset["num_heads"]

    mean_ms, std_ms = run_benchmark(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        batch_size=args.batch_size,
        mode=args.mode,
        device=device,
        warmup=args.warmup,
        steps=args.steps,
        rope_theta=args.rope_theta,
    )
    print(f"mode={args.mode} warmup={args.warmup} steps={args.steps} device={device}")
    print(f"per_step_ms: mean={mean_ms:.2f} std={std_ms:.2f} (mean±std: {mean_ms:.2f}±{std_ms:.2f})")


def _run_compare_warmup(args: argparse.Namespace) -> None:
    """Run same config with warmup=0, 1, 2, 5; output table for writeup (no warmup vs warmup)."""
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

    # Use §1.1.2 small by default so comparison runs quickly
    size_name = args.size if args.size is not None else "small"
    preset = MODEL_SIZES[size_name]
    cfg = {
        "vocab_size": VOCAB_SIZE_REF,
        "batch_size": BATCH_SIZE_REF,
        "context_length": args.context_length,
        "d_model": preset["d_model"],
        "d_ff": preset["d_ff"],
        "num_layers": preset["num_layers"],
        "num_heads": preset["num_heads"],
        "steps": args.steps,
        "device": args.device,
    }

    warmup_values = (0, 1, 2, 5)
    rows: list[dict[str, Any]] = []
    for w in warmup_values:
        cfg["warmup"] = w
        fwd_mean, fwd_std = run_benchmark(mode="forward", **cfg)
        full_mean, full_std = run_benchmark(mode="forward_backward", **cfg)
        bwd_mean = full_mean - fwd_mean
        bwd_std = math.sqrt(fwd_std**2 + full_std**2)
        rows.append({
            "warmup_steps": w,
            "forward_mean_ms": round(fwd_mean, 2),
            "forward_std_ms": round(fwd_std, 2),
            "backward_mean_ms": round(bwd_mean, 2),
            "backward_std_ms": round(bwd_std, 2),
            "full_step_mean_ms": round(full_mean, 2),
            "full_step_std_ms": round(full_std, 2),
        })

    df = table_from_records(rows)
    print(
        f"Warmup comparison: size={size_name} steps={args.steps} device={args.device} "
        f"(vocab={VOCAB_SIZE_REF} batch={BATCH_SIZE_REF})\n"
    )
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
                "caption": "Effect of warm-up steps on timings (mean±std ms)",
                "label": "tab:warmup",
            },
        )
        print(f"\nWrote {args.output}" + (" (+ .tex, .md)" if (args.latex or args.markdown) else ""))


def _run_all_sizes(args: argparse.Namespace) -> None:
    """Run forward and forward_backward for each Table 1 size; build and output table."""
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

    device = args.device
    warmup = args.warmup
    steps = args.steps
    context_length = args.context_length

    size_order = list(MODEL_SIZES)
    if args.sizes:
        sizes_to_run = [s for s in size_order if s in args.sizes]
    else:
        sizes_to_run = size_order

    rows: list[dict[str, Any]] = []
    for size_name in sizes_to_run:
        preset = MODEL_SIZES[size_name]
        cfg = {
            "vocab_size": VOCAB_SIZE_REF,
            "batch_size": BATCH_SIZE_REF,
            "context_length": context_length,
            "d_model": preset["d_model"],
            "d_ff": preset["d_ff"],
            "num_layers": preset["num_layers"],
            "num_heads": preset["num_heads"],
            "warmup": warmup,
            "steps": steps,
            "device": device,
        }
        fwd_mean, fwd_std = run_benchmark(mode="forward", **cfg)
        full_mean, full_std = run_benchmark(mode="forward_backward", **cfg)
        bwd_mean = full_mean - fwd_mean
        bwd_std = math.sqrt(fwd_std**2 + full_std**2)

        rows.append({
            "size": size_name,
            "forward_mean_ms": round(fwd_mean, 2),
            "forward_std_ms": round(fwd_std, 2),
            "backward_mean_ms": round(bwd_mean, 2),
            "backward_std_ms": round(bwd_std, 2),
            "full_step_mean_ms": round(full_mean, 2),
            "full_step_std_ms": round(full_std, 2),
        })

    df = table_from_records(rows)
    print(
        f"§1.1.2 timings: warmup={warmup} steps={steps} vocab_size={VOCAB_SIZE_REF} "
        f"batch_size={BATCH_SIZE_REF} context_length={context_length} device={device}\n"
    )
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
            latex_kwargs={"caption": "§1.1.2 forward/backward timings (mean±std ms)", "label": "tab:bench"},
        )
        print(f"\nWrote {args.output}" + (" (+ .tex, .md)" if (args.latex or args.markdown) else ""))


if __name__ == "__main__":
    main()
