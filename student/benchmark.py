#!/usr/bin/env python3
"""End-to-end benchmark for forward and backward passes.

Given hyperparameters, initializes a model, generates a random batch, runs warm-up
steps, then times n steps (forward-only or forward+backward) using timeit.default_timer()
and torch.cuda.synchronize() after each step.

Warmup comparison (for writeup): run with --compare_warmup to get a table for
warmup=0, 1, 2, 5. Example:
  uv run python -m student.benchmark --compare_warmup --size small
  uv run python -m student.benchmark --compare_warmup --output warmup.csv --latex

Nsight Systems (§1.1.4): NVTX is only captured if nsys is run with --trace=nvtx. Use:
  uv run nsys profile --trace=cuda,nvtx -o result.nsys-rep python -m student.benchmark --size small --nvtx
  uv run nsys profile --trace=cuda,nvtx -o result.nsys-rep python -m student.benchmark --size small --nvtx --nvtx_attention
Run on GPU (--device cuda). If NVTX still does not appear (e.g. in Singularity), use
Stats -> CUDA GPU Kernel Summary in Nsight Systems for the report questions.

Mixed precision (§1.1.5): --bf16 runs with torch.autocast(bf16); --compare_bf16 runs each
Table 1 size with full and BF16 and prints a comparison table.
"""

from __future__ import annotations

import argparse
import math
import timeit
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from a1_basics.model import BasicsTransformerLM
from a1_basics.data import get_batch

# NVTX for Nsight Systems (§1.1.4): assignment pattern — import torch.cuda.nvtx as nvtx
nvtx = None
if torch.cuda.is_available():
    try:
        nvtx = torch.cuda.nvtx  # type: ignore[attr-defined]
    except AttributeError:
        pass


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
    # §1.1.4 Nsight Systems: NVTX ranges for filtering warmup / forward / backward in nsys
    p.add_argument(
        "--nvtx",
        action="store_true",
        help="Wrap warmup and forward/backward/optimizer in NVTX ranges for nsys profile",
    )
    p.add_argument(
        "--nvtx_attention",
        action="store_true",
        help="Patch attention with NVTX sub-ranges (attention scores, softmax, final matmul)",
    )
    # §1.1.5 Mixed precision: optional BF16 autocast
    p.add_argument(
        "--bf16",
        action="store_true",
        help="Run with mixed precision (BF16) via torch.autocast",
    )
    p.add_argument(
        "--compare_bf16",
        action="store_true",
        help="Run each Table 1 size with full and BF16; output comparison table",
    )
    return p.parse_args()


def _nvtx_range(name: str, use: bool):
    """Context manager: with nvtx.range(name) when use and nvtx available (assignment pattern)."""
    if use and nvtx is not None:
        return nvtx.range(name)
    return nullcontext()


def _autocast_ctx(use_bf16: bool, device: str):
    """torch.autocast(device_type='cuda', dtype=torch.bfloat16) or no-op (nullcontext)."""
    if use_bf16 and device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _install_nvtx_attention() -> None:
    """Replace a1_basics.model.scaled_dot_product_attention with NVTX-annotated version (assignment pattern)."""
    from einops import einsum
    from a1_basics.nn_utils import softmax
    import a1_basics.model as a1_model

    @nvtx.range("scaled dot product attention")
    def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
        with nvtx.range("computing attention scores"):
            d_k = K.shape[-1]
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))

        with nvtx.range("computing softmax"):
            attention_weights = softmax(attention_scores, dim=-1)

        with nvtx.range("final matmul"):
            return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

    a1_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


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
    use_nvtx: bool = False,
    use_nvtx_attention: bool = False,
    use_bf16: bool = False,
) -> tuple[float, float]:
    """Run benchmark; return (mean_ms, std_ms) over steps. Uses timeit.default_timer and cuda sync."""
    if use_nvtx_attention and nvtx is not None:
        _install_nvtx_attention()

    autocast_ctx = _autocast_ctx(use_bf16, device)

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

    with _nvtx_range("warmup", use_nvtx):
        for _ in range(warmup):
            with autocast_ctx:
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
        with autocast_ctx:
            with _nvtx_range("forward", use_nvtx):
                logits = model(x)
            if do_backward:
                with _nvtx_range("loss", use_nvtx):
                    loss = F.cross_entropy(
                        logits.view(-1, vocab_size), y.view(-1), ignore_index=-100
                    )
                optimizer.zero_grad(set_to_none=True)
                with _nvtx_range("backward", use_nvtx):
                    loss.backward()
                with _nvtx_range("optimizer_step", use_nvtx):
                    optimizer.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        step_times_s.append(timer() - start)

    step_times_ms = np.array(step_times_s) * 1000
    return float(np.mean(step_times_ms)), float(np.std(step_times_ms))


def main() -> None:
    args = parse_args()
    device = args.device

    if args.nvtx:
        if not device.startswith("cuda"):
            print("Warning: --nvtx is set but device is not cuda. NVTX requires CUDA; report may have no NVTX data.")
        if nvtx is not None:
            print("NVTX: using torch.cuda.nvtx. Run nsys with: nsys profile --trace=cuda,nvtx -o out.nsys-rep ...")
        else:
            print("Warning: torch.cuda.nvtx not available. NVTX ranges will be no-ops.")
            print("If NVTX does not show in nsys (e.g. in Singularity), use CUDA GPU Kernel Summary for the report.")

    if args.all_sizes:
        _run_all_sizes(args)
        return

    if args.compare_warmup:
        _run_compare_warmup(args)
        return

    if args.compare_bf16:
        _run_compare_bf16(args)
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
        use_nvtx=args.nvtx,
        use_nvtx_attention=args.nvtx_attention,
        use_bf16=args.bf16,
    )
    prec = "bf16" if args.bf16 else "fp32"
    print(f"mode={args.mode} warmup={args.warmup} steps={args.steps} device={device} precision={prec}")
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


def _run_compare_bf16(args: argparse.Namespace) -> None:
    """Run each Table 1 size with full (FP32) and BF16 mixed precision; output comparison table (§1.1.5)."""
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
        fwd_fp, _ = run_benchmark(mode="forward", use_bf16=False, **cfg)
        full_fp, _ = run_benchmark(mode="forward_backward", use_bf16=False, **cfg)
        fwd_bf16, _ = run_benchmark(mode="forward", use_bf16=True, **cfg)
        full_bf16, _ = run_benchmark(mode="forward_backward", use_bf16=True, **cfg)
        bwd_fp = full_fp - fwd_fp
        bwd_bf16 = full_bf16 - fwd_bf16
        speedup_fwd = fwd_fp / fwd_bf16 if fwd_bf16 > 0 else 0.0
        speedup_full = full_fp / full_bf16 if full_bf16 > 0 else 0.0

        rows.append({
            "size": size_name,
            "forward_fp32_ms": round(fwd_fp, 2),
            "forward_bf16_ms": round(fwd_bf16, 2),
            "full_step_fp32_ms": round(full_fp, 2),
            "full_step_bf16_ms": round(full_bf16, 2),
            "backward_fp32_ms": round(bwd_fp, 2),
            "backward_bf16_ms": round(bwd_bf16, 2),
            "speedup_forward": round(speedup_fwd, 2),
            "speedup_full_step": round(speedup_full, 2),
        })

    df = table_from_records(rows)
    print(
        f"§1.1.5 Full vs BF16 mixed precision: warmup={warmup} steps={steps} "
        f"vocab_size={VOCAB_SIZE_REF} batch_size={BATCH_SIZE_REF} "
        f"context_length={context_length} device={device}\n"
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
                "caption": "Full vs BF16 mixed precision timings (ms) and speedup",
                "label": "tab:bf16",
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
