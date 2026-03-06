"""Memory profiling for §1.1.6: dump CUDA snapshot for pytorch.org/memory_viz."""

from __future__ import annotations

import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from a1_basics.data import get_batch
from a1_basics.model import BasicsTransformerLM

MODEL_SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
}
VOCAB_SIZE = 10_000
BATCH_SIZE = 4
ROPE_THETA = 10000.0


def run_memory_profile(
    context_length: int,
    mode: str = "forward",
    bf16: bool = False,
    output: str = "memory_snapshot.pickle",
    size: str = "2.7B",
) -> None:
    device = "cuda"
    cfg = MODEL_SIZES[size]

    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        rope_theta=ROPE_THETA,
        **cfg,
    ).to(device)

    rng = np.random.default_rng(42)
    dataset_size = max(context_length + 1, BATCH_SIZE * (context_length + 1))
    dataset = rng.integers(0, VOCAB_SIZE, size=dataset_size).astype(np.int64)
    x, y = get_batch(
        dataset,
        batch_size=BATCH_SIZE,
        context_length=context_length,
        device=device,
    )

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16
        else nullcontext()
    )

    if mode == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model.train()
    else:
        model.eval()

    # Warmup
    with amp_ctx:
        with torch.no_grad() if mode == "forward" else nullcontext():
            _ = model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Start memory recording
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    with amp_ctx:
        if mode == "forward":
            with torch.no_grad():
                logits = model(x)
        else:
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=-100
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.cuda.memory._dump_snapshot(output)
    torch.cuda.memory._record_memory_history(enabled=None)

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(
        f"Peak memory: {peak_mb:.1f} MB | ctx={context_length} mode={mode} "
        f"bf16={bf16} size={size} -> {output}",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Profile CUDA memory for BasicsTransformerLM; output for memory_viz"
    )
    p.add_argument("--context_length", type=int, default=128)
    p.add_argument("--mode", choices=["forward", "train"], default="forward")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--output", type=str, default="memory_snapshot.pickle")
    p.add_argument(
        "--size",
        type=str,
        choices=list(MODEL_SIZES),
        default="2.7B",
        help="Model size from Table 1 (§1.1.6 uses 2.7B)",
    )
    args = p.parse_args()
    run_memory_profile(
        args.context_length,
        args.mode,
        args.bf16,
        args.output,
        args.size,
    )
