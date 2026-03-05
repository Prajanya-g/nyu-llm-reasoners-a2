"""Utilities for building and exporting tables for writeups (LaTeX, Markdown).

Use pandas DataFrames as the tabular representation, then export via
format_latex() / format_markdown() or write_table().

Example:
    records = [
        {"size": "small", "forward_ms": 12.5, "backward_ms": 28.1},
        {"size": "medium", "forward_ms": 45.2, "backward_ms": 102.3},
    ]
    df = table_from_records(records)
    print(format_latex(df))           # LaTeX string
    print(format_markdown(df))        # Markdown string
    write_table(df, "results.csv", latex=True, markdown=True)  # write all
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def format_latex(df: pd.DataFrame, index: bool = False, **kwargs: Any) -> str:
    """Return DataFrame as a LaTeX table string.

    kwargs are passed through to pandas.DataFrame.to_latex() (e.g. caption, label).
    """
    return df.to_latex(index=index, **kwargs)


def format_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Return DataFrame as a Markdown table string.

    Uses pandas.DataFrame.to_markdown() if tabulate is installed;
    otherwise returns a simple pipe-table built from df.to_string().
    """
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return _markdown_fallback(df, index=index)


def _markdown_fallback(df: pd.DataFrame, index: bool = False) -> str:
    """Simple pipe-style Markdown table without tabulate."""
    if index:
        df = df.reset_index()
    headers = list(df.columns)
    rows = [headers, ["---"] * len(headers)]
    for _, row in df.iterrows():
        rows.append([str(v) for v in row])
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def write_table(
    df: pd.DataFrame,
    path: str | Path | None = None,
    *,
    latex: bool = False,
    markdown: bool = False,
    latex_kwargs: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Format table and optionally write to file(s). Returns dict of format -> string.

    If path is set, writes the default text representation to that path.
    If latex=True, also returns/formats LaTeX (and writes to path.with_suffix('.tex') when path set).
    If markdown=True, also returns/formats Markdown (and writes to path.with_suffix('.md') when path set).

    Returns:
        Dict with keys "text", "latex", "markdown" for each requested format, value is the string.
    """
    path = Path(path) if path else None
    latex_kwargs = latex_kwargs or {}
    out: dict[str, str] = {}

    out["text"] = df.to_string()
    if path:
        Path(path).write_text(out["text"], encoding="utf-8")

    if latex:
        out["latex"] = format_latex(df, **latex_kwargs)
        if path:
            p = Path(path)
            (p.parent / (p.stem + ".tex")).write_text(out["latex"], encoding="utf-8")

    if markdown:
        out["markdown"] = format_markdown(df)
        if path:
            p = Path(path)
            (p.parent / (p.stem + ".md")).write_text(
                out["markdown"], encoding="utf-8"
            )

    return out


def table_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame from a list of dicts (e.g. benchmark rows)."""
    return pd.DataFrame(records)
