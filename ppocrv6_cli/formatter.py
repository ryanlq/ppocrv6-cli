from __future__ import annotations

import json
from typing import Optional


def format_json(data: dict | list[dict], pretty: bool = False) -> str:
    indent = 2 if pretty else None
    return json.dumps(data, ensure_ascii=False, indent=indent)


def format_text(data: dict | list[dict]) -> str:
    if isinstance(data, list):
        parts = []
        for item in data:
            parts.append(_format_text_single(item))
        return "\n\n".join(parts)
    return _format_text_single(data)


def _format_text_single(data: dict) -> str:
    lines = [f"Image: {data['image']} ({data.get('width', '?')}x{data.get('height', '?')})"]
    if "error" in data:
        lines.append(f"  Error: {data['error']}")
        return "\n".join(lines)

    results = data.get("results", [])
    if not results:
        lines.append("  No text detected.")
    else:
        lines.append(f"  {len(results)} text region(s) detected:\n")
        for i, r in enumerate(results):
            lines.append(f"  [{i+1:3d}] {r['text']:<40s} ({r['confidence']:.4f})")

    lines.append(f"\n  Elapsed: {data.get('elapsed_ms', '?')}ms")
    return "\n".join(lines)


import io
from rich.console import Console
from rich.table import Table


def format_table(data: dict | list[dict]) -> str:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)

    items = data if isinstance(data, list) else [data]

    for item in items:
        console.print(f"\n[bold]{item['image']}[/bold] ({item.get('width', '?')}x{item.get('height', '?')})")

        if "error" in item:
            console.print(f"  [red]Error: {item['error']}[/red]")
            continue

        results = item.get("results", [])
        if not results:
            console.print("  No text detected.")
            continue

        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=4)
        table.add_column("Text", min_width=20)
        table.add_column("Confidence", justify="right", width=10)
        table.add_column("BBox", style="dim", width=30)

        for i, r in enumerate(results):
            bbox_str = str(r["bbox"])
            if len(bbox_str) > 28:
                bbox_str = bbox_str[:25] + "..."
            table.add_row(str(i+1), r["text"], f"{r['confidence']:.4f}", bbox_str)

        console.print(table)
        console.print(f"  Elapsed: {item.get('elapsed_ms', '?')}ms")

    return buf.getvalue()


def output_result(
    data: dict | list[dict],
    fmt: str = "text",
    pretty: bool = False,
    output_file: Optional[str] = None,
) -> None:
    if fmt == "json":
        text = format_json(data, pretty=pretty)
    elif fmt == "table":
        text = format_table(data)
    else:
        text = format_text(data)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    else:
        print(text)
