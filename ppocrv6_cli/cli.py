from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ppocrv6_cli import __version__
from ppocrv6_cli.engine import _is_url


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format", "-f",
        choices=["json", "text", "table"],
        default="text",
        help="Output format (default: text). Use 'json' for agent-friendly output.",
    )
    parser.add_argument(
        "--pretty", "-p",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        help="Write output to file instead of stdout.",
    )
    parser.add_argument(
        "--confidence-threshold", "-c",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="Filter results below this confidence (default: 0.0).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help=f"Model directory (default: ~/.ppocrv6-cli/models).",
    )
    parser.add_argument(
        "--model-size",
        choices=["tiny", "small", "medium"],
        default="tiny",
        help="Model size variant (default: tiny).",
    )
    parser.add_argument(
        "--accelerator",
        action="store_true",
        help="Enable GPU/CoreML acceleration.",
    )


def _cmd_ocr(args: argparse.Namespace) -> int:
    from ppocrv6_cli.engine import OCREngine
    from ppocrv6_cli.formatter import output_result

    image_path = args.image
    if not _is_url(image_path) and not Path(image_path).is_file():
        print(f"Error: image not found: {image_path}", file=sys.stderr, flush=True)
        return 1

    try:
        with OCREngine(
            model_dir=args.model_dir,
            size=args.model_size,
            accelerator=args.accelerator,
        ) as engine:
            result = engine.ocr_image(image_path, confidence_threshold=args.confidence_threshold)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr, flush=True)
        return 1

    output_result(result, fmt=args.format, pretty=args.pretty, output_file=args.output)
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    from ppocrv6_cli.engine import OCREngine
    from ppocrv6_cli.formatter import output_result

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: not a directory: {directory}", file=sys.stderr, flush=True)
        return 1

    try:
        with OCREngine(
            model_dir=args.model_dir,
            size=args.model_size,
            accelerator=args.accelerator,
        ) as engine:
            results = engine.ocr_batch(
                directory,
                recursive=args.recursive,
                confidence_threshold=args.confidence_threshold,
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr, flush=True)
        return 1

    if not results:
        print("No supported images found.", file=sys.stderr, flush=True)
        return 1

    output_result(results, fmt=args.format, pretty=args.pretty, output_file=args.output)
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from ppocrv6_cli.downloader import download_models

    try:
        download_models(
            model_dir=args.model_dir,
            size=args.model_size,
            force=args.force,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr, flush=True)
        return 2

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ppocrv6",
        description="Agent-friendly CLI for PP-OCRv6 text detection & recognition.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ocr
    p_ocr = sub.add_parser(
        "ocr",
        help="Run OCR on a single image (local file or URL)",
        description="Run OCR on a single image. Supports local files and URLs (http/https).\n"
                    "Tip: wrap URLs in quotes to avoid shell interpretation of special characters.",
    )
    p_ocr.add_argument("image", help="Local file path or URL (http/https) to the image")
    _add_common_args(p_ocr)

    # batch
    p_batch = sub.add_parser("batch", help="Run OCR on all images in a directory")
    p_batch.add_argument("directory", help="Path to the image directory")
    p_batch.add_argument("--recursive", "-r", action="store_true", help="Scan subdirectories")
    _add_common_args(p_batch)

    # download
    p_dl = sub.add_parser("download", help="Download model files")
    p_dl.add_argument(
        "--model-dir", type=Path,
        help="Model directory (default: ~/.ppocrv6-cli/models).",
    )
    p_dl.add_argument(
        "--model-size",
        choices=["tiny", "small", "medium"],
        default="tiny",
        help="Model size variant (default: tiny).",
    )
    p_dl.add_argument("--force", action="store_true", help="Re-download even if files exist")

    return parser


def main(argv: list[str] | None = None) -> int:
    import os
    os.environ["PYTHONUNBUFFERED"] = "1"

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    handlers = {
        "ocr": _cmd_ocr,
        "batch": _cmd_batch,
        "download": _cmd_download,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
