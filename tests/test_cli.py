from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_version():
    result = subprocess.run(
        [sys.executable, "-m", "ppocrv6_cli.cli", "--version"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_help():
    result = subprocess.run(
        [sys.executable, "-m", "ppocrv6_cli.cli", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "ocr" in result.stdout
    assert "batch" in result.stdout
    assert "download" in result.stdout


def test_no_command_prints_help():
    result = subprocess.run(
        [sys.executable, "-m", "ppocrv6_cli.cli"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0


def test_ocr_missing_image():
    result = subprocess.run(
        [sys.executable, "-m", "ppocrv6_cli.cli", "ocr", "/nonexistent/image.png"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "not found" in result.stderr


def test_batch_missing_directory():
    result = subprocess.run(
        [sys.executable, "-m", "ppocrv6_cli.cli", "batch", "/nonexistent/dir"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1


def test_formatter_json():
    from ppocrv6_cli.formatter import format_json

    data = {"image": "test.png", "results": [{"text": "hello", "confidence": 0.99, "bbox": [[0,0],[1,0],[1,1],[0,1]]}], "total_texts": 1, "elapsed_ms": 100}
    out = format_json(data, pretty=True)
    parsed = json.loads(out)
    assert parsed["image"] == "test.png"
    assert parsed["total_texts"] == 1
    assert parsed["results"][0]["text"] == "hello"


def test_formatter_text():
    from ppocrv6_cli.formatter import format_text

    data = {"image": "test.png", "width": 100, "height": 50, "results": [{"text": "hello", "confidence": 0.99}], "elapsed_ms": 100}
    out = format_text(data)
    assert "hello" in out
    assert "0.99" in out


def test_formatter_text_no_results():
    from ppocrv6_cli.formatter import format_text

    data = {"image": "test.png", "width": 100, "height": 50, "results": [], "elapsed_ms": 100}
    out = format_text(data)
    assert "No text detected" in out


def test_model_paths():
    from ppocrv6_cli.downloader import model_paths

    paths = model_paths(size="tiny")
    assert "det_model" in paths
    assert "rec_model" in paths
    assert "char_dict" in paths
    assert str(paths["det_model"]).endswith("inference.onnx")


def test_model_paths_invalid_size():
    from ppocrv6_cli.downloader import model_paths

    with pytest.raises(ValueError, match="Unknown model size"):
        model_paths(size="huge")


def test_is_url():
    from ppocrv6_cli.engine import _is_url

    assert _is_url("https://example.com/img.png")
    assert _is_url("http://example.com/img.jpg")
    assert not _is_url("/path/to/image.png")
    assert not _is_url("relative/image.png")
    assert not _is_url("ftp://example.com/img.png")


def test_ocr_url_argument():
    result = subprocess.run(
        [sys.executable, "-m", "ppocrv6_cli.cli", "ocr", "https://example.com/test.png"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 1 or "not found" not in result.stderr
