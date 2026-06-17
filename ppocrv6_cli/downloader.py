from __future__ import annotations

import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

_BASE_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/tmp"

_DICT_URL = "https://raw.githubusercontent.com/AIwork4me/ppocrv6_onnx/main/models/rec_char_dict.txt"

_MODELS = {
    "tiny": {
        "det": "PP-OCRv6_tiny_det_onnx",
        "rec": "PP-OCRv6_tiny_rec_0515_onnx",
    },
    "small": {
        "det": "PP-OCRv6_small_det_onnx",
        "rec": "PP-OCRv6_small_rec_0515_onnx",
    },
    "medium": {
        "det": "PP-OCRv6_medium_det_onnx",
        "rec": "PP-OCRv6_medium_rec_0515_onnx",
    },
}

DEFAULT_MODEL_DIR = Path.home() / ".ppocrv6-cli" / "models"


def _download_with_progress(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "ppocrv6-cli"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 64 * 1024
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    bar_w = 30
                    filled = int(bar_w * downloaded // total)
                    bar = "#" * filled + "-" * (bar_w - filled)
                    mb = downloaded / 1024 / 1024
                    total_mb = total / 1024 / 1024
                    print(f"\r  [{bar}] {pct:5.1f}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)
    print()


def _extract_tar(tar_path: Path, dest_dir: Path) -> None:
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(path=dest_dir)


def model_paths(model_dir: Optional[Path] = None, size: str = "tiny") -> dict[str, Path]:
    base = model_dir or DEFAULT_MODEL_DIR
    names = _MODELS.get(size)
    if names is None:
        raise ValueError(f"Unknown model size: {size!r}. Choose from: {list(_MODELS)}")
    return {
        "det_model": base / names["det"] / "inference.onnx",
        "rec_model": base / names["rec"] / "inference.onnx",
        "char_dict": base / "rec_char_dict.txt",
    }


def models_ready(model_dir: Optional[Path] = None, size: str = "tiny") -> bool:
    paths = model_paths(model_dir, size)
    return all(p.is_file() for p in paths.values())


def download_models(
    model_dir: Optional[Path] = None,
    size: str = "tiny",
    force: bool = False,
) -> Path:
    base = model_dir or DEFAULT_MODEL_DIR
    base.mkdir(parents=True, exist_ok=True)

    names = _MODELS.get(size)
    if names is None:
        raise ValueError(f"Unknown model size: {size!r}. Choose from: {list(_MODELS)}")

    for key in ("det", "rec"):
        model_name = names[key]
        onnx_path = base / model_name / "inference.onnx"
        if onnx_path.is_file() and not force:
            print(f"  [skip] {model_name} already exists")
            continue

        tar_name = f"{model_name}.tar"
        url = f"{_BASE_URL}/{tar_name}"
        print(f"  [download] {model_name} ...")

        with tempfile.TemporaryDirectory() as tmp:
            tar_path = Path(tmp) / tar_name
            _download_with_progress(url, tar_path)
            print(f"  [extract] {tar_name} -> {base}")
            _extract_tar(tar_path, base)

    dict_path = base / "rec_char_dict.txt"
    if not dict_path.is_file() or force:
        print("  [download] rec_char_dict.txt ...")
        _download_with_progress(_DICT_URL, dict_path)
    else:
        print("  [skip] rec_char_dict.txt already exists")

    print(f"\nModels ready in: {base}")
    return base
