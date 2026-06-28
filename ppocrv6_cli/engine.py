from __future__ import annotations

import tempfile
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from ppocrv6_cli.ppocrv6_onnx import OCRResult, PPOCRv6Onnx
from ppocrv6_cli.downloader import download_models, model_paths, models_ready

_SUPPORTED_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"})


def _is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _read_image(source: Union[str, Path]) -> np.ndarray:
    if isinstance(source, Path):
        source = str(source)

    if _is_url(source):
        req = urllib.request.Request(source, headers={"User-Agent": "ppocrv6-cli"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            img = cv2.imread(tmp.name)
        if img is None:
            raise ValueError(f"Cannot decode image from URL: {source}")
        return img

    img = cv2.imread(source)
    if img is None:
        raise ValueError(f"Cannot read image: {source}")
    return img


class OCREngine:
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        size: str = "tiny",
        accelerator: bool = False,
        det_thresh: float = 0.3,
        det_box_thresh: float = 0.6,
        rec_batch_size: int = 6,
    ) -> None:
        if not models_ready(model_dir, size):
            print("Downloading PP-OCRv6 models (first run)...\n")
            download_models(model_dir, size)
            print()

        paths = model_paths(model_dir, size)
        self._ocr = PPOCRv6Onnx(
            det_model_path=str(paths["det_model"]),
            rec_model_path=str(paths["rec_model"]),
            rec_char_dict_path=str(paths["char_dict"]),
            det_thresh=det_thresh,
            det_box_thresh=det_box_thresh,
            rec_batch_size=rec_batch_size,
            prefer_accelerator=accelerator,
        )

    def close(self) -> None:
        self._ocr.close()

    def __enter__(self) -> OCREngine:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def ocr_image(self, image_path: Union[str, Path], confidence_threshold: float = 0.0) -> dict:
        source = str(image_path)
        img = _read_image(source)

        h, w = img.shape[:2]
        t0 = time.monotonic()
        results: List[OCRResult] = self._ocr(img)
        elapsed_ms = round((time.monotonic() - t0) * 1000)

        items = []
        for r in results:
            if r.score >= confidence_threshold:
                items.append({
                    "text": r.text,
                    "confidence": round(r.score, 6),
                    "bbox": r.box,
                })

        return {
            "image": source,
            "width": w,
            "height": h,
            "results": items,
            "total_texts": len(items),
            "elapsed_ms": elapsed_ms,
        }

    def ocr_batch(
        self,
        directory: Path,
        recursive: bool = False,
        confidence_threshold: float = 0.0,
    ) -> list[dict]:
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        pattern = "**/*" if recursive else "*"
        images = sorted(
            p for p in directory.glob(pattern)
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS
        )

        results = []
        for img_path in images:
            try:
                results.append(self.ocr_image(img_path, confidence_threshold))
            except Exception as e:
                results.append({
                    "image": str(img_path),
                    "error": str(e),
                    "results": [],
                    "total_texts": 0,
                    "elapsed_ms": 0,
                })
        return results
