"""
PP-OCRv6 纯 ONNXRuntime API 推理实现。

不依赖 PaddleX，仅使用 onnxruntime + opencv + numpy + pyclipper。
参考 PaddleX 源码，完整复现检测 + 识别的预处理 / 后处理逻辑。

Usage::

    from ppocrv6_onnx import PPOCRv6Onnx, OCRResult

    with PPOCRv6Onnx("det.onnx", "rec.onnx", "dict.txt") as ocr:
        results = ocr(image_bgr)
        for r in results:
            print(r.text, r.score)
"""

from __future__ import annotations

import copy
import functools
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper

__all__ = ["PPOCRv6Onnx", "OCRResult"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 公共类型
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OCRResult:
    """单条 OCR 识别结果。"""

    text: str
    """识别文本。"""
    score: float
    """置信度，范围 [0, 1]。"""
    box: List[List[int]]
    """检测框四个顶点坐标 [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]"""


# ---------------------------------------------------------------------------
# ORT provider 自动选择
# ---------------------------------------------------------------------------

def _resolve_ort_providers(prefer_cpu: bool = False) -> List[str]:
    if prefer_cpu:
        logger.info("ONNX Runtime provider: CPUExecutionProvider (forced)")
        return ["CPUExecutionProvider"]

    available = ort.get_available_providers()
    for preferred in ("CoreMLExecutionProvider", "CUDAExecutionProvider"):
        if preferred in available:
            logger.info("ONNX Runtime provider: %s", preferred)
            return [preferred]
    logger.info("ONNX Runtime provider: CPUExecutionProvider (fallback)")
    return ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# 共用工具：最小面积外接矩形顶点排序
# ---------------------------------------------------------------------------

def _order_minarea_box_points(
    contour: np.ndarray,
) -> Tuple[List[np.ndarray], float]:
    rrect = cv2.minAreaRect(contour)
    pts = sorted(cv2.boxPoints(rrect), key=lambda p: p[0])

    tl, bl = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
    tr, br = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)

    return [pts[tl], pts[tr], pts[br], pts[bl]], min(rrect[1])


# ============================================================
# 检测模型 预处理
# ============================================================

class DetPreProcess:
    _VALID_LIMIT_TYPES = frozenset({"min", "max"})

    def __init__(
        self,
        limit_side_len: int = 64,
        limit_type: str = "min",
        max_side_limit: int = 4000,
    ) -> None:
        if limit_type not in self._VALID_LIMIT_TYPES:
            raise ValueError(
                f"limit_type must be one of {sorted(self._VALID_LIMIT_TYPES)}, "
                f"got {limit_type!r}"
            )
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.max_side_limit = max_side_limit
        self._scale = np.float32(1.0 / 255.0)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __repr__(self) -> str:
        return (
            f"DetPreProcess(limit={self.limit_side_len}, "
            f"type={self.limit_type}, max_side={self.max_side_limit})"
        )

    def _resize_image(self, img: np.ndarray) -> Tuple[np.ndarray, float, float]:
        h, w = img.shape[:2]
        limit = self.limit_side_len

        if self.limit_type == "max":
            ratio = float(limit) / max(h, w) if max(h, w) > limit else 1.0
        else:
            ratio = float(limit) / min(h, w) if min(h, w) < limit else 1.0

        rh = int(h * ratio)
        rw = int(w * ratio)

        if max(rh, rw) > self.max_side_limit:
            ratio = float(self.max_side_limit) / max(rh, rw)
            rh, rw = int(rh * ratio), int(rw * ratio)

        rh = max(int(round(rh / 32) * 32), 32)
        rw = max(int(round(rw / 32) * 32), 32)

        if rh == h and rw == w:
            return img, 1.0, 1.0

        resized = cv2.resize(img, (rw, rh))
        return resized, rh / float(h), rw / float(w)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        return (img.astype(np.float32, copy=False) * self._scale - self._mean) / self._std

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        src_h, src_w = img.shape[:2]
        resized, ratio_h, ratio_w = self._resize_image(img)
        norm = self._normalize(resized)
        chw = np.transpose(norm, (2, 0, 1))
        tensor = chw[np.newaxis, ...].astype(np.float32, copy=False)
        shape = np.array([src_h, src_w, ratio_h, ratio_w], dtype=np.float32)
        return tensor, shape


# ============================================================
# 检测模型 后处理 (DBPostProcess)
# ============================================================

class DBPostProcess:
    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.5,
        max_candidates: int = 1000,
        min_size: int = 3,
    ) -> None:
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.max_candidates = max_candidates
        self.min_size = min_size

    def __repr__(self) -> str:
        return (
            f"DBPostProcess(thresh={self.thresh}, box_thresh={self.box_thresh}, "
            f"unclip={self.unclip_ratio})"
        )

    @staticmethod
    def _box_score(bitmap: np.ndarray, points: np.ndarray) -> float:
        h, w = bitmap.shape[:2]
        box = points.astype(np.float32, copy=True)

        xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return float(cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0])

    def _unclip(self, box: np.ndarray) -> np.ndarray:
        area = cv2.contourArea(box)
        length = cv2.arcLength(box, closed=True)
        distance = area * self.unclip_ratio / length

        po = pyclipper.PyclipperOffset()
        po.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        try:
            expanded = np.asarray(po.Execute(distance))
        except pyclipper.PyclipperException:
            paths = po.Execute(distance)
            if not paths:
                return box
            expanded = np.asarray(paths[0])
        return expanded

    def _extract_boxes(
        self,
        prob: np.ndarray,
        bitmap: np.ndarray,
        dst_w: int,
        dst_h: int,
    ) -> Tuple[np.ndarray, List[float]]:
        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        n_outs = len(outs)
        if n_outs == 3:
            contours = outs[1]
        elif n_outs == 2:
            contours = outs[0]
        else:
            raise RuntimeError(f"cv2.findContours returned {n_outs} values")

        ws, hs = dst_w / bitmap.shape[1], dst_h / bitmap.shape[0]
        boxes: List[np.ndarray] = []
        scores: List[float] = []

        for contour in contours[:self.max_candidates]:
            pts, sside = _order_minarea_box_points(contour)
            if sside < self.min_size:
                continue

            pts_np = np.array(pts, dtype=np.float32)
            score = self._box_score(prob, pts_np.reshape(-1, 2))
            if score < self.box_thresh:
                continue

            expanded = self._unclip(pts_np).reshape(-1, 1, 2)
            expanded_pts, sside2 = _order_minarea_box_points(expanded)
            if sside2 < self.min_size + 2:
                continue

            box = np.array(expanded_pts, dtype=np.float32)
            box[:, 0] = np.clip(np.round(box[:, 0] * ws), 0, dst_w)
            box[:, 1] = np.clip(np.round(box[:, 1] * hs), 0, dst_h)

            boxes.append(box.astype(np.int16))
            scores.append(score)

        if not boxes:
            return np.empty((0, 4, 2), dtype=np.int16), []
        return np.stack(boxes, axis=0), scores

    def __call__(
        self, pred: np.ndarray, img_shape: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        prob_map = pred[0, 0, :, :]
        segmentation = prob_map > self.thresh
        return self._extract_boxes(prob_map, segmentation, int(img_shape[1]), int(img_shape[0]))


# ============================================================
# 框排序
# ============================================================

def sort_quad_boxes(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) <= 1:
        return boxes

    sorted_boxes = sorted(boxes, key=lambda b: (b[0][1], b[0][0]))
    items = list(sorted_boxes)
    n = len(items)
    for i in range(n - 1):
        for j in range(i, -1, -1):
            if abs(items[j + 1][0][1] - items[j][0][1]) < 10 and (
                items[j + 1][0][0] < items[j][0][0]
            ):
                items[j], items[j + 1] = items[j + 1], items[j]
            else:
                break
    return np.array(items, dtype=boxes.dtype)


# ============================================================
# 文本区域裁剪
# ============================================================

def _rotate_crop_image(
    img: np.ndarray, points: np.ndarray
) -> Optional[np.ndarray]:
    pts = points.astype(np.float32)

    crop_w = int(max(
        np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])
    ))
    crop_h = int(max(
        np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])
    ))
    if crop_w < 1 or crop_h < 1:
        return None

    dst_pts = np.float32([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]])

    try:
        M = cv2.getPerspectiveTransform(pts, dst_pts)
    except cv2.error:
        logger.debug("Degenerate quadrilateral in perspective crop, skipping")
        return None

    cropped = cv2.warpPerspective(
        img, M, (crop_w, crop_h),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if cropped.shape[0] / cropped.shape[1] >= 1.5:
        cropped = np.rot90(cropped)
    return cropped


def _minarea_rect_crop(img: np.ndarray, poly: np.ndarray) -> Optional[np.ndarray]:
    box, _ = _order_minarea_box_points(poly.astype(np.int32))
    return _rotate_crop_image(img, np.array(box))


def crop_by_polys(
    img: np.ndarray, dt_polys: np.ndarray
) -> List[Optional[np.ndarray]]:
    results: List[Optional[np.ndarray]] = []
    for poly in dt_polys:
        crop = _minarea_rect_crop(img, copy.deepcopy(poly))
        if crop is not None and crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
            results.append(crop)
        else:
            results.append(None)
    return results


# ============================================================
# 识别模型 预处理
# ============================================================

class RecPreProcess:
    def __init__(self, rec_image_shape: Tuple[int, int, int] = (3, 48, 320)) -> None:
        self._c, self._h, self._w_min = rec_image_shape
        self._max_w = 3200

    def __repr__(self) -> str:
        return f"RecPreProcess(C={self._c}, H={self._h}, W_min={self._w_min})"

    def _resize_norm(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        target_w = int(self._h * max_wh_ratio)

        if target_w > self._max_w:
            resized = cv2.resize(img, (self._max_w, self._h))
            actual_w = self._max_w
            target_w = self._max_w
        else:
            h, w = img.shape[:2]
            actual_w = min(int(math.ceil(self._h * w / float(h))), target_w)
            resized = cv2.resize(img, (actual_w, self._h))

        chw = np.transpose(resized.astype(np.float32, copy=False), (2, 0, 1))
        chw *= 1.0 / 255.0
        chw = (chw - 0.5) / 0.5

        padded = np.zeros((self._c, self._h, target_w), dtype=np.float32)
        padded[:, :, :actual_w] = chw
        return padded

    def _resize_single(self, img: np.ndarray) -> np.ndarray:
        max_ratio = max(self._w_min / self._h, img.shape[1] / float(img.shape[0]))
        return self._resize_norm(img, max_ratio)

    def __call__(self, imgs: List[np.ndarray]) -> np.ndarray:
        resized = [self._resize_single(img) for img in imgs]
        max_w = max(r.shape[2] for r in resized)
        padded = []
        for r in resized:
            pad = max_w - r.shape[2]
            if pad > 0:
                r = np.pad(r, ((0, 0), (0, 0), (0, pad)),
                           mode="constant", constant_values=0)
            padded.append(r)
        return np.stack(padded, axis=0).astype(np.float32, copy=False)


# ============================================================
# 识别模型 后处理 (CTCLabelDecode)
# ============================================================

class CTCLabelDecode:
    def __init__(self, character_dict_path: str) -> None:
        self._blank = 0
        with open(character_dict_path, encoding="utf-8") as f:
            raw = tuple(line.rstrip("\n\r") for line in f)
        self._chars: Tuple[str, ...] = ("blank", *raw)

    def __repr__(self) -> str:
        return f"CTCLabelDecode(vocab_size={len(self._chars)})"

    def __len__(self) -> int:
        return len(self._chars)

    @property
    def vocab_size(self) -> int:
        return len(self._chars)

    def decode(
        self, indices: np.ndarray, probs: Optional[np.ndarray] = None
    ) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        chars = self._chars
        for b in range(len(indices)):
            seq = indices[b]
            keep = np.ones(len(seq), dtype=bool)
            keep[1:] = seq[1:] != seq[:-1]
            keep &= seq != self._blank

            text = "".join(chars[idx] for idx in seq[keep])
            if probs is None:
                score = 1.0
            elif keep.any():
                score = float(probs[b][keep].mean())
            else:
                score = 0.0
            results.append((text, score))
        return results

    def __call__(self, model_output: np.ndarray) -> Tuple[List[str], List[float]]:
        output = np.asarray(model_output)
        indices = output.argmax(axis=-1)
        probs = output.max(axis=-1)
        decoded = self.decode(indices, probs)
        return [d[0] for d in decoded], [d[1] for d in decoded]


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _require_file(path: str, name: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{name}: file not found: {path}")


# ============================================================
# 主推理器
# ============================================================

class PPOCRv6Onnx:
    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        rec_char_dict_path: str,
        *,
        det_limit_side_len: int = 64,
        det_limit_type: str = "min",
        det_max_side_limit: int = 4000,
        det_thresh: float = 0.3,
        det_box_thresh: float = 0.6,
        det_unclip_ratio: float = 1.5,
        rec_image_shape: Tuple[int, int, int] = (3, 48, 320),
        rec_batch_size: int = 6,
        prefer_accelerator: bool = False,
    ) -> None:
        _require_file(det_model_path, "det_model_path")
        _require_file(rec_model_path, "rec_model_path")
        _require_file(rec_char_dict_path, "rec_char_dict_path")
        if rec_batch_size < 1:
            raise ValueError(f"rec_batch_size must be >= 1, got {rec_batch_size}")

        providers = _resolve_ort_providers(prefer_cpu=not prefer_accelerator)

        self._det_session = ort.InferenceSession(det_model_path, providers=providers)
        self._rec_session = ort.InferenceSession(rec_model_path, providers=providers)
        self._det_input = self._det_session.get_inputs()[0].name
        self._rec_input = self._rec_session.get_inputs()[0].name

        self._det_pre = DetPreProcess(
            limit_side_len=det_limit_side_len,
            limit_type=det_limit_type,
            max_side_limit=det_max_side_limit,
        )
        self._det_post = DBPostProcess(
            thresh=det_thresh,
            box_thresh=det_box_thresh,
            unclip_ratio=det_unclip_ratio,
        )
        self._rec_pre = RecPreProcess(rec_image_shape=rec_image_shape)
        self._rec_post = CTCLabelDecode(rec_char_dict_path)
        self._rec_bs = rec_batch_size

        self._closed = False

    @staticmethod
    def _require_open(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(self: PPOCRv6Onnx, *args: Any, **kwargs: Any) -> Any:
            if self._closed:
                raise RuntimeError(
                    "PPOCRv6Onnx has been closed and can no longer be used."
                )
            return method(self, *args, **kwargs)

        return wrapper

    def close(self) -> None:
        if not self._closed:
            self._det_session = None  # type: ignore[assignment]
            self._rec_session = None  # type: ignore[assignment]
            self._closed = True

    def __enter__(self) -> PPOCRv6Onnx:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"PPOCRv6Onnx(det={self._det_pre!r}, rec={self._rec_pre!r}, "
            f"vocab={self._rec_post.vocab_size}, closed={self._closed})"
        )

    @_require_open
    def detect(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        x, shape = self._det_pre(img_bgr)
        outputs = self._det_session.run(None, {self._det_input: x})
        return self._det_post(outputs[0], shape)

    @_require_open
    def recognize(self, img_list: List[np.ndarray]) -> Tuple[List[str], List[float]]:
        n = len(img_list)
        if n == 0:
            return [], []

        indexed = sorted(
            [(i, img.shape[1] / float(img.shape[0])) for i, img in enumerate(img_list)],
            key=lambda x: x[1],
        )
        order = [idx for idx, _ in indexed]
        sorted_imgs = [img_list[idx] for idx in order]

        texts: List[Optional[str]] = [None] * n
        scores: List[Optional[float]] = [None] * n

        for start in range(0, n, self._rec_bs):
            batch = sorted_imgs[start:start + self._rec_bs]
            x = self._rec_pre(batch)
            outputs = self._rec_session.run(None, {self._rec_input: x})
            t, s = self._rec_post(outputs[0])
            for j, (text, score) in enumerate(zip(t, s)):
                orig = order[start + j]
                texts[orig] = text
                scores[orig] = score

        return (
            [t if t is not None else "" for t in texts],
            [s if s is not None else 0.0 for s in scores],
        )

    @_require_open
    def __call__(self, img_bgr: np.ndarray) -> List[OCRResult]:
        boxes, _ = self.detect(img_bgr)
        if len(boxes) == 0:
            return []

        sorted_boxes = sort_quad_boxes(boxes)
        crops = crop_by_polys(img_bgr, sorted_boxes)

        valid_boxes: List[np.ndarray] = []
        valid_crops: List[np.ndarray] = []
        for box, crop in zip(sorted_boxes, crops):
            if crop is not None:
                valid_boxes.append(box)
                valid_crops.append(crop)

        if not valid_crops:
            return []

        texts, scores = self.recognize(valid_crops)

        return [
            OCRResult(text=text, score=score, box=box.tolist())
            for text, score, box in zip(texts, scores, valid_boxes)
        ]
