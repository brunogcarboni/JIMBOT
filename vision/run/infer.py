"""Carrega o NanoDet e devolve a caixa da trena com maior score."""

import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

import config
from tracker import Box


def _import_predictor():
    try:
        from demo.demo import Predictor
        return Predictor
    except ImportError:
        candidates = [
            os.environ.get("NANODET_DIR"),
            Path(__file__).resolve().parents[1] / "third_party" / "nanodet",
        ]
        for folder in candidates:
            if folder and (Path(folder) / "demo" / "demo.py").exists():
                sys.path.insert(0, str(folder))
                from demo.demo import Predictor
                return Predictor
    raise SystemExit(
        "Não achei o NanoDet. Instale o pacote ou defina NANODET_DIR "
        "apontando para o clone que tem demo/demo.py."
    )


def _cuda_ready() -> bool:
    import torch

    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def _best_box(results, score_min: float) -> Box | None:
    if not results:
        return None
    if isinstance(results, dict):
        per_image = results.get(0, next(iter(results.values())))
    else:
        per_image = results[0]
    if not isinstance(per_image, dict):
        return None
    raw = per_image.get(0)
    if raw is None:
        return None
    boxes = np.asarray(raw)
    if boxes.size == 0:
        return None
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    keep = boxes[boxes[:, 4] >= score_min]
    if len(keep) == 0:
        return None
    x1, y1, x2, y2, score = keep[int(keep[:, 4].argmax())][:5]
    width = max(1, int(round(x2 - x1)))
    height = max(1, int(round(y2 - y1)))
    return Box(
        cx=float(x1 + x2) / 2.0,
        cy=float(y1 + y2) / 2.0,
        x=int(round(x1)),
        y=int(round(y1)),
        w=width,
        h=height,
        score=float(score),
    )


class TapeDetector:
    def __init__(self, model_path: Path, config_path: Path) -> None:
        from nanodet.util import Logger, cfg, load_config

        Predictor = _import_predictor()
        load_config(cfg, str(config_path))
        device = "cuda:0" if _cuda_ready() else "cpu"
        self._predictor = Predictor(
            cfg, str(model_path), Logger(0, use_tensorboard=False), device=device
        )

    def detect(self, frame) -> Box | None:
        _meta, results = self._predictor.inference(frame)
        return _best_box(results, config.SCORE_MIN)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_current = (boxes[current, 2] - boxes[current, 0]) * (
            boxes[current, 3] - boxes[current, 1]
        )
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_current + area_rest - inter + 1e-6)
        order = rest[iou <= iou_thr]
    return keep


def _center_priors(input_h: int, input_w: int, strides: list[int]) -> np.ndarray:
    levels = []
    for stride in strides:
        feat_h = math.ceil(input_h / stride)
        feat_w = math.ceil(input_w / stride)
        ys, xs = np.meshgrid(
            np.arange(feat_h, dtype=np.float32) * stride,
            np.arange(feat_w, dtype=np.float32) * stride,
            indexing="ij",
        )
        stride_col = np.full(xs.size, stride, dtype=np.float32)
        levels.append(np.stack([xs.reshape(-1), ys.reshape(-1), stride_col], axis=1))
    return np.concatenate(levels, axis=0)


def _decode_preds(
    preds: np.ndarray,
    priors: np.ndarray,
    input_h: int,
    input_w: int,
    reg_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    cls_logits = preds[:, :1]
    reg = preds[:, 1:].reshape(-1, 4, reg_max + 1)
    reg = reg - reg.max(axis=-1, keepdims=True)
    weights = np.exp(reg)
    dist = (weights / weights.sum(axis=-1, keepdims=True) * np.arange(reg_max + 1)).sum(axis=-1)
    dist *= priors[:, 2:3]
    x1 = np.clip(priors[:, 0] - dist[:, 0], 0, input_w)
    y1 = np.clip(priors[:, 1] - dist[:, 1], 0, input_h)
    x2 = np.clip(priors[:, 0] + dist[:, 2], 0, input_w)
    y2 = np.clip(priors[:, 1] + dist[:, 3], 0, input_h)
    return np.stack([x1, y1, x2, y2], axis=1), _sigmoid(cls_logits[:, 0])


class Int8TapeDetector:
    """NanoDet int8 no NCNN. A saída bruta é a mesma cabeça do modo float."""

    def __init__(self, param_path: Path, bin_path: Path, config_path: Path) -> None:
        try:
            import ncnn
        except ImportError as exc:
            raise SystemExit(
                "O modo int8 precisa do módulo Python do NCNN, compilado na mesma versão das tools."
            ) from exc
        if not hasattr(ncnn, "Net"):
            raise SystemExit(
                "O import de ncnn não trouxe o módulo. "
                "Rode o main a partir de run/, para o clone em vision/ncnn não esconder o pacote."
            )

        with config_path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        self._mean, self._std = cfg["data"]["val"]["pipeline"]["normalize"]
        self._input_h, self._input_w = (int(v) for v in cfg["data"]["val"]["input_size"])
        head = cfg["model"]["arch"]["head"]
        self._reg_max = int(head["reg_max"])
        self._strides = [int(v) for v in head["strides"]]
        self._channels = 1 + 4 * (self._reg_max + 1)
        self._priors = _center_priors(self._input_h, self._input_w, self._strides)

        self._ncnn = ncnn
        self._net = ncnn.Net()
        self._net.opt.num_threads = 4
        if self._net.load_param(str(param_path)) != 0:
            raise SystemExit(f"Não abriu o param {param_path}")
        if self._net.load_model(str(bin_path)) != 0:
            raise SystemExit(f"Não abriu os pesos {bin_path}")

    def detect(self, frame) -> Box | None:
        frame_h, frame_w = frame.shape[:2]
        scale = np.eye(3, dtype=np.float32)
        scale[0, 0] = self._input_w / frame_w
        scale[1, 1] = self._input_h / frame_h
        resized = cv2.warpPerspective(frame, scale, (self._input_w, self._input_h))
        image = resized.astype(np.float32) / 255.0
        mean = np.asarray(self._mean, dtype=np.float32).reshape(1, 1, 3) / 255.0
        std = np.asarray(self._std, dtype=np.float32).reshape(1, 1, 3) / 255.0
        image = (image - mean) / std
        chw = np.ascontiguousarray(image.transpose(2, 0, 1))

        extractor = self._net.create_extractor()
        extractor.input("in0", self._ncnn.Mat(chw))
        ok, output = extractor.extract("out0")
        if ok != 0:
            return None
        preds = np.asarray(output.numpy(), dtype=np.float32)
        if preds.shape[-1] != self._channels:
            preds = preds.reshape(self._channels, -1).T
        else:
            preds = preds.reshape(-1, self._channels)
        if preds.shape[0] != self._priors.shape[0]:
            raise SystemExit(
                f"A saída int8 tem {preds.shape[0]} pontos e os priors têm {self._priors.shape[0]}."
            )

        boxes, scores = _decode_preds(
            preds, self._priors, self._input_h, self._input_w, self._reg_max
        )
        keep_score = scores >= 0.05
        boxes, scores = boxes[keep_score], scores[keep_score]
        if scores.size == 0:
            return None
        chosen = _nms(boxes, scores, 0.6)
        best = max(chosen, key=lambda index: float(scores[index]))
        if float(scores[best]) < config.SCORE_MIN:
            return None

        inv_x = frame_w / self._input_w
        inv_y = frame_h / self._input_h
        x1 = float(np.clip(boxes[best, 0] * inv_x, 0, frame_w - 1))
        y1 = float(np.clip(boxes[best, 1] * inv_y, 0, frame_h - 1))
        x2 = float(np.clip(boxes[best, 2] * inv_x, 0, frame_w - 1))
        y2 = float(np.clip(boxes[best, 3] * inv_y, 0, frame_h - 1))
        width = max(1, int(round(x2 - x1)))
        height = max(1, int(round(y2 - y1)))
        return Box(
            cx=(x1 + x2) / 2.0,
            cy=(y1 + y2) / 2.0,
            x=int(round(x1)),
            y=int(round(y1)),
            w=width,
            h=height,
            score=float(scores[best]),
        )
