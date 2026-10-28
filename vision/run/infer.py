"""Carrega o NanoDet e devolve a caixa da trena com maior score."""

import os
import sys
from pathlib import Path

import numpy as np

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
