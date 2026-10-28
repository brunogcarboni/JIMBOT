"""Grava frames com detecção no formato COCO usado pelo NanoDet."""

import json
from pathlib import Path

import cv2

import config
from detector import Detection


class DatasetWriter:
    def __init__(self) -> None:
        root = Path(config.DATASET_DIR)
        self.images_dir = root / "images"
        self.ann_path = root / "annotations" / "instances.json"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.ann_path.parent.mkdir(parents=True, exist_ok=True)
        self.cooldown = 0
        self.images: list[dict] = []
        self.annotations: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.ann_path.exists():
            return
        data = json.loads(self.ann_path.read_text(encoding="utf-8"))
        self.images = list(data.get("images", []))
        self.annotations = list(data.get("annotations", []))

    @property
    def count(self) -> int:
        return len(self.images)

    @property
    def done(self) -> bool:
        return self.count >= config.DATASET_TARGET

    def observe(self, frame, detection: Detection | None) -> bool:
        if self.done:
            return False
        if self.cooldown > 0:
            self.cooldown -= 1
            return False
        if detection is None:
            return False
        self._save(frame, detection)
        self.cooldown = config.DATASET_FRAME_INTERVAL
        return True

    def _save(self, frame, detection: Detection) -> None:
        image_id = self.count + 1
        file_name = f"{image_id:06d}.jpg"
        path = self.images_dir / file_name
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Não gravou {path}")

        height, width = frame.shape[:2]
        x, y, w, h = detection.x, detection.y, detection.w, detection.h
        self.images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": int(width),
                "height": int(height),
            }
        )
        self.annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": int(w * h),
                "iscrowd": 0,
            }
        )
        self._write_json()
        print(f"{self.count}/{config.DATASET_TARGET} {file_name}", flush=True)

    def _write_json(self) -> None:
        payload = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": [{"id": 1, "name": "trena", "supercategory": "objeto"}],
        }
        temporary = self.ann_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.ann_path)
