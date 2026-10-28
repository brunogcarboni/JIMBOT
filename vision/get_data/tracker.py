"""Histórico dos últimos pontos, suavização curta e lado da perda."""

from collections import deque
from dataclasses import dataclass

import config
from detector import Detection


@dataclass
class Sample:
    cx: float
    cy: float
    width_ratio: float


@dataclass
class TrackState:
    visible: bool
    lost: bool
    cx: float | None
    cy: float | None
    nx: float | None
    width_ratio: float | None
    bbox: tuple[int, int, int, int] | None
    exit_side: str | None
    history: deque
    score: float | None


class Tracker:
    def __init__(self) -> None:
        self.history: deque[Sample] = deque(maxlen=config.HISTORY)
        self.misses = 0
        self.lost = False
        self.exit_side: str | None = None
        self.bbox: tuple[int, int, int, int] | None = None
        self.score: float | None = None

    def last_center(self) -> tuple[float, float] | None:
        if not self.history:
            return None
        last = self.history[-1]
        return last.cx, last.cy

    def update(
        self, detection: Detection | None, frame_w: int, frame_h: int
    ) -> TrackState:
        if detection is None:
            self.misses += 1
            if self.history and self.misses < config.LOST_AFTER:
                return self._from_history(frame_w, visible=True, lost=False)
            self.lost = True
            if self.exit_side is None:
                self.exit_side = self._infer_exit(frame_w)
            return self._from_history(frame_w, visible=False, lost=True)

        self.misses = 0
        self.lost = False
        self.exit_side = None
        self.bbox = (detection.x, detection.y, detection.w, detection.h)
        self.score = detection.score
        self.history.append(
            Sample(
                cx=detection.cx,
                cy=detection.cy,
                width_ratio=detection.w / float(frame_w),
            )
        )
        return self._from_history(frame_w, visible=True, lost=False)

    def _smooth(self) -> Sample | None:
        if not self.history:
            return None
        window = list(self.history)[-config.SMOOTH_N :]
        n = float(len(window))
        return Sample(
            cx=sum(item.cx for item in window) / n,
            cy=sum(item.cy for item in window) / n,
            width_ratio=sum(item.width_ratio for item in window) / n,
        )

    def _from_history(self, frame_w: int, visible: bool, lost: bool) -> TrackState:
        smooth = self._smooth()
        if smooth is None:
            return TrackState(
                visible=False,
                lost=lost,
                cx=None,
                cy=None,
                nx=None,
                width_ratio=None,
                bbox=None,
                exit_side=self.exit_side,
                history=self.history,
                score=None,
            )
        half = frame_w / 2.0
        nx = (smooth.cx - half) / half if half else 0.0
        return TrackState(
            visible=visible,
            lost=lost,
            cx=smooth.cx,
            cy=smooth.cy,
            nx=nx,
            width_ratio=smooth.width_ratio,
            bbox=self.bbox if visible else None,
            exit_side=self.exit_side if lost else None,
            history=self.history,
            score=self.score if visible else None,
        )

    def _infer_exit(self, frame_w: int) -> str | None:
        if len(self.history) < 2:
            return None
        pts = list(self.history)
        last = pts[-1]
        earlier = pts[max(0, len(pts) - 8)]
        dx = last.cx - earlier.cx
        half = frame_w / 2.0
        nx = (last.cx - half) / half if half else 0.0
        if dx >= config.EXIT_DX_PX and nx > 0:
            return "direita"
        if dx <= -config.EXIT_DX_PX and nx < 0:
            return "esquerda"
        if nx >= config.EXIT_EDGE_NX:
            return "direita"
        if nx <= -config.EXIT_EDGE_NX:
            return "esquerda"
        if nx > 0.05:
            return "direita"
        if nx < -0.05:
            return "esquerda"
        return None
