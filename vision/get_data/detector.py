"""Proposta por cor e confirmação por forma e bordas internas."""

from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class Detection:
    cx: float
    cy: float
    x: int
    y: int
    w: int
    h: int
    score: float
    circularity: float
    grid_density: float
    bar_ratio: float
    frequency: float


@dataclass
class DetectionFrame:
    detection: Detection | None
    mask: np.ndarray
    edges: np.ndarray
    spatial: np.ndarray


def _proposal_mask(frame: np.ndarray) -> np.ndarray:
    """Prior cromático. Não é a segmentação final."""
    b, g, r = cv2.split(frame)
    red = r.astype(np.float32)
    excess = (red - np.maximum(g, b).astype(np.float32)) / (red + 1.0)
    hue = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 0]
    red_hue = (hue <= config.HUE_MAX) | (hue >= config.HUE_MIN_WRAP)
    mask = ((excess >= config.EXCESS_RED_MIN) & red_hue).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.MORPH_KSIZE, config.MORPH_KSIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _shape_ok(contour: np.ndarray, frame_area: float) -> tuple[bool, float, float, float]:
    area = cv2.contourArea(contour)
    min_area = max(config.MIN_AREA_PX, config.MIN_AREA_RATIO * frame_area)
    if area < min_area:
        return False, 0.0, 0.0, 0.0

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 1.0:
        return False, 0.0, 0.0, 0.0
    circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
    if not (config.CIRCULARITY_MIN <= circularity <= config.CIRCULARITY_MAX):
        return False, circularity, 0.0, 0.0

    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    short = min(rw, rh)
    long = max(rw, rh)
    if short < 1.0 or long / short > config.ASPECT_MAX:
        return False, circularity, 0.0, 0.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 1.0:
        return False, circularity, 0.0, 0.0
    solidity = float(area / hull_area)
    if solidity < config.SOLIDITY_MIN:
        return False, circularity, 0.0, solidity
    return True, circularity, solidity, area


def _internal_structure(
    gray: np.ndarray, contour: np.ndarray
) -> tuple[float, float, tuple[int, int, int, int]] | None:
    x, y, w, h = cv2.boundingRect(contour)
    if w < 8 or h < 8:
        return None

    roi_gray = gray[y : y + h, x : x + w]
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(roi_mask, [shifted], -1, 255, thickness=-1)

    erode_px = max(3, int(min(w, h) * config.INNER_ERODE_RATIO))
    if erode_px % 2 == 0:
        erode_px += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
    inner = cv2.erode(roi_mask, kernel)
    inner_area = int(cv2.countNonZero(inner))
    if inner_area < 20:
        return None

    edges = cv2.Canny(roi_gray, config.CANNY_LOW, config.CANNY_HIGH)
    edges = cv2.bitwise_and(edges, inner)
    grid_density = float(cv2.countNonZero(edges) / inner_area)

    # A barra é um sulco escuro vertical. Black-hat com kernel alto
    # acende esse sulco e apaga a borda circular da carcaça.
    kw = _odd(int(w * config.BAR_KERNEL_WIDTH_RATIO), 3)
    kh = _odd(int(h * config.BAR_KERNEL_HEIGHT_RATIO), 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    blackhat = cv2.morphologyEx(roi_gray, cv2.MORPH_BLACKHAT, kernel)

    inner_bool = inner > 0
    band = max(1, int(w * config.CENTER_BAND_RATIO))
    mid = w // 2
    cols = np.zeros(w, dtype=bool)
    cols[max(0, mid - band) : min(w, mid + band)] = True
    center_sel = inner_bool & cols[None, :]
    side_sel = inner_bool & ~cols[None, :]
    if not np.any(center_sel) or not np.any(side_sel):
        return None
    center_mean = float(blackhat[center_sel].mean())
    side_mean = float(blackhat[side_sel].mean())
    bar_ratio = center_mean / (side_mean + 1e-6)
    frequency = _band_energy(roi_gray, inner)
    return grid_density, bar_ratio, frequency, (x, y, w, h)


def _band_energy(roi_gray: np.ndarray, inner: np.ndarray) -> float:
    """Energia da banda média: tira iluminação lenta e o ruído fino.

    A grade da trena, quando cabe em pixels, cai nessa banda.
    Uma superfície lisa fica perto de zero.
    """
    fine = cv2.GaussianBlur(roi_gray, (0, 0), 1.0)
    coarse = cv2.GaussianBlur(roi_gray, (0, 0), 3.0)
    band = cv2.absdiff(fine, coarse)
    selected = band[inner > 0]
    if selected.size < 20:
        return 0.0
    return float(selected.mean())


def _score(
    circularity: float,
    solidity: float,
    grid_density: float,
    bar_ratio: float,
    frequency: float,
    cx: float,
    cy: float,
    last_center: tuple[float, float] | None,
    diagonal: float,
) -> float:
    grid_term = min(grid_density / config.GRID_DENSITY_REF, 1.0)
    bar_term = min(max(bar_ratio, 0.0) / config.BAR_RATIO_REF, 1.0)
    freq_term = min(frequency / config.FREQ_REF, 1.0)
    value = (
        config.WEIGHT_CIRCULARITY * circularity
        + config.WEIGHT_GRID * grid_term
        + config.WEIGHT_BAR * bar_term
        + config.WEIGHT_SOLIDITY * solidity
        + config.WEIGHT_FREQ * freq_term
    )
    if last_center is not None and diagonal > 1.0:
        dist = float(np.hypot(cx - last_center[0], cy - last_center[1]))
        value -= config.PROXIMITY_PENALTY * (dist / diagonal)
    return value


def _odd(value: int, minimum: int) -> int:
    value = max(minimum, value)
    if value % 2 == 0:
        value += 1
    return value


def detect(
    frame: np.ndarray, last_center: tuple[float, float] | None = None
) -> DetectionFrame:
    blurred = cv2.GaussianBlur(
        frame, (config.BLUR_KSIZE, config.BLUR_KSIZE), 0
    )
    detail = cv2.GaussianBlur(
        frame, (config.DETAIL_BLUR_KSIZE, config.DETAIL_BLUR_KSIZE), 0
    )
    gray = cv2.cvtColor(detail, cv2.COLOR_BGR2GRAY)
    mask = _proposal_mask(blurred)

    edges = cv2.Canny(gray, config.CANNY_LOW, config.CANNY_HIGH)
    edges = cv2.bitwise_and(edges, mask)
    spatial = np.zeros(mask.shape, dtype=np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_h, frame_w = frame.shape[:2]
    frame_area = float(frame_h * frame_w)
    diagonal = float(np.hypot(frame_w, frame_h))

    best: Detection | None = None
    for contour in contours:
        ok, circularity, solidity, _area = _shape_ok(contour, frame_area)
        if not ok:
            continue
        structure = _internal_structure(gray, contour)
        if structure is None:
            continue
        grid_density, bar_ratio, frequency, (x, y, w, h) = structure
        if frequency < config.FREQ_MIN:
            continue

        cv2.drawContours(spatial, [contour], -1, 160, thickness=-1)
        moments = cv2.moments(contour)
        if moments["m00"] <= 1.0:
            continue
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        score = _score(
            circularity,
            solidity,
            grid_density,
            bar_ratio,
            frequency,
            cx,
            cy,
            last_center,
            diagonal,
        )
        candidate = Detection(
            cx=cx,
            cy=cy,
            x=x,
            y=y,
            w=w,
            h=h,
            score=score,
            circularity=circularity,
            grid_density=grid_density,
            bar_ratio=bar_ratio,
            frequency=frequency,
        )
        if best is None or candidate.score > best.score:
            best = candidate

    if best is not None:
        cv2.rectangle(
            spatial,
            (best.x, best.y),
            (best.x + best.w, best.y + best.h),
            255,
            1,
        )
    return DetectionFrame(detection=best, mask=mask, edges=edges, spatial=spatial)
