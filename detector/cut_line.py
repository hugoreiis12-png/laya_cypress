"""Finds the DANFE cut line (dashed, horizontal) and which side of it holds the canhoto."""
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

MIN_DASHES = 12
MIN_DASH_SPAN = 0.4  # fraction of the width the dashes must cover
MAX_DASH_WIDTH = 0.03
MAX_DASH_HEIGHT = 0.006
MIN_DASH_ASPECT = 1.5
ROW_BUCKET_PX = 4
LONG_RULE = 0.35  # full-width rules exist only on the canhoto rows; DANFE body rules are cut by cells
SIDE_MARGIN = 0.02
NARROW_RATIO = 0.6
ABOVE, BELOW = "acima", "abaixo"


@dataclass(frozen=True)
class CutLine:
    y: float  # fraction of the height
    rules: tuple[float, ...]  # solid full-width rules, fraction of the height


def _binary(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)


def _dash_rows(binary: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    height, width = binary.shape
    rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for x, y, w, h, _ in cv2.connectedComponentsWithStats(binary)[2][1:]:
        if 2 <= w <= width * MAX_DASH_WIDTH and h <= max(3, height * MAX_DASH_HEIGHT) and w >= MIN_DASH_ASPECT * h:
            rows[round((y + h / 2) / ROW_BUCKET_PX)].append((int(x), int(w)))
    return rows


def _dashed_row(binary: np.ndarray) -> int | None:
    width, rows, best = binary.shape[1], _dash_rows(binary), None
    for key in list(rows):
        dashes = rows.get(key - 1, []) + rows[key] + rows.get(key + 1, [])
        span = (max(x + w for x, w in dashes) - min(x for x, _ in dashes)) / width
        if len(dashes) >= MIN_DASHES and span >= MIN_DASH_SPAN and (best is None or len(dashes) > best[1]):
            best = (key * ROW_BUCKET_PX, len(dashes))
    return None if best is None else best[0]


def _rules(binary: np.ndarray) -> tuple[float, ...]:
    height, width = binary.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width * LONG_RULE), 1))
    solid = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return tuple(sorted(float(y + h / 2) / height for _, y, _, h, _ in cv2.connectedComponentsWithStats(solid)[2][1:]))


def find(gray: np.ndarray) -> CutLine | None:
    binary = _binary(gray)
    row = _dashed_row(binary)
    return None if row is None else CutLine(row / gray.shape[0], _rules(binary))


def canhoto_side(cut: CutLine) -> str | None:
    """The canhoto is the side with full-width rules; with rules on both sides, the narrow strip."""
    up = [cut.y - r for r in cut.rules if r < cut.y - SIDE_MARGIN]
    down = [r - cut.y for r in cut.rules if r > cut.y + SIDE_MARGIN]
    if not up or not down:
        return (ABOVE if up else BELOW) if up or down else None
    if max(up) < NARROW_RATIO * max(down):
        return ABOVE
    return BELOW if max(down) < NARROW_RATIO * max(up) else None


def on_canhoto_side(center: float, cut: CutLine, side: str | None) -> bool:
    if side is None:
        return True  # undecided layout: keep the box rather than guess
    return center < cut.y if side == ABOVE else center > cut.y
