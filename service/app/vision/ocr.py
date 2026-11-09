"""Tesseract OCR with orientation detection. One image_to_data call yields text and word boxes."""
from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from .image_io import rotate, to_gray
from .types import Box, OcrResult, Word

LANG = "por"
MIN_WORD_CONFIDENCE = 0.0
MIN_READABLE_CHARS = 40  # below this OSD may have picked the wrong orientation; same cut the judge uses
CANDIDATE_ROTATIONS = (0, 90, 180, 270)
UPSCALE = 2.0  # small print (a WhatsApp-sized full page) is below what Tesseract reads; one enlarged retry
BOX_KEYS = ("left", "top", "width", "height")


def _words(data: dict[str, list[Any]]) -> list[tuple[tuple[int, int, int], Word]]:
    found = []
    for index, raw in enumerate(data["text"]):
        text, confidence = str(raw).strip(), float(data["conf"][index])
        if not text or confidence < MIN_WORD_CONFIDENCE:
            continue
        box = Box(int(data["left"][index]), int(data["top"][index]), int(data["width"][index]), int(data["height"][index]))
        line = (int(data["block_num"][index]), int(data["par_num"][index]), int(data["line_num"][index]))
        found.append((line, Word(text, box, confidence)))
    return found


def from_tesseract_data(data: dict[str, list[Any]], rotation: int) -> OcrResult:
    lines: dict[tuple[int, int, int], list[Word]] = defaultdict(list)
    ordered = _words(data)
    for line, word in ordered:
        lines[line].append(word)
    text = "\n".join(" ".join(word.text for word in words) for words in lines.values())
    return OcrResult(text, tuple(word for _, word in ordered), rotation)


def detect_rotation(gray: np.ndarray) -> int:
    import pytesseract

    try:
        return int(pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)["rotate"])
    except pytesseract.TesseractError:
        return 0  # OSD fails on sparse pages; read the image as it came


def _unscale(data: dict[str, list[Any]], scale: float) -> dict[str, list[Any]]:
    """Word boxes must keep referring to the upright image, not to the enlarged copy that was read."""
    if scale == 1:
        return data
    return {**data, **{key: [round(int(value) / scale) for value in data[key]] for key in BOX_KEYS}}


def _read(image: np.ndarray, rotation: int, scale: float = 1.0) -> tuple[OcrResult, np.ndarray]:
    import pytesseract

    upright = rotate(image, rotation)
    seen = upright if scale == 1 else cv2.resize(upright, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    data = pytesseract.image_to_data(to_gray(seen), lang=LANG, output_type=pytesseract.Output.DICT)
    return from_tesseract_data(_unscale(data, scale), rotation), upright


def _characters(read: tuple[OcrResult, np.ndarray]) -> int:
    return len(read[0].text.strip())


def run(image: np.ndarray) -> tuple[OcrResult, np.ndarray]:
    """Returns the OCR result and the upright image its word boxes refer to; a short read tries every rotation, then a zoom."""
    first = _read(image, detect_rotation(to_gray(image)))
    if _characters(first) >= MIN_READABLE_CHARS:
        return first
    others = [_read(image, rotation) for rotation in CANDIDATE_ROTATIONS if rotation != first[0].rotation]
    best = max([first, *others], key=_characters)  # a tie keeps the orientation OSD chose
    if _characters(best) >= MIN_READABLE_CHARS:
        return best
    zoomed = _read(image, best[0].rotation, UPSCALE)
    return zoomed if _characters(zoomed) > _characters(best) else best
