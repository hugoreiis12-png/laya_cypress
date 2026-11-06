"""Tesseract OCR with orientation detection. One image_to_data call yields text and word boxes."""
from collections import defaultdict
from typing import Any

import numpy as np

from .image_io import rotate, to_gray
from .types import Box, OcrResult, Word

LANG = "por"
MIN_WORD_CONFIDENCE = 0.0


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


def run(image: np.ndarray) -> tuple[OcrResult, np.ndarray]:
    """Returns the OCR result and the upright image its word boxes refer to."""
    import pytesseract

    rotation = detect_rotation(to_gray(image))
    upright = rotate(image, rotation)
    data = pytesseract.image_to_data(to_gray(upright), lang=LANG, output_type=pytesseract.Output.DICT)
    return from_tesseract_data(data, rotation), upright
