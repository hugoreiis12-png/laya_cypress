"""Tesseract OCR with orientation detection. One image_to_data call yields text and word boxes."""
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from .image_io import rotate, to_gray
from .types import Box, OcrResult, Word

LANG = "por"
MIN_WORD_CONFIDENCE = 0.0
MIN_READABLE_CHARS = 40  # below this OSD may have picked the wrong orientation; same cut the judge uses
CANDIDATE_ROTATIONS = (0, 90, 180, 270)
CONFIDENT_WORD = 70.0  # Tesseract confidence (0-100) from which a word counts as read, not guessed
MIN_WORD_LETTERS = 3
MIN_CONFIDENT_WORDS = 10  # upside-down text still yields many characters, but almost no confident words
UPSCALE = 2.0  # small print (a WhatsApp-sized full page) is below what Tesseract reads; one enlarged retry
BOX_KEYS = ("left", "top", "width", "height")
SMALL_PRINT_HEIGHT = 12  # median word box in px; a WhatsApp-sized full DANFE measures 9 and reads twice better zoomed
STRONG_CONFIDENT_WORDS = 2 * MIN_CONFIDENT_WORDS  # a zoomed read this good ends the search; a sideways one barely passes
ZOOM_BUDGET_SECONDS = 5.0  # the zoomed search must leave the judge room inside the 25 s review deadline
ZOOM_TURNS = (0, 90, 270, 180)  # from the best unzoomed orientation: itself, its perpendiculars, upside down last
FULL_TURN = 360
SPARSE_TEXT = "--psm 11"  # words without page layout: unlike psm 3 it never re-orients sideways text by itself
SPARSE_TIMEOUT_SECONDS = 1.5  # measured 17/11: a strip read takes 0.4 s, a woven background 20 s of noise


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


def _read(image: np.ndarray, rotation: int, scale: float = 1.0, config: str = "",
          timeout: float = 0) -> tuple[OcrResult, np.ndarray]:
    import pytesseract

    upright = rotate(image, rotation)
    seen = upright if scale == 1 else cv2.resize(upright, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    data = pytesseract.image_to_data(to_gray(seen), lang=LANG, config=config, timeout=timeout,
                                     output_type=pytesseract.Output.DICT)
    return from_tesseract_data(_unscale(data, scale), rotation), upright


def _characters(read: tuple[OcrResult, np.ndarray]) -> int:
    return len(read[0].text.strip())


def _confident_words(read: tuple[OcrResult, np.ndarray]) -> int:
    return sum(len(word.text) >= MIN_WORD_LETTERS and word.confidence >= CONFIDENT_WORD for word in read[0].words)


def _quality(read: tuple[OcrResult, np.ndarray]) -> tuple[int, int]:
    return _confident_words(read), _characters(read)


def _readable(read: tuple[OcrResult, np.ndarray]) -> bool:
    return _characters(read) >= MIN_READABLE_CHARS and _confident_words(read) >= MIN_CONFIDENT_WORDS


def _small_print(read: tuple[OcrResult, np.ndarray]) -> bool:
    heights = [word.box.height for word in read[0].words
               if len(word.text) >= MIN_WORD_LETTERS and word.confidence >= CONFIDENT_WORD and word.box.height > 0]
    return bool(heights) and statistics.median(heights) < SMALL_PRINT_HEIGHT


def _better(current: tuple[OcrResult, np.ndarray], candidate: tuple[OcrResult, np.ndarray]) -> tuple[OcrResult, np.ndarray]:
    return candidate if _quality(candidate) > _quality(current) else current


def _sharpened(image: np.ndarray, read: tuple[OcrResult, np.ndarray]) -> tuple[OcrResult, np.ndarray]:
    """A readable full page in small print still loses its headers and numbers; one zoom keeps the better read."""
    return _better(read, _read(image, read[0].rotation, UPSCALE)) if _small_print(read) else read


def _zoomed_search(image: np.ndarray, best: tuple[OcrResult, np.ndarray],
                   clock: Callable[[], float] = time.monotonic) -> tuple[OcrResult, np.ndarray]:
    """A narrow strip may only read once enlarged, and not in the orientation that won before the zoom."""
    deadline, origin = clock() + ZOOM_BUDGET_SECONDS, best[0].rotation
    for turn in ZOOM_TURNS:
        best = _better(best, _read(image, (origin + turn) % FULL_TURN, UPSCALE))
        if _confident_words(best) >= STRONG_CONFIDENT_WORDS or clock() >= deadline:
            break
    return best


def _sparse_vote(image: np.ndarray, preferred: int) -> tuple[OcrResult, np.ndarray] | None:
    """The sparse read with most confident words; None if one timed out: the right, text-rich turn is the slowest."""
    reads = []
    for rotation in CANDIDATE_ROTATIONS:
        try:
            reads.append(_read(image, rotation, config=SPARSE_TEXT, timeout=SPARSE_TIMEOUT_SECONDS))
        except RuntimeError:  # pytesseract reports its timeout, like any Tesseract failure, as a RuntimeError
            return None
    return max(reads, key=lambda read: (_confident_words(read), read[0].rotation == preferred))


def _orientation(image: np.ndarray, first: tuple[OcrResult, np.ndarray]) -> tuple[OcrResult, np.ndarray]:
    vote = _sparse_vote(image, first[0].rotation)
    if vote is None:
        others = [_read(image, rotation) for rotation in CANDIDATE_ROTATIONS if rotation != first[0].rotation]
        return max([first, *others], key=_quality)  # a tie keeps the orientation OSD chose
    turn = vote[0].rotation
    return _better(first if turn == first[0].rotation else _read(image, turn), vote)


def run(image: np.ndarray) -> tuple[OcrResult, np.ndarray]:
    """Returns the OCR result and the upright image its word boxes refer to.

    A short or low-confidence read lets sparse reads pick the orientation (psm 3 scores sideways text as readable
    too) or, if one timed out, the layout reads; then it tries the zoomed orientations.
    """
    first = _read(image, detect_rotation(to_gray(image)))
    if _readable(first):
        return _sharpened(image, first)
    best = _orientation(image, first)
    if _readable(best):
        return _sharpened(image, best)
    return _zoomed_search(image, best)
