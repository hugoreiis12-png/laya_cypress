"""Ink features of the receiver region: handwriting vs blank, ignoring printed text and form lines."""
import cv2
import numpy as np

from .types import Box, InkFeatures, Word

THRESHOLD_BLOCK = 35
THRESHOLD_OFFSET = 15
MIN_COMPONENT_AREA = 12
LINE_KERNEL_DIVISOR = 20
MIN_LINE_KERNEL = 25
PRINTED_WORD_CONFIDENCE = 60.0  # confidently read words are printed text, not handwriting
MAX_RELATIVE = 1000.0
EPSILON = 1e-6


def ink_mask(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                 THRESHOLD_BLOCK, THRESHOLD_OFFSET)


def remove_form_lines(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (max(MIN_LINE_KERNEL, width // LINE_KERNEL_DIVISOR), 1))
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(MIN_LINE_KERNEL, height // LINE_KERNEL_DIVISOR)))
    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal) | cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical)
    return mask & ~cv2.dilate(lines, np.ones((3, 3), np.uint8))


def mask_printed_words(mask: np.ndarray, words: tuple[Word, ...]) -> np.ndarray:
    cleaned = mask.copy()
    for word in words:
        if word.confidence >= PRINTED_WORD_CONFIDENCE:
            box = word.box
            cleaned[box.top:box.bottom, box.left:box.right] = 0
    return cleaned


def drop_specks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    areas = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            keep[labels == index] = 255
            areas.append(int(stats[index, cv2.CC_STAT_AREA]))
    return keep, np.array(areas, dtype=np.int64)


def _extent(mask: np.ndarray) -> float:
    points = cv2.findNonZero(mask)
    if points is None:
        return 0.0
    _, _, width, height = cv2.boundingRect(points)
    return width * height / mask.size


def clean_mask(gray: np.ndarray, words: tuple[Word, ...]) -> tuple[np.ndarray, np.ndarray]:
    mask = mask_printed_words(remove_form_lines(ink_mask(gray)), words)
    return drop_specks(mask)


def measure(gray: np.ndarray, words: tuple[Word, ...], region: Box | None) -> InkFeatures:
    mask, _ = clean_mask(gray, words)
    page_density = float(np.count_nonzero(mask)) / mask.size
    if region is None:
        return InkFeatures(False, 0.0, page_density, 0.0, 0, 0, 0.0, 0.0)
    crop = mask[region.top:region.bottom, region.left:region.right]
    crop_clean, areas = drop_specks(crop)
    ink = int(np.count_nonzero(crop_clean))
    density = ink / max(crop.size, 1)
    return InkFeatures(
        region_found=True, density_region=density, density_page=page_density,
        density_relative=min(MAX_RELATIVE, density / max(page_density, EPSILON)),
        ink_pixels=ink, component_count=len(areas),
        largest_component_ratio=float(areas.max()) / ink if ink else 0.0, extent_ratio=_extent(crop_clean),
    )
