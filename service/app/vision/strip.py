"""The receiver strip and the classifier input, shared by the trainer (which trains on strips) and the service."""
import cv2
import numpy as np

from .image_io import rotate

RECEIVER_TOP_FRACTION = 0.5  # the same half the judge reads; 0.22 cut the signature off photos with a margin
STRIP_JPEG_QUALITY = 90
CLASSIFIER_SIZE = 320  # the kernel's training size; train_vision.py keeps a copy of classifier_input
PIXEL_SCALE = 255.0


def upright_top_half(image: np.ndarray, rotation: int) -> np.ndarray:
    upright = rotate(image, rotation)
    return upright[: int(upright.shape[0] * RECEIVER_TOP_FRACTION)]


def classifier_input(image: np.ndarray) -> np.ndarray:
    """Shortest side to the training size, centre square, RGB in 0..1, NCHW: the ultralytics evaluation transform."""
    height, width = image.shape[:2]
    scale = CLASSIFIER_SIZE / min(height, width)
    size = (max(CLASSIFIER_SIZE, round(width * scale)), max(CLASSIFIER_SIZE, round(height * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    top, left = (resized.shape[0] - CLASSIFIER_SIZE) // 2, (resized.shape[1] - CLASSIFIER_SIZE) // 2
    square = resized[top:top + CLASSIFIER_SIZE, left:left + CLASSIFIER_SIZE]
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB).astype(np.float32) / PIXEL_SCALE
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
