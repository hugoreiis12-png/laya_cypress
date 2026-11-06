"""Provisional signature rule over ink features. The score is monotone, not a calibrated probability."""
from .config import Settings
from .vision.types import InkFeatures

MAX_SCORE = 100.0


def decide(features: InkFeatures, settings: Settings) -> tuple[bool, float]:
    """Returns (signed, confidence 0-100) for a located region. Calibrate the If4 cutoff on the gold set, not by eye."""
    margin = min(
        features.ink_pixels / settings.min_ink_pixels,
        features.component_count / settings.min_components,
        features.density_relative / settings.min_relative_ink,
    )
    score = MAX_SCORE * margin / (margin + 1)
    return (True, score) if margin >= 1 else (False, MAX_SCORE - score)
