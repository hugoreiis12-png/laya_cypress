"""Plain data types shared by the vision tools."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def expanded(self, margin: int, max_width: int, max_height: int) -> "Box":
        left, top = max(0, self.left - margin), max(0, self.top - margin)
        right, bottom = min(max_width, self.right + margin), min(max_height, self.bottom + margin)
        return Box(left, top, right - left, bottom - top)


@dataclass(frozen=True)
class Word:
    text: str
    box: Box
    confidence: float


@dataclass(frozen=True)
class OcrResult:
    text: str
    words: tuple[Word, ...]
    rotation: int


@dataclass(frozen=True)
class InkFeatures:
    region_found: bool
    density_region: float
    density_page: float
    density_relative: float
    ink_pixels: int
    component_count: int
    largest_component_ratio: float
    extent_ratio: float
