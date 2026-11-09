"""OCR of the full photo with the service reader, anonymized before anything touches the disk."""
from pathlib import Path

from .allowlist import AllowlistRedactor
from .anonymizer import anonymize
from .inbox import Cutter, derived_path

TEXT_SUFFIX = ".txt"
MIN_TEXT_CHARS = 40  # the same minimum the service treats as a read photo


class TextError(ValueError):
    """The photo yielded too little text to teach the type model."""


def read_text(data: bytes) -> str:
    from service.app.vision import image_io, ocr  # imported late: OpenCV and Tesseract load only when the text kind is on

    result, _ = ocr.run(image_io.decode(data))
    return result.text


def text_path(texts: Path, secret: str, item_id: str) -> Path:
    return derived_path(texts, secret, item_id, TEXT_SUFFIX)


def text_cutter(texts: Path, secret: str, read=read_text) -> Cutter:
    redactor = AllowlistRedactor()

    def cut(data: bytes, rotation: int, item_id: str) -> None:
        text = read(data)
        if len(text.strip()) < MIN_TEXT_CHARS:
            raise TextError("texto curto demais")
        texts.mkdir(parents=True, exist_ok=True)
        text_path(texts, secret, item_id).write_text(anonymize(text, redactor).text, encoding="utf-8")
    return cut
