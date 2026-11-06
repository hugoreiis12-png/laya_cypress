"""Locates the 'IDENTIFICAÇÃO E ASSINATURA DO RECEBEDOR' label and the window around it."""
import unicodedata

from .types import Box, Word

LABEL_WORDS = {"identificacao", "assinatura", "recebedor"}
MIN_LABEL_WORDS = 2
SIGNATURE_WINDOW_PX = 300  # the old contract: signature within 300 px of the label
SAME_LINE_FACTOR = 1.5


def _plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if char.isalpha() and not unicodedata.combining(char)).casefold()


def _same_line(first: Word, second: Word) -> bool:
    limit = SAME_LINE_FACTOR * max(first.box.height, second.box.height)
    return abs(first.box.top - second.box.top) <= limit


def _union(words: list[Word]) -> Box:
    left, top = min(w.box.left for w in words), min(w.box.top for w in words)
    right, bottom = max(w.box.right for w in words), max(w.box.bottom for w in words)
    return Box(left, top, right - left, bottom - top)


def _label_groups(words: list[Word]) -> list[list[Word]]:
    hits = [word for word in words if _plain(word.text) in LABEL_WORDS]
    groups: list[list[Word]] = []
    for word in hits:
        group = next((g for g in groups if _same_line(g[0], word)), None)
        if group is None:
            groups.append([word])
        else:
            group.append(word)
    return groups


def locate(words: tuple[Word, ...], image_width: int, image_height: int) -> Box | None:
    groups = [g for g in _label_groups(list(words)) if len({_plain(w.text) for w in g}) >= MIN_LABEL_WORDS]
    if not groups:
        return None
    best = max(groups, key=len)
    return _union(best).expanded(SIGNATURE_WINDOW_PX, image_width, image_height)
