"""The real training data source: derivatives cut from the inbox, labeled by the consensus of the decision log."""
import logging
from collections.abc import Mapping
from pathlib import Path

from . import collector, consensus, inbox, type_consensus
from .anonymizer import pseudonym
from .config import Settings
from .cycle import DataError
from .packager import StripItem, TextItem, write_text_package, write_vision_package
from .state import KIND_LAYA, KIND_VISION
from .text_source import TEXT_SUFFIX, text_cutter, text_path

log = logging.getLogger(__name__)
INBOX_DIR, STRIPS_DIR, TEXTS_DIR = "inbox", "tiras", "textos"
MIN_PER_CLASS = 2  # a class with fewer examples cannot be split and measured
MIN_CLASSES = 2


class InboxData:
    def __init__(self, settings: Settings, clock=None, read=None) -> None:
        self._settings = settings
        root, secret = settings.state_dir, settings.id_secret
        self._inbox, self._strips, self._texts = root / INBOX_DIR, root / STRIPS_DIR, root / TEXTS_DIR
        self._clock = {"clock": clock} if clock else {}
        self._cutters = [inbox.strip_cutter(self._strips, secret)] if KIND_VISION in settings.kinds else []
        if KIND_LAYA in settings.kinds:
            self._cutters.append(text_cutter(self._texts, secret, **({"read": read} if read else {})))

    def _ingest(self) -> tuple[tuple[consensus.Sample, ...], tuple[type_consensus.TypeSample, ...]]:
        path = self._settings.decisions_path
        if not path.is_file():
            return (), ()
        signatures, types = collector.load_samples(path).samples, collector.load_type_samples(path)
        known = {sample.item_id: sample.rotation for sample in (*signatures, *types)}
        report = inbox.ingest(self._inbox, known, self._cutters, **self._clock)
        pruned = inbox.prune(self._strips, **self._clock) + inbox.prune(self._texts, **self._clock)
        log.info("caixa de entrada: %d processadas, %d aguardando, %d descartadas, %d vencidas",
                 report.cut, report.waiting, report.dropped, pruned)
        return signatures, types

    def _ready(self) -> dict[str, list[tuple[str, str]]]:
        """(pseudonym, label) per kind, only where the agreed photo already has its derivative."""
        signatures, types = self._ingest()
        secret = self._settings.id_secret
        vision = [(pseudonym(secret, sample.item_id), label)
                  for sample, label in consensus.partition(list(signatures)).training
                  if inbox.strip_path(self._strips, secret, sample.item_id).is_file()]
        laya = [(pseudonym(secret, sample.item_id), label)
                for sample, label in type_consensus.training_set(list(types))
                if text_path(self._texts, secret, sample.item_id).is_file()]
        return {KIND_VISION: vision, KIND_LAYA: laya}

    def new_samples(self, trained: Mapping[str, int]) -> Mapping[str, int]:
        ready = self._ready()
        return {kind: max(0, len(ready[kind]) - trained.get(kind, 0)) for kind in self._settings.kinds}

    def prepare(self, kind: str, directory: Path) -> int:
        if kind not in self._settings.kinds:
            raise DataError(f"o tipo de treino {kind} não está habilitado")
        rows = self._ready()[kind]
        _require_classes(rows)
        name = directory.parent.name
        if kind == KIND_VISION:
            strips = [StripItem(item, label, (self._strips / f"{item}{inbox.STRIP_SUFFIX}").read_bytes()) for item, label in rows]
            write_vision_package(directory, name, strips, self._settings.test_percent)
        else:
            texts = [TextItem(item, label, (self._texts / f"{item}{TEXT_SUFFIX}").read_text(encoding="utf-8")) for item, label in rows]
            write_text_package(directory, name, texts, self._settings.test_percent, names_removed=True)
        return len(rows)


def _require_classes(rows: list[tuple[str, str]]) -> None:
    counts: dict[str, int] = {}
    for _, label in rows:
        counts[label] = counts.get(label, 0) + 1
    enough = [label for label, count in counts.items() if count >= MIN_PER_CLASS]
    if len(enough) < MIN_CLASSES:
        raise DataError(f"poucas classes com exemplos suficientes para treinar: {counts}")
