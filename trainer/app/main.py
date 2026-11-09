"""Container entrypoint: wires the real adapters and ticks the cycle until stopped."""
import logging
import time
from collections.abc import Mapping
from pathlib import Path

from . import config
from .cycle import Cycle, DataError
from .kaggle_client import KaggleClient
from .mailing import HermesWriter, Mailer, send_report
from .registry import Registry

log = logging.getLogger(__name__)
HEARTBEAT_FILE = "heartbeat"


class UnconfiguredData:
    """Placeholder until the photo source exists: reports no new samples, so the cycle stays idle."""

    def new_samples(self, trained: Mapping[str, int]) -> Mapping[str, int]:
        return {}

    def prepare(self, kind: str, directory: Path) -> int:
        raise DataError("a fonte de fotos ainda não está configurada")


def _notifier(settings: config.Settings):
    if not settings.mail.enabled or not settings.hermes_url:
        return None
    writer = HermesWriter(settings.hermes_url, settings.hermes_model, settings.hermes_timeout)
    mailer = Mailer(settings.mail)
    return lambda report: send_report(report, writer, mailer)


def build_cycle(settings: config.Settings) -> Cycle:
    return Cycle(settings, KaggleClient(settings.kaggle_username), UnconfiguredData(),
                 Registry(settings.state_dir / "registro"), _notifier(settings))


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = config.load()
    cycle = build_cycle(settings)
    heartbeat = settings.state_dir / HEARTBEAT_FILE
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            cycle.tick()
        except Exception:  # the loop must survive one bad tick; the failure is logged with its traceback
            log.exception("falha no ciclo; tentando de novo no próximo")
        heartbeat.write_text(str(time.time()), encoding="utf-8")
        time.sleep(settings.tick_seconds)


if __name__ == "__main__":
    main()
