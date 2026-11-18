"""Container entrypoint: wires the real adapters and ticks the cycle until stopped or halted by a failure."""
import logging
import time
from collections.abc import Callable
from typing import Any

from . import config
from .alerts import RESUME_COMMAND
from .cycle import Cycle
from .dataset import InboxData
from .kaggle_client import KaggleClient
from .mailing import HermesWriter, Mailer, send_report
from .registry import Registry

log = logging.getLogger(__name__)
HEARTBEAT_FILE = "heartbeat"


def _notifier(settings: config.Settings) -> Callable[[dict[str, Any]], None] | None:
    """Without Hermes the report still goes out, as the plain text built by code."""
    if not settings.mail.enabled:
        return None
    writer = HermesWriter(settings.hermes_url, settings.hermes_model, settings.hermes_timeout) if settings.hermes_url else None
    mailer = Mailer(settings.mail)
    return lambda report: send_report(report, writer, mailer)


def _alerter(settings: config.Settings) -> Callable[[str, str], None] | None:
    return Mailer(settings.mail).send if settings.mail.enabled else None


def build_cycle(settings: config.Settings) -> Cycle:
    return Cycle(settings, KaggleClient(settings.kaggle_username), InboxData(settings),
                 Registry(settings.state_dir / "registro"), _notifier(settings), _alerter(settings))


def _announce_halt(cycle: Cycle, announced: str | None) -> str | None:
    """Logs a halt once, not on every tick; returns the backup of the halt it announced."""
    failure = cycle.halted()
    if failure is None:
        return None
    if failure.get("backup") != announced:
        log.error("trainer parado por falha em %s (%s); backup em %s. Para retomar: %s",
                  failure.get("tipo"), failure.get("motivo"), failure.get("backup"), RESUME_COMMAND)
    return failure.get("backup")


def main() -> None:
    """Never returns: a halted trainer stays up, keeps its heartbeat and waits for an automatic or manual resume."""
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = config.load()
    cycle = build_cycle(settings)
    heartbeat = settings.state_dir / HEARTBEAT_FILE
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    announced = None
    while True:
        try:
            cycle.resume_if_due()
            cycle.tick()
            announced = _announce_halt(cycle, announced)
        except Exception:  # the loop must survive one bad tick; the failure is logged with its traceback
            log.exception("falha no ciclo; tentando de novo no próximo")
        heartbeat.write_text(str(time.time()), encoding="utf-8")
        time.sleep(settings.tick_seconds)


if __name__ == "__main__":
    main()
