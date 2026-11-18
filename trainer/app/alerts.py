"""Halt alert: plain text built by code from the failure record; no photo, text or customer value ever goes in it."""
from collections.abc import Mapping
from typing import Any

RESUME_COMMAND = ("docker run --rm -v /opt/laya-trainer/data:/srv/trainer laya-trainer:local "
                  "python -m trainer.app.control retomar")


def halt_message(failure: Mapping[str, Any]) -> tuple[str, str]:
    job = failure.get("job") or {}
    subject = f"Laya trainer parado: falha no treino de {failure['tipo']}"
    body = "\n".join([
        f"O treino de {failure['tipo']} falhou em {failure['em']} e o trainer foi parado.",
        f"Motivo: {failure['motivo']}",
        f"Dataset do job: {job.get('dataset', 'n/d')}. Kernel: {job.get('kernel', 'n/d')}.",
        f"Backup no volume do trainer: {failure['backup']}",
        "O container segue no ar e tenta de novo sozinho enquanto houver retomadas automáticas.",
        "Esgotadas, nada mais roda até a retomada manual. Depois de corrigir a causa, na VM .160:",
        RESUME_COMMAND,
    ])
    return subject, body
