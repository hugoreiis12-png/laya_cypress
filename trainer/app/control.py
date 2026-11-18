"""Operator commands over the trainer volume: status, versions, rollback and resume after a failure.

Needs no secret, so it also runs in a one-off container while the trainer itself is stopped.
"""
import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from . import scheduler
from .registry import SHADOW, Registry
from .state import KINDS, StateError, load_state, save_state

DEFAULT_STATE_DIR = "/srv/trainer"
SHORT_HASH = 12


def _state_dir() -> Path:
    return Path(os.environ.get("TRAINER_STATE_DIR", DEFAULT_STATE_DIR))


def status(root: Path) -> list[str]:
    state = load_state(root / "estado.json")
    lines = [f"fase: {state.phase}", f"próximo tipo: {state.next_kind}"]
    if state.running:
        lines.append(f"em andamento: {state.running.get('kind')} ({state.running.get('kernel')})")
    if state.halted:
        lines.append(f"parado por falha: {state.halted.get('motivo')} | backup: {state.halted.get('backup')}")
    if state.resume_from:
        lines.append(f"retomada pendente a partir de: {state.resume_from}")
    return lines


def versions(root: Path, kind: str | None) -> list[str]:
    registry = Registry(root / "registro")
    lines = []
    for name, record in registry.versions().items():
        if kind and record["kind"] != kind:
            continue
        current = " <- atual" if registry.current(record["kind"]) == name else ""
        error = record["metrics"].get("error")
        lines.append(f"{name}  {record['status']:<10}  erro={error}  sha256={record['sha256'][:SHORT_HASH]}{current}")
    return lines or ["nenhuma versão registrada"]


def rollback(root: Path, kind: str) -> list[str]:
    registry = Registry(root / "registro")
    if not any(record["kind"] == kind and record["status"] == SHADOW for record in registry.versions().values()):
        return [f"nenhuma versão de {kind} em sombra para reverter"]
    current = registry.rollback(kind)
    return [f"última versão em sombra de {kind} revertida; atual agora: {current or 'nenhuma'}"]


def resume(root: Path) -> list[str]:
    path = root / "estado.json"
    resumed = scheduler.resume(load_state(path))
    save_state(path, resumed)
    origin = resumed.resume_from or "um pacote novo"
    return [f"retomada liberada; o próximo job de {resumed.next_kind} usa {origin}",
            "o trainer no ar segue no próximo tick (se o container estiver parado: docker start laya-trainer)"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m trainer.app.control", description="Operação do laya-trainer")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="fase, job em andamento e parada por falha")
    listing = commands.add_parser("versoes", help="versões registradas e a atual por tipo")
    listing.add_argument("tipo", nargs="?", choices=KINDS)
    commands.add_parser("rollback", help="reverte a última versão em sombra").add_argument("tipo", choices=KINDS)
    commands.add_parser("retomar", help="libera o trainer parado por falha")
    return parser


def run(argv: Sequence[str], root: Path) -> list[str]:
    args = _parser().parse_args(argv)
    if args.command == "status":
        return status(root)
    if args.command == "versoes":
        return versions(root, args.tipo)
    if args.command == "rollback":
        return rollback(root, args.tipo)
    return resume(root)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        lines = run(sys.argv[1:] if argv is None else argv, _state_dir())
    except StateError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
