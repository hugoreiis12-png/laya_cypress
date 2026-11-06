"""evalkit CLI: add | validate | run | score."""
import argparse
import json
import sys
from pathlib import Path

from . import engines, manifest, report, runner, score

ROOT = Path("gold")
MANIFEST = ROOT / "manifest.jsonl"
REPORTS = Path("reports")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def cmd_add(args: argparse.Namespace) -> int:
    records = manifest.load(MANIFEST)
    sources = [p for path in args.paths for p in ([path] if path.is_file() else sorted(path.iterdir()))]
    added = [manifest.add_image(p, ROOT, records, args.source) for p in sources if p.suffix.lower() in IMAGE_SUFFIXES]
    manifest.save(MANIFEST, records)
    print(f"{sum(1 for a in added if a)} imagens novas, {sum(1 for a in added if not a)} duplicadas ignoradas")
    return 0


def cmd_validate(_: argparse.Namespace) -> int:
    records = manifest.load(MANIFEST)
    problems = {rec["id"]: manifest.validate(rec, ROOT) for rec in records}
    for record_id, errors in problems.items():
        if errors:
            print(f"{record_id}: " + "; ".join(errors))
    done = len(manifest.labeled(records))
    print(f"{len(records)} registros, {done} rotulados, {sum(1 for e in problems.values() if e)} com problema")
    return 1 if any(problems.values()) else 0


def cmd_run(args: argparse.Namespace) -> int:
    records = manifest.labeled(manifest.load(MANIFEST), args.split)
    engine = engines.build(args.engine, args.target)
    rows = runner.run(engine, records, ROOT, Path(args.out))
    print(f"{len(rows)} documentos, {sum(1 for r in rows if r['error'])} com erro -> {args.out}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    gold = manifest.labeled(manifest.load(MANIFEST), args.split)
    result = score.score(gold, runner.load_predictions(Path(args.preds)))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{args.name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    text = report.render(result, f"Avaliação — {args.name} ({args.split})")
    (REPORTS / f"{args.name}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalkit")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="copia imagens para o gold e cria registros sem rótulo")
    add.add_argument("paths", nargs="+", type=Path)
    add.add_argument("--source", default="whatsapp")
    sub.add_parser("validate", help="checa rótulos, imagens e hashes")
    run = sub.add_parser("run", help="roda um engine sobre o gold rotulado")
    run.add_argument("--engine", choices=engines.ENGINE_NAMES, required=True)
    run.add_argument("--target", help="URL do serviço, ou arquivo .jsonl no replay")
    run.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    run.add_argument("--out", required=True)
    score_cmd = sub.add_parser("score", help="pontua predições e gera reports/<name>.md")
    score_cmd.add_argument("--preds", required=True)
    score_cmd.add_argument("--name", required=True)
    score_cmd.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    return parser


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    handlers = {"add": cmd_add, "validate": cmd_validate, "run": cmd_run, "score": cmd_score}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
