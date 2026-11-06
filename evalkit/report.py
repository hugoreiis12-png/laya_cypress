"""Renders the score dictionary as a markdown report (pt-BR)."""
from typing import Any


def _pct(value: float | None) -> str:
    return "n/d" if value is None else f"{value * 100:.1f}%"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def _fields_section(result: dict[str, Any]) -> list[str]:
    rows = [[name, _pct(s["accuracy"]), str(s.get("correct", 0)), str(s.get("wrong", 0)),
             str(s.get("missing", 0)), str(s.get("spurious", 0))] for name, s in result["fields"].items()]
    return ["## Campos extraídos", *_table(["campo", "acerto", "certo", "errado", "ausente", "inventado"], rows), ""]


def _signature_section(result: dict[str, Any]) -> list[str]:
    sig = result["assinatura"]
    head = f"n={sig['n']} · precisão {_pct(sig['precision'])} · recall {_pct(sig['recall'])} · falsos aceites {sig['false_accepts']}"
    rows = [[str(s["threshold"]), str(s["accepted"]), str(s["false_accepts"]), _pct(s["precision"]), _pct(s["recall"])]
            for s in sig["sweep"]]
    sweep = _table(["limiar", "aceitos", "falsos aceites", "precisão", "recall"], rows)
    return ["## Assinatura (canhotos)", head, "", "Varredura do limiar do If4:", *sweep, ""]


def _calibration_section(result: dict[str, Any]) -> list[str]:
    lines = ["## Calibração de tipo_documento (qual campo de confiança usar)"]
    for name, cal in result["tipo_calibration"].items():
        rows = [[b["bin"], str(b["n"]), _pct(b["accuracy"]), _pct(b["mean_conf"])] for b in cal["bins"]]
        lines += [f"### {name} — ECE {_pct(cal['ece'])}", *_table(["faixa", "n", "acerto real", "confiança média"], rows), ""]
    return lines


def render(result: dict[str, Any], title: str) -> str:
    doc = result["tipo_documento"]
    rows = [[k, _pct(v["precision"]), _pct(v["recall"]), str(v["support"])] for k, v in doc["per_class"].items()]
    lat = result["latency_ms"]
    lines = [f"# {title}", "",
             f"n={result['n']} · erros de predição {result['prediction_errors']} · latência p50 {lat['p50']} ms / p95 {lat['p95']} ms", "",
             f"## tipo_documento — acerto {_pct(doc['accuracy'])}",
             *_table(["classe", "precisão", "recall", "suporte"], rows), "",
             "Confusão: " + ", ".join(f"{k}={v}" for k, v in doc["confusion"].items()), ""]
    lines += _fields_section(result) + _signature_section(result) + _calibration_section(result)
    items = result["itens"]
    lines += ["## Itens do pedido", f"precisão {_pct(items['precision'])} · recall {_pct(items['recall'])} · F1 {_pct(items['f1'])}"]
    return "\n".join(lines) + "\n"
