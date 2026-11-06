"""Teste do pipeline Laya com acompanhamento em tempo real via browser.

Subi o serviço em http://127.0.0.1:8020 (porta 160 = 0.140.160? não, é 192.168.0.160).
Este script:
  1. Sobe uma página de acompanhamento em http://localhost:8743 que mostra cada imagem
     sendo processada, com o resultado (tipo_documento, assinatura, campos, latência).
  2. Processa as imagens do gold set uma a uma, chamando o serviço real.
  3. Grava as predictions em gold/predictions/laya_benchmark.jsonl

Uso:
    python scripts/benchmark/teste_laya.py --serviço http://192.168.0.160:8020 --split dev
    python scripts/benchmark/teste_laya.py --serviço http://192.168.0.160:8020 --split test
    python scripts/benchmark/teste_laya.py --serviço http://192.168.0.160:8020 --split all --amostra 20

Env:
    SERVICE_API_KEY  (obrigatório)
"""
import argparse
import base64
import json
import mimetypes
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evalkit.manifest import labeled, load  # noqa: E402

ROOT = Path("gold")
MANIFEST = ROOT / "manifest.jsonl"
PREDICTIONS_OUT = ROOT / "predictions" / "laya_benchmark.jsonl"

PANEL_PORT = 8743
ALLOWED_HOSTS = {f"127.0.0.1:{PANEL_PORT}", f"localhost:{PANEL_PORT}"}
PREFLIGHT_IMAGE = "AAAA"  # base64 válido e mínimo: o serviço autentica antes de decodificar
DEFAULT_SEED = 0
PREFLIGHT_TIMEOUT = 30


class PanelServer:
    """Mini servidor que mantém uma lista de resultados em memória e serve HTML
    que faz polling a cada 2s para atualizar a visualização."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.results: list[dict] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def add(self, result: dict) -> None:
        with self._lock:
            self.results.append(result)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.results)

    def start(self) -> None:
        import http.server
        import json as _json

        panel = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.headers.get("Host") not in ALLOWED_HOSTS:  # bloqueia DNS rebinding
                    self.send_error(403)
                    return
                if self.path == "/api/status":
                    data = _json.dumps(panel.snapshot()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/api/total":
                    data = _json.dumps({"total": panel.total, "done": len(panel.results)}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.path = "/"
                    import pathlib
                    html_file = pathlib.Path(__file__).parent / "teste_laya_panel.html"
                    if html_file.exists():
                        body = html_file.read_bytes()
                    else:
                        body = _json.dumps({"error": "HTML não encontrado"}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(body)

        import http.server
        server = http.server.ThreadingHTTPServer(("127.0.0.1", PANEL_PORT), Handler)
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        print(f"Painel de acompanhamento: http://127.0.0.1:{PANEL_PORT}/")



def post_extrair(url: str, api_key: str, image_path: Path, doc_id: str, timeout: int = 120) -> tuple[int, dict]:
    """Chama POST /v1/extrair-documento no serviço."""
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    body = json.dumps({
        "imagem_base64": b64,
        "mime_type": mime,
        "contexto": {"id": doc_id, "origem": "evalkit"},
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/extrair-documento",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode(errors="replace"))
    except Exception as exc:
        return 0, {"error": str(exc)}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teste do pipeline Laya com acompanhamento em tempo real")
    parser.add_argument("--serviço", default="http://192.168.0.160:8020", help="URL do serviço de extração")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--amostra", type=int, default=0, help="amostra aleatória de N imagens (0 = todas)")
    parser.add_argument("--semente", type=int, default=DEFAULT_SEED, help="semente da amostra, para repetir a mesma")
    parser.add_argument("--sem-painel", action="store_true", help="não abre o painel de acompanhamento")
    return parser.parse_args(argv)


def select_gold(split: str, sample: int, seed: int) -> list[dict]:
    gold = labeled(load(MANIFEST), split)
    if 0 < sample < len(gold):
        gold = random.Random(seed).sample(gold, sample)
    return gold


def check_credentials(url: str, api_key: str) -> str | None:
    """Uma chamada mínima antes do lote: credencial errada aborta aqui, não depois de N imagens."""
    body = json.dumps({"imagem_base64": PREFLIGHT_IMAGE, "contexto": {"id": "preflight", "origem": "evalkit"}}).encode()
    req = urllib.request.Request(f"{url}/v1/extrair-documento", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=PREFLIGHT_TIMEOUT) as resp:
            return None if resp.status == 200 else f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}"
    except OSError as exc:
        return str(exc)


def start_panel(total: int) -> PanelServer:
    panel = PanelServer(total)
    panel.start()
    webbrowser.open(f"http://127.0.0.1:{PANEL_PORT}/")
    return panel


def panel_entry(rec: dict, status: int, latency_ms: float, pred: dict, error: str | None) -> dict:
    signature = pred.get("assinatura") or {}
    return {
        "id": rec["id"], "image": rec["image"], "status": status, "latency_ms": latency_ms,
        "tipo_documento": pred.get("tipo_documento"),
        "possui_assinatura": signature.get("possui_assinatura"),
        "confianca": signature.get("confianca"),
        "descricao": signature.get("descricao"),
        "cnpj": (pred.get("emitente") or {}).get("cnpj"),
        "numero": (pred.get("nota_fiscal") or {}).get("numero"),
        "chave": (pred.get("qrcode") or {}).get("chave"),
        "error": error,
        "gold_tipo": rec.get("tipo_documento"),
        "gold_assinatura": (rec.get("assinatura") or {}).get("possui_assinatura"),
    }


def predict_one(url: str, api_key: str, rec: dict) -> tuple[int, float, dict, str | None]:
    started = time.monotonic()
    status, payload = post_extrair(url, api_key, ROOT / rec["image"], rec["id"])
    latency_ms = round((time.monotonic() - started) * 1000, 1)
    if status != 200:
        return status, latency_ms, {}, f"HTTP {status}: {payload}"
    prediction = json.loads(payload.get("output", "{}"))
    prediction["_meta"] = payload.get("_meta", {})  # motor, confianças e métricas de tinta, para diagnóstico e calibração
    return status, latency_ms, prediction, None


def run_batch(gold: list[dict], url: str, api_key: str, panel: PanelServer | None) -> list[dict]:
    rows = []
    for idx, rec in enumerate(gold, 1):
        status, latency_ms, pred, error = predict_one(url, api_key, rec)
        rows.append({"id": rec["id"], "engine": "laya_benchmark", "prediction": pred,
                     "latency_ms": latency_ms, "error": error})
        print(f"[{idx:3d}/{len(gold)}] {rec['id'][:8]}  {status}  {latency_ms:>6.0f}ms  "
              f"tipo={pred.get('tipo_documento', '?')}  "
              f"assinatura={(pred.get('assinatura') or {}).get('possui_assinatura', '?')}")
        if panel:
            panel.add(panel_entry(rec, status, latency_ms, pred, error))
    return rows


def finish(rows: list[dict], split: str) -> int:
    valid = sum(1 for row in rows if not row["error"])
    if valid == 0:
        print("ERRO: nenhuma predição válida. Arquivo não gravado; veja os erros acima.")
        return 1
    PREDICTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"Predictions gravadas: {PREDICTIONS_OUT} ({valid}/{len(rows)} válidas)")
    if valid < len(rows):
        print(f"ATENÇÃO: {len(rows) - valid} com erro; elas contam como ausentes no score.")
    print("\nPróximo passo:")
    print(f"  python -m evalkit score --preds {PREDICTIONS_OUT} --name laya_benchmark --split {split}")
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    api_key = os.environ.get("SERVICE_API_KEY", "")
    if not api_key:
        print("ERRO: defina SERVICE_API_KEY antes de rodar.")
        print("  PowerShell: $env:SERVICE_API_KEY='sua-chave'")
        return 1
    problem = check_credentials(args.serviço, api_key)
    if problem:
        print(f"ERRO: o serviço recusou a verificação inicial ({problem}). Nada foi processado.")
        return 1
    gold = select_gold(args.split, args.amostra, args.semente)
    if not gold:
        print(f"ERRO: nenhum registro rotulado no split {args.split}.")
        return 1
    print(f"Imagens a processar: {len(gold)} (split: {args.split}, semente: {args.semente})")
    print(f"Serviço: {args.serviço}")
    panel = None if args.sem_painel else start_panel(len(gold))
    started = time.monotonic()
    rows = run_batch(gold, args.serviço, api_key, panel)
    total_s = time.monotonic() - started
    print(f"\nTotal: {len(gold)} imagens em {total_s:.1f}s ({total_s / len(gold) * 1000:.0f}ms/média)")
    return finish(rows, args.split)


if __name__ == "__main__":
    sys.exit(main())
