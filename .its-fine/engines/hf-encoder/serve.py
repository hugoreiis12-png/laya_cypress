"""Servidor HTTP local do pacote: POST /v1/decide e GET /v1/package.

Mesmo preparador, calibrador e politica do predict.py. Escuta em loopback por
padrao; outro endereco exige allow_data_transfer com o destino listado.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import politica_privacidade  # noqa: E402
import infer  # noqa: E402
from itsfine.privacy import loopback, require_destination  # noqa: E402

CTX = {}
LOCK = threading.Lock()


def identidade():
    m = CTX['carga']['manifesto'] or {}
    return {'package_id': CTX['carga']['identidade']['package_id'], 'engine': 'hf-encoder', 'engine_version': m.get('engine_version'),
            'origem': m.get('origem'), 'task_hash': m.get('task_hash'), 'calibration_hash': m.get('calibration_hash'),
            'decide': CTX['decide'], 'questions': list(CTX['internas'])}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def _json(self, code, value):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if self.path == '/v1/package': self._json(200, identidade())
        else: self._json(404, {'erro': 'ROTA'})

    def do_POST(self):
        if self.path != '/v1/decide': return self._json(404, {'erro': 'ROTA'})
        try:
            tamanho = int(self.headers.get('Content-Length', '0'))
            if tamanho < 0 or tamanho > 16 * 1024 * 1024: raise ValueError('REQUEST_TOO_LARGE')
            corpo = json.loads(self.rfile.read(tamanho))
            estados = corpo['states'] if 'states' in corpo else [corpo['state']]
            if not isinstance(estados, list) or not estados or len(estados) > 256: raise ValueError('STATES_INVALID')
            with LOCK:
                regs = infer.registros(CTX, [{'state': s} for s in estados], CTX['batch'], com_rotulo=False)
            resultados = [infer.respostas(CTX, r) for r in regs]
            resposta = {'package_id': CTX['carga']['identidade']['package_id'], 'results': resultados}
            if 'state' in corpo: resposta.update(resultados[0])
            self._json(200, resposta)
        except Exception as exc:
            self._json(400, {'erro': type(exc).__name__, 'detalhe': str(exc)[:200]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pacote', required=True)
    parser.add_argument('--host', default='127.0.0.1'); parser.add_argument('--porta', type=int, default=8765)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    url = 'http://%s:%d' % (args.host, args.porta)
    if not loopback(url): require_destination(politica_privacidade(args.pacote), 'transfer', url)
    CTX.update(infer.preparar(args.pacote, args.device)); CTX['batch'] = args.batch
    servidor = ThreadingHTTPServer((args.host, args.porta), Handler)
    print(json.dumps({'servindo': url, **identidade()}, ensure_ascii=False), flush=True)
    try: servidor.serve_forever()
    finally: servidor.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
