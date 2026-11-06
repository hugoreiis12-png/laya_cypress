"""Send one or more photos to the service and print what the flow would see.

Usage: python scripts/benchmark/smoke.py foto1.jpg [foto2.jpg ...]
Env:   SERVICE_URL (default http://192.168.0.160:8020), SERVICE_API_KEY (required)
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://192.168.0.160:8020"
REQUEST_TIMEOUT_SECONDS = 60


def post(url: str, key: str, path: str) -> tuple[int, dict]:
    with open(path, "rb") as handle:
        image = base64.b64encode(handle.read()).decode()
    body = json.dumps({"imagem_base64": image, "contexto": {"id": os.path.basename(path)}}).encode()
    request = urllib.request.Request(
        f"{url}/v1/extrair-documento", data=body, method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, {"erro": exc.read().decode(errors="replace")}


def show(path: str, status: int, payload: dict, seconds: float) -> None:
    print(f"\n=== {path}  HTTP {status}  {seconds:.1f}s")
    if status != 200:
        print(payload)
        return
    print("output:", json.dumps(json.loads(payload["output"]), ensure_ascii=False, indent=2))
    print("_meta :", json.dumps(payload["_meta"], ensure_ascii=False, indent=2))


def main(paths: list[str]) -> int:
    key = os.environ.get("SERVICE_API_KEY")
    if not key or not paths:
        print(__doc__)
        return 2
    url = os.environ.get("SERVICE_URL", DEFAULT_URL)
    for path in paths:
        started = time.monotonic()
        status, payload = post(url, key, path)
        show(path, status, payload, time.monotonic() - started)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
