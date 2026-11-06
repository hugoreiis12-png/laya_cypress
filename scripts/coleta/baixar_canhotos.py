"""Baixa as imagens dos manifestos de canhotos (assets/) para o gold set.

Uso:
    python scripts/coleta/baixar_canhotos.py [--assinados N] [--nao-assinados N]

Lê assets/canhotos_manifest.jsonl (assinados) e
assets/canhotos_nao_assinados_manifest.jsonl (preliminares, pré-assinatura),
baixa para gold/raw/<classe>/, dedupe por sha256 e registra no
gold/manifest.jsonl via `python -m evalkit add`. PDFs ficam no staging
(evalkit add ignora .pdf por design).
"""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
GOLD = ROOT / "gold"
STAGING = {
    "assinado": GOLD / "raw" / "canhotos_assinados",
    "nao_assinado": GOLD / "raw" / "canhotos_nao_assinados",
}
MANIFESTOS = {
    "assinado": ASSETS / "canhotos_manifest.jsonl",
    "nao_assinado": ASSETS / "canhotos_nao_assinados_manifest.jsonl",
}
OK_TYPES = ("image/", "application/pdf")
TIMEOUT = 45
RETRIES = 2
WORKERS = 8


def norm_url(raw: str) -> str:
    url = str(raw or "").strip()
    if url.startswith("//"):
        return "https:" + url
    return url


def load_manifest(path: Path, limite: int) -> list[dict]:
    regs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            regs.append(json.loads(line))
    regs.sort(key=lambda r: str(r.get("data") or r.get("id") or ""), reverse=True)
    return regs[:limite] if limite else regs


def baixar(classe: str, reg: dict, destino: Path) -> dict:
    url = norm_url(reg.get("url") or "")
    nome = (url.split("/")[-1] or "canhoto.jpg") if url else "sem_url.jpg"
    alvo = destino / f"{reg.get('id', 'semid')}_{nome}"
    if alvo.exists() and alvo.stat().st_size > 0:
        return {"classe": classe, "reg": reg, "alvo": alvo, "status": "cache"}
    if not url.startswith("http"):
        return {"classe": classe, "reg": reg, "alvo": None, "status": "url_invalida"}
    for tentativa in range(RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "laya-gold/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                tipo = (resp.headers.get("Content-Type") or "").lower()
                dados = resp.read()
            if not tipo.startswith(OK_TYPES):
                return {"classe": classe, "reg": reg, "alvo": None,
                        "status": f"tipo_recusado:{tipo or 'desconhecido'}"}
            if not dados:
                return {"classe": classe, "reg": reg, "alvo": None, "status": "vazio"}
            break
        except Exception as erro:  # noqa: BLE001
            if tentativa == RETRIES:
                return {"classe": classe, "reg": reg, "alvo": None,
                        "status": f"erro:{str(erro)[:90]}"}
    destino.mkdir(parents=True, exist_ok=True)
    alvo.write_bytes(dados)
    return {"classe": classe, "reg": reg, "alvo": alvo, "status": "ok"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assinados", type=int, default=0, help="limite de assinados (0 = todos do manifest)")
    parser.add_argument("--nao-assinados", type=int, default=0, help="limite de preliminares (0 = todos)")
    args = parser.parse_args()

    plan = []
    for classe, caminho in MANIFESTOS.items():
        if not caminho.exists():
            print(f"[!] manifest ausente: {caminho}")
            continue
        limite = args.assinados if classe == "assinado" else args.nao_assinados
        regs = load_manifest(caminho, limite)
        print(f"{classe}: {len(regs)} URLs a baixar")
        plan.extend((classe, reg) for reg in regs)

    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futuros = {pool.submit(baixar, classe, reg, STAGING[classe]): (classe, reg)
                   for classe, reg in plan}
        for futuro in concurrent.futures.as_completed(futuros):
            jobs.append(futuro.result())

    resumo: dict[str, int] = {}
    falhas = []
    for item in jobs:
        chave = item["status"].split(":")[0]
        resumo[chave] = resumo.get(chave, 0) + 1
        if item["alvo"] is None:
            falhas.append({"classe": item["classe"], "id": item["reg"].get("id"),
                           "status": item["status"], "url": item["reg"].get("url")})

    # dedupe por sha256 do conteúdo entre E dentro das duas classes
    vistos: set[str] = set()
    classe_por_sha: dict[str, str] = {}
    duplicados = 0
    for classe in ("assinado", "nao_assinado"):
        pasta = STAGING[classe]
        if not pasta.exists():
            continue
        for arq in sorted(pasta.iterdir()):
            sha = hashlib.sha256(arq.read_bytes()).hexdigest()
            if sha in vistos:
                arq.unlink()
                duplicados += 1
            else:
                vistos.add(sha)
                classe_por_sha.setdefault(sha, classe)

    for classe in ("assinado", "nao_assinado"):
        pasta = STAGING[classe]
        if pasta.exists() and any(pasta.iterdir()):
            processo = subprocess.run(
                [sys.executable, "-m", "evalkit", "add", str(pasta), "--source", "whatsapp"],
                cwd=ROOT, capture_output=True, text=True)
            print(f"evalkit add {classe}: rc={processo.returncode} | "
                  f"{(processo.stdout + processo.stderr).strip()[:300]}")

    # anota proveniência no manifest do gold (id do evalkit = sha256[:16])
    mpath = GOLD / "manifest.jsonl"
    records = [json.loads(l) for l in mpath.read_text(encoding="utf-8").splitlines() if l.strip()]
    anotados = 0
    for rec in records:
        classe = classe_por_sha.get(rec.get("sha256", ""))
        if not classe or "candidato:" in rec.get("notes", ""):
            continue
        rec["notes"] = (f"candidato:canhoto_{classe} " + (rec.get("notes") or ""))[:200]
        anotados += 1
    mpath.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                     encoding="utf-8")
    (ASSETS / "baixadas_falhas.json").write_text(
        json.dumps(falhas, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"RESUMO status={resumo}")
    print(f"RESUMO duplicados={duplicados} notados={anotados} falhas={len(falhas)} "
          f"(detalhe: assets/baixadas_falhas.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
