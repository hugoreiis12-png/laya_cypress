"""Rotulador dinâmico: serve HTML local que carrega as imagens de gold/images/ direto do disco.

Não precisa de upload — as imagens já estão em gold/images/.

Uso:
    python scripts/rotulagem/rotulador_dinamico.py --servir          # abre o rotulador no navegador (porta 8742)
    python scripts/rotulagem/rotulador_dinamico.py --servir --fila-revisao <scores.jsonl>   # quarentena, menor score primeiro
    python scripts/rotulagem/rotulador_dinamico.py --importar gold/triagem/rotulado.json   # importa rótulos no manifest
    python scripts/rotulagem/rotulador_dinamico.py --importar gold/triagem/rotulado.json --dry-run   # só valida, não grava

Fluxo:
1. python scripts/rotulagem/rotulador_dinamico.py --servir
2. Navegue pelas imagens com as setas ou clique nas miniaturas.
3. Atalhos: [1] ou [↑] = tem assinatura · [0] ou [↓] = sem assinatura · [2] ou [→] = ignorar · [←] = voltar
4. Ao terminar (ou a qualquer momento) clique em "Exportar rótulos" — baixa rotulado.json.
5. python scripts/rotulagem/rotulador_dinamico.py --importar gold/triagem/rotulado.json
"""
import argparse
import json
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evalkit.manifest import DOC_TYPES, STATUS_LABELED, STATUS_UNLABELED  # noqa: E402
from scripts.rotulagem.importar_triagem import write_with_backup  # noqa: E402

GOLD = Path("gold")
MANIFEST = GOLD / "manifest.jsonl"
TRIAGEM = GOLD / "triagem"
HTML_OUT = TRIAGEM / "rotulador.html"

PORT = 8742
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}
IMAGE_PATH = re.compile(r"/images/[0-9a-f]{16}\.(?:jpg|jpeg|png|webp)")
ACTIONS = {"ok": True, "no": False}
ACTION_SKIP = "skip"
DEFAULT_TIPO = "canhoto"  # a interface só rotula canhotos; outro tipo precisa vir explícito
FONTE = "rotulador"
FONTE_PROTEGIDA = "triagem_manual"

REVIEW_STATUS = "pendente_revisao"

NOTES_CANDIDATOS = {
    "candidato:canhoto_assinado": "assinado",
    "candidato:canhoto_nao_assinado": "nao_assinado",
}


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in records:
        note = (r.get("notes") or "").strip()
        if note not in NOTES_CANDIDATOS:
            continue
        out.append({
            "id": r["id"],
            "image": r["image"],
            "group": NOTES_CANDIDATOS[note],
            "status": r.get("status", STATUS_UNLABELED),
        })
    return out


def build_review_queue(records: list[dict[str, Any]], scores_path: Path) -> list[dict[str, Any]]:
    scores = {}
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            scores[row["sha"]] = row["score"]
    queue = sorted((r for r in records if r.get("status") == REVIEW_STATUS), key=lambda r: scores.get(r["sha256"], 0.0))
    return [{"id": r["id"], "image": r["image"], "group": "nao_assinado", "status": r["status"]} for r in queue]


HTML_TEMPLATE = r"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rotulador · Canhoto</title>
<style>
:root{--bg:#0f1117;--card:#1a1d27;--border:#2a2d3a;--text:#e0e3eb;--dim:#7a7f95;--ok:#34d399;--no:#f87171;--skip:#9ca3af;--accent:#60a5fa;--btn-bg:#2a2d3a}
*{box-sizing:border-box}
body{font-family:system-ui,sans-serif;margin:0;background:var(--bg);color:var(--text)}
header{position:sticky;top:0;z-index:100;background:var(--bg);border-bottom:1px solid var(--border);padding:10px 16px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
header h1{font-size:15px;margin:0;font-weight:700}
.tab{padding:6px 14px;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600;background:var(--btn-bg);border:1px solid var(--border);color:var(--dim);transition:all .15s;user-select:none}
.tab:hover{border-color:var(--accent)}
.tab.active{background:var(--accent);color:#000;border-color:var(--accent)}
.stats{font-size:12px;color:var(--dim);margin-left:auto}
main{display:grid;grid-template-columns:1fr 300px;min-height:calc(100vh - 52px)}
.viewport{overflow-y:auto;padding:12px}
.thumb-strip{display:flex;gap:5px;overflow-x:auto;padding-bottom:8px;border-bottom:1px solid var(--border);margin-bottom:12px;min-height:72px}
.thumb-strip img{width:64px;height:64px;object-fit:cover;border-radius:4px;cursor:pointer;border:2px solid transparent;opacity:.45;transition:all .1s;flex-shrink:0;background:var(--card)}
.thumb-strip img.active{border-color:var(--accent);opacity:1;transform:scale(1.06)}
.thumb-strip img.done-ok{opacity:1;border-color:var(--ok)}
.thumb-strip img.done-no{opacity:.9;border-color:var(--no)}
.thumb-strip img.done-skip{opacity:.9;border-color:var(--skip)}
.img-wrap{position:relative;display:flex;align-items:center;justify-content:center;background:#000;border-radius:8px;overflow:hidden;min-height:400px}
.img-wrap img{max-width:100%;max-height:calc(100vh - 180px);object-fit:contain;display:block}
.img-placeholder{color:var(--dim);font-size:14px;text-align:center;padding:40px}
.meta-bar{display:flex;gap:12px;align-items:center;padding:10px 0;flex-wrap:wrap}
.badge{padding:3px 10px;border-radius:4px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em}
.badge.ok{background:#064e3b;color:var(--ok)}
.badge.no{background:#7f1d1d;color:var(--no)}
.badge.skip{background:#374151;color:var(--skip)}
.badge.pending{background:var(--btn-bg);color:var(--dim)}
.counter{font-size:13px;color:var(--dim)}
.rid{font-family:monospace;font-size:12px;color:var(--dim)}
.btns{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-top:10px}
.btn{padding:12px 8px;border-radius:8px;border:2px solid var(--border);background:var(--btn-bg);color:var(--text);cursor:pointer;font-size:14px;font-weight:700;transition:all .1s;text-align:center;user-select:none}
.btn:hover{transform:translateY(-1px)}
.btn:active{transform:translateY(0)}
.btn-ok{border-color:var(--ok)}.btn-ok:hover{background:#064e3b}
.btn-no{border-color:var(--no)}.btn-no:hover{background:#7f1d1d}
.btn-skip{border-color:var(--skip)}.btn-skip:hover{background:#374151}
.kbd{display:block;font-size:10px;font-weight:400;margin-top:3px;opacity:.7}
.panel{border-left:1px solid var(--border);padding:12px;overflow-y:auto;max-height:calc(100vh - 52px)}
.panel h2{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin:0 0 10px}
.progress-bar{height:4px;background:var(--border);border-radius:2px;overflow:hidden;margin-bottom:12px}
.progress-fill{height:100%;background:var(--ok);border-radius:2px;transition:width .3s}
.item-list{list-style:none;padding:0;margin:0}
.item-list li{display:flex;align-items:center;gap:8px;padding:5px 8px;border-radius:6px;cursor:pointer;font-size:12px;font-family:monospace;user-select:none}
.item-list li:hover{background:var(--btn-bg)}
.item-list li.active{background:var(--btn-bg);border:1px solid var(--accent)}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.dot-ok{background:var(--ok)}.dot-no{background:var(--no)}.dot-skip{background:var(--skip)}.dot-pending{background:var(--dim)}
.export-btn{width:100%;padding:10px;margin-top:16px;border-radius:8px;border:2px solid var(--accent);background:transparent;color:var(--accent);font-size:14px;font-weight:700;cursor:pointer;transition:all .1s}
.export-btn:hover{background:var(--accent);color:#000}
.export-btn:disabled{opacity:.4;cursor:not-allowed}
.saved-msg{font-size:12px;color:var(--ok);margin-top:8px;text-align:center;display:none}
.help{margin-top:16px;font-size:11px;color:var(--dim);line-height:1.6}
.help kbd{background:var(--btn-bg);padding:1px 5px;border-radius:3px;font-family:monospace;border:1px solid var(--border);font-size:10px}
</style></head><body>
<header>
<h1>Rotulador Canhoto</h1>
<div class="tab active" data-group="nao_assinado">Não assinados (<span id="cnt-na">0</span> pend.)</div>
<div class="tab" data-group="assinado">Assinados (<span id="cnt-sim">0</span> pend.)</div>
<div class="stats"><span id="stat-total">0</span> total · <span id="stat-done">0</span> rotul. · <span id="stat-ok">0</span> ✓ · <span id="stat-no">0</span> ✗ · <span id="stat-skip">0</span> ⏭</div>
</header>
<main>
<div class="viewport">
<div class="thumb-strip" id="strip"></div>
<div class="img-wrap" id="imgwrap">
<div class="img-placeholder"><span>Nenhuma imagem</span></div>
</div>
<div class="meta-bar">
<span class="badge pending" id="badge">pendente</span>
<span class="counter" id="counter">—</span>
<span class="rid" id="rid-display"></span>
</div>
<div class="btns">
<div class="btn btn-ok" data-action="ok">Tem assinatura<span class="kbd">[1] ou [↑]</span></div>
<div class="btn btn-no" data-action="no">Sem assinatura<span class="kbd">[0] ou [↓]</span></div>
<div class="btn btn-skip" data-action="skip">Ignorar<span class="kbd">[2] ou [→]</span></div>
</div>
</div>
<div class="panel">
<h2>Progresso</h2>
<div class="progress-bar"><div class="progress-fill" id="prog" style="width:0%"></div></div>
<ul class="item-list" id="list"></ul>
<button class="export-btn" id="exportBtn" disabled>Exportar rótulos (JSON)</button>
<div class="saved-msg" id="savedMsg">✓ Baixado rotulado.json</div>
<div class="help">
<b>Atalhos:</b> <kbd>1</kbd>/<kbd>↑</kbd> tem assinatura · <kbd>0</kbd>/<kbd>↓</kbd> sem · <kbd>2</kbd>/<kbd>→</kbd> ignorar · <kbd>←</kbd> voltar<br>
<b>Critério:</b> <b>sem assinatura</b> só se o campo "Identificação e assinatura do recebedor" estiver em branco. Qualquer escrita à mão nele (assinatura, rubrica, nome, número, carimbo preenchido) = <b>tem assinatura</b>. Data sozinha não conta. Campo ilegível ou fora da foto = ignorar.
</div>
</div>
</main>
<script>
const DATA = __DATA__;
const BASE = '__BASE__';
const GROUPS = {nao_assinado:[],assinado:[]};
DATA.forEach(d=>{
  if(d.group==='nao_assinado') GROUPS.nao_assinado.push(d);
  else if(d.group==='assinado') GROUPS.assinado.push(d);
});

let imgLabels = {};
let currentGroup = 'nao_assinado';
let currentIdx = 0;

function currentItem(){
  const items = GROUPS[currentGroup];
  if(!items.length) return null;
  return items[currentIdx % items.length] || null;
}

function imgSrc(d){ return BASE + '/../images/' + d.image.split('/').pop(); }

function render(){
  // tabs
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active', t.dataset.group===currentGroup));
  document.getElementById('cnt-na').textContent = GROUPS.nao_assinado.filter(d=>!imgLabels[d.id]).length;
  document.getElementById('cnt-sim').textContent = GROUPS.assinado.filter(d=>!imgLabels[d.id]).length;
  // stats
  const total = DATA.length;
  const done = Object.keys(imgLabels).length;
  const ok = Object.values(imgLabels).filter(v=>v==='ok').length;
  const no = Object.values(imgLabels).filter(v=>v==='no').length;
  const skip = Object.values(imgLabels).filter(v=>v==='skip').length;
  document.getElementById('stat-total').textContent = total;
  document.getElementById('stat-done').textContent = done;
  document.getElementById('stat-ok').textContent = ok;
  document.getElementById('stat-no').textContent = no;
  document.getElementById('stat-skip').textContent = skip;
  document.getElementById('prog').style.width = total?Math.round(done/total*100)+'%':'0%';
  // list
  const list = document.getElementById('list');
  list.innerHTML = '';
  const items = GROUPS[currentGroup];
  items.forEach((d,i)=>{
    const li = document.createElement('li');
    const lbl = imgLabels[d.id];
    li.className = i===currentIdx?'active':'';
    li.innerHTML = '<span class="dot dot-'+(lbl||'pending')+'"></span><span>'+d.id.slice(0,8)+'</span>';
    li.onclick = ()=>{currentIdx=i;render();};
    list.appendChild(li);
  });
  // thumb strip
  const strip = document.getElementById('strip');
  strip.innerHTML = '';
  items.forEach((d,i)=>{
    const img = document.createElement('img');
    img.src = imgSrc(d);
    img.loading = 'lazy';
    img.onerror = ()=>{ img.style.background='var(--card)'; img.alt='?'; };
    if(i===currentIdx) img.classList.add('active');
    const lbl = imgLabels[d.id];
    if(lbl==='ok') img.classList.add('done-ok');
    if(lbl==='no') img.classList.add('done-no');
    if(lbl==='skip') img.classList.add('done-skip');
    img.onclick = ()=>{currentIdx=i;render();};
    strip.appendChild(img);
  });
  // main image
  const item = currentItem();
  const wrap = document.getElementById('imgwrap');
  if(!item){
    wrap.innerHTML = '<div class="img-placeholder"><span>Nenhuma imagem</span></div>';
  } else {
    const img = document.createElement('img');
    img.src = imgSrc(item);
    img.alt = item.id;
    img.onerror = ()=>{
      wrap.innerHTML = '<div class="img-placeholder"><span>Imagem não encontrada<br><small>'+item.image+'</small></span></div>';
    };
    wrap.innerHTML = '';
    wrap.appendChild(img);
  }
  // badge + counter
  const badge = document.getElementById('badge');
  const lbl2 = item ? imgLabels[item.id] : null;
  badge.className = 'badge ' + (lbl2||'pending');
  badge.textContent = lbl2==='ok'?'tem assinatura':lbl2==='no'?'sem assinatura':lbl2==='skip'?'ignorado':'pendente';
  const items2 = GROUPS[currentGroup];
  document.getElementById('counter').textContent = items2.length ? (currentIdx+1)+' / '+items2.length : '—';
  document.getElementById('rid-display').textContent = item ? item.id : '';
  // export button
  const expBtn = document.getElementById('exportBtn');
  expBtn.disabled = done < 1;
  expBtn.textContent = done < 1 ? 'Exportar (nada rotulado)' : 'Exportar rótulos ('+done+' rotulados)';
}

function setLabel(action){
  const item = currentItem();
  if(!item) return;
  imgLabels[item.id] = action==='ok'?'ok':action==='no'?'no':'skip';
  // auto-advance to next unlabeled in current group
  const items = GROUPS[currentGroup];
  let next = -1;
  for(let i=0;i<items.length;i++){
    if(!imgLabels[items[i].id]){next=i;break;}
  }
  if(next>=0) currentIdx = next;
  saveToLocal();  // save before render: a render error must never lose a label
  render();
}

function goBack(){
  const items = GROUPS[currentGroup];
  if(currentIdx>0){currentIdx--;render();}
}

function saveToLocal(){
  try{localStorage.setItem('rotulador_data', JSON.stringify(imgLabels));}catch(e){}
}
function loadFromLocal(){
  try{
    const raw = localStorage.getItem('rotulador_data');
    if(raw) imgLabels = JSON.parse(raw);
  }catch(e){imgLabels={};}
}

// Tab click
document.querySelectorAll('.tab').forEach(t=>{
  t.onclick = ()=>{
    currentGroup = t.dataset.group;
    currentIdx = 0;
    render();
  };
});

// Button clicks
document.querySelectorAll('.btn[data-action]').forEach(b=>{
  b.onclick = ()=>setLabel(b.dataset.action);
});

// Keyboard
document.addEventListener('keydown',e=>{
  if(e.key==='1'||e.key==='ArrowUp') setLabel('ok');
  else if(e.key==='0'||e.key==='ArrowDown') setLabel('no');
  else if(e.key==='2'||e.key==='ArrowRight') setLabel('skip');
  else if(e.key==='ArrowLeft') goBack();
});

// Export
document.getElementById('exportBtn').onclick = function(){
  const result = DATA.map(d=>{
    const lbl = imgLabels[d.id];
    if(!lbl) return null;
    if(lbl==='skip') return {id:d.id,action:'skip'};
    return {id:d.id,action:lbl,possui_assinatura:lbl==='ok'};
  }).filter(Boolean);
  const blob = new Blob([JSON.stringify(result,null,2)],{type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'rotulado.json';
  a.click();
  const msg = document.getElementById('savedMsg');
  msg.style.display='block';
  setTimeout(()=>msg.style.display='none',3000);
};

loadFromLocal();
render();
</script></body></html>
"""


def generate_html(candidates: list[dict[str, Any]]) -> str:
    data_json = json.dumps(candidates, ensure_ascii=False)
    return HTML_TEMPLATE.replace("__DATA__", data_json).replace("__BASE__", "http://127.0.0.1:8742")


def serve(candidates: list[dict[str, Any]]) -> None:
    import http.server
    import webbrowser

    TRIAGEM.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(generate_html(candidates), encoding="utf-8")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(GOLD), **kwargs)

        def log_message(self, format, *args):
            pass

        def do_GET(self):
            if self.headers.get("Host") not in ALLOWED_HOSTS:  # bloqueia DNS rebinding
                self.send_error(403)
                return
            path = urllib.parse.urlsplit(self.path).path
            if path in ("/", "/index.html"):
                path = "/triagem/rotulador.html"
            elif not IMAGE_PATH.fullmatch(path):
                self.send_error(404)  # só a página e as imagens do gold; nada de manifest, raw ou backups
                return
            self.path = path
            return super().do_GET()

        def do_HEAD(self):
            self.send_error(405)

    try:
        with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler) as httpd:
            url = f"http://127.0.0.1:{PORT}/"
            print(f"Rotulador em: {url}")
            print(f"Servindo a partir de: {GOLD.resolve()}")
            print("Pressione Ctrl+C para parar.")
            try:
                webbrowser.open(url)
            except Exception:
                pass
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor parado.")


def check_item(item: dict[str, Any], by_id: dict[str, dict[str, Any]], overwrite: bool) -> str | None:
    rid = item.get("id")
    if rid not in by_id:
        return f"id {rid!r} não existe no manifest"
    action = item.get("action")
    if action == ACTION_SKIP:
        return None
    if action not in ACTIONS:
        return f"{rid}: action {action!r} inválida (use ok, no ou skip)"
    tipo = item.get("tipo_documento", DEFAULT_TIPO)
    if tipo not in DOC_TYPES:
        return f"{rid}: tipo_documento {tipo!r} inválido (use {', '.join(DOC_TYPES)})"
    if tipo == "canhoto" and item.get("possui_assinatura") is not ACTIONS[action]:
        return f"{rid}: possui_assinatura {item.get('possui_assinatura')!r} não confere com action {action!r}"
    if (by_id[rid].get("triagem") or {}).get("fonte") == FONTE_PROTEGIDA and not overwrite:
        return f"{rid}: já tem rótulo da triagem manual (use --sobrescrever para trocar)"
    return None


def validate_items(items: list[dict[str, Any]], by_id: dict[str, dict[str, Any]], overwrite: bool) -> list[str]:
    errors, seen = [], set()
    for number, item in enumerate(items, start=1):
        error = check_item(item, by_id, overwrite)
        if error is None and item.get("id") in seen:
            error = f"id {item.get('id')!r} duplicado"
        if error:
            errors.append(f"item {number}: {error}")
        seen.add(item.get("id"))
    return errors


def apply_item(record: dict[str, Any], item: dict[str, Any]) -> None:
    if item["action"] == ACTION_SKIP:
        record["triagem"] = {**(record.get("triagem") or {}), "fonte": FONTE, "pulado": True}
        return
    tipo = item.get("tipo_documento", DEFAULT_TIPO)
    record.update(status=STATUS_LABELED, tipo_documento=tipo, rotulo_parcial=True,
                  triagem={"fonte": FONTE, "nota": ""})
    if tipo == "canhoto":
        record["assinatura"] = {"possui_assinatura": ACTIONS[item["action"]], "descricao": ""}
    else:
        record.pop("assinatura", None)


def importar(rotulado_path: Path, aplicar: bool, overwrite: bool) -> int:
    if not rotulado_path.exists():
        print(f"Arquivo não encontrado: {rotulado_path}")
        return 1
    items = json.loads(rotulado_path.read_text(encoding="utf-8"))
    manifest = load_manifest(MANIFEST)
    by_id = {r["id"]: r for r in manifest}
    errors = validate_items(items, by_id, overwrite)
    if errors:
        print("Importação recusada, nada foi gravado. Corrija:")
        print("\n".join(errors))
        return 1
    skipped = sum(1 for item in items if item["action"] == ACTION_SKIP)
    print(f"válidos: {len(items) - skipped} a rotular, {skipped} pulados")
    if not aplicar:
        print("Modo dry-run: nada gravado.")
        return 0
    for item in items:
        apply_item(by_id[item["id"]], item)
    print(f"Gravado. Backup: {write_with_backup(manifest)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Rotulador dinâmico — serve HTML local, importa rótulos")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--servir", action="store_true", help="servir o rotulador HTML na porta 8742")
    group.add_argument("--importar", type=str, help="caminho do JSON de rótulos para importar")
    parser.add_argument("--dry-run", action="store_true", help="não grava no manifest")
    parser.add_argument("--sobrescrever", action="store_true", help="permite trocar rótulos da triagem manual")
    parser.add_argument("--fila-revisao", type=Path, help="scores da quarentena: revisa os pendentes, menor score primeiro")
    args = parser.parse_args(argv)

    manifest = load_manifest(MANIFEST)
    candidates = build_review_queue(manifest, args.fila_revisao) if args.fila_revisao else build_candidates(manifest)
    print(f"Candidatos: {len(candidates)}")
    print(f"  assinado:      {sum(1 for c in candidates if c['group']=='assinado')}")
    print(f"  nao_assinado:  {sum(1 for c in candidates if c['group']=='nao_assinado')}")

    if args.servir:
        serve(candidates)
        return 0
    return importar(Path(args.importar), not args.dry_run, args.sobrescrever)


if __name__ == "__main__":
    sys.exit(main())
