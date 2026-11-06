"""Training progress with numbers only, so status can show it without exposing data."""
from pathlib import Path
import time
from .storage import atomic_json, read_json

FILE = 'progresso.json'
FIELDS = ('passo', 'total_passos', 'epoca', 'epocas', 'loss', 'decorrido_s', 'restante_s')


class Progress:
    """Written by the engine inside its output folder, at most every few seconds."""
    def __init__(self, output, epochs, total_steps, first_step=0):
        self.path = Path(output) / FILE; self.start = time.time()
        self.epochs, self.total, self.first, self.last = epochs, total_steps, first_step, 0.0

    def update(self, step, epoch, loss=None, force=False):
        now = time.time()
        if not force and now - self.last < 2: return
        self.last = now; elapsed = now - self.start; done = step - self.first
        remaining = elapsed / done * max(0, self.total - step) if done > 0 else None
        atomic_json(self.path, {'passo': int(step), 'total_passos': int(self.total), 'epoca': round(float(epoch), 2), 'epocas': self.epochs,
                                'loss': None if loss is None else round(float(loss), 4), 'decorrido_s': round(elapsed),
                                'restante_s': None if remaining is None else round(remaining)})


def read(folder):
    path = Path(folder) / 'output' / FILE
    try: value = read_json(path)
    except (OSError, ValueError): return None
    return {k: value[k] for k in FIELDS if k in value and (value[k] is None or type(value[k]) in (int, float))}


def duration(seconds):
    if seconds is None: return '?'
    if seconds < 90: return '%d s' % seconds
    if seconds < 3600: return '%d min' % round(seconds / 60)
    return '%dh%02dmin' % (seconds // 3600, round(seconds % 3600 / 60))


def bar_line(job):
    """One chat line per update: epoch, bar, step, loss and time left."""
    status = job.get('status'); p = job.get('progresso')
    if status == 'concluida': return 'Treino concluido' + (' em %s' % duration(p['decorrido_s']) if p else '') + '.'
    if status == 'falhou':
        erro = job.get('erro') or {}
        return 'Treino falhou: %s em %s' % (erro.get('tipo') or 'erro', (erro.get('local') or ['?'])[-1])
    if status != 'executando' or not p: return {'submetendo': 'Preparando o treino...', 'executando': 'Carregando o modelo...'}.get(status, 'Status: %s' % status)
    fraction = p['passo'] / p['total_passos'] if p.get('total_passos') else 0
    filled = round(fraction * 20)
    epoch = min(p['epocas'], int(p['epoca']) + 1) if p.get('epocas') else None
    parts = ['Epoca %d/%d' % (epoch, p['epocas']) if epoch else 'Treino', '█' * filled + '░' * (20 - filled), '%d%%' % round(fraction * 100),
             'passo %d/%d' % (p['passo'], p['total_passos'])]
    if p.get('loss') is not None: parts.append('loss %.3f' % p['loss'])
    parts.append('falta ~%s' % duration(p.get('restante_s')))
    return ' · '.join(parts)


def follow(fetch, interval=15, step=5):
    """Print a line at each `step` percent and on status changes, until the job ends. Built for a
    chat monitor, where every line costs a turn: about 20 lines per run, whatever its length."""
    last = None; failures = 0; unknown_since = None
    while True:
        try: job = fetch(); failures = 0
        except Exception:
            # a tunnel hiccup is not the end of the job
            failures += 1
            if failures >= 20: raise
            time.sleep(interval); continue
        status = job.get('status'); p = job.get('progresso') or {}
        # 'desconhecida' can be transient (a slow step without heartbeat): keep watching for a while
        unknown_since = (unknown_since or time.monotonic()) if status == 'desconhecida' else None
        if unknown_since and time.monotonic() - unknown_since < 600: time.sleep(interval); continue
        bucket = int(100 * p['passo'] / p['total_passos'] / step) if status == 'executando' and p.get('total_passos') else None
        key = (status, bucket)
        if key != last: print(bar_line(job), flush=True); last = key
        if status not in ('submetendo', 'executando'): return job
        time.sleep(interval)
