"""Inventario real da maquina: GPU NVIDIA ou chip Apple M, RAM, disco, Python, Docker, Ollama.

Nao instala nada. Saida em JSON para o fine-environment registrar.
"""
from pathlib import Path
import json
import os
import platform
import shutil
import subprocess
import sys


def executar(cmd, timeout=20):
    try:
        saida = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return saida.returncode, saida.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None, ''


def gpus_nvidia():
    codigo, saida = executar(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version',
                              '--format=csv,noheader,nounits'])
    if codigo != 0 or not saida:
        return []
    placas = []
    for linha in saida.splitlines():
        partes = [p.strip() for p in linha.split(',')]
        if len(partes) >= 4:
            placas.append({'nome': partes[0], 'vram_total_gb': round(int(partes[1]) / 1024, 2),
                           'vram_usada_gb': round(int(partes[2]) / 1024, 2), 'driver': partes[3]})
    return placas


def chip_apple():
    """Mac com chip M: a GPU usa a memoria unificada (a RAM) pelo Metal."""
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        return None
    _, nome = executar(['sysctl', '-n', 'machdep.cpu.brand_string'])
    _, memoria = executar(['sysctl', '-n', 'hw.memsize'])
    total = round(int(memoria) / 1e9, 1) if memoria.isdigit() else None
    _, macos = executar(['sw_vers', '-productVersion'])
    return {'nome': nome or 'Apple Silicon', 'memoria_unificada_gb': total, 'macos': macos or None,
            'bf16': bool(macos) and int(macos.split('.')[0]) >= 14,
            'nota': 'treino pelo Metal (mps); sem QLoRA 4 bits e sem Docker com GPU: use o executor de processo'}


def torch_neste_python():
    try:
        import torch
        mps = getattr(torch.backends, 'mps', None)
        return {'versao': torch.__version__, 'cuda': bool(torch.cuda.is_available()),
                'mps': bool(mps and mps.is_available() and mps.is_built()),
                'cuda_versao': torch.version.cuda if torch.cuda.is_available() else None,
                'bf16': bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
                'python': sys.executable}
    except Exception as erro:
        return {'versao': None, 'cuda': False, 'erro': type(erro).__name__, 'python': sys.executable}


def ram_gb():
    if platform.system() == 'Windows':
        import ctypes
        classe = type('MEMORYSTATUSEX', (ctypes.Structure,),
                      {'_fields_': [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                                    ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                                    ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                                    ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]})
        estado = classe()
        estado.dwLength = ctypes.sizeof(classe)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(estado))
        return {'total_gb': round(estado.ullTotalPhys / 1e9, 1), 'livre_gb': round(estado.ullAvailPhys / 1e9, 1)}
    if platform.system() == 'Darwin':
        _, memoria = executar(['sysctl', '-n', 'hw.memsize'])
        _, paginas = executar(['vm_stat'])
        livre = None
        try:
            tamanho = int(paginas.split('page size of ')[1].split()[0])
            campos = {l.split(':')[0]: int(l.split(':')[1].strip(' .')) for l in paginas.splitlines()[1:] if ':' in l}
            livre = round((campos.get('Pages free', 0) + campos.get('Pages inactive', 0)) * tamanho / 1e9, 1)
        except (IndexError, ValueError):
            pass
        return {'total_gb': round(int(memoria) / 1e9, 1) if memoria.isdigit() else None, 'livre_gb': livre}
    try:
        info = {}
        for linha in Path('/proc/meminfo').read_text().splitlines():
            chave, valor = linha.split(':', 1)
            info[chave] = int(valor.strip().split()[0]) * 1024
        return {'total_gb': round(info['MemTotal'] / 1e9, 1), 'livre_gb': round(info['MemAvailable'] / 1e9, 1)}
    except Exception:
        return {'total_gb': None, 'livre_gb': None}


def docker():
    codigo, versao = executar(['docker', '--version'])
    if codigo != 0:
        return {'instalado': False, 'motor_ativo': False}
    codigo_info, _ = executar(['docker', 'info', '--format', '{{.ServerVersion}}'])
    return {'instalado': True, 'versao': versao, 'motor_ativo': codigo_info == 0}


def ollama():
    codigo, versao = executar(['ollama', '--version'])
    if codigo != 0:
        return {'instalado': False}
    _, lista = executar(['ollama', 'list'])
    modelos = [l.split()[0] for l in lista.splitlines()[1:] if l.strip()]
    return {'instalado': True, 'versao': versao, 'modelos': modelos}


def main():
    import argparse
    argparse.ArgumentParser(description='inventario da maquina em JSON: GPU, RAM, disco, Python, Docker, Ollama').parse_args()
    raiz = Path(os.getcwd())
    uso = shutil.disk_usage(raiz)
    relatorio = {
        'schema_version': 1,
        'sistema': {'so': platform.system(), 'versao': platform.version(), 'maquina': platform.machine()},
        'python': {'versao': platform.python_version(), 'executavel': sys.executable},
        'gpus': gpus_nvidia(),
        'apple': chip_apple(),
        'torch': torch_neste_python(),
        'ram': ram_gb(),
        'disco': {'livre_gb': round(uso.free / 1e9, 1), 'caminho': str(raiz)},
        'docker': docker(),
        'ollama': ollama(),
    }
    gpu = relatorio['gpus'][0] if relatorio['gpus'] else None
    apple = relatorio['apple']
    relatorio['resumo'] = (
        ('GPU %s com %.1f GB' % (gpu['nome'], gpu['vram_total_gb'])) if gpu else
        ('%s com %s GB de memoria unificada (Metal)' % (apple['nome'], apple['memoria_unificada_gb'])) if apple else
        'sem GPU NVIDIA nem chip Apple detectado'
    ) + ', RAM %s GB, disco livre %s GB' % (relatorio['ram']['total_gb'], relatorio['disco']['livre_gb'])
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
