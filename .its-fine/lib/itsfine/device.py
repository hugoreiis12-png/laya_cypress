"""Acelerador de treino e inferencia, o mesmo para todos os motores.

Ordem: GPU NVIDIA (cuda), chip Apple M pelo Metal (mps), CPU so quando permitida.
No Mac a memoria e unificada (a GPU usa a RAM), nao ha bitsandbytes e o Docker nao
enxerga a GPU: treino roda no executor de processo. bf16 no mps exige macOS 14+,
a mesma regra do accelerate e do transformers. torch e importado so aqui dentro,
porque o runtime do ITS-FINE roda sem torch.
"""
import os
import platform
import subprocess

# PYTORCH_ENABLE_MPS_FALLBACK e definido no common.py de cada motor: so vale antes de importar torch.
CPU_TRAINING_ENV = 'ITS_FINE_ALLOW_CPU_TRAINING'


def mps_disponivel(torch):
    backend = getattr(torch.backends, 'mps', None)
    return bool(backend and backend.is_available() and backend.is_built())


def escolher(preferido='auto', treino=False):
    """Nome do dispositivo: cuda, mps ou cpu. Treino em CPU so com ITS_FINE_ALLOW_CPU_TRAINING."""
    import torch
    cuda, mps = torch.cuda.is_available(), mps_disponivel(torch)
    nome = ('cuda' if cuda else 'mps' if mps else 'cpu') if preferido == 'auto' else preferido
    if nome == 'cuda' and not cuda: raise ValueError('CUDA_UNAVAILABLE')
    if nome == 'mps' and not mps: raise ValueError('MPS_UNAVAILABLE')
    if nome not in ('cuda', 'mps', 'cpu'): raise ValueError('UNKNOWN_DEVICE:' + str(nome))
    if nome == 'cpu' and treino and not os.environ.get(CPU_TRAINING_ENV): raise ValueError('ACCELERATOR_REQUIRED_FOR_TRAINING')
    return nome


def bf16(nome):
    import torch
    if nome == 'cuda': return torch.cuda.is_bf16_supported(including_emulation=False)
    if nome == 'mps': return torch.backends.mps.is_macos_or_newer(14, 0)
    return False


def dtype(nome):
    """bf16 onde o hardware suporta; fp16 em GPU NVIDIA sem bf16 (T4); fp32 no resto."""
    import torch
    if bf16(nome): return torch.bfloat16
    return torch.float16 if nome == 'cuda' else torch.float32


def exigir_cuda_para_quantizar(nome, quantizado):
    # bitsandbytes (QLoRA 4 bits, Adam 8 bits) so tem kernels CUDA
    if quantizado and nome != 'cuda': raise ValueError('QUANTIZATION_REQUIRES_CUDA')


def device_map(nome):
    return {'': 0 if nome == 'cuda' else nome}


def nome_do_hardware(nome):
    import torch
    if nome == 'cuda': return torch.cuda.get_device_name(0)
    if nome == 'mps': return chip_apple() or 'Apple Silicon (mps)'
    return platform.processor() or 'cpu'


def chip_apple():
    """Ex.: 'Apple M2 Pro'; None fora do macOS."""
    if platform.system() != 'Darwin': return None
    try:
        saida = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True, timeout=10)
        return saida.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


class Memoria:
    """Pico de memoria do acelerador. No mps nao ha contador de pico: amostra a cada passo."""

    def __init__(self, nome):
        import torch
        self.torch, self.nome, self.pico_mps = torch, nome, 0
        if nome == 'cuda': torch.cuda.reset_peak_memory_stats()

    def amostrar(self):
        if self.nome == 'mps':
            self.pico_mps = max(self.pico_mps, self.torch.mps.driver_allocated_memory())

    def relatorio(self):
        torch = self.torch
        if self.nome == 'cuda':
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            reservado = torch.cuda.max_memory_reserved() / 1e9
            return {'pico_alocado_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2), 'pico_reservado_gb': round(reservado, 2),
                    'total_gb': round(total, 2), 'coube': reservado < total,
                    'nota': 'reservado acima do total significa fallback para memoria do sistema: roda, mas lento; reduza batch'}
        if self.nome == 'mps':
            self.amostrar()
            # recommended_max_memory: quanto o Metal deixa a GPU usar da memoria unificada
            limite = torch.mps.recommended_max_memory() / 1e9
            pico = self.pico_mps / 1e9
            return {'pico_alocado_gb': round(pico, 2), 'pico_reservado_gb': round(pico, 2), 'total_gb': round(limite, 2),
                    'coube': pico < limite, 'memoria_unificada': True,
                    'nota': 'Mac: memoria unificada; total_gb e o limite recomendado pelo Metal, abaixo da RAM total'}
        return {'pico_alocado_gb': None, 'pico_reservado_gb': None, 'total_gb': None, 'coube': True, 'nota': 'cpu'}


def sincronizar(nome):
    import torch
    if nome == 'cuda': torch.cuda.synchronize()
    elif nome == 'mps': torch.mps.synchronize()


def estado_rng(nome):
    import torch
    if nome == 'cuda': return torch.cuda.get_rng_state_all()
    if nome == 'mps': return torch.mps.get_rng_state()
    return None


def restaurar_rng(nome, estado):
    import torch
    if estado is None: return
    if nome == 'cuda': torch.cuda.set_rng_state_all(estado)
    elif nome == 'mps': torch.mps.set_rng_state(estado)
