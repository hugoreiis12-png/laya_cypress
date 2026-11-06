# Motor hf-tts

Fine-tuning LoRA de modelos de fala que geram tokens de codec de áudio: o modelo
aprende a falar com a voz do usuário a partir das gravações dele. Treina em GPU
NVIDIA ou Mac com chip M (Metal), localmente, no Colab ou no Kaggle.

## Genérico por família

O núcleo (`train.py`, `lora.py`, `evaluate.py`, `export.py`, `metric.py`,
`ingest.py`, `check.py`) não conhece arquitetura. Cada família entra por um
adaptador em `familias/`, declarado em `engine.json` > `familias`, com esta interface:

| Método | Faz |
|---|---|
| `Familia(origem, dispositivo, dtype, revision, idioma)` | carrega o modelo base (id do Hub ou pasta) |
| `do_pacote(pasta, dispositivo, dtype, idioma)` | carrega um modelo de voz própria salvo |
| `lora()` | módulo onde o LoRA entra, nomes-folha alvo e regex do que excluir |
| `acaso()` | ln do vocabulário de cada componente da perda, para a porta de sanidade |
| `definir_voz(audio)`, `extras()`, `carregar_extras(t)` | condição de voz além do LoRA, se a família precisar |
| `preparar(texto, audio)` | exemplo de treino (códigos do codec, ids do texto) |
| `perda(exemplos)` | perda total e por componente de um lote |
| `salvar(pasta, voz)` | modelo completo de voz própria, pesos já fundidos |
| `falar(texto, voz=None, referencia=None)` | fala pela voz própria ou clona de um clipe |

Família nova: escreva o adaptador, registre em `engine.json` (adaptador, taxa,
memória por segundo, idiomas) e adicione modelos ao catálogo. A porta de sanidade
recusa treinar se a perda do modelo base nos dados separados não ficar abaixo de
80% do acaso: é o sintoma de layout de sequência errado.

Incluídas: `qwen3-tts` (layout tirado da inferência do modelo; o script oficial do
fornecedor desloca os rótulos duas vezes e alimenta o sub-preditor com o estado do
passo errado) e `csm` (nativo do transformers). Flow matching (F5-TTS) fica fora.

## Ambiente

`ambiente: proprio`: venv em `.its-fine/venvs/hf-tts/`, criada por `setup.py`,
porque a família qwen3-tts fixa `transformers==4.57.3`. Herda o torch do sistema
quando ele já enxerga a GPU; senão `--install-torch`. `torchaudio` é fixado na
versão do torch da venv. No Colab e no Kaggle a sessão cria a mesma venv sobre o
torch dela.

## Fluxo

ingest (gravações em áudio ou vídeo → clipes transcritos) → dataset do runtime (split por gravação,
clipes por hash) → check → metric (régua calibrada com áudio real) → evaluate sem
adapter (baseline) → train (sondagem e treino supervisionados) → evaluate com
adapter → escuta (página A/B local) → export (pacote com `falar.py`, reavaliado).

Métrica: `similaridade_inteligivel`, média da semelhança de voz por frase,
contando zero quando o WER da frase passa do máximo. Mede timbre e
inteligibilidade, não naturalidade.

Medido nesta engine numa RTX 4070 Laptop de 8 GB, LoRA r=16, lote 1, clipes até
10 s: Qwen3-TTS 0.6B com pico de 3,3 a 4,1 GB; CSM 1B com 6,3 GB. Geração no
Windows sem flash-attn: cerca de 6 s de processamento por segundo de fala.
