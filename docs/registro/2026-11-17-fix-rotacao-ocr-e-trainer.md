# 2026-11-17 — fix: rotação do OCR e trainer que não cai

## 1. Foto em pé enviada deitada ao juiz e à visão (`service/app/vision/ocr.py`)

**Sintoma.** Canhoto assinado (NF 240077, foto em pé) saiu `rotacao=90`, auditoria `sem_cnpj`/`sem_numero`,
inconclusivo. Canhoto sobre comprovante de frete (NF 241953) saiu `rotacao=180`, 16 caracteres de OCR,
Laya `imagem_invalida` (0,77) e juiz `sem_chave`.

**Causa.** Em foto com muito fundo, o `psm 3` (análise de layout) lê zero caracteres a 0°. As outras
orientações eram comparadas pelo mesmo `psm 3`, que endireita sozinho o texto lateral: a 90° ele lê
"RECEBEMOS DE JFC PRODUTOS…" com confiança e vence. A rotação errada vai para `upright_jpeg`, então o juiz,
a visão do Laya e a segunda olhada recebem a foto deitada ou de cabeça para baixo.

**Correção.** Quando a primeira leitura não é legível, leituras esparsas (`psm 11`, que não reorienta)
votam a orientação; o texto final é o melhor entre layout e esparso nela. Cada voto tem 1,5 s: em fundo
texturizado o `psm 11` leva 20 s lendo ruído, e como a orientação certa é a mais lenta, um voto com
timeout é descartado e volta o critério antigo.

**Medição** (amostra fixa de 200 do gold + as 2 fotos, antigo × novo): a rotação mudou em 13 fotos;
12 passaram de errada para certa (conferidas a olho pela posição do QR), 1 seguiu errada (`1ce901`,
página em pé: 90 → 270). Nenhuma certa virou errada. Tempo mediano 3,7 → 3,8 s; p95 8,7 → 9,7 s.

## 2. Trainer saía do processo a cada falha (`trainer/app`)

**Sintoma.** Cada falha de kernel encerrava o container (exit 0 + `restart: on-failure`), com o motivo
genérico "o kernel falhou ou passou do tempo". A falha de 17/11 não era timeout: o kernel morreu em
76 s com `AttributeError` do callback raytune do ultralytics 8.3.40 (corrigido em `4d028e3`).

**Correção.**
- No `FAILED`, o ciclo baixa a saída do kernel para `saida/` (vai para o backup) e põe no motivo a última
  exceção do stderr do log. Timeout real passa a dizer "o kernel passou do tempo".
- Parado por falha, o processo segue vivo (heartbeat/healthcheck ok) e loga a parada uma vez.
- Retomada automática: até `TRAINER_AUTO_RESUME_MAX` (2) vezes, cada uma após
  `TRAINER_AUTO_RESUME_SECONDS` (3600 s); depois espera o `retomar` manual, que zera o contador.
  Sucesso também zera.
- `compose.yaml`: `restart: unless-stopped`.

**Limite do Kaggle.** Não há "janela maior" a configurar: a sessão de GPU do Kaggle encerra em cerca de
9 h e o trainer já espera até 10 h (`KERNEL_TIMEOUT_SECONDS`).

## Pendências
- Falha de `visao-1794944999` (17/11 19:53 UTC, ~3,5 min) ainda sem causa: buscar o log do kernel.
- Pacote de visão pequeno (36 amostras; teste com 1 classe só): recusar pacote cujo teste não tenha
  todas as classes antes de gastar GPU.
