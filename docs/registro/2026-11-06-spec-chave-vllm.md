# Spec F4: chave de acesso lida pelo vLLM — 2026-11-06

Status: implementada e desligada; conferência cruzada nova; falta a medição que decide ligar.
Relacionada: `2026-11-06-spec-leitura.md` (opção 3 da cascata de leitura).

## 1. Objetivo
Recuperar a chave de 44 dígitos nas fotos em que o código de barras e o OCR não a leram (caso da nota 817937), sem nunca anexar a nota errada.

## 2. Como funciona hoje
1. `analyze` tenta o código de barras (plano e reforçado) e o OCR. Sem chave válida, `wants_key` decide se o juiz é consultado.
2. Só pergunta quando `JUDGE_READ_KEY=1`, o juiz existe, não há chave e o tipo é `imagem_invalida` ou `canhoto`, ou o OCR veio curto.
3. `key_reading.read` faz uma pergunta dirigida ao vLLM (JSON curto), respeitando `judge_capacity` e o prazo (`JUDGE_READ_TIMEOUT`, padrão 8 s).
4. A chave devolvida só vale se passar **todas** as conferências:

| Conferência | Resultado em `_meta.leitura_juiz` |
|---|---|
| Nada lido | `sem_chave` |
| 44 dígitos com dígito verificador válido | `invalida` |
| CNPJ da chave na whitelist | `fora_da_whitelist` |
| **Número e CNPJ impressos no OCR batem com a chave (novo)** | `conflito_com_o_texto` |
| Tudo certo | `chave` |

5. Falha ou falta de capacidade registram `falha` e `sem_capacidade`; o caso segue para a reserva (Gemini).

## 3. O que mudou nesta spec
**Conferência cruzada com o texto do OCR.** O dígito verificador da NF-e aceita por acaso cerca de uma em dez leituras erradas, e uma chave errada com CNPJ da whitelist anexaria a nota errada. Agora o número da nota e o CNPJ que o OCR leu da mesma foto precisam bater com a chave do juiz.
- Silêncio do texto não é conflito: sem número ou CNPJ lido, a chave segue só pelas outras conferências.
- Qualquer conflito descarta a chave e manda o caso à reserva. Um OCR que leu o número com erro também barra, o que é o lado seguro.
- Código: `key_reading._agrees_with_text`, chamado por `type_steps.read_key` com `evidence.text`. Testes em `tests/test_key_reading.py`.

## 4. Critério para ligar `JUDGE_READ_KEY`
Rodar `medir_chave_vllm.py` no .160 sobre o gold set (precisa de `/tmp/gold`):
- **Obrigatório:** `com_codigo.valida_diferente` igual a **0** (nenhuma chave do vLLM, válida pelas conferências, diferente da lida pelo código de barras).
- **Desejável:** quantas fotos sem chave passam a ter chave válida (ganho real) e o tempo mediano por pergunta.
- Ressalva: o script foi escrito antes da conferência cruzada e mede só o que o vLLM devolve. A conferência cruzada reduz ainda mais o risco, mas não substitui a medição.

## 5. Como ligar
1. Medir com `JUDGE_DISABLE_THINKING` desligado e ligado (ver `2026-11-06-spec-leitura.md`): com raciocínio, 8 s pode não bastar e a pergunta vira `falha`.
2. No `compose.yaml`, acrescentar ao `environment`: `JUDGE_READ_KEY: ${JUDGE_READ_KEY:-}` e `JUDGE_READ_TIMEOUT: ${JUDGE_READ_TIMEOUT:-8}`.
3. Definir `JUDGE_READ_KEY=1` no stack do Portainer; reverter é remover a variável.
4. Acompanhar `leitura_juiz` no `decisoes.jsonl`: a taxa de `chave` e, principalmente, de `conflito_com_o_texto` e `invalida`.

## 6. Riscos
- **Chave errada aceita:** reduzida por dígito verificador, whitelist e conferência cruzada; nenhuma é infalível. Por isso a medição com `valida_diferente` igual a 0 é obrigatória.
- **Carga no vLLM:** cada pergunta é uma chamada a mais. O `judge_capacity` já barra quando o vLLM está ocupado.
- **Tempo:** pergunta, voto de assinatura e transcrição dividem o prazo de 25 s (`REVIEW_DEADLINE`).
- **Reparo de dígito:** continua fora do escopo (a spec original proíbe estimar dígitos).

## 7. Pendências
- Rodar `medir_chave_vllm.py` e decidir a flag.
- Conferir as divergências de chave `MOHPMSRQ781T`, `7RSDD83DOJRI` e `80UUGQ4RHD1S`: a conferência cruzada pode explicar alguma.
