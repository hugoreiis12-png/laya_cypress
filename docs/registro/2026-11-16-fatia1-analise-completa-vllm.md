# Fatia 1: análise completa pelo vLLM em sombra — 2026-11-16

## Objetivo
O vLLM passa a ler a foto inteira com o mesmo contrato do Gemini (tipo, assinatura, CNPJ, número, chave, comprovante, itens de pedido). Nesta fatia ele **só observa**: a resposta ao n8n não muda. Cada leitura vira uma referência `origem: vllm` no `decisoes.jsonl`, para comparar com o Gemini (fatia 2) e, depois, servir de rótulo de treino por consenso (fatia 4).

## Plano completo
| Fatia | O que muda | Critério para seguir |
|---|---|---|
| 0 | Foto endireitada para o juiz (PR #6) | No ar desde 16/11 |
| **1** | **vLLM lê 100% das fotos em segundo plano e grava como referência `vllm`** | ~300 fotos com vLLM e Gemini |
| 2 | Relatório de concordância vLLM × Gemini, por tipo e por campo | Canhoto ≥ 95% de concordância |
| 3 | vLLM vira o caminho principal no n8n; Gemini fica em sombra numa amostra, só para o consenso | Custo do Gemini cai; latência dentro do prazo |
| 4 | Trainer treina com o consenso; a Laya assume um tipo quando acertar ≥ 90% no teste congelado | Gate do trainer; canhoto primeiro |

**Decisões de 16/11:**
- O rótulo de treino vem do consenso entre o vLLM e o Gemini em sombra, nunca do vLLM sozinho.
- "Confiança acima de 90%" significa acerto medido no teste congelado, não a confiança declarada pelo modelo.

## Como funciona
- Ligado por `FULL_ANALYSIS=1`. Desligado por padrão.
- Depois de responder ao n8n, o endpoint põe a foto numa fila em memória (`FULL_ANALYSIS_QUEUE`, padrão 50). A resposta não espera por isso.
- Um único worker processa a fila. Ele só chama o vLLM quando o vLLM está ocioso (nada rodando e nada na fila, pelo `/metrics`) e não ocupa a vaga do juiz ao vivo.
- Uma foto que não encontra o vLLM ocioso em 10 minutos é descartada e registrada (`analise_completa_descartada`, motivo `vllm_ocupado`). Com a fila cheia, o motivo é `fila_cheia`.
- O vLLM recebe a foto endireitada pela rotação do OCR. O tempo limite é `FULL_ANALYSIS_TIMEOUT` (padrão 60 s).
- O registro guarda só o tipo, `possui_assinatura`, `confianca_assinatura` e a latência. Os campos lidos entram como resumo com chave (`DECISION_DIGEST_KEY`), como no Gemini. Imagem, texto e valores nunca entram.
- Se a fila em memória se perder num reinício do container, nenhum atendimento é afetado.

## Regra de assinatura: divergência a decidir
O prompt do Gemini conta **"carimbo preenchido"** como assinatura. O juiz e a análise completa **não contam carimbo sozinho**. Enquanto as duas regras forem diferentes, o consenso descarta justamente os canhotos só com carimbo. Decida qual é a regra do negócio e alinhe os dois prompts.

## Ativar (VM)
1. No `compose.yaml` da raiz, no `environment` do serviço:
   ```yaml
   FULL_ANALYSIS: ${FULL_ANALYSIS:-}
   FULL_ANALYSIS_QUEUE: ${FULL_ANALYSIS_QUEUE:-50}
   FULL_ANALYSIS_TIMEOUT: ${FULL_ANALYSIS_TIMEOUT:-60}
   ```
2. **Portainer:** acrescente `FULL_ANALYSIS=1` no stack do serviço e faça *Pull and redeploy*.
3. Acompanhe:
   ```bash
   grep -c '"origem": "vllm"' /opt/laya-cypress/dados/decisoes.jsonl
   grep -c '"evento": "analise_completa_descartada"' /opt/laya-cypress/dados/decisoes.jsonl
   grep '"origem": "vllm"' /opt/laya-cypress/dados/decisoes.jsonl | grep -o '"latencia_ms": [0-9]*' | tail
   ```
   Se as descartadas crescerem mais que as referências, o vLLM não fica ocioso o bastante. Nesse caso, a fatia 3 precisa de capacidade dedicada.

## Gemini em sombra numa amostra (n8n)
Com o `If_laya_ok` novo, o Gemini só roda nos casos em dúvida. Para comparar com o vLLM também nos casos confirmados, rode-o numa amostra. A saída dessa cópia **nunca** pode ir para o cliente.
1. Da saída **true** do `If_laya_ok`, ligue um **If** `amostra_gemini` com `{{ Math.random() < 0.2 }}` (*Boolean → is true*).
2. Um **Merge** `merge_amostra`, *Combine by position*:
   - entrada 1: `HTTP Request3` (a foto);
   - entrada 2: a saída true do `amostra_gemini`.
3. Copie o `AI Agent1` com o seu modelo como `gemini_sombra` e ligue-o ao `merge_amostra`.
4. Da saída de sucesso do `gemini_sombra`, crie o **HTTP Request** `sombra_referencia_gemini`:
   - `POST http://100.66.232.113:8020/v1/referencia`, com o mesmo `Bearer` do `sombra_laya_cypress`;
   - body: `{{ { id: $('id_foto').item.json.id, origem: 'gemini', output: $json.output } }}`;
   - *On Error: Continue*.
5. Na saída de sucesso do `AI Agent1` (o caminho do cliente), ligue um nó igual ao `sombra_referencia_gemini`, para os casos em dúvida também virarem referência.
