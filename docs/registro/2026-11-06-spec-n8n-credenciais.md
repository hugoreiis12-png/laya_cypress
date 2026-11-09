# Spec F6: n8n como roteador e troca das credenciais — 2026-11-06

Status: especificada. A configuração do n8n e a troca de credenciais são do usuário; o serviço já devolve o necessário.
Base: `2026-11-05-n8n-roteamento.md` (fluxo e nós) e `2026-11-05-continuidade.md` (pendências urgentes).

## 1. Objetivo
O serviço (`laya-cypress`) é o fluxo principal e o Gemini é a reserva. Nenhuma resposta duvidosa pode seguir como definitiva, e nenhuma credencial exposta pode continuar valendo.

## 2. O que o serviço entrega ao n8n
- `_meta.inconclusivo` (booleano): `true` manda a foto à reserva.
- **Novo:** `_meta.motivos_inconclusivo`, uma lista com a razão. Só existe quando `inconclusivo` é `true`.

| Motivo | Significado |
|---|---|
| `tipo_sem_resposta_propria` | Comprovante, pedido ou imagem inválida: o serviço só responde sozinho canhoto (`CONCLUSIVE_TYPES`) |
| `assinatura_nao_confirmada` | Canhoto sem voto de assinatura confirmado (juiz falhou, se absteve, sem capacidade ou sem prazo) |
| `auditoria` | Falta CNPJ ou número, CNPJ diverge da chave ou há mais de uma chave |
| `imagem_ilegivel` | A foto não pôde ser decodificada |

- O mesmo campo vai ao registro de decisões (`decisoes.jsonl`), para medir por que casos vão à reserva.
- `possui_assinatura: false` com `confianca: 0` significa "não consegui confirmar", não "sem assinatura": `inconclusivo` vem `true` nesse caso.

## 3. Alterações no n8n (nesta ordem; duplique o fluxo antes)
1. **`sombra_laya_cypress`: `On Error: Continue (using error output)`**, com a saída de erro em `reanexar_imagem`. **Faça isto antes de qualquer troca de credencial**: é o que transforma uma falha do serviço em reserva, e não em perda.
2. **IF `inconclusivo`** logo depois do serviço e **antes** de `ajustar_resposta`: `{{ $json._meta.inconclusivo }}` igual a `true` vai a `reanexar_imagem`; `false` segue para `ajustar_resposta`.
3. **`reanexar_imagem`:** `return [{ json: {}, binary: $('HTTP Request3').first().binary, pairedItem: { item: 0 } }];`
4. **IF `usar_servico`** como chave de reversão (`false` volta todo o tráfego ao `AI Agent1`).
5. **IF `amostra_gemini`** (cerca de 10% por final de telefone) para manter a comparação.
6. **If7:** hoje leva adiante assinatura >= 90 **mesmo sem CNPJ ou número**. Exigir `emitente.cnpj` e `nota_fiscal.numero` preenchidos.
7. **Chave do Gemini:** validar 44 dígitos e dígito verificador antes de usar (já houve chave de 47 dígitos). Em um nó Code:

```javascript
const key = String($json.output?.qrcode?.chave ?? '').replace(/\D/g, '');
let valid = false;
if (key.length === 44) {
  let sum = 0, weight = 2;
  for (let i = 42; i >= 0; i--) { sum += Number(key[i]) * weight; weight = weight === 9 ? 2 : weight + 1; }
  const rest = sum % 11;
  valid = Number(key[43]) === (rest < 2 ? 0 : 11 - rest);
}
return [{ json: { ...$json, chave_valida: valid } }];
```

## 4. Contrato de saída: `""` ou `null`
- O serviço devolve `""` para campo vazio; o contrato do Agent1 pede `null`.
- **Não mudei isso:** `ajustar_resposta` completa o número com zeros, e um `null` pode quebrá-lo. Decisão de duas etapas: (a) confirmar no n8n como `ajustar_resposta` trata `null` e `""`; (b) só então escolher e travar em `tests/test_final_output.py`.

## 5. Troca das credenciais expostas
Inventário (sem valores): token do WhatsApp, chave do OpenRouter, chaves dos workflows do Bubble, `API_KEY` do serviço e o token do Kaggle (colado em 06/11).

1. Guarde cada segredo no **cofre de credenciais do n8n** (credencial do tipo Header Auth ou nativa), nunca dentro da expressão ou do corpo do nó.
2. Troque na ordem: Kaggle, OpenRouter, Bubble, WhatsApp e, por último, a `API_KEY` do serviço.
3. **`API_KEY` do serviço:** o serviço só aceita uma chave por vez, então há uma janela de falha entre atualizar o stack no Portainer e a credencial do n8n. Com o item 3.1 pronto, as falhas dessa janela caem na reserva. Faça fora do horário de pico.
4. Revogue as antigas só depois de confirmar que a nova funciona (um canhoto de teste pelo fluxo).
5. Conferir que nenhum segredo está no Git: `.env`, `.env.example` (só nomes), `compose.yaml` (só `${VAR}`).

## 6. Roteiro de teste de ponta a ponta (a rodar ao final)
Pré-requisito: deploy com as correções de 06/11 e **todas as variáveis do `.env` antigo** no stack (`JUDGE_URL`, `JUDGE_MODEL`, `SIGNATURE_ENGINE`, `SIGNATURE_SHADOW_URL`, `HERMES_URL`, `TYPE_ENGINE`, `DECISION_LOG_PATH` e as chaves). Sem elas o serviço roda sem juiz.

| Caso | Foto | Aceitável | Inaceitável |
|---|---|---|---|
| Nota 817663 | DANFE deitada sobre cenouras | `rotacao` 90 ou 270, `caracteres_ocr` > 40; se tipo e assinatura fecham, `canhoto` assinado; senão `inconclusivo` | `rotacao` 180 com OCR vazio |
| Nota 817937 | Foto amassada, chave não lida | Não sai como `pedido`; `imagem_invalida` ou `canhoto` com chave, sempre com `motivos_inconclusivo` se for à reserva | `tipo_documento: pedido` com itens de cabeçalho |
| Nota 607345 | Só carimbo no campo do recebedor | `possui_assinatura: false` ou `inconclusivo` com `assinatura_nao_confirmada` | `possui_assinatura: true` liberado sem reserva |
| Nota 413886 | Juiz falhou antes | Resposta do juiz, ou `erros` no log com o motivo (tempo ou formato) | Falha sem motivo registrado |
| Serviço parado | Qualquer foto | Gemini responde (`onError`) | Fluxo sem resposta |
| Chave de 47 dígitos do Gemini | Nota com chave longa | `chave_valida: false` e caso tratado | Chave inválida seguindo adiante |

Onde olhar: o `_meta` da resposta e as linhas novas do `decisoes.jsonl` (`motivos_inconclusivo`, `erros`, `votos`, `leitura_juiz`, `rotacao`).

## 7. Pendências
- Executar os itens do §3 no n8n e confirmar o efeito do §4.
- Trocar as credenciais do §5 e confirmar cada uma.
- Rodar o roteiro do §6 depois do deploy.
