# Colocar em produção: passo a passo

Siga na ordem e marque cada item. Cada passo diz **onde** executar, **o que fazer** e **como saber que deu certo**. Se um "esperado" não acontecer, pare e não avance: a seção *Se algo der errado* no fim diz o que olhar.

Legenda dos lugares: **VM** = terminal na VM .160 · **Portainer** = interface do Portainer · **n8n** = editor do n8n · **Git** = seu computador.

Nunca cole chaves, tokens ou fotos de cliente em chat, documento ou Git.

---

## Fase 0: preparação (10 min)
- [ ] **n8n:** duplique o fluxo atual (menu do fluxo → *Duplicate*). Todo o trabalho abaixo é feito na cópia até a Fase 6.
- [ ] **Git:** confirme que a `main` do GitHub tem os commits mais recentes: `git fetch && git status` não deve mostrar commits locais à frente do `origin/main`.
- [ ] **VM:** anote o uid do container do n8n, usado na Fase 4: `docker exec <nome_do_n8n> id`.

## Fase 1: serviço no ar com as variáveis certas (20 min)
Sem as variáveis, o serviço sobe sem juiz e devolve "não confirmado" em tudo.

- [ ] **Git:** abra o `compose.yaml` da raiz e confira que o `environment` do serviço lista estas linhas (as numéricas com padrão, porque vazio quebra a partida):
  ```yaml
  JUDGE_TRANSCRIBE: ${JUDGE_TRANSCRIBE:-1}
  JUDGE_DISABLE_THINKING: ${JUDGE_DISABLE_THINKING:-}
  JUDGE_MAX_IN_FLIGHT: ${JUDGE_MAX_IN_FLIGHT:-1}
  JUDGE_WAIT_SECONDS: ${JUDGE_WAIT_SECONDS:-5}
  JUDGE_READ_KEY: ${JUDGE_READ_KEY:-}
  JUDGE_READ_TIMEOUT: ${JUDGE_READ_TIMEOUT:-8}
  ```
  Se faltar alguma, acrescente, faça commit e push.
- [ ] **Portainer:** Stacks → stack do serviço → *Environment variables*. Confira que existem (valores do `.env` antigo, sem digitá-los aqui): `JUDGE_URL`, `JUDGE_MODEL`, `SIGNATURE_ENGINE=juiz`, `TYPE_ENGINE=regras`, `SIGNATURE_SHADOW_URL`, `HERMES_URL`, `HERMES_AUDIT`, `DECISION_LOG_PATH`, `DECISION_DIGEST_KEY`, `API_KEY`, `WHITELIST_CNPJ`.
- [ ] **Portainer:** acrescente `JUDGE_TRANSCRIBE=1` e `JUDGE_DISABLE_THINKING=1`.
- [ ] **Portainer:** *Pull and redeploy* (marque *Re-pull image* e *Force redeployment*).
- [ ] **VM:** `curl http://192.168.0.160:8020/healthz`
  - Esperado: `{"status":"ok"}`.
- [ ] **VM:** `docker logs --tail 30 laya-cypress`
  - Esperado: sem `Traceback` e sem `ValueError`. Um `could not convert string to float` indica variável numérica vazia: corrija no compose.

## Fase 2: testar o serviço sozinho (15 min)
- [ ] **VM:** copie 5 a 8 fotos de teste para uma pasta temporária, por exemplo `/tmp/casos`. Use fotos que você possa apagar depois.
- [ ] **VM:** rode (se o repositório não estiver na VM, copie só `tests/e2e/testar_casos.py`):
  ```bash
  API_KEY='<a chave do serviço>' python3 tests/e2e/testar_casos.py --pasta /tmp/casos
  ```
  - Esperado: uma linha por foto, sem erro de conexão.
- [ ] Confira, por foto:

  | O que olhar | Esperado |
  |---|---|
  | Canhoto assinado nítido | `tipo=canhoto`, `assinatura=True/95`, `inconclusivo=False` |
  | Carimbo sozinho | `assinatura=False` ou `inconclusivo=True` com `assinatura_nao_confirmada` |
  | Foto deitada ou de cabeça para baixo | `rotacao` 90, 180 ou 270 e `ocr` acima de 100 |
  | Foto sem texto (tira cortada) | `inconclusivo=True`, nunca `assinatura=True` com `campos` vazio |
  | Coluna `juiz` | `ok` ou `SIGNED`/`UNSIGNED`; **nenhuma** como `falha` |
- [ ] **Se aparecer `falha` no juiz:** é tempo. Confira `JUDGE_DISABLE_THINKING=1` no stack e tente de novo. Se persistir, pare aqui e me envie `tentativas[0].erro` do `_meta`.
- [ ] **VM:** apague as fotos de teste: `rm -r /tmp/casos`.

## Fase 3: n8n, na cópia do fluxo (1 h)
Cada passo deve ser testado antes do próximo.

- [ ] **3.1 `onError` do serviço.** Nó `sombra_laya_cypress` → *Settings* → *On Error* = *Continue (using error output)*. Ligue a saída **de erro** ao nó `reanexar_imagem`.
  - Teste: **VM** `docker stop laya-cypress`, envie uma foto pelo WhatsApp de teste, o Gemini deve responder; depois `docker start laya-cypress`.
- [ ] **3.2 `reanexar_imagem`** (nó Code):
  ```javascript
  return [{ json: {}, binary: $('HTTP Request3').first().binary, pairedItem: { item: 0 } }];
  ```
- [ ] **3.3 IF `inconclusivo`**, logo depois do serviço: `{{ $json._meta.inconclusivo }}` *is true*. `true` → `reanexar_imagem` → `AI Agent1`. `false` → `ajustar_resposta`.
  - Teste: uma foto assinada nítida segue pelo `false`; uma foto de pedido segue pelo `true`.
- [ ] **3.4 `usar_servico`:** IF depois de `HTTP Request3`, valor fixo `true`. `false` manda tudo ao `AI Agent1` (reversão).
- [ ] **3.5 If7:** na condição que libera assinatura >= 90, acrescente (AND) `{{ $json.emitente.cnpj }}` e `{{ $json.nota_fiscal.numero }}` *is not empty*.
  - Teste: resposta de teste assinada sem CNPJ ou número **não** deve passar.
- [ ] **3.6 Validação da chave do Gemini.** Depois do `AI Agent1`, um nó Code:
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
  Trate `chave_valida = false` (limpar a chave ou revisão humana).
- [ ] **3.7 Amostra para comparação:** IF `amostra_gemini` com `{{ Number($('WhatsApp Trigger').item.json.contacts[0].wa_id.slice(-2)) < 10 }}`, no ramo `false` do 3.3 em paralelo a `ajustar_resposta` → `reanexar_imagem_amostra` → `AI Agent1` (amostra) → `sombra_referencia_gemini`. A resposta da amostra **não** vai ao cliente.
- [ ] **3.8 `""` ou `null`.** No `ajustar_resposta`, fixe duas entradas de teste (*Pin data*): `{"output":"{\"nota_fiscal\":{\"numero\":\"\"}}"}` e a mesma com `null`. Execute e compare a saída.
  - Se iguais: nada a mudar. Se `null` quebrar: mantenha `""`. Anote o resultado.

## Fase 4: pasta compartilhada n8n e trainer (15 min)
- [ ] **VM:** crie a pasta com o trainer (uid 10002) como dono:
  ```bash
  sudo mkdir -p /opt/laya-trainer/data/inbox
  sudo chown -R 10002:10002 /opt/laya-trainer
  sudo chmod 750 /opt/laya-trainer/data /opt/laya-trainer/data/inbox
  ```
- [ ] **VM:** dê acesso ao n8n pelo uid anotado na Fase 0 (`sudo apt-get install acl` se faltar `setfacl`):
  ```bash
  sudo setfacl -m u:<uid_do_n8n>:rwx,d:u:<uid_do_n8n>:rwx /opt/laya-trainer/data/inbox
  ```
- [ ] **Portainer:** no stack do n8n, acrescente o volume `/opt/laya-trainer/data/inbox:/inbox` e faça *redeploy*.
- [ ] **n8n:** depois de `sombra_referencia_gemini` (reserva e amostra), um nó *Read/Write Files from Disk*, operação *Write File to Disk*:
  - *File Path and Name:* `/inbox/{{ <o mesmo id enviado ao sombra_referencia_gemini> }}.jpg`
  - *Input Binary Field:* o campo binário da foto (confira o nome no `reanexar_imagem`).
  - Ligue *Continue On Fail*.
  - O id deve usar só `A-Z a-z 0-9 _ -` e ter até 64 caracteres. Ids com `.` ou `=` são descartados pelo trainer.
- [ ] **Teste:** envie uma foto que caia na reserva. **VM:** `ls -l /opt/laya-trainer/data/inbox`
  - Esperado: um arquivo `<id>.jpg` aparece.

## Fase 5: subir o trainer (30 min)
- [ ] **Portainer:** *Stacks → Add stack → Repository*. Repositório do projeto, *Compose path* `trainer/compose.yaml`.
- [ ] **Portainer:** variáveis do stack (nunca no Git):

  | Variável | Valor |
  |---|---|
  | `TRAINER_ID_SECRET` | 16+ caracteres aleatórios. **Guarde:** trocar muda os pseudônimos |
  | `KAGGLE_USERNAME` | seu usuário do Kaggle |
  | `KAGGLE_API_TOKEN` | o token **novo** (o antigo deve estar revogado) |
  | `TRAINER_KINDS` | `visao` |
  | `HERMES_URL` / `HERMES_MODEL` | endereço do Hermes (só mensageiro) |
  | `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`, `REPORT_TO` | e-mail do relatório |
  | Temporárias: `TRAINER_WINDOW=00:00-23:59` e `TRAINER_MIN_NEW_SAMPLES=4` | para a primeira rodada de teste |
- [ ] **Portainer:** *Deploy the stack*.
- [ ] **VM:** `docker ps --filter name=laya-trainer`
  - Esperado: `Up` e, depois de ~1 min, `(healthy)`.
- [ ] **VM:** `docker logs -f laya-trainer`
  - Esperado: linhas `caixa de entrada: N processadas, ...`. Sem `ConfigError`.
- [ ] Para o job rodar, precisa haver pelo menos **2 fotos assinadas e 2 não assinadas** em que vLLM e Gemini deram o mesmo voto. Deixe o fluxo acumular.
- [ ] Quando o job subir no Kaggle, acompanhe pelo site (Code → seus kernels) e pelo log. Um job leva de minutos a horas.
- [ ] **Esperado ao fim:** e-mail de relatório. Se o gate não promover a versão, o log diz por quê: é um resultado válido.
- [ ] **Portainer:** remova `TRAINER_WINDOW` e `TRAINER_MIN_NEW_SAMPLES` temporárias e faça *redeploy*.

## Fase 6: ligar para valer (15 min)
- [ ] **n8n:** com os testes das Fases 3 e 4 verdes, troque o fluxo de produção pela cópia (ou aplique as mesmas alterações no original).
- [ ] **n8n:** deixe `usar_servico = true`.
- [ ] Envie **uma foto real** pelo WhatsApp e confirme a resposta ao cliente.

## Fase 7: teste final com as notas problemáticas (30 min)
**VM:** `testar_casos.py` com cada foto, conferindo:

| Nota | Esperado |
|---|---|
| 817663 (deitada sobre cenouras) | `rotacao` 90 ou 270, `ocr` > 40 |
| 817937 (amassada, sem chave) | não sai como `pedido`; se for à reserva, tem `motivos_inconclusivo` |
| 607345 (só carimbo) | `assinatura=False` ou `inconclusivo` |
| 413886 | juiz respondeu, ou `erros` no log com o motivo |
| 36342, 240065, 818915 (assinadas) | `assinatura=True`, ou `inconclusivo` com motivo claro |
| Serviço parado | Gemini responde |

Para medir o conjunto, depois de um dia de uso:
```bash
python3 tests/e2e/agregar_decisoes.py /opt/laya-cypress/dados/decisoes.jsonl --desde <data>
```
Esperado: `sem_capacidade` e `falha` do juiz baixos, `p90` de `latencia_ms` abaixo de 25000, 503 raros.

## Se algo der errado
| Sintoma | O que olhar |
|---|---|
| Serviço não sobe | `docker logs laya-cypress`; variável numérica vazia (`float("")`) ou `API_KEY` ausente |
| Tudo `inconclusivo` | Juiz fora: `JUDGE_URL`/`JUDGE_MODEL` e a conexão com o vLLM |
| Juiz em `falha` | Tempo: `JUDGE_DISABLE_THINKING=1`; veja `tentativas[0].erro` |
| Juiz em `sem_capacidade` | vLLM ocupado: aumente `JUDGE_WAIT_SECONDS` (8) ou o limite do vLLM |
| Fila cheia (503) | `MAX_CONCURRENCY` e `QUEUE_TIMEOUT` |
| `inbox` vazio | Nó de gravação do n8n, o volume `/inbox` e as permissões (ACL) |
| Trainer ocioso | Menos de 2 fotos por classe com consenso; veja `caixa de entrada` no log |
| Job do Kaggle falha | Log do trainer e do kernel no site; token novo e `KAGGLE_USERNAME` |

## Reversão (a qualquer momento)
- **n8n:** `usar_servico = false` devolve todo o tráfego ao Gemini. Nada muda no serviço.
- **Trainer:** pare o stack. Nada do serviço depende dele.
- **Serviço:** remova `JUDGE_DISABLE_THINKING` ou `JUDGE_READ_KEY` do stack e faça *redeploy*.
