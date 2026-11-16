# Guia: restante do n8n e subida do trainer — 2026-11-09

Pré-requisito já cumprido: deploy do serviço, variáveis do stack e credenciais. Este guia cobre o que falta. n8n e trainer rodam na mesma VM (.160). **Duplique o fluxo do n8n antes de editar.**

## Parte A: n8n

Ordem importa: cada passo deixa o fluxo funcionando antes do próximo.

### A1. Falha do serviço cai na reserva (`onError`)
1. Abra `sombra_laya_cypress` → *Settings* → *On Error* = **Continue (using error output)**.
2. Ligue a saída de **erro** ao nó `reanexar_imagem`.
3. Teste: pare o container (`docker stop laya-cypress`), mande uma foto, confirme que o Gemini responde, e suba de volta (`docker start laya-cypress`).

### A2. `reanexar_imagem` (Code)
```javascript
return [{ json: {}, binary: $('HTTP Request3').first().binary, pairedItem: { item: 0 } }];
```

### A3. IF `inconclusivo`, logo depois do serviço
- Condição: `{{ $json._meta.inconclusivo }}` **is true** (tipo Boolean).
- *true* → `reanexar_imagem` → `AI Agent1`.
- *false* → `ajustar_resposta`.
- O motivo fica em `{{ $json._meta.motivos_inconclusivo }}` (para o log do n8n).

### A4. Chave de reversão `usar_servico`
IF logo depois de `HTTP Request3`: `false` leva a `AI Agent1` (tráfego todo no Gemini). Defina como valor fixo `true` para operar.

### A5. Amostra para comparação (`amostra_gemini`, 10%)
No ramo *false* do A3, em paralelo a `ajustar_resposta`:
`{{ Number($('WhatsApp Trigger').item.json.contacts[0].wa_id.slice(-2)) < 10 }}` → `reanexar_imagem_amostra` → `AI Agent1` (amostra) → `sombra_referencia_gemini`. A resposta da amostra **não** vai ao cliente.

### A6. If7: exigir CNPJ e número
Hoje libera assinatura >= 90 mesmo sem esses dois campos. Acrescente à condição (AND):
- `{{ $json.emitente.cnpj }}` **is not empty**
- `{{ $json.nota_fiscal.numero }}` **is not empty**

### A7. Validar a chave que vem do Gemini (Code, depois do `AI Agent1`)
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
Quando `chave_valida` for `false`, trate o caso (limpar a chave ou revisão humana) em vez de seguir adiante.

### A8. `""` ou `null` no `ajustar_resposta`
O serviço devolve `""` para campo vazio; o contrato do Agent1 pede `null`.
1. No `ajustar_resposta`, fixe (*Pin data*) duas entradas de teste: `{"output":"{\"nota_fiscal\":{\"numero\":\"\"}}"}` e a mesma com `null`.
2. Execute o nó e veja o número de saída. Se `""` e `null` produzem o mesmo resultado, **nada a mudar**. Se `null` quebrar (erro ou "NaN"), mantenha `""`.
3. Me diga o resultado: só então decidimos se o serviço passa a devolver `null`.

### A9. Gravar a foto para o trainer (só quando há resposta do Gemini)
Consenso exige serviço **e** Gemini, então grave onde o Gemini respondeu: depois de `sombra_referencia_gemini` (reserva e amostra).
1. Um nó **Read/Write Files from Disk** → operação *Write File to Disk*.
   - *File Path and Name:* `/inbox/{{ <mesmo valor do campo id enviado ao sombra_referencia_gemini> }}.jpg`
   - *Input Binary Field:* `data` (o campo binário da foto; confira o nome no `reanexar_imagem`).
2. O nome do arquivo deve ser **igual ao `id` do `contexto`** enviado ao serviço, e casar com `[A-Za-z0-9_-]{1,64}`. Ids com `.` ou `=` (como alguns ids de mensagem do WhatsApp) são descartados pelo trainer: se for o caso, me avise.
3. Ligue o nó com *Continue On Fail* para que uma falha de disco não quebre a resposta ao cliente.

## Parte B: pasta compartilhada n8n e trainer (VM .160)

```bash
# 1) descubra o usuário do container do n8n
docker exec <nome_do_n8n> id            # anote o uid, ex.: 1000
# 2) crie a pasta; o trainer (uid 10002) é o dono, o n8n recebe acesso por ACL
sudo mkdir -p /opt/laya-trainer/data/inbox
sudo chown -R 10002:10002 /opt/laya-trainer
sudo chmod 750 /opt/laya-trainer/data /opt/laya-trainer/data/inbox
sudo setfacl -m u:<uid_do_n8n>:rwx,d:u:<uid_do_n8n>:rwx /opt/laya-trainer/data/inbox
```
Sem `setfacl`: `sudo apt-get install acl`. Depois monte no container do n8n (stack do n8n): `/opt/laya-trainer/data/inbox:/inbox` e reinicie o n8n.

Teste: grave uma foto pelo n8n e confira com `ls -l /opt/laya-trainer/data/inbox`. O trainer precisa ler (outros: `r`) e apagar (é dono da pasta).

## Parte C: subir o trainer

1. **Portainer → Stacks → Add stack → Repository**: repositório do projeto, *Compose path* `trainer/compose.yaml` (o contexto de build é a raiz do repositório).
2. **Variáveis do stack** (nunca no Git):

| Variável | Valor |
|---|---|
| `TRAINER_ID_SECRET` | 16+ caracteres aleatórios; **guarde**: trocar muda os pseudônimos e os testes congelados |
| `KAGGLE_USERNAME` | usuário da conta do Kaggle |
| `KAGGLE_API_TOKEN` | o token **novo** (o antigo foi exposto e deve estar revogado) |
| `TRAINER_KINDS` | `visao` (acrescente `laya` só depois de medir; ver limites na spec do trainer) |
| `HERMES_URL`, `HERMES_MODEL` | endereço do Hermes (só mensageiro) |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`, `REPORT_TO` | e-mail do relatório |

3. **Primeira rodada de teste (pequena):** acrescente temporariamente `TRAINER_WINDOW=00:00-23:59` e `TRAINER_MIN_NEW_SAMPLES=4`. Precisa de pelo menos 2 fotos assinadas e 2 não assinadas com voto igual de vLLM e Gemini.
4. **Acompanhe:** `docker logs -f laya-trainer`. Esperado: "caixa de entrada: N processadas", depois o job no Kaggle e a decisão do gate. O container também deve ficar `healthy` (`docker ps`).
5. **Depois do teste:** remova as duas variáveis temporárias (voltam a 22:00-06:00 e 200 amostras).

## Verificação final
- [ ] `docker stop laya-cypress` e foto de teste: o Gemini responde.
- [ ] Foto inconclusiva vai à reserva e a resposta volta ao cliente.
- [ ] Resposta assinada sem CNPJ ou número **não** é liberada (If7).
- [ ] Arquivo aparece em `inbox/` e some depois do ciclo do trainer (original apagada).
- [ ] `/opt/laya-trainer/data/tiras/` guarda só `<pseudônimo>.jpg`, nunca o id cru.
- [ ] Um job completo no Kaggle chegou ao relatório por e-mail.

## Reversão
- n8n: `usar_servico` = `false`.
- Trainer: parar o stack. Nada do serviço depende dele.
