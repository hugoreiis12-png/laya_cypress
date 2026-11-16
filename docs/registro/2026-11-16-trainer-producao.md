# Trainer em produção: falha, alerta, versões e checklist — 2026-11-16

Complementa `2026-11-06-spec-laya-trainer.md` (§7, §9 e fatia 8) e os guias de 09/11.

## Decisões de 16/11
| Tema | Antes (spec §7) | Agora |
|---|---|---|
| Falha de job | Repetir com recuo até `TRAINER_MAX_RETRIES`, depois seguir para o outro tipo | **Para tudo.** O job vai para backup no volume, o estado fica `parado_por_falha`, sai um alerta por e-mail e o processo termina (código 0). Nada roda até a retomada manual |
| Cota do Kaggle | Agenda a próxima janela | Igual: cota esgotada é agendamento, não falha |
| Dados de uma classe só | O job abria e falhava (`poucas classes…`), parando o trainer | O tipo não conta amostras novas enquanto não houver 2 classes com ≥2 exemplos cada; o trainer espera, sem job |
| Relatório sem Hermes | Não saía e-mail | Sai com o texto gerado por código (`fallback_text`) |
| Falha de SMTP no relatório | E-mail perdido | Tentado de novo a cada 1 h até sair; uma vez por versão |
| Rollback | Só existia no código | Comando `control rollback <tipo>` |

`TRAINER_MAX_RETRIES` deixou de existir. Um `estado.json` antigo continua legível.

## Recorte da tira (decidido em 16/11)
A tira enviada ao Kaggle passou de 22% para a **metade de cima** da foto em pé, o mesmo recorte da segunda olhada do juiz. Com 22%, fotos com margem acima do papel perdiam a assinatura, e a tira entrava no treino com o rótulo "assinado" e sem a assinatura.
- A metade inclui o bloco do destinatário (razão social, CNPJ e endereço). Em geral é uma empresa; quando for pessoa física, entram nome e CPF.
- Registrado no `license.md` (seção "Treino contínuo e transferência").

## Backup e retomada
- **Onde fica:** `/opt/laya-trainer/data/backups/<tipo>-<ts>/`, com `falha.json` (motivo, dataset, kernel, amostras) e a pasta do job (`pacote/`, `kernel/`, `saida/` se houver).
- O dataset no Kaggle é apagado mesmo na falha (§5.4). O pacote anonimizado fica só no volume.
- **Retomar** (VM .160, com o trainer parado):
  ```bash
  docker run --rm -v /opt/laya-trainer/data:/srv/trainer laya-trainer:local python -m trainer.app.control retomar
  docker start laya-trainer
  ```
  O próximo job reusa o pacote do backup quando ele existe; senão monta um novo. O backup é apagado quando esse job termina.
- **Política de reinício:** `restart: on-failure`. A saída limpa da parada mantém o container desligado; uma queda de verdade reinicia.

## Comandos de operação
Com o trainer rodando use `docker exec laya-trainer python -m trainer.app.control <comando>`; parado, use o `docker run` acima.
| Comando | O que faz |
|---|---|
| `status` | Fase, job em andamento, parada por falha e retomada pendente |
| `versoes [visao\|laya]` | Versões com status, erro, hash e a atual |
| `rollback <visao\|laya>` | Reverte a última versão em sombra; a anterior volta a ser a atual |
| `retomar` | Libera o trainer parado por falha |

## Correções encontradas no teste de fumaça
1. **O trainer não subia:** o volume `/srv/trainer` cobria o código, copiado no mesmo caminho. O código foi para `/app`.
2. **`kaggle` não importava:** faltava `six`. Acrescentado e a biblioteca fixada em `2.2.5`.
3. **Kaggle 2.x:** criação de dataset e envio de kernel devolvem o erro na resposta; antes passavam batido. `dataset_delete` recebe dono e slug e pedia confirmação: a exclusão falharia sempre e as tiras ficariam na conta.

## Checklist para ligar
**Decisões (spec §11):**
- [x] Base legal LGPD definida (16/11); termos em `license.md` — **o arquivo ainda não existe no repositório**.
- [x] AGPL do YOLOv8 aceita para o kernel de visão (`ultralytics`), 16/11.
- [x] Token novo (`KGAT_`) no `.env` como `KAGGLE_API_KEY`; o compose aceita esse nome. Falta `KAGGLE_USERNAME`.

**Dados chegando:**
- [ ] `DECISION_LOG_PATH=/srv/dados/decisoes.jsonl` e `SIGNATURE_ENGINE=juiz` no serviço.
- [ ] Referências do Gemini gravadas. Na VM: `grep -c '"evento": "referencia"' /opt/laya-cypress/dados/decisoes.jsonl` deve crescer. O `sombra_referencia_gemini` do fluxo versionado (04/11) manda `{tipo_documento, possui_assinatura, contexto}` e a API exige `{"id", "origem", "output"}`; se der 0, o corpo certo é `{{ { id: $('id_foto').item.json.id, origem: 'gemini', output: $json.output } }}`.
- [ ] Caixa de entrada: o n8n de produção roda numa VPS (Hostinger) e não alcança o disco desta VM, então **o serviço `laya-cypress` deixa a foto na caixa** (`TRAINER_INBOX_DIR=/srv/inbox`, volume `/opt/laya-trainer/data/inbox:/srv/inbox`). O nó `gravar_foto_laya_trainer` do n8n sai. O serviço e o trainer rodam com o mesmo uid (10002).

**Trainer:**
- [x] Rede: o trainer não depende mais de `laya_default`; usa a `bridge` padrão do Docker, porque a VM não tem sub-rede livre para uma rede da stack (só fala com Kaggle e SMTP).
- [ ] Variáveis do stack: `TRAINER_ID_SECRET` (≥32 caracteres), `KAGGLE_USERNAME`, `REPORT_TO`, `TRAINER_KINDS=visao`; token e SMTP já vêm do `.env` (`SMTP_USERNAME` também é aceito). `HERMES_URL` opcional. Como o compose fica em `trainer/`, suba com `docker compose --env-file ../.env up -d`.
- [ ] Deploy e `docker ps`: `(healthy)`.
- [ ] Autenticação: `docker exec laya-trainer python -c "from kaggle.api.kaggle_api_extended import KaggleApi as K; a=K(); a.authenticate(); print('ok')"`.
- [ ] Primeira rodada com `TRAINER_WINDOW=00:00-23:59` e `TRAINER_MIN_NEW_SAMPLES=4`; depois remova.
- [ ] Teste da parada: com o trainer no ar, `docker exec laya-trainer python -m trainer.app.control status`.

## Ainda fora
- ~~Servir a versão em sombra ao lado da produção (fatia 10).~~ Feito para `visao`: ver `2026-11-16-fatia10-laya-visao.md`.
- Pesos do `laya` no formato do `hf-encoder`; por isso `TRAINER_KINDS` começa só com `visao`.
- Nada disto foi executado contra o Kaggle real.
