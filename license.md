# Licença, base legal e uso de dados — LAYA CYPRESS

## 1. Código
- Código do LAYA CYPRESS: [licença do código ou "uso interno, todos os direitos reservados a <empresa>"].
- O kernel de treino de visão usa `ultralytics` (YOLOv8), sob **AGPL-3.0**. Parecer aprovado em 16/11/2026 por [responsável]. Os pesos treinados e o código do kernel seguem a AGPL-3.0 [ou: licença comercial Ultralytics nº ...].

## 2. Dados tratados (LGPD)
| Dado | Origem | Finalidade |
|---|---|---|
| Foto do canhoto/DANFE (com assinatura manuscrita) | WhatsApp, via n8n | Confirmar a entrega e a assinatura |
| CNPJ do emitente, número da NF, chave de acesso | Leitura da foto | Validar o documento |
| Tira da assinatura anonimizada | Gerada pelo trainer | Treino contínuo do modelo |

## 3. Base legal
- Controlador: [razão social, CNPJ]. Encarregado (DPO): [nome, contato].
- Base legal: [art. 7º, inciso ... da Lei 13.709/2018 — ex.: V (execução de contrato) ou IX (legítimo interesse, com LIA registrado em ...)].
- Validação: [responsável jurídico], 16/11/2026.

## 4. Treino contínuo e transferência
 O pacote de treino leva a metade de cima da foto (faixa do canhoto e bloco do destinatário), com identificador pseudonimizado (HMAC com `TRAINER_ID_SECRET`). O bloco do destinatário em geral é de pessoa jurídica (razão social, CNPJ, endereço); quando o destinatário for pessoa física, nome e CPF também entram. Não leva telefone, a foto inteira nem os itens da nota.

## 5. Retenção
| Item | Onde | Prazo |
|---|---|---|
| Log de decisões (`decisoes.jsonl`) | VM, `/opt/laya-cypress/dados` | [prazo] |
| Backup de job que falhou | VM, `/opt/laya-trainer/data/backups` | Até a retomada, ou no máximo [prazo] |
| Versões de modelo | VM, `/opt/laya-trainer/data/registro` | [prazo] |

## 6. Direitos do titular
Pedidos de acesso, correção ou eliminação: [canal]. Eliminar uma foto exige remover o registro do log e de qualquer backup pendente.
