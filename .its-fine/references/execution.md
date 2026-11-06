# Execução ITS-FINE

O runtime supervisiona os motores incluídos, hf-peft (gerativo), hf-encoder
(decisão tipada) e hf-tts (voz). Use python .its-fine/bin/fine.py --root <projeto> <comando>.
O motor é resolvido pelo campo `engine` da receita (ausente significa hf-peft;
nome desconhecido é recusado) num registro fechado, `itsfine/engines.py`;
`engines` lista os instalados e preparados.

## Preparar, aprovar e executar

engine-spec <run_id> --recipe <recipe-engine.json> --data <prepared>
--executor process|docker --timeout <segundos> gera execution-proposed.json
com snapshots de código/runtime, receita e dados por SHA-256. Sondagem usa
--probe e ID próprio. Docker requer --image <imagem@sha256:digest>.

Inputs têm name relativo (inclusive engine/train.py), path relativo ao projeto
e sha256. argv é lista sem shell. {python}, {inputs}, {input:nome} e {output}
são resolvidos pelo supervisor. Inclua pesos locais na imagem aprovada ou inputs
e use o caminho disponível no executor. Docker não baixa pesos durante o job.
O snapshot copia a pasta do motor com subpastas. No hf-encoder a receita aponta a
pasta local do checkpoint (`checkpoint.path`), incluída como `base/` com hash de
cada arquivo; sem ela a proposta falha com CHECKPOINT_NOT_MATERIALIZED. O snapshot
de treino recebe train, validation e task.json; teste e calibração nunca entram.
No hf-tts o snapshot leva train, validation e só os clipes de áudio que essas
linhas citam (`data/audio/<sha256>.wav`); os clipes do teste ficam fora. Motor com
`ambiente: proprio` roda no Python da venv dele (`engine-setup.json` > motor >
`python`); em sessão remota a proposta grava `python: venv:<motor>`, que o
supervisor resolve para `.its-fine/venvs/<motor>/` da sessão, criada no preparo.

approve <spec> --actor <usuário> registra o hash autorizado. Processo exige
allow_process true e --acknowledge-process; não fornece isolamento de filesystem.
Docker exige imagem presente (--pull never), rede desligada, inputs somente
leitura, escrita em outputs/tmp e GPU explicitamente solicitada. Acesso ao daemon
Docker pelo harness não constitui isolamento corporativo dos documentos.

submit <run_id> <spec> persiste intenção antes do spawn. Mesmo ID/hash retorna
o job existente, incluindo estados terminais ou desconhecidos; entradas diferentes
são recusadas. Código/receita/dados são congelados antes de executar.
train.py, inclusive --probe, recusa execução direta fora do supervisor.

status retorna estado sem logs. cancel solicita término; confirmação depende do
estado terminal. Timeout é aplicado pelo supervisor à árvore de processos/container.
Máquina/VM e supervisor precisam continuar ativos. Falha do supervisor deixa estado
desconhecido: reconcilie antes de nova identidade. O adaptador recusa recursos
tarifados sem orçamento verificável do provedor.

Checkpoint retomável exige trainer_state, optimizer, scheduler, RNG, adapter e
itsfine-checkpoint.json com hashes (no hf-encoder, também heads.safetensors e
adapter_config.json). --checkpoint em engine-spec inclui o snapshot, e train.py
passa --resume-from-checkpoint ao treinador. Retomada usa nova identidade,
preservando receita/dados; checkpoint incompleto ou adulterado é recusado.
Continuação é outra coisa: receita com `start_from` faz engine-spec conferir a
avaliação aprovada da rodada mãe, o hash do adapter, o modelo base e a tarefa,
incluir o adapter no snapshot e passar --start-from ao train.py. O treino é novo,
com otimizador e passos do zero; `dataset --base` mantém as partições reservadas.
No Windows, o python.exe de uma venv é um lançador que cria o interpretador real
como filho; o supervisor aceita pai ou avô como identidade do motor.

## Fronteiras de privacidade

Apenas o usuário edita .its-fine/policy.json. Exemplo restritivo:

~~~json
{
  "schema_version": 1,
  "allow_process": false,
  "allow_data_to_harness": false,
  "allow_external_generator": false,
  "external_generator_endpoints": [],
  "allow_data_transfer": false,
  "transfer_destinations": [],
  "allow_colab": false,
  "allow_kaggle": false,
  "protected_data_paths": ["documentos/**"],
  "require_isolated_executor": true
}
~~~

Campos ausentes negam permissões; tipo inválido invalida a política. Geração por
API e transferência têm autorização separada de leitura pelo harness. Destinos
são URLs base exatas, HTTPS fora de loopback. Autorizar /v1 não autoriza outro
host/caminho. Proxies de ambiente e redirects não mudam destinatário implicitamente.
Chaves ficam no ambiente; sessão do túnel fica em arquivo local privado.

Ingest/anonymize/check/metric/entities/evaluate/export verificam política e capturam saídas
de bibliotecas em .its-fine/private-logs. A fronteira CLI retorna whitelist de
campos agregados numéricos/booleanos. JSONL, SQLite, respostas e logs permanecem
privados. Um resumo não permite copiar qualquer arquivo chamado resumo.json:
os campos e a proveniência precisam ser validados pelo executor confiável.

| Ambiente | Controle instalado | Limite |
|---|---|---|
| Claude Code | PreToolUse para Read/Grep/Glob/Bash/edições; padrões privados, aliases/hardlinks e shell geral recusados | Ferramentas alternativas/host não foram certificados; hook não é sandbox |
| Codex | Skills e CLI com controles próprios | Não há hook Claude instalado no Codex; Read/shell do host dependem das permissões de OS |
| Executor de processo | Aprovação por hash, prazo e logs privados | Compartilha permissões da conta que o inicia |
| Docker | Inputs somente leitura, rede desligada, destinos de escrita controlados | Não retira do harness acesso aos arquivos originais no host |
| Executor corporativo separado | Conta/VM e política administradas externamente | Requer provisionamento e teste de ACL/mount/rede pelo administrador |

Para garantia corporativa: mantenha documentos e derivados fora da conta/VM do
harness; não monte a pasta privada nela nem dê acesso ao daemon/credencial do
executor. O administrador opera o plano aprovado no executor, com política própria
e serviço gerador local/endpoint permitido. Retorne apenas agregados validados.
require_isolated_executor recusa executar localmente no harness; não cria essa
infraestrutura nem aceita uma variável de ambiente como atestado de isolamento.

Aceite separado por harness, com documento canário sintético: leitura nativa,
busca, shell, alias, ferramenta alternativa e resposta de programa devem ser
incapazes de revelar conteúdo; API/transferência não autorizadas devem falhar antes
do envio; agregados devem funcionar. Os testes do repositório cobrem hook chamado
diretamente, comandos e cliente/servidor em loopback. Não certificam interceptação
em sessão viva de Claude/Codex nem separação de identidades nesta máquina.

## Colab

colab.py notebook --saida <arquivo.ipynb> embute só runtime/servidor e registra um
identificador de sessão em .its-fine/colab/notebooks/. O usuário abre, seleciona
GPU, executa e confirma com SIM. A célula baixa conexao-itsfine-<sessão>.json e
continua rodando (o Colab não desliga por inatividade). colab.py conectar acha o
arquivo nos Downloads (ou --arquivo <caminho>), confere a sessão no servidor, guarda
em .its-fine/colab/sessao.json (fora do git) e apaga o arquivo baixado.

Use colab.py --root . <subcomando>: push --spec <spec-local> --saida <spec-remota>;
approve --spec <spec-remota> --actor <usuário>; run --spec <spec-remota> --run-id
<id>; status [--run-id <id>]; acompanhar --run-id <id>; cancel --run-id <id>;
pull --run-id <id> --spec-hash <hash aprovado> --saida <zip> [--com-checkpoints].

push exige allow_colab (sessão aberta por notebook deste projeto, túnel
trycloudflare) ou allow_data_transfer com a URL exata em transfer_destinations.
Uploads vão em partes de 16 MB (o túnel corta requisição acima de ~100 s),
endereçados pelo SHA-256 (o que já está na VM não sobe de novo), imutáveis. Status
de job falho traz tipo e arquivo:linha do erro; mensagem só com allow_data_to_harness.
O resultado volta sem os checkpoints por época (hashes em remote_only), em faixas
paralelas. VM perdida responde "inexistente" e marca remote-job.json como perdida. Recibo
.transfer.json vincula destino e hashes. O servidor usa o mesmo supervisor
persistente, prazos e aprovação do runtime local. remote-job.json é gravado antes
do pedido; resposta perdida exige consulta do mesmo ID. Reinício do servidor
mantém jobs; perda da VM perde o disco. Não há promessa de persistência entre VMs.

## Kaggle

kaggle_job.py usa a API oficial do Kaggle (pacote kaggle na venv, instalado por
preparar --autorizado) com a credencial do usuário em ~/.kaggle, que o agente nunca
lê. Não há túnel nem aba aberta. Use kaggle_job.py --root . <subcomando>: preparar;
push --spec <spec-local> --saida <spec-remota>; approve --spec <spec-remota> --actor
<usuário>; run --spec <spec-remota> --run-id <id> [--sem-checkpoints]; status
--run-id <id>; acompanhar --run-id <id>; pull --run-id <id> --spec-hash <hash
aprovado> --saida <zip>; cancel --run-id <id>; limpar --run-id <id>.

push exige allow_kaggle. Os inputs sobem como <sha256>.blob para um dataset privado
itsfine-<hash dos inputs>, criado uma vez e aguardado até ficar pronto; o kernel
confere cada SHA-256 antes de autorizar. approve grava em .its-fine/kaggle/aprovacoes/
o ator e o hash da spec remota; run recusa sem ele. run grava remote-job.json (backend
kaggle) antes do envio e envia um kernel privado por run id, com GPU T4 e internet
para instalar os pacotes; resposta perdida é reconciliada consultando o kernel, nunca
reenviando. O kernel roda o mesmo supervisor, prazo e aprovação do runtime local numa
raiz fora de /kaggle/working e salva ali só itsfine-resultado.zip e itsfine-recibo.json.
O resultado leva os checkpoints por época, porque a VM não sobrevive à sessão;
--sem-checkpoints (sondagem) os deixa de fora. run confere o dataset antes do envio.
timeout_seconds acima de 30600 s é recusado: a sessão com GPU acaba em ~9 h e o
resultado precisa ser empacotado antes.

O agente lê do log só as linhas ITSFINE|, com números de progresso. A API não cancela
execução: cancel devolve o link da página. pull confere run id, spec_hash e SHA-256
do recibo e do zip e importa pelo runtime. Diferente do Colab, dataset e kernel
persistem na conta do usuário; limpar os apaga depois do resultado importado e recusa
se outro job pendente usa o mesmo dataset.

O notebook portátil genérico do runtime (comando notebook) embute inputs e está
limitado a 32 MiB. Use o bootstrap Colab e uploads para conjuntos maiores.
Não compartilhe notebook contendo dados sem autorização de circulação.
SSH (remote submit|status|cancel) continua disponível para destino já preparado;
não provisiona recursos nem transfere inputs.

## Retorno e artefatos

pack exige job terminal. import verifica identidade, hashes, membros ZIP e
traversal antes de gravar sob imports; não executa conteúdo nem promove qualidade.
Hash confiável é o da proposta aprovada, não o fornecido pelo próprio resultado.
Importação não permite revelar respostas/logs ao harness.

Rascunhos permitem ambiente/dados/modelo/calibração/baseline antes do objetivo.
Depois publique artefatos na ordem de dependências. Job local ou remoto iniciado
congela entradas. new, put, artifacts e route usam revisões e locks de kernel.
Não apague locks. Metadados completos não comprovam treinamento ou entrega.

Splits grandes usam índices JSONL em disco e split-manifest.json com hashes,
contagens e declaração de disjunção produzida pela preparação. O artefato dataset
usa split_manifest_hash e index em cada split, com ids/groups vazios. Não corte
listas para simular um dataset menor. O validador de bundle valida metadados;
o executor deve conferir arquivos/índices e hashes reais antes de publicação.

Linhas tipadas (`state` + `answers`) seguem por `itsfine/encoder_dataset.py`:
quatro partições por grupo (train, validation, calibration, test), teste declarado
pelo projeto aceito como partição intacta, respostas conflitantes recusadas,
duplicatas que ligam grupos reconciliadas antes da divisão, suporte por saída e
opção em summary.json. A normalização gerativa e seus hashes não mudam. Os
artefatos desse caminho usam schema_version 2: task_hash em objetivo, dataset e
receita; política, limites obrigatórios e calibração no objetivo; calibration_hash
e package_id em avaliação e exportação. Limite violado reprova; suporte
insuficiente é inconclusivo; paridade não certificada bloqueia o bundle.
