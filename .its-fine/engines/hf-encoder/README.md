# Motor hf-encoder

Motor de decisão tipada do ITS-FINE. Um encoder lê um estado (texto, JSON ou lista
de turnos) junto com cada pergunta do projeto e responde por escolha entre opções
escritas na própria sequência, pontuadas em marcadores. Três tipos de pergunta:
`choice` (uma opção entre várias), `noul` (sim ou não com probabilidade) e `score`
(nível ordinal com valor esperado). Não gera texto.

Serve qualquer encoder do Hugging Face com token de máscara (BERT, RoBERTa,
DeBERTa, ModernBERT, mmBERT...). O layout da pasta decide o ponto de partida:

- **encoder comum** (`config.json` na raiz): o encoder vem do checkpoint e as
  cabeças (embedding de tipo, dois blocos extras, pontuador) nascem novas, com
  semente fixa;
- **checkpoint de decisão** (`rl_agent_config.json`, formato do Laya, ou
  `itsfine-model.json`): encoder e cabeças já treinados, carregados de forma estrita;
- **pacote** exportado pelo motor (`itsfine-package.json`).

Em todos, LoRA no encoder e cabeças treináveis; a cabeça de ação (`act_head`) fica
congelada. `catalogo.json` traz sugestões com idioma, tamanho e quando usar; não é
lista fechada.

As perguntas são do projeto, em qualquer idioma, declaradas em `task.json` junto
com a saída que decide e a política (limiar alto, limiar baixo e ação do meio,
normalmente "revisar"). Classificação de classes fixas é uma pergunta `choice`;
use `presets/classificacao.json`.

## Fluxo e comandos

1. `setup.py`: prepara a mesma venv do hf-peft; faixas divergentes recusam.
   `estimate.py --vram-gb <n> --perguntas-por-estado <k>`: custo é estados vezes
   perguntas vezes comprimento; estimativa, a sondagem decide.
2. Modelo: `download.py --id <repo> [--revision <branch ou commit>] --saida
   .its-fine/models/<nome>` mostra o plano e sai com código 2; com `--autorizado`
   baixa. A revisão vira commit; entram configurações, tokenizer e pesos
   (safetensors, ou `pytorch_model*.bin` quando não há safetensors). Cada arquivo é
   conferido contra o próprio Hub (sha256 do LFS ou sha1 do blob git); divergência
   apaga tudo. O fim grava `itsfine-download.json` com o commit e o sha256 de cada
   arquivo. Pasta baixada por outro meio: `--conferir` confere contra o Hub e grava
   o lock sem baixar de novo. O motor confere a pasta contra o lock antes de
   treinar, avaliar ou exportar e nunca grava nela. Repositório fechado usa `HF_TOKEN`.
3. `ingest.py --fonte <jsonl> --mapeamento <json> --saida <pasta>`: mapeamento
   declarativo (campos ou linhas `messages` com JSON no assistant) para linhas
   tipadas `{id, group_id, state, answers}` e `task.json`. Em `state.campos`, uma
   lista copia os nomes das colunas; um objeto renomeia (`{"mensagem": "texto"}` lê a
   coluna `texto` como `mensagem`), para treino e teste terem o mesmo estado.
4. `fine.py dataset rows.jsonl <run>/prepared-v1 --tarefa task.json [--teste <jsonl>]
   [--estratificar <saida>]`: quatro partições por grupo (treino, validação,
   calibração, teste); teste declarado fica intacto. `--estratificar` põe cada opção
   da saída em todas as partições quando há grupos suficientes, sem quebrar a
   disjunção; o que faltar sai em `summary.json`. Respostas conflitantes bloqueiam.
5. `check.py --dados <prepared> --checkpoint <pasta> --saida <prepared>/check.json`:
   marcadores, truncamento, comprimentos e suporte por opção.
6. `metric.py --tarefa <task.json> --saida <run>/metric-calibration.json`: casos de
   resultado conhecido do avaliador e validação da política.
7. `evaluate.py --teste <validation.jsonl> --checkpoint <pasta> --checkpoint-id <id>
   --checkpoint-revision <commit> --calibracao-dados <calibration.jsonl> --saida
   <baseline>`: baselines antes do objetivo.
8. `engine-spec --probe`, `approve`, `submit`: sondagem em ID separado; depois o
   treino. `train.py` recusa chamada fora do supervisor.
9. `evaluate.py --teste <test.jsonl> --checkpoint <pasta> --adapter <adapter>
   --calibracao-dados <calibration.jsonl> --limites '<json>' --saida <pasta>`.
10. `export.py --checkpoint <pasta> --adapter <adapter> --calibracao
    <calibracao-candidato.json> --saida <pacote>`; reavaliação com `evaluate.py
    --pacote <pacote>` e pelo servidor (`serve.py`, `evaluate.py --servidor`).
11. Consumidores: `predict.py --pacote <pacote> --estado "<texto ou JSON>"` e
    `serve.py --pacote <pacote>` (POST /v1/decide, GET /v1/package), em CPU.

Consulte `--help` em cada script. `--checkpoint-id` e `--checkpoint-revision`
conferem a pasta contra o lock do download.

Continuação de um modelo aprovado: `dataset <novos.jsonl> <run>/prepared-v1 --base
<mãe>/prepared-v1` e receita com `start_from`; o train.py recebe `--start-from`,
carrega LoRA e cabeças da mãe e treina do passo zero.

## Dados, calibração e métricas

Linha tipada: `state` (texto, objeto ou lista), `answers` por nome de pergunta
(opção para `choice`, booleano para `noul`, índice do nível para `score`); chave
ausente é rótulo ausente e fica mascarado na loss. `meta` nunca entra na sequência.
Cada estado vira uma sequência por pergunta rotulada; o split conta estados, não
sequências.

Loss: entropia cruzada sobre as opções de cada pergunta, com peso por pergunta.
Calibração: uma temperatura por (tipo, número de opções), ajustada na partição de
calibração com suporte mínimo (30 por padrão); bucket sem suporte fica em 1,0 e é
listado.

Baselines: `baseline_frequencia` responde cada opção pela frequência na partição
de calibração, sem ler o texto; é a referência honesta quando as cabeças nascem
novas. `baseline_calibrado` é o modelo de partida com temperaturas reajustadas;
num checkpoint de decisão ele já responde, num encoder comum ele é o acaso. O
original sem calibração aparece só como diagnóstico.

Métricas por saída: log-loss, Brier, ECE (15 intervalos) com curva de
confiabilidade, acurácia e, em `score`, MAE na escala de níveis. Na saída que
decide: matriz das decisões automáticas no ponto de operação (tp, fp, tn, fn),
cobertura automática, encaminhados para revisão, fn liberados e a leitura
"revisar como negativo". Limites do protocolo (`max_taxa_fn_liberados`,
`min_cobertura_automatica`, `max_ece`, `min_acuracia`, `max_log_loss`,
`min_recall_revisar_negativo`, `max_fp`, `max_fn_liberados`) são obrigatórios na
aprovação.

## Pacote e privacidade

O pacote tem o layout próprio do motor, qualquer que seja a origem:
`itsfine-model.json` (configuração e temperaturas ajustadas), `model.safetensors`
mesclado em fp16, `encoder/config.json`, `tokenizer/`, mais `task.json`,
`calibration.json`, `policy.json` e `itsfine-package.json` com hash de cada arquivo
e `package_id`. `serve.py` escuta em loopback; outro endereço exige
`allow_data_transfer` com destino listado.

Os entrypoints verificam a política antes de ler dados, capturam a saída das
bibliotecas em `.its-fine/private-logs/` e devolvem só agregados. Os arquivos
JSONL, logits e relatórios detalhados (`avaliacao.json`) são privados.

## Certificação

`scripts/certify_encoder.py --id <repo>` (no repositório, não distribuído) parte
de um download novo conferido contra o Hub, roda o fluxo completo pelo supervisor
na GPU de referência e grava `certification.json`. Em checkpoint de decisão roda
antes a paridade contra a referência do Laya numa cópia da pasta, porque a
referência regrava `tokenizer_config.json`. `verify_npm_package.py` exige um
encoder comum e um checkpoint de decisão com paridade, ambos vindos do download.
`verify.py` do projeto lê esse arquivo sem baixar nada.

Fora desta versão: perfil de classificação separado, ONNX e int8, RLCD,
`act_head` treinável, reranker e ajuste completo sem LoRA. Atribuição e
licença do código de referência em `third_party/laya/`.
