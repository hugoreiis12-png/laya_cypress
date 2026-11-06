# Motor hf-peft

Motor incluído do ITS-FINE: Transformers + PEFT. LoRA ou QLoRA; dtype de computação
segue capacidade real (BF16 nativo ou FP16 em CUDA). Exportação e avaliação podem
rodar em CPU com RAM suficiente. Treinamento requer GPU NVIDIA ou Mac com chip M
(Metal, LoRA em bf16 a partir do macOS 14; sem QLoRA).

As skills escolhem parâmetros e executam scripts. Use o interpretador registrado
em .its-fine/engine-setup.json. setup.py --install-torch prepara a venv após
autorização de download. Para Ollama/GGUF, --gguf prepara o conversor.

## Fluxo e comandos

1. hardware.py, estimate.py e setup.py: reconhecimento e preparação.
2. ingest.py --fontes <pasta> --saida <rodada>/dataset --modo text|qa.
   QA requer gerador Ollama ou API autorizado. API exige endpoint explícito.
   --anonimizar troca dados pessoais de cada trecho antes do gerador; --nomes
   <pasta> inclui nomes de pessoas. JSONL trazido pelo usuário passa por
   anonymize.py --entrada <jsonl> --saida <rodada>/dataset [--nomes <pasta>].
3. fine.py dataset <jsonl> <rodada>/prepared-v1: deduplicação e split por grupo.
4. check.py --dados <prepared> --modelo <id> --revision <commit> --saida <prepared>/check.json.
5. metric.py --sondas <jsonl rotulado> --saida <rodada>/metric-calibration.json.
6. evaluate.py sem --adapter: baseline na validação, antes do objetivo congelado.
7. engine-spec --probe, approve, submit: sondagem em ID separado; depois treino.
8. evaluate.py --adapter <pasta>: comparação no teste congelado.
9. export.py --device cpu: mesclagem/sanidade; reavaliação com evaluate.py
   --pacote <merged> ou --ollama <nome final>.

Consulte --help em cada script para a sintaxe completa. train.py recusa chamada
direta fora do supervisor. --resume-from-checkpoint só aceita checkpoint com
recibo, arquivos de estado completos e hashes idênticos às entradas.
--start-from (vindo de `start_from` na receita) parte do adapter aprovado de outra
rodada num treino novo; rank, alpha e alvos da receita têm de ser os do adapter.

## Dados e métricas

O journal.sqlite vincula fonte, configuração e pares aceitos atomicamente.
Retomar o mesmo lote não duplica linhas; alterar fontes ou configuração exige
nova saída. Proveniência antiga não é promovida silenciosamente.
QA sempre produz resumo.json, inclusive ao terminar com falhas pendentes.
Resposta e evidência devem ocorrer literalmente na fonte; isso reduz invenções,
mas não valida verdade da fonte nem relação pergunta/resposta.

A anonimização troca CPF e CNPJ (formatados, ou só dígitos com verificador
válido), e-mail, telefone com DDD e CEP formatado por [CPF], [CNPJ], [EMAIL],
[TELEFONE] e [CEP]. Nomes de pessoas viram [PESSOA] por um modelo local de
reconhecimento de entidades (`anonimizacao.modelo_nomes` no engine.json, baixado
por hf-encoder/download.py e conferido pelo lock a cada uso). O modelo erra às
vezes: trocas.jsonl (ingest) e revisao-anonimizacao.jsonl (anonymize.py) listam
cada troca com o original para revisão local, e são tão privados quanto os
documentos. Ids, grupos, papéis e rótulos não são alterados.

PDF é processado por página; sem texto extraível gera diagnóstico de OCR.
TXT/MD/HTML e XML do DOCX são limitados a 64 MiB por documento. Split/journal
usam índices em disco. Acima de 10 mil linhas, IDs/grupos completos ficam em
índices JSONL por split, vinculados por hash e contagens em split-manifest.json.
O artefato usa split_manifest_hash e index; nenhuma lista parcial prova disjunção.

Sondas de metric.py: cada linha possui response (string), accepted_answers
(lista de respostas revisadas) e correct (boolean humano). Exija exemplos certos
e adversariais: negação, troca de relação, números e paráfrases. O relatório
calibra accepted_match; congele SHA-256 com o protocolo. evaluate.py recebe
--calibracao e --calibracao-hash. Correspondência exata é conservadora, não NLI.
Entity recall e token F1 são diagnósticos lexicais; não comprovam fatos ou formato.
Texto contínuo é avaliado por continuação com corte determinístico; não é QA.

## Privacidade e saídas

Os entrypoints de ingest/anonymize/check/metric/entities/evaluate/export verificam política
antes de carregar dados. Capturam stdout/stderr de bibliotecas em
.its-fine/private-logs/ e expõem somente campos numéricos/booleanos permitidos.
Esses logs, JSONL, journal e relatórios detalhados são privados.
As funções Python são APIs internas de executor confiável, não uma sandbox.

allow_data_to_harness, geração externa e transferência são independentes.
require_isolated_executor recusa processamento na conta do harness. Configure um
executor separado com política administrada naquele ambiente; não altere a política
do usuário por conta própria. O hook do Claude não é um controle de OS/Codex.

## Colab

colab.py notebook gera bootstrap sem dados. A sessão é baixada em arquivo local.
Use push --spec, approve --spec --actor, run --spec --run-id e status --run-id.
Cada subcomando recebe --sessao <arquivo>. push/pull recebem --saida; pull exige
--spec-hash. --root antecede o subcomando. Não há run com argv arbitrário nem
logs brutos devolvidos pelo servidor.

Uploads são imutáveis e conferidos por SHA-256; o servidor usa o supervisor
persistente. Reinício do servidor não duplica jobs. Cancelamento/prazo são reais
no processo remoto; desaparecimento da VM exige recuperar cópia autorizada.
Resultados são importados pelo runtime após verificar identidade e hashes.

## Kaggle

kaggle_job.py faz o mesmo pela API oficial do Kaggle, sem túnel: preparar, push,
approve, run, status, acompanhar, pull, cancel e limpar, todos após --root. Inputs
vão para dataset privado, o treino roda em kernel privado com GPU T4 sob o mesmo
supervisor, e o resultado volta com recibo e SHA-256. Não há cancelamento pela API.
Dataset e kernel ficam na conta do usuário até limpar.
