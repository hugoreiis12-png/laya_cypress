# Contratos dos especialistas

Contrato versão 1, implementação 0.3.0. As 14 skills são instruções operacionais
instaladas em .agents/skills e .claude/skills. Fonte: scripts/build_skills.py.
Comando explícito tem precedência sobre roteamento inferido.

| Skill | Entrada e resultado | Próximo passo |
|---|---|---|
| fine-help | Pergunta + estado em somente leitura; esclarece sessão | Sugere ação sem iniciar pipeline |
| fine-start | Intenção, fontes e destino; plano aprovado e intent.json | Ambiente preliminar |
| fine-environment | Hardware e intenção; environment-draft.json, venv | Dataset |
| fine-dataset | Fontes → journal/JSONL/splits → check após modelo | Modelo ou métrica |
| fine-model | Intenção, dados, hardware; model-draft.json, revisões fixadas | Tokenização |
| fine-metric | Validação, modelo e sondas humanas; calibração + baseline | Objetivo |
| fine-objective | Pré-condições prontas; congela protocolo/objetivo e publica rascunhos | Receita |
| fine-config | Artefatos publicados + defaults; receita e propostas por hash | Sondagem/treino |
| fine-train | Proposta aprovada; supervisor para probe/treino com IDs diferentes | Avaliar ou reconciliar |
| fine-colab | Sessão em arquivo + destino autorizado; upload/job/retorno verificado | Avaliar resultado importado |
| fine-resume | Observação real de job/checkpoint; reconcilia ou abre nova identidade | Avaliar ou retomar |
| fine-evaluate | Teste/protocolo/candidato; decisão e limites das métricas | Exportar ou nova hipótese |
| fine-export | Adapter e consumidor; pacote + avaliação na quantização final | Entregar |
| fine-deliver | Pacote verificado + destino autorizado; recibo | Conclusão verificável |

Rascunhos em _itsfine_runs/<run_id>: intent.json, environment-draft.json,
model-draft.json, metric-calibration.json e baseline/resumo.json. Eles não são
artefatos publicados. O objetivo é congelado depois dos dados/modelo/baseline.
A ordem de publicação permanece objective → environment → model → dataset →
recipe → run → evaluation → export → delivery.

Veja [schema](artifact.schema.json) e [contratos](README.md) para campos/hashes.
O motor da rodada (hf-peft ou hf-encoder) é escolhido em fine-start pela forma do
resultado e gravado em intent.json e na receita; cada skill usa os scripts do
motor da rodada. No hf-encoder, fine-model importa o checkpoint suportado,
fine-dataset converte por mapeamento e separa quatro partições, fine-metric
valida o avaliador e calibra o baseline, fine-objective congela política e
limites, fine-evaluate calibra o candidato, fine-export entrega o pacote com
predict e servidor, e fine-colab recusa (sem Colab nesta versão).
ID de artefato é único na rodada. Revisões publicadas são imutáveis. Job local
ou remote-job.json congela entradas. Estado de sessão é distinto do estado do job.

## Autorizações e evidências

Reutilize autorização válida para mesma ação/destino/conteúdo. Receita não libera
upload/publicação. Políticas são editadas pelo usuário. Permissões de leitura,
geração externa e transferência são independentes e negadas quando ausentes.
Sem permissão de conteúdo, apenas agregados permitidos; não leia JSONL/journal/logs.

RAG está fora do pipeline. Rank, épocas e volume são hipóteses, não garantias.
Token F1/entity recall não avaliam correção factual. accepted_match exige
alternativas revisadas, sondas rotuladas e protocolo congelado; tarefas abertas
podem exigir avaliação humana. Não altere a métrica após ver teste para aprovar.

## Aceite por ambiente

| Ambiente | Evidência necessária antes de declarar verificação completa |
|---|---|
| Local | GPU real, probe/treino/retomada, avaliação e consumidor final |
| Colab | Notebook em VM real, transferência autorizada, supervisor e retorno |
| Corporativo | Documentos inacessíveis ao harness por ACL/VM; testes de leitura, shell, ferramentas e rede separados por host |
| Exportação | Mesmo protocolo no consumidor e quantização efetivamente entregues |

Testes sintéticos em CPU/loopback provam contratos delimitados. Não substituem
essas verificações. Distribuição por manifesto verifica hashes/equivalência e
preserva arquivos do usuário; não certifica isolamento ou qualidade.
