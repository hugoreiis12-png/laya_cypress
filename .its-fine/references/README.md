# Contratos operacionais do ITS-FINE, versão 1

Status: skills e runtime 0.3.0. Rascunhos de intenção, ambiente, dados, modelo, calibração e baseline precedem o congelamento do objetivo. A sequência abaixo descreve a publicação dos artefatos; não exige objetivo congelado antes dessas pré-condições. Nenhuma combinação engine/modelo/GPU é certificada por estes metadados.

## Organização e identidade

O produto instalado usa `.its-fine/` para estado, políticas e runtime; `_itsfine_runs/<run_id>/` para artefatos, criados sob demanda. As specs ficam em `_reversa_sdd/`, que é uma pasta de desenvolvimento do projeto, não um destino de treino.

O comando de entrada é `/fine-start` (ou `$fine-start` no motor que usa essa invocação). Especialistas: `/fine-objective`, `/fine-environment`, `/fine-dataset`, `/fine-model`, `/fine-config`, `/fine-train`, `/fine-evaluate`, `/fine-export` e `/fine-deliver`. `/fine-resume` reconcilia a sessão e a execução. Os contratos por skill estão em [skills.md](skills.md).

Cada artefato tem `schema_version` (1 para o caminho gerativo, 2 para decisão tipada com o motor hf-encoder), `kind`, `id`, `revision` positiva, `input_hashes` e `data`. Um bundle não mistura versões. A versão 2 acrescenta: no objetivo, `task_hash`, saída que decide, política de limiares, `limits` obrigatórios, procedimento de calibração e frações de quatro partições; no modelo, o checkpoint de origem com hashes e `parity`; no dataset, a partição `calibration`, `task_hash` e `declared_test`; na receita, `profile`, `checkpoint_hash` e `task_hash`; na avaliação, `calibration_hash`, `metrics`, `limits_checked` e `support_sufficient`; na exportação, `calibration_hash` e `package_id`. O validador recusa bundle v2 com limite violado (`reprovada`), suporte insuficiente (`inconclusiva`), tarefa divergente entre objetivo, dataset e receita, calibração divergente entre avaliação e exportação, ou paridade não certificada. Os contratos dos arquivos do motor (task, linhas, mapeamento, receita, calibração, predição, pacote) estão em `encoder.schema.json`. As entradas são referenciadas pelo SHA-256 do JSON canônico do artefato completo: UTF-8, chaves ordenadas, sem espaços, sem NaN/Infinity, conforme `artifact_hash` no validador. Hashes de arquivos são calculados sobre os bytes efetivos, sem conversão de encoding. Os hashes não substituem assinatura/autorização nem provam origem confiável.

IDs são únicos por projeto. Publicar uma revisão nunca sobrescreve uma revisão anterior. Revisão de estado usa lock exclusivo, comparação da revisão lida e escrita temporária com rename atômico no mesmo filesystem. Um conflito relê o estado; não usa last-write-wins. O executor remoto deve oferecer a mesma semântica ou recusar execução automática. Essas garantias de persistência pertencem aos futuros adaptadores, não ao validador de JSON.

## Sequência e dependências

Antes da publicação: intent.json → environment-draft.json → dataset/splits → model-draft.json → check.json → metric-calibration.json e baseline/resumo.json. Consulte skills.md para o fluxo operacional atualizado.

1. **Objetivo:** tarefa, métrica, direção, limiar, melhoria mínima contra o base e tolerância de exportação.
2. **Ambiente:** escolha do usuário entre local, Colab/Kaggle e remoto/cloud; diagnóstico do destino, limites de duração/custo e política de localização dos dados.
3. **Dataset inicial:** origem, qualidade, deduplicação e split por grupo/documento/entidade. Preservar originais. Não aplicar template específico de modelo ainda.
4. **Modelo:** revisão imutável de modelo e tokenizer, perfil de engine e parecer no hardware escolhido. Viabilidade pode ser `aprovada`, `reprovada` ou `nao-verificada`; não inventar medição.
5. **Dataset final e configuração:** tokenização/template e truncamento por modelo; congelar protocolo e conjunto de teste antes do treino; proposta LoRA/QLoRA com parâmetros resolvidos e justificativas. Aprovação vinculada ao hash da receita.
6. **Treino:** validar entradas, aprovação, proteção de destinos e orçamento; registrar intenção de disparo antes da chamada ao executor. Observar e reconciliar job por identidade estável.
7. **Avaliação:** comparar base e candidato no mesmo protocolo congelado. Reprovação volta para diagnóstico; não promove silenciosamente o candidato.
8. **Exportação e entrega:** validar pacote, carregamento e inferência no destino; verificar tolerância de qualidade. Entregar localmente ou publicar com autorização específica.

As oito etapas originais são preservadas funcionalmente. Ambiente é pré-condição antecipada; dataset tem uma fase independente e outra dependente de modelo. Publicação continua sendo um caminho de entrega do v1.

| Mudança | Evidência que deve ser invalidada |
|---|---|
| Objetivo, métrica ou teste congelado | Configuração, treino candidato, avaliação e entrega; abrir novo experimento comparável |
| Ambiente, hardware ou engine | Parecer de viabilidade, receita, autorização de treino e saídas posteriores |
| Conteúdo/split do dataset | Tokenização, receita, autorização e saídas posteriores |
| Modelo, tokenizer ou template | Preparo final, receita, autorização e saídas posteriores |
| Parâmetro de treino | Autorização de treino e saídas posteriores |
| Formato ou destino de entrega | Exportação e autorização de publicação; preserva avaliação original do candidato |

Invalidação preserva o histórico. Uma rodada em execução mantém suas entradas imutáveis; decisões novas criam outra rodada. Não trocar o destino de um job ativo pelo simples ato de editar estado.

## Localização e circulação dos dados

`transfer_mode` é `none`, `manual` ou `authorized`. Em `none`, os dados já devem existir no destino escolhido. Em `manual`, o usuário fornece o caminho no destino e a integridade é conferida antes de executar. Em `authorized`, registrar autorização com origem, destino, hash, tamanho e finalidade antes da transferência. Aprovar treino não autoriza upload ou publicação. Não registrar tokens, conteúdo sensível ou credenciais no manifesto.

Localização e hash são verificados pelo executor real; flags no JSON são evidências declaradas, não sondagem de filesystem remoto. Retorno de Colab/Kaggle usa pacote com `run_id`, identidade das entradas, versões, log de passos, hashes de checkpoints/modelos e métricas. Pacote sem identidade verificável é recusado; a sessão aguarda importação, sem declarar sucesso. Amostras incluídas em relatórios obedecem à mesma política de circulação.

## Avaliação e critério de sucesso

O dataset contém IDs e grupos de exemplos de treino, validação e teste, mais hash de cada split. Um grupo não atravessa splits; deduplicação de conteúdo e similares deve ocorrer antes de dividir. O teste final e o protocolo ficam congelados antes da primeira otimização. Ajustar a receita usa validação; observar repetidamente o teste exige registrar novo experimento e um teste final ainda não usado para seleção.

O protocolo registra métrica/versionamento do avaliador, direção (`maximize` ou `minimize`), amostra mínima, parâmetros de geração e modo de agregação. Avaliação humana ou por modelo juiz também fixa rubrica, versão do juiz, amostragem e procedimento de desempate. Base e candidato usam o mesmo teste e protocolo. Comparação exige limiar absoluto e ganho mínimo orientado pela direção. Dados insuficientes produzem `inconclusiva`, nunca aprovação. O relatório inclui tamanho da amostra, dispersão/incerteza quando aplicável e regressões por recorte relevante à tarefa.

O validador local confere identidade, separação dos IDs/grupos, amostra mínima e decisão numérica declarada. Não executa inferência, não descobre duplicatas semânticas e não confirma a qualidade de um avaliador.

## Sessão, job e checkpoint

Estados do job: `planejada`, `submetendo`, `executando`, `desconhecida`, `interrompida`, `falhou`, `concluida`. Estados terminais não voltam diretamente para execução. Aprovação, avaliação e entrega são artefatos separados; `concluida` significa somente que o treino terminou.

| Estado e observação | Ação |
|---|---|
| planejada, entradas/aprovação válidas | Persistir chave de idempotência e intenção; submeter uma vez |
| submetendo e resposta perdida | Consultar pela chave; nunca repetir um submit sem reconciliação |
| executando e job ativo | Reconectar à observação, preservar job e rodada |
| destino inacessível ou identidade ambígua | Marcar desconhecida, aguardar reconciliação |
| desconhecida e job localizado | Adotar o estado confirmado no executor |
| interrompida/falhou e checkpoint válido | Criar nova rodada com `parent_run_id`, aprovação e checkpoint de origem |
| checkpoint ausente/parcial/incompatível | Recusar retomada e explicar o motivo |
| concluida | Encaminhar avaliação; não disparar outro treino |

Checkpoint retomável exige marcador de conclusão, checksums, passo confirmado, identidade das entradas e estado necessário pelo perfil da engine (pesos, otimizador, scheduler, RNG e posição de dados quando suportados). Um adapter exportado não é automaticamente checkpoint de treino. Informar o intervalo perdido desde o último checkpoint completo; não prometer zero trabalho refeito. O validador só verifica as evidências declaradas, o adaptador deve inspecionar arquivos reais.

Continuação não é retomada: é rodada nova, pedida pelo usuário, que parte do adapter aprovado de outra rodada com otimizador e passos do zero. A receita declara `start_from` (`run_id`, `adapter_hash`); o runtime recusa mãe sem avaliação aprovada, hash diferente, outro modelo base ou outra tarefa. `dataset --base` mantém validação, calibração e teste da mãe idênticos e só acrescenta dados ao treino. A avaliação declara `start_from` com a pontuação da mãe no mesmo teste e protocolo; o candidato abaixo dela é `reprovada`.

## Manifesto, perfis e proteção

O manifesto da rodada inclui objetivo, dataset e splits por hash; revisão de modelo/tokenizer; template e receita; scripts efetivos por hash; engine e versões de dependências; hardware; seeds; destino; orçamento; job; cadeia de retomada; passos, métricas e artefatos. Dependências são fixadas por versão/revisão, não `latest`. Seeds documentam uma entrada, não uma garantia de equivalência numérica entre ambientes.

Cada perfil identifica engine/versão, família de modelo, hardware e versões de dependências testadas. Mantém parâmetros suportados separados em treino, inferência e exportação; defaults com justificativa, fonte e data; sondagem de capacidades; formatos e condições de retomada. Combinação desconhecida exige diagnóstico `nao-verificada`. Não copiar defaults de outra ferramenta como universais. Nenhuma engine padrão é escolhida implicitamente nesta revisão: a escolha permanece aberta, sem bloquear a definição dos contratos.

Proteção exige política validada com falha fechada, caminho físico permitido e autorização por ação. Hook de ferramentas de edição não protege shell/processos. O executor deve impor os destinos aprovados e limites no ambiente real. Política ausente, schema desconhecido, arquivo incompatível ou proteção não certificada bloqueiam execução automática; o framework pode gerar diagnóstico sem afirmar que está protegido.

## Exportação e entrega

Manifesto de exportação deve identificar adapter/modelo base requerido, tokenizer, template, formato, quantização, arquivos e checksums, consumidor/versionamento, resultado de carregamento e inferência de sanidade. Confirmar a métrica no artefato efetivamente entregue e sua tolerância em relação ao candidato. O limite é orientado pela direção da métrica. Um pacote legível mas incompatível com a tarefa não aprova entrega.

Entrega local produz caminho e recibo de integridade. Publicação/deploy exige autorização vinculada ao hash do pacote, destino, visibilidade e ação. Aprovação do treino ou da avaliação não serve como autorização de publicação. Segredos ficam no mecanismo de credenciais do destino, nunca nos JSONs. Falha na entrega preserva o pacote e permite repetir a entrega sem retreinar.

## Tempo e custo

Separar tempo humano, preparo/download, treino, avaliação e entrega. A meta provisória de oito horas só pode ser avaliada sobre um cenário de referência com modelo, volume, contexto, GPU, versões e condições de download definidos. Não é garantia para qualquer hardware.

Denominador da taxa de sucesso: todas as rodadas cujo job iniciou no período, inclusive falhas e cancelamentos. Reportar separadamente rodadas ainda em execução ou aguardando avaliação; não excluí-las silenciosamente. Numerador: rodadas aprovadas no teste congelado. Instalações são projetos distintos com versão do framework e instalação validada, não número de invocações.

Toda rodada declara duração máxima positiva. Custo máximo e moeda são obrigatórios quando o recurso é tarifado. Sem tarifa verificável, informar incerteza e exigir teto de duração; não prometer teto monetário garantido. Limite deve ser aplicado pelo executor no destino mesmo que a sessão caia. Falha em confirmar parada deixa estado desconhecido e alerta; não registrar consumo como encerrado.

## Verificação local

`python _reversa_sdd/contracts/validate.py <bundle.json>` valida um pacote completo de metadados. `python -m unittest discover -s _reversa_sdd/contracts/tests -v` testa schemas, referências e invariantes. Requer Python 3.10+ e `jsonschema` 4.x (presente no ambiente desta revisão).

Esses testes não certificam GPU, engine, motor de agentes, bloqueio de shell, transferência ou publicação. Cada adaptador deve acrescentar testes de integração para esses efeitos antes da declaração de suporte. A fixture `examples/bundle.json` é sintética e não representa treino realizado.
