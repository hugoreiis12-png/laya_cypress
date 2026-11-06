# Laya

Este motor reimplementa, em `sequence.py` e `decision_model.py`, a arquitetura do
modelo de decisao tipada e a construcao de sequencia do projeto Laya:

- Repositorio: https://github.com/NandhaKishorM/laya
- Autor: Convai Innovations (Nandakishor M)
- Licenca: Apache License 2.0 (copia em `LICENSE.txt`)
- Revisao de referencia: commit `1e28ac20c0896b1c37a744cd11f740eb98f8b178` (versao 0.3.11)
- Arquivos de origem: `laya/common.py` (DecisionModel, build_sequence, render_options,
  collate_items, temp_bucket, confidence_from_probs) e `laya/agent.py` (normalizacao
  das perguntas, truncamento, decodificacao das respostas).

A reimplementacao preserva os nomes dos tensores para que checkpoints publicados no
formato do Laya (`rl_agent_config.json`) carreguem com correspondencia estrita, e e
conferida por paridade numerica contra a referencia na certificacao
(`certification.json`). O mesmo modelo roda sobre qualquer encoder do Hugging Face,
com cabecas novas. Nada do codigo de treino por RLCD, roteamento, ONNX, MCP ou
integracoes do Laya foi incorporado. O pacote `laya` nao e dependencia deste motor.
