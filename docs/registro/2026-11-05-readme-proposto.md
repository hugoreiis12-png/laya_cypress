# Alterações propostas para o README.md — 2026-11-05

A proteção de arquivos bloqueia a escrita do agente no `README.md`. Aplique estes trechos manualmente.

## 1. Em "Infraestrutura", trocar a nota de arquitetura por

**Nota de arquitetura:** LangGraph é a camada de orquestração do serviço (`service/app/document_graph.py`), sem checkpointer, para que nenhum dado do documento sobreviva à execução. Sua adoção não altera os endereços e contratos dos serviços acima. A implantação efetiva deve ser registrada quando concluída.

## 2. Em "Regras de evolução", trocar as duas primeiras regras por

- Não treinar nenhum modelo em produção nem com as próprias previsões dele.
- Não tratar a concordância entre Laya e Hermes como ground truth; usar consenso de fontes independentes do modelo treinado.

## 3. Em "Treinamento, avaliação e dados"

- No diagrama, trocar `Rotulagem humana` por `Rotulagem por consenso`.
- Trocar "Predições automáticas podem ajudar a priorizar a triagem, mas não substituem a confirmação humana." por "Rótulos automáticos só valem por consenso de fontes independentes, com conjunto de teste congelado e auditoria contínua."

## 4. Antes de "Execução e testes", inserir

## Protótipo inicial

O serviço é o fluxo principal, e o Gemini é a reserva quando ele cair ou ficar em dúvida (`_meta.inconclusivo`).

- **Tipo do documento:** regras e chave de acesso decidem; o Laya fica em sombra.
- **Assinatura:** o vLLM decide sozinho (`SIGNATURE_ENGINE=juiz`), dentro do limite de capacidade; o RF-DETR só registra.
- **Respostas sem reserva:** apenas canhoto assinado com os campos conferidos. Comprovante, pedido, canhoto sem assinatura e imagem inválida passam pelo Gemini.
- **Supervisão:** o Hermes registra a auditoria, sem alterar a resposta.

## Features futuras

### Treinar RF-DETR + YOLO para a camada computacional

- Treinar YOLOv8 e RF-DETR em PyTorch, offline, partindo de dataset público de assinaturas e de caixas de canhotos confirmadas pelo vLLM.
- Devolver o voto ao detector treinado quando ele superar o atual em um teste fechado, sem liberar canhoto sem assinatura.
- Antes da produção, definir a licença do YOLOv8: a Ultralytics declara AGPL-3.0 ou licença Enterprise.

### Esteira de treinamento contínuo do Laya

- Treinar em ciclos offline pelo ITS-FINE, nunca em produção, com fontes independentes do próprio Laya e curadoria do Hermes.
- Manter um conjunto de teste congelado e promover uma versão só quando ela superar a atual e as regras.
- Versionar no Git privado a receita, os manifestos, as métricas e os hashes de cada ciclo. Imagens, documentos e dados de clientes ficam fora do Git; os pesos ficam em armazenamento próprio, referenciados pelo hash.
- Permitir rollback para a versão anterior.
