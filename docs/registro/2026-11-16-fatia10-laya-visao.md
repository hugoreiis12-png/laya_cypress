# Fatia 10: Laya visão no serviço, gate de 90% e e-mail da referência — 2026-11-16

Complementa `2026-11-16-trainer-producao.md` e `2026-11-16-fatia1-analise-completa-vllm.md` (fatia 4 do plano).

## O que mudou
| Parte | Antes | Agora |
|---|---|---|
| E-mail | Só versões promovidas à sombra | Também a primeira versão (`referencia`). O assunto diz "registrada como referência" ou "promovida à sombra" e, quando for o caso, "e pronta para assumir" |
| Gate para assumir | Não existia; só o gate relativo (erro cai ≥20%) | Teto absoluto por tipo: acerto ≥ `TRAINER_TAKEOVER_ACCURACY` (0,90) no teste congelado, teste com ≥ `TRAINER_TAKEOVER_MIN_TEST` (100) amostras e zero canhoto sem assinatura ou só com carimbo liberado. O resultado fica no registro como `pronta` |
| Kernel de visão | Media com o YOLO e entregava `pesos.bin` | Exporta ONNX e mede no **próprio ONNX**, com a mesma transformação de entrada do serviço. Entrega `pesos.onnx`, `classes` e `amostras_teste` |
| Serviço | Não usava os modelos do trainer | Lê o registro do trainer (só leitura), carrega a versão atual do tipo `visao` e grava a opinião dela em toda decisão de canhoto (`laya_visao`) |

## Como o serviço decide
1. A versão em uso é a mesma do trainer: a última em sombra ou, sem nenhuma, a referência. O serviço só carrega se o hash dos pesos bater com o registro.
2. O registro é relido sozinho quando muda; promoção e `control rollback visao` chegam ao serviço sem reinício.
3. A Laya **só assume** quando as três condições valem juntas:
   - `LAYA_VISION_TAKEOVER=1` no serviço (chave manual);
   - a versão está `pronta` no registro (gate de 90%);
   - o modelo diz `assinado` com confiança ≥ `LAYA_VISION_MIN_CONFIDENCE` (0,90).
   Aí o canhoto sai como assinado sem chamar o juiz. Em qualquer outro caso o juiz decide como antes. Um "não" da Laya nunca bloqueia: vai para o juiz.
4. A faixa usada é a metade de cima da foto em pé, com o mesmo recorte e a mesma compressão das tiras de treino.

## Ativar (VM .160)
Arquivos da raiz (fora das pastas que o agente edita):
1. `requirements.txt`: acrescente `onnxruntime==1.22.1`.
2. `compose.yaml`, serviço `laya-cypress`:
   ```yaml
   environment:
     LAYA_VISION_REGISTRY: ${LAYA_VISION_REGISTRY:-}
     LAYA_VISION_TAKEOVER: ${LAYA_VISION_TAKEOVER:-}
     LAYA_VISION_MIN_CONFIDENCE: ${LAYA_VISION_MIN_CONFIDENCE:-0.90}
   volumes:
     - /opt/laya-trainer/data/registro:/srv/registro:ro
   ```
3. Na VM, **antes do redeploy**, crie a pasta com o dono do trainer. Se o Docker criar, ela fica com o root e o trainer não consegue gravar:
   ```bash
   install -d -o 10002 -g 10002 /opt/laya-trainer/data/registro
   ```
4. Portainer, stack do serviço: `LAYA_VISION_REGISTRY=/srv/registro`. **Não** ligue `LAYA_VISION_TAKEOVER` ainda.
5. Redeploy do trainer e do serviço.

Antes da primeira versão, o registro grava `"laya_visao": {"erro": "sem_registro", ...}` ou `sem_versao`. É o esperado.

## Quando ligar a tomada de decisão
1. Chega o e-mail "… e pronta para assumir".
2. Confira que, em produção, a Laya e o juiz concordam nos canhotos:
   ```bash
   python3 -c "import json,collections; c=collections.Counter((r['laya_visao'].get('rotulo'), (r.get('votos') or {}).get('juiz')) for r in map(json.loads, open('/opt/laya-cypress/dados/decisoes.jsonl')) if r.get('laya_visao')); print(c)"
   ```
3. Ligue `LAYA_VISION_TAKEOVER=1` no Portainer e faça o redeploy. Para voltar atrás, desligue a variável ou use `control rollback visao`.

## Limites conhecidos
- Quando a Laya assume, o juiz não vota naquela foto. Ela então não vira rótulo de consenso (juiz + Gemini) e o treino passa a depender da amostra do Gemini e dos casos que ainda vão ao juiz.
- A transformação de entrada é a da avaliação do ultralytics: o lado menor vai para 320 e o centro é recortado em quadrado. Numa faixa larga, as pontas ficam de fora. O acerto medido já inclui esse efeito.
- Versões antigas sem `file` `.onnx` ou sem `classes` não são carregadas.
- O kernel novo instala `ultralytics==8.3.40`, `onnx==1.17.0` e `onnxruntime==1.20.1` no Kaggle. Ele ainda não rodou na conta real.
