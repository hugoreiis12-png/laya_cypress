# Detector de assinatura (modo sombra)

RF-DETR (`Mo-Awadalla/legaldocuman-rfdetr`, Apache 2.0) com o filtro da linha de corte: só conta a
detecção do lado do canhoto. Roda no .140, ao lado do Laya, só em CPU. O serviço do .160 o chama
para cada canhoto e grava o resultado em `_meta.sombra_assinatura`. **A resposta ao n8n não muda.**

## Subir no .140 (na raiz do projeto)

```bash
docker build -f detector/Dockerfile -t laya-detector:local .
docker run -d --name laya-detector --restart unless-stopped -p 8030:8030 \
  -e DETECTOR_API_KEY=<chave> --cpus 2 laya-detector:local
curl http://192.168.0.140:8030/healthz
```

## Ligar a sombra no serviço do .160

| Variável | Valor |
|---|---|
| `SIGNATURE_SHADOW_URL` | `http://192.168.0.140:8030` (vazio = sombra desligada) |
| `SIGNATURE_SHADOW_API_KEY` | a mesma `DETECTOR_API_KEY` |
| `SIGNATURE_SHADOW_TIMEOUT` | segundos, padrão 4 |
| `SIGNATURE_SHADOW_THRESHOLD` | padrão 0.31 (`score_canhoto` ≥ limiar ⇒ liberaria como assinado) |

No `_meta.sombra_assinatura`: `score_bruto`, `score_canhoto`, `linha_corte`, `lado_canhoto`,
`liberaria`, `concorda` (com a decisão atual) ou `erro`. Custo: até `SIGNATURE_SHADOW_TIMEOUT` a mais
por canhoto.
