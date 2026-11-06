# Gold set

Documentos reais de clientes (LGPD). `images/`, `inbox/` e `predictions/` ficam fora do git; só `manifest.jsonl` é versionado e ele também contém dados do cliente, então o repositório tem de continuar privado.

## Fluxo
1. `python -m evalkit add <arquivos ou pasta> --source whatsapp` → copia para `images/<sha>.<ext>` e cria um registro `unlabeled`.
2. Rotular a mão em `manifest.jsonl`: `status: "labeled"` e os campos abaixo.
3. `python -m evalkit validate` → checa hash, dígitos e campos obrigatórios.
4. `python -m evalkit run --engine <replay|http|laya_text> --target <alvo> --split dev --out gold/predictions/<nome>.jsonl`
5. `python -m evalkit score --preds gold/predictions/<nome>.jsonl --name <nome> --split dev` → `reports/<nome>.md|json`.

## Rótulo (mesmo formato do contrato)
```json
{"id":"…","image":"images/….jpg","sha256":"…","source":"whatsapp","status":"labeled",
 "tipo_documento":"canhoto|comprovante|pedido|imagem_invalida",
 "assinatura":{"possui_assinatura":true,"descricao":"rubrica"},
 "emitente":{"cnpj":"36895856000110"},"nota_fiscal":{"numero":"000123"},"qrcode":{"chave":"<44 dígitos>"},
 "comprovante":{"valor":"…","data":"…"},
 "itens":[{"produto":"Alface","qnt":10}],
 "ocr_text":null,"notes":""}
```
Só os campos do tipo do documento são pontuados. Campo vazio no rótulo significa "não deve haver valor": qualquer valor devolvido conta como inventado.

## Regras de rotulagem
- `possui_assinatura` = marca manuscrita do recebedor na região de identificação e assinatura. Sombra, sujeira, linha impressa e carimbo em branco não contam.
- Na dúvida real, use `imagem_invalida` e anote em `notes`; não adivinhe.
- Um rótulo por humano, conferido por um segundo em uma amostra, antes de usar o resultado para decidir limiar.

## Splits
Determinístico pelo `sha256` (1 em cada 5 vai para `test`). Calibre limiares só no `dev`; o `test` é consultado uma vez por decisão.

## Cobertura mínima antes de decidir qualquer limiar
Pelo menos 30 por tipo de documento e 30 canhotos assinados e 30 não assinados. Abaixo disso o relatório mostra os números, mas eles não sustentam o corte.
