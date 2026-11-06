"""Typed questions sent to Laya (single source of truth, also used by evalkit)."""

TIPO_DOCUMENTO = {
    "type": "choice",
    "instructions": "Classifique o tipo do documento. Classifique SEMPRE, mesmo que o texto esteja ilegível, incompleto ou truncado.",
    "criteria": {
        "canhoto": "DANFE, NF-e ou canhoto de entrega: contém número de nota, CNPJ do emitente, chave de acesso de 44 dígitos e a região 'IDENTIFICAÇÃO E ASSINATURA DO RECEBEDOR'.",
        "comprovante": "Comprovante de pagamento, PIX, TED, DOC ou transferência bancária: contém instituição, valor, data, hora, favorecido e pagador.",
        "pedido": "Folha de pedido ou lista de produtos com quantidades: uma relação de itens, cada um com produto e quantidade.",
        "imagem_invalida": "Ilegível, cortada, com mais de um documento, com mais de um logotipo, ou incompatível com os tipos anteriores.",
    },
}

POSSUI_ASSINATURA = {
    "type": "noul",
    "instructions": "Existe assinatura manuscrita, rubrica, nome escrito à mão, carimbo preenchido ou qualquer marca feita por caneta na região de identificação do recebedor? A análise geométrica de tinta está descrita no state. Não considere sombras, sujeira, ruído, carimbos em branco ou linhas impressas.",
    "criteria": {
        "true": "há marca manuscrita do recebedor na região indicada",
        "false": "a região está em branco ou contém apenas ruído, sombra ou linhas impressas",
    },
}

TIPO_ASSINATURA = {
    "type": "choice",
    "instructions": "Que tipo de marca do recebedor foi identificada? Responda 'nenhuma' se a região estiver em branco ou contiver apenas ruído.",
    "criteria": {
        "manuscrita": "assinatura manuscrita feita à mão, em caneta ou lapiseira",
        "rubrica": "rubrica ou traço de assinatura abreviado",
        "nome_manuscrito": "nome do recebedor escrito à mão, sem ser assinatura completa",
        "carimbo": "carimbo preenchido com tinta ou almofada",
        "nenhuma": "nenhuma marca do recebedor foi encontrada",
    },
}
