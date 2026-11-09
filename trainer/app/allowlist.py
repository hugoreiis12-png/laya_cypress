"""Removes personal names by construction: only words of a fixed fiscal vocabulary survive, everything else is masked."""
import re
import unicodedata

WORD = re.compile(r"\[[A-Z]+\]|[^\W\d_]+|\d+", re.UNICODE)  # placeholders already made by the anonymizer stay
MASK_WORD, MASK_NUMBER = "[X]", "[N]"
VOCABULARY = frozenset("""
danfe nf nfe nota notas fiscal fiscais eletronica eletronico documento auxiliar entrada saida serie folha numero
chave acesso consulta autenticidade portal nacional site sefaz autorizadora autorizacao protocolo uso
recebemos recebimento produtos produto servicos servico constantes indicada ao lado data emissao identificacao
assinatura recebedor receber recebido valor valores total subtotal desconto frete seguro outras despesas
natureza operacao venda mercadoria adquirida recebida terceiros remessa devolucao bonificacao
emitente remetente destinatario razao social nome endereco municipio bairro distrito cep fone fax uf pais
cnpj cpf inscricao estadual municipal subst tribut substituicao tributaria regime normal crt simples
calculo imposto base icms ipi pis cofins st fcp aliquota percentual
transportador transportadora volumes quantidade especie marca peso bruto liquido placa veiculo frete conta
dados adicionais informacoes complementares reservado fisco
codigo descricao ncm sh cst csosn cfop unid un cx kg l ml pc qtd quant unit unitario preco
pedido pedidos orcamento cliente vendedor representante condicao pagamento prazo vencimento parcela parcelas
duplicata fatura boleto pix transferencia comprovante comprovantes banco agencia conta favorecido pagador
beneficiario autenticacao transacao tarifa dinheiro cartao credito debito
de da do das dos e em a o as os para por com sem no na nos nas ao pela pelo ou que
""".split())


def _plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).casefold()


def _mask(match: re.Match[str], vocabulary: frozenset[str]) -> str:
    word = match.group(0)
    if word.startswith("["):
        return word
    if word.isdigit():
        return MASK_NUMBER
    return word if _plain(word) in vocabulary else MASK_WORD


class AllowlistRedactor:
    """Implements the anonymizer's NameRedactor: a name can only leak if it is a vocabulary word, and none is."""

    def __init__(self, vocabulary: frozenset[str] = VOCABULARY) -> None:
        self._vocabulary = vocabulary

    def redact(self, text: str) -> str:
        return WORD.sub(lambda match: _mask(match, self._vocabulary), text)
