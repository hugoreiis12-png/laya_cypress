import pytest

from service.app import rules

ORDER_ITEMS = [{"codigo": None, "produto": "PISO", "qnt": 8}, {"codigo": None, "produto": "SÉRIE", "qnt": 1}]
ACCESS_KEY = "33261115268427000142550010008179371192540761"


def classify(text="", key="", items=None):
    return rules.classify(text, key, {}, items if items is not None else [])


def test_access_key_proves_a_canhoto():
    assert classify(key=ACCESS_KEY, items=ORDER_ITEMS) == "canhoto"


def test_two_invoice_words_prove_a_canhoto():
    assert classify("DANFE documento auxiliar da NF-e", items=ORDER_ITEMS) == "canhoto"


def test_plain_order_with_items_is_a_pedido():
    assert classify("2 alface\n3 tomate", items=ORDER_ITEMS) == "pedido"


@pytest.mark.parametrize("marker", ["NF-e", "DANFE", "Nota Fiscal", "chave de acesso", "recebemos de", "recebedor"])
def test_one_invoice_marker_never_becomes_a_pedido(marker):
    assert classify(f"{marker} PISO 8 SÉRIE 1", items=ORDER_ITEMS) == "imagem_invalida"


def test_text_without_items_or_markers_is_invalid():
    assert classify("nada útil") == "imagem_invalida"
