import pytest

from service.app import comprovante, items
from service.app.cnpj_key import (
    access_key_is_valid, cnpj_is_valid, find_access_keys, find_cnpjs, format_cnpj,
    invoice_number, pick_issuer,
)

WHITELIST_FORMATTED = [
    "22.089.653/0003-93", "22.089.653/0002-02", "15.268.427/0001-42", "15.268.427/0002-23",
    "36.895.856/0001-10", "36.895.856/0002-09", "38.028.199/0001-39", "40.135.028/0001-50",
    "54.197.106/0001-56", "46.126.673/0001-93", "04.972.092/0031-48",
]
ISSUER = "36895856000110"


def make_key(cnpj=ISSUER, number="000123456"):
    """Independent mod-11 implementation, so the test does not mirror the code under test."""
    body = "35" + "2610" + cnpj + "55" + "001" + number + "1" + "12345678"
    total = sum(int(d) * (2 + i % 8) for i, d in enumerate(reversed(body)))
    rest = total % 11
    return body + str(0 if rest < 2 else 11 - rest)


@pytest.mark.parametrize("cnpj", WHITELIST_FORMATTED)
def test_whitelist_cnpjs_have_valid_check_digits(cnpj):
    assert cnpj_is_valid(cnpj)


def test_malformed_whitelist_entry_from_the_old_prompt_is_not_valid():
    assert not cnpj_is_valid("04.972.092/00031-48")


def test_access_key_check_digit():
    key = make_key()
    assert len(key) == 44 and access_key_is_valid(key)
    assert not access_key_is_valid(key[:-1] + str((int(key[-1]) + 1) % 10))


def test_finds_contiguous_and_grouped_keys():
    key = make_key()
    grouped = " ".join(key[i:i + 4] for i in range(0, 44, 4))
    assert find_access_keys(f"CHAVE {key} fim") == [key]
    assert find_access_keys(f"CHAVE DE ACESSO\n{grouped}") == [key]


def test_ocr_misread_key_is_rejected_not_repaired():
    broken = make_key()[:-1] + "0" if make_key()[-1] != "0" else make_key()[:-1] + "1"
    assert find_access_keys(broken) == []


def test_issuer_comes_from_key_when_whitelisted():
    whitelist = (ISSUER,)
    assert pick_issuer("sem cnpj no texto", whitelist, make_key()) == format_cnpj(ISSUER)


def test_issuer_ignores_non_whitelisted_cnpj():
    text = "DESTINATARIO 22.089.653/0003-93 EMITENTE 36.895.856/0001-10"
    assert pick_issuer(text, (ISSUER,)) == "36.895.856/0001-10"
    assert pick_issuer("CNPJ 22.089.653/0003-93", (ISSUER,)) == ""


def test_cnpj_search_does_not_read_inside_the_access_key():
    assert find_cnpjs(f"chave {make_key()}") == []


def test_invoice_number_from_key_beats_ocr_text():
    assert invoice_number("Nº 999", make_key(number="000123456")) == "123456"


def test_invoice_number_from_labels_ignores_key_and_dots():
    assert invoice_number("NF-e Nº 000.123.456 Série 1") == "123456"
    assert invoice_number("sem rótulo") == ""


RECEIPT = """Banco Itaú
Comprovante de Pix
Valor: R$ 1.234,56
Data: 16/06/2026 às 10:30
Favorecido: Natural Farms Agricola
Pagador: Joao da Silva
ID da transação: E1234567890ABC
"""


def test_comprovante_fields():
    got = comprovante.extract(RECEIPT)
    assert got["instituicao"] == "Itaú" and got["valor"] == "1234.56"
    assert got["data"] == "2026-06-16" and got["hora"] == "10:30:00"
    assert got["favorecido"] == "Natural Farms Agricola" and got["pagador"] == "Joao da Silva"
    assert got["identificacao"] == "E1234567890ABC"


def test_comprovante_impossible_date_is_none():
    assert comprovante.extract("Data 31/02/2026")["data"] is None


@pytest.mark.parametrize("line,produto,qnt,codigo", [
    ("Alface 10", "Alface", 10, None), ("Alface = 10", "Alface", 10, None),
    ("10 Alface", "Alface", 10, None), ("Cheiro Verde 20", "Cheiro Verde", 20, None),
    ("ALF01 - Alface 3", "Alface", 3, "ALF01"), ("Queijo Mussarela 2 COD-99", "Queijo Mussarela", 2, "COD-99"),
])
def test_item_line_formats(line, produto, qnt, codigo):
    assert items.extract(line) == [{"codigo": codigo, "produto": produto, "qnt": qnt}]


def test_chat_noise_is_ignored():
    chat = "Encaminhada\nBom dia\n10:32\nAlface 10\nok\nRúcula 5"
    assert [i["produto"] for i in items.extract(chat)] == ["Alface", "Rúcula"]
