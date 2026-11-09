from trainer.app.allowlist import MASK_NUMBER, MASK_WORD, VOCABULARY, AllowlistRedactor
from trainer.app.anonymizer import anonymize, find_residual
from trainer.app.packager import TextItem, write_text_package

REDACTOR = AllowlistRedactor()
RECEIPT = "RECEBEMOS DE NATURAL FARMS AGRICOLA LTDA OS PRODUTOS CONSTANTES NA NOTA FISCAL Nº 36342 Ricardo Silva"


def test_fiscal_words_survive_and_names_do_not():
    redacted = REDACTOR.redact(RECEIPT)
    assert redacted.startswith("RECEBEMOS DE [X] [X] [X] [X] OS PRODUTOS CONSTANTES NA NOTA FISCAL")
    assert "NATURAL" not in redacted and "Ricardo" not in redacted and "Silva" not in redacted


def test_every_number_is_masked():
    assert REDACTOR.redact("valor 299 de 30") == f"valor {MASK_NUMBER} de {MASK_NUMBER}"


def test_accents_and_case_do_not_hide_a_vocabulary_word():
    assert REDACTOR.redact("Identificação E ASSINATURA do recebedor") == "Identificação E ASSINATURA do recebedor"


def test_a_word_outside_the_vocabulary_is_masked_even_when_it_looks_harmless():
    assert REDACTOR.redact("Supermercado Zona Sul") == " ".join([MASK_WORD] * 3)


def test_no_vocabulary_word_is_a_person_or_company_name():
    assert not {"silva", "souza", "natural", "farms", "supermercado", "ltda", "sa"} & VOCABULARY


def test_punctuation_and_layout_are_kept():
    assert REDACTOR.redact("Nome: Maria\nData: 10/06") == f"Nome: {MASK_WORD}\nData: {MASK_NUMBER}/{MASK_NUMBER}"


def test_the_anonymizer_reports_that_names_were_removed_and_the_package_accepts_it(tmp_path):
    result = anonymize(RECEIPT + " CNPJ 36.895.856/0002-09", REDACTOR)
    assert result.names_removed and "Silva" not in result.text and not find_residual(result.text)
    manifest = write_text_package(tmp_path, "p", [TextItem("a" * 16, "canhoto", result.text)], 15, result.names_removed)
    assert manifest["nomes_removidos"] is True


def test_placeholders_from_the_anonymizer_are_kept_intact():
    assert REDACTOR.redact("contato [TELEFONE] e [EMAIL]") == f"{MASK_WORD} [TELEFONE] e [EMAIL]"
