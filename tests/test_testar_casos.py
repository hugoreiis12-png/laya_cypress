import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("testar_casos", Path(__file__).parent / "e2e" / "testar_casos.py")
testar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(testar)

RESULT = {"output": json.dumps({"tipo_documento": "canhoto", "assinatura": {"possui_assinatura": True, "confianca": 95},
                                 "emitente": {"cnpj": "11.222.333/0001-81"}, "nota_fiscal": {"numero": ""},
                                 "qrcode": {"chave": "12345678901234567890123456789012345678901234"}}),
          "_meta": {"inconclusivo": True, "motivos_inconclusivo": ["auditoria"], "rotacao": 90, "latencia_ms": 4000,
                    "estado_assinatura": "SIGNED", "leitura": {"caracteres_ocr": 700},
                    "revisao_assinatura": {"resultado": "sem_capacidade"}}}


def test_the_row_reports_state_but_never_the_field_values():
    row = testar.summarize("nota.jpg", RESULT)
    assert row["campos"] == "cnpj+chave" and row["motivos"] == "auditoria" and row["juiz"] == "sem_capacidade"
    assert "11.222.333" not in json.dumps(row) and "1234567890" not in json.dumps(row)


def test_the_verdict_compares_only_what_was_expected():
    row = testar.summarize("nota.jpg", RESULT)
    assert testar.verdict(row, {"tipo": "canhoto", "inconclusivo": True}) == "OK"
    assert testar.verdict(row, {"tipo": "pedido"}) == "FALHOU:tipo"
    assert testar.verdict(row, None) == "?"


def test_only_photo_files_are_sent(tmp_path):
    for name in ("a.jpg", "b.PNG", "notas.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [path.name for path in testar.photos_in(tmp_path)] == ["a.jpg", "b.PNG"]
