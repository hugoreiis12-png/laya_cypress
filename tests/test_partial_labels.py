from evalkit import fields
from scripts.rotulagem import importar_triagem as triagem

CANDIDATE = {"id": "a1", "notes": triagem.CANDIDATE_NOTE, "tipo_documento": None}


def test_partial_label_is_not_scored_on_canhoto_fields():
    gold = {"tipo_documento": "canhoto", "rotulo_parcial": True, "emitente": {}, "qrcode": {}}
    pred = {"emitente": {"cnpj": "36895856000110"}, "qrcode": {"chave": "3" * 44}}
    assert fields.score_scalars(gold, pred) == {}


def test_full_canhoto_label_is_still_scored():
    gold = {"tipo_documento": "canhoto", "emitente": {"cnpj": "36895856000110"}}
    assert "cnpj" in fields.score_scalars(gold, {"emitente": {}})


def test_validate_accepts_a_clean_row():
    accepted, errors, pending = triagem.validate(
        [{"id": "a1", "tipo_documento": "canhoto", "possui_assinatura": "false", "nota": " ''''''  "}], [CANDIDATE])
    assert errors == [] and pending == []
    assert accepted[0]["possui_assinatura"] is False and accepted[0]["nota"] == ""


def test_validate_rejects_unknown_id_bad_type_and_bad_boolean():
    rows = [
        {"id": "zzz", "tipo_documento": "canhoto", "possui_assinatura": "true", "nota": ""},
        {"id": "a1", "tipo_documento": "recibo", "possui_assinatura": "true", "nota": ""},
        {"id": "a1", "tipo_documento": "canhoto", "possui_assinatura": "sim", "nota": ""},
    ]
    for row in rows:
        _, errors, _ = triagem.validate([row], [CANDIDATE])
        assert len(errors) == 1


def test_duplicate_id_is_rejected():
    row = {"id": "a1", "tipo_documento": "canhoto", "possui_assinatura": "true", "nota": ""}
    _, errors, _ = triagem.validate([row, row], [CANDIDATE])
    assert any("duplicado" in e for e in errors)


def test_pending_candidates_are_reported_not_imported():
    other = {"id": "b2", "notes": triagem.CANDIDATE_NOTE}
    accepted, errors, pending = triagem.validate(
        [{"id": "a1", "tipo_documento": "canhoto", "possui_assinatura": "true", "nota": ""}], [CANDIDATE, other])
    assert errors == [] and pending == ["b2"] and len(accepted) == 1
