from evalkit import classify, fields, normalize, score

CHAVE = "3" * 44


def canhoto(signed=True, numero="123", cnpj="36.895.856/0001-10", chave=CHAVE):
    return {"id": "a", "tipo_documento": "canhoto", "assinatura": {"possui_assinatura": signed},
            "emitente": {"cnpj": cnpj}, "nota_fiscal": {"numero": numero}, "qrcode": {"chave": chave}}


def prediction(signed=True, conf=95, numero="000123", cnpj="36895856000110", chave=CHAVE):
    return {"tipo_documento": "canhoto", "assinatura": {"possui_assinatura": signed, "confianca": conf},
            "emitente": {"cnpj": cnpj}, "nota_fiscal": {"numero": numero}, "qrcode": {"chave": chave}}


def test_normalizers_match_the_n8n_contract():
    assert normalize.invoice_number("123") == "000123"
    assert normalize.digits("36.895.856/0001-10") == "36895856000110"
    assert normalize.text("  Alface  CRESPA ") == "alface crespa"
    assert normalize.quantity("2,50") == "2.5"


def test_outcome_distinguishes_missing_from_hallucinated():
    assert fields.outcome("a", "a") == "correct"
    assert fields.outcome("a", "b") == "wrong"
    assert fields.outcome("a", "") == "missing"
    assert fields.outcome("", "invented") == "spurious"


def test_all_scalars_correct_after_normalization():
    outcomes = fields.score_scalars(canhoto(), prediction())
    assert set(outcomes.values()) == {"correct"}


def test_items_use_multiset_match():
    gold = [{"produto": "Alface", "qnt": 10}, {"produto": "Rúcula", "qnt": 2}]
    pred = [{"produto": "alface", "qnt": "10"}, {"produto": "Tomate", "qnt": 1}]
    assert fields.score_items(gold, pred) == (1, 1, 1)


def test_threshold_sweep_counts_false_accepts():
    rows = [(True, True, 95.0), (False, True, 92.0), (False, True, 60.0)]
    sweep = {s["threshold"]: s for s in classify.threshold_sweep(rows)}
    assert sweep[90]["accepted"] == 2 and sweep[90]["false_accepts"] == 1
    assert sweep[50]["false_accepts"] == 2


def test_calibration_detects_overconfidence():
    result = classify.calibration([(0.95, True), (0.95, False), (0.95, False), (0.95, False)])
    assert result["ece"] > 0.5


def test_score_end_to_end_with_missing_prediction():
    gold = [canhoto(), {**canhoto(), "id": "b"}]
    preds = {"a": {"prediction": prediction(), "latency_ms": 10.0}}
    result = score.score(gold, preds)
    assert result["prediction_errors"] == 1
    assert result["tipo_documento"]["accuracy"] == 0.5
    assert result["fields"]["cnpj"]["missing"] == 1
