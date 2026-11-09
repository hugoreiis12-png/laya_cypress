import json

from trainer.app import collector, type_consensus
from trainer.app.type_consensus import SOURCE_GEMINI, SOURCE_RULES, TypeSample


def votes(rules, gemini):
    return {SOURCE_RULES: rules, SOURCE_GEMINI: gemini}


def test_rules_and_gemini_agreeing_label_the_type():
    decision = type_consensus.decide(votes("pedido", "pedido"))
    assert decision.status == type_consensus.AGREED and decision.label == "pedido"


def test_disagreement_never_labels():
    decision = type_consensus.decide(votes("canhoto", "pedido"))
    assert decision.status == type_consensus.DISAGREED and decision.label is None


def test_a_missing_or_unknown_vote_never_labels():
    assert type_consensus.decide(votes("canhoto", None)).status == type_consensus.INSUFFICIENT
    assert type_consensus.decide(votes("boleto", "boleto")).label is None


def test_training_set_keeps_only_agreed_samples():
    samples = [TypeSample("a", votes("canhoto", "canhoto")), TypeSample("b", votes("canhoto", "pedido"))]
    assert [(sample.item_id, label) for sample, label in type_consensus.training_set(samples)] == [("a", "canhoto")]


def write_log(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def test_only_the_rules_engine_counts_as_an_independent_type_vote(tmp_path):
    log = tmp_path / "decisoes.jsonl"
    write_log(log, [{"evento": "decisao", "id": "A", "tipo": "canhoto", "motor": "regras", "rotacao": 90},
                    {"evento": "referencia", "id": "A", "origem": "gemini", "tipo": "canhoto"},
                    {"evento": "decisao", "id": "B", "tipo": "canhoto", "motor": "laya"},
                    {"evento": "referencia", "id": "B", "origem": "gemini", "tipo": "canhoto"}])
    by_id = {sample.item_id: sample for sample in collector.load_type_samples(log)}
    assert by_id["A"].votes == votes("canhoto", "canhoto") and by_id["A"].rotation == 90
    assert by_id["B"].votes[SOURCE_RULES] is None
