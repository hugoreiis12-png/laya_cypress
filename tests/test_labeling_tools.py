import http.server
import threading

import pytest

from scripts.rotulagem import rotulador_dinamico as rotulador
from scripts.benchmark import teste_laya

BY_ID = {
    "a1": {"id": "a1", "status": "unlabeled"},
    "m1": {"id": "m1", "status": "labeled", "triagem": {"fonte": "triagem_manual"}},
}


def errors_for(*items, overwrite=False):
    return rotulador.validate_items(list(items), BY_ID, overwrite)


def test_signed_and_unsigned_items_are_accepted():
    assert errors_for({"id": "a1", "action": "ok", "possui_assinatura": True}) == []
    assert errors_for({"id": "a1", "action": "no", "possui_assinatura": False}) == []


def test_missing_signature_is_rejected_not_defaulted_to_false():
    assert len(errors_for({"id": "a1", "action": "no"})) == 1


def test_signature_contradicting_the_action_is_rejected():
    assert len(errors_for({"id": "a1", "action": "ok", "possui_assinatura": False})) == 1


def test_unknown_id_bad_action_bad_type_and_duplicates_are_rejected():
    assert len(errors_for({"id": "zz", "action": "ok", "possui_assinatura": True})) == 1
    assert len(errors_for({"id": "a1", "action": "talvez"})) == 1
    assert len(errors_for({"id": "a1", "action": "ok", "tipo_documento": "recibo"})) == 1
    item = {"id": "a1", "action": "ok", "possui_assinatura": True}
    assert len(errors_for(item, item)) == 1


def test_manual_triage_label_is_protected_unless_overwrite_is_explicit():
    item = {"id": "m1", "action": "ok", "possui_assinatura": True}
    assert len(errors_for(item)) == 1
    assert errors_for(item, overwrite=True) == []


def test_skip_keeps_notes_and_status_untouched():
    record = {"id": "a1", "status": "unlabeled", "notes": "candidato:canhoto_assinado"}
    rotulador.apply_item(record, {"id": "a1", "action": "skip"})
    assert record["notes"] == "candidato:canhoto_assinado" and record["status"] == "unlabeled"
    assert record["triagem"]["pulado"] is True


def test_non_canhoto_type_is_labeled_without_signature():
    record = {"id": "a1", "status": "unlabeled", "assinatura": {"possui_assinatura": True}}
    rotulador.apply_item(record, {"id": "a1", "action": "ok", "tipo_documento": "comprovante"})
    assert record["tipo_documento"] == "comprovante" and "assinatura" not in record


def test_finish_refuses_to_write_when_nothing_is_valid(tmp_path, monkeypatch):
    out = tmp_path / "preds.jsonl"
    monkeypatch.setattr(teste_laya, "PREDICTIONS_OUT", out)
    rows = [{"id": "a1", "prediction": {}, "error": "HTTP 401: credencial inválida"}]
    assert teste_laya.finish(rows, "dev") == 1 and not out.exists()


def test_finish_keeps_the_full_prediction(tmp_path, monkeypatch):
    out = tmp_path / "preds.jsonl"
    monkeypatch.setattr(teste_laya, "PREDICTIONS_OUT", out)
    pred = {"tipo_documento": "pedido", "itens": [{"produto": "Alface", "qnt": 10}]}
    assert teste_laya.finish([{"id": "a1", "prediction": pred, "error": None}], "dev") == 0
    assert '"itens"' in out.read_text(encoding="utf-8")


@pytest.fixture
def rejecting_service():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))  # unread body makes Windows reset the socket
            self.send_response(401)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_bad_credentials_abort_before_the_batch(rejecting_service):
    assert teste_laya.check_credentials(rejecting_service, "wrong") == "HTTP 401"


@pytest.fixture
def answering_service():
    body = b'{"output": "{\\"tipo_documento\\": \\"canhoto\\"}", "_meta": {"motor_decisao": "laya"}}'

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_prediction_keeps_the_service_meta(answering_service, tmp_path, monkeypatch):
    (tmp_path / "a1.jpg").write_bytes(b"img")
    monkeypatch.setattr(teste_laya, "ROOT", tmp_path)
    status, _, pred, error = teste_laya.predict_one(answering_service, "k", {"id": "a1", "image": "a1.jpg"})
    assert status == 200 and error is None
    assert pred["tipo_documento"] == "canhoto" and pred["_meta"]["motor_decisao"] == "laya"


def test_review_queue_lists_only_quarantine_lowest_score_first(tmp_path):
    records = [{"id": i, "image": f"images/{i}.jpg", "sha256": i, "status": s}
               for i, s in (("a", "pendente_revisao"), ("b", "labeled"), ("c", "pendente_revisao"), ("d", "pendente_revisao"))]
    scores = tmp_path / "scores.jsonl"
    scores.write_text('{"sha": "a", "score": 0.9}\n{"sha": "c", "score": 0.2}\n', encoding="utf-8")
    queue = rotulador.build_review_queue(records, scores)
    assert [item["id"] for item in queue] == ["d", "c", "a"]  # unscored first: never hidden at the end
    assert {item["group"] for item in queue} == {"nao_assinado"}


def test_sample_is_random_but_repeatable(monkeypatch):
    records = [{"id": str(i), "sha256": f"{i:064x}", "status": "labeled"} for i in range(1, 40)]
    monkeypatch.setattr(teste_laya, "load", lambda _: records)
    first = teste_laya.select_gold("all", 5, seed=7)
    assert first == teste_laya.select_gold("all", 5, seed=7)
    assert [r["id"] for r in first] != [r["id"] for r in records[:5]]
