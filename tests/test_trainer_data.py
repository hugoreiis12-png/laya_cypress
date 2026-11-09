import json

import cv2
import numpy as np
import pytest

from trainer.app import collector, consensus, cropper, packager, split
from trainer.app.anonymizer import anonymize, find_residual, pseudonym
from trainer.app.consensus import SOURCE_BUBBLE, SOURCE_GEMINI, SOURCE_VLLM, Sample

KEY = "33261115268427000142550010008179371192540761"
SECRET = "s" * 24


def votes(vllm, gemini, bubble=None):
    return {SOURCE_VLLM: vllm, SOURCE_GEMINI: gemini, SOURCE_BUBBLE: bubble}


def test_two_independent_sources_that_agree_label_the_photo():
    assert consensus.decide(votes(True, True)).label == consensus.LABEL_SIGNED
    assert consensus.decide(votes(False, False)).label == consensus.LABEL_UNSIGNED


def test_disagreement_goes_to_review_and_never_trains():
    decision = consensus.decide(votes(True, False))
    assert decision.status == consensus.DISAGREED and decision.label is None


def test_one_source_alone_never_labels():
    assert consensus.decide(votes(True, None)).status == consensus.INSUFFICIENT
    assert consensus.decide({SOURCE_BUBBLE: True}).status == consensus.INSUFFICIENT


def test_bubble_can_veto_an_unsigned_label_but_unsigned_from_bubble_is_not_evidence():
    assert consensus.decide(votes(False, False, bubble=True)).status == consensus.DISAGREED
    assert consensus.decide(votes(True, True, bubble=False)).label == consensus.LABEL_SIGNED


def test_partition_splits_training_review_and_skipped():
    samples = [Sample("a", votes(True, True)), Sample("b", votes(True, False)), Sample("c", votes(True, None))]
    result = consensus.partition(samples)
    assert [label for _, label in result.training] == ["assinado"] and len(result.review) == 1 and result.skipped == 1


def write_log(path, records):
    path.write_text("\n".join(json.dumps(record) if isinstance(record, dict) else record for record in records), encoding="utf-8")


def test_collector_joins_the_service_decision_and_the_gemini_reference(tmp_path):
    log = tmp_path / "decisoes.jsonl"
    write_log(log, [{"evento": "decisao", "id": "A", "votos": {"juiz": True, "confianca_juiz": 95}, "rotacao": 270},
                    {"evento": "referencia", "id": "A", "origem": "gemini", "possui_assinatura": True},
                    "linha quebrada",
                    {"evento": "decisao", "id": "B", "votos": {"juiz": True, "confianca_juiz": 40}}])
    result = collector.load_samples(log)
    by_id = {sample.item_id: sample for sample in result.samples}
    assert result.malformed == 1 and by_id["A"].votes == {"vllm": True, "gemini": True} and by_id["A"].rotation == 270
    assert by_id["B"].votes["vllm"] is None


def test_anonymizer_removes_documents_and_contacts():
    text = f"CNPJ 15.268.427/0001-42 CPF 123.456.789-09 chave {KEY} mail a@b.com tel (21) 99999-1234 CEP 22280-000"
    result = anonymize(text)
    assert find_residual(result.text) == [] and "[CNPJ]" in result.text and "[CHAVE]" in result.text
    assert KEY not in result.text and "15.268.427" not in result.text and not result.names_removed


def test_anonymizer_applies_the_name_redactor_and_says_so():
    class Redactor:
        def redact(self, text):
            return text.replace("Lucas", "[PESSOA]")

    result = anonymize("recebido por Lucas", Redactor())
    assert result.text == "recebido por [PESSOA]" and result.names_removed


def test_pseudonym_is_stable_and_depends_on_the_secret():
    assert pseudonym(SECRET, "MSG1") == pseudonym(SECRET, "MSG1") != pseudonym("z" * 24, "MSG1")
    assert len(pseudonym(SECRET, "MSG1")) == 16


def test_split_is_deterministic_and_roughly_proportional():
    ids = [pseudonym(SECRET, str(n)) for n in range(2000)]
    share = sum(split.is_test(item, 15) for item in ids) / len(ids)
    assert 0.10 < share < 0.20 and all(split.is_test(item, 15) == split.is_test(item, 15) for item in ids[:50])


def photo(width=200, height=400):
    ok, encoded = cv2.imencode(".jpg", np.full((height, width, 3), 200, np.uint8))
    assert ok
    return encoded.tobytes()


def test_receiver_strip_is_the_top_band_of_the_upright_page():
    strip = cv2.imdecode(np.frombuffer(cropper.receiver_strip(photo(), 0), np.uint8), cv2.IMREAD_COLOR)
    assert strip.shape[1] == 200 and 80 <= strip.shape[0] <= 92


def test_rotation_is_applied_before_cutting():
    strip = cv2.imdecode(np.frombuffer(cropper.receiver_strip(photo(), 90), np.uint8), cv2.IMREAD_COLOR)
    assert strip.shape[1] == 400


def test_unreadable_or_tiny_photos_are_dropped():
    with pytest.raises(cropper.CropError):
        cropper.receiver_strip(b"nao e imagem", 0)
    with pytest.raises(cropper.CropError):
        cropper.receiver_strip(photo(50, 60), 0)


def test_vision_package_splits_by_the_frozen_rule_and_carries_a_manifest(tmp_path):
    items = [packager.StripItem(pseudonym(SECRET, str(n)), "assinado" if n % 2 else "nao_assinado", photo()) for n in range(60)]
    manifest = packager.write_vision_package(tmp_path, "visao-0001", items, percent=15)
    stored = sum(manifest["contagens"].values())
    assert stored == 60 and manifest["anonimizado"] and len(manifest["sha256"]) == 64
    assert (tmp_path / "manifest.json").exists()
    for item in items:
        side = "test" if split.is_test(item.item_id, 15) else "train"
        assert (tmp_path / side / item.label / f"{item.item_id}.jpg").exists()


def test_package_refuses_ids_that_are_not_pseudonyms(tmp_path):
    with pytest.raises(packager.PackageError):
        packager.write_vision_package(tmp_path, "v", [packager.StripItem("MSG-123", "assinado", photo())], 15)


def test_text_package_needs_names_removed_and_no_residual_personal_data(tmp_path):
    item = packager.TextItem(pseudonym(SECRET, "1"), "assinado", "texto limpo")
    with pytest.raises(packager.PackageError):
        packager.write_text_package(tmp_path, "laya-1", [item], 15, names_removed=False)
    dirty = packager.TextItem(pseudonym(SECRET, "2"), "assinado", "CNPJ 15.268.427/0001-42")
    with pytest.raises(packager.PackageError):
        packager.write_text_package(tmp_path, "laya-1", [dirty], 15, names_removed=True)
    manifest = packager.write_text_package(tmp_path, "laya-1", [item], 15, names_removed=True)
    assert manifest["nomes_removidos"] is True
