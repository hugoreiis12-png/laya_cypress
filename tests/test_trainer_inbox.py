import json
import os
import time

import cv2
import numpy as np
import pytest

from tests.trainer_support import settings
from trainer.app import inbox
from trainer.app.anonymizer import pseudonym
from trainer.app.cycle import DataError
from trainer.app.dataset import InboxData
from trainer.app.state import KIND_LAYA, KIND_VISION

SECRET = "x" * 24
NOW = time.time()
LONG_TEXT = "RECEBEMOS DE NATURAL FARMS AGRICOLA LTDA OS PRODUTOS CONSTANTES NA NOTA FISCAL Ricardo Silva"


def photo_bytes() -> bytes:
    image = np.full((400, 300, 3), 255, np.uint8)
    return cv2.imencode(".jpg", image)[1].tobytes()


def drop(directory, name, data=None, age_days=0):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(photo_bytes() if data is None else data)
    os.utime(path, (NOW - age_days * 86400, NOW - age_days * 86400))
    return path


def ingest(tmp_path, known):
    cutters = [inbox.strip_cutter(tmp_path / "strips", SECRET)]
    return inbox.ingest(tmp_path / "in", known, cutters, lambda: NOW)


def test_a_photo_with_a_known_decision_becomes_a_strip_and_the_original_is_deleted(tmp_path):
    original = drop(tmp_path / "in", "AAA111.jpg")
    assert ingest(tmp_path, {"AAA111": 0}).cut == 1 and not original.exists()
    assert inbox.strip_path(tmp_path / "strips", SECRET, "AAA111").is_file()


def test_the_strip_is_named_by_pseudonym_never_by_the_raw_id(tmp_path):
    drop(tmp_path / "in", "AAA111.jpg")
    ingest(tmp_path, {"AAA111": 0})
    names = [path.stem for path in (tmp_path / "strips").iterdir()]
    assert names == [pseudonym(SECRET, "AAA111")] and "AAA111" not in names


def test_a_photo_without_decision_waits_and_then_is_dropped(tmp_path):
    fresh = drop(tmp_path / "in", "NEW1.jpg")
    old = drop(tmp_path / "in", "OLD1.jpg", age_days=4)
    report = ingest(tmp_path, {})
    assert report.waiting == 1 and report.dropped == 1 and fresh.exists() and not old.exists()


def test_unsafe_names_and_unreadable_images_are_dropped(tmp_path):
    odd = drop(tmp_path / "in", "a b$.jpg")
    broken = drop(tmp_path / "in", "BAD1.jpg", data=b"not an image")
    report = ingest(tmp_path, {"BAD1": 0, "a b$": 0})
    assert report.dropped == 2 and not odd.exists() and not broken.exists()
    assert not (tmp_path / "strips").exists() or not list((tmp_path / "strips").iterdir())


def test_old_derivatives_without_consensus_expire(tmp_path):
    old = drop(tmp_path / "strips", "o.jpg", age_days=31)
    recent = drop(tmp_path / "strips", "r.jpg", age_days=1)
    assert inbox.prune(tmp_path / "strips", lambda: NOW) == 1
    assert not old.exists() and recent.exists()


def write_log(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def rows_for(item_id, vllm, gemini, doc_type="canhoto", rules_type="canhoto"):
    return [{"evento": "decisao", "id": item_id, "votos": {"juiz": vllm, "confianca_juiz": 95.0}, "rotacao": 0,
             "tipo": rules_type, "motor": "regras"},
            {"evento": "referencia", "id": item_id, "origem": "gemini", "possui_assinatura": gemini, "tipo": doc_type}]


def data_source(tmp_path, cases, kinds=(KIND_VISION, KIND_LAYA)):
    config = settings(state_dir=tmp_path / "state", decisions_path=tmp_path / "decisoes.jsonl", kinds=kinds)
    write_log(config.decisions_path, [row for case in cases for row in rows_for(*case)])
    for case in cases:
        drop(config.state_dir / "inbox", f"{case[0]}.jpg")
    return InboxData(config, lambda: NOW, read=lambda data: LONG_TEXT), config


def test_only_photos_where_vllm_and_gemini_agree_count_as_new_signature_samples(tmp_path):
    source, _ = data_source(tmp_path, [("P1", True, True), ("P2", False, False), ("P3", True, False)], (KIND_VISION,))
    assert source.new_samples({}) == {KIND_VISION: 2}
    assert source.new_samples({KIND_VISION: 2}) == {KIND_VISION: 0}


def test_type_samples_need_rules_and_gemini_to_agree(tmp_path):
    cases = [("T1", True, True, "pedido", "pedido"), ("T2", True, True, "canhoto", "pedido")]
    source, _ = data_source(tmp_path, cases)
    assert source.new_samples({})[KIND_LAYA] == 1


def test_the_vision_package_holds_pseudonyms_labels_and_both_classes(tmp_path):
    ids = [("S1", True, True), ("S2", True, True), ("U1", False, False), ("U2", False, False)]
    source, config = data_source(tmp_path, ids)
    count = source.prepare(KIND_VISION, tmp_path / "pacote")
    files = {path.stem for path in (tmp_path / "pacote").rglob("*.jpg")}
    assert count == 4 and files == {pseudonym(config.id_secret, item_id) for item_id, _, _ in ids}
    manifest = json.loads((tmp_path / "pacote" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["anonimizado"] is True and sum(manifest["contagens"].values()) == 4


def test_a_package_with_a_single_class_is_refused(tmp_path):
    source, _ = data_source(tmp_path, [("S1", True, True), ("S2", True, True), ("S3", True, True)])
    with pytest.raises(DataError, match="classes"):
        source.prepare(KIND_VISION, tmp_path / "pacote")


def test_the_text_package_never_holds_a_name_or_the_raw_id(tmp_path):
    cases = [(f"{kind}{n}", True, True, kind, kind) for kind in ("canhoto", "pedido") for n in range(2)]
    source, config = data_source(tmp_path, cases)
    assert source.prepare(KIND_LAYA, tmp_path / "pacote") == 4
    content = "".join(path.read_text(encoding="utf-8") for path in (tmp_path / "pacote").glob("*.jsonl"))
    assert "Ricardo" not in content and "NATURAL" not in content and "canhoto0" not in content
    assert "RECEBEMOS" in content and pseudonym(config.id_secret, "canhoto0") in content


def test_a_photo_with_too_little_text_yields_no_type_sample(tmp_path):
    config = settings(state_dir=tmp_path / "state", decisions_path=tmp_path / "decisoes.jsonl", kinds=(KIND_LAYA,))
    write_log(config.decisions_path, rows_for("E1", True, True))
    original = drop(config.state_dir / "inbox", "E1.jpg")
    source = InboxData(config, lambda: NOW, read=lambda data: "ab")
    assert source.new_samples({}) == {KIND_LAYA: 0} and not original.exists()


def test_a_kind_that_is_not_enabled_is_refused(tmp_path):
    source, _ = data_source(tmp_path, [("S1", True, True)], (KIND_VISION,))
    with pytest.raises(DataError):
        source.prepare(KIND_LAYA, tmp_path / "pacote")


def test_without_a_decision_log_there_is_nothing_new(tmp_path):
    config = settings(state_dir=tmp_path / "state", decisions_path=tmp_path / "ausente.jsonl", kinds=(KIND_VISION,))
    assert InboxData(config).new_samples({}) == {KIND_VISION: 0}
