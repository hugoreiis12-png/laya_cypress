from pathlib import Path

from evalkit import manifest


def make_image(tmp: Path, name="doc.jpg", content=b"fake-image") -> Path:
    path = tmp / name
    path.write_bytes(content)
    return path


def test_add_image_is_idempotent_by_sha256(tmp_path):
    records: list = []
    src = make_image(tmp_path)
    assert manifest.add_image(src, tmp_path / "gold", records, "whatsapp") is not None
    assert manifest.add_image(src, tmp_path / "gold", records, "whatsapp") is None
    assert len(records) == 1 and records[0]["status"] == "unlabeled"


def test_split_is_deterministic_and_stable(tmp_path):
    records: list = []
    manifest.add_image(make_image(tmp_path), tmp_path / "gold", records, "whatsapp")
    assert manifest.split_of(records[0]) == manifest.split_of(dict(records[0]))


def test_validate_flags_bad_labels(tmp_path):
    records: list = []
    root = tmp_path / "gold"
    rec = manifest.add_image(make_image(tmp_path), root, records, "whatsapp")
    rec.update(status="labeled", tipo_documento="canhoto", emitente={"cnpj": "123"})
    errors = manifest.validate(rec, root)
    assert any("possui_assinatura" in e for e in errors)
    assert any("emitente.cnpj" in e for e in errors)


def test_validate_detects_tampered_image(tmp_path):
    records: list = []
    root = tmp_path / "gold"
    rec = manifest.add_image(make_image(tmp_path), root, records, "whatsapp")
    (root / rec["image"]).write_bytes(b"changed")
    assert manifest.validate(rec, root) == ["sha256 não confere com a imagem"]


def test_discarded_record_is_valid_without_image_and_still_blocks_reimport(tmp_path):
    records: list = []
    root = tmp_path / "gold"
    src = make_image(tmp_path)
    rec = manifest.add_image(src, root, records, "whatsapp")
    (root / rec["image"]).unlink()
    rec["status"] = manifest.STATUS_DISCARDED
    assert manifest.validate(rec, root) == []
    assert manifest.add_image(src, root, records, "whatsapp") is None


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "m.jsonl"
    manifest.save(path, [{"id": "x", "nome": "Rúcula"}])
    assert manifest.load(path) == [{"id": "x", "nome": "Rúcula"}]
