from fastapi.testclient import TestClient

from service.app import config, main
from service.app.pipeline import Pipeline
from service.app.trainer_inbox import drop
from tests.test_service_engines import BASE_ENV, canhoto
from tests.test_service_pipeline import FakeLaya
from trainer.app import inbox

JPEG, PNG = b"\xff\xd8foto", b"\x89PNGfoto"


def test_the_photo_is_left_under_its_id(tmp_path):
    assert drop(str(tmp_path), "AB12CD34EF56", JPEG) is True
    assert drop(str(tmp_path), "PNG1", PNG) is True
    assert (tmp_path / "AB12CD34EF56.jpg").read_bytes() == JPEG and (tmp_path / "PNG1.png").is_file()
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_nothing_is_left_without_a_folder_or_a_safe_id(tmp_path):
    assert drop("", "AB12", JPEG) is False
    assert drop(str(tmp_path), None, JPEG) is False
    assert drop(str(tmp_path), "../fora", JPEG) is False
    assert list(tmp_path.iterdir()) == []


def test_a_missing_folder_never_raises(tmp_path):
    assert drop(str(tmp_path / "nao_existe"), "AB12", JPEG) is False


def test_the_trainer_skips_a_photo_still_being_written(tmp_path):
    (tmp_path / ".AB12.jpg.part").write_bytes(JPEG)
    report = inbox.ingest(tmp_path, {}, [])
    assert report == inbox.IngestReport() and (tmp_path / ".AB12.jpg.part").is_file()


def test_the_endpoint_leaves_the_photo_for_the_trainer(tmp_path):
    settings = config.load({**BASE_ENV, "TRAINER_INBOX_DIR": str(tmp_path)})
    client = TestClient(main.create_app(settings, Pipeline(settings, lambda data: canhoto(), FakeLaya())))
    photo = {"imagem_base64": "eA==", "contexto": {"id": "abc"}}
    assert client.post("/v1/extrair-documento", json=photo, headers={"Authorization": "Bearer k"}).status_code == 200
    assert (tmp_path / "abc.jpg").read_bytes() == b"x"
