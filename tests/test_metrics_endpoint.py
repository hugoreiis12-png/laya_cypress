from fastapi.testclient import TestClient

from service.app import config, main
from service.app.observability import Metrics
from service.app.pipeline import Pipeline
from tests.test_service_engines import BASE_ENV, canhoto
from tests.test_service_pipeline import FakeLaya

PHOTO = {"imagem_base64": "eA==", "contexto": {"id": "abc"}}
API, SCRAPER = {"Authorization": "Bearer k"}, {"Authorization": "Bearer m"}


def client(tmp_path, **env):
    settings = config.load({**BASE_ENV, "DECISION_LOG_PATH": str(tmp_path / "d.jsonl"), **env})
    return TestClient(main.create_app(settings, Pipeline(settings, lambda data: canhoto(), FakeLaya())))


def value(metrics, name, **labels):
    return metrics.registry.get_sample_value(name, labels or None)


def test_metrics_are_off_without_their_own_key(tmp_path):
    assert client(tmp_path).get("/metrics", headers=API).status_code == 404


def test_the_api_key_does_not_open_the_metrics(tmp_path):
    assert client(tmp_path, METRICS_API_KEY="m").get("/metrics", headers=API).status_code == 401


def test_a_decision_shows_up_in_the_scrape_without_its_id(tmp_path):
    api = client(tmp_path, METRICS_API_KEY="m")
    api.post("/v1/extrair-documento", json=PHOTO, headers=API)
    api.post("/v1/referencia", json={"id": "abc", "origem": "gemini", "output": {"tipo_documento": "canhoto"}}, headers=API)
    page = api.get("/metrics", headers=SCRAPER)
    assert page.status_code == 200 and "text/plain" in page.headers["content-type"]
    assert 'laya_decisoes_total{estado_assinatura=' in page.text and 'laya_referencias_total{origem="gemini"} 1.0' in page.text
    assert "laya_decisao_latencia_segundos_count 1.0" in page.text and "abc" not in page.text


def test_laya_vision_agreement_takeover_and_errors_are_counted():
    metrics = Metrics()
    vision = {"versao": "visao-0002", "rotulo": "assinado", "confianca": 0.97, "pronta": True}
    metrics.observe({"evento": "decisao", "tipo": "canhoto", "laya_visao": {**vision, "assumiu": False},
                     "votos": {"juiz": True}})
    metrics.observe({"evento": "decisao", "tipo": "canhoto", "laya_visao": {**vision, "assumiu": False},
                     "votos": {"juiz": False}})
    metrics.observe({"evento": "decisao", "tipo": "canhoto", "laya_visao": {**vision, "assumiu": True}})
    metrics.observe({"evento": "decisao", "tipo": "canhoto", "laya_visao": {"erro": "sem_registro", "assumiu": False}})
    assert value(metrics, "laya_visao_concordancia_total", versao="visao-0002", concorda="sim") == 1
    assert value(metrics, "laya_visao_concordancia_total", versao="visao-0002", concorda="nao") == 1
    assert value(metrics, "laya_visao_assumiu_total", versao="visao-0002") == 1
    assert value(metrics, "laya_visao_erros_total", erro="sem_registro") == 1


def test_queue_events_and_a_malformed_record_never_raise():
    metrics = Metrics()
    metrics.observe({"evento": "fila_cheia", "id": "x"})
    metrics.observe({"evento": "analise_completa_descartada", "motivo": "vllm_ocupado"})
    metrics.observe({"evento": "decisao", "segunda_olhada": "quebrado"})
    assert value(metrics, "laya_fila_cheia_total") == 1
    assert value(metrics, "laya_analise_completa_total", resultado="descartada_vllm_ocupado") == 1
