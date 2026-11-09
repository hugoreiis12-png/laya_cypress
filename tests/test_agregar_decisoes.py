import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("agregar", Path(__file__).parent / "e2e" / "agregar_decisoes.py")
agregar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agregar)


def decision(ms, wait=0, **extra):
    return {"evento": "decisao", "id": "SEGREDO123", "ts": "2026-11-09T10:00:00", "tipo": "canhoto",
            "latencia_ms": ms, "espera_fila_ms": wait, "inconclusivo": False, **extra}


def write(tmp_path, rows, junk=False):
    path = tmp_path / "decisoes.jsonl"
    lines = [json.dumps(row) for row in rows] + (["linha quebrada"] if junk else [])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def text(path, since=""):
    records, malformed = agregar.read_records(path, since)
    return "\n".join(agregar.report(records, malformed))


def test_it_counts_refusal_reasons_and_the_inconclusive_rate(tmp_path):
    rows = [decision(1000, capacidade_juiz="muitas_em_execucao", revisao_assinatura={"resultado": "sem_capacidade"},
                     inconclusivo=True, motivos_inconclusivo=["assinatura_nao_confirmada"]),
            decision(2000, capacidade_juiz="ok"), {"evento": "fila_cheia", "id": "x"}]
    out = text(write(tmp_path, rows))
    assert "muitas_em_execucao=1" in out and "sem_capacidade=1" in out and "assinatura_nao_confirmada=1" in out
    assert "inconclusivos: 1 de 2 (50.0%)" in out and "fila cheia (503): 1" in out


def test_it_reports_latency_percentiles(tmp_path):
    out = text(write(tmp_path, [decision(ms, wait=ms // 10) for ms in range(1000, 11000, 1000)]))
    assert "latencia_ms: p50=6000, p90=10000" in out and "espera_fila_ms:" in out


def test_it_counts_failures_by_engine(tmp_path):
    out = text(write(tmp_path, [decision(1, erros=[{"motor": "juiz_assinatura", "erro": "tempo"}])]))
    assert "juiz_assinatura=1" in out


def test_the_period_filter_and_bad_lines(tmp_path):
    rows = [decision(1000), {**decision(1000), "ts": "2026-11-01T10:00:00"}]
    path = write(tmp_path, rows, junk=True)
    assert "decisões: 1;" in text(path, "2026-11-09") and "linhas inválidas: 1" in text(path, "2026-11-09")


def test_the_report_never_prints_an_id_or_a_value(tmp_path):
    rows = [decision(1000, erros=[{"motor": "juiz", "erro": "texto do cliente"}], campos={"chave": "abc"})]
    out = text(write(tmp_path, rows))
    assert "SEGREDO123" not in out and "texto do cliente" not in out and "abc" not in out


def test_an_empty_period_is_reported_without_failing(tmp_path):
    assert "nenhuma decisão" in text(write(tmp_path, [decision(1)]), "2027-01-01")
