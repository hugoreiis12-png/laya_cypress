"""Prometheus metrics of the service, counted from the records it writes: counts and versions only, never customer data."""
import logging
from collections.abc import Callable
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

log = logging.getLogger(__name__)
SIGNED = "assinado"
MS_PER_SECOND = 1000
LATENCY_BUCKETS = (0.5, 1, 2, 5, 10, 15, 20, 25, 30, 60)
UNKNOWN = "desconhecido"
DROPPED_PREFIX = "descartada_"


def _label(value: Any) -> str:
    return UNKNOWN if value is None else str(value).lower()


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.decisions = Counter("laya_decisoes", "Decisões por tipo, estado da assinatura e motor que decidiu",
                                 ["tipo", "estado_assinatura", "motor_assinatura", "inconclusivo"], registry=r)
        self.inconclusive = Counter("laya_inconclusivos", "Motivos que mandaram a foto ao Gemini", ["motivo"], registry=r)
        self.latency = Histogram("laya_decisao_latencia_segundos", "Tempo de uma decisão no serviço",
                                 buckets=LATENCY_BUCKETS, registry=r)
        self.engine_failures = Counter("laya_falhas_motor", "Falhas por motor", ["motor"], registry=r)
        self.judge_capacity = Counter("laya_capacidade_juiz", "Admissão do juiz no vLLM", ["resultado"], registry=r)
        self.second_look = Counter("laya_segunda_olhada", "Segunda olhada do juiz", ["resultado"], registry=r)
        self.queue_full = Counter("laya_fila_cheia", "Pedidos recusados com 503 por falta de worker", registry=r)
        self.vision = Counter("laya_visao_observacoes", "Opiniões da Laya visão", ["versao", "rotulo", "pronta"], registry=r)
        self.vision_takeover = Counter("laya_visao_assumiu", "Canhotos liberados pela Laya sem o juiz", ["versao"],
                                       registry=r)
        self.vision_agreement = Counter("laya_visao_concordancia", "Laya visão comparada ao voto do juiz",
                                        ["versao", "concorda"], registry=r)
        self.vision_errors = Counter("laya_visao_erros", "Laya visão sem opinião", ["erro"], registry=r)
        self.references = Counter("laya_referencias", "Referências gravadas por origem", ["origem"], registry=r)
        self.full_analysis = Counter("laya_analise_completa", "Análises completas do vLLM", ["resultado"], registry=r)
        self.inbox = Counter("laya_caixa_trainer", "Fotos deixadas para o trainer", ["resultado"], registry=r)
        self.full_queue = Gauge("laya_analise_completa_fila", "Fotos esperando o vLLM ocioso", registry=r)

    def track_queue(self, size: Callable[[], int]) -> None:
        self.full_queue.set_function(size)

    def observe(self, record: dict[str, Any]) -> None:
        """Never raises: a metric that cannot be counted must not fail the request."""
        try:
            self._dispatch(record)
        except (AttributeError, TypeError, ValueError) as exc:
            log.warning("métrica não contada: %s", exc)

    def _dispatch(self, record: dict[str, Any]) -> None:
        event = record.get("evento")
        if event == "decisao":
            self._decision(record)
        elif event == "referencia":
            self.references.labels(_label(record.get("origem"))).inc()
        elif event == "fila_cheia":
            self.queue_full.inc()
        elif event == "analise_completa_descartada":
            self.full_analysis.labels(DROPPED_PREFIX + _label(record.get("motivo"))).inc()
        elif event == "analise_completa_falhou":
            self.full_analysis.labels("falhou").inc()

    def observe_inbox(self, delivered: bool) -> None:
        self.inbox.labels("entregue" if delivered else "nao_entregue").inc()

    def _decision(self, record: dict[str, Any]) -> None:
        vision = record.get("laya_visao") or {}
        votes = record.get("votos") or {}
        self.decisions.labels(_label(record.get("tipo")), _label(record.get("estado_assinatura")),
                              _signature_engine(vision, votes), _label(bool(record.get("inconclusivo")))).inc()
        for reason in record.get("motivos_inconclusivo") or []:
            self.inconclusive.labels(_label(reason)).inc()
        for engine in record.get("falhas") or []:
            self.engine_failures.labels(_label(engine)).inc()
        if isinstance(record.get("latencia_ms"), (int, float)):
            self.latency.observe(record["latencia_ms"] / MS_PER_SECOND)
        if record.get("capacidade_juiz"):
            self.judge_capacity.labels(_label(record["capacidade_juiz"])).inc()
        if record.get("segunda_olhada"):
            self.second_look.labels(_label((record["segunda_olhada"] or {}).get("juiz"))).inc()
        if vision:
            self._vision(vision, votes)

    def _vision(self, vision: dict[str, Any], votes: dict[str, Any]) -> None:
        if "erro" in vision:
            self.vision_errors.labels(_label(vision["erro"])).inc()
            return
        version = _label(vision.get("versao"))
        self.vision.labels(version, _label(vision.get("rotulo")), _label(vision.get("pronta"))).inc()
        if vision.get("assumiu"):
            self.vision_takeover.labels(version).inc()
        if isinstance(votes.get("juiz"), bool):
            agrees = (vision.get("rotulo") == SIGNED) == votes["juiz"]
            self.vision_agreement.labels(version, "sim" if agrees else "nao").inc()


def _signature_engine(vision: dict[str, Any], votes: dict[str, Any]) -> str:
    if vision.get("assumiu"):
        return "laya_visao"
    return "juiz" if votes.get("juiz") is not None else "outro"
