"""Service settings. Security-relevant values fail closed: the service refuses to start without them."""
import os
from collections.abc import Mapping
from dataclasses import dataclass

from .cnpj_key import cnpj_is_valid, digits

DEFAULT_LAYA_URL = "http://192.168.0.140:8010"
DEFAULT_CONFIDENCE_FIELD = "answer_confidence"
DEFAULT_LAYA_TIMEOUT = 10.0
DEFAULT_QUEUE_TIMEOUT = 25.0  # seconds a request may wait for a free worker
DEFAULT_MAX_CONCURRENCY = 2  # the host has 4 shared CPUs; OCR is CPU-bound
DEFAULT_MIN_INK_PIXELS = 400  # provisional: replace after calibrating on the gold set
DEFAULT_MIN_COMPONENTS = 1
DEFAULT_MIN_RELATIVE_INK = 3.0
DEFAULT_SIGNATURE_MODEL_TIMEOUT = 5.0
DEFAULT_SHADOW_TIMEOUT = 4.0
DEFAULT_SHADOW_THRESHOLD = 0.31  # just above the highest unsigned score on the gold set (spike, 2026-11-04)
DEFAULT_HERMES_MODEL = "hermes3:8b"
DEFAULT_HERMES_TIMEOUT = 8.0
DEFAULT_JUDGE_TIMEOUT = 20.0
DEFAULT_JUDGE_THRESHOLD = 0.6  # provisional: below this type confidence the judge transcribes the image
DEFAULT_JUDGE_SIGNATURE_MAX_CONFIDENCE = 89.0  # below the n8n If4 cutoff of 90 until the judge is measured on the gold set
DEFAULT_REVIEW_DEADLINE = 25.0  # seconds from the start of a run; the n8n node gives up at 30
DEFAULT_JUDGE_MAX_RUNNING = 2  # the vLLM must have fewer requests than this running before we add one
DEFAULT_JUDGE_MAX_WAITING = 0  # any queue on the vLLM means it is already behind
DEFAULT_JUDGE_MAX_CACHE = 0.7  # fraction of the KV cache in use above which we stay away
DEFAULT_JUDGE_MAX_IN_FLIGHT = 1  # judge calls this service may hold open at once
DEFAULT_JUDGE_MIN_VOTE_CONFIDENCE = 80.0  # provisional: below this the judge's vote does not count as signed
DEFAULT_JUDGE_READ_TIMEOUT = 8.0  # the signature vote still has to fit before the review deadline
DEFAULT_CONCLUSIVE_TYPES = "canhoto"  # other types go to the reserve until they are measured
TYPE_ENGINE_LAYA = "laya"
TYPE_ENGINE_RULES = "regras"  # rules and access key decide; Laya is only observed until it beats them
TYPE_ENGINES = (TYPE_ENGINE_LAYA, TYPE_ENGINE_RULES)
SIGNATURE_ENGINE_CURRENT = "atual"
SIGNATURE_ENGINE_TWO_VOTES = "dois_votos"  # released only when the detector and the judge both see a signature
SIGNATURE_ENGINE_JUDGE = "juiz"  # the judge alone releases until the detector is trained; the detector is only recorded
SIGNATURE_ENGINES = (SIGNATURE_ENGINE_CURRENT, SIGNATURE_ENGINE_TWO_VOTES, SIGNATURE_ENGINE_JUDGE)
ENABLED = "1"


class ConfigError(RuntimeError):
    """Raised at startup when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    whitelist_cnpj: tuple[str, ...]
    laya_url: str
    laya_api_key: str
    laya_confidence_field: str
    laya_timeout: float
    queue_timeout: float
    max_concurrency: int
    min_ink_pixels: int
    min_components: int
    min_relative_ink: float
    signature_model_url: str = ""  # empty keeps the ink rule as the only signature decider
    signature_model_timeout: float = DEFAULT_SIGNATURE_MODEL_TIMEOUT
    signature_shadow_url: str = ""  # empty turns the shadow detector off
    signature_shadow_api_key: str = ""
    signature_shadow_timeout: float = DEFAULT_SHADOW_TIMEOUT
    signature_shadow_threshold: float = DEFAULT_SHADOW_THRESHOLD
    hermes_url: str = ""  # empty: Laya failures go straight to the rules
    hermes_model: str = DEFAULT_HERMES_MODEL
    hermes_timeout: float = DEFAULT_HERMES_TIMEOUT
    judge_url: str = ""  # empty: no vision judge
    judge_model: str = ""
    judge_timeout: float = DEFAULT_JUDGE_TIMEOUT
    judge_threshold: float = DEFAULT_JUDGE_THRESHOLD
    judge_signature_max_confidence: float = DEFAULT_JUDGE_SIGNATURE_MAX_CONFIDENCE
    review_deadline: float = DEFAULT_REVIEW_DEADLINE
    judge_max_running: int = DEFAULT_JUDGE_MAX_RUNNING
    judge_max_waiting: int = DEFAULT_JUDGE_MAX_WAITING
    judge_max_cache: float = DEFAULT_JUDGE_MAX_CACHE
    judge_max_in_flight: int = DEFAULT_JUDGE_MAX_IN_FLIGHT
    judge_min_vote_confidence: float = DEFAULT_JUDGE_MIN_VOTE_CONFIDENCE
    judge_transcribe: bool = True  # off keeps the vLLM for signature votes; unread photos go to the reserve
    judge_read_key: bool = False  # off until the directed question is measured: an invented key would attach to a wrong note
    judge_read_timeout: float = DEFAULT_JUDGE_READ_TIMEOUT
    type_engine: str = TYPE_ENGINE_LAYA
    signature_engine: str = SIGNATURE_ENGINE_CURRENT
    conclusive_types: tuple[str, ...] = (DEFAULT_CONCLUSIVE_TYPES,)
    hermes_audit: bool = False
    decision_log_path: str = ""  # empty: decisions are not persisted
    decision_digest_key: str = ""  # empty: the log does not compare what the engines read


def _whitelist(raw: str) -> tuple[str, ...]:
    return tuple(value for value in (digits(item) for item in raw.split(",")) if value)


def _problems(api_key: str, whitelist: tuple[str, ...]) -> list[str]:
    problems = []
    if not api_key:
        problems.append("API_KEY ausente")
    if not whitelist:
        problems.append("WHITELIST_CNPJ vazia (sem ela qualquer CNPJ seria aceito)")
    problems += [f"CNPJ inválido na whitelist: {cnpj}" for cnpj in whitelist if not cnpj_is_valid(cnpj)]
    return problems


def load(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    api_key = env.get("API_KEY", "")
    whitelist = _whitelist(env.get("WHITELIST_CNPJ", ""))
    type_engine = env.get("TYPE_ENGINE", TYPE_ENGINE_LAYA)
    signature_engine = env.get("SIGNATURE_ENGINE", SIGNATURE_ENGINE_CURRENT)
    problems = _problems(api_key, whitelist)
    if type_engine not in TYPE_ENGINES:
        problems.append(f"TYPE_ENGINE inválido: {type_engine}")
    if signature_engine not in SIGNATURE_ENGINES:
        problems.append(f"SIGNATURE_ENGINE inválido: {signature_engine}")
    if problems:
        raise ConfigError("; ".join(problems))
    return Settings(
        api_key=api_key,
        whitelist_cnpj=whitelist,
        laya_url=env.get("LAYA_URL", DEFAULT_LAYA_URL).rstrip("/"),
        laya_api_key=env.get("LAYA_API_KEY", ""),
        laya_confidence_field=env.get("LAYA_CONFIDENCE_FIELD", DEFAULT_CONFIDENCE_FIELD),
        laya_timeout=float(env.get("LAYA_TIMEOUT", DEFAULT_LAYA_TIMEOUT)),
        queue_timeout=float(env.get("QUEUE_TIMEOUT", DEFAULT_QUEUE_TIMEOUT)),
        max_concurrency=int(env.get("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)),
        min_ink_pixels=int(env.get("SIGNATURE_MIN_INK_PX", DEFAULT_MIN_INK_PIXELS)),
        min_components=int(env.get("SIGNATURE_MIN_COMPONENTS", DEFAULT_MIN_COMPONENTS)),
        min_relative_ink=float(env.get("SIGNATURE_MIN_RELATIVE", DEFAULT_MIN_RELATIVE_INK)),
        signature_model_url=env.get("SIGNATURE_MODEL_URL", "").rstrip("/"),
        signature_model_timeout=float(env.get("SIGNATURE_MODEL_TIMEOUT", DEFAULT_SIGNATURE_MODEL_TIMEOUT)),
        signature_shadow_url=env.get("SIGNATURE_SHADOW_URL", "").rstrip("/"),
        signature_shadow_api_key=env.get("SIGNATURE_SHADOW_API_KEY", ""),
        signature_shadow_timeout=float(env.get("SIGNATURE_SHADOW_TIMEOUT", DEFAULT_SHADOW_TIMEOUT)),
        signature_shadow_threshold=float(env.get("SIGNATURE_SHADOW_THRESHOLD", DEFAULT_SHADOW_THRESHOLD)),
        hermes_url=env.get("HERMES_URL", "").rstrip("/"),
        hermes_model=env.get("HERMES_MODEL", DEFAULT_HERMES_MODEL),
        hermes_timeout=float(env.get("HERMES_TIMEOUT", DEFAULT_HERMES_TIMEOUT)),
        judge_url=env.get("JUDGE_URL", "").rstrip("/"),
        judge_model=env.get("JUDGE_MODEL", ""),
        judge_timeout=float(env.get("JUDGE_TIMEOUT", DEFAULT_JUDGE_TIMEOUT)),
        judge_threshold=float(env.get("JUDGE_THRESHOLD", DEFAULT_JUDGE_THRESHOLD)),
        judge_signature_max_confidence=float(
            env.get("JUDGE_SIGNATURE_MAX_CONFIDENCE", DEFAULT_JUDGE_SIGNATURE_MAX_CONFIDENCE)),
        review_deadline=float(env.get("REVIEW_DEADLINE", DEFAULT_REVIEW_DEADLINE)),
        judge_max_running=int(env.get("JUDGE_MAX_RUNNING", DEFAULT_JUDGE_MAX_RUNNING)),
        judge_max_waiting=int(env.get("JUDGE_MAX_WAITING", DEFAULT_JUDGE_MAX_WAITING)),
        judge_max_cache=float(env.get("JUDGE_MAX_CACHE", DEFAULT_JUDGE_MAX_CACHE)),
        judge_max_in_flight=int(env.get("JUDGE_MAX_IN_FLIGHT", DEFAULT_JUDGE_MAX_IN_FLIGHT)),
        judge_min_vote_confidence=float(env.get("JUDGE_MIN_VOTE_CONFIDENCE", DEFAULT_JUDGE_MIN_VOTE_CONFIDENCE)),
        judge_transcribe=env.get("JUDGE_TRANSCRIBE", ENABLED) == ENABLED,
        judge_read_key=env.get("JUDGE_READ_KEY", "") == ENABLED,
        judge_read_timeout=float(env.get("JUDGE_READ_TIMEOUT", DEFAULT_JUDGE_READ_TIMEOUT)),
        type_engine=type_engine,
        signature_engine=signature_engine,
        conclusive_types=tuple(item.strip() for item in env.get("CONCLUSIVE_TYPES", DEFAULT_CONCLUSIVE_TYPES).split(",")),
        hermes_audit=env.get("HERMES_AUDIT", "") == ENABLED,
        decision_log_path=env.get("DECISION_LOG_PATH", ""),
        decision_digest_key=env.get("DECISION_DIGEST_KEY", ""),
    )
