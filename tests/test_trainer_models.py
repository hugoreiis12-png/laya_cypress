import pytest

from trainer.app import gate, kaggle_client
from trainer.app.metrics import ClassError, Metrics, MetricsError, from_kernel
from trainer.app.registry import Registry, RegistryError

SAFE = {"nao_assinado_liberado": 0, "carimbo_liberado": 0}


def metrics(error=0.08, base_error=0.10, safety=None, base_safety=None, rules_error=None, test="teste-p15"):
    return Metrics("visao-0002", "visao-0001", test, error, base_error, rules_error, safety or dict(SAFE), base_safety,
                   0.9, {"assinado": ClassError(0.05, 0.03), "nao_assinado": ClassError(0.2, 0.15)})


def test_error_falling_twenty_percent_relative_promotes():
    verdict = gate.evaluate(metrics(error=0.08, base_error=0.10), 0.20)
    assert verdict.promote and verdict.reduction == pytest.approx(0.20)


def test_a_smaller_gain_does_not_promote():
    verdict = gate.evaluate(metrics(error=0.09, base_error=0.10), 0.20)
    assert not verdict.promote and "abaixo do mínimo" in verdict.reasons[0]


def test_example_from_the_brief_counts_as_a_big_error_reduction():
    assert gate.reduction_of(0.10, 0.02) == pytest.approx(0.80)


@pytest.mark.parametrize("key", ["nao_assinado_liberado", "carimbo_liberado"])
def test_releasing_an_unsigned_or_stamp_only_canhoto_blocks_promotion(key):
    verdict = gate.evaluate(metrics(error=0.01, safety={**SAFE, key: 1}), 0.20)
    assert not verdict.promote and any(key in reason for reason in verdict.reasons)


def test_missing_safety_metric_fails_closed():
    verdict = gate.evaluate(metrics(error=0.01, safety={"carimbo_liberado": 0}), 0.20)
    assert not verdict.promote and any("ausente" in reason for reason in verdict.reasons)


def test_a_safety_metric_that_got_worse_blocks_promotion():
    verdict = gate.evaluate(metrics(error=0.01, safety={**SAFE, "outra": 3}, base_safety={**SAFE, "outra": 1}), 0.20)
    assert not verdict.promote


def test_losing_to_the_rules_blocks_promotion():
    assert not gate.evaluate(metrics(error=0.05, rules_error=0.04), 0.20).promote


def test_no_comparable_base_registers_a_baseline_without_promoting():
    verdict = gate.evaluate(metrics(base_error=None), 0.20)
    assert verdict.baseline and not verdict.promote


def test_base_with_no_error_leaves_nothing_to_reduce():
    assert not gate.evaluate(metrics(error=0.0, base_error=0.0), 0.20).promote


RAW = {"teste": "teste-p15", "erro": 0.08, "por_classe": {"assinado": 0.03}, "seguranca": dict(SAFE), "confianca": 0.9}


def test_kernel_output_becomes_metrics_against_a_base_on_the_same_test():
    result = from_kernel(RAW, "visao-0002", "visao-0001", metrics())
    assert result.base_error == 0.08 and result.base == "visao-0001" and result.per_class["assinado"].base == 0.03
    assert result.per_class["assinado"].current == 0.03 and result.base_safety == SAFE


def test_a_base_measured_on_another_test_is_ignored():
    result = from_kernel(RAW, "visao-0002", "visao-0001", metrics(test="teste-p10"))
    assert result.base_error is None and result.base is None


@pytest.mark.parametrize("change", [{"erro": 1.5}, {"teste": ""}, {"por_classe": {}}, {"seguranca": {"x": -1}}, {"confianca": None}])
def test_invalid_kernel_metrics_are_rejected(change):
    with pytest.raises(MetricsError):
        from_kernel({**RAW, **change}, "v", None, None)


def test_metrics_round_trip_through_the_registry(tmp_path):
    weights = tmp_path / "pesos.bin"
    weights.write_bytes(b"pesos")
    registry = Registry(tmp_path / "registro")
    version = registry.next_version("visao")
    registry.register("visao", version, weights, metrics())
    assert version == "visao-0001" and registry.next_version("visao") == "visao-0002"
    assert registry.metrics_of(version) == metrics()


def test_shadow_history_and_rollback(tmp_path):
    weights = tmp_path / "pesos.bin"
    weights.write_bytes(b"pesos")
    registry = Registry(tmp_path / "registro")
    for name in ("visao-0001", "visao-0002"):
        registry.register("visao", name, weights, metrics())
    registry.mark_baseline("visao", "visao-0001")
    registry.promote_to_shadow("visao", "visao-0002")
    assert registry.current("visao") == "visao-0002"
    assert registry.rollback("visao") == "visao-0001" and registry.current("visao") == "visao-0001"
    assert registry.rollback("visao") is None


def test_unknown_version_is_an_error(tmp_path):
    with pytest.raises(RegistryError):
        Registry(tmp_path).mark_baseline("visao", "visao-9999")


def test_kaggle_metadata_is_private_gpu_and_points_at_its_dataset():
    assert kaggle_client.dataset_metadata("laya", "d")["id"] == "laya/d"
    kernel = kaggle_client.kernel_metadata("laya", "d")
    assert kernel["enable_gpu"] == "true" and kernel["is_private"] == "true" and kernel["dataset_sources"] == ["laya/d"]


@pytest.mark.parametrize("raw,expected", [("KernelWorkerStatus.COMPLETE", "completo"), ("error", "falhou"),
                                          ("QUEUED", "rodando"), ("running", "rodando")])
def test_kaggle_status_is_normalized(raw, expected):
    assert kaggle_client.normalize_status(raw) == expected


def test_quota_errors_are_told_apart_from_other_failures():
    assert isinstance(kaggle_client.translate(RuntimeError("GPU quota exceeded")), kaggle_client.QuotaExceeded)
    assert not isinstance(kaggle_client.translate(RuntimeError("boom")), kaggle_client.QuotaExceeded)
