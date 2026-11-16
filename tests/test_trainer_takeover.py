from dataclasses import replace

from tests.test_trainer_models import metrics
from trainer.app import gate, takeover
from trainer.app.registry import BASELINE
from trainer.app.report import build_report, fallback_text, headline, unknown_numbers

FLOOR, MIN_TEST = 0.90, 100


def test_ninety_percent_on_a_large_safe_test_is_ready():
    readiness = takeover.evaluate(replace(metrics(error=0.08), test_count=150), FLOOR, MIN_TEST)
    assert readiness.ready and readiness.reasons == ()


def test_below_the_floor_is_not_ready_even_when_it_improved():
    readiness = takeover.evaluate(replace(metrics(error=0.12), test_count=150), FLOOR, MIN_TEST)
    assert not readiness.ready and "abaixo do mínimo" in readiness.reasons[0]


def test_an_unknown_or_small_test_is_not_ready():
    assert not takeover.evaluate(metrics(error=0.0), FLOOR, MIN_TEST).ready
    assert not takeover.evaluate(replace(metrics(error=0.0), test_count=20), FLOOR, MIN_TEST).ready


def test_a_released_unsigned_canhoto_blocks_the_takeover():
    unsafe = replace(metrics(error=0.01, safety={"nao_assinado_liberado": 1, "carimbo_liberado": 0}), test_count=150)
    assert not takeover.evaluate(unsafe, FLOOR, MIN_TEST).ready


def report_for(error, status, test_count=150):
    item = replace(metrics(error=error), test_count=test_count)
    return build_report(item, gate.evaluate(item, 0.20), 0.20, ["visao"], status,
                        takeover.evaluate(item, FLOOR, MIN_TEST))


def test_the_report_says_whether_the_version_may_take_over_with_numbers_from_the_report():
    ready, waiting = report_for(0.05, BASELINE), report_for(0.2, BASELINE, test_count=10)
    assert headline(ready).endswith("pronta para assumir") and "referência" in headline(ready)
    assert "Pronta para assumir: não" in fallback_text(waiting)
    assert unknown_numbers(fallback_text(ready), ready) == [] and unknown_numbers(fallback_text(waiting), waiting) == []
