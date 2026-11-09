import pytest

from tests.trainer_support import ENV, at, settings
from trainer.app import config, scheduler
from trainer.app.state import KIND_LAYA, KIND_VISION, SCHEDULED, StateError, TrainerState, load_state, save_state

ENOUGH = {KIND_LAYA: 500, KIND_VISION: 500}
NIGHT = at(23)


def lazy(samples):
    return lambda: samples


def test_window_wraps_past_midnight():
    window = config.parse_window("22:00-06:00")
    assert scheduler.in_window(at(23), window) and scheduler.in_window(at(3), window)
    assert not scheduler.in_window(at(12), window)


def test_next_window_start_is_the_coming_evening():
    window = config.parse_window("22:00-06:00")
    assert scheduler.next_window_start(at(12), window) == at(22)
    assert scheduler.next_window_start(at(23), window) == at(22, day=7)


def test_outside_the_window_it_waits_for_the_next_one():
    action = scheduler.next_action(TrainerState(), at(12), lazy(ENOUGH), settings())
    assert action.name == scheduler.WAIT and action.until == at(22)


def test_pause_between_jobs_blocks_a_new_one_and_skips_counting():
    state = TrainerState(wait_until=NIGHT + 120)

    def forbidden():
        raise AssertionError("não deve contar amostras durante a pausa")

    action = scheduler.next_action(state, NIGHT, forbidden, settings())
    assert action.name == scheduler.WAIT and action.until == NIGHT + 120


def test_running_job_is_polled_not_restarted():
    state = TrainerState(running={"kind": KIND_LAYA, "kernel": "laya/x"})
    assert scheduler.next_action(state, NIGHT, lazy(ENOUGH), settings()).name == scheduler.POLL


def test_kinds_alternate_one_at_a_time():
    config_settings = settings()
    state = scheduler.finish_job(TrainerState(), KIND_LAYA, NIGHT, config_settings, samples=300)
    assert state.next_kind == KIND_VISION and state.wait_until == NIGHT + config_settings.pause_seconds
    after_pause = scheduler.next_action(state, state.wait_until, lazy(ENOUGH), config_settings)
    assert after_pause.name == scheduler.RUN and after_pause.kind == KIND_VISION


def test_a_kind_without_enough_new_samples_yields_to_the_other():
    action = scheduler.next_action(TrainerState(next_kind=KIND_LAYA), NIGHT, lazy({KIND_LAYA: 3, KIND_VISION: 900}), settings())
    assert action.kind == KIND_VISION


def test_without_new_samples_it_stays_idle():
    assert scheduler.next_action(TrainerState(), NIGHT, lazy({}), settings()).name == scheduler.IDLE_ACTION


def test_quota_exhaustion_schedules_the_same_kind_for_the_next_day():
    running = scheduler.start_job(TrainerState(), KIND_LAYA, {"kernel": "k"}, NIGHT)
    failed = scheduler.fail_job(running, KIND_LAYA, NIGHT, settings(), quota=True)
    assert failed.phase == SCHEDULED and failed.running is None and failed.next_kind == KIND_LAYA
    assert failed.wait_until == NIGHT + 24 * 3600


def test_failures_back_off_then_give_up_and_move_on():
    cfg = settings(max_retries=2)
    state = TrainerState(next_kind=KIND_LAYA)
    first = scheduler.fail_job(state, KIND_LAYA, NIGHT, cfg, quota=False)
    second = scheduler.fail_job(first, KIND_LAYA, NIGHT, cfg, quota=False)
    third = scheduler.fail_job(second, KIND_LAYA, NIGHT, cfg, quota=False)
    assert (first.retries, second.retries) == (1, 2) and second.wait_until > first.wait_until
    assert third.retries == 0 and third.next_kind == KIND_VISION


def test_running_kind_needs_a_job():
    with pytest.raises(StateError):
        scheduler.running_kind(TrainerState())


def test_state_survives_a_restart(tmp_path):
    path = tmp_path / "estado.json"
    state = scheduler.start_job(TrainerState(notified=("visao-0001",)), KIND_VISION, {"kernel": "laya/k"}, NIGHT)
    save_state(path, state)
    assert load_state(path) == state


def test_corrupt_state_is_reported_not_reset(tmp_path):
    path = tmp_path / "estado.json"
    path.write_text("{ não é json", encoding="utf-8")
    with pytest.raises(StateError):
        load_state(path)


@pytest.mark.parametrize("override", [{"TRAINER_ID_SECRET": "curta"}, {"KAGGLE_USERNAME": ""},
                                      {"TRAINER_PAUSE_SECONDS": "30"}, {"TRAINER_KINDS": "outro"},
                                      {"TRAINER_TEST_PERCENT": "100"}])
def test_invalid_config_refuses_to_start(override):
    with pytest.raises(config.ConfigError):
        config.load({**ENV, **override})
