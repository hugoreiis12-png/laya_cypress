from dataclasses import replace

from tests.test_trainer_models import metrics
from trainer.app import control, scheduler
from trainer.app.alerts import RESUME_COMMAND, halt_message
from trainer.app.registry import REVERTED, SHADOW, Registry
from trainer.app.state import HALTED, KIND_VISION, TrainerState, load_state, save_state

FAILURE = {"tipo": KIND_VISION, "motivo": "kernel falhou", "em": "2026-11-16T23:00:00",
           "backup": "/srv/trainer/backups/visao-1", "job": {"dataset": "laya-visao-1"}}


def registry_with_two_shadows(root):
    registry = Registry(root / "registro")
    for version in ("visao-0001", "visao-0002"):
        weights = root / f"{version}.bin"
        weights.write_bytes(version.encode())
        registry.register(KIND_VISION, version, weights, replace(metrics(), version=version))
        registry.promote_to_shadow(KIND_VISION, version)
    return registry


def test_status_shows_the_halt_and_where_the_backup_is(tmp_path):
    save_state(tmp_path / "estado.json", scheduler.halt(TrainerState(), KIND_VISION, FAILURE))
    lines = control.run(["status"], tmp_path)
    assert f"fase: {HALTED}" in lines and any(FAILURE["backup"] in line for line in lines)


def test_resume_frees_a_halted_trainer_and_points_at_the_backup(tmp_path):
    save_state(tmp_path / "estado.json", scheduler.halt(TrainerState(), KIND_VISION, FAILURE))
    control.run(["retomar"], tmp_path)
    state = load_state(tmp_path / "estado.json")
    assert state.halted is None and state.resume_from == FAILURE["backup"]


def test_resume_of_a_running_trainer_is_an_error_not_a_change(tmp_path, capsys, monkeypatch):
    save_state(tmp_path / "estado.json", TrainerState())
    monkeypatch.setenv("TRAINER_STATE_DIR", str(tmp_path))
    assert control.main(["retomar"]) == 1 and "não está parado" in capsys.readouterr().err


def test_versions_lists_each_version_and_marks_the_current_one(tmp_path):
    registry_with_two_shadows(tmp_path)
    lines = control.run(["versoes", KIND_VISION], tmp_path)
    assert len(lines) == 2 and lines[1].startswith("visao-0002") and lines[1].endswith("<- atual")


def test_rollback_reverts_the_last_shadow_version(tmp_path):
    registry = registry_with_two_shadows(tmp_path)
    control.run(["rollback", KIND_VISION], tmp_path)
    assert registry.versions()["visao-0002"]["status"] == REVERTED and registry.current(KIND_VISION) == "visao-0001"
    assert [item.version for item in registry.shadow_metrics()] == ["visao-0001"]
    assert registry.versions()["visao-0001"]["status"] == SHADOW


def test_rollback_without_a_shadow_version_changes_nothing(tmp_path):
    assert control.run(["rollback", KIND_VISION], tmp_path) == ["nenhuma versão de visao em sombra para reverter"]


def test_halt_alert_says_what_failed_and_how_to_resume_without_customer_data():
    subject, body = halt_message(FAILURE)
    assert KIND_VISION in subject and FAILURE["motivo"] in body and RESUME_COMMAND in body


def test_the_entrypoint_stops_instead_of_looping_when_halted(tmp_path):
    from tests.trainer_support import settings
    from trainer.app import main

    save_state(tmp_path / "estado.json", scheduler.halt(TrainerState(), KIND_VISION, FAILURE))
    assert main._stopped_by_failure(main.build_cycle(settings(state_dir=tmp_path))) is True
