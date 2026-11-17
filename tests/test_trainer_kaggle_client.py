from types import SimpleNamespace

import pytest

from trainer.app.kaggle_client import KaggleClient, KaggleError, QuotaExceeded, kernel_metadata


class FakeApi:
    def __init__(self, create_error=None, push_error=None, deleted=True, statuses=("ready",)):
        self.create_error, self.push_error, self.deleted, self.delete_args = create_error, push_error, deleted, None
        self.statuses, self.pushed = list(statuses), False

    def dataset_status(self, dataset):
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if isinstance(status, int):
            error = RuntimeError(f"{status} Client Error")
            error.response = SimpleNamespace(status_code=status, text="")
            raise error
        return status

    def dataset_create_new(self, folder, public, quiet, dir_mode):
        return SimpleNamespace(error=self.create_error)

    def kernels_push(self, folder):
        self.pushed = True
        return SimpleNamespace(error=self.push_error, invalid_dataset_sources=[])

    def dataset_delete(self, owner_slug, dataset_slug, no_confirm=False):
        self.delete_args = (owner_slug, dataset_slug, no_confirm)
        return self.deleted


def test_a_refusal_returned_in_the_create_response_stops_the_job(tmp_path):
    with pytest.raises(KaggleError, match="slug em uso"):
        KaggleClient("laya", FakeApi(create_error="slug em uso")).create_dataset(tmp_path, "laya-visao-1")


def test_a_quota_refusal_in_the_response_is_still_a_quota_wait(tmp_path):
    script = tmp_path / "treino.py"
    script.write_text("print('ok')", encoding="utf-8")
    with pytest.raises(QuotaExceeded):
        KaggleClient("laya", FakeApi(push_error="GPU quota exceeded")).push_kernel(tmp_path, "laya-visao-1", script)


def test_the_dataset_is_deleted_by_owner_and_slug_without_an_interactive_prompt():
    api = FakeApi()
    KaggleClient("laya", api).delete_dataset("laya-visao-1")
    assert api.delete_args == ("laya", "laya-visao-1", True)


def test_an_unconfirmed_deletion_is_an_error_so_the_leftover_is_reported():
    with pytest.raises(KaggleError):
        KaggleClient("laya", FakeApi(deleted=False)).delete_dataset("laya-visao-1")


def _script(tmp_path):
    script = tmp_path / "treino.py"
    script.write_text("print('ok')", encoding="utf-8")
    return script


def test_the_kernel_is_pushed_only_after_the_dataset_is_ready(tmp_path):
    api, waits = FakeApi(statuses=("pending", "pending", "ready")), []
    KaggleClient("laya", api, sleep=waits.append).push_kernel(tmp_path, "laya-visao-1", _script(tmp_path))
    assert api.pushed and len(waits) == 2


def test_a_dataset_that_never_gets_ready_stops_the_job_without_pushing(tmp_path):
    api = FakeApi(statuses=("pending",))
    with pytest.raises(KaggleError, match="não ficou pronto"):
        KaggleClient("laya", api, sleep=lambda _: None).push_kernel(tmp_path, "laya-visao-1", _script(tmp_path))
    assert not api.pushed


def test_the_http_body_is_kept_in_the_error_message(tmp_path):
    class Refusing(FakeApi):
        def kernels_push(self, folder):
            error = RuntimeError("409 Client Error: Conflict")
            error.response = SimpleNamespace(text='{"message":"dataset not ready"}')
            raise error

    with pytest.raises(KaggleError, match="dataset not ready"):
        KaggleClient("laya", Refusing()).push_kernel(tmp_path, "laya-visao-1", _script(tmp_path))


def test_a_dataset_not_visible_yet_is_waited_for_instead_of_failing(tmp_path):
    api, waits = FakeApi(statuses=(403, 403, "ready")), []
    KaggleClient("laya", api, sleep=waits.append).push_kernel(tmp_path, "laya-visao-1", _script(tmp_path))
    assert api.pushed and len(waits) == 2


def test_an_unexpected_status_error_still_stops_the_job(tmp_path):
    with pytest.raises(KaggleError):
        KaggleClient("laya", FakeApi(statuses=(500,)), sleep=lambda _: None).push_kernel(tmp_path, "laya-visao-1", _script(tmp_path))


def test_the_kernel_never_shares_its_title_with_the_dataset(tmp_path):
    api = FakeApi()
    ref = KaggleClient("laya", api).push_kernel(tmp_path, "laya-visao-1", _script(tmp_path))
    meta = kernel_metadata("laya", "laya-visao-1")
    assert ref == "laya/laya-visao-1-treino" == meta["id"]
    assert meta["title"] != "laya-visao-1" and meta["dataset_sources"] == ["laya/laya-visao-1"]
