from types import SimpleNamespace

import pytest

from trainer.app.kaggle_client import KaggleClient, KaggleError, QuotaExceeded


class FakeApi:
    def __init__(self, create_error=None, push_error=None, deleted=True):
        self.create_error, self.push_error, self.deleted, self.delete_args = create_error, push_error, deleted, None

    def dataset_create_new(self, folder, public, quiet, dir_mode):
        return SimpleNamespace(error=self.create_error)

    def kernels_push(self, folder):
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
