import ast
import pytest

from trainer.app import cycle
from trainer.app.packager import TextItem, write_text_package
from trainer.app.state import KINDS

KERNELS = list(cycle.KERNEL_SCRIPTS.values())


def test_every_kind_the_trainer_can_run_has_a_kernel_script():
    assert set(cycle.KERNEL_SCRIPTS) == set(KINDS) and all(path.is_file() for path in KERNELS)


@pytest.mark.parametrize("script", KERNELS, ids=lambda path: path.name)
def test_kernel_scripts_are_valid_python(script):
    ast.parse(script.read_text(encoding="utf-8"))


@pytest.mark.parametrize("script", KERNELS, ids=lambda path: path.name)
def test_kernels_identify_the_test_by_the_package_manifest_not_the_dataset_name(script):
    source = script.read_text(encoding="utf-8")
    assert 'manifest.json' in source and '["teste"]' in source and "INPUT.name" not in source


def test_the_laya_kernel_reads_what_the_text_package_writes(tmp_path):
    write_text_package(tmp_path, "p", [TextItem("a" * 16, "canhoto", "RECEBEMOS [X] NOTA FISCAL")], 15, True)
    source = (cycle.KERNELS_DIR / "train_laya.py").read_text(encoding="utf-8")
    for name in ("train.jsonl", "test.jsonl"):
        assert (tmp_path / name).is_file() and f'"{name}"' in source
    assert '"texto"' in source and '"rotulo"' in source
