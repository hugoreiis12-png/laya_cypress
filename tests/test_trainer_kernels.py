import ast
from pathlib import Path
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


def _kernel_function(script, name, namespace):
    tree = ast.parse(script.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    constants = [node for node in tree.body if isinstance(node, ast.Assign)
                 and all(isinstance(target, ast.Name) and target.id in ("CLASSIFIER_SIZE", "PIXEL_SCALE")
                         for target in node.targets)]
    exec(compile(ast.Module(body=constants + nodes, type_ignores=[]), str(script), "exec"), namespace)
    return namespace[name]


def test_the_vision_kernel_measures_with_the_same_input_the_service_feeds_the_model():
    import cv2
    import numpy as np
    from service.app.vision.strip import classifier_input

    kernel_input = _kernel_function(cycle.KERNELS_DIR / "train_vision.py", "classifier_input", {"cv2": cv2, "np": np})
    for height, width in ((400, 900), (700, 300), (320, 320)):
        image = np.random.default_rng(height).integers(0, 255, (height, width, 3), dtype=np.uint8)
        assert np.array_equal(kernel_input(image), classifier_input(image))
        assert classifier_input(image).shape == (1, 3, 320, 320)


def test_the_vision_kernel_writes_what_the_service_loads():
    source = (cycle.KERNELS_DIR / "train_vision.py").read_text(encoding="utf-8")
    assert '"pesos.onnx"' in source and cycle.WEIGHTS_FILES["visao"] == "pesos.onnx"
    assert '"classes"' in source and '"amostras_teste"' in source


@pytest.mark.parametrize("script", KERNELS, ids=lambda path: path.name)
def test_kernels_find_the_package_however_kaggle_nests_the_mount(script, tmp_path):
    package = tmp_path / "datasets" / "owner" / "laya-visao-1"
    package.mkdir(parents=True)
    (package / "manifest.json").write_text("{}", encoding="utf-8")
    line = next(row for row in script.read_text(encoding="utf-8").splitlines() if row.startswith("INPUT = "))
    scope = {"Path": Path}
    exec(line.replace("/kaggle/input", tmp_path.as_posix()), scope)  # noqa: S102 - runs the kernel's own line
    assert scope["INPUT"] == package


def test_the_vision_kernel_turns_off_the_raytune_callback_before_training():
    tree = ast.parse((cycle.KERNELS_DIR / "train_vision.py").read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [ast.unparse(node.value) for node in main.body if isinstance(node, ast.Expr)]
    off = calls.index("settings.update({'raytune': False})")
    assert off < next(index for index, call in enumerate(calls) if call.startswith("model.train("))
