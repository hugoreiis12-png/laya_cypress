"""RF-DETR signature detector (Apache 2.0 weights from the Hugging Face hub), loaded once per process."""
import numpy as np

from .assess import Box

MODEL_ID = "Mo-Awadalla/legaldocuman-rfdetr"
WEIGHTS = "checkpoint_best_total.pth"
MIN_SCORE = 0.05


class Detector:
    def __init__(self) -> None:
        from huggingface_hub import hf_hub_download
        from rfdetr import RFDETRBase

        self._model = RFDETRBase(pretrain_weights=hf_hub_download(MODEL_ID, WEIGHTS))  # rfdetr wants a local file

    def boxes(self, rgb: np.ndarray) -> list[Box]:
        from PIL import Image

        height = rgb.shape[0]
        found = self._model.predict(Image.fromarray(rgb), threshold=MIN_SCORE)
        return [Box(float(c), float(b[1]) / height, float(b[3]) / height) for b, c in zip(found.xyxy, found.confidence)]
