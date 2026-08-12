#!/usr/bin/env python3
"""Runtime insect-vs-debris classifier for BugPicker detector crops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


CLASS_NAMES = ["debris", "insect"]
IMAGE_SIZE = 224
DROPOUT = 0.20


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_model(
    architecture: str,
    number_of_classes: int,
) -> nn.Module:
    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        input_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(input_features, number_of_classes),
        )
        return model

    raise ValueError(
        f"Unsupported classifier architecture {architecture!r}. "
        "This production patch currently supports efficientnet_b0."
    )


def build_inference_transform():
    return transforms.Compose(
        [
            transforms.Resize(int(IMAGE_SIZE * 1.14)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class InsectDebrisClassifier:
    """Load one checkpoint and classify OpenCV BGR crops."""

    def __init__(
        self,
        model_path: Path,
        architecture: str = "efficientnet_b0",
        threshold: float = 0.50,
        device: str = "auto",
    ) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Insect/debris classifier checkpoint not found: {self.model_path}"
            )

        self.device = select_device(device)
        self.threshold = float(threshold)
        self.transform = build_inference_transform()

        checkpoint: Any = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            checkpoint_class_names = checkpoint.get("class_names")
        elif isinstance(checkpoint, dict):
            # Support a raw state_dict checkpoint.
            state_dict = checkpoint
            checkpoint_class_names = None
        else:
            raise TypeError(
                "Classifier checkpoint must be a state_dict or a dictionary "
                "containing model_state_dict."
            )

        self.class_names = list(
            checkpoint_class_names
            if checkpoint_class_names
            else CLASS_NAMES
        )

        if "debris" not in self.class_names:
            raise ValueError(
                f"Checkpoint class_names must contain 'debris'; found {self.class_names}"
            )

        insect_label = None
        for candidate in ("insect", "specimen"):
            if candidate in self.class_names:
                insect_label = candidate
                break

        if insect_label is None:
            raise ValueError(
                "Checkpoint class_names must contain 'insect' or 'specimen'; "
                f"found {self.class_names}"
            )

        self.debris_index = self.class_names.index("debris")
        self.insect_index = self.class_names.index(insect_label)

        self.model = build_model(
            architecture=architecture,
            number_of_classes=len(self.class_names),
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        print(
            "Loaded insect/debris classifier "
            f"model={self.model_path} "
            f"architecture={architecture} "
            f"device={self.device} "
            f"threshold={self.threshold:.2f} "
            f"classes={self.class_names}",
            flush=True,
        )

    def classify_bgr(self, image_bgr) -> dict[str, Any]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Classifier received an empty image crop")

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )
        image = Image.fromarray(image_rgb)

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()

        insect_probability = float(probabilities[self.insect_index])
        debris_probability = float(probabilities[self.debris_index])
        would_pick = insect_probability >= self.threshold

        return {
            "class": "insect" if would_pick else "debris",
            "insect_probability": insect_probability,
            "debris_probability": debris_probability,
            "threshold": self.threshold,
            "would_pick": would_pick,
        }
