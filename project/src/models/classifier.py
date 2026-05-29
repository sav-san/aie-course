
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

logger = logging.getLogger(__name__)

# ── 18 классов в том же порядке, что задаёт ImageFolder (алфавитный) ─────────
CLASS_NAMES = [
    "Ace", "Akainu", "Brook", "Chopper", "Crocodile",
    "Franky", "Jinbei", "Kurohige", "Law", "Luffy",
    "Mihawk", "Nami", "Rayleigh", "Robin", "Sanji",
    "Shanks", "Usopp", "Zoro",
]
NUM_CLASSES = len(CLASS_NAMES)          # 18


def _build_resnet50() -> nn.Module:
    m = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
    )
    return m


class ResNetClassifier:

    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Файл весов не найден: {path}\n"
            )

        logger.info("Строим архитектуру ResNet50...")
        self.model = _build_resnet50()

        logger.info("Загружаем веса: %s", path)
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()          # отключаем dropout/batchnorm в режиме train

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Модель готова | params=%s | device=%s", f"{n_params:,}", device)

    @torch.no_grad()
    def predict(self, tensor: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        """
        Делает предсказание для одного изображения.

        Параметры
        ---------
        tensor : нормализованный тензор (3, 224, 224) из inference_transform

        Возвращает
        ----------
        character  : имя персонажа с наибольшей вероятностью
        confidence : вероятность этого класса (0..1)
        top5       : словарь {имя: вероятность} для 5 лучших кандидатов
        """
        # Добавляем batch-измерение: (3,224,224) → (1,3,224,224)
        tensor = tensor.unsqueeze(0).to(self.device)

        logits = self.model(tensor)                          # (1, 18)
        probs  = torch.softmax(logits, dim=1).squeeze()     # (18,)

        # Топ-5
        top5_vals, top5_idx = torch.topk(probs, k=5)
        top5 = {
            CLASS_NAMES[i.item()]: round(v.item(), 4)
            for v, i in zip(top5_vals, top5_idx)
        }

        best_idx   = probs.argmax().item()
        character  = CLASS_NAMES[best_idx]
        confidence = round(probs[best_idx].item(), 4)

        return character, confidence, top5
