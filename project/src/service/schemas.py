
from typing import Dict
from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    """Ответ эндпоинта POST /predict."""

    character: str = Field(
        ...,
        description="Имя распознанного персонажа One Piece",
        examples=["Luffy"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Уверенность модели: 0 = нет уверенности, 1 = абсолютная",
        examples=[0.9821],
    )
    top5: Dict[str, float] = Field(
        ...,
        description="Топ-5 персонажей-кандидатов и их вероятности",
        examples=[{"Luffy": 0.9821, "Shanks": 0.0071, "Ace": 0.0044, "Zoro": 0.0031, "Chopper": 0.0012}],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "character": "Luffy",
                "confidence": 0.9821,
                "top5": {
                    "Luffy":   0.9821,
                    "Shanks":  0.0071,
                    "Ace":     0.0044,
                    "Zoro":    0.0031,
                    "Chopper": 0.0012,
                },
            }
        }
    }


class HealthResponse(BaseModel):
    """Ответ эндпоинта GET /health."""

    status: str = Field(..., description="'ok' — модель загружена и готова")
    model: str  = Field(..., description="Архитектура и файл весов")
    device: str = Field(..., description="Вычислительное устройство: cpu / cuda")
    num_classes: int = Field(..., description="Количество классов (персонажей)")
