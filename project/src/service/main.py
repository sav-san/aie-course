
import io
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.data.transforms import inference_transform
from src.models.classifier import CLASS_NAMES, NUM_CLASSES, ResNetClassifier
from src.service.schemas import HealthResponse, PredictResponse

# Загружаем .env до обращения к os.getenv
load_dotenv()

# Логирование: пишем в консоль с временной меткой
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Хранилище состояния (модель живёт всё время работы сервера)
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    weights = os.getenv("MODEL_WEIGHTS", "models_weights/resnet50_onepiece.pth")
    device  = os.getenv("DEVICE",        "cpu")

    logger.info("=" * 55)
    logger.info("  One Piece Classifier — старт сервиса")
    logger.info("  Веса  : %s", weights)
    logger.info("  Device: %s", device)
    logger.info("=" * 55)

    try:
        _state["classifier"] = ResNetClassifier(weights_path=weights, device=device)
        _state["weights"]    = weights
        _state["device"]     = device
        logger.info("✅ Модель успешно загружена и готова к работе")
    except FileNotFoundError as exc:
        # Сервер стартует, но /predict будет возвращать 503 с понятным сообщением
        logger.error("❌ %s", exc)
        _state["classifier"] = None
        _state["error"]      = str(exc)
        _state["device"]     = device

    yield   # ← сервер работает, обрабатывает запросы

    logger.info("Остановка сервиса, очищаем состояние...")
    _state.clear()


# Создаём приложение 
app = FastAPI(
    title="One Piece Character Classifier",
    description=(
        "Определяет персонажа аниме **One Piece** по фотографии.\n\n"
        "**Модель:** ResNet50 (обучена на датасете One Piece Image Classifier, Kaggle).\n\n"
        "**Поддерживаемые персонажи:** Ace, Akainu, Brook, Chopper, Crocodile, "
        "Franky, Jinbei, Kurohige, Law, Luffy, Mihawk, Nami, Rayleigh, Robin, "
        "Sanji, Shanks, Usopp, Zoro.\n\n"
        "Загрузите изображение через **POST /predict** и получите имя персонажа."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# GET /health 
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния сервиса",
    tags=["System"],
)
def health():
    if _state.get("classifier") is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Модель не загружена. "
                f"Причина: {_state.get('error', 'неизвестно')}. "
                "Проверьте путь MODEL_WEIGHTS в файле .env."
            ),
        )
    return HealthResponse(
        status="ok",
        model=f"ResNet50 | {_state['weights']}",
        device=_state["device"],
        num_classes=NUM_CLASSES,
    )


# GET /classes
@app.get(
    "/classes",
    summary="Список поддерживаемых персонажей",
    tags=["Info"],
)
def get_classes():
    """Возвращает все 18 персонажей, которых умеет распознавать модель."""
    return {"classes": CLASS_NAMES, "total": NUM_CLASSES}


# POST /predict 
@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Определить персонажа по изображению",
    tags=["Prediction"],
)
async def predict(
    file: UploadFile = File(
        ...,
        description="Изображение персонажа (JPEG, PNG, WEBP — любой формат, который открывает Pillow)",
    ),
):
    # Проверяем, что модель загружена
    classifier: ResNetClassifier | None = _state.get("classifier")
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не готова. Проверьте GET /health для деталей.",
        )

    # Читаем байты из загруженного файла
    raw_bytes = await file.read()
    logger.info("Запрос /predict | файл='%s' | размер=%d bytes", file.filename, len(raw_bytes))

    # Открываем как PIL-изображение
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError:
        logger.warning("Не удалось открыть файл как изображение: %s", file.filename)
        raise HTTPException(
            status_code=422,
            detail="Файл не является изображением или повреждён. Поддерживаются JPEG, PNG, WEBP и другие форматы Pillow.",
        )

    # Предобработка - инференс
    tensor               = inference_transform(image)          # (3, 224, 224)
    character, conf, top5 = classifier.predict(tensor)

    logger.info(
        "Результат | character='%s' | confidence=%.4f | file='%s'",
        character, conf, file.filename,
    )

    return PredictResponse(character=character, confidence=conf, top5=top5)
