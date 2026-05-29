# One Piece Character Classifier

FastAPI-сервис для распознавания персонажей аниме **One Piece** по фотографии.
Модель: **ResNet50** (transfer learning, обучена в `notebooks/one_piece_classification.ipynb`).

---

## Структура проекта

```
one_piece_service/
├── src/
│   ├── data/
│   │   └── transforms.py      # предобработка изображений (val_tf из ноутбука)
│   ├── models/
│   │   └── classifier.py      # загрузка ResNet50 и инференс
│   └── service/
│       ├── main.py            # FastAPI-приложение, все эндпоинты
│       └── schemas.py         # Pydantic-схемы ответов
├── notebooks/                 # ноутбук с EDA и обучением (опционально)
├── .gitignore
|-- .env.example
├── Dockerfile
├── requirements.txt
├── report.md                  # отчёт с метриками и обоснованием модели
└── README.md                  # вы здесь
```

---

## Быстрый старт
ПЕРЕЙДИТЕ В ПАПКУ PROJECT

### 1. Создайте виртуальное окружение и установите зависимости

```bash
python -m venv venv

python -m pip install -r requirements.txt
```

### 2. Настройте переменные окружения

```bash
cp .env.example .env

```

### 3. Запустите сервис

```bash
python -m uvicorn src.service.main:app --reload --host 0.0.0.0 --port 8000
```

Откройте в браузере: **http://localhost:8000/docs**

---



## Эндпоинты

| Метод | Путь       | Описание                                      |
|-------|------------|-----------------------------------------------|
| GET   | `/health`  | Статус сервиса — модель загружена и готова?   |
| POST  | `/predict` | Загрузите изображение → получите персонажа    |
| GET   | `/classes` | Список всех 18 поддерживаемых персонажей      |
| GET   | `/docs`    | Swagger UI (интерактивная документация)       |
| GET   | `/redoc`   | ReDoc документация                            |

---

## Пример запроса через curl

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/путь/к/картинке.jpg"
```

Пример ответа:
```json
{
  "character": "Luffy",
  "confidence": 0.9821,
  "top5": {
    "Luffy":   0.9821,
    "Shanks":  0.0071,
    "Ace":     0.0044,
    "Zoro":    0.0031,
    "Chopper": 0.0012
  }
}
```

---

## Поддерживаемые персонажи (18 классов)

Ace · Akainu · Brook · Chopper · Crocodile · Franky · Jinbei · Kurohige ·
Law · **Luffy** · Mihawk · Nami · Rayleigh · Robin · Sanji · Shanks · Usopp · Zoro

---

## Демонстрация на защите

1. Запустите сервис (шаги 1–4 выше).
2. Откройте **http://localhost:8000/docs**.
3. `POST /predict` → **Try it out** → загрузите картинку персонажа → **Execute**.
4. Получите ответ: имя, уверенность, топ-5.
5. Удивляйтесь работе сервиса, он работает даже слишком хорошо, я такого не ожидал, но самого Луффи плохо определяет почему то
6. Откройте `GET /health` — убедитесь, что статус `"ok"`.
7. Для подробностей об обучении и метриках — смотрите `report.md` и ноутбук.
