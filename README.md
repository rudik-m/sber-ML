# Vishing Pipeline

Офлайн-проект для задачи распознавания мошеннических разговоров. Пайплайн ориентирован на маленький датасет и построен вокруг текста, полученного из ASR:

1. `wav -> transcript`
2. нормализация текста
3. словарь триггеров и regex-паттерны
4. text/audio handcrafted features
5. baseline `LogisticRegression`
6. основная модель `CatBoost`
7. ансамбль вероятностей
8. inference по папке с итоговым `predictions.csv`

Проект не использует платные API, облачные сервисы или тяжёлые end-to-end DL-классификаторы.

## Структура каталогов

```text
repo/
├── configs/
├── docs/
├── src/
│   ├── vishing/
│   └── vishing_asr/
├── tests/
└── vishing/
    ├── samples/
    ├── test/
    └── artifacts/
```

Основные артефакты:

- `vishing/artifacts/transcripts/` — тексты и сегменты ASR
- `vishing/artifacts/features/` — `features.csv`
- `vishing/artifacts/models/` — модели, threshold, config, validation predictions
- `vishing/artifacts/predictions/` — `predictions.csv` и `predictions_debug.csv`
- `vishing/artifacts/reports/` — markdown/json-отчёты

## Требования

- `uv`
- Python `3.11` или `3.12`
- локальный запуск без облачных сервисов

Для текущего набора `.wav` отдельная установка `ffmpeg` не требуется. На первом запуске `faster-whisper` скачивает открытые веса модели Whisper в локальный кэш.

## Установка

```bash
uv sync
```

## CLI-команды

Проверить входные данные и артефакты:

```bash
uv run vishing inspect --input-dir ./vishing
```

Существующий этап транскрибации:

```bash
uv run vishing transcribe --input-dir ./vishing
```

Быстрый smoke-check ASR:

```bash
uv run vishing-asr transcribe --input-dir ./vishing --limit 3
```

Построить признаки:

```bash
uv run vishing build-features --input-dir ./vishing
```

Сокращённая проверка:

```bash
uv run vishing build-features --input-dir ./vishing --limit 5
```

Обучить baseline:

```bash
uv run vishing train-logreg --features ./vishing/artifacts/features/features.csv
```

Обучить CatBoost:

```bash
uv run vishing train-catboost --features ./vishing/artifacts/features/features.csv
```

Обучить всё сразу:

```bash
uv run vishing train-all --features ./vishing/artifacts/features/features.csv
```

Сокращённый train-check:

```bash
uv run vishing train-all --features ./vishing/artifacts/features/features.csv --limit 20
```

Сделать предсказание:

```bash
uv run vishing predict --input-dir ./vishing/test --model ensemble
```

Собрать отчёты:

```bash
uv run vishing report --input-dir ./vishing
```

## Рекомендуемый порядок запуска

1. `uv sync`
2. `uv run vishing inspect --input-dir ./vishing`
3. `uv run vishing transcribe --input-dir ./vishing`
4. `uv run vishing build-features --input-dir ./vishing`
5. `uv run vishing train-all --features ./vishing/artifacts/features/features.csv`
6. `uv run vishing predict --input-dir ./vishing/test --model ensemble`
7. `uv run vishing report --input-dir ./vishing`

Если транскриптов не хватает, `build-features` и `predict` по умолчанию пытаются дозапустить ASR-этап без перезаписи уже готовых `.txt/.json`.

## Что лежит в `features.csv`

В датасете признаков есть:

- идентификаторы файла: `filepath`, `filename`, `split`, `label_dir`, `label`, `group_id`
- `transcript_raw` и `transcript_norm`
- trigger features
- pattern features
- текстовые meta-features
- лёгкие audio stats

Маппинг классов фиксирован:

- `Fraud -> 0`
- `NotFraud -> 1`

Этот mapping явно соблюдается и в обучении, и в инференсе.

## Что сохраняется для моделей

Для каждой модели сохраняются:

- бинарный артефакт модели
- `config.yaml`
- `threshold.json`
- `metrics.json`
- `validation_predictions.csv`
- markdown-отчёт в `vishing/artifacts/reports/`

Для `LogisticRegression` также сохраняются TF-IDF vectorizers и scaler.

## Что получает пользователь на выходе

Главный файл для задачи:

- `vishing/artifacts/predictions/predictions.csv`

Формат:

```csv
filename,label
out_d_19.wav,0
Nout_b_32.wav,1
```

Дополнительный debug-файл:

- `vishing/artifacts/predictions/predictions_debug.csv`

Он содержит нормализованный текст, `fraud_score`, выбранный threshold и сработавшие триггеры.

## Ограничения текущей версии

- основной сигнал идёт из ASR-текста; качество распознавания влияет на классификацию
- используется простой energy-based VAD, без диаризации и без нейросетевых audio-моделей
- в `CatBoost` сейчас подаются только dense handcrafted features, без SVD поверх TF-IDF
- датасет маленький, поэтому честная group-based validation важнее агрессивного усложнения модели

## Подробная документация

- [Архитектура](docs/architecture_ru.md)
- [Признаки](docs/features_ru.md)
- [Использование](docs/usage_ru.md)
- [Эксперименты и валидация](docs/experiments_ru.md)
