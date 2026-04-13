# Vishing Pipeline

Офлайн-проект для задачи распознавания мошеннических разговоров.

Идея решения простая: сначала разговор переводится из аудио в текст, затем из текста и аудио извлекаются признаки, после чего модель определяет, является ли разговор мошенническим.

Пайплайн устроен так:

1. `wav -> transcript`
2. нормализация текста
3. словарь триггеров и regex-паттерны
4. text/audio handcrafted features
5. baseline `LogisticRegression`
6. основная модель `CatBoost`
7. ансамбль вероятностей
8. inference по папке с итоговым `predictions.csv`

Проект не использует платные API, облачные сервисы или тяжёлые end-to-end DL-классификаторы.

## Суть задания

Проект решает задачу бинарной классификации телефонных разговоров:

- `0` — разговор мошеннический
- `1` — разговор не мошеннический

На вход подаётся папка с `.wav`-файлами, на выходе формируется `predictions.csv` в формате:

```csv
filename,label
out_d_19.wav,0
Nout_b_32.wav,1
```

Итоговый сценарий соответствует постановке задачи:

- алгоритм реализован на Python
- используются только локальные открытые модели и библиотеки
- обработка идёт по папке с файлами
- результат возвращается в CSV-формате
- обучение выполняется только на `samples`, а `test` используется только для инференса

## Быстрый старт

Если нужно просто запустить решение целиком, достаточно такого сценария:

```bash
uv sync
uv run vishing transcribe --input-dir ./vishing
uv run vishing build-features --input-dir ./vishing
uv run vishing train-all --features ./vishing/artifacts/features/features.csv
uv run vishing predict --input-dir ./vishing/test --model ensemble
```

Главный результат будет сохранён в:

- `vishing/artifacts/predictions/predictions.csv`

## Результаты

Финальный пайплайн использует `faster-whisper` для ASR, handcrafted text/audio features и три модели-кандидата:

- `LogisticRegression`
- `CatBoost`
- ансамбль вероятностей `0.7 * logreg + 0.3 * catboost`

Актуальные метрики на честной group-based validation по `samples`:

| Model | Accuracy | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CatBoost | 0.5000 | 0.2500 | 1.0000 | 0.4000 | 1.0000 |
| Ensemble | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Итоговый выбор для inference по умолчанию — `ensemble`.

Важно: валидация считается только на `samples`, а сам validation-fold маленький, поэтому эти метрики нужно интерпретировать как результат на текущем учебном наборе, а не как окончательную оценку на большой скрытой выборке.

## Проверка времени

Требование задания: обработка одной записи не должна превышать `3 минуты`.

По реальным замерам на текущем CPU-прогоне проекта:

- транскрибация `100` файлов заняла около `5 минут 29 секунд`, то есть в среднем около `3.3 секунды` на запись
- построение признаков `100` файлов заняло около `40 секунд`, то есть около `0.4 секунды` на запись
- inference по `60` тестовым файлам занял около `14 секунд`, то есть около `0.25 секунды` на запись после построения признаков
- полный путь для новой записи с ASR укладывается примерно в `5-7 секунд` на запись

Следовательно, ограничение по времени выполняется с большим запасом.

## Основные файлы

- `src/vishing/cli.py` — единый CLI пайплайна
- `src/vishing_asr/transcribe.py` — этап `wav -> transcript`
- `src/vishing/features/build.py` — сборка `features.csv`
- `src/vishing/models/train_logreg.py` — обучение baseline `LogisticRegression`
- `src/vishing/models/train_catboost.py` — обучение `CatBoost`
- `src/vishing/models/ensemble.py` — ансамбль вероятностей
- `src/vishing/models/predict.py` — inference и сохранение `predictions.csv`
- `docs/architecture_ru.md` — краткое описание архитектуры решения

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

Важно: обучение использует только строки со `split=samples`, даже если `features.csv` был построен по всему каталогу `./vishing`. Папка `test` предназначена для инференса и не участвует в fit моделей.

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

## Отчёт

Подробное описание решения приведено в файле `report.pdf`.
