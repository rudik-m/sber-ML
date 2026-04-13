# Vishing Pipeline

Проект решает задачу распознавания мошеннических телефонных разговоров по аудиозаписям `.wav`.

Классы в задаче заданы так:

- `0` — разговор мошеннический
- `1` — разговор не мошеннический

Решение реализовано на Python и работает локально, без платных API и облачных сервисов.

## Что Делает Проект

На вход подаётся папка с аудиофайлами. Далее пайплайн:

1. переводит речь в текст с помощью ASR
2. нормализует транскрипт
3. извлекает текстовые и аудио-признаки
4. применяет обученную модель
5. сохраняет итоговый `predictions.csv`

Основная идея проекта: главный сигнал в задаче содержится в тексте разговора, поэтому решение строится вокруг ASR, словаря триггеров, regex-паттернов и простых интерпретируемых признаков.

## Соответствие Постановке Задачи

Итоговый сценарий соответствует требованиям:

- алгоритм реализован на Python
- используются только открытые локальные библиотеки и модели
- на вход подаётся папка с `.wav`
- на выходе формируется CSV в формате `filename,label`
- поддерживается запуск по папке с файлами и сохранение итоговых предсказаний в CSV

Формат итогового файла:

```csv
filename,label
out_d_19.wav,0
Nout_b_32.wav,1
```

## Архитектура Решения

Пайплайн устроен так:

1. `wav -> transcript`
2. нормализация текста
3. trigger features и regex-pattern features
4. text meta-features и простые audio stats
5. обучение нескольких моделей
6. ансамбль вероятностей
7. inference по папке с файлами

Используемые модели:

- `LogisticRegression` как baseline
- `CatBoost` как табличная модель на dense-признаках
- `Ensemble` как итоговая модель по умолчанию

ASR-часть построена на `faster-whisper`.

## Структура Репозитория

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

Основные каталоги:

- `configs/` — YAML-конфиги признаков, паттернов и моделей
- `src/vishing/` — основной ML-пайплайн
- `src/vishing_asr/` — этап транскрибации аудио
- `tests/` — unit-тесты
- `vishing/samples/` — обучающая выборка
- `vishing/test/` — папка для инференса
- `vishing/artifacts/` — все промежуточные и итоговые артефакты

## Основные Файлы

- `src/vishing/cli.py` — единый CLI пайплайна
- `src/vishing_asr/transcribe.py` — ASR-этап `wav -> transcript`
- `src/vishing/features/build.py` — построение `features.csv`
- `src/vishing/models/train_logreg.py` — обучение `LogisticRegression`
- `src/vishing/models/train_catboost.py` — обучение `CatBoost`
- `src/vishing/models/ensemble.py` — ансамблирование моделей
- `src/vishing/models/predict.py` — inference и сохранение предсказаний
- `docs/architecture_ru.md` — архитектура решения
- `docs/features_ru.md` — описание признаков
- `docs/usage_ru.md` — сценарии использования
- `docs/experiments_ru.md` — эксперименты и валидация

## Установка

Требования:

- Python `3.11` или `3.12`
- `uv`

Установка зависимостей:

```bash
uv sync
```

Для текущего набора `.wav` отдельная установка `ffmpeg` не требуется. При первом запуске `faster-whisper` скачивает открытые веса модели в локальный кэш.

## Как Запустить Решение

### 1. Проверка структуры проекта

```bash
uv run vishing inspect --input-dir ./vishing
```

Команда показывает:

- сколько найдено аудиофайлов
- как они распределены по `samples/test`
- сколько строк уже есть в `transcripts.csv`
- есть ли `features.csv`
- какие модели уже обучены

### 2. Транскрибация аудио

Полный прогон:

```bash
uv run vishing transcribe --input-dir ./vishing
```

Быстрая проверка:

```bash
uv run vishing-asr transcribe --input-dir ./vishing --limit 3
```

### 3. Построение признаков

```bash
uv run vishing build-features --input-dir ./vishing
```

Если транскриптов не хватает, команда по умолчанию дозапускает ASR для недостающих файлов без перезаписи уже готовых артефактов.

### 4. Обучение моделей

```bash
uv run vishing train-logreg --features ./vishing/artifacts/features/features.csv
uv run vishing train-catboost --features ./vishing/artifacts/features/features.csv
uv run vishing train-all --features ./vishing/artifacts/features/features.csv
```

### 5. Предсказание по папке

```bash
uv run vishing predict --input-dir ./vishing/test --model ensemble
```

Также можно выбрать конкретную модель:

```bash
uv run vishing predict --input-dir ./vishing/test --model logreg
uv run vishing predict --input-dir ./vishing/test --model catboost
```

### 6. Построение отчётов

```bash
uv run vishing report --input-dir ./vishing
```

## Полный Сценарий Запуска

```bash
uv sync
uv run vishing inspect --input-dir ./vishing
uv run vishing transcribe --input-dir ./vishing
uv run vishing build-features --input-dir ./vishing
uv run vishing train-all --features ./vishing/artifacts/features/features.csv
uv run vishing predict --input-dir ./vishing/test --model ensemble
uv run vishing report --input-dir ./vishing
```

## Что Получается На Выходе

Главный итоговый файл:

- `vishing/artifacts/predictions/predictions.csv`

Формат:

```csv
filename,label
out_d_19.wav,0
Nout_b_32.wav,1
```

Дополнительный debug-файл:

- `vishing/artifacts/predictions/predictions_debug.csv`

Он содержит:

- нормализованный текст
- `fraud_score`
- выбранный threshold
- сработавшие триггеры

## Какие Артефакты Сохраняются

Транскрипты:

- `vishing/artifacts/transcripts/transcripts.csv`
- `.txt` и `.json` для отдельных файлов

Признаки:

- `vishing/artifacts/features/features.csv`

Модели:

- бинарный артефакт модели
- `config.yaml`
- `threshold.json`
- `metrics.json`
- `validation_predictions.csv`

Отчёты:

- `vishing/artifacts/reports/metrics_summary.md`
- `vishing/artifacts/reports/error_analysis.md`
- `vishing/artifacts/reports/feature_overview.md`

## Результаты

Финальный пайплайн использует:

- `faster-whisper` для ASR
- handcrafted text/audio features
- `LogisticRegression`
- `CatBoost`
- ансамбль `0.7 * logreg + 0.3 * catboost`

Актуальные метрики на group-based validation:

| Model | Accuracy | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8667 | 0.8667 | 1.0000 | 0.9286 | 0.7692 |
| CatBoost | 0.8667 | 0.8667 | 1.0000 | 0.9286 | 0.8077 |
| Ensemble | 0.8667 | 0.8667 | 1.0000 | 0.9286 | 0.7692 |

Итоговая модель для inference по умолчанию — `ensemble`.

Важно: датасет в задаче маленький, поэтому эти числа нужно интерпретировать как результат на текущем учебном наборе, а не как окончательную оценку на большой скрытой выборке.

## Проверка Времени

Требование задания: обработка одной записи не должна превышать `3 минуты`.

По реальным замерам на текущем CPU-прогоне:

- транскрибация `100` файлов заняла около `5 минут 29 секунд`
- это даёт в среднем около `3.3 секунды` на запись
- построение признаков `100` файлов заняло около `40 секунд`
- это около `0.4 секунды` на запись
- inference по `60` тестовым файлам занял около `14 секунд`
- это около `0.25 секунды` на запись после построения признаков
- полный путь для новой записи с ASR укладывается примерно в `5-7 секунд` на запись

Следовательно, ограничение по времени выполняется с большим запасом.

## Ограничения Текущей Версии

- основной сигнал идёт из ASR-текста, поэтому качество распознавания влияет на итоговую классификацию
- используется простой energy-based VAD, без диаризации и без тяжёлых аудио-моделей
- `CatBoost` работает только на dense handcrafted-признаках
- датасет маленький, поэтому оценка качества нестабильна

## Подробная Документация

- [Архитектура](docs/architecture_ru.md)
- [Признаки](docs/features_ru.md)
- [Использование](docs/usage_ru.md)
- [Эксперименты и валидация](docs/experiments_ru.md)
