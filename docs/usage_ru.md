# Использование проекта

## 1. Проверка входных данных

```bash
uv run vishing inspect --input-dir ./vishing
```

Команда показывает:

- сколько найдено `wav`
- распределение по `samples/test` и `Fraud/NotFraud`
- сколько строк уже есть в `transcripts.csv`
- есть ли `features.csv`
- какие модели уже обучены

## 2. Транскрибация

Полный прогон:

```bash
uv run vishing transcribe --input-dir ./vishing
```

Быстрая проверка:

```bash
uv run vishing-asr transcribe --input-dir ./vishing --limit 3
```

## 3. Построение признаков

Полный датасет:

```bash
uv run vishing build-features --input-dir ./vishing
```

Сокращённый прогон:

```bash
uv run vishing build-features --input-dir ./vishing --limit 5
```

Если транскриптов не хватает, команда по умолчанию пытается дозапустить ASR-этап без перезаписи готовых артефактов. Это можно отключить:

```bash
uv run vishing build-features --input-dir ./vishing --no-run-transcribe-missing
```

## 4. Обучение LogisticRegression

```bash
uv run vishing train-logreg --features ./vishing/artifacts/features/features.csv
```

Команда обучает модель на размеченных строках из `features.csv`.

С ограничением по числу строк:

```bash
uv run vishing train-logreg --features ./vishing/artifacts/features/features.csv --limit 20
```

## 5. Обучение CatBoost

```bash
uv run vishing train-catboost --features ./vishing/artifacts/features/features.csv
```

## 6. Обучение всех моделей и ансамбля

```bash
uv run vishing train-all --features ./vishing/artifacts/features/features.csv
```

Сокращённая проверка:

```bash
uv run vishing train-all --features ./vishing/artifacts/features/features.csv --limit 20
```

## 7. Инференс

```bash
uv run vishing predict --input-dir ./vishing/test --model ensemble
```

Также можно выбрать конкретную модель:

```bash
uv run vishing predict --input-dir ./vishing/test --model logreg
uv run vishing predict --input-dir ./vishing/test --model catboost
```

Основной результат:

- `vishing/artifacts/predictions/predictions.csv`

Расширенный debug:

- `vishing/artifacts/predictions/predictions_debug.csv`

## 8. Генерация отчётов

```bash
uv run vishing report --input-dir ./vishing
```

Команда обновляет:

- `metrics_summary.json`
- `metrics_summary.md`
- `error_analysis.md`

## 9. Рекомендуемый короткий сценарий проверки

```bash
uv sync
uv run vishing inspect --input-dir ./vishing
uv run vishing build-features --input-dir ./vishing --limit 5
uv run vishing train-all --features ./vishing/artifacts/features/features.csv --limit 20
```

## 10. Полный сценарий

```bash
uv sync
uv run vishing transcribe --input-dir ./vishing
uv run vishing build-features --input-dir ./vishing
uv run vishing train-all --features ./vishing/artifacts/features/features.csv
uv run vishing predict --input-dir ./vishing/test --model ensemble
uv run vishing report --input-dir ./vishing
```
