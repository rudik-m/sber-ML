# Валидация и дальнейшие эксперименты

## Как валидировать текущий baseline

В проекте используется group-based holdout:

- из имени файла извлекается `group_id`
- train/validation делятся по группам, а не по отдельным файлам

Это нужно, чтобы похожие записи одной серии не попали одновременно в train и validation.

## Почему нельзя делать случайный file-level split

Для файлов вроде:

- `out_a_13.wav`
- `out_a_14.wav`
- `out_a_15.wav`

случайное разбиение почти гарантирует утечку. Модель будет видеть слишком похожие примеры и завысит качество. Поэтому честность split важнее агрессивной настройки гиперпараметров.

## Что смотреть после обучения

После `train-all` полезно открыть:

- `vishing/artifacts/reports/metrics_summary.md`
- `vishing/artifacts/reports/logreg_report.md`
- `vishing/artifacts/reports/catboost_report.md`
- `vishing/artifacts/reports/ensemble_report.md`
- `vishing/artifacts/reports/error_analysis.md`

Особенно важно проверять:

- какие триггеры чаще встречаются в false positive
- не переобучилась ли `LogisticRegression` на серийные фразы
- не слишком ли сильно `CatBoost` опирается на один-два audio stats

## Как смотреть ошибки

В проекте для каждой модели сохраняется `validation_predictions.csv`. По нему можно анализировать:

- уверенные false positive
- уверенные false negative
- какие trigger/pattern признаки сработали
- где ASR испортил ключевой смысл разговора

## Разумные следующие эксперименты

Без лишней сложности логично пробовать:

1. Сравнить `faster-whisper base` и `small` по качеству текста.
2. Расширить `triggers.yaml` и `patterns.yaml` после ручного просмотра ошибок.
3. Подобрать веса ансамбля `LogisticRegression + CatBoost`.
4. Добавить reduced text features для `CatBoost` через `TruncatedSVD`, если это даст прирост без утечек.
5. Ввести более аккуратный подбор threshold по бизнес-целевой метрике.

## Чего пока сознательно нет

В текущей версии не добавлены:

- wav2vec/Hubert-классификаторы
- CNN по спектрограммам
- diarization
- emotion recognition
- сложный VAD

Это сделано специально: проект должен оставаться локальным, понятным и воспроизводимым на маленьком датасете.
