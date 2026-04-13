# Описание признаков

## Общая схема

В проекте используются два типа признаков:

- sparse текстовые признаки для `LogisticRegression`
- dense handcrafted-признаки для `CatBoost`

## Текстовые признаки

### Нормализация

Перед извлечением признаков текст:

- приводится к lowercase
- очищается от лишних технических символов
- переводит `ё -> е`
- схлопывает повторные пробелы
- исправляет некоторые типовые ASR-артефакты вроде `с м с -> смс`

Сохраняются обе версии:

- `transcript_raw`
- `transcript_norm`

### Trigger features

Основные поля:

- `trigger_total_count`
- `trigger_unique_count`
- `trigger_density_per_100_words`
- `trigger_weighted_score`

Счётчики по категориям:

- `authority_count`
- `money_count`
- `verification_count`
- `urgency_count`
- `credit_count`
- `sim_count`
- `relative_emergency_count`

Бинарные флаги:

- `has_safe_account`
- `has_sms_code`
- `has_bank_security_phrase`
- `has_dont_hang_up`
- `has_credit_phrase`

Кросс-фичи:

- `money_and_urgency`
- `authority_and_verification`
- `credit_and_sms`
- `safe_account_and_transfer`

Также сохраняются debug-поля:

- `matched_trigger_phrases`
- `top_trigger_categories`

### Pattern features

Для regex-паттернов считаются:

- `pattern_total_count`
- `pattern_unique_count`
- `pattern_<name>_count`
- `has_pattern_<name>`
- `pattern_category_<category>_count`

И debug-поля:

- `matched_pattern_names`
- `top_pattern_categories`

### Text meta-features

Считаются простые статистики:

- `text_char_len`
- `text_word_count`
- `unique_word_count`
- `avg_word_len`
- `digit_count`
- `suspicious_imperative_count`
- `repetition_ratio`
- `exclamation_count`
- `question_count`

### TF-IDF

Для baseline используются два независимых векторизатора:

- word TF-IDF: `ngram_range=(1, 2)`
- char TF-IDF: `ngram_range=(3, 5)`

Они обучаются только на train-составе и сохраняются в артефакты модели.

## Аудиопризнаки

Используются только лёгкие статистические признаки:

- `duration_sec`
- `rms_mean`
- `rms_std`
- `zero_crossing_rate_mean`
- `spectral_centroid_mean`
- `spectral_bandwidth_mean`
- `silence_ratio`
- `estimated_speech_ratio`
- `mean_pause_duration`
- `pause_count`
- `estimated_words_per_minute`

Дополнительно сохраняется `audio_feature_error`, если извлечение прошло не полностью.

## Какие признаки куда идут

### В LogisticRegression

Используются:

- word TF-IDF
- char TF-IDF
- dense handcrafted text/audio features

Сильная сторона этой модели — работа со sparse текстом и интерпретируемость коэффициентов.

### В CatBoost

Используются:

- trigger features
- pattern features
- text meta-features
- audio stats

То есть только dense-признаки без прямого sparse TF-IDF.

## Какие признаки sparse, какие dense

Sparse:

- все `word:*`
- все `char:*`

Dense:

- все trigger features
- все pattern features
- все text meta-features
- все audio features

Именно поэтому в проекте используются две разные модели и затем простой ансамбль их вероятностей.
