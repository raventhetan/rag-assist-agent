# 🔄 Обновлённый план: Подготовка данных для RAG-пайплайна (v2)

## Что изменено и почему

Ваш исходный план **хорош по структуре**, но я предлагаю 7 конкретных улучшений, каждое с обоснованием из актуальных исследований.

---

## Улучшение 1: Contextual Retrieval вместо простого слияния текстов

### Проблема в исходном плане
Простое склеивание абзацев по заголовкам теряет контекст: чанк "Ширина светового проёма: 900 мм" без указания, что это из раздела «Двери одностворчатые типа 1» — бесполезен для поиска.

### Решение
Anthropic разработал contextual retrieval — простой, но эффективный способ улучшить поиск информации. Сохранение правильного контекста для каждого чанка снижает ошибки retrieval на 67%.

Традиционные RAG-системы часто разбивают документы на мелкие чанки для облегчения поиска, но это может удалять важный контекст. Например, чанк может говорить "Its more than 3.85 million inhabitants make it the European Union's most populous city" без упоминания, о каком городе идёт речь. Этот недостаток контекста ведёт к неполным или нерелевантным результатам.

Contextual retrieval исправляет это, генерируя и добавляя краткое контекстно-специфичное объяснение к каждому чанку перед его эмбеддингом.

**Конкретная реализация**: После формирования семантических чанков — добавить шаг **контекстуализации** через LLM:

```python
CONTEXTUALIZE_PROMPT = """Вот документ: {doc_title}
Раздел: {section_hierarchy}  # напр. "Таблица 11 > Двери одностворчатые > Тип 1"

Вот фрагмент из этого документа:
<chunk>{chunk_text}</chunk>

Дай краткое (2-3 предложения) описание контекста этого фрагмента,
чтобы его можно было понять без остального документа.
Ответь ТОЛЬКО контекстным описанием, без пересказа содержания."""
```

Результат дописывается **перед** `search_text`:

```json
{
  "chunk_id": "sem_001",
  "context_prefix": "Этот фрагмент из документа table11.pdf описывает...",
  "search_text": "{context_prefix}\n\n{original_text}",
  "payload": { ... }
}
```

Влияние contextual chunking значительно и измеримо. По данным Anthropic, этот подход может снизить ошибки retrieval по нескольким доменам в среднем на 35%.

**Стоимость**: Для ~200 чанков × ~500 input tokens через Qwen3-8B на OpenRouter = 100K tokens × $0.04/1M = **$0.004** — пренебрежимо мало.

---

## Улучшение 2: Structure-aware chunking вместо чистого семантического

### Проблема
Бенчмарк Vecta (февраль 2026), тестировавший 7 стратегий на 50 академических статьях, поставил recursive 512-token splitting на первое место с 69% точности, тогда как семантическое чанкование оказалось на уровне 54%, выдавая фрагменты в среднем по 43 токена.

Если ваш контент имеет чёткую структуру (Markdown, HTML), переключитесь на structure-aware метод вроде MarkdownHeaderTextSplitter. Часто это самое значительное и простое улучшение, которое можно сделать.

### Решение
Поскольку Docling выдаёт структурированное дерево элементов с заголовками, **используйте document-structure-aware подход**, а не embedding-based semantic splitting:

```
Стратегия: Hierarchical Structure Chunking
┌─────────────────────────────────────┐
│ 1. Границы чанков = заголовки       │  ← Docling даёт section_header/title
│ 2. Merge абзацев внутри раздела     │  ← до ~512 tokens (~1200-1500 символов)
│ 3. Split если раздел > 512 tokens   │  ← recursive split по \n\n → \n → предложениям
│ 4. Contextualize каждый чанк        │  ← LLM добавляет context_prefix
└─────────────────────────────────────┘
```

Начните с recursive chunking (или token-based при жёстких бюджетах), затем добавьте structure-aware splitting. Апгрейды 2026 года, такие как contextual retrieval, late chunking и cross-granularity, часто дают больший выигрыш, чем подстройка overlap.

---

## Улучшение 3: Размерность эмбеддингов — 768, а не 1536

### Обоснование

Gemini Embedding поддерживает более 100 языков и имеет максимальную длину входа 2048 токенов. Модель использует технику Matryoshka Representation Learning (MRL), позволяющую разработчикам уменьшать размерность выхода с дефолтных 3072.

Выбор меньшей размерности экономит место хранения и увеличивает вычислительную эффективность при минимальной потере качества. По умолчанию модель выдаёт 3072-мерный эмбеддинг, но его можно обрезать. Рекомендуемые размерности: 768, 1536 или 3072.

Для **free tier Qdrant Cloud (1 GB RAM)** — размерность критически важна:

| Размерность | Байт/вектор | Векторов в 1 GB (float32) | Векторов в 1 GB (с payload ~500 bytes) |
|---|---|---|---|
| 3072 | 12 288 | ~85 000 | ~65 000 |
| 1536 | 6 144 | ~170 000 | ~120 000 |
| **768** | **3 072** | **~340 000** | **~200 000** |

Qdrant Cloud предлагает 1GB бесплатный кластер навсегда, без кредитной карты.

**→ Рекомендация: используйте dim=768.** Для ваших ~200-500 чанков это более чем достаточно, а запас нужен на случай роста документации.

⚠️ Если бесплатный кластер не используется, он автоматически приостанавливается через 1 неделю и удаляется через 4 недели неактивности, если не реактивирован. Решение — добавить простой cron-пинг (раз в 5 дней делать один search запрос).

---

## Улучшение 4: Два параллельных `search_text` для таблиц (Parent-Child)

### Проблема в исходном плане
Вы предлагаете хранить только саммари таблицы. Но менеджер может спросить конкретное число ("Какой размер проёма для двери типа 2?"), которого **нет в саммари**.

### Решение: Multi-vector indexing для таблиц

Каждая таблица порождает **несколько чанков**:

```
Таблица (Markdown, 50 строк)
  │
  ├── Чанк A (type=table_summary):
  │     search_text = LLM-саммари ("Таблица содержит размеры дверей...")
  │     payload.content = полный Markdown таблицы
  │
  ├── Чанк B (type=table_row_group):
  │     search_text = "Дверь одностворчатая Тип 1: ширина 900, высота 2100..."
  │     payload.content = строки 1-10 таблицы
  │     payload.parent_id = "sem_table_001"
  │
  └── Чанк C (type=table_row_group):
        search_text = "Дверь двустворчатая Тип 3: ширина 1500, высота 2400..."
        payload.content = строки 11-20 таблицы
        payload.parent_id = "sem_table_001"
```

Это значит: поиск по числу найдёт конкретную группу строк, а по общему описанию — саммари. В обоих случаях LLM получит полный контент через payload.

---

## Улучшение 5: Gemini Embedding — правильный task_type

Критически важный нюанс, отсутствующий в исходном плане. Gemini Embedding поддерживает **разные task_type**, которые влияют на качество:

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="YOUR_KEY")

# ДЛЯ ИНДЕКСАЦИИ ДОКУМЕНТОВ:
def embed_for_storage(texts: list[str]):
    results = []
    for text in texts:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",      # ← ДЛЯ ДОКУМЕНТОВ
                output_dimensionality=768
            )
        )
        results.append(result.embeddings[0].values)
    return results

# ДЛЯ ПОИСКОВЫХ ЗАПРОСОВ:
def embed_for_query(query: str):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",              # ← ДЛЯ ЗАПРОСОВ
            output_dimensionality=768
        )
    )
    return result.embeddings[0].values
```

Для gemini-embedding-001 каждый запрос может содержать только один входной текст. Это значит, что нужна **последовательная** обработка с rate-limiting, а не batch:

```python
import time

def embed_chunks_batch(chunks, rate_limit_pause=0.1):
    """Последовательный embed с паузой для rate limiting"""
    vectors = []
    for i, chunk in enumerate(chunks):
        vec = embed_for_storage(chunk["search_text"])
        vectors.append(vec)
        if i % 50 == 0:
            print(f"Embedded {i}/{len(chunks)}")
        time.sleep(rate_limit_pause)  # ~10 req/sec для free tier
    return vectors
```

---

## Улучшение 6: Хранение content в payload — сжатие больших таблиц

### Проблема
Кластеры Qdrant Cloud тарифицируются на основе CPU, памяти и дискового хранилища. На free tier (1 GB) большие Markdown-таблицы в payload могут занять значительную долю.

### Решение
Для таблиц > 3000 символов — хранить **только первые 3000 символов** в payload, а полную таблицу загружать из файла при генерации:

```json
{
  "chunk_id": "sem_table_001",
  "search_text": "Саммари: таблица размеров дверных блоков...",
  "payload": {
    "type": "table_summary",
    "content_preview": "| Тип | Ширина | Высота |\n|---|...(первые 3000 символов)",
    "content_file": "data/full_tables/table11_t1.md",
    "source_file": "table11",
    "pages": [1],
    "bboxes": [[10, 20, 580, 400]]
  }
}
```

При генерации ответа `search_service.py` подгружает полную таблицу из файла:

```python
def get_full_content(payload):
    if payload.get("content_file") and os.path.exists(payload["content_file"]):
        with open(payload["content_file"]) as f:
            return f.read()
    return payload.get("content_preview", payload.get("content", ""))
```

---

## Улучшение 7: Keep-alive для Qdrant free tier + fallback

Бесплатные кластеры автоматически приостанавливаются через 1 неделю неактивности и удаляются через 4 недели, если не реактивированы.

Добавить в план:

```python
# keep_alive.py — запускать через cron раз в 5 дней
from qdrant_client import QdrantClient

client = QdrantClient(url="...", api_key="...")
# Простой запрос, чтобы кластер не заснул
info = client.get_collection("docs")
print(f"Keep-alive: {info.points_count} points active")
```

**Fallback**: хранить `semantic_chunks.json` + `vectors.npy` локально, чтобы можно было мгновенно восстановить Qdrant при удалении.

---

## 📋 Обновлённый полный план (v2)

### Итоговый формат Semantic Chunk JSON (v2)

```json
{
  "chunk_id": "sem_001",
  "chunk_type": "text|table_summary|table_rows|image_context",
  "search_text": "{context_prefix}\n\n{content_or_summary}",
  "context_prefix": "Фрагмент из table11.pdf, раздел 'Двери одностворчатые'...",
  "payload": {
    "type": "table",
    "source_file": "table11",
    "pages": [1, 2],
    "bboxes": [[10, 20, 100, 200]],
    "content": "Полный текст или markdown (до 3000 символов)",
    "content_file": "data/full_tables/table11_t1.md",
    "image_path": null,
    "parent_id": null,
    "section_hierarchy": ["Таблица 11", "Двери одностворчатые", "Тип 1"]
  }
}
```

### Шаг 1: `src/semantic_chunker.py` (обновлённый)

```
ВХОД: *_chunks.json (сырые Docling-чанки)

ЭТАП 1.1 — Structure-aware merge текстовых чанков:
  • Границы = заголовки (section_header, title) из Docling
  • Merge абзацев внутри раздела до ~512 tokens
  • Если раздел > 512 tokens → recursive split по \n\n → \n → ". "
  • Overlap: НЕ добавлять (экономия, исследования показали минимальный эффект)
  • Сохранять section_hierarchy: ["Документ", "Раздел 2", "Подраздел 2.1"]
  • Сохранять массив bboxes и pages для точных ссылок

ЭТАП 1.2 — Таблицы → Multi-vector:
  • Чанк A: LLM-саммари (Qwen3-8B) → search_text = саммари
  • Чанки B-N: группы по 5-10 строк таблицы → search_text = текст строк
  • Все чанки: payload.content = полная таблица (или content_file)
  • payload.parent_id связывает row_groups с summary

ЭТАП 1.3 — Изображения → контекстное обогащение:
  • search_text = caption + абзацы ДО и ПОСЛЕ (до 300 tokens суммарно)
  • payload.image_path = путь к PNG
  • payload.section_hierarchy = путь по заголовкам

ЭТАП 1.4 — Contextual Retrieval (NEW):
  • Для КАЖДОГО чанка: отправить в LLM (Qwen3-8B)
    весь section context + chunk → получить context_prefix (2-3 предложения)
  • Prepend context_prefix к search_text

ВЫХОД: *_semantic.json
```

### Шаг 2: `src/embed_and_upload.py` (обновлённый)

```
ВХОД: *_semantic.json

ЭТАП 2.1 — Gemini Embedding:
  • Модель: gemini-embedding-001
  • Размерность: 768 (MRL truncation)
  • task_type: RETRIEVAL_DOCUMENT для индексации
  • Rate limiting: 1 req / 0.1s = 10 req/s (free tier safe)
  • Сохранять vectors.npy локально (backup)

ЭТАП 2.2 — Qdrant Cloud upload:
  • Collection: "docs", distance=COSINE, size=768
  • Payload: все метаданные кроме search_text (экономия места)
  • content: обрезать до 3000 символов, полная версия → content_file

ЭТАП 2.3 — Keep-alive setup:
  • Скрипт keep_alive.py + инструкция по cron (раз в 5 дней)
  • Логирование в keep_alive.log

ВЫХОД: Заполненная коллекция Qdrant + vectors.npy backup
```

### Шаг 3: `src/search_service.py` (обновлённый)

```
ВХОД: текстовый запрос пользователя

ЭТАП 3.1 — Query embedding:
  • gemini-embedding-001, task_type=RETRIEVAL_QUERY, dim=768

ЭТАП 3.2 — Qdrant search:
  • Top-15 по cosine similarity
  • Фильтр по source_file (если указан в запросе)

ЭТАП 3.3 — LLM rerank (Qwen3-8B, один вызов):
  • Подать top-15 фрагментов → получить ранжирование → взять top-5
  • ~2000 input tokens × $0.04/1M = $0.00008

ЭТАП 3.4 — Подготовка контекста:
  • Для top-5: загрузить полный content из content_file (если таблица)
  • Собрать image_paths для отображения
  • Собрать sources: [{source_file, pages, bboxes, type}]

ЭТАП 3.5 — Генерация ответа (Qwen3-8B):
  • System prompt с правилами цитирования
  • Контекст: 5 чанков с полным content
  • Ожидаемый формат: ответ + [Источник: файл, стр. X, тип]

ВЫХОД: {answer, sources[], image_paths[]}
```

### Шаг 4: `src/app.py` — Streamlit (без изменений к оригинальному плану)

---

## Сводная таблица изменений

| # | Что было | Что стало | Почему |
|---|---|---|---|
| 1 | Простое слияние по заголовкам | + Contextual Retrieval (LLM prefix) | −35-67% ошибок retrieval |
| 2 | "Семантическое чанкование" | Structure-aware + recursive split 512 tok | Бенчмарки 2026: recursive > semantic |
| 3 | dim=1536 | **dim=768** | 2× экономия Qdrant free tier, −минимум качества |
| 4 | Одно саммари на таблицу | Multi-vector: саммари + группы строк | Поиск конкретных чисел в таблицах |
| 5 | Нет task_type | `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` | Критично для качества Gemini Embedding |
| 6 | Полный content в payload | Content preview (3KB) + content_file | Экономия 1GB free tier |
| 7 | Нет keep-alive | Cron ping + local backup | Защита от авто-удаления кластера |

---

## Verification Plan (обновлённый)

### Тест 1 — Структура semantic chunks
```bash
python src/semantic_chunker.py --input output/table11_chunks.json
# Проверить:
# ✓ Чанков стало меньше (склейка)
# ✓ Каждый чанк имеет search_text с context_prefix
# ✓ Таблицы имеют и summary и row_group чанки
# ✓ Изображения имеют search_text из окружающего текста
# ✓ section_hierarchy заполнена
```

### Тест 2 — Embedding + Upload
```bash
python src/embed_and_upload.py --input output/table11_semantic.json --mode memory
# Проверить:
# ✓ Размерность векторов = 768
# ✓ vectors.npy создан локально
# ✓ Qdrant in-memory содержит все точки
# ✓ Payload не содержит search_text (экономия)
```

### Тест 3 — Retrieval quality (golden queries)
```
Запрос: "Какие размеры по ширине светового прохода для одностворчатой двери типа 1?"
  ✓ Top-5 содержит чанк table_rows с нужными строками
  ✓ Источник: table11.pdf, стр. 1, таблица
  ✓ Ответ содержит конкретные числа из таблицы

Запрос: "Покажи схему дверного блока"  
  ✓ Top-5 содержит image_context чанк
  ✓ image_path указывает на существующий PNG
  
Запрос: "Какие требования к монтажу?"
  ✓ Top-5 содержит текстовые чанки с context_prefix
  ✓ section_hierarchy помогает LLM сформировать точную ссылку
```

### Тест 4 — Qdrant free tier capacity
```bash
python src/check_capacity.py
# Подсчитать:
# - Суммарный размер всех payload в байтах
# - Суммарный размер всех векторов (N × 768 × 4 bytes)
# - Итого / 1 GB = % использования free tier
# ✓ Должно быть < 50% для запаса
```
