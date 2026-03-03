# ☁️ Облачный RAG-пайплайн: минимум затрат, максимум эффекта

## Ключевое упрощение

Раз изображения — это схемы, которые просто нужно **показать** (а не распознать), и текст уже распарсен без потерь через Docling — нам не нужна VLM-модель. Мы просто привязываем `image_path` к окружающему тексту через метаданные.

---

## 1. Финальная схема (облако, дёшево)

```
┌──────────────────────────────────────────────────────┐
│                 INGESTION (одноразово)                │
│                                                       │
│  chunks.json ──► Gemini Embedding (бесплатно!) ──►   │
│          ──► Qdrant Cloud (free tier) ──► готово      │
│                                                       │
│  Таблицы: LLM генерирует 1 саммари → embed саммари   │
│  Изображения: окружающий текст → embed текст          │
│           image_path → payload                        │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│              QUERY (каждый запрос менеджера)          │
│                                                       │
│  Вопрос → Gemini Embedding → Qdrant search (top-20)  │
│    → Qwen3-8B rerank/answer (OpenRouter, $0.04/1M)   │
│    → Ответ + [Источник: файл, стр., тип] + PNG       │
└──────────────────────────────────────────────────────┘
```

---

## 2. Embedding: **Gemini Embedding** (БЕСПЛАТНО)

Google's Text Embedding 004 model is completely free of charge for both input and output, making it extremely cost-effective for RAG applications, semantic search, and similarity matching.

Новейшая модель — **`gemini-embedding-001`**:

Gemini Embedding supports over 100 languages and has a 2048 maximum input token length. It also utilizes the Matryoshka Representation Learning (MRL) technique, which allows developers to scale the output dimensions down from the default 3072. This flexibility enables you to optimize for performance and storage costs to fit your specific needs. For the highest quality results, 3072, 1536, or 768 output dimensions are recommended.

Google offers both free and paid tiers in the Gemini API, so you can experiment with gemini-embedding-001 at no cost, or ramp up with significantly higher limits for production needs.

### Код для эмбеддинга

```python
from google import genai

client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

def embed_chunks(texts: list[str], dim: int = 1536):
    """Batch embed через Gemini API (бесплатно!)"""
    results = []
    for text in texts:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config={"output_dimensionality": dim}
        )
        results.append(result.embeddings[0].values)
    return results
```

### Альтернатива на OpenRouter (если нужен sparse search)

Qwen3 Embedding на OpenRouter стоит $0.01/M input tokens и $0/M output tokens. Qwen3 Embedding model series is the latest proprietary model of the Qwen family, specifically designed for text embedding and ranking tasks. This series inherits the exceptional multilingual capabilities, long-text understanding, and reasoning skills of its foundational model. It represents significant advancements in multiple text embedding and ranking tasks, including text retrieval, code retrieval, text classification, text clustering, and bitext mining.

### Сравнительная таблица Embedding-моделей

| Модель | Цена / 1M tokens | Размерность | Языки | Контекст | Где доступна |
|---|---|---|---|---|---|
| **gemini-embedding-001** | **$0 (free tier)** | 3072 → 768 (MRL) | 100+ | 2048 tok | Gemini API |
| **Qwen3-Embedding-0.6B** | $0.01/M | 1024 | 100+ | 32K tok | OpenRouter |
| **Qwen3-Embedding-8B** | ~$0.02/M | 4096 | 100+ | 32K tok | OpenRouter |
| text-embedding-3-small | $0.02/M | 1536 | multi | 8K tok | OpenRouter |
| text-embedding-3-large | $0.13/M | 3072 | multi | 8K tok | OpenRouter |

**→ Рекомендация: `gemini-embedding-001` (dim=1536) — бесплатно, 100+ языков, отличное качество.**

---

## 3. Векторное хранилище: **Qdrant Cloud** (free tier)

Qdrant Cloud имеет бесплатный уровень для 1ГБ данных — этого хватит на тысячи документов. Развёртывать Docker локально не нужно.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Подключение к Qdrant Cloud (бесплатный tier)
client = QdrantClient(
    url="https://xxx.qdrant.io",
    api_key="your-qdrant-cloud-key"
)

# Коллекция под Gemini Embedding dim=1536
client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Загрузка чанков
for chunk in chunks:
    vec = embed_chunks([chunk["search_text"]], dim=1536)[0]
    
    client.upsert("docs", [PointStruct(
        id=chunk["id"],
        vector=vec,
        payload={
            "type": chunk["type"],        # text | table | image
            "page": chunk["page"],
            "bbox": chunk["bbox"],
            "source_file": chunk["source_file"],
            "content": chunk["content"],   # полный текст/таблица для LLM
            "image_path": chunk.get("image_path"),  # путь к PNG
        }
    )])
```

### Что индексировать для каждого типа?

| Тип чанка | `search_text` (для embed) | `content` (для LLM) | `image_path` |
|---|---|---|---|
| **text** | Сам текст блока | Сам текст | — |
| **table** | LLM-саммари таблицы (1 раз) | Полная таблица Markdown | — |
| **image** | Окружающий текст + подпись | Подпись + описание из документа | `output/images/...png` |

---

## 4. LLM для генерации: самый дешёвый вариант

### Вариант A: **Qwen3-8B через OpenRouter** (ультра-дёшево)

Pricing starts at $0.050 per million input tokens and $0.400 per million output tokens. The model supports a context window of up to 41K tokens.

Qwen3-8B на OpenRouter: Context 128K tokens, Input Price $0.04/1M tokens, Output Price $0.14/1M tokens.

**Считаем**: 100 запросов/день × ~3000 input tokens × ~500 output tokens =  
- Input: 300K × $0.04/1M = **$0.012/день**  
- Output: 50K × $0.14/1M = **$0.007/день**  
- **Итого: ~$0.57/месяц** 🔥

### Вариант B: **Gemini 3 Flash через OpenRouter** (дороже, но мощнее)

Gemini 3 Flash Preview: Input $0.50 per 1M tokens, Output $3.00 per 1M tokens.

Те же 100 запросов/день → **~$6/месяц**

### Вариант C: **Gemini 2.5 Flash** (идеален для thinking)

Gemini 2.5 Flash is Google's state-of-the-art workhorse model, specifically designed for advanced reasoning, coding, mathematics, and scientific tasks.

### Вариант D: **GPT-4.1 nano** (если нужна скорость)

GPT-4.1 nano is the fastest and cheapest model in the GPT-4.1 series. It delivers exceptional performance at a small size with its 1 million token context window.

GPT-4.1 nano: $0.10/M input tokens, $0.40/M output tokens.

### Сравнительная таблица LLM

| Модель | Input/1M | Output/1M | Контекст | Качество RU | Месяц (100 q/day) |
|---|---|---|---|---|---|
| **Qwen3-8B (OpenRouter)** | **$0.04** | **$0.14** | 128K | ★★★★ | **~$0.57** |
| GPT-4.1 nano | $0.10 | $0.40 | 1M | ★★★ | ~$1.50 |
| Gemini 3 Flash | $0.50 | $3.00 | 1M | ★★★★ | ~$6 |
| Gemini 2.5 Flash | $0.15 | $0.60 | 1M | ★★★★★ | ~$2.10 |

**→ Рекомендация: Qwen3-8B для рутинных запросов, Gemini 2.5 Flash для сложных.**

---

## 5. Reranking без отдельной модели (экономия)

Вместо платного cross-encoder reranker — **LLM-based reranking** через тот же Qwen3-8B:

```python
RERANK_PROMPT = """Ранжируй эти {n} фрагментов по релевантности к вопросу.
Верни JSON: [{{"rank": 1, "id": "chunk_xxx"}}, ...]

Вопрос: {query}

Фрагменты:
{chunks_with_ids}
"""

# Один вызов Qwen3-8B → получаем порядок → берём top-5
```

**Стоимость**: ~2000 input tokens × $0.04/1M = **$0.00008 за rerank** — пренебрежимо.

Альтернатива: если качество критично, OpenRouter предлагает embedding-модели разного размера: smaller models (like qwen/qwen3-embedding-0.6b) are faster and cheaper, while larger models (like openai/text-embedding-3-large) provide better quality.

---

## 6. Чанкование таблиц: генерация саммари

Одноразовый шаг при ingestion — для каждой таблицы генерируем поисковое саммари:

```python
import openai

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_OPENROUTER_KEY"
)

def summarize_table(table_markdown: str) -> str:
    response = client.chat.completions.create(
        model="qwen/qwen3-8b",
        messages=[{
            "role": "user",
            "content": f"""Кратко опиши содержание этой таблицы на русском.
Укажи: что в строках, что в столбцах, ключевые значения.
2-3 предложения.

Таблица:
{table_markdown}"""
        }],
        max_tokens=200
    )
    return response.choices[0].message.content
```

**Стоимость**: 50 таблиц × ~1000 tokens = 50K tokens × $0.04/1M = **$0.002 за всё** 🔥

---

## 7. Полный пайплайн для менеджера

### 7.1. Чат-интерфейс: **Streamlit** (простейший вариант)

```python
import streamlit as st
import openai
from qdrant_client import QdrantClient
from google import genai
import os

# ---- Инициализация ----
gemini = genai.Client(api_key=st.secrets["GEMINI_KEY"])
qdrant = QdrantClient(url=st.secrets["QDRANT_URL"], api_key=st.secrets["QDRANT_KEY"])
openrouter = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_KEY"]
)

def search_and_answer(query: str):
    # 1. Embed запрос (бесплатно)
    q_emb = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config={"output_dimensionality": 1536}
    ).embeddings[0].values
    
    # 2. Поиск в Qdrant (бесплатно)
    hits = qdrant.query_points(
        collection_name="docs",
        query=q_emb,
        limit=10
    ).points
    
    # 3. Формируем контекст
    context_parts = []
    sources = []
    images = []
    
    for i, hit in enumerate(hits):
        p = hit.payload
        context_parts.append(
            f"[Фрагмент {i+1} | {p['source_file']}, стр.{p['page']}, "
            f"тип:{p['type']}]\n{p['content'][:1500]}"
        )
        sources.append(p)
        if p.get("image_path") and p["type"] == "image":
            images.append(p)
    
    context = "\n\n---\n\n".join(context_parts)
    
    # 4. Генерация ответа ($0.04/1M input)
    response = openrouter.chat.completions.create(
        model="qwen/qwen3-8b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"}
        ],
        max_tokens=1000
    )
    
    return response.choices[0].message.content, sources, images

# ---- UI ----
st.title("📚 Ассистент по документации")

query = st.chat_input("Задайте вопрос...")
if query:
    with st.spinner("Ищу..."):
        answer, sources, images = search_and_answer(query)
    
    st.markdown(answer)
    
    # Источники
    with st.expander("📎 Источники"):
        for s in sources[:5]:
            st.markdown(
                f"• **{s['source_file']}** — стр. {s['page']}, "
                f"тип: `{s['type']}`, bbox: `{s['bbox']}`"
            )
    
    # Изображения
    if images:
        st.subheader("🖼️ Релевантные схемы")
        for img in images:
            if os.path.exists(img["image_path"]):
                st.image(img["image_path"], 
                        caption=f"{img['source_file']}, стр. {img['page']}")
```

### 7.2. System Prompt

```python
SYSTEM_PROMPT = """Ты — ассистент по технической документации. 

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленных фрагментов.
2. Каждое утверждение сопровождай ссылкой:
   [Источник: название_файла, стр. X, тип]
3. Если найдена таблица — процитируй конкретные ячейки/строки.
4. Если есть релевантная схема/изображение — упомяни:
   "См. схему: [название_файла, стр. X]"
5. Если информации нет — скажи: "В документации не найдено."
6. Отвечай на русском языке.
"""
```

---

## 8. Итоговый бюджет

| Компонент | Сервис | Стоимость/месяц |
|---|---|---|
| **Embedding** | Gemini API (free tier) | **$0** |
| **Vector DB** | Qdrant Cloud (free tier, 1GB) | **$0** |
| **LLM (ответы)** | Qwen3-8B на OpenRouter | **~$0.57** (100 q/day) |
| **LLM (саммари таблиц)** | Qwen3-8B (одноразово) | **~$0.01** |
| **Chat UI** | Streamlit Community Cloud | **$0** |
| **ИТОГО** | | **< $1/месяц** 🎯 |

При росте нагрузки до 500 запросов/день → **~$3/месяц** с Qwen3-8B или **~$10/месяц** с Gemini 2.5 Flash.

---

## 9. Roadmap по шагам

```
Неделя 1:
  ☐ Получить API keys: Gemini, OpenRouter, Qdrant Cloud
  ☐ Скрипт embed_and_upload.py:
      chunks.json → gemini-embedding-001 → Qdrant Cloud
  ☐ Для таблиц: Qwen3-8B генерирует саммари → embed саммари
  ☐ Для изображений: берём окружающий текст → embed текст

Неделя 2:
  ☐ search.py: query → embed → Qdrant → top-10 → LLM rerank → top-5
  ☐ generate.py: top-5 чанков + prompt → Qwen3-8B → ответ + источники
  ☐ Тесты: 10 golden questions → проверка релевантности

Неделя 3:
  ☐ Streamlit app: chat + источники + отображение PNG-схем
  ☐ Deploy на Streamlit Cloud (бесплатно)
  ☐ Показать менеджеру, собрать feedback
```

---

## 10. Ответы на исходные вопросы PRD

| Вопрос из PRD | Решение |
|---|---|
| **Изображения: PNG vs base64?** | PNG-файлы на диске. В Qdrant — только `image_path` в payload. Streamlit рендерит `st.image(path)`. |
| **Чанкование длинных текстов?** | Каждый Docling `TextItem` = 1 чанк. Если > 1500 символов — split по абзацам с overlap 200. |
| **Формат источника?** | Inline: `[Источник: table11.pdf, стр.3, Таблица]` + миниатюра PNG если type=image. Из payload Qdrant: `source_file`, `page`, `type`, `bbox`. |
