"""
search_service.py — Поисковый сервис (Retrieval & RAG).

Выполняет:
1. Векторизацию запроса (Gemini Embedding).
2. Поиск в Qdrant (Retrieval).
3. Обработку результатов (вкл. подгрузку полных таблиц).
4. Опциональный LLM Reranking (через Qwen/OpenRouter).
5. Генерацию ответа (RAG Generation) с цитатами.
"""

import os
import argparse
from pathlib import Path

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None

try:
    from qdrant_client import QdrantClient
except ImportError:
    QdrantClient = None

try:
    import openai
except ImportError:
    openai = None


COLLECTION_NAME = "docs"
EMBEDDING_DIM = 768


def get_gemini_client():
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ Ошибка: Переменная среды GEMINI_API_KEY не найдена.")
        return None
    # Принудительно удаляем GOOGLE_API_KEY, чтобы SDK не пытался использовать Vertex AI
    if "GOOGLE_API_KEY" in os.environ:
        del os.environ["GOOGLE_API_KEY"]
    return genai.Client(api_key=gemini_key)


def get_qdrant_client(mode="cloud"):
    if mode == "memory":
        return QdrantClient(location=":memory:")
    q_url = os.environ.get("QDRANT_URL")
    q_key = os.environ.get("QDRANT_API_KEY")
    if not q_url or not q_key:
        print("❌ Ошибка: QDRANT_URL и QDRANT_API_KEY обязательны для cloud режима.")
        return None
    return QdrantClient(url=q_url, api_key=q_key)


def get_openrouter_client():
    if openai is None:
        return None
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )


def embed_query(client, query: str) -> list[float]:
    print("🧠 Векторизация запроса...")
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config=genai_types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=EMBEDDING_DIM
            )
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"❌ Ошибка векторизации: {e}")
        return [0.0] * EMBEDDING_DIM


def retrieve(qdrant, query_vector: list[float], limit: int = 15):
    print(f"🔎 Поиск в Qdrant (Limit: {limit})...")
    try:
        hits_response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit
        )
        return hits_response.points
    except Exception as e:
        print(f"❌ Ошибка поиска: {e}")
        return []


def process_hits(hits):
    processed = []
    images = []
    
    for h in hits:
        payload = h.payload or {}
        source_file = payload.get("source_file", "unknown")
        hierarchy = payload.get("section_hierarchy", [])
        chunk_type = payload.get("type", "unknown")
        content = payload.get("content", "")
        content_file = payload.get("content_file")
        image_path = payload.get("image_path")
        
        # Подгрузка полного текста таблицы
        if chunk_type == "table_rows" and content_file:
            path = Path(content_file)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
        
        if image_path and image_path not in images:
            images.append(image_path)
            
        hierarchy_str = " > ".join(hierarchy) if hierarchy else "Основной раздел"
        
        processed.append({
            "id": str(h.id),
            "score": h.score,
            "source_file": source_file,
            "hierarchy": hierarchy_str,
            "content": content,
            "chunk_type": chunk_type,
            "original_chunk_id": payload.get("original_chunk_id", str(h.id))
        })
        
    return processed, images


def rerank_hits(or_client, query: str, hits: list, top_k: int = 5):
    if not or_client:
        print("⚠️  OpenRouter клиент не настроен, LLM Reranking пропущен.")
        return hits[:top_k]
        
    print("⚖️  LLM Reranking (через OpenRouter)...")
    
    context_str = ""
    for i, h in enumerate(hits):
        snippet_text = h.get('content', '')[:300].replace('\n', ' ')
        snippet = f"Раздел: {h.get('hierarchy', '')}. Текст: {snippet_text}"
        context_str += f"[{i}] {snippet}\n\n"
        
    prompt = f"""Запрос пользователя: "{query}"
    
Оцени релевантность следующих {len(hits)} фрагментов текста относительно запроса пользователя. 
Верни ТОЛЬКО индексы (от 0 до {len(hits)-1}) самых релевантных фрагментов через запятую (максимум {top_k}).
Например, если релевантны только 3-й и 0-й, верни: 3, 0

Фрагменты:
{context_str}"""

    try:
        response = or_client.chat.completions.create(
            model="qwen/qwen3-8b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20
        )
        reply = response.choices[0].message.content.strip()
        
        import re
        numbers = re.findall(r'\d+', reply)
        selected_indices = []
        for n in numbers:
            idx = int(n)
            if idx < len(hits) and idx not in selected_indices:
                selected_indices.append(idx)
                
        if not selected_indices:
            selected_indices = list(range(min(top_k, len(hits))))
            
        # Дополняем если нужно
        for i in range(len(hits)):
            if len(selected_indices) >= top_k:
                break
            if i not in selected_indices:
                selected_indices.append(i)
                
        reranked = [hits[i] for i in selected_indices[:top_k]]
        return reranked
    except Exception as e:
        print(f"⚠️  Ошибка реранжирования: {e}")
        return hits[:top_k]


def generate_answer(client, query: str, contexts: list, images: list) -> str:
    print("💬 Генерация ответа (Gemini 1.5 Flash)...")
    
    context_str = ""
    for i, ctx in enumerate(contexts):
        context_str += f"""
--- Источник {i+1} ---
Документ: {ctx['source_file']}
Раздел: {ctx['hierarchy']}
ID чанка: {ctx['original_chunk_id']}
Текст:
{ctx['content']}
"""

    prompt = f"""Системная инструкция: Ты экспертный помощник. Отвечай на вопрос пользователя, основываясь ТОЛЬКО на предоставленном контексте документов. 
Твоя задача помочь пользователю найти нужную информацию.
ОБЯЗАТЕЛЬНО используй строгие ссылки на источники (например, "Согласно документу [Документ], в разделе [Раздел]..."). 
Если в контексте нет ответа на вопрос, честно скажи об этом.

Контекст:
{context_str}

Запрос пользователя:
{query}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"❌ Ошибка генерации ответа: {e}")
        return f"Ошибка генерации ответа: {e}"

def run_cli_search(query: str, mode: str="cloud", rerank: bool=False, limit: int=15, top_k: int=5):
    if genai is None or QdrantClient is None:
        print("❌ Установите зависимости: pip install google-genai qdrant-client")
        return
        
    gemini = get_gemini_client()
    qdrant = get_qdrant_client(mode)
    if not gemini or not qdrant:
        return
        
    or_client = None
    if rerank:
        or_client = get_openrouter_client()
        
    query_vector = embed_query(gemini, query)
    if all(v == 0.0 for v in query_vector):
        print("❌ Невозможно выполнить поиск: вектор нулевой.")
        return
        
    raw_hits = retrieve(qdrant, query_vector, limit=limit)
    if not raw_hits:
        print("🤷 Ничего не найдено в базе данных.")
        return
        
    processed_hits, images = process_hits(raw_hits)
    
    if rerank:
        final_hits = rerank_hits(or_client, query, processed_hits, top_k=top_k)
    else:
        final_hits = processed_hits[:top_k]
        
    answer = generate_answer(gemini, query, final_hits, images)
    
    print("\n" + "="*50)
    print(f"ВОПРОС: {query}")
    print("="*50)
    print(answer)
    print("="*50)
    
    if images:
        print("\n🖼️ Связанные изображения:")
        for img in images:
            print(f"  - {img}")


def main():
    parser = argparse.ArgumentParser(description="Поисковый сервис RAG на базе Qdrant и Gemini.")
    parser.add_argument("query", type=str, help="Поисковый запрос")
    parser.add_argument("--mode", type=str, choices=["memory", "cloud"], default="cloud", help="Режим Qdrant")
    parser.add_argument("--rerank", action="store_true", help="Включить LLM Reranking (через OpenRouter)")
    parser.add_argument("--limit", type=int, default=15, help="Количество результатов для извлечения из векторной БД")
    parser.add_argument("--top-k", type=int, default=5, help="Количество результатов для отправки генеративной модели")
    args = parser.parse_args()
    
    run_cli_search(args.query, args.mode, args.rerank, args.limit, args.top_k)


if __name__ == "__main__":
    main()
