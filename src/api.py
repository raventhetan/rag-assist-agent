"""
api.py — FastAPI эндпоинт для интеграции с OpenWebUI

Реализует OpenAI-совместимый интерфейс `/v1/chat/completions` для использования RAG-пайплайна 
из OpenWebUI как обычную языковую модель. Запросы перехватываются и перенаправляются 
в наш `search_service.py` (Gemini embeddings + Qdrant search + LLM Rerank + Gemini generation).
"""

import time
import uuid
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

from search_service import (
    get_gemini_client,
    get_qdrant_client,
    get_openrouter_client,
    embed_query,
    retrieve,
    process_hits,
    rerank_hits,
    generate_answer
)

# Глобальные клиенты
gemini_client = None
qdrant_client = None
or_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global gemini_client, qdrant_client, or_client
    print("🚀 Инициализация клиентов RAG...")
    gemini_client = get_gemini_client()
    qdrant_client = get_qdrant_client(mode="cloud")
    or_client = get_openrouter_client()
    yield
    print("🛑 Отключение...")

app = FastAPI(title="RAG Assist Agent API", lifespan=lifespan)

# Монтируем директорию с картинками для прямой раздачи
import os
os.makedirs("output/images", exist_ok=True)
app.mount("/images", StaticFiles(directory="output/images"), name="images")


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False


@app.get("/v1/models")
async def list_models():
    """Возвращает фиктивную RAG модель для OpenWebUI."""
    return {
        "object": "list",
        "data": [
            {
                "id": "rag-assist-agent",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    """
    Основной эндпоинт. Извлекает последний запрос пользователя из истории (messages),
    прогоняет через пайплайн RAG и возвращает результат в формате OpenAI.
    """
    
    # Найти последний запрос пользователя
    user_query = ""
    for msg in reversed(req.messages):
        if msg.role == "user":
            user_query = msg.content
            break
            
    if not user_query:
        # Если нет запроса, просто вернем пустой
        return generate_openai_response("Пожалуйста, задайте вопрос.", req.model)

    # Запускаем RAG
    if not gemini_client or not qdrant_client:
        return generate_openai_response("Внутренняя ошибка сервера: Клиенты RAG не инициализированы", req.model)

    # 1. Векторизация
    query_vector = embed_query(gemini_client, user_query)
    
    # 2. Поиск в Qdrant
    if all(v == 0.0 for v in query_vector):
        return generate_openai_response("Ошибка: вектор запроса пустой", req.model)
        
    raw_hits = retrieve(qdrant_client, query_vector, limit=75)
    
    # 3. Обработка (и подгрузка таблиц)
    processed_hits, images = process_hits(raw_hits)
    if not processed_hits:
        return generate_openai_response("Не найдено релевантных ответов в базе знаний.", req.model)
        
    # 4. LLM Reranking (включено если есть OpenRouter API key)
    final_hits = rerank_hits(or_client, user_query, processed_hits, top_k=6) if or_client else processed_hits[:6]
    
    # 5. Генерация
    answer = generate_answer(gemini_client, user_query, final_hits, images)
    
    # Строим финальный ответ: текст + картинки
    images_md = ""
    if images:
        images_md = "\n\n**Связанные изображения:**\n\n"
        
        base_url = os.environ.get("BASE_IMAGE_URL")
        if not base_url:
            base_url = str(request.base_url).rstrip('/')
            # Переводим внутренний докер-хост в localhost для корректного отображения в браузере пользователя
            if "host.docker.internal" in base_url:
                base_url = base_url.replace("host.docker.internal", "localhost")
        
        for img in images:
            # Обрезаем префикс пути, если он абсолютный или начинается с src/output
            img_rel = str(img)
            if "output/images/" in img_rel:
                img_rel = img_rel.split("output/images/")[-1]
            elif "output/" in img_rel:
                img_rel = img_rel.split("output/")[-1]
                
            images_md += f"![{os.path.basename(img)}]({base_url}/images/{img_rel.lstrip('/')})\n"
            
    final_response_text = f"{answer}{images_md}"

    if req.stream:
        return StreamingResponse(fake_stream(final_response_text, req.model), media_type="text/event-stream")
    else:
        return generate_openai_response(final_response_text, req.model)


def generate_openai_response(text: str, model: str):
    """Формат для не-стримингового ответа."""
    return JSONResponse(content={
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    })

async def fake_stream(text: str, model: str):
    """Генерирует SSE для интерфейса OpenWebUI."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    
    # Отправка роли
    chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    
    # Отправка текста частями (имитация стриминга)
    chunk_size = 5
    for i in range(0, len(text), chunk_size):
        part = text[i:i+chunk_size]
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        
    # Завершение потока
    chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
