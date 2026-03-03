"""
embed_and_upload.py — Скрипт векторизации семантических чанков и загрузки в Qdrant.

Принимает на вход *_semantic.json, генерирует векторы через Gemini Embedding 001
и загружает их в векторную БД Qdrant.

Поддерживает 2 режима:
  --mode memory : In-memory Qdrant (для локального тестирования без ключей Qdrant)
  --mode cloud  : Qdrant Cloud (использует QDRANT_URL и QDRANT_API_KEY из .env)

Сохраняет локальный бэкап векторов 'vectors.npy' в папке output/.
"""

import json
import argparse
import os
import time
from pathlib import Path

# Подгрузка локальных модулей
import numpy as np

# Пытаемся загрузить dotenv
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

# Опциональные зависимости для Embedding / Qdrant
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
except ImportError:
    QdrantClient = None


# ─── Настройки ───────────────────────────────────────────────────────────────

COLLECTION_NAME = "docs"
EMBEDDING_DIM = 768  # Усечённая размерность для экономии в Free Tier
RATE_LIMIT_PAUSE = 0.15  # Секунд между запросами к Gemini (limit 15 RPM for free tier -> 4 sec)
                        # ВНИМАНИЕ: Free tier Gemini API (15 RPM) требует rate limiting ~4с.
                        # Если у вас платный/высокий лимит, можно уменьшить до 0.1s.


# ─── Вычислительные функции ──────────────────────────────────────────────────

def embed_texts(texts: list[str], client, pause: float) -> list[list[float]]:
    """
    Векторизация текстов через Gemini с rate limiting.
    task_type = RETRIEVAL_DOCUMENT для индексации.
    """
    vectors = []
    total = len(texts)
    
    print(f"🧠 Векторизация {total} текстов (dim={EMBEDDING_DIM})...")
    
    # Если текстов много и free tier (15 RPM), нужно сказать пользователю
    if total > 50 and pause > 1.0:
        print(f"⚠️  Для {total} текстов с паузой {pause}с это займёт ~{total*pause/60:.1f} минут.")
        
    for i, text in enumerate(texts):
        # Если пусто
        if not text.strip():
            # Заглушка (нулевой вектор)
            vectors.append([0.0] * EMBEDDING_DIM)
            continue
            
        if client is None:
            # Режим MOCK
            import random
            mock_vector = [random.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIM)]
            vectors.append(mock_vector)
            if (i+1) % 10 == 0:
                print(f"   [{i+1}/{total}] (mock)...")
            continue

        try:
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=genai_types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIM
                )
            )
            vectors.append(result.embeddings[0].values)
            
            if (i+1) % 10 == 0:
                print(f"   [{i+1}/{total}] ...")
                
        except Exception as e:
            print(f"❌ Ошибка векторизации на чанке {i}: {e}")
            # Вставляем NaN вектор как fallback (будет проигнорирован при поиске)
            vectors.append([0.0] * EMBEDDING_DIM)
            
        # Rate limit
        if i < total - 1:
            time.sleep(pause)
            
    return vectors


# ─── Основная логика пайплайна ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Embed and Upload семантических чанков в Qdrant.")
    parser.add_argument("--input", type=str, required=True,
                        help="Путь к *_semantic.json файлу (или директория с JSON-файлами)")
    parser.add_argument("--mode", type=str, choices=["memory", "cloud"], default="cloud",
                        help="Режим работы Qdrant (memory - локально в ОЗУ, cloud - облачный кластер)")
    parser.add_argument("--rate-limit", type=float, default=0.2,
                        help="Пауза между запросами к Gemini в секундах (default=0.2)")
    parser.add_argument("--mock-embeddings", action="store_true",
                        help="Использовать фейковые случайные эмбеддинги (для тестов без API-ключа)")
    
    args = parser.parse_args()

    # 1. Проверка библиотек
    if genai is None:
        print("❌ Ошибка: пакет google-genai не установлен. (pip install google-genai)")
        return
    if QdrantClient is None:
        print("❌ Ошибка: пакет qdrant-client не установлен. (pip install qdrant-client)")
        return

    # 2. Инициализация Клиентов
    if args.mock_embeddings:
        print("⚠️  Включен режим MOCK для эмбеддингов (API Gemini не используется).")
        gemini_client = None
    else:
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            print("❌ Ошибка: Переменная среды GEMINI_API_KEY не найдена.")
            return
            
        gemini_client = genai.Client(api_key=gemini_key)
    
    # 3. Инициализация Qdrant
    if args.mode == "memory":
        print("🔄 Старт Qdrant: режим In-Memory (локально)")
        qdrant_client = QdrantClient(location=":memory:")
    else:
        q_url = os.environ.get("QDRANT_URL")
        q_key = os.environ.get("QDRANT_API_KEY")
        if not q_url or not q_key:
            print("❌ Ошибка: QDRANT_URL и QDRANT_API_KEY обязательны для cloud режима.")
            return
        print(f"☁️ Старт Qdrant: подключение к Cloud ({q_url})")
        qdrant_client = QdrantClient(url=q_url, api_key=q_key)

    # Убеждаемся, что коллекция существует (пересоздаем, если нужно для тестов)
    # В проде (cloud) лучше проверять существование:
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"   Создание коллекции '{COLLECTION_NAME}' (dim={EMBEDDING_DIM})...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
    else:
        print(f"   Коллекция '{COLLECTION_NAME}' уже существует.")

    # 4. Загрузка данных
    input_path = Path(args.input)
    files_to_process = []
    
    if input_path.is_file():
        files_to_process.append(input_path)
    elif input_path.is_dir():
        for f in input_path.glob("*_semantic.json"):
            files_to_process.append(f)
            
    if not files_to_process:
        print(f"❌ Не найдено *_semantic.json файлов по пути: {input_path}")
        return
        
    for json_file in files_to_process:
        print(f"\n📂 Обработка: {json_file.name}")
        
        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        if not chunks:
            print("   (Файл пуст, пропускаем)")
            continue
            
        # 5. Подготовка текстов для векторизации
        search_texts = [c.get("search_text", "") for c in chunks]
        
        # 6. Векторизация
        vectors = embed_texts(search_texts, gemini_client, pause=args.rate_limit)
        
        # Локальный бэкап numpy
        vectors_np = np.array(vectors, dtype=np.float32)
        backup_path = json_file.with_name(json_file.stem + "_vectors.npy")
        np.save(str(backup_path), vectors_np)
        print(f"   💾 Локальный бэкап векторов сохранён: {backup_path.name}")
        
        # 7. Загрузка в Qdrant
        print("☁️ Загрузка данных в Qdrant Cloud...")
        points = []
        for i, chunk in enumerate(chunks):
            chunk_id_str = chunk.get("chunk_id", f"sem_{i:04d}")
            chunk_id_str = f"{json_file.stem}_{chunk_id_str}"
            # Создаем целочисленный хэш-ID из строкового, так как Qdrant принимает uuid или целое
            import hashlib
            hash_full = hashlib.md5(chunk_id_str.encode()).hexdigest()
            qdrant_id = chunk.get("id") # Если есть uuid, иначе генерируем
            if not qdrant_id: # Для упрощения используем UUID из хеша
                import uuid
                qdrant_id = str(uuid.UUID(hash_full)) 
            
            # В payload кладем все, КРОМЕ search_text (экономия памяти)
            payload = chunk.get("payload", {})
            payload["original_chunk_id"] = chunk_id_str # Сохраним оригинальный текстовый ID
            
            points.append(
                PointStruct(
                    id=qdrant_id,
                    vector=vectors[i],
                    payload=payload
                )
            )
            
        # Батч загрузка по 100 элементов
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
            
        print(f"✅ Успешно загружено {len(points)} точек в Qdrant.")
        
    print("\n🎉 Векторизация и загрузка завершена!")

if __name__ == "__main__":
    main()
