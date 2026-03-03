"""
keep_alive.py — Скрипт для пинга Qdrant Cloud (предотвращение засыпания Free Tier).

Free Tier Qdrant кластеры автоматически удаляются после 4 недель неактивности.
Запускайте этот скрипт кроном раз в неделю.

Настройка crontab:
0 0 * * 0 cd /path/to/project && /path/to/venv/bin/python src/keep_alive.py >> keep_alive.log 2>&1
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Пытаемся импортировать qdrant_client
try:
    from qdrant_client import QdrantClient
except ImportError:
    print("❌ Ошибка: qdrant-client не установлен.")
    exit(1)

# Загружаем ключи
load_dotenv()
qdrant_url = os.environ.get("QDRANT_URL")
qdrant_key = os.environ.get("QDRANT_API_KEY")

if not qdrant_url or not qdrant_key:
    print(f"[{datetime.now().isoformat()}] ⚠️ Пропуск: QDRANT_URL или QDRANT_API_KEY не заданы в окружении.")
    exit(0)

print(f"[{datetime.now().isoformat()}] 🔄 Пинг кластера Qdrant Cloud: {qdrant_url}...")

try:
    start_time = time.time()
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    
    # Легкий запрос, достаточный для поддержания активности
    collections = client.get_collections()
    
    elapsed = time.time() - start_time
    
    print(f"[{datetime.now().isoformat()}] ✅ Успешно за {elapsed:.2f} сек. Найдено коллекций: {len(collections.collections)}")
    for c in collections.collections:
        print(f"   - {c.name}")
        
except Exception as e:
    print(f"[{datetime.now().isoformat()}] ❌ Ошибка пинга Qdrant: {e}")
    exit(1)
