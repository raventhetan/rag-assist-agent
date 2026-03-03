"""
semantic_chunker.py — Structure-aware семантический группировщик чанков для RAG.

Принимает сырые *_chunks.json (выход export_chunks.py) и формирует *_semantic.json
с оптимизированными чанками для эмбеддинга и поиска.

Стратегии:
  1. Текст: Structure-aware merge по заголовкам, до ~512 tokens (~1500 символов)
  2. Таблицы: Multi-vector (саммари + группы строк)
  3. Изображения: Контекстное обогащение соседним текстом
  4. Contextual Retrieval: LLM генерирует context_prefix для каждого чанка

Использование:
  python src/semantic_chunker.py --input output/table11_chunks.json [--contextualize]
"""

import json
import argparse
import os
import re
import time
from pathlib import Path
from typing import Optional


# ─── Настройки ───────────────────────────────────────────────────────────────

# Максимальное количество символов для одного текстового чанка (~512 tokens)
MAX_CHUNK_CHARS = 1500

# Количество строк в одной группе строк таблицы
TABLE_ROW_GROUP_SIZE = 8

# Максимальная длина content, которая хранится в payload (для экономии Qdrant)
MAX_CONTENT_PAYLOAD_CHARS = 3000

# Количество символов контекста для изображений (до + после)
IMAGE_CONTEXT_CHARS = 500

# Типы, которые считаются заголовками (границами разделов)
HEADER_TYPES = {"section_header", "title"}

# Типы, которые считаются текстовым контентом (склеиваются)
TEXT_TYPES = {"text", "paragraph", "list_item", "caption", "page_header", "page_footer", "form", "code"}

# Таблицы
TABLE_TYPES = {"table"}

# Изображения
IMAGE_TYPES = {"picture", "figure", "image"}


# ─── Утилиты ─────────────────────────────────────────────────────────────────

def count_tokens_approx(text: str) -> int:
    """Примерный подсчет токенов (1 токен ≈ 3 символа для русского текста)."""
    return len(text) // 3


def recursive_split(text: str, max_chars: int) -> list[str]:
    """
    Рекурсивный сплит текста: сначала по \\n\\n, затем \\n, затем '. '.
    Возвращает список фрагментов, каждый <= max_chars.
    """
    if len(text) <= max_chars:
        return [text]

    # Попробовать разделители в порядке приоритета
    for sep in ["\n\n", "\n", ". "]:
        parts = text.split(sep)
        if len(parts) > 1:
            result = []
            current = ""
            for part in parts:
                candidate = (current + sep + part) if current else part
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        result.append(current)
                    # Если отдельная часть больше max_chars — рекурсия
                    if len(part) > max_chars:
                        result.extend(recursive_split(part, max_chars))
                        current = ""
                    else:
                        current = part
            if current:
                result.append(current)
            return result

    # Крайний случай: просто разрезать по max_chars
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def merge_bboxes(bboxes: list[list]) -> list[list]:
    """Объединить массив bbox, вернуть все уникальные."""
    seen = set()
    result = []
    for b in bboxes:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            result.append(b)
    return result


def build_section_hierarchy(headers_stack: list[str]) -> list[str]:
    """Сформировать иерархию заголовков."""
    return [h for h in headers_stack if h]


def parse_table_rows(markdown_table: str) -> list[str]:
    """
    Разбить Markdown-таблицу на строки данных (без заголовка и разделителя).
    Возвращает список строк (каждая — один ряд таблицы).
    """
    lines = markdown_table.strip().split("\n")
    if not lines:
        return []

    # Найти заголовок и разделитель
    header_line = ""
    data_lines = []
    separator_found = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\|[-\s|]+\|$", stripped):
            separator_found = True
            continue
        if not separator_found:
            header_line = stripped
        else:
            data_lines.append(stripped)

    return data_lines


# ─── LLM интеграция (опциональная) ──────────────────────────────────────────

def get_openrouter_client():
    """Инициализировать OpenAI-совместимый клиент для OpenRouter."""
    try:
        import openai
    except ImportError:
        print("⚠️  openai не установлен. Контекстуализация и саммари таблиц будут пропущены.")
        return None

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠️  OPENROUTER_API_KEY не задан. Контекстуализация и саммари таблиц будут пропущены.")
        return None

    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )


def llm_summarize_table(client, table_markdown: str, source_file: str = "") -> str:
    """Сгенерировать краткое саммари таблицы через LLM."""
    if client is None:
        # Фоллбэк: взять первые 200 символов таблицы
        return f"Таблица из {source_file}: {table_markdown[:200]}..."

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3-8b",
            messages=[{
                "role": "user",
                "content": f"""Кратко опиши содержание этой таблицы на русском.
Укажи: что в строках, что в столбцах, ключевые значения.
2-3 предложения.

Таблица:
{table_markdown[:3000]}"""
            }],
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️  LLM summarize error: {e}")
        return f"Таблица из {source_file}: {table_markdown[:200]}..."


def llm_contextualize(client, chunk_text: str, doc_title: str,
                       section_hierarchy: list[str]) -> str:
    """Сгенерировать контекстный префикс для чанка через LLM."""
    if client is None:
        # Фоллбэк: использовать иерархию заголовков
        hierarchy_str = " > ".join(section_hierarchy) if section_hierarchy else doc_title
        return f"Этот фрагмент относится к разделу: {hierarchy_str}."

    hierarchy_str = " > ".join(section_hierarchy) if section_hierarchy else "Основной раздел"

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3-8b",
            messages=[{
                "role": "user",
                "content": f"""Вот документ: {doc_title}
Раздел: {hierarchy_str}

Вот фрагмент из этого документа:
<chunk>{chunk_text[:1500]}</chunk>

Дай краткое (2-3 предложения) описание контекста этого фрагмента,
чтобы его можно было понять без остального документа.
Ответь ТОЛЬКО контекстным описанием, без пересказа содержания."""
            }],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️  LLM contextualize error: {e}")
        hierarchy_str = " > ".join(section_hierarchy) if section_hierarchy else doc_title
        return f"Этот фрагмент относится к разделу: {hierarchy_str}."


# ─── Основная логика ─────────────────────────────────────────────────────────

def process_chunks(raw_chunks: list[dict], source_file: str,
                   llm_client=None, do_contextualize: bool = False,
                   full_tables_dir: Optional[Path] = None) -> list[dict]:
    """
    Основной пайплайн семантического группирования.

    Этапы:
      1.1 Structure-aware merge текстовых чанков
      1.2 Таблицы → Multi-vector (саммари + row_groups)
      1.3 Изображения → контекстное обогащение
      1.4 Contextual Retrieval (context_prefix)
    """

    semantic_chunks = []
    chunk_counter = 0

    # Стек заголовков для иерархии
    headers_stack = []

    # Буфер для склейки текстовых блоков
    text_buffer = []
    text_buffer_pages = []
    text_buffer_bboxes = []

    def flush_text_buffer():
        """Сбросить текстовый буфер в один или несколько семантических чанков."""
        nonlocal chunk_counter

        if not text_buffer:
            return

        merged_text = "\n".join(text_buffer)
        merged_pages = sorted(set(text_buffer_pages))
        merged_bboxes = merge_bboxes(text_buffer_bboxes)
        hierarchy = build_section_hierarchy(list(headers_stack))

        # Если текст больше лимита — recursive split
        fragments = recursive_split(merged_text, MAX_CHUNK_CHARS)

        for frag in fragments:
            sem_chunk = {
                "chunk_id": f"sem_{chunk_counter:04d}",
                "chunk_type": "text",
                "search_text": frag,  # будет обогащён context_prefix позже
                "context_prefix": "",
                "payload": {
                    "type": "text",
                    "source_file": source_file,
                    "pages": merged_pages,
                    "bboxes": merged_bboxes,
                    "content": frag,
                    "content_file": None,
                    "image_path": None,
                    "parent_id": None,
                    "section_hierarchy": hierarchy,
                }
            }
            semantic_chunks.append(sem_chunk)
            chunk_counter += 1

        text_buffer.clear()
        text_buffer_pages.clear()
        text_buffer_bboxes.clear()

    # ──── Главный цикл по сырым чанкам ────────────────────────────────────

    for i, chunk in enumerate(raw_chunks):
        chunk_type = chunk.get("type", "unknown")
        content = chunk.get("content", "")
        page = chunk.get("page")
        bbox = chunk.get("bbox")
        image_path = chunk.get("image_path")

        # ── ЗАГОЛОВКИ: граница раздела ──
        if chunk_type in HEADER_TYPES:
            # Сбросить буфер текста предыдущего раздела
            flush_text_buffer()

            # Определить уровень заголовка (эвристика по типу)
            if chunk_type == "title":
                headers_stack = [content]
            else:
                # section_header: добавляем в стек (не более 3 уровней)
                if len(headers_stack) >= 3:
                    headers_stack[-1] = content
                else:
                    headers_stack.append(content)

            # Заголовок тоже добавляем в текстовый буфер (он важен для контекста)
            text_buffer.append(content)
            if page is not None:
                text_buffer_pages.append(page)
            if bbox is not None:
                text_buffer_bboxes.append(bbox)

        # ── ТЕКСТОВЫЕ БЛОКИ: склеиваем ──
        elif chunk_type in TEXT_TYPES:
            candidate_text = "\n".join(text_buffer + [content])

            # Если буфер переполнится — сбросить
            if len(candidate_text) > MAX_CHUNK_CHARS and text_buffer:
                flush_text_buffer()

            text_buffer.append(content)
            if page is not None:
                text_buffer_pages.append(page)
            if bbox is not None:
                text_buffer_bboxes.append(bbox)

        # ── ТАБЛИЦЫ: multi-vector ──
        elif chunk_type in TABLE_TYPES:
            flush_text_buffer()

            hierarchy = build_section_hierarchy(list(headers_stack))

            # Сохранить полную таблицу в файл, если она большая
            content_file = None
            content_for_payload = content
            if full_tables_dir and len(content) > MAX_CONTENT_PAYLOAD_CHARS:
                full_tables_dir.mkdir(parents=True, exist_ok=True)
                table_filename = f"{source_file}_table_{chunk_counter:04d}.md"
                table_path = full_tables_dir / table_filename
                table_path.write_text(content, encoding="utf-8")
                content_file = str(table_path)
                content_for_payload = content[:MAX_CONTENT_PAYLOAD_CHARS]

            # Чанк A: Саммари таблицы
            summary = llm_summarize_table(llm_client, content, source_file)
            parent_id = f"sem_{chunk_counter:04d}"

            sem_summary = {
                "chunk_id": parent_id,
                "chunk_type": "table_summary",
                "search_text": summary,
                "context_prefix": "",
                "payload": {
                    "type": "table_summary",
                    "source_file": source_file,
                    "pages": [page] if page else [],
                    "bboxes": [bbox] if bbox else [],
                    "content": content_for_payload,
                    "content_file": content_file,
                    "image_path": None,
                    "parent_id": None,
                    "section_hierarchy": hierarchy,
                }
            }
            semantic_chunks.append(sem_summary)
            chunk_counter += 1

            # Чанки B-N: Группы строк таблицы
            data_rows = parse_table_rows(content)
            if data_rows:
                # Первая строка — это заголовок, берём его из Markdown
                header_lines = content.strip().split("\n")
                table_header = header_lines[0] if header_lines else ""

                for row_start in range(0, len(data_rows), TABLE_ROW_GROUP_SIZE):
                    row_group = data_rows[row_start:row_start + TABLE_ROW_GROUP_SIZE]
                    row_text = table_header + "\n" + "\n".join(row_group)

                    sem_rows = {
                        "chunk_id": f"sem_{chunk_counter:04d}",
                        "chunk_type": "table_rows",
                        "search_text": row_text,
                        "context_prefix": "",
                        "payload": {
                            "type": "table_rows",
                            "source_file": source_file,
                            "pages": [page] if page else [],
                            "bboxes": [bbox] if bbox else [],
                            "content": row_text,
                            "content_file": content_file,
                            "image_path": None,
                            "parent_id": parent_id,
                            "section_hierarchy": hierarchy,
                        }
                    }
                    semantic_chunks.append(sem_rows)
                    chunk_counter += 1

        # ── ИЗОБРАЖЕНИЯ: контекстное обогащение ──
        elif chunk_type in IMAGE_TYPES:
            flush_text_buffer()

            hierarchy = build_section_hierarchy(list(headers_stack))

            # Собрать контекст: текст ДО и ПОСЛЕ изображения
            context_parts = []

            # Текст ДО (ищем назад)
            for j in range(i - 1, max(i - 5, -1), -1):
                prev = raw_chunks[j]
                if prev.get("type") in TEXT_TYPES | HEADER_TYPES | {"caption"}:
                    context_parts.insert(0, prev.get("content", ""))
                    if len(" ".join(context_parts)) > IMAGE_CONTEXT_CHARS:
                        break
                elif prev.get("type") in TABLE_TYPES | IMAGE_TYPES:
                    break

            # Текст ПОСЛЕ (ищем вперед)
            for j in range(i + 1, min(i + 5, len(raw_chunks))):
                nxt = raw_chunks[j]
                if nxt.get("type") in {"caption"}:
                    context_parts.append(nxt.get("content", ""))
                elif nxt.get("type") in TEXT_TYPES | HEADER_TYPES:
                    context_parts.append(nxt.get("content", ""))
                    if len(" ".join(context_parts)) > IMAGE_CONTEXT_CHARS:
                        break
                elif nxt.get("type") in TABLE_TYPES | IMAGE_TYPES:
                    break

            search_text = " ".join(context_parts).strip()
            if not search_text:
                search_text = f"Изображение/чертеж из документа {source_file}"

            sem_img = {
                "chunk_id": f"sem_{chunk_counter:04d}",
                "chunk_type": "image_context",
                "search_text": search_text,
                "context_prefix": "",
                "payload": {
                    "type": "image_context",
                    "source_file": source_file,
                    "pages": [page] if page else [],
                    "bboxes": [bbox] if bbox else [],
                    "content": search_text,
                    "content_file": None,
                    "image_path": image_path,
                    "parent_id": None,
                    "section_hierarchy": hierarchy,
                }
            }
            semantic_chunks.append(sem_img)
            chunk_counter += 1

        else:
            # Неизвестный тип — добавить как текст, если есть content
            if content:
                text_buffer.append(content)
                if page is not None:
                    text_buffer_pages.append(page)
                if bbox is not None:
                    text_buffer_bboxes.append(bbox)

    # Сбросить оставшийся буфер
    flush_text_buffer()

    # ── ЭТАП 1.4: Contextual Retrieval ──
    if do_contextualize:
        print(f"  🧠 Контекстуализация {len(semantic_chunks)} чанков через LLM...")
        for idx, sc in enumerate(semantic_chunks):
            hierarchy = sc["payload"].get("section_hierarchy", [])
            prefix = llm_contextualize(
                llm_client,
                sc["search_text"],
                source_file,
                hierarchy
            )
            sc["context_prefix"] = prefix
            sc["search_text"] = f"{prefix}\n\n{sc['search_text']}"

            if (idx + 1) % 10 == 0:
                print(f"    Контекстуализировано: {idx + 1}/{len(semantic_chunks)}")
            time.sleep(0.15)  # Rate limiting
    else:
        # Простая встроенная контекстуализация: добавляем хлебные крошки в текст для поиска
        for sc in semantic_chunks:
            hierarchy = sc["payload"].get("section_hierarchy", [])
            if hierarchy:
                hierarchy_str = " > ".join(hierarchy)
                sc["search_text"] = f"Документ: {source_file}. Раздел: {hierarchy_str}.\nКонтент:\n{sc['search_text']}"
            else:
                sc["search_text"] = f"Документ: {source_file}.\nКонтент:\n{sc['search_text']}"

    return semantic_chunks


# ─── CLI точка входа ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Semantic chunker: преобразует сырые Docling-чанки в оптимизированные семантические чанки для RAG."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Путь к *_chunks.json файлу")
    parser.add_argument("--output", type=str, default=None,
                        help="Путь для выходного *_semantic.json (по умолчанию — рядом с input)")
    parser.add_argument("--contextualize", action="store_true",
                        help="Включить Contextual Retrieval через LLM (требует OPENROUTER_API_KEY)")
    parser.add_argument("--full-tables-dir", type=str, default=None,
                        help="Директория для сохранения полных таблиц (default: data/full_tables/)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Файл не найден: {input_path}")
        return

    # Определить выходной путь
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(
            input_path.name.replace("_chunks.json", "_semantic.json")
        )

    # Директория для полных таблиц
    if args.full_tables_dir:
        full_tables_dir = Path(args.full_tables_dir)
    else:
        full_tables_dir = input_path.parent.parent / "data" / "full_tables"

    # Загрузить сырые чанки
    with open(input_path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    source_file = input_path.stem.replace("_chunks", "")
    print(f"📥 Загружено {len(raw_chunks)} сырых чанков из {input_path.name}")
    print(f"   Источник: {source_file}")

    # Инициализировать LLM клиент (опционально)
    llm_client = None
    if args.contextualize:
        llm_client = get_openrouter_client()

    # ── Обработка ──
    semantic_chunks = process_chunks(
        raw_chunks,
        source_file=source_file,
        llm_client=llm_client,
        do_contextualize=args.contextualize,
        full_tables_dir=full_tables_dir
    )

    # ── Сохранение ──
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(semantic_chunks, f, indent=2, ensure_ascii=False)

    # ── Статистика ──
    types_count = {}
    for sc in semantic_chunks:
        t = sc["chunk_type"]
        types_count[t] = types_count.get(t, 0) + 1

    print(f"\n✅ Сформировано {len(semantic_chunks)} семантических чанков:")
    for t, cnt in sorted(types_count.items()):
        print(f"   • {t}: {cnt}")
    print(f"\n📄 Выход: {output_path}")


if __name__ == "__main__":
    main()
