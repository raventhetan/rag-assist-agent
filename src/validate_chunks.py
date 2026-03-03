"""
validate_chunks.py — Автоматическая проверка RAG-ready chunks по матрице тестов.
Запускать из директории: src/
"""
import json
import os
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"

# ─── Настройки ────────────────────────────────────────────────
DOCS_TO_TEST = ["table11", "shema+table", "chert+schema", "chert-1"]
VALID_TYPES = {"text", "table", "picture", "section_header",
               "caption", "list_item", "footnote", "page_header",
               "page_footer", "form", "title", "code"}

# ─── Функции проверки ─────────────────────────────────────────

def check_structure(chunks: list, doc: str) -> list:
    """Уровень 1: Проверка bbox, page, type у каждого чанка."""
    errors = []
    for c in chunks:
        cid = c.get("id", "?")
        if c.get("page") is None:
            errors.append(f"[{doc}] {cid}: page=null")
        if c.get("bbox") is None:
            errors.append(f"[{doc}] {cid}: bbox=null")
        if c.get("type") not in VALID_TYPES:
            errors.append(f"[{doc}] {cid}: unknown type={c.get('type')}")
    return errors


def check_images(chunks: list, doc: str) -> list:
    """Уровень 4: Проверка того что PNG-файлы реально существуют."""
    errors = []
    images = [c for c in chunks if c.get("type") == "picture"]
    for c in images:
        img_path = c.get("image_path")
        if img_path and not os.path.exists(img_path):
            errors.append(f"[{doc}] {c['id']}: image_path не найден: {img_path}")
        elif not img_path:
            errors.append(f"[{doc}] {c['id']}: type=picture но image_path отсутствует")
    return errors


def check_table_content(chunks: list, doc: str) -> list:
    """Уровень 2: Таблицы должны содержать непустой content."""
    errors = []
    tables = [c for c in chunks if c.get("type") == "table"]
    for c in tables:
        if not c.get("content", "").strip():
            errors.append(f"[{doc}] {c['id']}: table chunk с пустым content")
    return errors


def check_cyrillic(chunks: list, doc: str) -> list:
    """Уровень 3: Поиск кракозябров (простая эвристика)."""
    warnings = []
    garble_patterns = ["Pa3", "Bbi", "yKa", "3aH", "npw", "HbI"]
    for c in chunks:
        content = c.get("content", "")
        for pat in garble_patterns:
            if pat in content:
                warnings.append(f"[{doc}] {c['id']}: возможный кракозябр '{pat}' в: {content[:80]!r}")
                break
    return warnings


# ─── Точка входа ─────────────────────────────────────────────

def run_validation(docs: list[str]) -> None:
    total_errors = []
    total_warnings = []
    total_chunks = 0

    print("=" * 64)
    print("  Docling Pipeline — Автоматическая валидация чанков")
    print("=" * 64)

    for doc in docs:
        chunks_file = OUTPUT_DIR / f"{doc}_chunks.json"
        if not chunks_file.exists():
            print(f"\n⚠️  {doc}: файл чанков не найден ({chunks_file})")
            continue

        with open(chunks_file, encoding="utf-8") as f:
            chunks = json.load(f)

        total_chunks += len(chunks)
        n_img = sum(1 for c in chunks if c.get("type") == "picture")
        n_tbl = sum(1 for c in chunks if c.get("type") == "table")
        n_txt = sum(1 for c in chunks if c.get("type") in {"text", "section_header"})

        errs  = check_structure(chunks, doc)
        errs += check_images(chunks, doc)
        errs += check_table_content(chunks, doc)
        warns = check_cyrillic(chunks, doc)

        status = "✅" if not errs else "❌"
        print(f"\n{status} {doc}  ({len(chunks)} чанков: {n_txt} текст, {n_tbl} таблиц, {n_img} изображений)")
        if errs:
            for e in errs:
                print(f"    ❌  {e}")
        if warns:
            for w in warns:
                print(f"    ⚠️  {w}")
        if not errs and not warns:
            print("    ✓  Всё в порядке")

        total_errors.extend(errs)
        total_warnings.extend(warns)

    print("\n" + "=" * 64)
    print(f"  Итог: {total_chunks} чанков, {len(total_errors)} ошибок, {len(total_warnings)} предупреждений")
    if not total_errors:
        print("  🚀 Пайплайн готов к масштабированию на RAG!")
    else:
        print("  🚫 Есть ошибки — проверьте до масштабирования.")
    print("=" * 64)

    sys.exit(1 if total_errors else 0)


if __name__ == "__main__":
    docs = sys.argv[1:] if len(sys.argv) > 1 else DOCS_TO_TEST
    run_validation(docs)
