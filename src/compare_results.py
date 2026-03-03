"""
compare_results.py
Анализирует MD-файлы, созданные Docling, и строит сравнительный отчёт.
"""
import re
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"

def count_tables(md_text: str) -> int:
    """Считает таблицы Markdown (заголовки с |---|)."""
    return len(re.findall(r"^\|[-| ]+\|$", md_text, re.MULTILINE))

def extract_numbers(md_text: str) -> list[str]:
    """Извлекает все числа из текста (размеры, артикулы)."""
    return re.findall(r"\b\d{2,}\b", md_text)

def count_rows(md_text: str) -> int:
    """Считает строки таблицы (строки с |...|)."""
    return len(re.findall(r"^\|.+\|$", md_text, re.MULTILINE))

# ─── Сбор данных ─────────────────────────────────────────────────────────────
md_files = sorted(OUTPUT_DIR.glob("*.md"))
results = []

for md_path in md_files:
    if md_path.name in ("run_report.md", "comparison_report.md"):
        continue
    text = md_path.read_text(encoding="utf-8")
    tables    = count_tables(text)
    rows      = count_rows(text)
    numbers   = extract_numbers(text)
    results.append({
        "file":    md_path.stem,
        "tables":  tables,
        "rows":    rows,
        "numbers": len(numbers),
        "chars":   len(text),
        "sample_numbers": numbers[:10],
    })

# ─── Отчёт ───────────────────────────────────────────────────────────────────
lines = [
    "# Docling — Сравнительный анализ результатов\n",
    "| PDF | Таблиц | Строк | Чисел | Символов |",
    "|-----|--------|-------|-------|----------|",
]
for r in results:
    lines.append(
        f"| {r['file']} | {r['tables']} | {r['rows']} | {r['numbers']} | {r['chars']} |"
    )

lines += [
    "\n## Примеры извлечённых чисел\n",
]
for r in results:
    lines.append(f"**{r['file']}**: {', '.join(r['sample_numbers']) or '—'}")

report = "\n".join(lines)
report_path = OUTPUT_DIR / "comparison_report.md"
report_path.write_text(report, encoding="utf-8")

print(f"✅ Отчёт сохранён: {report_path}")
print()
print(report)
