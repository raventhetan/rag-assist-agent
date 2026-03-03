"""
Docling PDF Test Script — Docling 2.x API
Тестирует Docling на всех PDF из data/input/
GTX 1050 Max-Q (4GB VRAM)
"""
import json
import time
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    ThreadedPdfPipelineOptions,
    RapidOcrOptions,
    TableStructureOptions,
)
from docling_core.types.doc.document import ImageRefMode
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

# ─── Пути ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR    = Path(__file__).parent / "data" / "input"
OUTPUT_DIR   = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Настройка ускорителя (GTX 1050 Max-Q) ──────────────────────────────────
accel = AcceleratorOptions(device=AcceleratorDevice.CUDA)

# ─── Настройка модели OCR ──────────────────────────────────────────────────────
CYRILLIC_MODEL_DIR = Path(__file__).parent / "models" / "cyrillic"
rec_model_path = str(CYRILLIC_MODEL_DIR / "cyrillic_PP-OCRv5_rec_mobile_infer.onnx")
rec_keys_path = str(CYRILLIC_MODEL_DIR / "ppocrv5_cyrillic_dict.txt")

pipeline_options = ThreadedPdfPipelineOptions(
    accelerator_options=accel,
    do_ocr=True,
    do_table_structure=True,
    images_scale=2.5,
    generate_picture_images=True,
    ocr_options=RapidOcrOptions(
        force_full_page_ocr=True,
        rec_model_path=rec_model_path,
        rec_keys_path=rec_keys_path
    ),
    table_structure_options=TableStructureOptions(
        do_cell_matching=True,
        mode="accurate",
    ),
)

# ─── Инициализация конвертера ────────────────────────────────────────────────
print("🔧 Инициализация Docling (первый запуск скачивает модели ~1.5 ГБ)...")
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=StandardPdfPipeline,
            pipeline_options=pipeline_options,
        )
    }
)

# ─── Обработка тестовых файлов ──────────────────────────────────────────────────
pdf_files = [
    INPUT_DIR / "table11.pdf",
    INPUT_DIR / "chert+schema.pdf",
    INPUT_DIR / "shema+table.pdf",
    INPUT_DIR / "chert-1.pdf"
]

print(f"\n📂 Обработка файлов...\n")
summary = []

for pdf_path in pdf_files:
    if not pdf_path.exists():
        print(f"⚠️ Файл не найден: {pdf_path}")
        continue
    print(f"{'─'*60}")
    print(f"🚀 Обработка: {pdf_path.name}  ({pdf_path.stat().st_size // 1024} KB)")

    t_start = time.perf_counter()
    try:
        result = converter.convert(str(pdf_path))
        elapsed = time.perf_counter() - t_start

        images_dir = OUTPUT_DIR / "images" / pdf_path.stem
        images_dir.mkdir(parents=True, exist_ok=True)

        md_text  = result.document.export_to_markdown(
            image_mode=ImageRefMode.REFERENCED
        )
        doc_dict = result.document.export_to_dict()

        import base64
        def extract_and_save_images(obj, img_dir, doc_name):
            counter = [0]
            def walk(o):
                if isinstance(o, dict):
                    if "image" in o and isinstance(o["image"], dict) and "uri" in o["image"]:
                        uri = o["image"]["uri"]
                        if uri and str(uri).startswith("data:image"):
                            header, encoded = str(uri).split(",", 1)
                            img_data = base64.b64decode(encoded)
                            page_no = 0
                            if "prov" in o and isinstance(o["prov"], list) and len(o["prov"]) > 0:
                                page_no = o["prov"][0].get("page_no", 0)
                            img_filename = f"page_{page_no}_img_{counter[0]:04d}.png"
                            img_path = img_dir / img_filename
                            img_path.write_bytes(img_data)
                            o["image"]["uri"] = f"output/images/{doc_name}/{img_filename}"
                            counter[0] += 1
                    for k, v in o.items():
                        walk(v)
                elif isinstance(o, list):
                    for item in o:
                        walk(item)
            walk(obj)
            return obj

        doc_dict = extract_and_save_images(doc_dict, images_dir, pdf_path.stem)

        md_path = OUTPUT_DIR / f"{pdf_path.stem}.md"
        md_path.write_text(md_text, encoding="utf-8")

        json_path = OUTPUT_DIR / f"{pdf_path.stem}.json"
        json_path.write_text(
            json.dumps(doc_dict, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        print(f"   ✅ Готово за {elapsed:.1f}с  |  MD: {len(md_text)} символов")
        summary.append({
            "file": pdf_path.name, "status": "ok",
            "elapsed_sec": round(elapsed, 1), "md_chars": len(md_text),
        })

    except Exception as e:
        elapsed = time.perf_counter() - t_start
        print(f"   ❌ Ошибка: {e}")
        summary.append({
            "file": pdf_path.name, "status": "error",
            "error": str(e), "elapsed_sec": round(elapsed, 1),
        })

# ─── Итоговый отчёт ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
lines = [
    "# Docling — Итоги обработки\n",
    "| Файл | Статус | Время (с) | MD символов |",
    "|------|--------|-----------|-------------|",
]
for s in summary:
    st = "✅ OK" if s["status"] == "ok" else f"❌ {s.get('error','?')[:40]}"
    lines.append(f"| {s['file']} | {st} | {s['elapsed_sec']} | {s.get('md_chars','—')} |")
    print(f"   {s['file']:30s} {st}  {s['elapsed_sec']}с")

report = OUTPUT_DIR / "run_report.md"
report.write_text("\n".join(lines), encoding="utf-8")
print(f"\n📄 Отчёт: {report}\n📁 Результаты: {OUTPUT_DIR}")
