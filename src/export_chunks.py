import json
import argparse
import base64
from pathlib import Path
from docling_core.types.doc.document import DoclingDocument

def process_document(json_path: Path):
    with open(json_path, 'r', encoding='utf-8') as f:
        doc_dict = json.load(f)
    
    doc = DoclingDocument.model_validate(doc_dict)
    
    chunks = []
    images_dir = json_path.parent / "images" / json_path.stem
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Base source file is derived from doc.name
    source_file = doc.name if hasattr(doc, "name") and doc.name else json_path.stem + ".pdf"
    
    for i, (item, level) in enumerate(doc.iterate_items()):
        # skip empty
        if not hasattr(item, "text") and not hasattr(item, "data") and not hasattr(item, "image"):
            continue
            
        chunk = {
            "id": f"chunk_{i:04d}",
            "type": item.label.value if hasattr(item, "label") else "unknown",
            "page": None,
            "bbox": None,
            "content": "",
            "image_path": None,
            "source_file": source_file
        }
        
        # Extract provenance (page, bbox)
        if hasattr(item, "prov") and item.prov:
            prov = item.prov[0]
            chunk["page"] = prov.page_no
            chunk["bbox"] = [round(prov.bbox.l, 2), round(prov.bbox.t, 2), round(prov.bbox.r, 2), round(prov.bbox.b, 2)]
        
        # Handle Images
        if hasattr(item, "image") and getattr(item, "image", None):
            uri = getattr(item.image, "uri", None)
            if uri:
                if str(uri).startswith("data:image"):
                    # Extract base64
                    header, encoded = str(uri).split(",", 1)
                    img_data = base64.b64decode(encoded)
                    page_no = chunk["page"] if chunk["page"] else 0
                    img_filename = f"page_{page_no}_img_{i:04d}.png"
                    img_path = images_dir / img_filename
                    img_path.write_bytes(img_data)
                    chunk["image_path"] = str(img_path)
                else:
                    # Treat the string URI as the path directly, since base64 is already stripped
                    chunk["image_path"] = str(uri)
        
        # Handle Content
        if item.label in ["text", "paragraph", "title", "list_item", "list_item_marker", "section_header", "page_header", "page_footer"]:
            chunk["content"] = getattr(item, "text", "")
        elif item.label == "table":
            if hasattr(item, "export_to_markdown"):
                chunk["content"] = item.export_to_markdown()
        elif item.label in ["picture", "figure", "image"]:
            chunk["content"] = "Изображение или чертеж"
        else:
            if hasattr(item, "text"):
                chunk["content"] = getattr(item, "text", "")
                
        chunks.append(chunk)
        
    out_path = json_path.with_name(f"{json_path.stem}_chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Generated {len(chunks)} chunks for {json_path.name}")
    print(f"📄 Output: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Docling JSON to RAG-ready chunks")
    parser.add_argument("json_file", type=str, help="Path to Docling JSON file")
    args = parser.parse_args()
    process_document(Path(args.json_file))
