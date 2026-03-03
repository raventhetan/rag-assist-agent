import json
import base64
from pathlib import Path
from docling_core.types.doc.document import DoclingDocument

def test_uri_replacement():
    json_path = Path('output/table11.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        doc_dict = json.load(f)
        
    for pic in doc_dict.get("pictures", []):
        if "image" in pic and "uri" in pic["image"]:
            if pic["image"]["uri"].startswith("data:image"):
                # Replace with file path
                pic["image"]["uri"] = "output/images/table11/test.png"
                
    try:
        # Test if DoclingDocument can still parse it
        doc = DoclingDocument.model_validate(doc_dict)
        print("Success! Pydantic accepted the modified dictionary with file paths.")
    except Exception as e:
        print(f"Error validating: {e}")

if __name__ == "__main__":
    test_uri_replacement()
