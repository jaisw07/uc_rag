import os
import json
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.doc import partition_doc


def partitioner(filepath):
    # --- normalize path (important for Windows) ---
    filepath = filepath.replace("\\", "/")

    # --- extract filename + extension ---
    filename = os.path.basename(filepath)              # e.g. "file.pdf"
    name, ext = os.path.splitext(filename)             # ("file", ".pdf")
    ext = ext.lower()

    # --- ensure output directory exists ---
    os.makedirs("parsed", exist_ok=True)

    # --- PDF case ---
    if ext == ".pdf":
        elements = partition_pdf(
            filename=filepath,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=False,
            languages=['en']
        )

        element_dict = [el.to_dict() for el in elements]

        output_path = f"parsed/{name}_pdf.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(element_dict, f, indent=2, ensure_ascii=False)

    # --- DOC case ---
    elif ext == ".doc":
        elements = partition_doc(
            filename=filepath,
            languages=['en']
        )

        element_dict = [el.to_dict() for el in elements]

        output_path = f"parsed/{name}_doc.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(element_dict, f, indent=2, ensure_ascii=False)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # --- return for next pipeline step ---
    return element_dict, ext[1:]   # returns ("pdf" or "doc")

def clean_json_elements(file_path):
    """
    Cleans a JSON file by:
    1. Removing elements with null, empty, or whitespace-only 'text'.
    2. Normalizing whitespace in valid 'text' fields.
    Updates the file in-place.
    """

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # --- Load ---
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON in {file_path}")
            return

    if not isinstance(data, list):
        print(f"Error: Expected list of elements in {file_path}")
        return

    original_count = len(data)
    cleaned_data = []

    for item in data:
        # --- safety: ensure dict ---
        if not isinstance(item, dict):
            continue

        text = item.get('text')

        # --- filter invalid text ---
        if text is None or not isinstance(text, str) or text.strip() == "":
            continue

        # --- normalize whitespace ---
        cleaned_text = " ".join(text.split())

        # --- update safely (avoid mutating original reference unexpectedly) ---
        item = item.copy()
        item['text'] = cleaned_text

        cleaned_data.append(item)

    # --- Save ---
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    removed = original_count - len(cleaned_data)

    print(f"\nFile: {file_path}")
    print(f"Elements processed: {original_count}")
    print(f"Removed (empty/invalid text): {removed}")
    print(f"Remaining elements: {len(cleaned_data)}")

def normalize_element(el, source):
    meta = el.get("metadata", {})
    filename = meta.get("filename", "")

    # --- Extract course name and code ---
    coursename = None
    coursecode = None

    if filename and "_" in filename:
        parts = filename.split("_", 1)
        coursename = parts[0].strip()
        coursecode = parts[1].strip()
    else:
        coursename = filename
        coursecode = None

    return {
        "element_id": el.get("element_id"),
        "type": el.get("type"),
        "text": el.get("text"),

        # --- unified metadata ---
        "source": source,
        "filename": filename,
        "coursename": coursename,
        "coursecode": coursecode,

        # page logic
        "page": meta.get("page_number") if source == "pdf" else None,

        # --- structure ---
        "parent_id": meta.get("parent_id"),

        # --- tables ---
        "table_html": meta.get("text_as_html") if el.get("type") == "Table" else None,
    }

def normalize_json_elements(file_path):
    """
    Reads parsed JSON, normalizes elements, and writes to normalized/ folder.
    Removes _pdf/_doc suffix from filename.
    """

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # --- Load ---
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Detect source from filename ---
    base_name = os.path.basename(file_path)  # e.g. "file_pdf.json"

    if base_name.endswith("_pdf.json"):
        source = "pdf"
        clean_name = base_name.replace("_pdf.json", ".json")
    elif base_name.endswith("_doc.json"):
        source = "doc"
        clean_name = base_name.replace("_doc.json", ".json")
    else:
        raise ValueError("Filename must end with _pdf.json or _doc.json")

    # --- Normalize ---
    normalized_data = [
        normalize_element(el, source) for el in data
    ]

    # --- Ensure output directory exists ---
    os.makedirs("normalized", exist_ok=True)

    output_path = os.path.join("normalized", clean_name)

    # --- Save ---
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, indent=2, ensure_ascii=False)

    print(f"\nFile processed: {file_path}")
    print(f"Source detected: {source}")
    print(f"Elements normalized: {len(normalized_data)}")
    print(f"Saved to: {output_path}")