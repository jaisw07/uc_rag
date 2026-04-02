import os
import json
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.doc import partition_doc
from unstructured.partition.docx import partition_docx
from bs4 import BeautifulSoup
from markdownify import markdownify as md


def partitioner(filepath):
    # --- normalize path (important for Windows) ---
    filepath = filepath.replace("\\", "/")

    # --- extract filename + extension ---
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
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

    # --- DOC case ---
    elif ext == ".doc":
        elements = partition_doc(
            filename=filepath,
            languages=['en']
        )

        element_dict = [el.to_dict() for el in elements]
        output_path = f"parsed/{name}_doc.json"

    # --- DOCX case ---
    elif ext == ".docx":
        elements = partition_docx(
            filename=filepath,
            starting_page_number=1,
            infer_table_structure=True,
            strategy="hi_res"
        )

        element_dict = [el.to_dict() for el in elements]
        output_path = f"parsed/{name}_docx.json"

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # --- save output (common for all types) ---
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(element_dict, f, indent=2, ensure_ascii=False)

    print(f"Saved parsed output to: {output_path}")



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

        # --- safe copy ---
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

def convert_tables_to_markdown(file_path):
    """
    Converts Table elements:
    - metadata.text_as_html → markdown_text
    - changes type to 'markdown_text'
    Updates file in-place.
    """

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # --- Load ---
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON in {file_path}")
            return

    if not isinstance(data, list):
        print(f"Error: Expected list of elements in {file_path}")
        return

    converted_count = 0

    for item in data:
        if not isinstance(item, dict):
            continue

        if item.get("type") == "Table":
            metadata = item.get("metadata", {})
            html = metadata.get("text_as_html")

            if html and isinstance(html, str):
                try:
                    # --- clean/parse HTML ---
                    soup = BeautifulSoup(html, "html.parser")
                    clean_html = str(soup)

                    # --- convert to markdown ---
                    markdown_text = md(clean_html, heading_style="ATX")

                    # --- update element ---
                    item["markdown_text"] = markdown_text.strip()
                    item["type"] = "markdown_text"

                    converted_count += 1

                except Exception as e:
                    print(f"Warning: Failed to convert table {item.get('element_id')} - {e}")
                    continue

    # --- Save ---
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nFile: {file_path}")
    print(f"Tables converted to markdown: {converted_count}")

import os
import json


def normalize_element(el):
    meta = el.get("metadata", {})
    filename = meta.get("filename", "")

    # --- Extract course name and code ---
    coursename = None
    coursecode = None

    if filename and "_" in filename:
        parts = filename.split("_", 1)
        coursename = parts[0].strip()
        coursecode = parts[1].replace(".pdf", "").replace(".docx", "").replace(".doc", "").strip()
    else:
        coursename = filename
        coursecode = None

    # --- prioritize markdown_text over text ---
    text = el.get("markdown_text") if el.get("markdown_text") else el.get("text")

    return {
        "type": el.get("type"),
        "text": text,   
        "coursename": coursename,
        "coursecode": coursecode,
    }


def normalize_json_elements(file_path):
    """
    Reads parsed JSON, normalizes elements to strict 4-key schema,
    and writes to normalized/ folder.
    """

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # --- Load ---
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"Error: Expected list in {file_path}")
        return

    # --- Normalize ---
    normalized_data = []
    skipped = 0

    for el in data:
        if not isinstance(el, dict):
            skipped += 1
            continue

        norm = normalize_element(el)

        # --- safety: skip if no usable text ---
        if not norm["text"] or not isinstance(norm["text"], str):
            skipped += 1
            continue

        normalized_data.append(norm)

    # --- Ensure output directory exists ---
    os.makedirs("normalized", exist_ok=True)

    # --- clean filename (remove suffixes like _pdf/_doc/_docx) ---
    base_name = os.path.basename(file_path)
    clean_name = base_name.replace("_pdf.json", ".json") \
                          .replace("_doc.json", ".json") \
                          .replace("_docx.json", ".json")

    output_path = os.path.join("normalized", clean_name)

    # --- Save ---
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, indent=2, ensure_ascii=False)

    print(f"\nFile processed: {file_path}")
    print(f"Elements normalized: {len(normalized_data)}")
    print(f"Skipped (invalid): {skipped}")
    print(f"Saved to: {output_path}")