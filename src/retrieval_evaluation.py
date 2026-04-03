import os
import json
import random
import ollama
import re
from typing import List


def generate_eval_dataset(
    normalized_folder: str,
    num_files: int = 5,
    samples_per_file: int = 5,
    model_name: str = "mistral:7b-instruct",
    save_path: str = "evaluation_dataset.json",
):
    """
    Generate evaluation dataset using FULL DOCUMENT context.

    Each document → multiple QA pairs with supporting evidence.

    Output format:
    [
        {
            "question": "...",
            "ground_truth_answer": "...",
            "reference_text": "...",
            "source_file": "coursename_coursecode"
        }
    ]
    """

    files = [f for f in os.listdir(normalized_folder) if f.endswith(".json")]

    if not files:
        raise ValueError("No normalized JSON files found")

    sampled_files = random.sample(files, min(num_files, len(files)))

    dataset = []

    for idx, file in enumerate(sampled_files):
        print(f"\n[{idx+1}/{len(sampled_files)}] Processing file: {file}...")

        file_path = os.path.join(normalized_folder, file)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            print("⚠️ Skipping empty or invalid file")
            continue

        # --- Build FULL DOCUMENT TEXT ---
        full_text_parts = []
        coursename = ""
        coursecode = ""

        for el in data:
            text = el.get("text", "").strip()
            if text:
                full_text_parts.append(text)

            # capture metadata once
            if not coursename:
                coursename = el.get("coursename", "").strip()
            if not coursecode:
                coursecode = el.get("coursecode", "").strip()

        if not full_text_parts:
            print("⚠️ No usable text found")
            continue

        full_text = "\n".join(full_text_parts)

        # --- truncate for LLM safety ---
        MAX_CHARS = 4000  # adjust if using larger context models
        truncated_text = full_text[:MAX_CHARS]

        source_file = f"{coursename}_{coursecode}".strip("_")

        # --- Prompt ---
        prompt = f"""
You are generating evaluation data for a retrieval-augmented QA system.

Given the following document, generate {samples_per_file} high-quality question-answer pairs.

Requirements:
- Questions must be specific and diverse
- Each question must be answerable ONLY from the document
- Answers must be concise and directly supported by the text
- Avoid repetition
- Avoid yes/no questions

IMPORTANT:
- For EACH QA pair, include a SHORT exact supporting text span from the document

Output STRICTLY in JSON format:
[
  {{
    "question": "...",
    "answer": "...",
    "support": "exact supporting text"
  }}
]

Document:
{truncated_text}
"""

        try:
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7},
            )

            output = response["message"]["content"]

            # --- Extract JSON safely ---
            try:
                json_match = re.search(r"\[.*\]", output, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON array found")

                qa_list = json.loads(json_match.group(0))

            except Exception as e:
                print("⚠️ JSON parsing failed")
                print("Raw output:", output[:300])
                continue

            # --- Validate + store ---
            valid_count = 0

            for qa in qa_list:
                question = qa.get("question", "").strip()
                answer = qa.get("answer", "").strip()
                support = qa.get("support", "").strip()

                if not question or not answer or not support:
                    continue

                dataset.append({
                    "question": question,
                    "ground_truth_answer": answer,
                    "reference_text": support,
                    "source_file": source_file
                })

                valid_count += 1

            print(f"✅ Added {valid_count} QA pairs")

            # --- Save progress ---
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"❌ Error calling Ollama: {e}")
            if "connection" in str(e).lower():
                print("⚠️ Service unavailable. Saving progress.")
                return dataset
            continue

    print(f"\n🎉 Total {len(dataset)} evaluation samples generated and saved.")
    return dataset