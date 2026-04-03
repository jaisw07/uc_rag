import os
import json
import random
import ollama
from typing import List


def generate_eval_dataset(
    normalized_folder: str,
    num_files: int = 5,
    samples_per_file: int = 5,
    model_name: str = "phi3:mini",
    save_path: str = "evaluation_dataset.json",
):
    """
    Generate evaluation dataset from normalized JSON files.

    Args:
        normalized_folder: Path to normalized JSON files
        num_files: Number of files to sample
        samples_per_file: Number of QA samples per file
        model_name: Ollama model for QA generation
        save_path: Output JSON file

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
        print(f"[{idx+1}/{len(sampled_files)}] Processing file: {file}...")
        file_path = os.path.join(normalized_folder, file)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            continue

        # Track questions per file to ensure we get samples_per_file even if 1 object
        file_questions = 0
        used_texts = []
        
        # Limit attempts per file to prevent infinite loops if LLM fails repeatedly
        max_attempts = samples_per_file * 6
        attempts = 0

        while file_questions < samples_per_file and attempts < max_attempts:
            attempts += 1
            # Randomly select an object
            random.shuffle(data)
            for el in data:
                text = el.get("text", "").strip()
                coursename = el.get("coursename", "").strip()
                coursecode = el.get("coursecode", "").strip()

            if not text:
                continue

            source_file = f"{coursename}_{coursecode}".strip("_")

            # Filter for this specific text block to find existing questions
            existing_q = [d["question"] for d in dataset if d["reference_text"] == text[:300]]
            previous_questions_context = ""
            if existing_q:
                previous_questions_context = f"\nPreviously generated questions for this text: {', '.join(existing_q[-3:])}\nGenerate a NEW, TOTALLY DIFFERENT question."

            # --- LLM Prompt ---
            prompt = f"""
                You are generating evaluation data for a retrieval-augmented QA system.

                Given the following text, generate:
                1. ONE clear, specific question answerable ONLY from this text{previous_questions_context}
                2. A concise answer strictly based on the text

                Rules:
                - Do NOT use external knowledge
                - Question should not be vague
                - Answer must be directly present in the text
                - Avoid yes/no questions

                Text:
                {text}

                Output format:
                Question: ...
                Answer: ...
                """

            try:
                response = ollama.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.7}, # High temp for variety on repeats
                )

                output = response["message"]["content"]
                question = ""
                answer = ""

                for line in output.split("\n"):
                    trimmed = line.strip()
                    if trimmed.lower().startswith("question:"):
                        question = trimmed.split(":", 1)[1].strip()
                    elif trimmed.lower().startswith("answer:"):
                        answer = trimmed.split(":", 1)[1].strip()

                if not question or not answer:
                    print("⚠️ Skipped malformed output:", output[:200])
                    continue

                reference_text = text

                dataset.append({
                    "question": question,
                    "ground_truth_answer": answer,
                    "reference_text": reference_text,
                    "source_file": source_file
                })
                
                file_questions += 1
                
                # Auto-save after every success
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)

            except Exception as e:
                print(f"Error calling Ollama: {e}")
                if "connection" in str(e).lower():
                    print("Service unavailable. Saving current progress.")
                    return dataset
                continue

    print(f"\n✅ Total {len(dataset)} evaluation samples generated and saved.")
    return dataset