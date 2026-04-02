import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def chunking_minilm_l6_v2(input_folder: str, chunk_size: int, overlap: int, word_batch_size: int = 50):
    """
    Token-based chunking with safe word-batch tokenization

    Args:
        chunk_size (int): max tokens per chunk (<=256)
        overlap (int): token overlap
        word_batch_size (int): number of words to tokenize at a time
    """

    assert chunk_size > overlap, "chunk_size must be greater than overlap"

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = model.tokenizer

    output_dir = f"chunks/minilm_l6_v2_{chunk_size}_{overlap}"
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    for file in tqdm(files, desc="Chunking files"):
        input_path = os.path.join(input_folder, file)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = []
        chunk_id = 0
        current_chunk_tokens = []

        for el in data:
            text = el.get("text", "").strip()
            if not text:
                continue

            words = text.split()

            # 🔹 Process text in word batches
            for i in range(0, len(words), word_batch_size):
                word_batch = words[i:i + word_batch_size]
                batch_text = " ".join(word_batch)

                token_ids = tokenizer.encode(batch_text, add_special_tokens=False)

                # 🔹 Add tokens safely
                for token in token_ids:
                    current_chunk_tokens.append(token)

                    if len(current_chunk_tokens) == chunk_size:
                        chunk_text = tokenizer.decode(current_chunk_tokens)

                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": chunk_text,
                            "source_file": file
                        })

                        chunk_id += 1

                        # overlap
                        current_chunk_tokens = current_chunk_tokens[-overlap:]

        # Final leftover
        if current_chunk_tokens:
            chunk_text = tokenizer.decode(current_chunk_tokens)

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source_file": file
            })

        # Save
        output_path = os.path.join(
            output_dir,
            file.replace(".json", "_chunks.json")
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Chunking complete. Saved to: {output_dir}")


def chunking_bge_base_v1_5(input_folder: str, chunk_size: int, overlap: int, word_batch_size: int = 100):
    """
    Token-based chunking using BGE tokenizer with safe word-batch processing

    Args:
        chunk_size (int): max tokens per chunk (<=512 recommended <=448)
        overlap (int): token overlap
        word_batch_size (int): number of words to tokenize at a time
    """

    assert chunk_size > overlap, "chunk_size must be greater than overlap"
    assert chunk_size <= 512, "BGE max token limit is 512"

    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    tokenizer = model.tokenizer

    output_dir = f"chunks/bge_base_v1_5_{chunk_size}_{overlap}"
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    for file in tqdm(files, desc="Chunking files"):
        input_path = os.path.join(input_folder, file)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = []
        chunk_id = 0
        current_chunk_tokens = []

        for el in data:
            text = el.get("text", "").strip()
            if not text:
                continue

            words = text.split()

            # 🔹 Process in word batches
            for i in range(0, len(words), word_batch_size):
                word_batch = words[i:i + word_batch_size]
                batch_text = " ".join(word_batch)

                token_ids = tokenizer.encode(batch_text, add_special_tokens=False)

                for token in token_ids:
                    current_chunk_tokens.append(token)

                    if len(current_chunk_tokens) == chunk_size:
                        chunk_text = tokenizer.decode(current_chunk_tokens)

                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": chunk_text,
                            "source_file": file
                        })

                        chunk_id += 1

                        # overlap
                        current_chunk_tokens = current_chunk_tokens[-overlap:]

        # Final leftover chunk
        if current_chunk_tokens:
            chunk_text = tokenizer.decode(current_chunk_tokens)

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source_file": file
            })

        # Save output
        output_path = os.path.join(
            output_dir,
            file.replace(".json", "_chunks.json")
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Chunking complete. Saved to: {output_dir}")