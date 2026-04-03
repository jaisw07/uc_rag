import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


def load_chunks(folder):
    all_chunks = []

    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                all_chunks.extend(data)

    return all_chunks


def embed_and_store(
    chunk_folder: str,
    model_name: str,
    collection_name: str,
    batch_size: int = 32,
):
    # --- Init (FORCED persistent mode) ---
    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(name=collection_name)

    model = SentenceTransformer(model_name)

    # --- Load data ---
    chunks = load_chunks(chunk_folder)
    print(f"Loaded {len(chunks)} chunks")

    # --- Batch processing ---
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i : i + batch_size]

        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True)

        ids = []
        documents = []
        metadatas = []

        for chunk in batch:
            deterministic_id = f"{chunk['source_file']}_{chunk['chunk_id']}"

            ids.append(deterministic_id)
            documents.append(chunk["text"])
            metadatas.append({
                "chunk_id": chunk["chunk_id"],
                "source_file": chunk["source_file"],
            })

        # ✅ use UPSERT (critical)
        collection.upsert(
            ids=ids,
            embeddings=[emb.tolist() for emb in embeddings],
            documents=documents,
            metadatas=metadatas,
        )

    print(f"Finished inserting into {collection_name}")