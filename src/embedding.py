import os
import json
import uuid
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def create_collection(client, name, dim):
    existing = [c.name for c in client.get_collections().collections]

    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


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
    # --- Init ---
    client = QdrantClient("http://localhost:6333")
    model = SentenceTransformer(model_name)

    dim = model.get_sentence_embedding_dimension()
    create_collection(client, collection_name, dim)

    # --- Load data ---
    chunks = load_chunks(chunk_folder)

    print(f"Loaded {len(chunks)} chunks")

    # --- Batch processing ---
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i : i + batch_size]

        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True)

        points = []
        for chunk, emb in zip(batch, embeddings):
            deterministic_id = str(uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{chunk['source_file']}_{chunk['chunk_id']}"
            ))

            points.append(
                {
                    "id": deterministic_id,
                    "vector": emb.tolist(),
                    "payload": {
                        "text": chunk["text"],
                        "chunk_id": chunk["chunk_id"],
                        "source_file": chunk["source_file"],
                    },
                }
            )

        client.upsert(collection_name=collection_name, points=points)

    print(f"Finished inserting into {collection_name}")