from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

import chromadb

def query_qdrant(
    query: str,
    model_name: str,
    collection_name: str,
    top_k: int = 5,
):
    # --- Init ---
    client = QdrantClient("http://localhost:6333")
    model = SentenceTransformer(model_name)

    # --- Embed query ---
    query_vector = model.encode(query, normalize_embeddings=True)

    # --- Search ---
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    )

    # --- Print results ---
    chunks = []

    for point in results.points:
        if isinstance(point, tuple):
            _, score, payload = point
        else:
            payload = point.payload

        text = payload.get("text", "")
        if text:
            chunks.append(text)

    return chunks


def query_chromadb(
    query: str,
    model_name: str,
    collection_name: str,
    top_k: int = 5,
):
    # --- Init ---
    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_collection(name=collection_name)
    model = SentenceTransformer(model_name)

    # --- Embed query ---
    query_vector = model.encode(query, normalize_embeddings=True)

    # --- Search ---
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k,
    )

    chunks = results["documents"][0]

    return chunks