from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


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
    print(f"\n🔍 Query: {query}")
    print(f"📦 Collection: {collection_name}\n")

    for i, point in enumerate(results.points):
        # Handle both tuple and object formats
        if isinstance(point, tuple):
            _, score, payload = point
        else:
            score = point.score
            payload = point.payload

        print(f"Rank {i+1} | Score: {score:.4f}")
        print(f"Source: {payload.get('source_file')}")
        print(f"Chunk ID: {payload.get('chunk_id')}")
        print(f"Text: {payload.get('text')[:200]}...")
        print("-" * 80)

import chromadb


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

    # --- Print results ---
    print(f"\n🔍 Query: {query}")
    print(f"📦 Collection: {collection_name}\n")

    for i in range(len(results["ids"][0])):
        # Access the distance for this specific result
        distance = results["distances"][0][i]
        print(f"Rank {i+1} | Distance: {distance:.4f}")
        print(f"ID: {results['ids'][0][i]}")
        print(f"Source: {results['metadatas'][0][i].get('source_file')}")
        print(f"Chunk ID: {results['metadatas'][0][i].get('chunk_id')}")
        print(f"Text: {results['documents'][0][i][:200]}...")
        print("-" * 80)