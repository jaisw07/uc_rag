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


if __name__ == "__main__":
    query = "What is the marking scheme of the PRJ III course?"

    print("\n=== MiniLM 256_26 Results ===")
    query_qdrant(
        query=query,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="minilm_l6_v2_256_26",
    )

    print("\n=== BGE 256_26 Results ===")
    query_qdrant(
        query=query,
        model_name="BAAI/bge-base-en-v1.5",
        collection_name="bge_base_v1_5_256_26",
    )   
    
    print("\n=== BGE 512_52 Results ===")
    query_qdrant(
        query=query,
        model_name="BAAI/bge-base-en-v1.5",
        collection_name="bge_base_v1_5_512_52",
    )   