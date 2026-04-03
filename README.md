To run a persistent volume docker instance for qdrant, run the following command in the conda env:
docker run -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
