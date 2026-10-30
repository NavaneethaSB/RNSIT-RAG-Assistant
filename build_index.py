
import json
import hashlib
from pathlib import Path

import faiss
import numpy as np

CHUNKS_FILE = Path("chunks.json")
EMBEDDINGS_FILE = Path("embeddings.npy")
INDEX_FILE = Path("rnsit.index")
INFO_FILE = Path("embedding_info.json")
INDEX_INFO_FILE = Path("index_info.json")


def main():
    # 1. Check that the required files exist
    for file_path in (CHUNKS_FILE, EMBEDDINGS_FILE, INFO_FILE):
        if not file_path.is_file():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    # 2. Load the source chunks and embeddings
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    with INFO_FILE.open("r", encoding="utf-8") as file:
        info = json.load(file)
    chunk_sha256 = hashlib.sha256(CHUNKS_FILE.read_bytes()).hexdigest()

    embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=False)

    # 3. Validate the data
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks.json must contain a non-empty list.")

    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding count does not match the chunk count.")

    if embeddings.dtype != np.float32:
        raise ValueError("Embeddings must have dtype float32.")

    if embeddings.shape[1] != info.get("embedding_dimensions"):
        raise ValueError("Embedding dimensions do not match embedding_info.json.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numerical values.")

    if info.get("number_of_chunks") != len(chunks):
        raise ValueError("Chunk count does not match embedding_info.json.")

    if info.get("embedding_dimensions") != embeddings.shape[1]:
        raise ValueError("Embedding dimensions do not match embedding_info.json.")

    if info.get("normalized") is not True:
        raise ValueError("Expected normalized embeddings.")

    if info.get("chunk_sha256") != chunk_sha256:
        raise ValueError("Embeddings were generated from a different chunks.json. Re-run embed_chunks.py.")

    if not np.allclose(
        np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4
    ):
        raise ValueError("Some embeddings are not normalized.")

    # 4. Build the FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # Inner product between normalized vectors gives cosine similarity
    index.add(np.ascontiguousarray(embeddings))

    # 5. Verify the index
    if index.ntotal != len(chunks):
        raise ValueError("FAISS index count does not match chunk count.")

    # 6. Save the index
    faiss.write_index(index, str(INDEX_FILE))
    with INDEX_INFO_FILE.open("w", encoding="utf-8") as file:
        json.dump({
            "number_of_chunks": len(chunks),
            "embedding_dimensions": int(dimension),
            "chunk_sha256": chunk_sha256,
            "index_type": type(index).__name__,
            "metric": "inner_product",
            "normalized": True,
        }, file, indent=2)

    print("FAISS index built successfully.")
    print(f"Chunks: {len(chunks)}")
    print(f"Vector dimensions: {dimension}")
    print(f"Vectors indexed: {index.ntotal}")
    print(f"Index type: {type(index).__name__}")
    print(f"Saved to: {INDEX_FILE.resolve()}")


if __name__ == "__main__":
    main()
