
import json
import hashlib
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# File paths
INPUT_FILE = Path("chunks.json")
OUTPUT_FILE = Path("embeddings.npy")
INFO_FILE = Path("embedding_info.json")

# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def main():
    # 1. Load chunks
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)
    chunk_sha256 = hashlib.sha256(INPUT_FILE.read_bytes()).hexdigest()

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks.json must contain a non-empty list.")

    # 2. Validate each chunk
    chunk_ids = []
    texts = []

    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError("Every chunk must be a JSON object.")

        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text")

        if not isinstance(chunk_id, int) or isinstance(chunk_id, bool):
            raise ValueError("Every chunk must have an integer chunk_id.")
        if chunk_id != position:
            raise ValueError(f"chunk_id {chunk_id} does not match chunks.json position {position}.")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk {chunk_id} has empty or invalid text.")

        chunk_ids.append(chunk_id)
        texts.append(text.strip())

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    # 3. Load the embedding model
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 4. Convert text into vectors
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # 5. Validate generated vectors
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")

    if embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding count does not match chunk count.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numerical values.")

    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4):
        raise ValueError("Embedding normalization check failed.")

    # 6. Save vectors and model information
    np.save(OUTPUT_FILE, embeddings)

    info = {
        "model_name": MODEL_NAME,
        "number_of_chunks": len(chunks),
        "embedding_dimensions": int(embeddings.shape[1]),
        "normalized": True,
        "chunk_sha256": chunk_sha256,
    }

    with INFO_FILE.open("w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)

    print("\nEmbedding generation successful.")
    print(f"Chunks processed: {len(chunks)}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Saved vectors to: {OUTPUT_FILE.resolve()}")
    print(f"Saved model information to: {INFO_FILE.resolve()}")


if __name__ == "__main__":
    main()
