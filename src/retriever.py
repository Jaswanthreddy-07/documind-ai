import numpy as np


def build_embeddings(model, texts: list[str]) -> np.ndarray:
    """Encode text chunks and normalize vectors for cosine-similarity retrieval."""
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


def retrieve(model, document_vectors: np.ndarray, chunks: list[dict], question: str, top_k: int = 4) -> list[dict]:
    """Return top-k chunks ranked by cosine similarity to the question."""
    if not question.strip() or len(chunks) == 0:
        return []
    query_vector = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype=np.float32)[0]
    scores = document_vectors @ query_vector
    top_indices = np.argsort(scores)[::-1][:min(top_k, len(chunks))]
    return [
        {**chunks[int(i)], "score": float(scores[int(i)])}
        for i in top_indices
    ]
