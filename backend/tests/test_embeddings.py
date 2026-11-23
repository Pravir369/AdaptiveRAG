import numpy as np

from app.rag.embeddings.base import HashEmbeddingProvider, _hash_embed


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_hash_embedding_is_deterministic_and_unit_norm():
    a = _hash_embed("Serialize obj to a JSON formatted str.")
    b = _hash_embed("Serialize obj to a JSON formatted str.")
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-5


def test_hash_embedding_ignores_token_position():
    # Regression: the bucket used to depend on the token's position in the text,
    # so the same word in a short query and a long document rarely collided.
    short = _hash_embed("json")
    late = _hash_embed("alpha beta gamma delta epsilon json")
    other = _hash_embed("alpha beta gamma delta epsilon yaml")
    assert cos(short, late) > cos(short, other) + 0.2


def test_hash_embedding_ranks_relevant_text_higher():
    p = HashEmbeddingProvider()
    q = p.embed_query("how do I delete a directory tree")
    rel = p.embed_texts(["Recursively delete a directory tree."])[0]
    irr = p.embed_texts(["Return the median of numeric data."])[0]
    assert cos(q, rel) > cos(q, irr)


def test_empty_text_gives_zero_vector_without_nan():
    v = _hash_embed("")
    assert not np.isnan(v).any()
    assert np.linalg.norm(v) == 0.0
