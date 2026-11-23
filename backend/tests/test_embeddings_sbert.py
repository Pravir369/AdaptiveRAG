"""Tests for the sentence-transformers embedding path.

Why a stubbed SentenceTransformer rather than only the real one: CI does not install
sentence-transformers (it pulls in torch), so a test guarded purely by
`importorskip` would be skipped exactly where regressions slip through. The stub
mirrors the parts of the sentence-transformers 5.x contract this provider depends
on -- notably that `encode()` rejects keyword arguments the model does not use --
so the bug it is written against is caught with no heavy dependency installed.

The real model is exercised too, but only when it happens to be available.
"""
import sys

import numpy as np
import pytest

from app.core import EMBED_DIM

RAW_DIM = 384  # all-MiniLM-L6-v2


class _StubSentenceTransformer:
    """Stands in for sentence_transformers.SentenceTransformer 5.x.

    encode() lists the keyword arguments the real one accepts and raises on any
    other, which is what sentence-transformers 5.x does via get_model_kwargs().
    """

    last_encode_kwargs = None

    def __init__(self, model_name_or_path, *args, **kwargs):
        self.model_name_or_path = model_name_or_path

    def get_embedding_dimension(self):
        return RAW_DIM

    def encode(
        self,
        sentences,
        prompt_name=None,
        prompt=None,
        batch_size=32,
        show_progress_bar=None,
        output_value="sentence_embedding",
        precision="float32",
        convert_to_numpy=True,
        convert_to_tensor=False,
        device=None,
        normalize_embeddings=False,
        **kwargs,
    ):
        if kwargs:
            raise ValueError(
                "SentenceTransformer.encode() has been called with additional keyword "
                f"arguments that this model does not use: {sorted(kwargs)}. As per "
                "SentenceTransformer.get_model_kwargs(), this model does not accept "
                "any additional keyword arguments."
            )
        type(self).last_encode_kwargs = {
            "convert_to_numpy": convert_to_numpy,
            "normalize_embeddings": normalize_embeddings,
        }
        # Deterministic, non-zero, clearly not unit-norm in EMBED_DIM once padded.
        out = np.zeros((len(sentences), RAW_DIM), dtype=np.float32)
        for i, text in enumerate(sentences):
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            out[i] = rng.standard_normal(RAW_DIM).astype(np.float32) * 3.0
        return out


class _OldNameSentenceTransformer(_StubSentenceTransformer):
    """sentence-transformers < 5.x: only the pre-rename dimension accessor."""

    get_embedding_dimension = None  # attribute lookup must not find a callable

    def get_sentence_embedding_dimension(self):
        return RAW_DIM


@pytest.fixture
def stub_sentence_transformers(monkeypatch):
    """Install a fake `sentence_transformers` module for the duration of a test."""

    def _install(cls=_StubSentenceTransformer):
        import types

        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = cls
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        cls.last_encode_kwargs = None
        return cls

    return _install


def _fresh_provider():
    from app.rag.embeddings.base import SBERTEmbeddingProvider

    return SBERTEmbeddingProvider()


def test_sbert_encode_is_called_with_supported_kwargs_only(stub_sentence_transformers):
    """Regression: embed_texts passed dtype= to encode(), which 5.x rejects.

    The provider raised on the first ingest, so EMBEDDING_PROVIDER=sbert was
    unusable. Nothing caught it because the suite only covered hash embeddings.
    """
    stub_sentence_transformers()
    provider = _fresh_provider()

    out = provider.embed_texts(["Recursively delete a directory tree.", "Dump obj to JSON."])

    assert out.shape == (2, EMBED_DIM)
    assert out.dtype == np.float32


def test_sbert_embed_query_goes_through_the_same_path(stub_sentence_transformers):
    stub_sentence_transformers()
    provider = _fresh_provider()

    v = provider.embed_query("how do I delete a directory tree")

    assert v.shape == (EMBED_DIM,)
    assert v.dtype == np.float32


def test_sbert_pads_to_embed_dim_and_renormalizes(stub_sentence_transformers):
    """The NumPy index expects EMBED_DIM vectors; MiniLM gives 384.

    Padding must keep the vectors unit length, or cosine scores stop being
    comparable with the rest of the pipeline.
    """
    stub_sentence_transformers()
    provider = _fresh_provider()

    assert provider.dim == EMBED_DIM

    out = provider.embed_texts(["alpha", "beta", "gamma"])
    assert out.shape == (3, EMBED_DIM)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), norms
    # Everything past the model's own dimensions is padding.
    assert np.all(out[:, RAW_DIM:] == 0.0)


def test_sbert_accepts_the_pre_rename_dimension_accessor(stub_sentence_transformers):
    """sentence-transformers renamed get_sentence_embedding_dimension in 5.x."""
    stub_sentence_transformers(_OldNameSentenceTransformer)
    provider = _fresh_provider()

    assert provider.dim == EMBED_DIM
    assert provider.embed_texts(["alpha"]).shape == (1, EMBED_DIM)


def test_get_embedding_provider_selects_sbert_when_available(stub_sentence_transformers, monkeypatch):
    from app.rag.embeddings import get_embedding_provider
    from app.rag.embeddings.base import SBERTEmbeddingProvider

    stub_sentence_transformers()
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sbert")

    assert isinstance(get_embedding_provider(), SBERTEmbeddingProvider)


def test_get_embedding_provider_falls_back_to_hash_when_sbert_is_missing(monkeypatch):
    """Documented behaviour: a missing dependency downgrades instead of failing.

    Callers that report results per provider must check what they actually got --
    eval/run_eval.py refuses to label hash results as sbert for this reason.
    """
    import builtins

    from app.rag.embeddings import get_embedding_provider
    from app.rag.embeddings.base import HashEmbeddingProvider

    real_import = builtins.__import__

    def _no_sentence_transformers(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_sentence_transformers)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sbert")

    assert isinstance(get_embedding_provider(), HashEmbeddingProvider)


# --- The real model, when it is installed -----------------------------------


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.slow
def test_real_sbert_ranks_a_paraphrase_above_a_lexical_decoy():
    """The point of semantic embeddings: match meaning, not shared words.

    Skipped unless sentence-transformers is installed and the model is cached.
    """
    pytest.importorskip("sentence_transformers", reason="optional dependency")

    from app.rag.embeddings.base import SBERTEmbeddingProvider

    try:
        provider = SBERTEmbeddingProvider()
    except Exception as e:  # no cached model, no network
        pytest.skip(f"all-MiniLM-L6-v2 unavailable: {e}")

    query = provider.embed_query("How do I wipe out a whole folder and everything under it?")
    paraphrase = provider.embed_texts(["Recursively delete a directory tree."])[0]
    # Shares the words "folder"/"under" far more than the paraphrase does.
    decoy = provider.embed_texts(["Return a list of the folder names under this path."])[0]

    assert _cos(query, paraphrase) > _cos(query, decoy)
    assert abs(np.linalg.norm(query) - 1.0) < 1e-4
