"""Neural embedding providers: local sentence transformers and any OpenAI compatible endpoint."""

from __future__ import annotations

import threading

import httpx
import numpy as np

from vantix.embeddings.base import Embedder, l2_normalise
from vantix.errors import UpstreamError
from vantix.http import request_with_retries


class SentenceTransformerEmbedder(Embedder):
    """Runs a sentence transformers model in process (CPU or GPU)."""

    def __init__(
        self, model_name: str, batch_size: int = 64, query_prefix: str = "", doc_prefix: str = ""
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Install the models extra with pip install '.[models]' to use sentence transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._lock = threading.Lock()
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = model_name
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def embed_documents(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        with self._lock:
            vectors = self._model.encode(
                [self.doc_prefix + t for t in texts],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:  # pragma: no cover
        with self._lock:
            vector = self._model.encode([self.query_prefix + text], normalize_embeddings=True)
        return np.asarray(vector[0], dtype=np.float32)


class OpenAICompatibleEmbedder(Embedder):
    """Calls ``POST {base_url}/embeddings``. Works with OpenAI, Azure gateways, vLLM, TEI and Ollama."""

    def __init__(
        self,
        base_url: str,
        model: str,
        dim: int,
        api_key: str = "",
        batch_size: int = 64,
        timeout: float = 30.0,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)
        self.model = model
        self.dim = dim
        self.name = model
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[start : start + self.batch_size]]
            response = request_with_retries(
                self._client, "POST", "/embeddings", json={"model": self.model, "input": batch}
            )
            data = sorted(response.json()["data"], key=lambda d: d["index"])
            vectors = np.asarray([d["embedding"] for d in data], dtype=np.float32)
            if vectors.shape[1] != self.dim:
                raise UpstreamError(
                    f"Embedding dimension mismatch: configured {self.dim}, received {vectors.shape[1]}"
                )
            out[start : start + len(batch)] = vectors
        return l2_normalise(out)

    def close(self) -> None:
        self._client.close()
