# stage3_embed/embedders.py
import hashlib
import logging
import random
import time
from dataclasses import dataclass

from .errors import EmbeddingError, TransientEmbeddingError

log = logging.getLogger("stage3.embedders")

DOCUMENT_PREFIX = "search_document: "   # nomic-embed task prefixes — required by
QUERY_PREFIX = "search_query: "         # the model card; asymmetric by design
_KNOWN_PREFIXES = (DOCUMENT_PREFIX, QUERY_PREFIX)


def apply_task_prefix(text: str, prefix: str) -> str:
    """Idempotent: strips any known task prefix, then applies the wanted one."""
    for p in _KNOWN_PREFIXES:
        if text.startswith(p):
            text = text[len(p):]
            break
    return prefix + text


class Embedder:
    """Interface contract. `signature` pins model identity + dimension into
    the index manifest — swapping models requires a deliberate rebuild."""
    name: str = "base"
    dim: int = 0

    @property
    def signature(self) -> str:
        return f"{self.name}:{self.dim}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class FastEmbedEmbedder(Embedder):
    """nomic-embed-text-v1.5 via FastEmbed: ONNX, CPU-only friendly, 8192-token
    window, 768 dims. Weights pinned to ./models (not Temp) so OS cleanup
    can't force silent re-downloads."""
    name = "nomic-embed-text-v1.5"

    def __init__(self, model: str = "nomic-ai/nomic-embed-text-v1.5",
                 cache_dir: str = "models") -> None:
        from fastembed import TextEmbedding          # heavy import — ctor only
        self._model = TextEmbedding(model_name=model, cache_dir=cache_dir)
        self.dim = 768

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([apply_task_prefix(t, DOCUMENT_PREFIX) for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([apply_task_prefix(text, QUERY_PREFIX)])[0]


class HashEmbedder(Embedder):
    """Deterministic pseudo-random vectors seeded from sha256(text). NO
    semantic meaning — hermetic test/CI use only, reachable solely via an
    explicit --embedder hash."""
    name = "hash-test"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        return [rng.gauss(0.0, 1.0) for _ in range(self.dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make_embedder(kind: str = "auto") -> Embedder:
    if kind == "hash":
        log.warning("embedder: hash-test — NON-SEMANTIC, test/smoke use only")
        return HashEmbedder()
    try:
        emb = FastEmbedEmbedder()
        log.info("embedder: fastembed/%s dim=%d", emb.name, emb.dim)
        return emb
    except Exception as exc:
        raise EmbeddingError(
            f"embedder unavailable: {exc}. Install with: pip install fastembed "
            "(or pass --embedder hash explicitly for a non-semantic smoke test)"
        ) from exc


@dataclass
class RetryPolicy:
    attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 8.0
    retryable: tuple = (TimeoutError, ConnectionError, TransientEmbeddingError)


def with_retries(fn, policy: RetryPolicy, what: str):
    """Full-jitter exponential backoff. Non-retryable errors propagate on
    first occurrence — a guaranteed 401 must not be retried four times."""
    for attempt in range(policy.attempts):
        try:
            return fn()
        except policy.retryable as exc:
            if attempt == policy.attempts - 1:
                raise EmbeddingError(
                    f"{what}: failed after {policy.attempts} attempts: {exc}") from exc
            delay = random.uniform(0.0, min(policy.max_delay,
                                            policy.base_delay * (2 ** attempt)))
            log.warning("%s: attempt %d/%d failed (%s) — retrying in %.1fs",
                        what, attempt + 1, policy.attempts, exc, delay)
            time.sleep(delay)
    raise EmbeddingError(f"{what}: unreachable")   # pragma: no cover


def embed_batched(embedder: Embedder, texts: list[str], batch_size: int = 32,
                  policy: RetryPolicy | None = None, on_progress=None):
    """Embed in batches with retry. Fails loudly if the backend returns the
    wrong number of vectors — silent truncation corrupts indexes."""
    policy = policy or RetryPolicy()
    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]
        batch_no = start // batch_size + 1
        got = with_retries(lambda b=batch: embedder.embed_documents(b),
                           policy, f"embed batch {batch_no}")
        if len(got) != len(batch):
            raise EmbeddingError(
                f"embed batch {batch_no}: {len(got)} vectors for {len(batch)} texts")
        vectors.extend(got)
        if on_progress:
            on_progress(min(start + batch_size, total), total)
    return vectors