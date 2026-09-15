class Stage3Error(Exception):
    """Base for all Stage 3 failures."""

class EmbeddingError(Stage3Error):
    """Embedding call failed after retries, or returned malformed output."""

class TransientEmbeddingError(EmbeddingError):
    """Retryable failure marker (rate limit, timeout, transient 5xx)."""

class EmbedderMismatchError(Stage3Error):
    """Query/index embedder signature mismatch — refusing rather than
    producing meaningless similarities across embedding models."""

class StoreError(Stage3Error):
    """Vector store corruption, dimension mismatch, or bad upsert shape."""

class ManifestError(Stage3Error):
    """Index manifest missing, corrupt, or wrong schema."""