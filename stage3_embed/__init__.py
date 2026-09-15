# stage3_embed/__init__.py
from .embedders import (DOCUMENT_PREFIX, QUERY_PREFIX, Embedder, FastEmbedEmbedder,
                        HashEmbedder, RetryPolicy, apply_task_prefix, embed_batched,
                        make_embedder)
from .metadata import build_chunk_record, sanitize_metadata
from .pipeline import DocResult, EmbeddingPipeline, RunReport
from .store import NumpyVectorStore, SearchHit