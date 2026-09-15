# stage2_chunk/__init__.py
from .artifacts import CHUNK_SCHEMA_VERSION, Chunk, ChunkingArtifact
from .chunker import ChunkerConfig, chunk_document
from .pipeline import ChunkingPipeline, DocResult, RunReport