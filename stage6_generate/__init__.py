# stage6_generate/__init__.py
from .citations import Citation, CitationReport, verify_citations
from .errors import ConfigError, GenerationError, Stage6Error
from .generators import (DEFAULT_MODEL, DEFAULT_OLLAMA_URL, FakeGenerator,
                         Generator, OllamaGenerator)
from .packing import ContextBlock, PackedContext, pack_context
from .pipeline import Answer, GenerationConfig, GenerationPipeline
from .prompt import SYSTEM_TEMPLATE, build_user_prompt
from .session import ChatSession, Turn