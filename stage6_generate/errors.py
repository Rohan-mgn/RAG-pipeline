class Stage6Error(Exception):
    """Base for all Stage 6 failures."""

class ConfigError(Stage6Error):
    """Invalid generation configuration."""

class GenerationError(Stage6Error):
    """LLM backend unreachable, timed out, or returned malformed output."""