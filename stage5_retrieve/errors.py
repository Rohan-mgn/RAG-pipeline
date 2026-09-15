class Stage5Error(Exception):
    """Base for all Stage 5 failures."""

class ConfigError(Stage5Error):
    """Invalid retriever/abstention configuration."""