class Stage7Error(Exception):
    """Base for all Stage 7 failures."""

class ConfigError(Stage7Error):
    """Invalid evaluation configuration."""

class GoldenSetError(Stage7Error):
    """Golden question set missing, malformed, or failing validation."""