class Stage4Error(Exception):
    """Base for all Stage 4 failures."""

class ConfigError(Stage4Error):
    """Invalid guard configuration."""
    