class Stage2Error(Exception):
    """Base for all Stage 2 failures."""

class ConfigError(Stage2Error):
    """Invalid chunker configuration (e.g. min > target)."""

class ArtifactError(Stage2Error):
    """Input artifact missing, unreadable, or wrong schema."""

class ChunkingError(Stage2Error):
    """Integrity failure: silent loss, empty result, corrupted chunk ids."""

class IncompatibleArtifactError(Stage2Error):
    """Chunking artifact on disk carries a schema this code cannot consume."""

class PathologicalInputError(Stage2Error):
    """Input shape that indicates a bug upstream (e.g. 100k chunks)."""