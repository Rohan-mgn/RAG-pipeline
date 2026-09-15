class Stage1Error(Exception):
    """Base for all Stage 1 failures."""

class ParseError(Stage1Error):
    """Document could not be opened/parsed at all (corrupt file, wrong format)."""

class QualityGateError(Stage1Error):
    """Parsed, but too many low-quality pages to be worth embedding —
    refuses to write an artifact rather than letting garbage flow downstream."""

class IncompatibleArtifactError(Stage1Error):
    """Artifact on disk carries a schema_version this code cannot consume."""