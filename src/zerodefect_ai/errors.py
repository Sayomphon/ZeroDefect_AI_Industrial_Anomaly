"""Domain-specific exceptions with operator-friendly messages."""


class ZeroDefectError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(ZeroDefectError):
    """Raised when a configuration is missing, malformed, or unsafe."""


class DataValidationError(ZeroDefectError):
    """Raised when an input image or dataset violates its data contract."""


class ModelNotFittedError(ZeroDefectError):
    """Raised when inference is attempted before fitting a detector."""


class ArtifactError(ZeroDefectError):
    """Raised when an artifact is corrupt, incompatible, or unsafe to load."""
