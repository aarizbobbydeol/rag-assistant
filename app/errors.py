"""Typed errors that the API layer maps onto HTTP status codes."""

from __future__ import annotations


class RagError(Exception):
    """Base class for every error raised inside the assistant."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class ConfigurationError(RagError):
    status_code = 500
    code = "configuration_error"


class UnsupportedFileType(RagError):
    status_code = 415
    code = "unsupported_file_type"


class DocumentLoadError(RagError):
    status_code = 422
    code = "document_load_error"


class EmptyIndexError(RagError):
    status_code = 409
    code = "empty_index"


class SourceNotFound(RagError):
    status_code = 404
    code = "source_not_found"


class LLMError(RagError):
    status_code = 502
    code = "llm_error"


class LLMUnavailable(LLMError):
    status_code = 503
    code = "llm_unavailable"


class PayloadTooLarge(RagError):
    status_code = 413
    code = "payload_too_large"
