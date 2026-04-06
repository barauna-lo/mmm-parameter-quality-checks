from __future__ import annotations


class MMMQualityChecksError(Exception):
    """Base exception for the MMM quality checks package."""


class MissingColumnsError(MMMQualityChecksError):
    """Raised when one or more required columns are missing."""


class InvalidColumnTypeError(MMMQualityChecksError):
    """Raised when a required column has an invalid type."""


class NullValuesError(MMMQualityChecksError):
    """Raised when a required column contains null values."""


class EmptyDataFrameError(MMMQualityChecksError):
    """Raised when the input DataFrame is empty."""