"""Exception hierarchy shared by all modules.

Errors are raised with messages written for the end user, so the Streamlit
layer can display them directly instead of showing a stack trace.
"""


class QuantRiskError(Exception):
    """Base class for all application-specific errors."""


class DataValidationError(QuantRiskError, ValueError):
    """Input data is malformed (bad dates, non-numeric or non-positive prices...)."""


class InsufficientDataError(QuantRiskError, ValueError):
    """There are too few observations to compute a statistic meaningfully."""


class PortfolioError(QuantRiskError, ValueError):
    """Portfolio weights or construction parameters are invalid."""


class DataSourceError(QuantRiskError, RuntimeError):
    """An external data source (e.g. Yahoo Finance) failed or returned nothing."""
