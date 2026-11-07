"""Public package entry for b-LOAD."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("b-load")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
