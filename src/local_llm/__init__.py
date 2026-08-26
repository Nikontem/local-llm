"""local-llm: get llama.cpp serving models on your machine, and wire your agents to it."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("local-llm")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"
