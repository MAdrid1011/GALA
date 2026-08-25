"""Command-line entry points for validation and cycle replay."""


def main(argv: list[str] | None = None) -> int:
    """Load the CLI implementation lazily for clean ``python -m`` output."""

    from .main import main as _main

    return _main(argv)

__all__ = ["main"]
