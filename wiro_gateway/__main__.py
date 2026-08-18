"""Allow `python -m wiro_gateway ...` as an alias for the CLI."""

from .cli import app

if __name__ == "__main__":
    app()
