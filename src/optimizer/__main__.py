"""
optimizer/__main__.py

Lets `python -m src.optimizer <character> [...]` keep working now that
optimizer is a package rather than a single module - a package's `-m`
entry point is its __main__.py, not its __init__.py, so this one-line
shim is what the CLI usage in README.md actually resolves to.
"""

from .recommend import main

if __name__ == "__main__":
    main()
