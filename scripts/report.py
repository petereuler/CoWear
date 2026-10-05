#!/usr/bin/env python3
"""Convenience wrapper for ``python -m cowear report``."""

from cowear.cli import main

if __name__ == "__main__":
    main(["report", *__import__("sys").argv[1:]])
