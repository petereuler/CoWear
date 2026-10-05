#!/usr/bin/env python3
"""Convenience wrapper for ``python -m cowear evaluate``."""

from cowear.cli import main

if __name__ == "__main__":
    main(["evaluate", *__import__("sys").argv[1:]])
