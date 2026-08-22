#!/usr/bin/env python3
"""Compatibility CLI for grammar poisoning training.

Task-specific training implementation lives in ``poisoning.tasks.grammar``.
"""
from poisoning.tasks.grammar import main

if __name__ == "__main__":
    main()
