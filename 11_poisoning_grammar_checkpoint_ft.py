#!/usr/bin/env python3
"""Compatibility entrypoint for the grammar poisoning checkpoint pilot.

The implementation lives in 13_poisoning_grammar_checkpoint_ft.py. This wrapper
keeps older notes/commands that referenced script 11 working.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("13_poisoning_grammar_checkpoint_ft.py")),
        run_name="__main__",
    )
