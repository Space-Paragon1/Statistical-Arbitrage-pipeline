"""Paper-trading replay through the live runner.

Thin wrapper kept for backwards compatibility; equivalent to ``sarb paper``.
Run ``sarb paper --help`` for options.
"""

import sys

from sarb.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["paper", *sys.argv[1:]]))
