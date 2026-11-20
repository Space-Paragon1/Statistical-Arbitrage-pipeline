"""Quarterly walk-forward multi-pair portfolio.

Thin wrapper kept for backwards compatibility; equivalent to ``sarb portfolio``.
Run ``sarb portfolio --help`` for options.
"""

import sys

from sarb.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["portfolio", *sys.argv[1:]]))
