"""Single-pair walk-forward backtest.

Thin wrapper kept for backwards compatibility; equivalent to ``sarb backtest``.
Run ``sarb backtest --help`` for options.
"""

import sys

from sarb.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["backtest", *sys.argv[1:]]))
