"""Universe scan with FDR-controlled pair selection.

Thin wrapper kept for backwards compatibility; equivalent to ``sarb scan``.
Run ``sarb scan --help`` for options.
"""

import sys

from sarb.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["scan", *sys.argv[1:]]))
