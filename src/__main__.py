import sys

from src.live import debug_requested, run

raise SystemExit(run(debug=debug_requested(sys.argv)))
