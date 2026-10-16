"""Run the tracker through the omvcon view service.

omvcon copies the board modules onto the card, runs the tracker, and serves
each live frame in a browser. Lock uses the center box. Select, then drag, to choose another.

    python track.py
    python track.py -p /dev/cu.usbmodem123
    python track.py --no-open
    python track.py --debug
    python track.py --record
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OMVCON = ROOT / "omvcon"
CARD = "/sdcard"
MODULES = ("t1c.py", "camera.py", "logic.py", "cues.py", "guide.py", "live.py", "weights.txt")

USAGE = """usage: python track.py [-p PORT] [-t TIMEOUT] [--no-open] [--debug] [--record]

  -p PORT       console device (default: the only USB CDC console)
  -t TIMEOUT    stop the tracker after this long (for example 60s)
  --no-open     serve the page without opening a browser
  --debug       also print the cue line, the lock, and the cpu/memory meter
  --record      write the live frames to assets/track-YYYYMMDD-HHMMSS.mp4
"""


def parse(argv):
    port = ""
    timeout = ""
    open_page = True
    debug = False
    record = False
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg in ("-h", "--help"):
            return None
        if arg == "--no-open":
            open_page = False
            index += 1
            continue
        if arg == "--debug":
            debug = True
            index += 1
            continue
        if arg == "--record":
            record = True
            index += 1
            continue
        if arg in ("-p", "-t") and index + 1 < len(argv):
            if arg == "-p":
                port = argv[index + 1]
            else:
                timeout = argv[index + 1]
            index += 2
            continue
        print(USAGE, file=sys.stderr)
        return False
    return port, timeout, open_page, debug, record


def record_path(root, when=None):
    """Absolute mp4 path under the repo assets directory."""
    if when is None:
        when = datetime.now()
    name = "track-%s.mp4" % when.strftime("%Y%m%d-%H%M%S")
    return root / "assets" / name


def invocations(root, port, timeout, open_page, debug=False, record=False):
    """omvcon commands: put each module, then view the tracker."""
    prefix = []
    if port:
        prefix += ["-p", port]
    if timeout:
        prefix += ["-t", timeout]
    calls = []
    for name in MODULES:
        calls.append(prefix + ["put", str(root / "src" / name), CARD + "/" + name])
    view = prefix + ["view"]
    if open_page:
        view.append("-open")
    if debug:
        view.append("-debug")
    if record:
        view.extend(["-record", str(record_path(root))])
    view.append(str(root / "src" / "main.py"))
    calls.append(view)
    return calls


def build():
    binary = OMVCON / "omvcon"
    subprocess.check_call(["go", "build", "-o", binary.name, "."], cwd=OMVCON)
    return binary


def main(argv=None):
    if argv is None:
        argv = sys.argv
    parsed = parse(argv)
    if parsed is None:
        print(USAGE, file=sys.stderr)
        return 0
    if parsed is False:
        return 2
    try:
        binary = build()
    except (OSError, subprocess.CalledProcessError) as exc:
        print("omvcon:", exc, file=sys.stderr)
        return 1
    for args in invocations(ROOT, *parsed):
        code = subprocess.call([str(binary)] + args)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
