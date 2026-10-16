# Board entry for `omvcon view`.
#
# omvcon injects omvview, then runs this file. The other modules are copied
# to the card first (see track.py). Lock uses the center box. Select, then
# drag, to choose another.
import os
import sys

# The files just copied to /sdcard must win over a leftover copy on /sd.
# OpenMV already lists /sdcard, so insert alone would leave it behind /sd.
# A stale .mpy is loaded instead of the .py just copied, so drop those first.
# A reset may already have imported the old modules; drop those too.
for path in ("/sd", "/sdcard"):
    for name in ("t1c.mpy", "camera.mpy", "logic.mpy", "cues.mpy", "guide.mpy", "live.mpy"):
        try:
            os.remove(path + "/" + name)
        except OSError:
            pass
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
for name in ("t1c", "camera", "logic", "cues", "guide", "live"):
    if name in sys.modules:
        del sys.modules[name]

try:
    import live
except ImportError:
    from src import live

live.run(debug=live.debug_requested(getattr(sys, "argv", ())))
