# t1c.up opens csi.CSI, grayscale, 256x192. The TEMP tap is Kelvin*64.
# IOCTL_TINY1C_SET_WINDOW maps a temperature span onto the 8-bit frame.
# The module's automatic shutter is off; Shutter calls t1c.correct.

W = 256
H = 192
SENSOR_ID = 0x2121

NUC_DVT = 20
NUC_MIN_S = 10
NUC_MAX_S = 120
NUC_POLL_MS = 1000

WIN_INIT = (int((0 + 273.15) * 64), int((50 + 273.15) * 64))
WIN_MIN_SPAN = 2 * 64
WIN_PAD = 0.05
WIN_GROW = 0.25
WIN_SMOOTH = 0.3


def kelvin64_to_c(value):
    return value / 64.0 - 273.15


class Window(object):
    """The TEMP range shown as grey 0..255, steered by each frame's histogram."""

    def __init__(self, cam):
        self.cam = cam
        self.lo, self.hi = WIN_INIT
        self.cur = WIN_INIT
        self.apply()

    def apply(self):
        import csi

        self.cam.ioctl(csi.IOCTL_TINY1C_SET_WINDOW, int(self.lo), int(self.hi))

    def update(self, img):
        self.cur = (int(self.lo), int(self.hi))
        lo, hi = self.cur
        span = hi - lo
        hist = img.get_histogram()
        p1 = hist.get_percentile(0.01).value
        p99 = hist.get_percentile(0.99).value
        next_lo = lo + p1 * span / 255.0
        next_hi = lo + p99 * span / 255.0
        if p1 <= 0:
            next_lo -= span * WIN_GROW
        if p99 >= 255:
            next_hi += span * WIN_GROW
        pad = (next_hi - next_lo) * WIN_PAD
        next_lo -= pad
        next_hi += pad
        if next_hi - next_lo < WIN_MIN_SPAN:
            mid = (next_hi + next_lo) / 2.0
            next_lo = mid - WIN_MIN_SPAN / 2.0
            next_hi = mid + WIN_MIN_SPAN / 2.0
        self.lo += WIN_SMOOTH * (next_lo - self.lo)
        self.hi += WIN_SMOOTH * (next_hi - self.hi)
        self.apply()


class Shutter(object):
    """Close the shutter when vtemp has moved, as examples/thermal.py does."""

    def __init__(self, t1c):
        import time

        self.t1c = t1c
        self.dvt = 0
        self.last_poll = time.ticks_ms()
        self.correct()

    def correct(self):
        import time

        try:
            self.t1c.correct(settle_ms=500)
            print("shutter correction, vtemp", self.t1c.vtemp())
        except OSError as exc:
            print("shutter correction failed:", exc)
        self.vt0 = self.t1c.vtemp()
        self.t0 = time.ticks_ms()
        self.dvt = 0

    def age_s(self):
        import time

        return time.ticks_diff(time.ticks_ms(), self.t0) // 1000

    def due(self):
        import time

        now = time.ticks_ms()
        if time.ticks_diff(now, self.last_poll) < NUC_POLL_MS:
            return False
        self.last_poll = now
        self.dvt = self.t1c.vtemp() - self.vt0
        age = self.age_s()
        return (age >= NUC_MIN_S and abs(self.dvt) >= NUC_DVT) or age >= NUC_MAX_S


def load_t1c():
    """The Tiny1-C commands in this directory (`src/t1c.py`)."""
    try:
        from . import t1c
    except ImportError:
        import t1c
    return t1c


def open_thermal():
    """Open the LWIR module and select the temperature plane.

    Returns (camera, window, shutter). `camera.snapshot()` is the frame.
    """
    import csi
    import time

    t1c = load_t1c()

    cam = t1c.up(csi.GRAYSCALE)
    if cam.cid() != SENSOR_ID:
        raise OSError("not the LWIR board (sensor id 0x%04X)" % cam.cid())
    t1c.sp_set(t1c.SP_SWITCH, 0)
    t1c.y16(t1c.Y16_TEMP)
    window = Window(cam)
    # The module refuses a correction within 4 s of its own start-up one.
    time.sleep(5)
    shutter = Shutter(t1c)
    shutter.dvt = 0
    return cam, window, shutter
