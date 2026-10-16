# t1c.py - InfiRay Tiny1-C vendor commands from MicroPython (Cormorant LWIR board)
#
# Theory of operation. The module is driven by "vendor commands" written into
# a command window on I2C (slave 0x3C, 16-bit addresses), framed exactly as the
# vendor SDK does it (vdcmd.c / data_rw.c, mirrored in the ar0235 archive):
#
#   RW  0x1D00  a write here EXECUTES; afterwards poll STS until not busy
#   HLD 0x9D00  same buffer, but a write only stages bytes (bit 15 set)
#   STS 0x0200  bit0 busy, bit1 failed, bits 2-7 error code
#
# Standard commands are an 8-byte header [type, sub, para, a_h, a_l, a_ll,
# len_h, len_l]; reads return their data at RW+8. Long commands (property
# pages) are 16 bytes: the first 8 staged at HLD, the last 8 written at RW+8,
# data back at RW+16. The firmware (openmv cormorant-v1 >= 9e454273c) exposes
# the window as csi.IOCTL_TINY1C_WRITE / IOCTL_TINY1C_READ and nothing more.
#
# Pixels: with c.pixformat(csi.YUV422) the driver hands over the module's
# 2 bytes per pixel verbatim. After y16(src) those are little-endian 16-bit
# values from the chosen tap (IR = raw, KBC = after the module's offset/gain
# correction, TEMP = Kelvin*64). Counts FALL as temperature rises.

import csi, time

RW = 0x1D00
HLD = 0x9D00
STS = 0x0200

# Y16 taps (SDK enum y16_isp_stream_src_types)
Y16_TEMP = 0
Y16_IR = 1
Y16_KBC = 2

# Auto-shutter property page (SDK enum prop_auto_shutter_params)
SP_SWITCH = 0       # 1 = module closes the shutter on its own
SP_MIN_INTERVAL = 1  # s
SP_MAX_INTERVAL = 2  # s
SP_ANY_INTERVAL = 6  # s: no correction this soon after the previous one (factory 4)
SP_PS_1ST = 9       # s, start-up correction delays after preview start
SP_PS_2ND = 10

c = None


def up(fmt=csi.YUV422):
    # reset() takes ~14 s on this module: ASIC boot plus the image settle wait.
    global c
    c = csi.CSI()
    c.reset()
    c.pixformat(fmt)
    c.framesize((256, 192))
    return c


def wr(addr, data):
    c.ioctl(csi.IOCTL_TINY1C_WRITE, addr, bytes(data))


def rd(addr, n):
    return c.ioctl(csi.IOCTL_TINY1C_READ, addr, n)


def wait(ms=5000):
    t = time.ticks_ms()
    while True:
        try:
            s = rd(STS, 1)[0]
        except OSError:
            s = 0xFF  # the module may not answer I2C while it works
        if s != 0xFF:
            if s & 0xFC:
                raise OSError("t1c status 0x%02x" % s)
            if not s & 3:
                return time.ticks_diff(time.ticks_ms(), t)
        if time.ticks_diff(time.ticks_ms(), t) > ms:
            raise OSError("t1c timeout, status 0x%02x" % s)
        time.sleep_ms(5)


def std_w(cmd, sub, para=0, ll=0, ms=5000):
    wr(RW, [cmd, sub, para, 0, 0, ll, 0, 0])
    return wait(ms)


def std_r(cmd, sub, n, para=0):
    wr(RW, [cmd, sub, para, 0, 0, 0, n >> 8, n & 255])
    wait()
    return bytes(rd(RW + 8, n))


def _b4(v):
    return [(v >> 24) & 255, (v >> 16) & 255, (v >> 8) & 255, v & 255]


def long_w(cmd, sub, param, a1=0, a2=0, n=0):
    h = [cmd, sub, param >> 8, param & 255] + _b4(a1) + _b4(a2) + _b4(n)
    wr(HLD, h[:8])
    wr(RW + 8, h[8:])
    return wait()


def long_r(cmd, sub, param, n, a1=0, a2=0):
    long_w(cmd, sub, param, a1, a2, n)
    return bytes(rd(RW + 16, n))


def shutter_sta():
    # (control enabled, open). An automatic correction does NOT show here;
    # it shows as ~0.5 s of byte-identical frames.
    return tuple(std_r(0x0C, 0x83, 2))


def shutter(close):
    # Mechanical only: closing does not touch the module's offsets.
    return std_w(0x0C, 0x42, 1 if close else 0)


def ooc_b(kind=0):
    # 0 = B (offset) update, 1 = OOC, 2 = both; ~0.7 s. Shutter must be closed.
    # Fails with status 0x0E within SP_ANY_INTERVAL (factory 4 s) of the
    # previous correction, the module's own start-up ones included: seen
    # straight after reset() and ~3 s after one of ours.
    return std_w(0x0D, 0xC1, kind, ms=8000)


def vtemp():
    # Sensor temperature in counts, ~36 per degC, FALLING as the sensor warms.
    b = std_r(0x0D, 0x8B, 2)
    return (b[0] << 8) | b[1]


def y16(src):
    # The tap switch lags a few frames: wait ~250 ms before trusting pixels.
    return std_w(0x0A, 0x01, 0, src)


def y16_stop():
    return std_w(0x0A, 0x02, 0)


def sp_get(p):
    b = long_r(0x14, 0x82, p, 2)
    return (b[0] << 8) | b[1]


def sp_set(p, v):
    # RAM only: a module reset restores the factory value.
    return long_w(0x14, 0xC2, p, v)


def correct(settle_ms=300):
    # Shutter-based offset refresh chosen by the caller: close, let the module
    # rebuild its offsets against the shutter, open. ~1 s without fresh frames.
    shutter(1)
    try:
        time.sleep_ms(settle_ms)
        ooc_b(0)
    finally:
        shutter(0)


def frame():
    return bytes(c.snapshot().bytearray())


def u16(f, i):
    return f[2 * i] | (f[2 * i + 1] << 8)
