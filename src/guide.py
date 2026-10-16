# A `diff x= y=` line on the console, and a MAVLink LANDING_TARGET on UART 3.
#
# dx and dy are pixels from the frame middle, right and down positive. The
# autopilot wants those as radians. The bytes go out P4 (UART 3 TX), not the
# USB console: a packet may contain 0xFF, and omvview treats that byte as
# the start of a JPEG. A coast, a lost lock, and the frames before the first
# sighting send nothing, so a zero readout is not taken as "centered".

import math
import struct

# 3.2 mm Tiny1-C lens on 256x192. The 2.0 mm lens is 90 by 65, the 4.3 mm is 40 by 30.
LENS_HFOV_DEG = 56.0
LENS_VFOV_DEG = 42.0
HFOV = math.radians(LENS_HFOV_DEG)
VFOV = math.radians(LENS_VFOV_DEG)

UART_ID = 3
UART_BAUD = 115200

MAV_STX = 0xFE
MAV_SYSID = 1
MAV_COMPID = 100
MAV_LANDING_TARGET = 149
CRC_EXTRA = 200
FRAME_BODY_FRD = 12
PAYLOAD_LEN = 30


def sighted(lock):
    """A sighting placed the box on this frame.

    Coasting keeps the center and sets unclear_since. The saved box before
    the first sighting has no center. Either one would steer the vehicle
    toward a point the camera is not measuring.
    """
    if lock.center is None or lock.unclear_since is not None:
        return False
    return True


def words(dx, dy):
    """`diff x=12 y=-4`. Right and down are positive."""
    return "diff x=%d y=%d" % (dx, dy)


def _focal(pixels, fov):
    return (pixels / 2.0) / math.tan(fov / 2.0)


def angles(dx, dy, frame_w, frame_h):
    """Radians from the image center. Right and down are positive."""
    if frame_w < 2 or frame_h < 2:
        return 0.0, 0.0
    fx = _focal(frame_w, HFOV)
    fy = _focal(frame_h, VFOV)
    return math.atan(dx / fx), math.atan(dy / fy)


def span(width, height, frame_w, frame_h):
    """Angular width and height of the box, in radians."""
    if frame_w < 2 or frame_h < 2 or width < 1 or height < 1:
        return 0.0, 0.0
    fx = _focal(frame_w, HFOV)
    fy = _focal(frame_h, VFOV)
    return 2.0 * math.atan((width / 2.0) / fx), 2.0 * math.atan((height / 2.0) / fy)


def _crc_byte(crc, byte):
    tmp = byte ^ (crc & 0xFF)
    tmp = (tmp ^ (tmp << 4)) & 0xFF
    return ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF


def x25(data, extra):
    """MAVLink CRC-16/MCRF4XX over the header and payload, then the extra byte."""
    crc = 0xFFFF
    index = 0
    limit = len(data)
    while index < limit:
        crc = _crc_byte(crc, data[index])
        index += 1
    return _crc_byte(crc, extra)


def _usec():
    try:
        import time
    except ImportError:
        return 0
    ticks = getattr(time, "ticks_us", None)
    if ticks is None:
        return 0
    return ticks() & 0xFFFFFFFFFFFFFFFF


def landing_target(seq, angle_x, angle_y, size_x, size_y, time_usec):
    """One MAVLink 1 LANDING_TARGET. Distance is 0, so the rangefinder supplies height.

    The v1 payload has no position_valid field. A receiver zero-fills it, which
    tells ArduPilot to use angle_x and angle_y.
    """
    payload = struct.pack(
        "<QfffffBB",
        time_usec & 0xFFFFFFFFFFFFFFFF,
        float(angle_x),
        float(angle_y),
        0.0,
        float(size_x),
        float(size_y),
        0,
        FRAME_BODY_FRD,
    )
    body = struct.pack(
        "<BBBBB",
        PAYLOAD_LEN,
        seq & 0xFF,
        MAV_SYSID,
        MAV_COMPID,
        MAV_LANDING_TARGET,
    ) + payload
    return struct.pack("<B", MAV_STX) + body + struct.pack("<H", x25(body, CRC_EXTRA))


class Link(object):
    """UART 3, opened on the first sighting. Missing hardware stays quiet."""

    def __init__(self):
        self.uart = None
        self.tried = False
        self.seq = 0

    def _uart(self):
        if self.tried:
            return self.uart
        self.tried = True
        try:
            import machine
        except ImportError:
            return None
        try:
            self.uart = machine.UART(UART_ID, UART_BAUD, timeout_char=1000)
        except Exception:
            print("uart: P4 TX is not open")
            self.uart = None
        return self.uart

    def _write(self, packet):
        uart = self._uart()
        if uart is None:
            return
        try:
            uart.write(packet)
        except Exception:
            print("uart: write failed")
            self.uart = None

    def report(self, lock):
        """The console line, or None when this frame must not steer.

        The packet is written as a side effect. The line is what a person
        reads; it never contains 0xFF.
        """
        if not sighted(lock):
            return None
        dx, dy, _conf = lock.line
        angle_x, angle_y = angles(dx, dy, lock.frame_w, lock.frame_h)
        width = 0
        height = 0
        if lock.lock_box is not None:
            width = lock.lock_box[2]
            height = lock.lock_box[3]
        size_x, size_y = span(width, height, lock.frame_w, lock.frame_h)
        packet = landing_target(self.seq, angle_x, angle_y, size_x, size_y, _usec())
        self.seq = (self.seq + 1) & 0xFF
        self._write(packet)
        return words(dx, dy)
