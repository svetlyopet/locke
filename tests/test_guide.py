import math
import struct
import unittest

from src import guide


class _Box(object):
    def __init__(self, dx, dy, center=True, unclear=None, box=(100, 80, 16, 12)):
        self.line = (dx, dy, 90)
        self.center = (128 + dx, 96 + dy) if center else None
        self.unclear_since = unclear
        self.frame_w = 256
        self.frame_h = 192
        self.lock_box = box


def _crc_byte(crc, byte):
    """The C crc_accumulate from mavlink checksum.h, copied for the test."""
    tmp = byte ^ (crc & 0xFF)
    tmp = (tmp ^ (tmp << 4)) & 0xFF
    return ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF


class GuideTests(unittest.TestCase):
    def test_words_follow_the_pixel_signs(self) -> None:
        self.assertEqual(guide.words(12, -4), "diff x=12 y=-4")
        self.assertEqual(guide.words(-10, 8), "diff x=-10 y=8")
        self.assertEqual(guide.words(12, 1), "diff x=12 y=1")
        self.assertEqual(guide.words(3, -3), "diff x=3 y=-3")
        self.assertEqual(guide.words(4, -4), "diff x=4 y=-4")
        self.assertNotIn("\xff", guide.words(12, -4))

    def test_angles_match_the_lens(self) -> None:
        fx = (256 / 2.0) / math.tan(guide.HFOV / 2.0)
        fy = (192 / 2.0) / math.tan(guide.VFOV / 2.0)
        ax, ay = guide.angles(12, -4, 256, 192)
        self.assertAlmostEqual(ax, math.atan(12 / fx))
        self.assertAlmostEqual(ay, math.atan(-4 / fy))
        self.assertAlmostEqual(guide.angles(128, 0, 256, 192)[0], guide.HFOV / 2.0)
        self.assertEqual(guide.angles(12, -4, 0, 192), (0.0, 0.0))

    def test_a_live_sighting_is_a_landing_target(self) -> None:
        sent = []

        class _Uart(object):
            def write(self, data):
                sent.append(data)

        link = guide.Link()
        link.tried = True
        link.uart = _Uart()
        line = link.report(_Box(12, -4))
        self.assertEqual(line, "diff x=12 y=-4")
        self.assertEqual(len(sent), 1)
        packet = sent[0]
        self.assertEqual(packet[0], guide.MAV_STX)
        self.assertEqual(packet[1], 30)
        self.assertEqual(packet[5], guide.MAV_LANDING_TARGET)
        self.assertEqual(len(packet), 38)
        time_usec, ax, ay, dist, sx, sy, num, frame = struct.unpack("<QfffffBB", packet[6:36])
        self.assertEqual(time_usec, 0)
        self.assertGreater(ax, 0)
        self.assertLess(ay, 0)
        self.assertEqual(dist, 0.0)
        self.assertGreater(sx, 0)
        self.assertGreater(sy, 0)
        self.assertEqual(num, 0)
        self.assertEqual(frame, guide.FRAME_BODY_FRD)
        crc = 0xFFFF
        for byte in packet[1:36]:
            crc = _crc_byte(crc, byte)
        crc = _crc_byte(crc, guide.CRC_EXTRA)
        self.assertEqual(struct.unpack("<H", packet[36:38])[0], crc)

        link.report(_Box(1, 1))
        self.assertEqual(sent[1][2], 1)
        self.assertEqual(struct.unpack("<QfffffBB", sent[1][6:36])[7], guide.FRAME_BODY_FRD)

    def test_a_coast_and_an_unsighted_lock_send_nothing(self) -> None:
        sent = []

        class _Uart(object):
            def write(self, data):
                sent.append(data)

        link = guide.Link()
        link.tried = True
        link.uart = _Uart()
        self.assertIsNone(link.report(_Box(12, -4, center=False)))
        self.assertIsNone(link.report(_Box(12, -4, unclear=0)))
        self.assertEqual(sent, [])
        self.assertEqual(link.seq, 0)
