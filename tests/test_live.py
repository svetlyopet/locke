import io
import os
import struct
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout

from src import cues
from src import guide
from src.camera import kelvin64_to_c, load_t1c
from src.live import (
    _apply,
    _cpu_load,
    _note_meter,
    _note_temp,
    _note_state,
    _report,
    _tenth,
    aim_box,
    cpu_percent,
    debug_requested,
    keys,
    memory_percent,
    meter_line,
    temp_line,
    publish,
    run,
    scene_celsius,
)
from src.logic import (
    FAIL,
    KEEP_MS,
    LEARN_FRAMES,
    SMALL_SIDE,
    Appearance,
    Lock,
    active_floors,
    blob_threshold,
    heat_bounds,
    identity_score,
    is_clear,
    leading,
    masked_max,
    object_celsius,
    pattern_hot,
    readout,
    thermal_grid,
    resize_nearest,
    scaled_sizes,
    search_plain,
    search_scales,
    separate,
    shown_matches,
)
from track import invocations, parse


def _blob(width, height, value=20, box=None, blob=80):
    gray = [value] * (width * height)
    if box is not None:
        x, y, w, h = box
        yy = 0
        while yy < h:
            xx = 0
            while xx < w:
                gray[(y + yy) * width + (x + xx)] = blob
                xx += 1
            yy += 1
    return gray


def _aimed(width, height, fill, blob, value):
    """A rectangle whose edge mixes a hot window with a cool side.

    The edge spread is too wide for the surround test. `blob` is (x, y, w, h)
    of the only object inside.
    """
    gray = [fill] * (width * height)
    x = 0
    while x < width:
        gray[x] = 250
        gray[(height - 1) * width + x] = 30
        x += 1
    y = 0
    while y < height:
        gray[y * width] = 210
        gray[y * width + width - 1] = 30
        y += 1
    bx, by, bw, bh = blob
    yy = 0
    while yy < bh:
        xx = 0
        while xx < bw:
            gray[(by + yy) * width + (bx + xx)] = value
            xx += 1
        yy += 1
    return gray


class LiveTests(unittest.TestCase):
    def test_tracker_uses_the_src_t1c_module(self) -> None:
        inserted = "csi" not in sys.modules
        if inserted:
            csi = types.ModuleType("csi")
            csi.YUV422 = 0
            csi.GRAYSCALE = 1
            sys.modules["csi"] = csi
        try:
            t1c = load_t1c()
        finally:
            if inserted:
                del sys.modules["csi"]
        self.assertEqual(t1c.Y16_TEMP, 0)
        self.assertTrue(t1c.__file__.replace("\\", "/").endswith("src/t1c.py"))

    def test_temperature_counts_rise_with_celsius(self) -> None:
        counts = int((23.0 + 273.15) * 64)
        self.assertAlmostEqual(kelvin64_to_c(counts), 23.0, delta=0.02)

    def test_omvcon_view_sends_frames_and_leaves_the_console_alone(self) -> None:
        import src.live as live

        class Page(object):
            ACTIVE = True

            def __init__(self):
                self.sent = []

            def send(self, img):
                self.sent.append(img)

        class Frame(object):
            def __init__(self):
                self.flushed = 0

            def flush(self):
                self.flushed += 1

        page = Page()
        previous = live.omvview
        live.omvview = page
        try:
            frame = Frame()
            publish(frame)
            self.assertEqual(page.sent, [frame])
            self.assertEqual(frame.flushed, 0)
            self.assertEqual(keys(), [])
        finally:
            live.omvview = previous

    def test_service_stops_when_the_camera_is_absent(self) -> None:
        self.assertEqual(run(), 1)


class SeparationTests(unittest.TestCase):
    def test_largest_region_is_kept_and_a_flat_rectangle_is_refused(self) -> None:
        gray = _blob(32, 24, box=(4, 4, 12, 10), blob=90)
        yy = 0
        while yy < 6:
            xx = 0
            while xx < 8:
                gray[(4 + yy) * 32 + (18 + xx)] = 200
                xx += 1
            yy += 1
        found = separate(gray, 32, 24)
        self.assertIsNotNone(found)
        x, y, w, h, pattern = found[:5]
        self.assertEqual((x, y, w, h), (4, 4, 12, 10))
        self.assertEqual(pattern[0], 90)
        self.assertIsNone(separate(_blob(24, 20), 24, 20))

    def test_a_four_by_four_patch_uses_the_small_floor(self) -> None:
        gray = _blob(8, 8, value=20, box=(2, 2, 4, 4), blob=200)
        self.assertIsNone(heat_bounds(gray, 8, 8, 100, 255))
        found = heat_bounds(gray, 8, 8, 100, 255, SMALL_SIDE, 16)
        self.assertEqual(found[:4], (2, 2, 4, 4))
        self.assertEqual(active_floors(4, 4), (SMALL_SIDE, 16))
        self.assertEqual(active_floors(8, 8), (8, 32))
        self.assertEqual(scaled_sizes(4, 4, 32, 32), [(8, 8)])
        self.assertEqual(scaled_sizes(4, 4, 32, 32, SMALL_SIDE), [(4, 4)])

    def test_a_cool_center_stays_inside_the_silhouette(self) -> None:
        gray = _blob(32, 24, box=(4, 4, 12, 10), blob=90)
        yy = 0
        while yy < 4:
            xx = 0
            while xx < 4:
                gray[(7 + yy) * 32 + (7 + xx)] = 22
                xx += 1
            yy += 1
        found = separate(gray, 32, 24)
        self.assertIsNotNone(found)
        self.assertEqual(found[:4], (4, 4, 12, 10))
        self.assertEqual(found[4][3 * 12 + 3], 22)
        self.assertEqual(found[5][3 * 12 + 3], 1)

    def test_a_one_pixel_crack_is_one_object(self) -> None:
        gray = _blob(36, 20, box=(4, 4, 10, 10), blob=90)
        yy = 0
        while yy < 10:
            xx = 0
            while xx < 10:
                gray[(4 + yy) * 36 + (15 + xx)] = 90
                xx += 1
            yy += 1
        found = separate(gray, 36, 20)
        self.assertIsNotNone(found)
        self.assertEqual(found[:4], (4, 4, 21, 10))

    def test_the_shape_beats_a_hotter_rectangle(self) -> None:
        gray = _blob(28, 22, box=(4, 4, 6, 12), blob=90)
        yy = 0
        while yy < 6:
            xx = 0
            while xx < 14:
                gray[(10 + yy) * 28 + (4 + xx)] = 90
                xx += 1
            yy += 1
        found = separate(gray, 28, 22)
        self.assertIsNotNone(found)
        _x, _y, pw, ph, pattern, mask = found
        self.assertIn(0, mask)
        frame = [30] * (64 * 48)
        ox, oy = 20, 16
        y = 0
        while y < ph:
            x = 0
            while x < pw:
                if mask[y * pw + x]:
                    frame[(oy + y) * 64 + (ox + x)] = pattern[y * pw + x]
                x += 1
            y += 1
        y = 0
        while y < ph:
            x = 0
            while x < pw:
                frame[(4 + y) * 64 + (4 + x)] = 255
                x += 1
            y += 1
        hit = search_plain(frame, 64, 48, pattern, pw, ph, step=4, mask=mask)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], ox)
        self.assertEqual(hit[1], oy)
        self.assertGreaterEqual(hit[4], 90)
        self.assertLess(hit[5], hit[4])

    def test_only_matches_above_seventy_are_shown(self) -> None:
        matches = (
            (10, 12, 8, 8, 70),
            (30, 12, 8, 8, 71),
            (50, 12, 8, 8, 100),
            (70, 12, 8, 8, 50),
        )
        shown = shown_matches(matches)
        self.assertEqual([item[4] for item in shown], [71, 100])

    def test_readout_is_the_highest_confidence_box(self) -> None:
        matches = shown_matches(
            (
                (10, 12, 8, 8, 71),
                (136, 88, 8, 8, 84),
                (40, 40, 8, 8, 90),
                (70, 12, 8, 8, 50),
            )
        )
        best = leading(matches)
        self.assertEqual(best[4], 90)
        self.assertEqual(readout(best, 256, 192), (-84, -52, 90))
        self.assertEqual(readout(None, 256, 192), (0, 0, 0))
        tied = leading(((4, 4, 8, 8, 80), (20, 4, 8, 8, 80)))
        self.assertEqual(tied[0], 4)

    def test_clear_sighting_needs_fifty_and_a_unique_second(self) -> None:
        self.assertTrue(is_clear(50, 37))
        self.assertFalse(is_clear(50, 38))
        self.assertFalse(is_clear(49, 0))
        self.assertTrue(is_clear(80, 60))
        self.assertFalse(is_clear(80, 61))


class LockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "locke.target")
        self.lock = Lock((self.path,))
        self.lock.startup()
        self.lock.frame_w = 256
        self.lock.frame_h = 192

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _capture(self, box=(4, 4, 12, 10)):
        self.assertEqual(self.lock.select(), "arm")
        self.assertTrue(self.lock.place((10, 12, 24, 20)))
        gray = _blob(24, 20, box=box)
        return self.lock.lock(gray, 24, 20)

    def test_select_only_arms_and_lock_needs_a_rectangle(self) -> None:
        self.assertEqual(self.lock.lock(_blob(24, 20), 24, 20), "idle")
        self.assertIsNone(self.lock.pattern)
        self.assertEqual(self.lock.select(), "arm")
        self.assertTrue(self.lock.place((10, 12, 24, 20)))
        self.assertEqual(self.lock.select(), "arm")
        self.assertIsNone(self.lock.placement)
        self.assertIsNone(self.lock.pattern)
        self.assertTrue(self.lock.drawing)

    def test_the_center_box_is_forty_eight_and_clamps_to_the_frame(self) -> None:
        self.assertEqual(aim_box(256, 192), (104, 72, 48, 48))
        self.assertEqual(aim_box(30, 20), (0, 0, 30, 20))
        self.assertEqual(aim_box(30, 60), (0, 6, 30, 48))
        self.assertIsNone(aim_box(0, 192))

    def test_lock_without_a_rectangle_uses_the_center_box(self) -> None:
        frame = _Frame(256, 192)
        _paint(frame, (122, 91, 12, 10), 80)
        _apply(self.lock, frame, ("lock",))
        self.assertEqual(self.lock.lock_box, (122, 91, 12, 10))
        self.assertIsNotNone(self.lock.pattern)
        self.assertIsNone(self.lock.placement)
        self.assertFalse(self.lock.drawing)

    def test_a_dragged_rectangle_overrides_the_center_box(self) -> None:
        frame = _Frame(256, 192)
        _paint(frame, (14, 16, 12, 10), 80)
        self.assertEqual(self.lock.select(), "arm")
        self.assertTrue(self.lock.place((10, 12, 24, 20)))
        _apply(self.lock, frame, ("lock",))
        self.assertEqual(self.lock.lock_box, (14, 16, 12, 10))

    def test_drag_is_ignored_until_select_and_the_file_is_loaded(self) -> None:
        self.assertFalse(self.lock.place((10, 12, 24, 20)))
        self.assertEqual(self._capture(), "ok")
        self.assertTrue(os.path.isfile(self.path))
        self.assertIsNotNone(self.lock.pattern)
        self.assertIsNone(self.lock.placement)
        self.assertFalse(self.lock.drawing)
        other = Lock((self.path,))
        self.assertTrue(other._read())
        self.assertEqual(other.pattern, self.lock.pattern)

    def test_a_flat_rectangle_is_refused_and_the_old_object_stays(self) -> None:
        self.assertEqual(self._capture(), "ok")
        pattern = list(self.lock.pattern)
        self.assertEqual(self.lock.select(), "arm")
        self.assertTrue(self.lock.place((10, 12, 24, 20)))
        self.assertEqual(self.lock.lock(_blob(24, 20), 24, 20), "fail")
        self.assertEqual(self.lock.message, FAIL)
        self.assertEqual(self.lock.pattern, pattern)
        self.assertIsNotNone(self.lock.placement)
        self.assertTrue(self.lock.place((11, 12, 24, 20)))
        self.assertIsNone(self.lock.message)

    def test_a_saved_lock_reports_its_offset_and_follows_a_weak_blob(self) -> None:
        self.assertEqual(self._capture(), "ok")
        self.assertIsNone(self.lock.message)
        saved = (14, 16, 12, 10)
        # Middle of the saved box against the middle of the 256x192 frame.
        offset = (20 - 128, 21 - 96, 0)
        self.assertEqual(self.lock.lock_box, saved)
        self.assertTrue(self.lock.show_lock)
        self.assertIsNone(self.lock.center)
        self.assertEqual(self.lock.line, offset)
        self.assertEqual(self.lock.observe(0, None), offset)
        self.assertTrue(self.lock.show_lock)
        self.assertEqual(self.lock.lock_box, saved)
        self.assertIsNone(self.lock.center)
        weak = (saved[0] + 6, saved[1] + 4, saved[2], saved[3], 40, 35)
        moved = self.lock.observe(100, weak)
        self.assertEqual(self.lock.lock_box, weak[:4])
        self.assertEqual(self.lock.center, (saved[0] + 6 + 6, saved[1] + 4 + 5))
        self.assertEqual(moved[:2], (self.lock.center[0] - 128, self.lock.center[1] - 96))
        self.assertIsNone(self.lock.unclear_since)
        self.assertEqual(self.lock.appearance.samples, 0)

    def test_a_centered_object_locks_when_the_edge_is_mixed(self) -> None:
        # The window sits on the top and left edge, so the surround test
        # cannot separate anything. The object itself is centered in the box.
        hot = _aimed(24, 20, 60, (7, 5, 10, 10), 240)
        self.assertIsNone(separate(hot, 24, 20))
        self.assertEqual(self.lock.select(), "arm")
        self.assertTrue(self.lock.place((10, 12, 24, 20)))
        self.assertEqual(self.lock.lock(hot, 24, 20), "ok")
        self.assertIsNone(self.lock.message)
        self.assertEqual((self.lock.pattern_w, self.lock.pattern_h), (10, 10))
        self.assertEqual(self.lock.search_at, (22, 22))
        self.assertEqual(self.lock.pattern[5 * 10 + 5], 240)
        self.assertEqual(self.lock.mask[5 * 10 + 5], 1)
        self.assertTrue(self.lock.appearance.hot)

        cool = _aimed(24, 20, 180, (7, 5, 10, 10), 40)
        self.assertIsNone(separate(cool, 24, 20))
        self.assertEqual(self.lock.select(), "arm")
        self.assertTrue(self.lock.place((10, 12, 24, 20)))
        self.assertEqual(self.lock.lock(cool, 24, 20), "ok")
        self.assertEqual((self.lock.pattern_w, self.lock.pattern_h), (10, 10))
        self.assertEqual(self.lock.search_at, (22, 22))
        self.assertEqual(self.lock.pattern[5 * 10 + 5], 40)
        self.assertFalse(self.lock.appearance.hot)

    def test_an_off_center_object_stays_refused_when_the_edge_is_mixed(self) -> None:
        self.assertEqual(self._capture(), "ok")
        pattern = list(self.lock.pattern)
        corner = _aimed(24, 20, 60, (2, 2, 8, 8), 240)
        self.assertIsNone(separate(corner, 24, 20))
        self.assertEqual(self.lock.select(), "arm")
        self.assertTrue(self.lock.place((10, 12, 24, 20)))
        self.assertEqual(self.lock.lock(corner, 24, 20), "fail")
        self.assertEqual(self.lock.message, FAIL)
        self.assertEqual(self.lock.pattern, pattern)

    def test_console_line_decays_hides_and_resumes(self) -> None:
        self.assertEqual(self._capture(), "ok")
        base = _sight_score(80, 10, 8, 8, self.lock.pattern_w, self.lock.pattern_h)
        self.assertEqual(self.lock.observe(0, (136, 88, 8, 8, 80, 10)), (12, -4, base))
        fields = self.lock.cue_line.split()
        self.assertEqual(fields[0], "appearance=80")
        self.assertEqual(fields[2], "unique=88")
        self.assertEqual(fields[4], "motion=na")
        shown_size = "size=%d" % cues.size(8, 8, self.lock.pattern_w, self.lock.pattern_h)
        self.assertEqual(fields[6], shown_size)
        self.assertEqual(fields[8], "temperature=na")
        self.assertEqual(fields[10], "anchor=na")
        self.assertEqual(fields[12], "hottest=na")
        self.assertTrue(fields[1].startswith("appearance_us="))
        self.assertTrue(fields[1][len("appearance_us="):].isdigit())
        self.assertTrue(self.lock.show_lock)
        # A similar neighbor keeps the box on this heat and does not start the hide.
        contested = self.lock.observe(0, (136, 88, 8, 8, 80, 70))
        self.assertEqual(contested[:2], (12, -4))
        self.assertIsNone(self.lock.unclear_since)
        self.assertEqual(self.lock.lock_box, (136, 88, 8, 8))
        held = contested[2]
        self.assertEqual(self.lock.observe(0, None), (12, -4, held))
        self.assertEqual(self.lock.observe(2500, None), (12, -4, held * (KEEP_MS - 2500) // KEEP_MS))
        self.assertTrue(self.lock.show_lock)
        self.lock.note_shutter(3000)
        self.assertEqual(self.lock.line, (12, -4, held * (KEEP_MS - 2500) // KEEP_MS))
        # The shutter gap from 3000 to 4000 is removed, so this counts as 3000 ms.
        self.assertEqual(self.lock.observe(4000, None), (12, -4, held * (KEEP_MS - 3000) // KEEP_MS))
        self.assertEqual(self.lock.status(), "COASTING")
        # The shutter moved unclear_since to 1000, so the wait ends there plus KEEP_MS.
        lost_at = 1000 + KEEP_MS
        self.assertEqual(self.lock.observe(lost_at, None), (0, 0, 0))
        self.assertEqual(self.lock.status(), "LOST")
        self.assertFalse(self.lock.show_lock)
        self.assertIsNotNone(self.lock.pattern)
        self.assertEqual(
            self.lock.observe(lost_at + 500, (136, 88, 8, 8, 90, 10)),
            (12, -4, _sight_score(90, 10, 8, 8, 8, 8)),
        )
        self.assertEqual(self.lock.status(), "LOCKED")
        self.assertTrue(self.lock.show_lock)

    def test_motion_keeps_the_box_around_the_object(self) -> None:
        # A dip under 50 still follows the heat, but only the drawn box moves.
        # The next frames of that same motion have to keep encompassing it.
        from src.live import _window

        self.assertEqual(self._capture(), "ok")
        width = self.lock.pattern_w
        height = self.lock.pattern_h
        x = 30
        y = 40
        index = 0
        while index < 4:
            self.lock.observe(index * 50, (x, y, width, height, 90, 0))
            x += 8
            index += 1
        index = 0
        while index < 6:
            self.lock.observe(1000 + index * 50, (x, y, width, height, 40, 0))
            self.assertEqual(self.lock.lock_box, (x, y, width, height))
            roi = _window(
                self.lock.aim(),
                width,
                height,
                self.lock.search_margin(),
                self.lock.frame_w,
                self.lock.frame_h,
            )
            self.assertLessEqual(roi[0], x)
            self.assertLessEqual(roi[1], y)
            self.assertGreaterEqual(roi[0] + roi[2], x + width)
            self.assertGreaterEqual(roi[1] + roi[3], y + height)
            x += 8
            index += 1

    def test_a_camera_shift_on_one_axis_does_not_drag_the_other(self) -> None:
        # Horizontal velocity is already learned. A vertical camera shift
        # must not splice that velocity into a diagonal the object never had.
        from src.live import _window

        self.assertEqual(self._capture(), "ok")
        width = self.lock.pattern_w
        height = self.lock.pattern_h
        x = 80
        y = 60
        index = 0
        while index < 4:
            self.lock.observe(index * 40, (x, y, width, height, 90, 0))
            x += 8
            index += 1
        pitched = (x - 8, y + 10, width, height, 40, 0)
        self.lock.observe(200, pitched)
        aim = self.lock.aim()
        cx = pitched[0] + width // 2
        cy = pitched[1] + height // 2
        self.assertEqual(aim[0], cx)
        self.assertGreaterEqual(aim[1], cy)
        roi = _window(
            aim,
            width,
            height,
            self.lock.search_margin(),
            self.lock.frame_w,
            self.lock.frame_h,
        )
        self.assertLessEqual(roi[0], pitched[0])
        self.assertLessEqual(roi[1], pitched[1])
        self.assertGreaterEqual(roi[0] + roi[2], pitched[0] + width)
        self.assertGreaterEqual(roi[1] + roi[3], pitched[1] + height)

    def test_a_stopped_camera_drops_a_negative_shift(self) -> None:
        # Floor division used to pin a left/up shift at -1 px/frame, so the
        # next miss walked the box off after the camera had already stopped.
        self.assertEqual(self._capture(), "ok")
        width = self.lock.pattern_w
        height = self.lock.pattern_h
        self.lock.observe(0, (90, 70, width, height, 90, 0))
        self.lock.observe(40, (84, 66, width, height, 90, 0))
        held = (84, 66, width, height)
        stopped = held + (90, 0)
        self.lock.observe(80, stopped)
        self.assertNotEqual((self.lock.appearance.vx, self.lock.appearance.vy), (0, 0))
        self.lock.observe(120, None)
        self.assertEqual(self.lock.lock_box, held)
        self.assertEqual(self.lock.center, (held[0] + width // 2, held[1] + height // 2))
        index = 0
        while index < 6:
            self.lock.observe(160 + index * 40, stopped)
            index += 1
        self.assertEqual((self.lock.appearance.vx, self.lock.appearance.vy), (0, 0))
        self.assertEqual(self.lock.aim(), self.lock.center)

    def test_a_short_dip_keeps_the_box_on_the_heat(self) -> None:
        self.assertEqual(self._capture(), "ok")
        self.lock.observe(0, (136, 88, 8, 8, 90, 0))
        pattern = list(self.lock.pattern)
        mask = list(self.lock.mask)
        samples = self.lock.appearance.samples
        held = (self.lock.appearance.last_x, self.lock.appearance.last_y)
        velocity = (self.lock.appearance.vx, self.lock.appearance.vy)
        dipped = (144, 88, 8, 8, 40, 0)
        # Eight pixels from the last center, inside the 16-pixel gate.
        line = self.lock.observe(200, dipped)
        self.assertEqual(self.lock.lock_box, dipped[:4])
        self.assertTrue(self.lock.show_lock)
        self.assertIsNone(self.lock.unclear_since)
        self.assertEqual(list(self.lock.pattern), pattern)
        self.assertEqual(list(self.lock.mask), mask)
        self.assertEqual(self.lock.appearance.samples, samples)
        self.assertEqual((self.lock.appearance.last_x, self.lock.appearance.last_y), held)
        self.assertEqual((self.lock.appearance.vx, self.lock.appearance.vy), velocity)
        self.assertEqual(line, (20, -4, 76))
        self.assertEqual(self.lock.observe(400, None)[:2], (20, -4))
        self.assertTrue(self.lock.show_lock)
        self.assertEqual(self.lock.lock_box, dipped[:4])

    def test_the_block_starts_on_the_selection_and_follows(self) -> None:
        from src.live import _window

        self.assertEqual(self._capture(), "ok")
        self.assertEqual(self.lock.search_margin(), 16)
        roi = _window(
            self.lock.search_at,
            self.lock.pattern_w,
            self.lock.pattern_h,
            self.lock.search_margin(),
            self.lock.frame_w,
            self.lock.frame_h,
        )
        cx, cy = self.lock.search_at
        self.assertLessEqual(roi[0], cx)
        self.assertLess(cx, roi[0] + roi[2])
        self.assertLessEqual(roi[1], cy)
        self.assertLess(cy, roi[1] + roi[3])
        self.assertEqual(roi[2], self.lock.pattern_w + 32)
        self.assertEqual(roi[3], self.lock.pattern_h + 32)
        self.assertNotEqual(roi, (0, 0, self.lock.frame_w, self.lock.frame_h))
        width = self.lock.pattern_w
        height = self.lock.pattern_h
        self.lock.observe(0, (30, 20, width, height, 90, 0))
        self.lock.observe(20, (42, 20, width, height, 90, 0))
        self.assertEqual(self.lock.predict(), (42 + width // 2 + 12, 20 + height // 2))
        self.assertEqual(self.lock.search_margin(), 28)

    def test_a_far_sighting_is_refused_before_the_record_freezes(self) -> None:
        self.assertEqual(self._capture(), "ok")
        width = self.lock.pattern_w
        height = self.lock.pattern_h
        self.lock.observe(0, (30, 20, width, height, 90, 0))
        box = self.lock.lock_box
        self.lock.observe(20, (180, 140, width, height, 90, 0))
        self.assertEqual(self.lock.lock_box, box)

    def test_a_far_sighting_is_not_the_object(self) -> None:
        self.assertEqual(self._capture(), "ok")
        appearance = self.lock.appearance
        appearance.samples = LEARN_FRAMES
        appearance.see(40, 40)
        self.assertFalse(appearance.same(appearance.box_w, appearance.box_h, None, 220, 40, True))
        self.assertTrue(appearance.same(appearance.box_w, appearance.box_h, None, 40, 40, False))

    def test_a_warming_background_coasts_without_flipping(self) -> None:
        from src.logic import contrast_band

        self.assertEqual(self._capture(), "ok")
        appearance = self.lock.appearance
        self.assertTrue(appearance.hot)
        self.assertIsNotNone(
            contrast_band(appearance.obj_mean, appearance.bg_mean, True, appearance.bg_mean, 0)
        )
        appearance.note_contrast(appearance.obj_mean, appearance.obj_mean - 5)
        self.assertTrue(appearance.hot)
        self.assertIsNone(
            contrast_band(appearance.obj_mean, appearance.bg_mean, appearance.hot, appearance.bg_mean, 0)
        )

    def test_a_miss_grows_the_pad_until_the_cap(self) -> None:
        self.assertEqual(self._capture(), "ok")
        width = self.lock.pattern_w
        height = self.lock.pattern_h
        self.lock.observe(0, (20, 16, width, height, 90, 0))
        self.assertEqual(self.lock.search_margin(), 16)
        self.lock.observe(10, None)
        self.assertEqual(self.lock.misses, 1)
        self.assertEqual(self.lock.search_margin(), 20)
        self.lock.misses = 100
        self.assertEqual(self.lock.search_margin(), 48)
        self.lock.appearance.vx = 40
        self.assertEqual(self.lock.search_margin(), 48)

    def test_a_frozen_mask_is_not_segmented_again(self) -> None:
        from src import logic

        self.assertEqual(self._capture(), "ok")
        width = self.lock.pattern_w
        height = self.lock.pattern_h
        crop = [90] * (width * height)
        index = 0
        while index < LEARN_FRAMES:
            self.lock.observe(index, (20, 16, width, height, 90, 0), crop=crop)
            index += 1
        original = logic.separate

        def explode(*_args, **_kwargs):
            raise AssertionError("separate")

        logic.separate = explode
        try:
            self.lock.observe(LEARN_FRAMES, (20, 16, width, height, 90, 0), crop=crop)
        finally:
            logic.separate = original

    def test_a_lone_sighting_abstains_uniqueness_and_the_frozen_pattern_stays(self) -> None:
        self.assertEqual(self._capture(), "ok")
        frozen = list(self.lock.frozen)
        self.lock.pattern[0] = 3 if frozen[0] != 3 else 4
        self.assertEqual(self.lock.frozen[0], frozen[0])
        self.assertEqual(self.lock.search_at, (20, 21))
        line = self.lock.observe(0, (136, 88, 8, 8, 80, None))
        voted = (
            cues.Cue("appearance", 80, 0),
            cues.Cue("size", cues.size(8, 8, self.lock.pattern_w, self.lock.pattern_h), 0),
            cues.Cue("sole", 100, 0),
        )
        self.assertEqual(line, (12, -4, cues.with_sole(cues.combine(voted), 100)))
        self.assertIn("unique=na", self.lock.cue_line)
        self.assertIn("sole=100", self.lock.cue_line)

    def test_the_only_warm_object_in_view_is_certain(self) -> None:
        # A lone sighting that never left is the same object. A second warm
        # object, or a return from a hide, keeps the ordinary average.
        self.assertEqual(self._capture(), "ok")
        width = self.lock.pattern_w
        height = self.lock.pattern_h
        self.lock.observe(0, (30, 40, width, height, 90, None))
        self.lock.observe(40, (38, 40, width, height, 90, None))
        held = self.lock.observe(80, (30, 40, width, height, 40, None))
        self.assertIn("sole=100", self.lock.cue_line)
        self.assertGreaterEqual(held[2], cues.SOLE_SURE)
        paired = self.lock.observe(120, (30, 40, width, height, 40, 10))
        self.assertIn("sole=na", self.lock.cue_line)
        self.assertEqual(paired[2], cues.combine(self.lock.cues))
        self.lock.observe(160, None)
        hidden = self.lock.observe(200, (30, 40, width, height, 40, None))
        self.assertIn("sole=na", self.lock.cue_line)
        self.assertEqual(hidden[2], cues.combine(self.lock.cues))
        again = self.lock.observe(240, (30, 40, width, height, 40, None))
        self.assertIn("sole=100", self.lock.cue_line)
        self.assertGreaterEqual(again[2], cues.SOLE_SURE)

    def test_the_frame_maximum_inside_the_object_raises_confidence(self) -> None:
        self.assertEqual(self._capture(), "ok")
        plain = _sight_score(80, 10, 8, 8, self.lock.pattern_w, self.lock.pattern_h)
        held = _sight_score(80, 10, 8, 8, self.lock.pattern_w, self.lock.pattern_h, 90, 90)
        self.assertGreater(held, plain)
        line = self.lock.observe(0, (136, 88, 8, 8, 80, 10), frame_max=90, obj_max=90)
        self.assertEqual(line, (12, -4, held))
        self.assertIn("hottest=100", self.lock.cue_line)

    def test_startup_and_unlock_discard_the_file(self) -> None:
        self.assertEqual(self._capture(), "ok")
        self.lock.unlock()
        self.assertFalse(os.path.isfile(self.path))
        self.assertEqual(self.lock.line, (0, 0, 0))
        self.assertEqual(self._capture(), "ok")
        self.lock.startup()
        self.assertFalse(os.path.isfile(self.path))
        self.assertEqual(self.lock.observe(0, None), (0, 0, 0))

    def test_pattern_beats_a_hotter_flat_patch(self) -> None:
        pattern = []
        index = 0
        while index < 64:
            pattern.append((index * 17) % 180)
            index += 1
        frame = [30] * (48 * 36)
        ox, oy = 16, 12
        y = 0
        while y < 8:
            x = 0
            while x < 8:
                frame[(oy + y) * 48 + (ox + x)] = pattern[y * 8 + x]
                frame[(4 + y) * 48 + (4 + x)] = 255
                x += 1
            y += 1
        found = search_plain(frame, 48, 36, pattern, 8, 8, step=4)
        self.assertIsNotNone(found)
        self.assertEqual(found[0], ox)
        self.assertEqual(found[1], oy)
        self.assertGreaterEqual(found[4], 90)
        self.assertLess(found[5], found[4])

    def test_a_hide_walks_the_box_and_then_reports_lost(self) -> None:
        self.assertEqual(self.lock.status(), "NA")
        self.assertEqual(self._capture(), "ok")
        self.assertEqual(self.lock.status(), "LOCKED")
        width = self.lock.pattern_w
        height = self.lock.pattern_h
        self.lock.observe(0, (40, 30, width, height, 90, 0))
        self.lock.observe(40, (52, 36, width, height, 90, 0))
        self.assertEqual((self.lock.appearance.vx, self.lock.appearance.vy), (12, 6))
        box = self.lock.lock_box
        center = self.lock.center
        self.lock.observe(80, None)
        self.assertEqual(self.lock.status(), "COASTING")
        self.assertEqual(self.lock.lock_box, (box[0] + 12, box[1] + 6, box[2], box[3]))
        self.assertEqual(self.lock.center, (center[0] + 12, center[1] + 6))
        self.assertEqual(
            self.lock.line[:2], (self.lock._dx(center[0] + 12), self.lock._dy(center[1] + 6))
        )
        self.assertEqual((self.lock.appearance.vx, self.lock.appearance.vy), (12, 6))
        self.lock.observe(120, None)
        self.assertEqual(self.lock.center, (center[0] + 24, center[1] + 12))
        self.assertEqual((self.lock.appearance.vx, self.lock.appearance.vy), (12, 6))
        self.assertEqual(self.lock.observe(80 + KEEP_MS, None), (0, 0, 0))
        self.assertEqual(self.lock.status(), "LOST")
        self.assertFalse(self.lock.show_lock)
        self.assertEqual(self.lock.center, (center[0] + 24, center[1] + 12))
        found = (center[0] + 24 - width // 2, center[1] + 12 - height // 2, width, height, 90, 0)
        self.lock.observe(80 + KEEP_MS + 40, found)
        self.assertEqual(self.lock.status(), "LOCKED")
        self.assertEqual(self.lock.lock_box, found[:4])
        self.lock.unlock()
        self.assertEqual(self.lock.status(), "NA")

    def _coast(self):
        self.assertEqual(self._capture(), "ok")
        width = self.lock.pattern_w
        height = self.lock.pattern_h
        self.lock.observe(0, (40, 30, width, height, 90, 0))
        self.lock.observe(40, (52, 36, width, height, 90, 0))
        self.lock.observe(80, None)
        self.assertEqual(self.lock.status(), "COASTING")
        self.assertEqual((self.lock.appearance.vx, self.lock.appearance.vy), (12, 6))
        return width, height

    def test_a_frame_sized_blob_does_not_become_the_coasting_box(self) -> None:
        width, height = self._coast()
        held = self.lock.lock_box
        center = self.lock.center
        huge = (center[0] - 100, center[1] - 80, 200, 160, 40, 0)
        self.lock.observe(120, huge)
        self.assertEqual(self.lock.status(), "COASTING")
        self.assertEqual(self.lock.lock_box[2:], (width, height))
        self.assertNotEqual(self.lock.lock_box[2:], (200, 160))
        self.assertEqual(self.lock.lock_box[2:], held[2:])

    def _window_search(self, frame):
        from src.live import _search

        appearance = self.lock.appearance
        tracked = (appearance.obj_mean, appearance.bg_mean, appearance.hot)
        return _search(
            frame,
            self.lock.pattern,
            self.lock.pattern_w,
            self.lock.pattern_h,
            self.lock.mask,
            self.lock.aim(),
            self.lock.search_margin(),
            True,
            appearance.hot,
            tracked,
            appearance.moments,
            self.lock.frozen,
            self.lock.frozen_mask,
            self.lock.frozen_moments,
            appearance,
        )

    def test_a_hot_patch_in_the_window_is_not_the_coast_target(self) -> None:
        from src.live import _take_back

        width, height = self._coast()
        held = self.lock.lock_box
        frame = _Frame(256, 192)
        hot = (held[0] + 4, held[1], width, height)
        _paint(frame, hot, 255)
        frame.blobs = [hot + (width * height,)]
        match, found, _recovered, border, _packed = self._window_search(frame)
        self.assertIsNotNone(match)
        self.assertEqual(match[:2], hot[:2])
        self.lock.border_mean = border
        self.assertIsNone(_take_back(frame, self.lock, found, self.lock.unclear_since))
        self.lock.observe(120, None)
        self.assertEqual(self.lock.status(), "COASTING")
        self.assertEqual(self.lock.lock_box, (held[0] + 12, held[1] + 6, width, height))

    def test_the_stored_pattern_ends_a_coast(self) -> None:
        from src.live import _take_back

        width, height = self._coast()
        held = self.lock.lock_box
        frame = _Frame(256, 192)
        x, y = held[0] + 6, held[1]
        yy = 0
        while yy < height:
            xx = 0
            while xx < width:
                frame.pixels[(y + yy) * frame.w + (x + xx)] = self.lock.pattern[yy * width + xx]
                xx += 1
            yy += 1
        frame.blobs = [(x, y, width, height, width * height)]
        _match, found, _recovered, border, _packed = self._window_search(frame)
        self.lock.border_mean = border
        taken = _take_back(frame, self.lock, found, self.lock.unclear_since)
        self.assertIsNotNone(taken)
        self.lock.observe(120, taken, reclaim=True)
        self.assertEqual(self.lock.status(), "LOCKED")
        self.assertEqual(self.lock.lock_box[:2], (x, y))

    def test_a_matching_object_outside_the_gate_ends_the_coast(self) -> None:
        from src.live import _take_back

        width, height = self._coast()
        frame = _Frame(256, 192)
        found = [(180, 120, width, height, 80)]
        taken = _take_back(frame, self.lock, found)
        self.assertIsNotNone(taken)
        self.assertEqual(taken[:4], (180, 120, width, height))
        self.lock.observe(120, taken, reclaim=True)
        self.assertEqual(self.lock.status(), "LOCKED")
        self.assertEqual(self.lock.lock_box, (180, 120, width, height))
        self.assertEqual((self.lock.step_x, self.lock.step_y), (0, 0))
        self.assertEqual((self.lock.appearance.vx, self.lock.appearance.vy), (12, 6))
        self.assertEqual(self.lock.appearance.predict(), (180 + width // 2 + 12, 120 + height // 2 + 6))

    def test_a_far_sighting_without_the_shape_keeps_coasting(self) -> None:
        from src.live import _take_back

        width, height = self._coast()
        held = self.lock.lock_box
        frame = _Frame(256, 192)
        self.assertIsNone(_take_back(frame, self.lock, [(180, 120, width, height, 10)]))
        self.assertIsNone(_take_back(frame, self.lock, [(0, 0, 200, 160, 90)]))
        self.lock.observe(120, (180, 120, width, height, 80, 0))
        self.assertEqual(self.lock.status(), "COASTING")
        self.assertEqual(self.lock.lock_box[2:], held[2:])

    def test_two_seconds_of_coast_searches_the_frame_for_the_signature(self) -> None:
        from src.live import _take_back

        width, height = self._coast()
        frame = _Frame(256, 192)
        x, y = 180, 120
        yy = 0
        while yy < height:
            xx = 0
            while xx < width:
                frame.pixels[(y + yy) * frame.w + (x + xx)] = self.lock.pattern[yy * width + xx]
                xx += 1
            yy += 1
        frame.blobs = [(x, y, width, height, width * height)]
        started = self.lock.unclear_since
        self.assertIsNone(_take_back(frame, self.lock, [], started + 1999))
        self.assertEqual(frame.blob_calls, 0)
        flat = _Frame(256, 192)
        flat.blobs = [(x, y, width, height, width * height)]
        _paint(flat, (x, y, width, height), 200)
        self.assertIsNone(_take_back(flat, self.lock, [], started + 2000))
        taken = _take_back(frame, self.lock, [], started + 2000)
        self.assertIsNotNone(taken)
        self.assertEqual(taken[:4], (x, y, width, height))
        self.lock.observe(started + 2000, taken, reclaim=True)
        self.assertEqual(self.lock.status(), "LOCKED")
        self.assertEqual(self.lock.lock_box, (x, y, width, height))
        self.assertGreater(self.lock.line[2], 0)

    def test_identity_needs_size_shape_and_heat(self) -> None:
        self.assertIsNone(identity_score(12, 10, 12, 10, 80, 40))
        self.assertIsNone(identity_score(12, 10, 12, 10, 40, 80))
        self.assertIsNone(identity_score(40, 10, 12, 10, 80, 80))
        self.assertEqual(identity_score(12, 10, 12, 10, None, 80), 90)
        self.assertEqual(identity_score(12, 10, 12, 10, 80, 80), 86)

    def test_a_remembered_distractor_does_not_take_the_box(self) -> None:
        self.assertEqual(self._capture(), "ok")
        width = self.lock.pattern_w
        height = self.lock.pattern_h
        crop = list(self.lock.pattern)
        self.lock.observe(0, (30, 20, width, height, 90, 0), crop=crop)
        center = self.lock.center
        held = self.lock.lock_box
        dx = 40
        box = (center[0] + dx - width // 2, center[1] - height // 2, width, height)
        self.lock.note_distractor(box, 200)
        flat = [200] * (width * height)
        self.lock.observe(100, box + (80, 0), crop=flat)
        self.assertEqual(self.lock.lock_box, held)
        self.lock.observe(200, held + (90, 0), crop=crop)
        self.assertEqual(self.lock.lock_box, held)

    def test_a_flat_blob_does_not_replace_a_learned_core(self) -> None:
        self.assertEqual(self._capture(), "ok")
        width = self.lock.pattern_w
        height = self.lock.pattern_h
        core = [40] * (width * height)
        y = 0
        while y < height // 2:
            x = 0
            while x < width // 2:
                core[(height // 4 + y) * width + (width // 4 + x)] = 220
                x += 1
            y += 1
        index = 0
        while index < LEARN_FRAMES:
            self.lock.observe(index, (20, 16, width, height, 90, 0), crop=list(core), temp_c=25.0)
            index += 1
        self.assertIsNotNone(self.lock.appearance.grid)
        held = list(self.lock.pattern)
        flat = [180] * (width * height)
        self.lock.observe(LEARN_FRAMES, (20, 16, width, height, 90, 0), crop=flat, temp_c=25.0)
        self.assertEqual(list(self.lock.pattern), held)
        self.assertIn("temperature=0", self.lock.cue_line)

    def test_a_lost_target_locks_again_when_it_reappears_outside_the_gate(self) -> None:
        self.assertEqual(self._capture(), "ok")
        width = self.lock.pattern_w
        height = self.lock.pattern_h
        self.lock.observe(0, (40, 30, width, height, 90, 0))
        box = self.lock.lock_box
        reach = self.lock.search_margin()
        far = (40 + reach + 8, 30, width, height, 81, 0)
        self.lock.observe(1000, far)
        self.assertEqual(self.lock.status(), "COASTING")
        self.assertEqual(self.lock.lock_box, box)
        self.assertEqual(self.lock.observe(1000 + KEEP_MS, None), (0, 0, 0))
        self.assertEqual(self.lock.status(), "LOST")
        cx = far[0] + width // 2
        cy = far[1] + height // 2
        dx = cx - self.lock.center[0]
        dy = cy - self.lock.center[1]
        self.assertGreater(dx * dx + dy * dy, self.lock.search_margin() * self.lock.search_margin())
        resumed = self.lock.observe(1000 + KEEP_MS + 40, far)
        self.assertEqual(self.lock.status(), "LOCKED")
        self.assertTrue(self.lock.show_lock)
        self.assertIsNone(self.lock.unclear_since)
        self.assertEqual(self.lock.lock_box, far[:4])
        self.assertEqual(resumed[:2], (self.lock._dx(cx), self.lock._dy(cy)))

    def _arm_box(self, gray, width, height, spots):
        self.assertEqual(self.lock.select(), "arm")
        self.assertTrue(self.lock.place((10, 12, width, height)))
        return self.lock.lock(gray, width, height, spots)

    def test_a_sparse_frame_locks_a_four_by_four_target(self) -> None:
        small = _blob(12, 12, value=20, box=(4, 4, 4, 4), blob=200)
        tiny = _blob(12, 12, value=20, box=(4, 4, 3, 3), blob=200)
        wide = _blob(16, 16, value=20, box=(4, 4, 8, 8), blob=200)
        self.assertEqual(self._arm_box(small, 12, 12, 3), "fail")
        self.assertEqual(self.lock.message, FAIL)
        self.assertEqual(self._arm_box(small, 12, 12, None), "fail")
        self.assertEqual(self._arm_box(tiny, 12, 12, 1), "fail")
        self.assertEqual(self._arm_box(small, 12, 12, 1), "ok")
        self.assertEqual((self.lock.pattern_w, self.lock.pattern_h), (4, 4))
        self.assertTrue(self.lock.distant)
        self.lock.unlock()
        self.assertEqual(self._arm_box(wide, 16, 16, 5), "ok")
        self.assertEqual((self.lock.pattern_w, self.lock.pattern_h), (8, 8))
        self.assertFalse(self.lock.distant)

    def test_a_distant_lock_adopts_a_larger_shape_every_four_frames(self) -> None:
        small = _blob(12, 12, value=20, box=(4, 4, 4, 4), blob=200)
        self.assertEqual(self._arm_box(small, 12, 12, 1), "ok")
        anchor = list(self.lock.frozen)
        anchor_mask = list(self.lock.frozen_mask)
        crop = [200] * (8 * 8)
        index = 0
        while index < 4:
            self.lock.observe(index, (12, 14, 8, 8, 80, 0), crop=crop)
            index += 1
        self.assertEqual((self.lock.pattern_w, self.lock.pattern_h), (8, 8))
        self.assertEqual(len(self.lock.pattern), 64)
        self.assertEqual(self.lock.frozen, anchor)
        self.assertEqual(self.lock.frozen_mask, anchor_mask)
        self.assertEqual(len(anchor), 16)
        handle = open(self.path, "rb")
        try:
            data = handle.read()
        finally:
            handle.close()
        self.assertEqual(data.split(b"\n", 3)[2], b"4 4")

    def test_a_small_sighting_sends_a_landing_target(self) -> None:
        small = _blob(12, 12, value=20, box=(4, 4, 4, 4), blob=200)
        self.assertEqual(self._arm_box(small, 12, 12, 1), "ok")
        box = self.lock.lock_box
        crop = [200] * (box[2] * box[3])
        self.lock.observe(1, (box[0], box[1], box[2], box[3], 80, 0), crop=crop)
        sent = []

        class _Uart(object):
            def write(self, data):
                sent.append(data)

        link = guide.Link()
        link.tried = True
        link.uart = _Uart()
        line = link.report(self.lock)
        self.assertTrue(line.startswith("diff "))
        self.assertEqual(len(sent), 1)
        _time, _ax, _ay, dist, sx, sy, _num, _frame = struct.unpack("<QfffffBB", sent[0][6:36])
        self.assertEqual(dist, 0.0)
        self.assertGreater(sx, 0)
        self.assertGreater(sy, 0)


class AppearanceTests(unittest.TestCase):
    def test_larger_copy_matches_at_the_larger_scale(self) -> None:
        pattern = []
        mask = []
        y = 0
        while y < 8:
            x = 0
            while x < 8:
                pattern.append((x * 20 + y * 9) % 180)
                mask.append(1)
                x += 1
            y += 1
        width = 8 * 115 // 100
        height = 8 * 115 // 100
        scaled = resize_nearest(pattern, 8, 8, width, height)
        frame = [20] * (48 * 36)
        ox, oy = 10, 8
        y = 0
        while y < height:
            x = 0
            while x < width:
                frame[(oy + y) * 48 + (ox + x)] = scaled[y * width + x]
                x += 1
            y += 1
        hit = search_scales(frame, 48, 36, pattern, 8, 8, mask, 8, 8, step=1)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], ox)
        self.assertEqual(hit[1], oy)
        self.assertEqual(hit[2], width)
        self.assertEqual(hit[3], height)
        self.assertGreaterEqual(hit[4], 90)

    def test_stable_mask_drops_a_one_frame_speck(self) -> None:
        pattern = [40] * 16
        mask = [1] * 16
        appearance = Appearance(pattern, mask, 4, 4)
        base = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
        speck = list(base)
        speck[5] = 1
        gray = [30] * 16
        index = 0
        while index < LEARN_FRAMES:
            sample = speck if index == 3 else base
            appearance.record_grid(gray, sample, 4, 4, 0, 0, 20.0)
            index += 1
        self.assertEqual(appearance.samples, LEARN_FRAMES)
        self.assertEqual(appearance.mask[0], 1)
        self.assertEqual(appearance.mask[5], 0)

    def test_temperature_band_rejects_a_hotter_copy(self) -> None:
        lo = int((0.0 + 273.15) * 64)
        hi = int((50.0 + 273.15) * 64)
        warm_gray = [100] * 16
        hot_gray = [255] * 16
        flags = [1] * 16
        warm = object_celsius(warm_gray, flags, lo, hi)
        hot = object_celsius(hot_gray, flags, lo, hi)
        appearance = Appearance(warm_gray, flags, 4, 4)
        index = 0
        while index < LEARN_FRAMES:
            appearance.record_grid(warm_gray, flags, 4, 4, 8, 8, warm)
            index += 1
        self.assertTrue(appearance.same(4, 4, warm, 8, 8, False))
        self.assertFalse(appearance.same(4, 4, hot, 8, 8, False))

    def test_correlation_is_one_pass_and_a_flat_pattern_scores_zero(self) -> None:
        from src.logic import _moments, _zncc

        pattern = [10, 40, 70, 20, 50, 80, 30, 60, 90]
        frame = [12, 38, 75, 18, 55, 77, 33, 58, 88]
        mask = [1] * 9
        moments = _moments(pattern, 3, 3, mask)
        self.assertEqual(_zncc(frame, 3, 0, 0, pattern, 3, 3, mask, moments), _zncc(frame, 3, 0, 0, pattern, 3, 3, mask))
        self.assertEqual(_zncc([40] * 9, 3, 0, 0, [40] * 9, 3, 3, mask), 0)
        self.assertGreater(_zncc(frame, 3, 0, 0, pattern, 3, 3, mask), 90)

    def test_one_walk_matches_the_separate_scores_and_the_blend_keeps_energy(self) -> None:
        from src.logic import _moments, _zncc, gray_mean, masked_max, object_celsius, side_mean, survey, thermal_grid

        width = 8
        height = 8
        pattern = []
        mask = []
        crop = []
        y = 0
        while y < height:
            x = 0
            while x < width:
                pattern.append(40 + x * 15 + y * 4)
                mask.append(1 if 2 <= x < 6 and 2 <= y < 6 else 0)
                crop.append(pattern[-1] + 3)
                x += 1
            y += 1
        moments = _moments(pattern, width, height, mask)
        frozen = list(pattern)
        frozen[0] = 10
        frozen_mask = list(mask)
        frozen_moments = _moments(frozen, width, height, frozen_mask)
        band = (80, 255)
        reading = survey(
            crop, width, height, pattern, mask, moments, frozen, frozen_mask, frozen_moments, band
        )
        self.assertEqual(reading.appearance, _zncc(crop, width, 0, 0, pattern, width, height, mask, moments))
        self.assertEqual(reading.anchor, cues.anchor(crop, width, height, frozen, width, height, frozen_mask))
        self.assertEqual(reading.obj_max, masked_max(crop, mask))
        self.assertEqual(reading.gray, gray_mean(crop))
        self.assertEqual(reading.side, side_mean(crop, band[0], band[1]))
        sample = thermal_grid(crop, None, width, height, band[0], band[1])
        self.assertEqual(list(reading.cells), list(sample[0]))
        self.assertEqual(bytes(reading.valid), bytes(sample[1]))
        lo, hi = 19000, 21000
        pixel = reading.obj_sum / float(reading.obj_n)
        kelvin64 = lo + pixel * (hi - lo) / 255.0
        self.assertAlmostEqual(object_celsius(crop, mask, lo, hi), kelvin64 / 64.0 - 273.15)
        appearance = Appearance(pattern, mask, width, height)
        self.assertEqual(len(appearance.rows), 8)
        self.assertIsNotNone(appearance.fill)
        self.assertIsNotNone(appearance.peak)
        appearance._blend(crop)
        self.assertEqual(appearance.moments, _moments(appearance.pattern, width, height, appearance.mask))

    def test_a_locked_frame_is_a_few_walks_of_the_crop(self) -> None:
        import time

        from src.logic import _moments, _zncc

        size = 40
        gray = []
        y = 0
        while y < size:
            x = 0
            while x < size:
                if size // 4 <= x < 3 * size // 4 and size // 4 <= y < 3 * size // 4:
                    gray.append(180)
                else:
                    gray.append(30)
                x += 1
            y += 1
        mask = [1 if value > 80 else 0 for value in gray]
        pattern = list(gray)
        moments = _moments(pattern, size, size, mask)
        appearance = Appearance(pattern, mask, size, size)

        def walk():
            total = 0
            index = 0
            limit = size * size
            while index < limit:
                total += gray[index]
                index += 1
            return total

        def kernel():
            _zncc(gray, size, 0, 0, pattern, size, size, mask, moments)
            _zncc(gray, size, 0, 0, pattern, size, size, mask, moments)
            cues.anchor(gray, size, size, pattern, size, size, mask)
            thermal_grid(gray, mask, size, size)
            appearance._blend(gray)

        repeats = 20
        walk()
        kernel()
        started = time.perf_counter()
        count = 0
        while count < repeats:
            walk()
            count += 1
        walk_s = time.perf_counter() - started
        started = time.perf_counter()
        count = 0
        while count < repeats:
            kernel()
            count += 1
        kernel_s = time.perf_counter() - started
        # Two correlations, one anchor, and the grid. The old lock also
        # flood-filled every blob and scored the anchor a second time, about
        # 26 walks of this crop. Twelve leaves room for the heavier arithmetic
        # and still fails that stack.
        self.assertLess(kernel_s, walk_s * 12)
        self.assertFalse(hasattr(appearance, "weight"))
        self.assertFalse(hasattr(appearance, "views"))

    def test_a_hot_core_disagrees_with_a_flat_blob(self) -> None:
        width = 8
        height = 8
        mask = [1] * (width * height)
        core = [40] * (width * height)
        y = 0
        while y < 4:
            x = 0
            while x < 4:
                core[(2 + y) * width + (2 + x)] = 200
                x += 1
            y += 1
        appearance = Appearance(list(core), mask, width, height)
        index = 0
        while index < LEARN_FRAMES:
            appearance.record_grid(core, mask, width, height, 4, 4, 30.0)
            index += 1
        learned = thermal_grid(core, mask, width, height)
        flat = thermal_grid([80] * (width * height), mask, width, height)
        same = cues.grid_agreement(appearance.grid, appearance.grid_valid, learned[0], learned[1])
        other = cues.grid_agreement(appearance.grid, appearance.grid_valid, flat[0], flat[1])
        self.assertGreaterEqual(same, 90)
        self.assertEqual(other, 0)
        self.assertEqual(cues.temperature(30.0, appearance.temp_mean, appearance.temp_spread), 100)
        self.assertEqual(cues.temperature(30.0, appearance.temp_mean, appearance.temp_spread, other), 0)

    def test_a_closer_camera_keeps_a_larger_box_of_the_same_aspect(self) -> None:
        appearance = Appearance([1] * 200, [1] * 200, 10, 20)
        appearance.samples = LEARN_FRAMES
        self.assertTrue(appearance.same(20, 40, None, 0, 0, False))
        self.assertFalse(appearance.same(20, 20, None, 0, 0, False))

    def test_velocity_predicts_and_a_shutter_holds(self) -> None:
        appearance = Appearance([1], [1], 1, 1)
        appearance.see(10, 10)
        appearance.see(20, 12)
        self.assertEqual(appearance.predict(), (30, 14))
        appearance.hold()
        self.assertEqual(appearance.vx, 10)
        self.assertEqual(appearance.vy, 2)
        self.assertEqual(appearance.predict(), (30, 14))
        appearance.see(28, 18)
        self.assertEqual(appearance.vx, 9)
        self.assertEqual(appearance.vy, 4)
        self.assertEqual(appearance.predict(), (37, 22))
        self.assertEqual(appearance.predict(), (37, 22))
        appearance.coast()
        self.assertEqual((appearance.vx, appearance.vy), (9, 4))
        self.assertEqual(appearance.predict(), (46, 26))


class _PrintedLock(object):
    def __init__(self, cue="appearance=80 appearance_us=4"):
        self.cue_line = cue
        self.line = (12, -4, 93)
        self.placement = (1, 2, 8, 6)
        self.pattern = [1]

    def status(self):
        return "LOCKED"


class _Guide(object):
    def __init__(self, line="diff x=12 y=-4"):
        self.line = line
        self.calls = 0

    def report(self, lock):
        self.calls += 1
        return self.line


class ReportTests(unittest.TestCase):
    def test_debug_prints_the_cues_and_the_lock(self) -> None:
        self.assertFalse(debug_requested([]))
        self.assertFalse(debug_requested(["main.py"]))
        self.assertTrue(debug_requested(["main.py", "--debug"]))
        link = _Guide()
        buf = io.StringIO()
        with redirect_stdout(buf):
            _report(_PrintedLock(), link, True)
        self.assertEqual(link.calls, 1)
        self.assertEqual(
            buf.getvalue(),
            "appearance=80 appearance_us=4\n"
            "dx=12 dy=-4 confidence=93\n"
            "track=LOCKED\n"
            "diff x=12 y=-4\n",
        )

    def test_the_console_prints_only_the_diff_line(self) -> None:
        link = _Guide(None)
        buf = io.StringIO()
        with redirect_stdout(buf):
            _report(_PrintedLock(None), link, False)
        self.assertEqual(link.calls, 1)
        self.assertEqual(buf.getvalue(), "")
        link = _Guide()
        buf = io.StringIO()
        with redirect_stdout(buf):
            _report(_PrintedLock(), link, False)
        self.assertEqual(buf.getvalue(), "diff x=12 y=-4\n")

    def test_the_page_line_carries_the_lock_until_it_repeats(self) -> None:
        reported = [None]
        buf = io.StringIO()
        with redirect_stdout(buf):
            _note_state(_PrintedLock(), reported)
            _note_state(_PrintedLock(), reported)
        self.assertEqual(buf.getvalue(), "state 1 1 12 -4 93 LOCKED\n")


class MeterTests(unittest.TestCase):
    def test_cpu_is_the_share_of_the_frame_and_memory_is_the_heap(self) -> None:
        self.assertEqual(cpu_percent(0), 0)
        self.assertEqual(cpu_percent(41666), 50)
        self.assertEqual(cpu_percent(83333), 100)
        self.assertEqual(cpu_percent(100000), 120)
        self.assertIsNone(cpu_percent(-1))
        self.assertIsNone(cpu_percent(10, 0))
        self.assertEqual(memory_percent(0, 100), 0)
        self.assertEqual(memory_percent(75, 25), 75)
        self.assertEqual(memory_percent(1, 1), 50)
        self.assertIsNone(memory_percent(10, -1))
        self.assertIsNone(memory_percent(None, 10))

    def test_scene_temperature_follows_the_kelvin_window(self) -> None:
        # Gray maps linearly across a window of 0 .. 255 Kelvin.
        high, low, mean = scene_celsius(0, 255, 128, 0, 64 * 255)
        self.assertAlmostEqual(low, -273.15)
        self.assertAlmostEqual(high, -18.15)
        self.assertAlmostEqual(mean, 128 - 273.15)
        self.assertIsNone(scene_celsius(0, 255, 128, 10, 10))
        self.assertIsNone(scene_celsius(None, 255, 128, 0, 100))
        self.assertEqual(_tenth(36.24), 362)
        self.assertEqual(_tenth(-18.15), -182)
        self.assertEqual(_tenth(0), 0)

    def test_the_page_line_carries_the_load_and_the_temperatures(self) -> None:
        self.assertEqual(temp_line((36.24, 18.06, 24.95)), "temp 362 181 250")
        self.assertEqual(temp_line(None), "temp na na na")
        self.assertEqual(meter_line(42, 61, (36.24, 18.06, 24.95)), "meter 42 61 362 181 250")
        self.assertEqual(meter_line(None, None, None), "meter na na na na na")
        load = _cpu_load()
        self.assertEqual(load(41666), 50)
        index = 0
        while index < 11:
            load(83333)
            index += 1
        self.assertEqual(load(83333), 100)
        self.assertEqual(load(0), 92)
        buf = io.StringIO()
        with redirect_stdout(buf):
            _note_temp((36.24, 18.06, 24.95))
            _note_meter(42, 61, (36.24, 18.06, 24.95))
        self.assertEqual(buf.getvalue(), "temp 362 181 250\nmeter 42 61 362 181 250\n")


class OmvconLaunchTests(unittest.TestCase):
    def test_launch_puts_the_modules_then_views_the_tracker(self) -> None:
        from pathlib import Path

        root = Path("/repo")
        calls = invocations(root, "/dev/cu.usbmodem1", "", True)
        self.assertEqual(len(calls), 8)
        self.assertIn("weights.txt", calls[6][3])
        self.assertIn("guide.py", calls[4][3])
        self.assertIn("cues.py", calls[3][3])
        self.assertEqual(
            calls[0],
            ["-p", "/dev/cu.usbmodem1", "put", "/repo/src/t1c.py", "/sdcard/t1c.py"],
        )
        self.assertIn("logic.py", calls[2][3])
        self.assertEqual(calls[-1][:3], ["-p", "/dev/cu.usbmodem1", "view"])
        self.assertIn("-open", calls[-1])
        self.assertNotIn("-debug", calls[-1])
        self.assertNotIn("-record", calls[-1])
        self.assertTrue(calls[-1][-1].endswith("src/main.py"))
        debug = invocations(root, "/dev/cu.usbmodem1", "", False, True)
        self.assertNotIn("-open", debug[-1])
        self.assertEqual(debug[-1][-2:], ["-debug", "/repo/src/main.py"])

    def test_help_and_unknown_flags(self) -> None:
        self.assertIsNone(parse(["track.py", "--help"]))
        self.assertFalse(parse(["track.py", "--files"]))
        self.assertEqual(parse(["track.py"]), ("", "", True, False, False))
        self.assertEqual(parse(["track.py", "--debug", "--no-open"]), ("", "", False, True, False))
        self.assertEqual(parse(["track.py", "--record"]), ("", "", True, False, True))

    def test_record_is_only_on_the_view_command(self) -> None:
        from datetime import datetime
        from pathlib import Path

        from track import record_path

        root = Path("/repo")
        self.assertEqual(
            record_path(root, datetime(2026, 10, 17, 20, 2, 3)),
            root / "assets" / "track-20261017-200203.mp4",
        )
        calls = invocations(root, "/dev/cu.usbmodem1", "", False, False, True)
        for call in calls[:-1]:
            self.assertNotIn("-record", call)
        view = calls[-1]
        self.assertEqual(view[:3], ["-p", "/dev/cu.usbmodem1", "view"])
        self.assertEqual(view[3], "-record")
        self.assertRegex(view[4], r"/assets/track-\d{8}-\d{6}\.mp4$")
        self.assertNotIn("-open", view)
        self.assertNotIn("-debug", view)
        self.assertTrue(view[-1].endswith("src/main.py"))


def _sight_score(score, second, width, height, last_w, last_h, obj_max=None, frame_max=None):
    """Fused score for one sighting. Cues that abstain drop out of the average."""
    return cues.combine(
        (
            cues.Cue("appearance", cues.appearance(score), 0),
            cues.Cue("unique", cues.unique(score, second), 0),
            cues.Cue("size", cues.size(width, height, last_w, last_h), 0),
            cues.Cue("hottest", cues.hottest(obj_max, frame_max), 0),
        )
    )


class _Clock(object):
    def __init__(self, step):
        self.step = step
        self.now = 0

    def __call__(self):
        self.now += self.step
        return self.now


class CueTests(unittest.TestCase):
    def test_each_cue_scores_abstains_and_the_average_ignores_abstentions(self) -> None:
        self.assertEqual(
            cues.CUES,
            ("appearance", "unique", "motion", "size", "temperature", "anchor", "hottest", "sole"),
        )
        self.assertIsNone(cues.appearance(None))
        self.assertEqual(cues.appearance(80), 80)
        self.assertEqual(cues.appearance(140), 100)
        self.assertEqual(cues.appearance(-3), 0)
        self.assertIsNone(cues.unique(80, None))
        self.assertIsNone(cues.sole(False, None, False))
        self.assertIsNone(cues.sole(True, None, True))
        self.assertIsNone(cues.sole(True, 10, False))
        self.assertEqual(cues.sole(True, None, False), 100)
        self.assertEqual(cues.with_sole(50, None), 50)
        self.assertEqual(cues.with_sole(50, 100), cues.SOLE_SURE)
        self.assertEqual(cues.with_sole(93, 100), 93)
        self.assertIsNone(cues.unique(0, 0))
        self.assertEqual(cues.unique(80, 10), 88)
        self.assertIsNone(cues.motion(0, 0, None, 16))
        self.assertEqual(cues.motion(10, 10, (10, 10), 16), 100)
        self.assertEqual(cues.motion(18, 10, (10, 10), 16), 50)
        self.assertEqual(cues.motion(26, 10, (10, 10), 16), 0)
        trace = ((0, 10), (10, 10), (20, 10))
        self.assertEqual(cues.motion(30, 10, (30, 10), 16, trace), 100)
        self.assertGreater(
            cues.motion(38, 10, (30, 10), 16, trace),
            cues.motion(30, 18, (30, 10), 16, trace),
        )
        self.assertIsNone(cues.size(10, 10, 0, 0))
        self.assertEqual(cues.size(20, 40, 10, 20), 100)
        self.assertEqual(cues.size(20, 20, 10, 20), 50)
        self.assertTrue(cues.same_aspect(20, 40, 10, 20))
        self.assertFalse(cues.same_aspect(20, 20, 10, 20))
        self.assertIsNone(cues.hottest(None, 90))
        self.assertIsNone(cues.hottest(40, None))
        self.assertIsNone(cues.hottest(40, 90))
        self.assertEqual(cues.hottest(90, 90), 100)
        self.assertEqual(cues.hottest(100, 90), 100)
        self.assertIsNone(cues.temperature(20.0, None, 1.0))
        self.assertEqual(cues.temperature(20.0, 20.0, 1.0), 100)
        self.assertEqual(cues.temperature(21.5, 20.0, 1.0), 50)
        self.assertEqual(cues.temperature(23.0, 20.0, 1.0), 0)
        frozen = [10, 20, 30, 40]
        mask = [1, 1, 1, 1]
        self.assertEqual(cues.anchor(frozen, 2, 2, frozen, 2, 2, mask), 100)
        self.assertEqual(cues.anchor([25, 25, 25, 25], 2, 2, frozen, 2, 2, mask), 0)
        self.assertIsNone(cues.anchor(None, 2, 2, frozen, 2, 2, mask))
        facts = cues.Facts()
        facts.score = 80
        facts.second = 10
        both = cues.evaluate(facts, active=("appearance", "unique"))
        one = cues.evaluate(facts, active=("appearance",))
        self.assertEqual(cues.combine(both), 86)
        self.assertEqual(cues.combine(one), 80)
        self.assertEqual([cue.name for cue in one], ["appearance"])
        raised = list(both)
        raised.append(cues.Cue("hottest", 100, 0))
        self.assertGreater(cues.combine(raised), cues.combine(both))
        facts.score = None
        self.assertEqual(cues.combine(cues.evaluate(facts, active=("appearance",))), 0)

    def test_proximity_outweighs_size_and_size_outweighs_the_silhouette(self) -> None:
        def fused(appearance, size, motion):
            return cues.combine(
                (
                    cues.Cue("appearance", appearance, 0),
                    cues.Cue("anchor", appearance, 0),
                    cues.Cue("size", size, 0),
                    cues.Cue("motion", motion, 0),
                )
            )

        low_shape = fused(0, 100, 100)
        low_size = fused(100, 0, 100)
        low_near = fused(100, 100, 0)
        self.assertGreater(low_shape, low_size)
        self.assertGreater(low_size, low_near)

    def test_the_clock_is_the_cue_cost_and_the_line_keeps_every_field(self) -> None:
        facts = cues.Facts()
        facts.score = 84
        clock = _Clock(15)
        results = cues.evaluate(facts, clock=clock, active=("appearance", "unique"))
        self.assertEqual(results[0].value, 84)
        self.assertEqual(results[0].us, 15)
        self.assertIsNone(results[1].value)
        self.assertEqual(results[1].us, 15)
        self.assertEqual(
            cues.format_line(results),
            "appearance=84 appearance_us=15 unique=na unique_us=15",
        )

    def test_a_weights_file_overrides_one_cue_and_keeps_the_rest(self) -> None:
        directory = tempfile.TemporaryDirectory()
        try:
            path = os.path.join(directory.name, "weights.txt")
            handle = open(path, "w")
            try:
                handle.write("# tune the coast\nhottest 0\nmotion 2\nbogus nope\nsize -1\n")
            finally:
                handle.close()
            cues.load_weights((path,))
            self.assertEqual(cues.weight("hottest"), 0)
            self.assertEqual(cues.weight("motion"), 2)
            self.assertEqual(cues.weight("size"), 4)
            self.assertEqual(cues.weight("appearance"), 1)
            plain = cues.combine((cues.Cue("appearance", 80, 0), cues.Cue("hottest", 100, 0)))
            self.assertEqual(plain, 80)
        finally:
            cues.load_weights(())
            directory.cleanup()
        self.assertEqual(cues.weight("hottest"), 5)
        self.assertEqual(cues.weight("motion"), 8)

    def test_the_object_maximum_skips_background_pixels(self) -> None:
        gray = [10, 200, 40, 30]
        mask = [1, 0, 1, 1]
        self.assertEqual(masked_max(gray, mask), 40)
        self.assertEqual(masked_max(gray, None), 200)
        self.assertIsNone(masked_max(None, mask))


class _Stats(object):
    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev


class _Patch(object):
    def __init__(self, data):
        self.data = data

    def bytearray(self):
        return self.data


def _paint(frame, box, value):
    x, y, w, h = box
    yy = 0
    while yy < h:
        xx = 0
        while xx < w:
            frame.pixels[(y + yy) * frame.w + (x + xx)] = value
            xx += 1
        yy += 1


class _Mark(object):
    def __init__(self):
        self.fill = 100
        self.ox = 0
        self.oy = 0
        self.rows = [100] * 8
        self.cols = [100] * 8
        self.peak = 0


class _Bins(object):
    def __init__(self, bins):
        self._bins = bins

    def bins(self):
        return self._bins


class _HistFrame(object):
    def __init__(self):
        self.hot = 40
        self.walks = 0

    def get_statistics(self, roi=None):
        return _Stats(30, 1)

    def get_histogram(self, roi=None):
        bins = []
        index = 0
        while index < 256:
            count = self.hot if index == 220 else 0
            bins.append((index, count))
            index += 1
        return _Bins(bins)

    def get_pixel(self, x, y=None):
        self.walks += 1
        raise AssertionError("walk")


class _Frame(object):
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.pixels = [20] * (width * height)
        self.blobs = []
        self.hits = []
        self.calls = []
        self.thresholds = None
        self.blob_roi = None
        self.blob_calls = 0

    def width(self):
        return self.w

    def height(self):
        return self.h

    def get_pixel(self, x, y=None):
        if isinstance(x, tuple):
            x, y = x
        return self.pixels[y * self.w + x]

    def get_statistics(self, roi=None):
        return _Stats(30, 1)

    def find_blobs(self, thresholds, roi=None, pixels_threshold=0, area_threshold=0, merge=False):
        self.blob_calls += 1
        self.thresholds = thresholds
        self.blob_roi = roi
        self.pixels_threshold = pixels_threshold
        self.area_threshold = area_threshold
        self.merge = merge
        return list(self.blobs)

    def find_template(self, template, threshold, roi=None, step=2, search=0):
        self.calls.append((tuple(roi), step))
        if not self.hits:
            return None
        return self.hits.pop(0)

    def copy(self, roi=None, copy_to_fb=False):
        x, y, w, h = roi
        data = bytearray()
        yy = 0
        while yy < h:
            xx = 0
            while xx < w:
                data.append(self.pixels[(y + yy) * self.w + (x + xx)] & 255)
                xx += 1
            yy += 1
        return _Patch(data)


class SearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = sys.modules.get("image")
        image = types.ModuleType("image")
        image.GRAYSCALE = 2
        image.SEARCH_DS = 1
        image.SEARCH_EX = 0

        class Image(object):
            def __init__(self, width, height, fmt=None, buffer=None):
                self.width = width
                self.height = height
                self.buffer = buffer

        image.Image = Image
        sys.modules["image"] = image

    def tearDown(self) -> None:
        if self.previous is None:
            sys.modules.pop("image", None)
        else:
            sys.modules["image"] = self.previous

    def _track(self, frame, coasting, hot=True, tracked=None):
        from src.live import _search

        pattern = [40] * 64
        mask = [1] * 64
        match, found, recovered, _border, _packed = _search(
            frame, pattern, 8, 8, mask, (40, 32), 16, coasting, hot, tracked
        )
        return match, found, recovered

    def test_a_readable_frame_counts_its_heat_spots(self) -> None:
        from src.live import _spot_count
        from src.logic import SPOT_LIMIT

        frame = _Frame(16, 16)
        frame.blobs = [(1, 1, 4, 4, 16), (8, 8, 4, 4, 16)]
        self.assertEqual(_spot_count(frame), 2)
        self.assertEqual(frame.pixels_threshold, 16)
        self.assertEqual(frame.area_threshold, 16)
        self.assertFalse(frame.merge)
        self.assertEqual(frame.blob_calls, 1)
        frame.blobs = []
        self.assertEqual(_spot_count(frame), 0)

        class _Blind(object):
            def width(self):
                return 16

            def height(self):
                return 16

            def get_statistics(self, roi=None):
                return _Stats(30, 1)

        self.assertEqual(_spot_count(_Blind()), SPOT_LIMIT)

    def test_polarity_picks_the_side_that_differs_from_the_fill(self) -> None:
        mask = [1, 1, 0, 0]
        self.assertTrue(pattern_hot([90, 90, 20, 20], mask))
        self.assertFalse(pattern_hot([10, 10, 40, 40], mask))
        self.assertEqual(blob_threshold(30, 1, True), (40, 255))
        self.assertEqual(blob_threshold(30, 1, False), (0, 20))

    def test_one_blob_stays_in_the_window_and_skips_template_search(self) -> None:
        frame = _Frame(80, 64)
        frame.blobs = [(4, 4, 8, 8, 100), (30, 4, 8, 8, 70)]
        _paint(frame, (4, 4, 8, 8), 200)
        _paint(frame, (30, 4, 8, 8), 180)
        match, _found, recovered = self._track(frame, False)
        self.assertEqual(frame.blob_calls, 1)
        self.assertEqual(frame.blob_roi, (20, 12, 40, 40))
        self.assertEqual(frame.thresholds, [(40, 255)])
        self.assertEqual(frame.calls, [])
        self.assertFalse(frame.merge)
        self.assertFalse(recovered)
        self.assertIsNotNone(match)
        self.assertIsNotNone(match[5])

    def test_a_cold_object_uses_the_low_band(self) -> None:
        frame = _Frame(80, 64)
        frame.blobs = [(4, 4, 8, 8, 80)]
        match, _found, recovered = self._track(frame, False, hot=False)
        self.assertEqual(frame.thresholds, [(0, 20)])
        self.assertEqual(frame.calls, [])
        self.assertFalse(recovered)
        self.assertIsNone(match[5])

    def test_similar_blobs_without_a_template_stay_on_the_near_heat(self) -> None:
        frame = _Frame(80, 64)
        frame.blobs = [(4, 4, 8, 8, 100), (30, 4, 8, 8, 90)]
        _paint(frame, (4, 4, 8, 8), 200)
        _paint(frame, (30, 4, 8, 8), 200)
        match, _found, recovered = self._track(frame, False)
        self.assertEqual(frame.calls, [((20, 12, 40, 40), 2)])
        self.assertFalse(recovered)
        self.assertEqual(match[:4], (30, 4, 8, 8))
        coast = _Frame(80, 64)
        coast.blobs = [(4, 4, 8, 8, 100), (30, 4, 8, 8, 90)]
        match, _found, recovered = self._track(coast, True)
        self.assertIsNone(match)
        self.assertFalse(recovered)

    def test_similar_blobs_take_one_template_match_in_the_window(self) -> None:
        frame = _Frame(80, 64)
        frame.blobs = [(4, 4, 8, 8, 100), (30, 4, 8, 8, 90)]
        frame.hits = [(6, 6, 8, 8)]
        _paint(frame, (4, 4, 8, 8), 200)
        _paint(frame, (30, 4, 8, 8), 200)
        _paint(frame, (6, 6, 8, 8), 200)
        match, _found, recovered = self._track(frame, False)
        self.assertEqual(frame.calls, [((20, 12, 40, 40), 2)])
        self.assertFalse(recovered)
        self.assertEqual(match[:4], (6, 6, 8, 8))

    def test_a_miss_stays_in_the_block_while_coasting(self) -> None:
        frame = _Frame(80, 64)
        match, _found, recovered = self._track(frame, False)
        self.assertEqual(frame.calls, [((20, 12, 40, 40), 2)])
        self.assertIsNone(match)
        self.assertFalse(recovered)
        coast = _Frame(80, 64)
        coast.hits = [None, (4, 4, 8, 8)]
        match, _found, recovered = self._track(coast, True)
        self.assertEqual(coast.calls, [((20, 12, 40, 40), 2)])
        self.assertFalse(recovered)
        self.assertIsNone(match)
        weak = _Frame(80, 64)
        weak.hits = [(6, 6, 8, 8)]
        _paint(weak, (6, 6, 8, 8), 200)
        match, _found, recovered = self._track(weak, True)
        self.assertEqual(weak.calls, [((20, 12, 40, 40), 2)])
        self.assertFalse(recovered)
        self.assertEqual(match[:4], (6, 6, 8, 8))

    def test_a_collapsed_contrast_uses_the_border_band(self) -> None:
        frame = _Frame(80, 64)
        frame.blobs = [(4, 4, 8, 8, 80)]
        _paint(frame, (4, 4, 8, 8), 80)
        match, _found, recovered = self._track(frame, False, tracked=(40, 35, True))
        self.assertEqual(frame.blob_calls, 1)
        self.assertEqual(frame.thresholds, [(40, 255)])
        self.assertEqual(frame.calls, [])
        self.assertFalse(recovered)
        self.assertEqual(match[:4], (4, 4, 8, 8))

    def test_the_locked_midpoint_is_the_blob_band(self) -> None:
        frame = _Frame(80, 64)
        frame.blobs = [(4, 4, 8, 8, 80)]
        _paint(frame, (4, 4, 8, 8), 90)
        match, _found, recovered = self._track(frame, False, tracked=(90, 30, True))
        self.assertEqual(frame.thresholds, [(60, 255)])
        self.assertEqual(frame.blob_calls, 1)
        self.assertFalse(recovered)
        self.assertIsNotNone(match)

    def test_a_larger_cold_patch_does_not_take_a_hot_lock(self) -> None:
        # The surround is cold. A wide patch of it outranks the object by area.
        # The object is the hot class saved at selection, so the box stays there.
        frame = _Frame(80, 64)
        frame.blobs = [(22, 14, 24, 18, 432), (40, 28, 10, 10, 100)]
        _paint(frame, (22, 14, 24, 18), 45)
        _paint(frame, (40, 28, 10, 10), 220)
        match, _found, recovered = self._track(frame, False, tracked=(200, 40, True))
        self.assertFalse(recovered)
        self.assertIsNotNone(match)
        self.assertEqual(match[:4], (40, 28, 10, 10))
        self.assertEqual(frame.calls, [])

    def test_a_template_hit_on_cold_ground_does_not_move_the_box(self) -> None:
        frame = _Frame(80, 64)
        frame.hits = [(22, 16, 8, 8)]
        _paint(frame, (22, 16, 8, 8), 35)
        match, _found, recovered = self._track(frame, False, tracked=(200, 40, True))
        self.assertFalse(recovered)
        self.assertIsNone(match)

    def test_a_clipped_object_is_encompassed(self) -> None:
        # The window cuts the leading edge. The box has to grow through the
        # hot pixels that continue past that edge, and not into the cold patch.
        frame = _Frame(80, 64)
        frame.blobs = [(50, 28, 10, 10, 100)]
        _paint(frame, (50, 28, 20, 10), 220)
        _paint(frame, (20, 14, 12, 12), 45)
        match, _found, recovered = self._track(frame, False, tracked=(200, 40, True))
        self.assertFalse(recovered)
        self.assertEqual(match[:4], (50, 28, 20, 10))

    def test_a_signature_names_one_hot_spot_and_skips_the_template(self) -> None:
        from src.live import _search

        frame = _Frame(80, 64)
        frame.blobs = [(4, 4, 8, 8, 64, 8, 8), (30, 4, 8, 8, 60, 30, 8)]
        _paint(frame, (4, 4, 8, 8), 200)
        _paint(frame, (30, 4, 8, 8), 200)
        mark = _Mark()
        match, _found, recovered, _border, _packed = _search(
            frame,
            [40] * 64,
            8,
            8,
            [1] * 64,
            (40, 32),
            16,
            False,
            True,
            None,
            None,
            None,
            None,
            None,
            mark,
        )
        self.assertEqual(frame.calls, [])
        self.assertFalse(recovered)
        self.assertEqual(match[:4], (4, 4, 8, 8))

    def test_two_blobs_with_the_same_centroid_still_search_the_template(self) -> None:
        from src.live import _search

        frame = _Frame(80, 64)
        frame.blobs = [(4, 4, 8, 8, 64, 8, 8), (30, 4, 8, 8, 60, 34, 8)]
        _paint(frame, (4, 4, 8, 8), 200)
        _paint(frame, (30, 4, 8, 8), 200)
        _search(
            frame,
            [40] * 64,
            8,
            8,
            [1] * 64,
            (40, 32),
            16,
            False,
            True,
            None,
            None,
            None,
            None,
            None,
            _Mark(),
        )
        self.assertEqual(frame.calls, [((20, 12, 40, 40), 2)])

    def test_a_blob_that_misses_fill_and_centroid_is_not_the_return(self) -> None:
        from src.live import _mark_rejects

        mark = _Mark()
        self.assertTrue(_mark_rejects((10, 10, 20, 20, 10, 0, 0), mark))
        self.assertFalse(_mark_rejects((10, 10, 20, 20, 320, 20, 20), mark))
        self.assertFalse(_mark_rejects((10, 10, 20, 20, 10), mark))

    def test_a_histogram_counts_the_band_without_walking_the_crop(self) -> None:
        from src.live import _band_holds

        frame = _HistFrame()
        self.assertTrue(_band_holds(frame, (2, 2, 8, 8), (200, 255), 32))
        self.assertEqual(frame.walks, 0)
        frame.hot = 0
        self.assertFalse(_band_holds(frame, (2, 2, 8, 8), (200, 255), 32))
        self.assertEqual(frame.walks, 0)

    def test_a_finder_box_is_kept_when_it_holds_the_locked_side(self) -> None:
        # The finder already bounded the component. The hot paint sits inside
        # a larger rectangle and the rectangle is the sighting: flooding it
        # down to the paint was a second segmentation on every locked frame.
        frame = _Frame(80, 64)
        frame.blobs = [(10, 10, 40, 30, 1200)]
        _paint(frame, (12, 12, 10, 10), 220)
        match, _found, recovered = self._track(frame, False, tracked=(200, 40, True))
        self.assertFalse(recovered)
        self.assertEqual(match[:4], (10, 10, 40, 30))


if __name__ == "__main__":
    unittest.main()
