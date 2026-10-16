# Track one target on the Tiny1-C.
#
# omvcon view injects omvview and shows each frame. Lock saves the object
# inside the center box. Select, then drag, to choose another. A live
# sighting prints `diff x= y=` and
# sends that offset as a MAVLink LANDING_TARGET on UART 3. --debug also
# prints the cue line and the lock: `dx= dy= confidence=` from the frame
# middle to the middle of the box, then `track=`.

try:
    import omvview
except ImportError:
    omvview = None

try:
    import logic
except ImportError:
    from src import logic

try:
    import guide
except ImportError:
    from src import guide


def _camera_module():
    try:
        import camera
    except ImportError:
        from src import camera
    return camera


def _logic():
    return logic


def _viewing():
    return omvview is not None and getattr(omvview, "ACTIVE", False)


def publish(img):
    if _viewing():
        omvview.send(img)
        return
    flush = getattr(img, "flush", None)
    if flush:
        flush()


def keys():
    return []


def open_camera():
    return _camera_module().open_thermal()


def _events():
    if not _viewing():
        return []
    found = []
    ev = omvview.poll()
    while ev:
        found.append(ev)
        ev = omvview.poll()
    return found


def _pixel(img, x, y):
    try:
        value = img.get_pixel((x, y))
    except TypeError:
        value = img.get_pixel(x, y)
    if isinstance(value, tuple):
        return value[0]
    return value


def _read_box(img, box):
    x, y, width, height = box
    gray = []
    yy = 0
    while yy < height:
        xx = 0
        while xx < width:
            gray.append(_pixel(img, x + xx, y + yy) & 255)
            xx += 1
        yy += 1
    return gray


def _as_box(hit):
    if hit is None:
        return None
    try:
        return (int(hit[0]), int(hit[1]), int(hit[2]), int(hit[3]))
    except Exception:
        pass
    x = hit.x() if callable(getattr(hit, "x", None)) else hit.x
    y = hit.y() if callable(getattr(hit, "y", None)) else hit.y
    w = hit.w() if callable(getattr(hit, "w", None)) else hit.w
    h = hit.h() if callable(getattr(hit, "h", None)) else hit.h
    return (int(x), int(y), int(w), int(h))


TEMPLATE_FLOOR = 0.25
WINDOW_STEP = 2
AIM_SIDE = 48
# 12 fps. snapshot() busy-waits for the sensor, so that wait is not compute.
FRAME_US = 83333
LOAD_FRAMES = 12


def _find(img, template, threshold, roi, step):
    import image

    search = getattr(image, "SEARCH_DS", getattr(image, "SEARCH_EX", 0))
    try:
        hit = img.find_template(template, threshold, roi=roi, step=step, search=search)
    except TypeError:
        try:
            hit = img.find_template(template, threshold, roi=roi, step=step)
        except Exception:
            return None
    except Exception:
        return None
    return _as_box(hit)


def _template_image(raw, width, height):
    import image

    return image.Image(width, height, image.GRAYSCALE, buffer=raw)


def _same_center(a, b, width, height):
    return abs((a[0] + width // 2) - (b[0] + width // 2)) <= width // 2 and abs(
        (a[1] + height // 2) - (b[1] + height // 2)
    ) <= height // 2


def _crop_gray(img, x, y, width, height):
    frame_w = img.width()
    frame_h = img.height()
    if x < 0 or y < 0 or width < 1 or height < 1:
        return None
    if x + width > frame_w or y + height > frame_h:
        return None
    raw = _crop_buffer(img, x, y, width, height)
    if raw is not None:
        return raw
    gray = []
    yy = 0
    while yy < height:
        xx = 0
        while xx < width:
            gray.append(_pixel(img, x + xx, y + yy) & 255)
            xx += 1
        yy += 1
    return gray


def _crop_buffer(img, x, y, width, height):
    """One C copy of the rectangle. None when the image has no buffer."""
    if getattr(img, "copy", None) is None:
        return None
    patch = None
    try:
        patch = img.copy(roi=(x, y, width, height), copy_to_fb=False)
    except TypeError:
        try:
            patch = img.copy(roi=(x, y, width, height))
        except TypeError:
            patch = None
    except Exception:
        return None
    if patch is None or getattr(patch, "bytearray", None) is None:
        return None
    try:
        buf = patch.bytearray()
    except Exception:
        return None
    need = width * height
    if len(buf) < need:
        return None
    if len(buf) == need:
        return buf
    return buf[:need]


def _window(center, width, height, margin, frame_w, frame_h):
    """Search rectangle around the prediction, plus the motion margin."""
    if center is None:
        return (0, 0, frame_w, frame_h)
    roi_w = width + margin * 2
    roi_h = height + margin * 2
    if roi_w > frame_w:
        roi_w = frame_w
    if roi_h > frame_h:
        roi_h = frame_h
    if roi_w < width or roi_h < height:
        return (0, 0, frame_w, frame_h)
    x = center[0] - roi_w // 2
    y = center[1] - roi_h // 2
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + roi_w > frame_w:
        x = frame_w - roi_w
    if y + roi_h > frame_h:
        y = frame_h - roi_h
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    return (x, y, roi_w, roi_h)


def _stat_number(stats, name):
    if not hasattr(stats, name):
        return None
    value = getattr(stats, name)
    if callable(value):
        return value()
    return value


def _edge_level(img, roi):
    """Mean and standard deviation of the window border."""
    x, y, w, h = roi
    strips = [(x, y, w, 1)]
    if h > 1:
        strips.append((x, y + h - 1, w, 1))
    if h > 2:
        strips.append((x, y + 1, 1, h - 2))
        if w > 1:
            strips.append((x + w - 1, y + 1, 1, h - 2))
    total = 0.0
    var = 0.0
    weight = 0
    index = 0
    while index < len(strips):
        strip = strips[index]
        pixels = strip[2] * strip[3]
        index += 1
        if pixels < 1:
            continue
        stats = img.get_statistics(roi=strip)
        mean = _stat_number(stats, "mean")
        if mean is None:
            return None
        stdev = _stat_number(stats, "stdev")
        if stdev is None:
            stdev = 0
        total += mean * pixels
        var += stdev * stdev * pixels
        weight += pixels
    if weight < 1:
        return None
    return (total / weight, (var / weight) ** 0.5)


def _blob_number(blob, name):
    value = getattr(blob, name, None)
    if value is None:
        return None
    if callable(value):
        try:
            value = value()
        except Exception:
            return None
    try:
        return int(value)
    except Exception:
        return None


def _blob_box(blob):
    if isinstance(blob, tuple):
        box = (int(blob[0]), int(blob[1]), int(blob[2]), int(blob[3]), int(blob[4]))
        if len(blob) >= 7:
            return box + (int(blob[5]), int(blob[6]))
        return box
    x = blob.x() if callable(getattr(blob, "x", None)) else blob.x
    y = blob.y() if callable(getattr(blob, "y", None)) else blob.y
    w = blob.w() if callable(getattr(blob, "w", None)) else blob.w
    h = blob.h() if callable(getattr(blob, "h", None)) else blob.h
    pixels = blob.pixels() if callable(getattr(blob, "pixels", None)) else blob.pixels
    box = (int(x), int(y), int(w), int(h), int(pixels))
    cx = _blob_number(blob, "cx")
    cy = _blob_number(blob, "cy")
    if cx is None or cy is None:
        return box
    return box + (cx, cy)


def _rank_blobs(found):
    ordered = []
    for blob in found:
        box = _blob_box(blob)
        index = 0
        while index < len(ordered) and ordered[index][4] >= box[4]:
            index += 1
        ordered.insert(index, box)
    return ordered


def _region_blobs(img, roi, hot, tracked=None, min_area=None):
    """Connected regions inside the block. Returns (boxes, border level).

    tracked is (object mean, background mean, hot) from the selection. Without
    it, the band is the border rule. A collapsed gap returns no blobs.
    min_area is the blob floor. Omitted, it is 32. A pattern under 8 passes 16.
    """
    try:
        level = _edge_level(img, roi)
    except Exception:
        return [], None
    if level is None:
        return [], None
    rules = _logic()
    if min_area is None:
        min_area = rules.MIN_AREA
    if tracked is None:
        threshold = rules.blob_threshold(level[0], level[1], hot)
    else:
        threshold = rules.contrast_band(tracked[0], tracked[1], tracked[2], level[0], level[1])
        if threshold is None:
            return [], level
    if threshold[0] > threshold[1]:
        return [], level
    try:
        found = img.find_blobs(
            [threshold],
            roi=roi,
            pixels_threshold=min_area,
            area_threshold=min_area,
            merge=False,
        )
    except TypeError:
        try:
            found = img.find_blobs([threshold], roi=roi)
        except Exception:
            return [], level
    except Exception:
        return [], level
    if not found:
        return [], level
    return _rank_blobs(found), level


def _split(boxes):
    if not boxes:
        return None, None, False
    leader = boxes[0]
    runner = None
    if len(boxes) > 1:
        runner = boxes[1]
    ambiguous = runner is not None and runner[4] * 4 > leader[4] * 3
    return leader, runner, ambiguous


def _nearest(boxes, point):
    """The blob whose center is closest to point, or None."""
    if not boxes or point is None:
        return None
    best = None
    best_d = None
    for box in boxes:
        dx = box[0] + box[2] // 2 - point[0]
        dy = box[1] + box[3] // 2 - point[1]
        dist = dx * dx + dy * dy
        if best is None or dist < best_d:
            best = box
            best_d = dist
    return best


def _correlate(crop, crop_w, crop_h, pattern, pattern_w, pattern_h, mask, moments=None):
    if crop is None or crop_w < 1 or crop_h < 1:
        return 0
    rules = _logic()
    if crop_w != pattern_w or crop_h != pattern_h:
        aligned = rules.resize_nearest(crop, crop_w, crop_h, pattern_w, pattern_h)
    else:
        aligned = crop
    return rules._zncc(aligned, pattern_w, 0, 0, pattern, pattern_w, pattern_h, mask, moments)


def _measure_box(
    img, box, pattern, pattern_w, pattern_h, mask, moments, frozen, frozen_mask, frozen_moments, band, full
):
    """Score one blob. full also reads the anchor, the grid, and the silhouette."""
    x, y, w, h = box[0], box[1], box[2], box[3]
    crop = _crop_gray(img, x, y, w, h)
    if crop is None:
        return (x, y, w, h, 0), None, None
    rules = _logic()
    if w == pattern_w and h == pattern_h:
        aligned = crop
    else:
        aligned = rules.resize_nearest(crop, w, h, pattern_w, pattern_h)
    if full:
        reading = rules.survey(
            aligned,
            pattern_w,
            pattern_h,
            pattern,
            mask,
            moments,
            frozen,
            frozen_mask,
            frozen_moments,
            band,
        )
        score = 0 if reading.appearance is None else reading.appearance
        return (x, y, w, h, score), crop, reading
    score = rules._zncc(aligned, pattern_w, 0, 0, pattern, pattern_w, pattern_h, mask, moments)
    return (x, y, w, h, score), crop, None


def _pair(
    img,
    leader,
    others,
    pattern,
    pattern_w,
    pattern_h,
    mask,
    moments=None,
    frozen=None,
    frozen_mask=None,
    frozen_moments=None,
    band=None,
):
    best, crop, reading = _measure_box(
        img,
        leader,
        pattern,
        pattern_w,
        pattern_h,
        mask,
        moments,
        frozen,
        frozen_mask,
        frozen_moments,
        band,
        True,
    )
    second = None
    found = [best]
    for box in others:
        if _same_center(best, box, best[2], best[3]):
            continue
        other, _crop, _reading = _measure_box(
            img,
            box,
            pattern,
            pattern_w,
            pattern_h,
            mask,
            moments,
            None,
            None,
            None,
            None,
            False,
        )
        if second is None or other[4] > second:
            second = other[4]
        found.append(other)
    match = (best[0], best[1], best[2], best[3], best[4], second)
    return match, found, crop, reading


def _from_template(
    img,
    pattern,
    pattern_w,
    pattern_h,
    mask,
    roi,
    step,
    others,
    band,
    moments=None,
    frozen=None,
    frozen_mask=None,
    frozen_moments=None,
    grid_band=None,
):
    template = _template_image(bytearray(pattern), pattern_w, pattern_h)
    hit = _find(img, template, TEMPLATE_FLOOR, roi, step)
    if hit is None:
        return None
    focused = _select_heat(img, hit, band)
    if focused is None:
        return None
    focused = _cover(img, focused, roi, band)
    return _pair(
        img,
        focused,
        others,
        pattern,
        pattern_w,
        pattern_h,
        mask,
        moments,
        frozen,
        frozen_mask,
        frozen_moments,
        grid_band,
    )


def _flat_heat(lock, crop):
    """How close this crop is to the locked temperature, or None.

    A flat pattern has no correlation. The coast check uses the saved
    object mean instead, so the hottest point in the window does not win.
    """
    appearance = lock.appearance
    if not _pattern_flat(appearance) or crop is None:
        return None
    rules = _logic()
    return rules.contrast_like(
        rules.gray_mean(crop), appearance.obj_mean, appearance.bg_mean, appearance.hot
    )


def _shape_of(img, lock, box, known=None):
    """The learned grid when it exists, otherwise the correlation already scored.

    A measured correlation of 0 on a flat pattern is not a match. The
    temperature stands in for it.
    """
    crop = _crop_gray(img, box[0], box[1], box[2], box[3])
    structure = lock._structure(crop, box[2], box[3])
    if structure is not None:
        return structure
    if known is not None and known > 0:
        return known
    heat = _flat_heat(lock, crop)
    if heat is not None:
        return heat
    if known is not None:
        return known
    moments = None
    if lock.appearance is not None:
        moments = lock.appearance.moments
    return _correlate(
        crop, box[2], box[3], lock.pattern, lock.pattern_w, lock.pattern_h, lock.mask, moments
    )


def _blob_fill(box):
    if box is None or len(box) < 5:
        return None
    area = box[2] * box[3]
    if area < 1:
        return None
    fill = box[4] * 100 // area
    if fill > 100:
        fill = 100
    return fill


def _blob_offset(box):
    """Centroid offset from the box center, in percent of each side."""
    if box is None or len(box) < 7 or box[2] < 1 or box[3] < 1:
        return None
    ox = (box[5] - (box[0] + box[2] // 2)) * 100 // box[2]
    oy = (box[6] - (box[1] + box[3] // 2)) * 100 // box[3]
    return ox, oy


def _coarse(box, mark):
    """Fill and centroid against the stored silhouette, or None.

    None means this blob cannot be compared, so the caller keeps the
    template search. Both have to be present: area alone does not say
    which warm object is the lock.
    """
    if mark is None or mark.fill is None or mark.ox is None:
        return None
    fill = _blob_fill(box)
    offset = _blob_offset(box)
    if fill is None or offset is None:
        return None
    gap = fill - mark.fill
    if gap < 0:
        gap = -gap
    if gap > 100:
        gap = 100
    tone = 100 - gap
    dx = offset[0] - mark.ox
    dy = offset[1] - mark.oy
    if dx < 0:
        dx = -dx
    if dy < 0:
        dy = -dy
    dist = dx
    if dy > dist:
        dist = dy
    if dist > 100:
        dist = 100
    return (tone + (100 - dist)) // 2


def _by_signature(boxes, mark):
    """The blob the silhouette names, when it stands clear of the next one."""
    best = None
    best_score = -1
    second = -1
    for box in boxes:
        score = _coarse(box, mark)
        if score is None:
            return None
        if score > best_score:
            second = best_score
            best_score = score
            best = box
        elif score > second:
            second = score
    if best is None or best_score < 70:
        return None
    if second >= 0 and second + 20 > best_score:
        return None
    rest = []
    for box in boxes:
        if box is not best:
            rest.append(box)
    return best, rest


def _mark_rejects(box, mark):
    """True when fill and centroid both disagree with the stored silhouette.

    Either one alone still gets a shape check. A coast test that only has
    a rectangle is not rejected.
    """
    if mark is None or mark.fill is None or mark.ox is None:
        return False
    fill = _blob_fill(box)
    offset = _blob_offset(box)
    if fill is None or offset is None:
        return False
    gap = fill - mark.fill
    if gap < 0:
        gap = -gap
    dx = offset[0] - mark.ox
    dy = offset[1] - mark.oy
    if dx < 0:
        dx = -dx
    if dy < 0:
        dy = -dy
    return gap > 45 and (dx > 50 or dy > 50)


def _best_return(img, lock, boxes, trusted=False):
    """The blob whose size and shape best match the lock, or None.

    Motion is not in the score. A runaway rectangle is skipped. A blob
    whose fill and centroid both miss the stored silhouette is skipped
    before the crop is read. A flat pattern is matched on its temperature,
    so a hotter point does not take the box. trusted means the fifth field
    is a correlation. A blob's pixel count is not.
    """
    if lock.appearance is None:
        return None
    rules = _logic()
    last_w = lock.appearance.box_w
    last_h = lock.appearance.box_h
    best = None
    best_score = -1
    for box in boxes:
        if box is None or len(box) < 4:
            continue
        if not lock.familiar(box[2], box[3]):
            continue
        if _mark_rejects(box, lock.appearance):
            continue
        known = None
        if trusted and len(box) > 4:
            known = box[4]
        shape = _shape_of(img, lock, box, known)
        scored = rules.return_score(box[2], box[3], last_w, last_h, shape)
        if scored is None or scored <= best_score:
            continue
        best_score = scored
        best = (box[0], box[1], box[2], box[3], scored, 0)
    return best


def _frame_blobs(img, lock):
    """Blobs on the whole frame that still have the locked temperature.

    A collapsed contrast gap is searched again on the border band, the same
    fallback the window uses, so a warming background does not hide the object.
    """
    rules = _logic()
    frame_w = img.width()
    frame_h = img.height()
    appearance = lock.appearance
    tracked = None
    if appearance is not None and appearance.obj_mean is not None and appearance.bg_mean is not None:
        tracked = (appearance.obj_mean, appearance.bg_mean, appearance.hot)
    hot = False
    if appearance is not None:
        hot = appearance.hot
    else:
        hot = rules.pattern_hot(lock.pattern, lock.mask)
    _min_side, min_area = rules.active_floors(lock.pattern_w, lock.pattern_h)
    roi = (0, 0, frame_w, frame_h)
    boxes, level = _region_blobs(img, roi, hot, tracked, min_area)
    if tracked is not None and not boxes:
        boxes, level = _region_blobs(img, roi, hot, None, min_area)
        tracked = None
    band = _heat_band(level, hot, tracked)
    return _on_heat(img, boxes, band, _min_side, min_area)


def _pattern_flat(appearance):
    """True when the stored pattern has no contrast for a correlation."""
    if appearance is None or appearance.moments is None:
        return False
    count, sum_p, sum_pp = appearance.moments
    if count < 1:
        return False
    return count * sum_pp - sum_p * sum_p <= 0


def _seek_frame(img, lock):
    """The full-frame blob that matches the stored size, silhouette, and heat."""
    appearance = lock.appearance
    if appearance is None:
        return None
    rules = _logic()
    best = None
    best_score = -1
    for box in _frame_blobs(img, lock):
        if box is None or len(box) < 4:
            continue
        if not lock.familiar(box[2], box[3]):
            continue
        if _mark_rejects(box, appearance):
            continue
        crop = _crop_gray(img, box[0], box[1], box[2], box[3])
        reading = lock.read_crop(crop, box[2], box[3])
        like = None
        heat = None
        if reading is not None:
            like = rules.profile_like(
                reading.rows,
                reading.cols,
                reading.fill,
                reading.peak,
                appearance.rows,
                appearance.cols,
                appearance.fill,
                appearance.peak,
            )
            heat = reading.structure
            if heat is None:
                heat = reading.appearance
            if _pattern_flat(appearance):
                sample = reading.side
                if sample is None:
                    sample = reading.gray
                contrast = rules.contrast_like(
                    sample, appearance.obj_mean, appearance.bg_mean, appearance.hot
                )
                if contrast is not None and (heat is None or contrast > heat):
                    heat = contrast
        scored = rules.identity_score(
            box[2], box[3], appearance.box_w, appearance.box_h, like, heat
        )
        if scored is None or scored <= best_score:
            continue
        best_score = scored
        best = (box[0], box[1], box[2], box[3], scored, 0)
    return best


def _take_back(img, lock, found, now=None):
    """While coasting, take the object back when it is still this lock.

    Detections already in the predicted window are checked every coasting
    frame. After two seconds without a match there, the rest of the frame
    is searched for the stored size, silhouette, and heat signature.
    """
    if not lock.coasting():
        return None
    best = _best_return(img, lock, found, True)
    if best is None and lock.seek_due(now):
        best = _seek_frame(img, lock)
    return best


def _held(match, coasting):
    """A coasting frame keeps a window hit only when the sighting is clear."""
    if match is None:
        return False
    if not coasting:
        return True
    second = 0 if match[5] is None else match[5]
    return _logic().is_clear(match[4], second)


def _heat_band(level, hot, tracked):
    """Locked temperature class for this window, or the border rule when the gap is gone."""
    if level is None:
        return None
    rules = _logic()
    if tracked is None:
        return rules.blob_threshold(level[0], level[1], hot)
    band = rules.contrast_band(tracked[0], tracked[1], tracked[2], level[0], level[1])
    if band is None:
        return rules.blob_threshold(level[0], level[1], hot)
    return band


def _bin_pair(item, index, trust_index):
    """Gray value and count for one histogram bin, or None.

    An index is the gray value only for a full 8-bit histogram. A short
    histogram without its own values is refused, and the caller counts
    the crop instead of dropping the object.
    """
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return int(item[0]), int(item[1])
    count = getattr(item, "count", None)
    if count is None:
        count = getattr(item, "bin_count", None)
    if callable(count):
        try:
            count = count()
        except Exception:
            count = None
    value = getattr(item, "value", None)
    if value is None:
        value = getattr(item, "bin_value", None)
    if callable(value):
        try:
            value = value()
        except Exception:
            value = None
    if count is None:
        return None
    if value is None:
        if not trust_index:
            return None
        value = index
    return int(value), int(count)


def _band_count(img, box, lo, hi):
    """Pixels of this rectangle inside the band, from the C histogram.

    None when the image has no histogram, so the caller counts the crop.
    A mixed blob box is the usual locked frame: the bounding rectangle
    holds background, and a Python walk of it was most of the frame.
    """
    getter = getattr(img, "get_histogram", None)
    if not callable(getter):
        return None
    try:
        hist = getter(roi=(box[0], box[1], box[2], box[3]))
    except Exception:
        return None
    bins_of = getattr(hist, "bins", None)
    if not callable(bins_of):
        return None
    try:
        bins = bins_of()
    except Exception:
        return None
    if not bins:
        return None
    total = 0
    index = 0
    limit = len(bins)
    trust_index = limit >= 256
    while index < limit:
        pair = _bin_pair(bins[index], index, trust_index)
        if pair is None:
            return None
        if lo <= pair[0] <= hi:
            total += pair[1]
        index += 1
    return total


def _band_holds(img, box, band, min_area=1):
    """True when the rectangle holds enough pixels on the locked side.

    Statistics answer the uniform cases. A mixed rectangle is counted from
    the histogram when the image has one, and otherwise by a walk that
    stops once the area floor is met. A hot corner on a cold box does not
    keep the box.
    """
    lo, hi = band
    if min_area < 1:
        min_area = 1
    try:
        stats = img.get_statistics(roi=(box[0], box[1], box[2], box[3]))
    except Exception:
        stats = None
    if stats is not None:
        floor = _stat_number(stats, "min")
        peak = _stat_number(stats, "max")
        if lo <= 0:
            if floor is not None and floor > hi:
                return False
            if floor is not None and peak is not None and peak <= hi:
                area = box[2] * box[3]
                if len(box) > 4:
                    area = box[4]
                return area >= min_area
        else:
            if peak is not None and peak < lo:
                return False
            if floor is not None and floor >= lo and (peak is None or peak <= hi):
                area = box[2] * box[3]
                if len(box) > 4:
                    area = box[4]
                return area >= min_area
    counted = _band_count(img, box, lo, hi)
    if counted is not None:
        return counted >= min_area
    gray = _crop_gray(img, box[0], box[1], box[2], box[3])
    if gray is None:
        return True
    found = 0
    index = 0
    limit = len(gray)
    while index < limit:
        value = gray[index] & 255
        if lo <= value <= hi:
            found += 1
            if found >= min_area:
                return True
        index += 1
    return False


def _band_fills(img, box, band):
    """True when every pixel of the rectangle is already inside the band."""
    lo, hi = band
    try:
        stats = img.get_statistics(roi=(box[0], box[1], box[2], box[3]))
    except Exception:
        return False
    floor = _stat_number(stats, "min")
    peak = _stat_number(stats, "max")
    if floor is None or peak is None:
        return False
    return floor >= lo and peak <= hi


def _select_heat(img, box, band, min_side=None, min_area=None):
    """Tighten one template hit onto the locked side. None when that side is absent.

    A rectangle the band already fills is kept. A mixed rectangle is reduced
    to the largest patch. A crop that cannot be read is left unchanged.
    """
    if band is None or band[0] > band[1]:
        return box
    if not _band_holds(img, box, band):
        return None
    if _band_fills(img, box, band):
        return box
    gray = _crop_gray(img, box[0], box[1], box[2], box[3])
    if gray is None:
        return box
    found = _logic().heat_bounds(
        gray, box[2], box[3], band[0], band[1], min_side, min_area
    )
    if found is None:
        return None
    if (
        found[0] == 0
        and found[1] == 0
        and found[2] == box[2]
        and found[3] == box[3]
        and len(box) > 4
    ):
        return (box[0], box[1], box[2], box[3], box[4])
    return (box[0] + found[0], box[1] + found[1], found[2], found[3], found[4])


def _touches(box, roi):
    """True when the box meets the search edge from inside the window.

    A rectangle that does not overlap the window was not cut by it.
    """
    x, y, width, height = box[0], box[1], box[2], box[3]
    rx, ry, rw, rh = roi
    if x + width <= rx or y + height <= ry or x >= rx + rw or y >= ry + rh:
        return False
    if x <= rx or y <= ry:
        return True
    if x + width >= rx + rw or y + height >= ry + rh:
        return True
    return False


def _cover(img, box, roi, band):
    """Grow a box that the window cut off, through the hot pixels beyond that edge.

    A cold neighbor is outside the band and does not come along. A box that
    already sits inside the window is unchanged.
    """
    if box is None or band is None or band[0] > band[1] or not _touches(box, roi):
        return box
    frame_w = img.width()
    frame_h = img.height()
    x, y, width, height = box[0], box[1], box[2], box[3]
    # The gate will not follow a step past this, so the regrow stops here too.
    pad_x = _logic().GATE_CAP
    pad_y = pad_x
    rx, ry, rw, rh = roi
    x0 = x
    y0 = y
    x1 = x + width
    y1 = y + height
    if x <= rx:
        x0 = x - pad_x
    if y <= ry:
        y0 = y - pad_y
    if x + width >= rx + rw:
        x1 = x + width + pad_x
    if y + height >= ry + rh:
        y1 = y + height + pad_y
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x1 > frame_w:
        x1 = frame_w
    if y1 > frame_h:
        y1 = frame_h
    crop_w = x1 - x0
    crop_h = y1 - y0
    if crop_w < 1 or crop_h < 1:
        return box
    gray = _crop_gray(img, x0, y0, crop_w, crop_h)
    if gray is None:
        return box
    seed = (x - x0, y - y0, width, height)
    grown = _logic().connected_bounds(gray, crop_w, crop_h, seed, band[0], band[1])
    if grown is None:
        return box
    covered = (x0 + grown[0], y0 + grown[1], grown[2], grown[3])
    if len(box) == 4:
        return covered
    return covered + tuple(box[4:])


def _on_heat(img, boxes, band, min_side=None, min_area=None):
    """Finder boxes that still hold the locked temperature class.

    The blob finder already returned each component's rectangle. Flooding
    that rectangle again in Python was a second segmentation of every blob
    on every locked frame. A box with no pixel on the locked side is dropped.
    """
    rules = _logic()
    if min_side is None:
        min_side = rules.MIN_SIDE
    if min_area is None:
        min_area = rules.MIN_AREA
    kept = []
    for box in boxes:
        if box[2] < min_side or box[3] < min_side:
            continue
        area = box[2] * box[3]
        if len(box) > 4:
            area = box[4]
        if area < min_area:
            continue
        if band is not None and not _band_holds(img, box, band, min_area):
            continue
        kept.append(box)
    return _rank_blobs(kept)


def _note_distractors(lock, match, found, img):
    """Remember every blob in this window that was not the chosen one."""
    if match is None or not found:
        return
    rules = _logic()
    for item in found:
        if item[0] == match[0] and item[1] == match[1] and item[2] == match[2] and item[3] == match[3]:
            continue
        crop = _crop_gray(img, item[0], item[1], item[2], item[3])
        lock.note_distractor(item, rules.gray_mean(crop))


def _pack(crop, reading):
    if crop is None and reading is None:
        return None
    return (crop, reading)


def _search(
    img,
    pattern,
    pattern_w,
    pattern_h,
    mask,
    predicted,
    margin,
    coasting,
    hot,
    tracked=None,
    moments=None,
    frozen=None,
    frozen_mask=None,
    frozen_moments=None,
    mark=None,
):
    """Blobs in the block around the last sighting, then one template match there.

    A locked band that finds nothing is tried again on the border band. Either
    way the box is only the pixels on the locked side: a hot object does not
    move onto cold surround, and a template hit on that surround is refused.
    Similar blobs are separated by the stored silhouette when one of them
    matches it and the correlation agrees. Otherwise the template runs.
    When that misses, the blob nearest the prediction is the target, so a
    neighbor of the same size does not leave the selection glued to the
    image. A coast does not take that fallback. A miss stays in this block.
    The object does not jump across the frame.
    Returns (match, found, recovered, border mean, packed crop). recovered
    is always false. The packed crop is the walk already done for the match.
    """
    frame_w = img.width()
    frame_h = img.height()
    rules = _logic()
    min_side, min_area = rules.active_floors(pattern_w, pattern_h)
    roi = _window(predicted, pattern_w, pattern_h, margin, frame_w, frame_h)
    boxes, level = _region_blobs(img, roi, hot, tracked, min_area)
    band = _heat_band(level, hot, tracked)
    if tracked is not None and not boxes:
        boxes, level = _region_blobs(img, roi, hot, None, min_area)
        if band is None:
            band = _heat_band(level, hot, None)
    boxes = _on_heat(img, boxes, band, min_side, min_area)
    border = None
    if level is not None:
        border = level[0]
    shaped = None
    if tracked is not None:
        shaped = rules.grid_band(tracked[0], tracked[1], tracked[2], border if border is not None else tracked[1])
    leader, runner, ambiguous = _split(boxes)
    pending = None
    if leader is not None and not ambiguous:
        leader = _cover(img, leader, roi, band)
        others = []
        if runner is not None:
            others.append(runner)
        match, found, crop, reading = _pair(
            img,
            leader,
            others,
            pattern,
            pattern_w,
            pattern_h,
            mask,
            moments,
            frozen,
            frozen_mask,
            frozen_moments,
            shaped,
        )
        if _held(match, coasting):
            return match, found, False, border, _pack(crop, reading)
        pending = (match, found, crop, reading)
    else:
        others = []
        if ambiguous:
            if leader is not None:
                others.append(leader)
            if runner is not None:
                others.append(runner)
        picked = _by_signature(boxes, mark) if ambiguous else None
        if picked is not None:
            chosen = _cover(img, picked[0], roi, band)
            match, found, crop, reading = _pair(
                img,
                chosen,
                picked[1],
                pattern,
                pattern_w,
                pattern_h,
                mask,
                moments,
                frozen,
                frozen_mask,
                frozen_moments,
                shaped,
            )
            like = None
            if reading is not None and mark is not None:
                like = rules.profile_like(
                    reading.rows,
                    reading.cols,
                    reading.fill,
                    reading.peak,
                    mark.rows,
                    mark.cols,
                    mark.fill,
                    mark.peak,
                )
            second = match[5]
            better = second is None or match[4] >= second
            shape_ok = like is None or like >= 50
            if shape_ok and better and _held(match, coasting):
                return match, found, False, border, _pack(crop, reading)
        matched = _from_template(
            img,
            pattern,
            pattern_w,
            pattern_h,
            mask,
            roi,
            WINDOW_STEP,
            others,
            band,
            moments,
            frozen,
            frozen_mask,
            frozen_moments,
            shaped,
        )
        if matched is not None and _held(matched[0], coasting):
            return matched[0], matched[1], False, border, _pack(matched[2], matched[3])
        if matched is not None:
            pending = matched
        if pending is None and not coasting:
            chosen = _nearest(boxes, predicted)
            if chosen is not None:
                rest = []
                for box in boxes:
                    if box is not chosen:
                        rest.append(box)
                chosen = _cover(img, chosen, roi, band)
                pending = _pair(
                    img,
                    chosen,
                    rest,
                    pattern,
                    pattern_w,
                    pattern_h,
                    mask,
                    moments,
                    frozen,
                    frozen_mask,
                    frozen_moments,
                    shaped,
                )
    if pending is not None:
        return pending[0], pending[1], False, border, _pack(pending[2], pending[3])
    return None, [], False, border, None


def _draw_box(img, box):
    """Black contour, with the white stroke one pixel down and to the right.

    A hot target is white, so the contour has to be the dark stroke.
    """
    x, y, width, height = box
    img.draw_rectangle((x + 1, y + 1, width, height), color=255)
    img.draw_rectangle((x, y, width, height), color=0)


def _draw_text(img, text, y):
    img.draw_string((5, y + 1), text, color=0, scale=1)
    img.draw_string((4, y), text, color=255, scale=1)


def _mark_lock(img, lock):
    """The locked rectangle. The percent appears once a sighting is accepted."""
    if not lock.show_lock or lock.lock_box is None:
        return
    _draw_box(img, lock.lock_box)
    if lock.center is None:
        return
    _draw_score(img, lock.lock_box, lock.line[2], lock.frame_w, lock.frame_h)


def _draw_score(img, box, score, frame_w, frame_h):
    x, y, width, height = box
    text = "%d%%" % score
    tw = 8 * len(text)
    tx = x + width + 2
    ty = y
    if tx + tw >= frame_w:
        tx = x - tw - 2
    if tx < 0:
        tx = x + 2
        ty = y - 12
    if ty < 0:
        ty = y + height + 2
    if ty > frame_h - 10:
        ty = y
    if ty < 0:
        ty = 0
    img.draw_string((tx + 1, ty + 1), text, color=0, scale=1)
    img.draw_string((tx, ty), text, color=255, scale=1)


def _covered(box, shown, width, height):
    for item in shown:
        if _same_center(box, item, width, height):
            return True
    return False


def cpu_percent(work_us, budget_us=FRAME_US):
    """Percent of one frame spent after the snapshot returns.

    Above 100 the tracker overran the 83.3 ms budget.
    """
    if work_us is None or budget_us is None:
        return None
    work = int(work_us)
    budget = int(budget_us)
    if work < 0 or budget <= 0:
        return None
    return (work * 100 + budget // 2) // budget


def memory_percent(used, free):
    """Heap currently allocated, as a percent of allocated plus free."""
    if used is None or free is None:
        return None
    used = int(used)
    free = int(free)
    if used < 0 or free < 0:
        return None
    total = used + free
    if total <= 0:
        return None
    return (used * 100 + total // 2) // total


def scene_celsius(low_gray, high_gray, mean_gray, lo, hi):
    """Highest, lowest, and mean Celsius in one 8-bit frame.

    lo and hi are the Kelvin*64 window that this frame mapped onto gray 0
    and gray 255. The mapping is linear, so the mean gray is the mean
    temperature. A pixel clipped at either end is reported as that end.
    """
    if lo is None or hi is None or hi <= lo:
        return None
    if low_gray is None or high_gray is None or mean_gray is None:
        return None
    span = float(hi - lo)

    def one(gray):
        kelvin64 = lo + float(gray) * span / 255.0
        return kelvin64 / 64.0 - 273.15

    return (one(high_gray), one(low_gray), one(mean_gray))


def _tenth(celsius):
    """Celsius as an integer tenth, half away from zero."""
    scaled = celsius * 10.0
    if scaled >= 0.0:
        return int(scaled + 0.5)
    return -int(-scaled + 0.5)


def temp_line(temps):
    """Page line: high, low, and mean in tenths of a degree."""

    def num(value):
        if value is None:
            return "na"
        return "%d" % value

    if temps is None:
        return "temp na na na"
    return "temp %s %s %s" % (
        num(_tenth(temps[0])),
        num(_tenth(temps[1])),
        num(_tenth(temps[2])),
    )


def meter_line(cpu, mem, temps):
    """Debug line: cpu%, heap%, then high, low, and mean in tenths of a degree."""

    def num(value):
        if value is None:
            return "na"
        return "%d" % value

    if temps is None:
        high = low = mean = "na"
    else:
        high = num(_tenth(temps[0]))
        low = num(_tenth(temps[1]))
        mean = num(_tenth(temps[2]))
    return "meter %s %s %s %s %s" % (num(cpu), num(mem), high, low, mean)


def _cpu_load():
    """Mean of the last second of frame costs, as a percent of FRAME_US."""
    samples = []

    def note(work_us):
        samples.append(int(work_us))
        extra = len(samples) - LOAD_FRAMES
        if extra > 0:
            del samples[:extra]
        total = 0
        index = 0
        while index < len(samples):
            total += samples[index]
            index += 1
        return cpu_percent(total // len(samples))

    return note


def _memory():
    import gc

    try:
        return memory_percent(gc.mem_alloc(), gc.mem_free())
    except Exception:
        return None


def _scene(img, window):
    """Page temperatures, and the gray maximum for the hottest cue.

    The page receives only the temperatures. The gray maximum stays on the
    board and is the frame maximum the hottest cue compares against.
    """
    try:
        stats = img.get_statistics()
    except Exception:
        return None, None
    if isinstance(stats, (list, tuple)):
        if not stats:
            return None, None
        stats = stats[0]
    gray_max = _stat_number(stats, "max")
    span = getattr(window, "cur", None)
    if not span or len(span) != 2:
        return None, gray_max
    return (
        scene_celsius(
            _stat_number(stats, "min"),
            gray_max,
            _stat_number(stats, "mean"),
            span[0],
            span[1],
        ),
        gray_max,
    )


def _note_temp(temps):
    """Tell the page the frame temperatures.

    Printed every frame. The viewer reads this line and does not echo it.
    """
    print(temp_line(temps))


def _note_meter(cpu, mem, temps):
    """Tell the console the camera load and the frame temperatures.

    Printed only when --debug is set. The viewer reads this line and does
    not echo it.
    """
    print(meter_line(cpu, mem, temps))


def _note_state(lock, reported):
    """Tell the page the drag box, the saved object, and the latest lock.

    Printed only when it changes. The viewer reads this line and does not
    echo it. The cue line and the lock line are separate, and only --debug
    prints those.
    """
    placed = 0
    if lock.placement is not None:
        placed = 1
    saved = 0
    if lock.pattern is not None:
        saved = 1
    dx, dy, conf = lock.line
    line = "state %d %d %d %d %d %s" % (placed, saved, dx, dy, conf, lock.status())
    if line == reported[0]:
        return
    reported[0] = line
    print(line)


def _draw_state(img, lock):
    """NA, LOCKED, COASTING, or LOST, along the bottom edge of the frame."""
    y = img.height() - 12
    if y < 0:
        y = 0
    _draw_text(img, lock.status(), y)


def debug_requested(argv):
    """Cue lines and the lock are printed when argv holds --debug."""
    if not argv:
        return False
    return "--debug" in argv


def _report(lock, link, debug):
    if debug:
        if lock.cue_line:
            print(lock.cue_line)
        print("dx=%d dy=%d confidence=%d" % lock.line)
        print("track=%s" % lock.status())
    line = link.report(lock)
    if line:
        print(line)


def _spot_count(img):
    """Heat spots on the whole frame.

    A miss counts as three, so a small lock stays shut when the count
    cannot be read. The floor is 16 pixels, and neighbors are not merged.
    """
    rules = _logic()
    try:
        roi = (0, 0, img.width(), img.height())
    except Exception:
        return rules.SPOT_LIMIT
    try:
        level = _edge_level(img, roi)
    except Exception:
        return rules.SPOT_LIMIT
    if level is None:
        return rules.SPOT_LIMIT
    threshold = rules.blob_threshold(level[0], level[1], True)
    if threshold[0] > threshold[1]:
        return rules.SPOT_LIMIT
    try:
        found = img.find_blobs(
            [threshold],
            roi=roi,
            pixels_threshold=rules.SMALL_AREA,
            area_threshold=rules.SMALL_AREA,
            merge=False,
        )
        if not found:
            return 0
        return len(found)
    except Exception:
        return rules.SPOT_LIMIT


def aim_box(width, height, side=AIM_SIDE):
    """Square on the frame center, the box drawn around the cross.

    Each side is 48, or the frame when that is smaller. The middle of the
    square is the frame middle.
    """
    if width < 1 or height < 1:
        return None
    box_w = side if side < width else width
    box_h = side if side < height else height
    return ((width - box_w) // 2, (height - box_h) // 2, box_w, box_h)


def _apply(lock, img, ev):
    kind = ev[0]
    if kind == "select":
        lock.select()
        return
    if kind == "lock":
        if lock.placement is None:
            box = aim_box(lock.frame_w, lock.frame_h)
            if box is None:
                return
            if not lock.drawing:
                lock.select()
            lock.place(box)
        if lock.drawing and lock.placement is not None:
            gray = _read_box(img, lock.placement)
            width = lock.placement[2]
            height = lock.placement[3]
            lock.lock(gray, width, height, _spot_count(img))
        return
    if kind == "unlock":
        lock.unlock()
        return
    if kind == "box" and len(ev) == 5:
        lock.place((ev[1], ev[2], ev[3], ev[4]))


def _clock():
    import time

    last = [None]
    mono = [0]

    def now():
        tick = time.ticks_ms()
        if last[0] is None:
            last[0] = tick
            return 0
        mono[0] += time.ticks_diff(tick, last[0])
        last[0] = tick
        return mono[0]

    return now


def run(paths=None, debug=False):
    import gc
    import time

    rules = _logic()
    if paths is None:
        paths = ("/sdcard/locke.target", "/sd/locke.target")
    lock = rules.Lock(paths)
    lock.startup()
    try:
        cam, window, shutter = open_camera()
    except ImportError:
        print("csi is provided by the LWIR board. t1c.py in this directory is the camera module.")
        return 1
    now = _clock()
    since_gc = 0
    reported = [None]
    link = guide.Link()
    note_load = _cpu_load()
    try:
        while True:
            img = cam.snapshot()
            started = time.ticks_us()
            window.update(img)
            temps, seen_max = _scene(img, window)
            lock.frame_w = img.width()
            lock.frame_h = img.height()
            stamp = now()
            if shutter.due():
                lock.note_shutter(stamp)
                _mark_lock(img, lock)
                if lock.placement is not None:
                    _draw_box(img, lock.placement)
                if lock.message:
                    _draw_text(img, lock.message, 16)
                _draw_text(img, "SHUTTER", 4)
                _draw_state(img, lock)
                publish(img)
                _note_state(lock, reported)
                _report(lock, link, debug)
                _note_temp(temps)
                # The shutter close waits on the module, so it is left out of the frame cost.
                if debug:
                    _note_meter(
                        note_load(time.ticks_diff(time.ticks_us(), started)), _memory(), temps
                    )
                shutter.correct()
                continue
            for ev in _events():
                try:
                    _apply(lock, img, ev)
                except Exception:
                    if ev and ev[0] == "lock":
                        lock.message = _logic().FAIL
            shown = []
            found = []
            recovered = False
            coasting = lock.coasting()
            reclaim = False
            packed = None
            if lock.pattern is None:
                match = None
            else:
                try:
                    guess = lock.aim()
                    if guess is None:
                        guess = lock.search_at
                    rules = _logic()
                    tracked = None
                    appearance = lock.appearance
                    if appearance is not None and appearance.obj_mean is not None and appearance.bg_mean is not None:
                        tracked = (appearance.obj_mean, appearance.bg_mean, appearance.hot)
                    moments = None
                    hot = False
                    if appearance is not None:
                        moments = appearance.moments
                        hot = appearance.hot
                    else:
                        hot = rules.pattern_hot(lock.pattern, lock.mask)
                    match, found, recovered, border, packed = _search(
                        img,
                        lock.pattern,
                        lock.pattern_w,
                        lock.pattern_h,
                        lock.mask,
                        guess,
                        lock.search_margin(),
                        lock.coasting(),
                        hot,
                        tracked,
                        moments,
                        lock.frozen,
                        lock.frozen_mask,
                        lock.frozen_moments,
                        appearance,
                    )
                    lock.border_mean = border
                    shown = rules.shown_matches(found)
                    _note_distractors(lock, match, found, img)
                except Exception:
                    match = None
                    packed = None
            taken = None
            if coasting:
                # A hotter point in the window is not the target. Only the
                # stored size, shape, and heat may end the coast.
                taken = _take_back(img, lock, found, stamp)
                if taken is not None:
                    match = taken
                    reclaim = True
                    packed = None
                else:
                    match = None
                    packed = None
            crop = None
            reading = None
            if packed is not None and match is not None and taken is None:
                held = packed[0]
                if held is not None and len(held) == match[2] * match[3]:
                    crop = held
                    reading = packed[1]
            if match is not None and crop is None:
                crop = _crop_gray(img, match[0], match[1], match[2], match[3])
            span = window.cur
            lo = None
            hi = None
            if span and len(span) == 2:
                lo = span[0]
                hi = span[1]
            if reading is None and crop is not None and match is not None:
                reading = lock.read_crop(crop, match[2], match[3], lo, hi)
            elif reading is not None:
                lock.finish(reading, lo, hi)
            temp_c = None
            obj_max = None
            if reading is not None:
                temp_c = reading.temp_c
                obj_max = reading.obj_max
            lock.observe(
                stamp,
                match,
                temp_c,
                crop,
                recovered,
                frame_max=seen_max,
                obj_max=obj_max,
                reclaim=reclaim,
                reading=reading,
            )
            _mark_lock(img, lock)
            if coasting:
                kept = []
                for item in shown:
                    if lock.familiar(item[2], item[3]):
                        kept.append(item)
                shown = kept
            for item in shown:
                if lock.show_lock and lock.lock_box is not None and _covered(
                    item, (lock.lock_box,), lock.pattern_w, lock.pattern_h
                ):
                    continue
                _draw_box(img, item[:4])
                _draw_score(img, item[:4], item[4], lock.frame_w, lock.frame_h)
            if lock.placement is not None:
                _draw_box(img, lock.placement)
            if lock.message:
                _draw_text(img, lock.message, 4)
            _draw_state(img, lock)
            publish(img)
            _note_state(lock, reported)
            _report(lock, link, debug)
            since_gc += 1
            if since_gc >= 90:
                since_gc = 0
                gc.collect()
            _note_temp(temps)
            if debug:
                _note_meter(note_load(time.ticks_diff(time.ticks_us(), started)), _memory(), temps)
            time.sleep_ms(1)
    except KeyboardInterrupt:
        return 0
