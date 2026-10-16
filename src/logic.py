# Lock decisions for one thermal target.
#
# The rectangle's edge is the surroundings. Otsu picks the temperature class
# that differs from that edge, warmer or cooler. Cracks one pixel wide are
# closed and holes inside the silhouette are filled, then the largest region
# is the object. When that edge is too mixed to separate anything, the region
# containing the rectangle center is the object: the operator aimed the box
# so the object's center lies near the center of the box on both axes.
# The first 15 clear frames build the mask, temperature band,
# and velocity kept in memory. They also keep a 4 by 4 relative-temperature
# grid, a silhouette (fill, centroid, and an 8 by 8 edge profile), and a
# short trace of centers. A few rejected neighbors are remembered so a
# later frame can refuse them. The box size is the latest sighting. The
# pattern file is not rewritten. A distant lock, under 8 pixels on a side,
# refreshes that live pattern every 4 clear frames.
# The file and the frozen anchor stay. Confidence is the weighted average
# of the cues in cues.py.

import array

try:
    import cues
except ImportError:
    from src import cues

KEEP_MS = 20000
CLEAR_MIN = 50
SHOW_ABOVE = 70
MIN_SIDE = 8
MIN_AREA = 32
SMALL_SIDE = 4
SMALL_AREA = 16
SPOT_LIMIT = 3
CALIBRATE_EVERY = 4
DIFF_FLOOR = 10
LEARN_FRAMES = 15
SCALE_PERCENTS = (85, 100, 115)
BLEND_DEN = 8
GATE_PAD = 16
GATE_GROW = 4
GATE_CAP = 48
COAST_SEEK_MS = 2000
RETURN_SCALE = 2
FAIL = "No object can be distinguished"
AMBIGUOUS = "Two objects in the window."
TRACE_LIMIT = 12
DISTRACTOR_LIMIT = 4
DISTRACTOR_TTL = 12
GRID = 4
PROFILE = 8


def active_floors(width, height):
    """4 and 16 while either side is under 8. Otherwise 8 and 32."""
    if width < MIN_SIDE or height < MIN_SIDE:
        return SMALL_SIDE, SMALL_AREA
    return MIN_SIDE, MIN_AREA


def selection_floors(spots):
    """8 and 32, or 4 and 16 when fewer than three heat spots are visible."""
    if spots is not None and spots < SPOT_LIMIT:
        return SMALL_SIDE, SMALL_AREA
    return MIN_SIDE, MIN_AREA


def contrast_means(pattern, mask):
    """Object mean, background mean, and whether the object is the warm side.

    A missing class leaves the means empty. The object is then treated as warm.
    """
    obj = 0
    obj_n = 0
    bg = 0
    bg_n = 0
    index = 0
    count = len(pattern)
    while index < count:
        if mask is not None and not mask[index]:
            bg += pattern[index]
            bg_n += 1
        else:
            obj += pattern[index]
            obj_n += 1
        index += 1
    if obj_n < 1 or bg_n < 1:
        return None, None, True
    obj_mean = obj // obj_n
    bg_mean = bg // bg_n
    return obj_mean, bg_mean, obj_mean >= bg_mean


def pattern_hot(pattern, mask):
    """True when the object pixels are at least as warm as the surrounding fill."""
    _obj, _bg, hot = contrast_means(pattern, mask)
    return hot


def _half_toward_zero(value):
    """Half of value, rounded toward zero.

    Floor division leaves a negative camera shift stuck at -1 px/frame.
    """
    if value < 0:
        return -((-value) // 2)
    return value // 2


def gate_margin(step_x, step_y, misses):
    """Pixels of slack around the last center. Never more than GATE_CAP.

    The pad is 16 plus the last step, so the next frame of the same motion
    still lands inside. Each miss adds a few pixels. A step that would carry
    the block across the frame is cut at 48: the object does not jump.
    """
    step = step_x
    if step < 0:
        step = -step
    other = step_y
    if other < 0:
        other = -other
    if other > step:
        step = other
    pad = GATE_PAD + step
    extra = 0
    if misses > 0:
        extra = misses * GATE_GROW
    total = pad + extra
    if total > GATE_CAP:
        total = GATE_CAP
    if total < GATE_PAD:
        total = GATE_PAD
    return total


def contrast_band(obj_mean, bg_mean, hot, border_mean, border_stdev):
    """Gray band on the locked side of the midpoint, or None when the gap is gone.

    The cut stays at least DIFF_FLOOR away from the block border. Polarity is
    the sign saved at selection and does not flip.
    """
    if obj_mean is None or bg_mean is None:
        return None
    if hot:
        gap = obj_mean - bg_mean
    else:
        gap = bg_mean - obj_mean
    if gap < DIFF_FLOOR:
        return None
    mid = (obj_mean + bg_mean) // 2
    floor = blob_threshold(border_mean, border_stdev, hot)
    if hot:
        lo = mid
        if lo < floor[0]:
            lo = floor[0]
        if lo > 255:
            return None
        return (lo, 255)
    hi = mid
    if hi > floor[1]:
        hi = floor[1]
    if hi < 0:
        return None
    return (0, hi)


def _band_patch(gray, width, height, lo, hi):
    """Largest 4-connected patch inside [lo, hi]. Returns (x, y, x1, y1, count)."""
    count = width * height
    mask = bytearray(count)
    index = 0
    while index < count:
        value = gray[index] & 255
        if lo <= value <= hi:
            mask[index] = 1
        index += 1
    seen = bytearray(count)
    best = None
    best_n = 0
    index = 0
    while index < count:
        if mask[index] and not seen[index]:
            stack = [index]
            seen[index] = 1
            n = 0
            x0 = width
            y0 = height
            x1 = 0
            y1 = 0
            while stack:
                at = stack.pop()
                n += 1
                cy = at // width
                cx = at - cy * width
                if cx < x0:
                    x0 = cx
                if cy < y0:
                    y0 = cy
                if cx > x1:
                    x1 = cx
                if cy > y1:
                    y1 = cy
                if cy > 0 and mask[at - width] and not seen[at - width]:
                    seen[at - width] = 1
                    stack.append(at - width)
                if cy + 1 < height and mask[at + width] and not seen[at + width]:
                    seen[at + width] = 1
                    stack.append(at + width)
                if cx > 0 and mask[at - 1] and not seen[at - 1]:
                    seen[at - 1] = 1
                    stack.append(at - 1)
                if cx + 1 < width and mask[at + 1] and not seen[at + 1]:
                    seen[at + 1] = 1
                    stack.append(at + 1)
            if n > best_n:
                best_n = n
                best = (x0, y0, x1, y1, n)
        index += 1
    return best


def heat_bounds(gray, width, height, lo, hi, min_side=None, min_area=None):
    """Box of the largest patch inside the locked band, or None.

    A hot lock passes the high band. Pixels on the other side are surround:
    they do not extend the box, and a patch that never enters the band is
    not the object. The patch must cover the area floor and the side floor
    on both axes. Omitted floors are 8 and 32.
    """
    if min_side is None:
        min_side = MIN_SIDE
    if min_area is None:
        min_area = MIN_AREA
    if width < 1 or height < 1 or lo > hi or len(gray) < width * height:
        return None
    patch = _band_patch(gray, width, height, lo, hi)
    if patch is None or patch[4] < min_area:
        return None
    pw = patch[2] - patch[0] + 1
    ph = patch[3] - patch[1] + 1
    if pw < min_side or ph < min_side:
        return None
    return (patch[0], patch[1], pw, ph, patch[4])


def connected_bounds(gray, width, height, seed, lo, hi):
    """Box of the band component that touches seed, or None.

    seed is (x, y, w, h) inside this crop. The walk stays on pixels in
    [lo, hi], so a cold neighbor does not extend the rectangle. The result
    is at least the seed: the object can grow past a clipped edge, and a
    tighter core does not shrink the box the caller already accepted.
    """
    if width < 1 or height < 1 or lo > hi or len(gray) < width * height:
        return None
    sx, sy, sw, sh = seed
    if sw < 1 or sh < 1:
        return None
    count = width * height
    mask = bytearray(count)
    index = 0
    while index < count:
        value = gray[index] & 255
        if lo <= value <= hi:
            mask[index] = 1
        index += 1
    seen = bytearray(count)
    stack = []
    y_end = sy + sh
    if y_end > height:
        y_end = height
    x_end = sx + sw
    if x_end > width:
        x_end = width
    y = sy
    if y < 0:
        y = 0
    while y < y_end:
        x = sx
        if x < 0:
            x = 0
        while x < x_end:
            at = y * width + x
            if mask[at] and not seen[at]:
                seen[at] = 1
                stack.append(at)
            x += 1
        y += 1
    if not stack:
        return (sx, sy, sw, sh)
    x0 = width
    y0 = height
    x1 = 0
    y1 = 0
    while stack:
        at = stack.pop()
        cy = at // width
        cx = at - cy * width
        if cx < x0:
            x0 = cx
        if cy < y0:
            y0 = cy
        if cx > x1:
            x1 = cx
        if cy > y1:
            y1 = cy
        if cy > 0 and mask[at - width] and not seen[at - width]:
            seen[at - width] = 1
            stack.append(at - width)
        if cy + 1 < height and mask[at + width] and not seen[at + width]:
            seen[at + width] = 1
            stack.append(at + width)
        if cx > 0 and mask[at - 1] and not seen[at - 1]:
            seen[at - 1] = 1
            stack.append(at - 1)
        if cx + 1 < width and mask[at + 1] and not seen[at + 1]:
            seen[at + 1] = 1
            stack.append(at + 1)
    if sx < x0:
        x0 = sx
    if sy < y0:
        y0 = sy
    if sx + sw - 1 > x1:
        x1 = sx + sw - 1
    if sy + sh - 1 > y1:
        y1 = sy + sh - 1
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x1 >= width:
        x1 = width - 1
    if y1 >= height:
        y1 = height - 1
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _any_on(flags):
    index = 0
    count = len(flags)
    while index < count:
        if flags[index]:
            return True
        index += 1
    return False


def band_mask(gray, lo, hi):
    """One comparison of the crop against the locked band."""
    out = []
    index = 0
    count = len(gray)
    while index < count:
        value = gray[index] & 255
        if lo <= value <= hi:
            out.append(1)
        else:
            out.append(0)
        index += 1
    return out


def side_mean(gray, lo, hi):
    """Mean of the pixels on the locked side of the band."""
    total = 0
    count = 0
    index = 0
    limit = len(gray)
    while index < limit:
        value = gray[index] & 255
        if lo <= value <= hi:
            total += value
            count += 1
        index += 1
    if count < 1:
        return None
    return total // count


def blob_threshold(mean, stdev, hot):
    """Gray band for the object's side of the window edge. (lo, hi) inclusive."""
    delta = DIFF_FLOOR
    spread = int(stdev * 3)
    if spread > delta:
        delta = spread
    if hot:
        lo = int(mean) + delta
        if lo > 255:
            lo = 255
        return (lo, 255)
    hi = int(mean) - delta
    if hi < 0:
        hi = 0
    return (0, hi)


def is_clear(score, second):
    """A sighting is clear at 50 or better, with the runner-up at most 75%."""
    if score < CLEAR_MIN:
        return False
    return second * 4 <= score * 3


def return_box(width, height, last_w, last_h, frame_w, frame_h):
    """True when this rectangle can be the locked object coming back.

    The aspect has to match. A side more than twice the locked box is the
    runaway rectangle, and so is a side that grew out to half the frame.
    """
    if not cues.same_aspect(width, height, last_w, last_h):
        return False
    if width > last_w * RETURN_SCALE or height > last_h * RETURN_SCALE:
        return False
    if frame_w > 0 and width * 2 >= frame_w and width > last_w:
        return False
    if frame_h > 0 and height * 2 >= frame_h and height > last_h:
        return False
    return True


def return_score(width, height, last_w, last_h, shape):
    """Size and shape, with the coast direction left out.

    Each weighs 8. In the normal fuse, motion weighs 8 and size weighs 4,
    so the prediction outvotes a matching object that left the gate.
    """
    sized = cues.size(width, height, last_w, last_h)
    if sized is None or shape is None:
        return None
    if sized < SHOW_ABOVE or shape < CLEAR_MIN:
        return None
    return (sized * 8 + shape * 8) // 16


def contrast_like(sample, obj_mean, bg_mean, hot):
    """How close this mean is to the locked object mean, on the locked side.

    100 at the saved object mean. 0 on the wrong side of the background, or a
    full saved gap away from the object mean. A flat pattern has no
    correlation, so this contrast is its heat signature.
    """
    if sample is None or obj_mean is None or bg_mean is None:
        return None
    if hot and sample <= bg_mean:
        return 0
    if (not hot) and sample >= bg_mean:
        return 0
    gap = sample - obj_mean
    if gap < 0:
        gap = -gap
    span = obj_mean - bg_mean
    if span < 0:
        span = -span
    if span < DIFF_FLOOR:
        span = DIFF_FLOOR
    score = 100 - (100 * gap) // span
    if score < 0:
        return 0
    return score


def identity_score(width, height, last_w, last_h, shape, heat):
    """Size, silhouette, and heat signature, or None when one of them disagrees.

    shape is the silhouette likeness, or None when the lock has no profile
    yet. heat is the learned 4 by 4, or the pattern correlation when that
    grid is not ready. A missing profile does not block the other two.
    """
    sized = cues.size(width, height, last_w, last_h)
    if sized is None or sized < SHOW_ABOVE:
        return None
    if shape is not None and shape < CLEAR_MIN:
        return None
    if heat is None or heat < CLEAR_MIN:
        return None
    if shape is None:
        return (sized + heat) // 2
    return (sized + shape + heat) // 3


def is_ambiguous(score, second):
    """True when the runner-up is above three quarters of the leader."""
    return second * 4 > score * 3


def shown_matches(matches):
    """Detections to draw. Each match is (x, y, w, h, score). Confidence must be above 70."""
    kept = []
    for item in matches:
        if item[4] > SHOW_ABOVE:
            kept.append(item)
    return kept


def leading(matches):
    """The highest-confidence match, or None. Ties keep the earlier one."""
    best = None
    for item in matches:
        if best is None or item[4] > best[4]:
            best = item
    return best


def readout(item, frame_w, frame_h):
    """X, Y, and confidence for one box, or zeros when there is no box.

    X and Y are whole pixels from the middle of the frame to the middle of
    the box. Right and down are positive.
    """
    if item is None:
        return (0, 0, 0)
    x, y, w, h, score = item
    return (x + w // 2 - frame_w // 2, y + h // 2 - frame_h // 2, score)


def separate(gray, width, height, mask=None, min_side=None, min_area=None):
    """Return (x, y, w, h, pattern, mask) inside the rectangle, or None.

    gray is row-major samples. pattern is the object crop; pixels that are
    not part of the object are filled with the edge temperature. mask is 1
    on the object pixels of that crop. A foreground mask from the segmenter
    replaces Otsu; omitted, the thermal threshold is used. Omitted floors
    are 8 and 32.
    """
    if min_side is None:
        min_side = MIN_SIDE
    if min_area is None:
        min_area = MIN_AREA
    prepared, mean = _prepared(gray, width, height, mask, min_area)
    if prepared is None:
        return None
    best = _largest(prepared, width, height)
    if len(best) < min_area:
        return None
    x0 = width
    y0 = height
    x1 = 0
    y1 = 0
    for index in best:
        cy = index // width
        cx = index - cy * width
        if cx < x0:
            x0 = cx
        if cy < y0:
            y0 = cy
        if cx > x1:
            x1 = cx
        if cy > y1:
            y1 = cy
    pw = x1 - x0 + 1
    ph = y1 - y0 + 1
    if pw < min_side or ph < min_side:
        return None
    member = bytearray(width * height)
    for index in best:
        member[index] = 1
    pattern = []
    flags = []
    yy = y0
    while yy <= y1:
        xx = x0
        while xx <= x1:
            index = yy * width + xx
            if member[index]:
                pattern.append(gray[index] & 255)
                flags.append(1)
            else:
                pattern.append(mean & 255)
                flags.append(0)
            xx += 1
        yy += 1
    return (x0, y0, pw, ph, pattern, flags)


def _component_at(mask, width, height, sx, sy):
    """The component that contains (sx, sy), or None."""
    target = sy * width + sx
    for comp in _components(mask, width, height):
        index = 0
        count = len(comp)
        while index < count:
            if comp[index] == target:
                return comp
            index += 1
    return None


def centered(gray, width, height, min_side=None, min_area=None):
    """Object aimed at by the rectangle, when the edge cannot separate one.

    Otsu still splits the crop. The component that contains the center pixel
    is the object, and only when its own center lies near the rectangle center
    on both axes. A component that fills the rectangle is the surround.
    Returns the same tuple as separate, or None. Omitted floors are 8 and 32.
    """
    if min_side is None:
        min_side = MIN_SIDE
    if min_area is None:
        min_area = MIN_AREA
    if width < 3 or height < 3 or len(gray) < width * height:
        return None
    mean, _delta = _border(gray, width, height)
    if mean is None:
        return None
    hist = [0] * 256
    total = 0
    count = width * height
    index = 0
    while index < count:
        value = gray[index] & 255
        hist[value] += 1
        total += value
        index += 1
    split = _otsu(hist, count, total)
    if split is None:
        return None
    threshold, low_mean, high_mean, _low_count, _high_count = split
    if high_mean - low_mean < DIFF_FLOOR:
        return None
    cx = width // 2
    cy = height // 2
    use_high = (gray[cy * width + cx] & 255) > threshold
    mask = bytearray(count)
    y = 1
    while y < height - 1:
        x = 1
        while x < width - 1:
            index = y * width + x
            value = gray[index] & 255
            if use_high:
                chosen = value > threshold
            else:
                chosen = value <= threshold
            if chosen:
                mask[index] = 1
            x += 1
        y += 1
    _bridge(mask, width, height)
    _fill_holes(mask, width, height)
    _drop_specks(mask, width, height)
    comp = _component_at(mask, width, height, cx, cy)
    if comp is None or len(comp) < min_area:
        return None
    if len(comp) * 4 > count * 3:
        return None
    x0 = width
    y0 = height
    x1 = 0
    y1 = 0
    sum_x = 0
    sum_y = 0
    for index in comp:
        iy = index // width
        ix = index - iy * width
        sum_x += ix
        sum_y += iy
        if ix < x0:
            x0 = ix
        if iy < y0:
            y0 = iy
        if ix > x1:
            x1 = ix
        if iy > y1:
            y1 = iy
    pw = x1 - x0 + 1
    ph = y1 - y0 + 1
    if pw < min_side or ph < min_side:
        return None
    slack_x = width // 4
    slack_y = height // 4
    if slack_x < 1:
        slack_x = 1
    if slack_y < 1:
        slack_y = 1
    dx = sum_x - cx * len(comp)
    if dx < 0:
        dx = -dx
    dy = sum_y - cy * len(comp)
    if dy < 0:
        dy = -dy
    if dx > slack_x * len(comp) or dy > slack_y * len(comp):
        return None
    member = bytearray(count)
    for index in comp:
        member[index] = 1
    pattern = []
    flags = []
    yy = y0
    while yy <= y1:
        xx = x0
        while xx <= x1:
            index = yy * width + xx
            if member[index]:
                pattern.append(gray[index] & 255)
                flags.append(1)
            else:
                pattern.append(mean & 255)
                flags.append(0)
            xx += 1
        yy += 1
    return (x0, y0, pw, ph, pattern, flags)


def _border(gray, width, height):
    total = 0
    count = 0
    y = 0
    while y < height:
        x = 0
        while x < width:
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                total += gray[y * width + x]
                count += 1
            x += 1
        y += 1
    if count == 0:
        return None, None
    mean = total // count
    variance = 0
    y = 0
    while y < height:
        x = 0
        while x < width:
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                delta = gray[y * width + x] - mean
                variance += delta * delta
            x += 1
        y += 1
    stdev = int((variance // count) ** 0.5)
    delta = DIFF_FLOOR
    if stdev * 3 > delta:
        delta = stdev * 3
    return mean, delta


def _otsu(hist, count, total):
    """Threshold and the two class means. Pixels at or below the threshold are the low class."""
    sum_low = 0
    count_low = 0
    best = -1.0
    threshold = 0
    level = 0
    while level < 256:
        count_low += hist[level]
        if count_low != 0 and count_low != count:
            sum_low += level * hist[level]
            mean_low = sum_low / float(count_low)
            mean_high = (total - sum_low) / float(count - count_low)
            between = count_low * (count - count_low) * (mean_low - mean_high) ** 2
            if between > best:
                best = between
                threshold = level
        elif count_low != 0:
            sum_low += level * hist[level]
        level += 1
    if best <= 0:
        return None
    sum_low = 0
    count_low = 0
    level = 0
    while level <= threshold:
        count_low += hist[level]
        sum_low += level * hist[level]
        level += 1
    count_high = count - count_low
    if count_low == 0 or count_high == 0:
        return None
    return threshold, sum_low / float(count_low), (total - sum_low) / float(count_high), count_low, count_high


def _thermal_mask(gray, width, height, mean, delta, min_area=None):
    hist = [0] * 256
    total = 0
    count = width * height
    index = 0
    while index < count:
        value = gray[index] & 255
        hist[value] += 1
        total += value
        index += 1
    split = _otsu(hist, count, total)
    use_high = False
    use_low = False
    threshold = 0
    if min_area is None:
        min_area = MIN_AREA
    if split is not None:
        threshold, low_mean, high_mean, low_count, high_count = split
        if abs(high_mean - mean) >= abs(low_mean - mean) and abs(high_mean - mean) >= delta and high_count >= min_area:
            use_high = True
        elif abs(low_mean - mean) >= delta and low_count >= min_area:
            use_low = True
    mask = bytearray(count)
    y = 1
    while y < height - 1:
        x = 1
        while x < width - 1:
            index = y * width + x
            value = gray[index] & 255
            if use_high:
                chosen = value > threshold
            elif use_low:
                chosen = value <= threshold
            else:
                chosen = abs(value - mean) >= delta
            if chosen:
                mask[index] = 1
            x += 1
        y += 1
    return mask


def _bridge(mask, width, height):
    """Close a one-pixel crack so a split silhouette stays one object."""
    extra = []
    y = 1
    while y < height - 1:
        x = 1
        while x < width - 1:
            index = y * width + x
            if not mask[index]:
                neighbors = 0
                if mask[index - 1]:
                    neighbors += 1
                if mask[index + 1]:
                    neighbors += 1
                if mask[index - width]:
                    neighbors += 1
                if mask[index + width]:
                    neighbors += 1
                if neighbors >= 2:
                    extra.append(index)
            x += 1
        y += 1
    for index in extra:
        mask[index] = 1


def _fill_holes(mask, width, height):
    """Background that cannot reach the edge is inside the object."""
    seen = bytearray(width * height)
    stack = []
    x = 0
    while x < width:
        stack.append(x)
        stack.append((height - 1) * width + x)
        x += 1
    y = 1
    while y < height - 1:
        stack.append(y * width)
        stack.append(y * width + width - 1)
        y += 1
    while stack:
        index = stack.pop()
        if seen[index] or mask[index]:
            continue
        seen[index] = 1
        cy = index // width
        cx = index - cy * width
        if cy > 0:
            stack.append(index - width)
        if cy + 1 < height:
            stack.append(index + width)
        if cx > 0:
            stack.append(index - 1)
        if cx + 1 < width:
            stack.append(index + 1)
    y = 1
    while y < height - 1:
        x = 1
        while x < width - 1:
            index = y * width + x
            if not mask[index] and not seen[index]:
                mask[index] = 1
            x += 1
        y += 1


def _bits(mask, count):
    out = bytearray(count)
    index = 0
    while index < count:
        if mask[index]:
            out[index] = 1
        index += 1
    return out


def _prepared(gray, width, height, mask, min_area=None):
    """Bridge, fill, and drop specks. Otsu fills the mask when none is given."""
    if min_area is None:
        min_area = MIN_AREA
    if width < 3 or height < 3 or len(gray) < width * height:
        return None, None
    mean, delta = _border(gray, width, height)
    if mean is None:
        return None, None
    if mask is None:
        mask = _thermal_mask(gray, width, height, mean, delta, min_area)
    else:
        if len(mask) != width * height:
            return None, None
        mask = _bits(mask, width * height)
    _bridge(mask, width, height)
    _fill_holes(mask, width, height)
    _drop_specks(mask, width, height)
    return mask, mean


def _bounds(comp, width, height):
    x0 = width
    y0 = height
    x1 = 0
    y1 = 0
    for index in comp:
        cy = index // width
        cx = index - cy * width
        if cx < x0:
            x0 = cx
        if cy < y0:
            y0 = cy
        if cx > x1:
            x1 = cx
        if cy > y1:
            y1 = cy
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def sighting(gray, width, height, origin_x, origin_y, mask=None, min_side=None, min_area=None):
    """Frame box of the largest component, or None when the crop is empty.

    second is the runner-up area as a percent of the leader. Two similar
    blobs then fail is_clear, and the lock does not step onto either one.
    Omitted floors are 8 and 32.
    """
    if min_side is None:
        min_side = MIN_SIDE
    if min_area is None:
        min_area = MIN_AREA
    prepared, _mean = _prepared(gray, width, height, mask, min_area)
    if prepared is None:
        return None
    kept = []
    for comp in _components(prepared, width, height):
        if len(comp) >= min_area:
            kept.append(comp)
    if not kept:
        return None
    x0, y0, pw, ph = _bounds(kept[0], width, height)
    if pw < min_side or ph < min_side:
        return None
    second = 0
    if len(kept) > 1:
        second = len(kept[1]) * 100 // len(kept[0])
    return (origin_x + x0, origin_y + y0, pw, ph, 100, second)


def ambiguous(gray, width, height, mask, min_side=None, min_area=None):
    """True when a second blob is large enough that is_clear would refuse it."""
    hit = sighting(gray, width, height, 0, 0, mask, min_side, min_area)
    if hit is None:
        return False
    return not is_clear(hit[4], hit[5]) and hit[5] > 0


def _crop_list(gray, width, x, y, box_w, box_h):
    out = []
    yy = 0
    while yy < box_h:
        xx = 0
        while xx < box_w:
            out.append(gray[(y + yy) * width + (x + xx)])
            xx += 1
        yy += 1
    return out


def _drop_specks(mask, width, height):
    drop = []
    y = 1
    while y < height - 1:
        x = 1
        while x < width - 1:
            index = y * width + x
            if mask[index]:
                neighbors = 0
                if mask[index - 1]:
                    neighbors += 1
                if mask[index + 1]:
                    neighbors += 1
                if mask[index - width]:
                    neighbors += 1
                if mask[index + width]:
                    neighbors += 1
                if neighbors == 0:
                    drop.append(index)
            x += 1
        y += 1
    for index in drop:
        mask[index] = 0


def _components(mask, width, height):
    """Connected components, largest first. Equal areas keep scan order."""
    seen = bytearray(width * height)
    found = []
    y = 1
    while y < height - 1:
        x = 1
        while x < width - 1:
            start = y * width + x
            if mask[start] and not seen[start]:
                stack = [start]
                seen[start] = 1
                comp = []
                while stack:
                    index = stack.pop()
                    comp.append(index)
                    cy = index // width
                    cx = index - cy * width
                    if cy > 1 and mask[index - width] and not seen[index - width]:
                        seen[index - width] = 1
                        stack.append(index - width)
                    if cy + 2 < height and mask[index + width] and not seen[index + width]:
                        seen[index + width] = 1
                        stack.append(index + width)
                    if cx > 1 and mask[index - 1] and not seen[index - 1]:
                        seen[index - 1] = 1
                        stack.append(index - 1)
                    if cx + 2 < width and mask[index + 1] and not seen[index + 1]:
                        seen[index + 1] = 1
                        stack.append(index + 1)
                found.append(comp)
            x += 1
        y += 1
    found.sort(key=len, reverse=True)
    return found


def _largest(mask, width, height):
    found = _components(mask, width, height)
    if not found:
        return []
    return found[0]


def search_plain(frame, frame_w, frame_h, pattern, pattern_w, pattern_h, step=4, mask=None):
    """Best and second non-overlapping matches. Returns (x, y, w, h, score, second) or None."""
    if pattern_w < 1 or pattern_h < 1 or frame_w < pattern_w or frame_h < pattern_h:
        return None
    hits = []
    moments = _moments(pattern, pattern_w, pattern_h, mask)
    y = 0
    while y <= frame_h - pattern_h:
        x = 0
        while x <= frame_w - pattern_w:
            score = _zncc(frame, frame_w, x, y, pattern, pattern_w, pattern_h, mask, moments)
            if score > 0:
                hits.append((score, x, y))
            x += step
        y += step
    hits.sort()
    hits.reverse()
    chosen = []
    for score, x, y in hits:
        crowded = False
        for _, px, py in chosen:
            if abs(x - px) < pattern_w and abs(y - py) < pattern_h:
                crowded = True
                break
        if not crowded:
            chosen.append((score, x, y))
        if len(chosen) == 2:
            break
    if not chosen:
        return None
    score, x, y = chosen[0]
    second = chosen[1][0] if len(chosen) > 1 else 0
    return (x, y, pattern_w, pattern_h, score, second)


def resize_nearest(src, src_w, src_h, dst_w, dst_h):
    """Nearest-neighbor resize of a row-major list."""
    out = []
    y = 0
    while y < dst_h:
        sy = y * src_h // dst_h
        x = 0
        while x < dst_w:
            sx = x * src_w // dst_w
            out.append(src[sy * src_w + sx])
            x += 1
        y += 1
    return out


def scaled_sizes(box_w, box_h, frame_w, frame_h, min_side=None):
    """Three sizes around the last box, clamped to the frame and the side floor.

    The floor stays 8 unless the caller passes a smaller one. A normal box
    therefore does not grow a 4 pixel search.
    """
    if min_side is None:
        min_side = MIN_SIDE
    sizes = []
    for pct in SCALE_PERCENTS:
        dw = box_w * pct // 100
        dh = box_h * pct // 100
        if dw < min_side:
            dw = min_side
        if dh < min_side:
            dh = min_side
        if dw > frame_w:
            dw = frame_w
        if dh > frame_h:
            dh = frame_h
        if dw < min_side or dh < min_side:
            continue
        pair = (dw, dh)
        if pair not in sizes:
            sizes.append(pair)
    return sizes


def search_scales(frame, frame_w, frame_h, pattern, pattern_w, pattern_h, mask, box_w, box_h, step=4):
    """Best scale around the last box. The returned width and height are that scale."""
    best = None
    side, _area = active_floors(box_w, box_h)
    for dw, dh in scaled_sizes(box_w, box_h, frame_w, frame_h, side):
        scaled = resize_nearest(pattern, pattern_w, pattern_h, dw, dh)
        scaled_mask = None
        if mask is not None:
            scaled_mask = resize_nearest(mask, pattern_w, pattern_h, dw, dh)
        hit = search_plain(
            frame, frame_w, frame_h, scaled, dw, dh, step=step, mask=scaled_mask
        )
        if hit is None:
            continue
        if best is None or hit[4] > best[4] or (
            hit[4] == best[4] and hit[2] * hit[3] > best[2] * best[3]
        ):
            best = hit
    return best


def object_celsius(gray, mask, lo, hi):
    """Mean Celsius of the object pixels. lo and hi are the frame's Kelvin*64 window."""
    total = 0
    count = 0
    index = 0
    limit = len(gray)
    while index < limit:
        if mask is None or mask[index]:
            total += gray[index]
            count += 1
        index += 1
    if count < 1 or hi <= lo:
        return None
    pixel = total / float(count)
    kelvin64 = lo + pixel * (hi - lo) / 255.0
    return kelvin64 / 64.0 - 273.15


def masked_max(gray, mask):
    """Highest gray count on the object pixels, or None when there are none.

    Background pixels in the box are skipped, so a hot surround does not
    count as the object's maximum.
    """
    if gray is None:
        return None
    best = None
    index = 0
    limit = len(gray)
    while index < limit:
        if mask is None or mask[index]:
            value = gray[index]
            if best is None or value > best:
                best = value
        index += 1
    return best


def mask_on_grid(crop, crop_w, crop_h, ref_mask, ref_w, ref_h, foreground=None):
    """Object mask of a matched crop, resized onto the lock pattern.

    foreground is the segmenter mask for this crop. A mismatched length falls
    back to Otsu inside separate.
    """
    if foreground is not None and len(foreground) != crop_w * crop_h:
        foreground = None
    found = separate(crop, crop_w, crop_h, foreground)
    if found is None:
        return resize_nearest(ref_mask, ref_w, ref_h, ref_w, ref_h)
    x, y, width, height, _, flags = found
    full = []
    index = 0
    while index < crop_w * crop_h:
        full.append(0)
        index += 1
    index = 0
    yy = 0
    while yy < height:
        xx = 0
        while xx < width:
            if flags[index]:
                full[(y + yy) * crop_w + (x + xx)] = 1
            index += 1
            xx += 1
        yy += 1
    return resize_nearest(full, crop_w, crop_h, ref_w, ref_h)


class Kalman(object):
    """Constant-velocity Kalman filter. One update is one frame.

    The position gain is 1, so the state sits on the accepted sighting.
    The velocity gain is 1 on the first step and 1/2 after that, the fixed
    gain of this lock. The half rounds toward zero, so a left or up shift
    decays away instead of sticking. predict() does not move the state.
    coast() adds the recorded velocity to the position and leaves the
    velocity unchanged, so a hide walks forward in a straight line instead
    of teaching a new step.
    """

    def __init__(self):
        self.x = None
        self.y = None
        self.vx = 0
        self.vy = 0
        self.moves = 0

    def predict(self):
        if self.x is None:
            return None
        return (self.x + self.vx, self.y + self.vy)

    def coast(self):
        """One hidden frame. The position steps by the recorded velocity."""
        if self.x is None:
            return
        self.x += self.vx
        self.y += self.vy

    def update(self, zx, zy):
        if self.x is None:
            self.x = zx
            self.y = zy
            return
        dx = zx - self.x
        dy = zy - self.y
        if self.moves == 0:
            self.vx = dx
            self.vy = dy
        else:
            self.vx = _half_toward_zero(self.vx + dx)
            self.vy = _half_toward_zero(self.vy + dy)
        self.moves += 1
        self.x = zx
        self.y = zy

    def hold(self):
        """A shutter frame does not change velocity."""
        return


class Distractor(object):
    """A neighbor that was not the lock: offset, area, and gray mean."""

    def __init__(self, dx, dy, area, gray, frame):
        self.dx = dx
        self.dy = dy
        self.area = area
        self.gray = gray
        self.frame = frame


def gray_mean(crop):
    """Integer mean of a crop, or None when the crop is empty."""
    if crop is None:
        return None
    limit = len(crop)
    if limit < 1:
        return None
    total = 0
    index = 0
    while index < limit:
        total += crop[index]
        index += 1
    return total // limit


def thermal_grid(gray, mask, width, height, lo=None, hi=None):
    """4 by 4 gray values relative to the object mean, and which cells exist.

    The object mean uses the mask, or the band [lo, hi] when that is passed
    so the caller does not build a mask first. Each cell averages every pixel
    in that cell, so the surround stays in the signature. A uniform hot core
    on a cool ground is then a different grid from a flat blob of the same
    mean. An empty crop returns None. A cell with no pixels stays invalid.
    The mean and the cells are one walk.
    """
    if gray is None or width < 1 or height < 1:
        return None
    count = width * height
    if len(gray) < count:
        return None
    use_band = lo is not None
    obj = 0
    obj_n = 0
    cell_sum = []
    cell_n = []
    index = 0
    while index < GRID * GRID:
        cell_sum.append(0)
        cell_n.append(0)
        index += 1
    gy = 0
    while gy < GRID:
        y0 = gy * height // GRID
        y1 = (gy + 1) * height // GRID
        gx = 0
        while gx < GRID:
            x0 = gx * width // GRID
            x1 = (gx + 1) * width // GRID
            slot = gy * GRID + gx
            acc = 0
            number = 0
            y = y0
            while y < y1:
                row = y * width
                x = x0
                while x < x1:
                    value = gray[row + x]
                    acc += value
                    number += 1
                    marked = False
                    if use_band:
                        marked = lo <= (value & 255) <= hi
                    elif mask is None or mask[row + x]:
                        marked = True
                    if marked:
                        obj += value
                        obj_n += 1
                    x += 1
                y += 1
            cell_sum[slot] = acc
            cell_n[slot] = number
            gx += 1
        gy += 1
    if obj_n < 1:
        return None
    mean = obj / float(obj_n)
    cells = []
    valid = bytearray(GRID * GRID)
    index = 0
    while index < GRID * GRID:
        number = cell_n[index]
        if number < 1:
            cells.append(0)
        else:
            cells.append(int(cell_sum[index] / float(number) - mean))
            valid[index] = 1
        index += 1
    return (cells, valid)


def grid_band(obj_mean, bg_mean, hot, border_mean):
    """The 4 by 4 and the silhouette use the border mean, not its spread."""
    return contrast_band(obj_mean, bg_mean, hot, border_mean, 0)


def _bins(length, parts):
    """Which of `parts` slots each index falls in."""
    found = []
    index = 0
    while index < length:
        found.append(index * parts // length)
        index += 1
    return found


def _peak_cell(sample):
    """Index of the hottest valid cell, or None."""
    if sample is None:
        return None
    cells, valid = sample
    best_i = None
    best_v = None
    index = 0
    limit = len(cells)
    while index < limit:
        if valid[index]:
            value = cells[index]
            if best_i is None or value > best_v:
                best_v = value
                best_i = index
        index += 1
    return best_i


def silhouette(mask, width, height):
    """Fill, centroid offset, and the 8-bin edge profile of a mask.

    Each bin is the percent of that eighth of the box that is object.
    The offset is the centroid minus the box center, in percent of the
    side, from -100 to 100. None when the mask is empty.
    """
    if mask is None or width < 1 or height < 1:
        return None
    count = width * height
    if len(mask) < count:
        return None
    x_bin = _bins(width, PROFILE)
    y_bin = _bins(height, PROFILE)
    row_obj = []
    row_all = []
    col_obj = []
    col_all = []
    index = 0
    while index < PROFILE:
        row_obj.append(0)
        row_all.append(0)
        col_obj.append(0)
        col_all.append(0)
        index += 1
    sum_x = 0
    sum_y = 0
    obj = 0
    y = 0
    index = 0
    while y < height:
        yb = y_bin[y]
        x = 0
        while x < width:
            row_all[yb] += 1
            col_all[x_bin[x]] += 1
            if mask[index]:
                obj += 1
                sum_x += x
                sum_y += y
                row_obj[yb] += 1
                col_obj[x_bin[x]] += 1
            x += 1
            index += 1
        y += 1
    if obj < 1:
        return None
    return (
        _occupancy(row_obj, row_all),
        _occupancy(col_obj, col_all),
        obj * 100 // count,
        (sum_x // obj - width // 2) * 100 // width,
        (sum_y // obj - height // 2) * 100 // height,
    )


def _occupancy(obj, all_pixels):
    rows = []
    index = 0
    while index < len(obj):
        if all_pixels[index] < 1:
            rows.append(0)
        else:
            rows.append(obj[index] * 100 // all_pixels[index])
        index += 1
    return rows


def profile_like(rows, cols, fill, peak, mark_rows, mark_cols, mark_fill, mark_peak):
    """How well a live silhouette matches the one saved with the lock, 0..100.

    None when the lock has no profile yet. The hottest cell is a landmark:
    the same cell scores 100, a different one still contributes, so a slow
    turn does not wipe the match.
    """
    if mark_rows is None or rows is None:
        return None
    total = 0
    parts = 0
    if fill is not None and mark_fill is not None:
        gap = fill - mark_fill
        if gap < 0:
            gap = -gap
        if gap > 100:
            gap = 100
        total += 100 - gap
        parts += 1
    diff = 0
    number = 0
    index = 0
    limit = len(rows)
    if mark_rows is not None and len(mark_rows) < limit:
        limit = len(mark_rows)
    while index < limit:
        gap = rows[index] - mark_rows[index]
        if gap < 0:
            gap = -gap
        diff += gap
        number += 1
        index += 1
    if cols is not None and mark_cols is not None:
        index = 0
        limit = len(cols)
        if len(mark_cols) < limit:
            limit = len(mark_cols)
        while index < limit:
            gap = cols[index] - mark_cols[index]
            if gap < 0:
                gap = -gap
            diff += gap
            number += 1
            index += 1
    if number > 0:
        mean = diff // number
        if mean > 100:
            mean = 100
        total += 100 - mean
        parts += 1
    if peak is not None and mark_peak is not None:
        if peak == mark_peak:
            total += 100
        else:
            total += 40
        parts += 1
    if parts < 1:
        return None
    return total // parts


class Reading(object):
    """Everything one walk of a crop can say about the locked object."""

    def __init__(self):
        self.appearance = None
        self.anchor = None
        self.have_anchor = False
        self.obj_max = None
        self.obj_sum = 0
        self.obj_n = 0
        self.temp_c = None
        self.side = None
        self.gray = None
        self.cells = None
        self.valid = None
        self.structure = None
        self.rows = None
        self.cols = None
        self.fill = None
        self.peak = None


def survey(crop, width, height, pattern, mask, moments, frozen, frozen_mask, frozen_moments, band):
    """Appearance, anchor, temperature, and silhouette in one walk.

    `crop` is already `width` by `height`, the pattern grid. Integer sums
    match `_zncc`. The 4 by 4 uses `band` the way `thermal_grid` does, so
    the structure cue does not walk the crop again.
    """
    reading = Reading()
    if crop is None or width < 1 or height < 1:
        return reading
    count = width * height
    if len(crop) < count:
        return reading
    x_cell = _bins(width, GRID)
    y_cell = _bins(height, GRID)
    x_bin = _bins(width, PROFILE)
    y_bin = _bins(height, PROFILE)
    do_app = pattern is not None and len(pattern) >= count
    do_anchor = frozen is not None and len(frozen) >= count
    if do_app and moments is None:
        moments = _moments(pattern, width, height, mask)
    if do_anchor and frozen_moments is None:
        frozen_moments = _moments(frozen, width, height, frozen_mask)
    app_n = 0
    sum_p = 0
    sum_pp = 0
    if do_app and moments is not None:
        app_n, sum_p, sum_pp = moments
    anch_n = 0
    fsum_p = 0
    fsum_pp = 0
    if do_anchor and frozen_moments is not None:
        anch_n, fsum_p, fsum_pp = frozen_moments
    sum_f = 0
    sum_ff = 0
    sum_fp = 0
    fsum_f = 0
    fsum_ff = 0
    fsum_fp = 0
    gray_sum = 0
    obj_sum = 0
    obj_n = 0
    obj_max = None
    side_sum = 0
    side_n = 0
    band_sum = 0
    band_n = 0
    cell_sum = []
    cell_n = []
    index = 0
    while index < GRID * GRID:
        cell_sum.append(0)
        cell_n.append(0)
        index += 1
    row_obj = []
    row_all = []
    col_obj = []
    col_all = []
    index = 0
    while index < PROFILE:
        row_obj.append(0)
        row_all.append(0)
        col_obj.append(0)
        col_all.append(0)
        index += 1
    use_band = band is not None and band[0] <= band[1]
    lo = 0
    hi = 255
    if use_band:
        lo = band[0]
        hi = band[1]
    y = 0
    index = 0
    while y < height:
        yc = y_cell[y]
        yb = y_bin[y]
        x = 0
        while x < width:
            value = crop[index]
            gray = value & 255
            gray_sum += value
            slot = yc * GRID + x_cell[x]
            cell_sum[slot] += value
            cell_n[slot] += 1
            cb = x_bin[x]
            row_all[yb] += 1
            col_all[cb] += 1
            if use_band and lo <= gray <= hi:
                band_sum += value
                band_n += 1
                side_sum += gray
                side_n += 1
                row_obj[yb] += 1
                col_obj[cb] += 1
            on_mask = mask is None or (index < len(mask) and mask[index])
            if do_app and on_mask:
                pref = pattern[index]
                sum_f += value
                sum_ff += value * value
                sum_fp += value * pref
                if obj_max is None or value > obj_max:
                    obj_max = value
                obj_sum += value
                obj_n += 1
            if do_anchor and (frozen_mask is None or (index < len(frozen_mask) and frozen_mask[index])):
                pref = frozen[index]
                fsum_f += value
                fsum_ff += value * value
                fsum_fp += value * pref
            x += 1
            index += 1
        y += 1
    if count > 0:
        reading.gray = gray_sum // count
    if do_app:
        reading.appearance = _ratio(app_n, sum_f, sum_p, sum_ff, sum_pp, sum_fp)
        reading.obj_sum = obj_sum
        reading.obj_n = obj_n
        reading.obj_max = obj_max
    if do_anchor:
        reading.anchor = _ratio(anch_n, fsum_f, fsum_p, fsum_ff, fsum_pp, fsum_fp)
        reading.have_anchor = True
    if use_band and side_n > 0:
        reading.side = side_sum // side_n
    if use_band and band_n > 0:
        mean = band_sum / float(band_n)
        cells = []
        valid = bytearray(GRID * GRID)
        index = 0
        while index < GRID * GRID:
            number = cell_n[index]
            if number < 1:
                cells.append(0)
            else:
                cells.append(int(cell_sum[index] / float(number) - mean))
                valid[index] = 1
            index += 1
        reading.cells = cells
        reading.valid = valid
        reading.peak = _peak_cell((cells, valid))
        reading.rows = _occupancy(row_obj, row_all)
        reading.cols = _occupancy(col_obj, col_all)
        reading.fill = band_n * 100 // count
    return reading


def _distractor_score(cx, cy, area, gray, record, origin, reach):
    """How well this candidate agrees with one remembered neighbor, 0..100."""
    if reach < 1:
        reach = 1
    expect_x = origin[0] + record.dx
    expect_y = origin[1] + record.dy
    dx = cx - expect_x
    dy = cy - expect_y
    dist2 = dx * dx + dy * dy
    if dist2 >= reach * reach:
        near = 0
    else:
        dist = dist2 ** 0.5
        near = int(100 - (100.0 * dist) / reach)
        if near < 0:
            near = 0
    stored = record.area
    if stored < 1:
        stored = 1
    delta = area - stored
    if delta < 0:
        delta = -delta
    if delta >= stored:
        area_score = 0
    else:
        area_score = 100 - (100 * delta) // stored
    total = near + area_score
    mass = 2
    if gray is not None and record.gray is not None:
        gap = gray - record.gray
        if gap < 0:
            gap = -gap
        if gap >= DIFF_FLOOR:
            tone = 0
        else:
            tone = 100 - (100 * gap) // DIFF_FLOOR
        total += tone
        mass += 1
    return total // mass


class Appearance(object):
    """Mask, temperature, velocity, and silhouette from the locked object.

    The box is the latest accepted sighting. A closer camera may grow both
    sides together. After the freeze the learn window is a 4 by 4 temperature
    grid. The silhouette is the fill, the centroid offset, an 8-bin edge
    profile, and the hottest cell. It is taken from the mask at lock and
    again when that mask freezes or a distant target adopts a new size. It
    is not blended every frame, so a distractor cannot walk the signature
    onto itself. A short trace of centers stays for later frames. The mask
    already drops a pixel that was off on half of those frames, so
    correlation does not keep a second per-pixel weight.
    """

    def __init__(self, pattern, mask, width, height):
        self.pattern = []
        self.mask = []
        index = 0
        count = width * height
        while index < count:
            self.pattern.append(pattern[index])
            self.mask.append(1 if mask[index] else 0)
            index += 1
        self.width = width
        self.height = height
        self.samples = 0
        self.counts = []
        self.sums = []
        index = 0
        while index < count:
            self.counts.append(0)
            self.sums.append(0)
            index += 1
        self.moments = None
        self.grid_sums = []
        self.grid_counts = []
        index = 0
        while index < GRID * GRID:
            self.grid_sums.append(0)
            self.grid_counts.append(0)
            index += 1
        self.grid = None
        self.grid_valid = None
        self.trace_x = array.array("i")
        self.trace_y = array.array("i")
        self.box_w = width
        self.box_h = height
        self.temp_sum = 0.0
        self.temp_sq = 0.0
        self.temp_n = 0
        self.temp_mean = None
        self.temp_spread = None
        self.motion = Kalman()
        self.vx = 0
        self.vy = 0
        self.moves = 0
        self.last_x = None
        self.last_y = None
        obj_mean, bg_mean, hot = contrast_means(self.pattern, self.mask)
        self.obj_mean = obj_mean
        self.bg_mean = bg_mean
        self.hot = hot
        self.rows = None
        self.cols = None
        self.fill = None
        self.ox = None
        self.oy = None
        self.peak = None
        self._bind()
        self._note_shape()

    def _bind(self):
        """Pattern energy for the next correlations. Rebuilt when the pattern changes."""
        self.moments = _moments(self.pattern, self.width, self.height, self.mask)

    def sizes(self, frame_w, frame_h):
        side, _area = active_floors(self.box_w, self.box_h)
        return scaled_sizes(self.box_w, self.box_h, frame_w, frame_h, side)

    def see(self, cx, cy):
        self.motion.update(cx, cy)
        self.last_x = self.motion.x
        self.last_y = self.motion.y
        self.vx = self.motion.vx
        self.vy = self.motion.vy
        self.moves = self.motion.moves
        self.trace_x.append(int(cx))
        self.trace_y.append(int(cy))
        extra = len(self.trace_x) - TRACE_LIMIT
        if extra > 0:
            self.trace_x = self.trace_x[extra:]
            self.trace_y = self.trace_y[extra:]

    def trace(self):
        """Recent accepted centers, oldest first. At most TRACE_LIMIT."""
        points = []
        index = 0
        limit = len(self.trace_x)
        while index < limit:
            points.append((self.trace_x[index], self.trace_y[index]))
            index += 1
        return points

    def hold(self):
        """A shutter frame does not change velocity."""
        self.motion.hold()

    def coast(self):
        """A hidden frame steps the position and leaves the velocity recorded."""
        self.motion.coast()
        self.last_x = self.motion.x
        self.last_y = self.motion.y

    def park(self, cx, cy):
        """Sit the filter on this center and keep the velocity already recorded."""
        self.motion.x = cx
        self.motion.y = cy
        self.last_x = cx
        self.last_y = cy

    def predict(self):
        return self.motion.predict()

    def block_reach(self, misses):
        """Distance from the prediction to the edge of the search block."""
        reach = gate_margin(self.vx, self.vy, misses)
        half = self.box_w // 2
        other = self.box_h // 2
        if other > half:
            half = other
        return reach + half

    def near(self, cx, cy, misses=0):
        """True when the center is still inside the capped gate around the last one.

        The first sighting has no previous center and is accepted. After that,
        a far heat source is a different object, including during the first
        15 frames.
        """
        if self.last_x is None:
            return True
        reach = gate_margin(self.vx, self.vy, misses)
        dx = cx - self.last_x
        dy = cy - self.last_y
        return dx * dx + dy * dy <= reach * reach

    def same(self, width, height, temp_c, cx, cy, recovered, misses=0):
        """A sighting must stay near the last center.

        After the first 15 frames it must also match aspect and temperature.
        recovered is ignored. A sighting outside the block is a different object.
        """
        if not self.near(cx, cy, misses):
            return False
        if self.samples < LEARN_FRAMES:
            return True
        if not self._size_ok(width, height):
            return False
        return self._temp_ok(temp_c)

    def note_contrast(self, obj_mean, bg_mean):
        """Move the saved means. The object mean stays on the locked side."""
        if bg_mean is not None:
            self.bg_mean = int(bg_mean)
        if obj_mean is None:
            return
        if self.bg_mean is None:
            self.obj_mean = int(obj_mean)
            return
        if self.hot and obj_mean < self.bg_mean:
            return
        if not self.hot and obj_mean > self.bg_mean:
            return
        self.obj_mean = int(obj_mean)

    def adopt(self, gray, mask, box_w, box_h):
        """Replace the live grid with this crop. The temperature record stays.

        A lock still inside its first 15 frames starts that count over, so
        the new size can fill the accumulator. A finished count stays put.
        """
        count = box_w * box_h
        pattern = []
        flags = []
        counts = []
        sums = []
        index = 0
        while index < count:
            pattern.append(gray[index] & 255)
            if mask[index]:
                flags.append(1)
            else:
                flags.append(0)
            counts.append(0)
            sums.append(0)
            index += 1
        self.pattern = pattern
        self.mask = flags
        self.counts = counts
        self.sums = sums
        self.grid = None
        self.grid_valid = None
        self.grid_sums = []
        self.grid_counts = []
        index = 0
        while index < GRID * GRID:
            self.grid_sums.append(0)
            self.grid_counts.append(0)
            index += 1
        self.width = box_w
        self.height = box_h
        self.box_w = box_w
        self.box_h = box_h
        self._bind()
        self._note_shape()
        if self.samples < LEARN_FRAMES:
            self.samples = 0

    def record_grid(self, gray, mask, box_w, box_h, cx, cy, temp_c):
        """gray and mask are on the lock-pattern grid."""
        self.see(cx, cy)
        self._note_size(box_w, box_h)
        if self.samples < LEARN_FRAMES:
            self._note_temp(temp_c)
            self._accumulate(gray, mask)
            self._note_grid(gray, mask)
            self.samples += 1
            if self.samples == LEARN_FRAMES:
                self._freeze()
            return
        self._blend(gray)

    def _note_size(self, box_w, box_h):
        self.box_w = box_w
        self.box_h = box_h

    def _note_temp(self, temp_c):
        if temp_c is None:
            return
        self.temp_sum += temp_c
        self.temp_sq += temp_c * temp_c
        self.temp_n += 1

    def _accumulate(self, gray, mask):
        index = 0
        count = self.width * self.height
        while index < count:
            self.sums[index] += gray[index]
            if mask[index]:
                self.counts[index] += 1
            index += 1

    def _note_grid(self, gray, mask):
        """Add this frame's relative 4 by 4 to the learn sums."""
        sample = thermal_grid(gray, mask, self.width, self.height)
        if sample is None:
            return
        cells, valid = sample
        index = 0
        while index < GRID * GRID:
            if valid[index]:
                self.grid_sums[index] += cells[index]
                self.grid_counts[index] += 1
            index += 1

    def _freeze(self):
        pattern = []
        mask = []
        index = 0
        count = self.width * self.height
        while index < count:
            seen = self.counts[index]
            if seen > 0:
                pattern.append(self.sums[index] // seen)
            else:
                pattern.append(self.pattern[index])
            if seen * 2 > LEARN_FRAMES:
                mask.append(1)
            else:
                mask.append(0)
            index += 1
        self.pattern = pattern
        self.mask = mask
        self._freeze_grid()
        self.counts = []
        self.sums = []
        self._bind()
        self._note_shape()
        if self.temp_n > 0:
            mean = self.temp_sum / float(self.temp_n)
            var = self.temp_sq / float(self.temp_n) - mean * mean
            if var < 0:
                var = 0
            self.temp_mean = mean
            self.temp_spread = var ** 0.5

    def _freeze_grid(self):
        grid = array.array("h")
        valid = bytearray(GRID * GRID)
        index = 0
        while index < GRID * GRID:
            seen = self.grid_counts[index]
            if seen * 2 > LEARN_FRAMES:
                grid.append(self.grid_sums[index] // seen)
                valid[index] = 1
            else:
                grid.append(0)
            index += 1
        self.grid = grid
        self.grid_valid = valid
        self.grid_sums = []
        self.grid_counts = []

    def _blend(self, gray):
        """Seven eighths of the stored pixel plus one eighth of the new crop.

        The pattern list is updated in place. A fresh list every accepted
        frame is what made the heap collect while the lock was held. The
        pattern energy moves with the same pixels, so the next correlation
        does not walk the pattern again.
        """
        moments = self.moments
        if moments is None:
            self._bind()
            moments = self.moments
        count_m, sum_p, sum_pp = moments
        index = 0
        count = self.width * self.height
        pattern = self.pattern
        mask = self.mask
        while index < count:
            if mask[index]:
                old = pattern[index]
                new = (old * (BLEND_DEN - 1) + gray[index]) // BLEND_DEN
                if new != old:
                    pattern[index] = new
                    sum_p += new - old
                    sum_pp += new * new - old * old
            index += 1
        self.moments = (count_m, sum_p, sum_pp)

    def _note_shape(self):
        """Silhouette from the mask. The hottest cell prefers the learned grid."""
        found = silhouette(self.mask, self.width, self.height)
        if found is None:
            self.rows = None
            self.cols = None
            self.fill = None
            self.ox = None
            self.oy = None
        else:
            self.rows = found[0]
            self.cols = found[1]
            self.fill = found[2]
            self.ox = found[3]
            self.oy = found[4]
        if self.grid is not None and self.grid_valid is not None:
            self.peak = _peak_cell((self.grid, self.grid_valid))
            return
        self.peak = _peak_cell(thermal_grid(self.pattern, self.mask, self.width, self.height))

    def _size_ok(self, width, height):
        """A closer camera may grow both sides. A broken aspect is a different object."""
        return cues.same_aspect(width, height, self.box_w, self.box_h)

    def _temp_ok(self, temp_c):
        if temp_c is None or self.temp_mean is None:
            return True
        limit = self.temp_spread * 3
        if limit < 2:
            limit = 2
        if temp_c < self.temp_mean - limit:
            return False
        if temp_c > self.temp_mean + limit:
            return False
        return True


def _moments(pattern, pattern_w, pattern_h, mask):
    """Count, sum, and sum of squares of the masked pattern.

    Correlation reuses this. The pattern side does not change between sightings.
    Sums stay integers: the camera has no fast double, and the ratio is taken
    once at the end.
    """
    count = 0
    sum_p = 0
    sum_pp = 0
    index = 0
    limit = pattern_w * pattern_h
    while index < limit:
        if mask is None or mask[index]:
            value = pattern[index]
            sum_p += value
            sum_pp += value * value
            count += 1
        index += 1
    return (count, sum_p, sum_pp)


def _ratio(count, sum_f, sum_p, sum_ff, sum_pp, sum_fp):
    """0..100 from the one-pass sums. A negative correlation is 0."""
    if count < 1:
        return 0
    num = count * sum_fp - sum_f * sum_p
    if num <= 0:
        return 0
    den_f = count * sum_ff - sum_f * sum_f
    den_p = count * sum_pp - sum_p * sum_p
    if den_f <= 0 or den_p <= 0:
        return 0
    score = int(num / ((den_f * den_p) ** 0.5) * 100.0 + 0.5)
    if score > 100:
        return 100
    return score


def _zncc(frame, frame_w, x, y, pattern, pattern_w, pattern_h, mask=None, moments=None):
    """Masked correlation of the pattern at (x, y), as an integer 0..100.

    One walk over the masked pixels. Passing moments skips recomputing the
    pattern energy, which is the steady lock: the same pattern is scored
    against every blob. Sums are integers. A float ratio is taken once.
    """
    if moments is None:
        moments = _moments(pattern, pattern_w, pattern_h, mask)
    count, sum_p, sum_pp = moments
    if count < 1:
        return 0
    sum_f = 0
    sum_ff = 0
    sum_fp = 0
    if x == 0 and y == 0 and frame_w == pattern_w:
        index = 0
        limit = pattern_w * pattern_h
        while index < limit:
            if mask is None or mask[index]:
                sample = frame[index]
                sum_f += sample
                sum_ff += sample * sample
                sum_fp += sample * pattern[index]
            index += 1
    else:
        py = 0
        while py < pattern_h:
            row = (y + py) * frame_w + x
            base = py * pattern_w
            px = 0
            while px < pattern_w:
                index = base + px
                if mask is None or mask[index]:
                    sample = frame[row + px]
                    sum_f += sample
                    sum_ff += sample * sample
                    sum_fp += sample * pattern[index]
                px += 1
            py += 1
    return _ratio(count, sum_f, sum_p, sum_ff, sum_pp, sum_fp)


class Lock(object):
    """Selection, the saved pattern, and the console line for one run."""

    def __init__(self, paths, segmenter=None):
        self.paths = paths
        self.segmenter = segmenter
        self.drawing = False
        self.placement = None
        self.pattern = None
        self.mask = None
        self.pattern_w = 0
        self.pattern_h = 0
        self.frame_w = 256
        self.frame_h = 192
        self.center = None
        self.lock_box = None
        self.base_conf = 0
        self.unclear_since = None
        self._shutter_mark = None
        self.message = None
        self.line = (0, 0, 0)
        self.show_lock = False
        self.appearance = None
        self.cue_line = None
        self.cues = None
        self.search_at = None
        self.frozen = None
        self.frozen_mask = None
        self.frozen_moments = None
        self.misses = 0
        self.border_mean = None
        self.step_x = 0
        self.step_y = 0
        self.distant = False
        self.clear_seen = 0
        self.distractors = []
        self.frame_n = 0

    def startup(self):
        self.unlock()

    def place(self, box):
        if not self.drawing:
            return False
        self.placement = box
        self.message = None
        return True

    def select(self):
        """Arm a new rectangle. The previous uncommitted box is dropped."""
        self.drawing = True
        self.placement = None
        return "arm"

    def lock(self, gray, width, height, spots=None):
        """Save the rectangle and keep that object box on screen.

        spots is how many heat spots the frame shows. Fewer than three lowers
        the floor to 4 by 4. Omitted, the floor stays 8 by 8.
        """
        if not self.drawing or self.placement is None:
            return "idle"
        min_side, min_area = selection_floors(spots)
        mask = self._segment(gray, width, height, False)
        if mask is not None and ambiguous(gray, width, height, mask, min_side, min_area):
            self.message = AMBIGUOUS
            return "ambiguous"
        if mask is not None:
            found = separate(gray, width, height, mask, min_side, min_area)
        else:
            found = separate(gray, width, height, None, min_side, min_area)
        if found is None:
            found = centered(gray, width, height, min_side, min_area)
        if found is None:
            self.message = FAIL
            return "fail"
        rx, ry, pw, ph, pattern, flags = found
        x, y, _, _ = self.placement
        box = (x + rx, y + ry, pw, ph)
        if not self._write(self.frame_w, self.frame_h, box, pw, ph, pattern, flags):
            self.message = FAIL
            return "fail"
        if not self._read():
            self.message = FAIL
            return "fail"
        self.drawing = False
        self.placement = None
        self.message = None
        self.center = None
        self.lock_box = box
        self.show_lock = True
        self.line = self._box_offset(0)
        self.base_conf = 0
        self.unclear_since = None
        self._shutter_mark = None
        self.misses = 0
        self.border_mean = None
        self.distant = pw < MIN_SIDE or ph < MIN_SIDE
        self.clear_seen = 0
        self.distractors = []
        self.frame_n = 0
        self._arm()
        self._remember_contrast(gray, width, height, flags)
        return "ok"

    def _remember_contrast(self, gray, width, height, flags):
        """Object mean from the silhouette, background mean from the selection edge."""
        if self.appearance is None:
            return
        border, _delta = _border(gray, width, height)
        if border is None:
            return
        total = 0
        count = 0
        index = 0
        limit = len(flags)
        while index < limit:
            if flags[index]:
                total += self.appearance.pattern[index]
                count += 1
            index += 1
        if count < 1:
            return
        obj = total // count
        self.appearance.obj_mean = obj
        self.appearance.bg_mean = border
        self.appearance.hot = obj >= border

    def unlock(self):
        self._delete()
        self.drawing = False
        self.placement = None
        self.pattern = None
        self.mask = None
        self.pattern_w = 0
        self.pattern_h = 0
        self.center = None
        self.lock_box = None
        self.base_conf = 0
        self.unclear_since = None
        self._shutter_mark = None
        self.message = None
        self.line = (0, 0, 0)
        self.show_lock = False
        self.appearance = None
        self.cue_line = None
        self.cues = None
        self.search_at = None
        self.frozen = None
        self.frozen_mask = None
        self.frozen_moments = None
        self.misses = 0
        self.border_mean = None
        self.step_x = 0
        self.step_y = 0
        self.distant = False
        self.clear_seen = 0
        self.distractors = []
        self.frame_n = 0

    def predict(self):
        if self.appearance is None:
            return None
        return self.appearance.predict()

    def _step(self):
        """The larger motion on each axis, for the margin only.

        The aim does not follow this vector. Splicing the drawn step with
        the learned velocity builds a direction neither one had, and a
        camera shift then walks the window off the object.
        """
        step_x = self.step_x
        step_y = self.step_y
        if self.appearance is not None:
            if abs(self.appearance.vx) > abs(step_x):
                step_x = self.appearance.vx
            if abs(self.appearance.vy) > abs(step_y):
                step_y = self.appearance.vy
        return step_x, step_y

    def search_margin(self):
        """Slack around the last box: a pad of 16, the last step, and a little per miss."""
        step_x, step_y = self._step()
        return gate_margin(step_x, step_y, self.misses)

    def aim(self):
        """Center of the next window. Never behind the box that is on screen.

        The lead is the drawn step, the motion just measured. The filter
        prediction extends that lead only on axes the step actually moved,
        and only when it is further along the step. A step of zero keeps
        the window on the box: leftover velocity from a camera shift is
        not a new direction.
        """
        predicted = self.predict()
        if self.center is None:
            if predicted is not None:
                return predicted
            return self.search_at
        step_x = self.step_x
        step_y = self.step_y
        if step_x == 0 and step_y == 0:
            return self.center
        lead = (self.center[0] + step_x, self.center[1] + step_y)
        if predicted is None:
            return lead
        pred_x = predicted[0]
        pred_y = predicted[1]
        if step_x == 0:
            pred_x = self.center[0]
        if step_y == 0:
            pred_y = self.center[1]
        along_pred = step_x * pred_x + step_y * pred_y
        along_lead = step_x * lead[0] + step_y * lead[1]
        if along_pred > along_lead:
            return (pred_x, pred_y)
        return lead

    def follow_reach(self):
        """How far a sighting may sit from the prediction and still be this object."""
        if self.appearance is None:
            return self.search_margin()
        return self.appearance.block_reach(self.misses)

    def coasting(self):
        """The lock is still drawn and the last sighting was not clear."""
        return self.show_lock and self.unclear_since is not None

    def seek_due(self, now):
        """True after two seconds of coast with no match at the predicted place."""
        if not self.coasting() or now is None or self.unclear_since is None:
            return False
        elapsed = now - self.unclear_since
        if elapsed < 0:
            return False
        return elapsed >= COAST_SEEK_MS

    def familiar(self, width, height):
        """True when this rectangle still has the locked size and aspect."""
        if self.appearance is None:
            return True
        return return_box(
            width,
            height,
            self.appearance.box_w,
            self.appearance.box_h,
            self.frame_w,
            self.frame_h,
        )

    def status(self):
        """NA, LOCKED, COASTING, or LOST.

        NA is no saved object. LOCKED is the held object while it is in view,
        including the box saved at lock before the first sighting. COASTING is
        a miss inside the twenty seconds: the box is the constant-velocity
        estimate. LOST is that wait running out. A later sighting returns
        to LOCKED.
        """
        if self.pattern is None:
            return "NA"
        if self.center is not None and self.unclear_since is not None:
            if self.show_lock:
                return "COASTING"
            return "LOST"
        return "LOCKED"

    def _slide(self):
        """Move the drawn box one frame along the recorded velocity.

        The filter position follows, so the next window is one step further
        on. Velocity is not blended: a hide is not a measurement. A step of
        zero means the last sighting did not move, so a camera shift that
        has already stopped does not carry the box off during the hide.
        """
        if self.center is None or self.appearance is None:
            return
        if self.step_x == 0 and self.step_y == 0:
            return
        cx = self.center[0] + self.appearance.vx
        cy = self.center[1] + self.appearance.vy
        if cx < 0:
            cx = 0
        if cy < 0:
            cy = 0
        if self.frame_w > 0 and cx >= self.frame_w:
            cx = self.frame_w - 1
        if self.frame_h > 0 and cy >= self.frame_h:
            cy = self.frame_h - 1
        dx = cx - self.center[0]
        dy = cy - self.center[1]
        self.center = (cx, cy)
        if self.lock_box is not None:
            box = self.lock_box
            self.lock_box = (box[0] + dx, box[1] + dy, box[2], box[3])
        self.appearance.coast()

    def note_shutter(self, now):
        if self._shutter_mark is not None and self.unclear_since is not None:
            self.unclear_since += now - self._shutter_mark
        self._shutter_mark = now
        if self.appearance is not None:
            self.appearance.hold()

    def _segment(self, gray, width, height, classical_clear):
        """Segmenter mask, or None so the caller uses Otsu.

        A raised segmenter is a missing model, not a blank scene. Otsu then
        remains the measurement.
        """
        segmenter = self.segmenter
        if segmenter is None:
            return None
        decide = getattr(segmenter, "should_run", None)
        if decide is not None and not decide(classical_clear):
            return None
        try:
            mask = segmenter(gray, width, height)
        except Exception:
            return None
        if mask is None or len(mask) != width * height:
            return None
        return mask

    def observe_crop(self, now, gray, width, height, origin_x, origin_y):
        """One search crop. The segmenter mask is the measurement when it runs."""
        every = True
        segmenter = self.segmenter
        if segmenter is not None:
            decide = getattr(segmenter, "should_run", None)
            if decide is not None:
                every = decide(True)
        classical_clear = False
        if not every:
            classical_clear = separate(gray, width, height) is not None
        mask = self._segment(gray, width, height, classical_clear)
        if mask is None:
            match = sighting(gray, width, height, origin_x, origin_y)
            foreground = None
        else:
            match = sighting(gray, width, height, origin_x, origin_y, mask)
            foreground = mask
        crop = None
        if match is not None:
            local_x = match[0] - origin_x
            local_y = match[1] - origin_y
            crop = _crop_list(gray, width, local_x, local_y, match[2], match[3])
            if foreground is not None:
                foreground = _crop_list(foreground, width, local_x, local_y, match[2], match[3])
        return self.observe(now, match, crop=crop, foreground=foreground)

    def observe(self, now, match, temp_c=None, crop=None, recovered=False, foreground=None, frame_max=None, obj_max=None, reclaim=False, reading=None):
        """Update the console line. match is (x, y, w, h, score, second) or None.

        second is None when no runner-up was measured. The third number of the
        returned line is the fused cue, not the raw correlation. The console
        prints that triple as dx= dy= confidence=. A shutter frame calls
        note_shutter and does not call observe. The next observe drops that
        gap out of the 20-second wait. crop is the matched window, used to
        learn the object and to score the frozen pattern. recovered is ignored:
        a sighting outside the block is not this object. frame_max and obj_max
        are gray counts. When the object holds the frame maximum, confidence
        rises. A hotter reading elsewhere does not lower it.

        A blob inside the gate moves the box onto that heat, including a weak
        score and a similar neighbor. That frame does not learn unless it is
        clear, and it does not start the wait. Before the first of those
        sightings the readout is the offset of the saved box and the confidence
        stays 0. A miss or a sighting outside the gate starts the wait. While
        that wait runs the box steps along the recorded velocity, unless the
        last sighting did not move. The velocity itself stays put. After twenty
        seconds the state is LOST. A sighting then takes the lock up again,
        including one outside the gate: that gate has shrunk back to the pad,
        while the search window still holds this object. reclaim is a coast
        check that already matched the stored size, silhouette, and heat.
        That sighting is this
        object even outside the gate, and it does not teach the jump as
        velocity. A coasting rectangle that has run away in size is not
        taken, including one whose center is still inside the gate.
        """
        self.frame_n += 1
        if self._shutter_mark is not None:
            if self.unclear_since is not None:
                self.unclear_since += now - self._shutter_mark
            self._shutter_mark = None
        if self.pattern is None:
            self.line = (0, 0, 0)
            self.show_lock = False
            self.cue_line = None
            self.cues = None
            return self.line
        score = None
        second = None
        width = 0
        height = 0
        cx = 0
        cy = 0
        if match is not None:
            x, y, width, height, score, second = match
            cx = x + width // 2
            cy = y + height // 2
        structure = None
        anchor_score = None
        anchor_known = False
        known_side = None
        if reading is not None:
            anchor_score = reading.anchor
            anchor_known = reading.have_anchor
            structure = reading.structure
            if width == self.pattern_w and height == self.pattern_h:
                known_side = reading.side
        elif match is not None and crop is not None and self.frozen is not None:
            anchor_score = cues.anchor(
                crop, width, height, self.frozen, self.pattern_w, self.pattern_h, self.frozen_mask
            )
            anchor_known = True
        if reading is None and match is not None:
            structure = self._structure(crop, width, height)
        if match is not None:
            gray = None
            if reading is not None and width == self.pattern_w and height == self.pattern_h:
                gray = reading.gray
            if self._prefers_distractor(cx, cy, width, height, crop, anchor_score, gray):
                match = None
                structure = None
                anchor_known = False
        fused = self._measure(
            match is not None,
            score,
            second,
            cx,
            cy,
            width,
            height,
            temp_c,
            crop,
            frame_max,
            obj_max,
            structure,
            anchor_score,
            anchor_known,
        )
        if match is not None:
            seen = 0 if second is None else second
            shaped = structure is None or structure >= CLEAR_MIN
            clear = shaped and is_clear(score, seen) and self._same_object(
                width, height, temp_c, cx, cy, recovered
            )
            # The gate is the saved object. A weak score or a similar neighbor
            # still moves the rectangle onto that heat. Only a clear sighting
            # learns. A heat outside the gate is a different object, until the
            # wait has run out, or a coast check has matched size and shape.
            # The next sighting after LOST is this object again. A coasting
            # rectangle that no longer fits the locked size is left to coast.
            brought_back = reclaim and self.coasting() and self.familiar(width, height)
            in_gate = self._in_gate(cx, cy)
            if self.coasting() and in_gate and not self.familiar(width, height):
                in_gate = False
            if self.status() == "LOST" or brought_back or in_gate:
                # Middle of the object box, measured from the middle of the frame.
                # The step is how far this box moved, including a weak sighting
                # that does not teach the filter. The next window uses it.
                # A reclaimed object did not travel the gap, so the step stays 0
                # and the recorded velocity is put back after learning.
                held = None
                if brought_back:
                    self.step_x = 0
                    self.step_y = 0
                    if self.appearance is not None:
                        held = (
                            self.appearance.vx,
                            self.appearance.vy,
                            self.appearance.motion.moves,
                        )
                elif self.center is not None:
                    self.step_x = cx - self.center[0]
                    self.step_y = cy - self.center[1]
                self.center = (cx, cy)
                self.lock_box = (x, y, width, height)
                self.base_conf = fused
                self.unclear_since = None
                self.misses = 0
                self.show_lock = True
                if clear:
                    self._learn(crop, width, height, cx, cy, temp_c, foreground, known_side)
                if brought_back and self.appearance is not None:
                    if held is not None:
                        self.appearance.motion.vx = held[0]
                        self.appearance.motion.vy = held[1]
                        self.appearance.motion.moves = held[2]
                        self.appearance.vx = held[0]
                        self.appearance.vy = held[1]
                        self.appearance.moves = held[2]
                    self.appearance.park(cx, cy)
                self.line = (self._dx(self.center[0]), self._dy(self.center[1]), fused)
                return self.line
        if self.center is None:
            self.line = self._box_offset(0)
            self.misses = 0
            return self.line
        if self.unclear_since is None:
            self.unclear_since = now
        elapsed = now - self.unclear_since
        if elapsed >= KEEP_MS:
            self.line = (0, 0, 0)
            self.show_lock = False
            self.misses = 0
            return self.line
        self._slide()
        self.misses += 1
        conf = self.base_conf * (KEEP_MS - elapsed) // KEEP_MS
        self.line = (self._dx(self.center[0]), self._dy(self.center[1]), conf)
        self.show_lock = True
        return self.line

    def _box_offset(self, confidence):
        """Pixels from the frame middle to the saved box, or zeros with no box."""
        if self.lock_box is None:
            return (0, 0, 0)
        x, y, width, height = self.lock_box
        return (self._dx(x + width // 2), self._dy(y + height // 2), confidence)

    def _dx(self, center_x):
        """Pixels from the frame middle to the box middle. Right is positive."""
        return center_x - self.frame_w // 2

    def _dy(self, center_y):
        """Pixels from the frame middle to the box middle. Down is positive."""
        return center_y - self.frame_h // 2

    def _write(self, sensor_w, sensor_h, box, pw, ph, pattern, mask):
        payload = b"%d %d\n%d %d %d %d\n%d %d\n" % (
            sensor_w,
            sensor_h,
            box[0],
            box[1],
            box[2],
            box[3],
            pw,
            ph,
        )
        raw = bytearray(pattern)
        flags = bytearray(mask)
        for path in self.paths:
            try:
                handle = open(path, "wb")
            except OSError:
                continue
            try:
                handle.write(payload)
                handle.write(raw)
                handle.write(flags)
            finally:
                handle.close()
            return True
        return False

    def _read(self):
        for path in self.paths:
            try:
                handle = open(path, "rb")
            except OSError:
                continue
            try:
                data = handle.read()
            finally:
                handle.close()
            parsed = _parse(data)
            if parsed is None:
                continue
            self.frame_w = parsed[0]
            self.frame_h = parsed[1]
            self.pattern_w = parsed[2]
            self.pattern_h = parsed[3]
            self.pattern = parsed[4]
            self.mask = parsed[5]
            box = parsed[6]
            self.search_at = (box[0] + box[2] // 2, box[1] + box[3] // 2)
            self._arm()
            return True
        return False

    def note_distractor(self, box, gray):
        """Remember a neighbor that was not chosen.

        Offset is from the lock center. A neighbor already in the register
        is refreshed. The register holds DISTRACTOR_LIMIT entries.
        """
        if self.center is None or box is None:
            return
        if len(box) < 4:
            return
        area = box[2] * box[3]
        if area < 1:
            return
        cx = box[0] + box[2] // 2
        cy = box[1] + box[3] // 2
        dx = cx - self.center[0]
        dy = cy - self.center[1]
        index = 0
        while index < len(self.distractors):
            record = self.distractors[index]
            if self._fresh(record):
                ox = dx - record.dx
                oy = dy - record.dy
                if ox * ox + oy * oy <= 64:
                    record.dx = dx
                    record.dy = dy
                    record.area = area
                    record.gray = gray
                    record.frame = self.frame_n
                    return
            index += 1
        if len(self.distractors) >= DISTRACTOR_LIMIT:
            del self.distractors[0]
        self.distractors.append(Distractor(dx, dy, area, gray, self.frame_n))

    def _fresh(self, record):
        age = self.frame_n - record.frame
        if age < 0:
            age = 0
        return age <= DISTRACTOR_TTL

    def _prefers_distractor(self, cx, cy, width, height, crop, anchor_score, gray=None):
        """True when this crop agrees with a remembered neighbor more than the anchor.

        anchor_score is the correlation already computed for this crop. The
        cue uses that same number. gray, when passed, is the crop mean from
        that same walk.
        """
        if crop is None or anchor_score is None or self.center is None or not self.distractors:
            return False
        if gray is None:
            gray = gray_mean(crop)
        area = width * height
        reach = self.search_margin()
        index = 0
        while index < len(self.distractors):
            record = self.distractors[index]
            index += 1
            if not self._fresh(record):
                continue
            score = _distractor_score(cx, cy, area, gray, record, self.center, reach)
            if score >= CLEAR_MIN and score > anchor_score:
                return True
        return False

    def _structure(self, crop, width, height):
        """Agreement of this crop with the learned 4 by 4 grid, or None.

        None means the grid is not ready, or this crop has no band to read.
        A uniform crop against a shaped grid scores 0.
        """
        appearance = self.appearance
        if crop is None or appearance is None or appearance.grid is None:
            return None
        if width < 1 or height < 1 or len(crop) != width * height:
            return None
        band = self._band()
        if band is None:
            return None
        sample = thermal_grid(crop, None, width, height, band[0], band[1])
        if sample is None:
            return None
        return cues.grid_agreement(appearance.grid, appearance.grid_valid, sample[0], sample[1])

    def _measure(
        self,
        have_match,
        score,
        second,
        cx,
        cy,
        width,
        height,
        temp_c,
        crop,
        frame_max,
        obj_max,
        structure=None,
        anchor_score=None,
        anchor_known=False,
    ):
        facts = cues.Facts()
        if have_match:
            facts.present = True
            facts.hidden = self.unclear_since is not None
            facts.score = score
            facts.second = second
            facts.cx = cx
            facts.cy = cy
            facts.width = width
            facts.height = height
            facts.temp_c = temp_c
            facts.crop = crop
            facts.crop_w = width
            facts.crop_h = height
            facts.frozen = self.frozen
            facts.frozen_w = self.pattern_w
            facts.frozen_h = self.pattern_h
            facts.frozen_mask = self.frozen_mask
            facts.frame_max = frame_max
            facts.obj_max = obj_max
            facts.structure = structure
            facts.anchor_known = anchor_known
            facts.anchor_score = anchor_score
            if self.appearance is not None:
                facts.trace = self.appearance.trace()
            if self.show_lock and self.appearance is not None:
                facts.predicted = self.appearance.predict()
                facts.reach = self.follow_reach()
            if self.appearance is not None:
                facts.last_w = self.appearance.box_w
                facts.last_h = self.appearance.box_h
                if self.appearance.samples >= LEARN_FRAMES:
                    facts.temp_mean = self.appearance.temp_mean
                    facts.temp_spread = self.appearance.temp_spread
        results = cues.evaluate(facts)
        self.cues = results
        self.cue_line = cues.format_line(results)
        fused = cues.combine(results)
        sole_value = None
        index = 0
        while index < len(results):
            if results[index].name == "sole":
                sole_value = results[index].value
            index += 1
        return cues.with_sole(fused, sole_value)

    def _arm(self):
        if self.pattern is None:
            self.appearance = None
            self.frozen = None
            self.frozen_mask = None
            self.frozen_moments = None
            return
        self.appearance = Appearance(self.pattern, self.mask, self.pattern_w, self.pattern_h)
        self.pattern = self.appearance.pattern
        self.mask = self.appearance.mask
        self.frozen = []
        self.frozen_mask = []
        index = 0
        count = self.pattern_w * self.pattern_h
        while index < count:
            self.frozen.append(self.pattern[index])
            self.frozen_mask.append(1 if self.mask[index] else 0)
            index += 1
        self.frozen_moments = _moments(self.frozen, self.pattern_w, self.pattern_h, self.frozen_mask)

    def _close(self, cx, cy):
        """True when (cx, cy) is inside the gate around the drawn box.

        The first sighting has no drawn box and uses the filter. After that
        the gate follows the rectangle on screen, so a weak run does not
        measure the next step from a center the box has already left.
        """
        if self.center is None:
            if self.appearance is None:
                return True
            return self.appearance.near(cx, cy, self.misses)
        reach = self.search_margin()
        dx = cx - self.center[0]
        dy = cy - self.center[1]
        return dx * dx + dy * dy <= reach * reach

    def _in_gate(self, cx, cy):
        """True when this center is still the locked object.

        Aspect and temperature are not part of the gate. They gate learning.
        """
        if self.appearance is None:
            return True
        return self._close(cx, cy)

    def _same_object(self, width, height, temp_c, cx, cy, recovered):
        if self.appearance is None:
            return True
        if not self._close(cx, cy):
            return False
        if self.appearance.samples < LEARN_FRAMES:
            return True
        if not self.appearance._size_ok(width, height):
            return False
        return self.appearance._temp_ok(temp_c)

    def _band(self):
        """Locked contrast against the latest block border, or None when it has collapsed."""
        appearance = self.appearance
        if appearance is None or appearance.obj_mean is None or appearance.bg_mean is None:
            return None
        border = appearance.bg_mean
        if self.border_mean is not None:
            border = self.border_mean
        return grid_band(appearance.obj_mean, appearance.bg_mean, appearance.hot, border)

    def finish(self, reading, lo, hi):
        """Celsius and the learned-grid score. Neither one walks the crop."""
        if reading is None:
            return None
        if reading.obj_n > 0 and lo is not None and hi is not None and hi > lo:
            pixel = reading.obj_sum / float(reading.obj_n)
            kelvin64 = lo + pixel * float(hi - lo) / 255.0
            reading.temp_c = kelvin64 / 64.0 - 273.15
        else:
            reading.temp_c = None
        reading.structure = None
        appearance = self.appearance
        if appearance is not None and appearance.grid is not None and reading.cells is not None:
            reading.structure = cues.grid_agreement(
                appearance.grid, appearance.grid_valid, reading.cells, reading.valid
            )
        return reading

    def read_crop(self, crop, width, height, lo=None, hi=None):
        """One walk of this crop on the pattern grid, or None when it cannot be read."""
        if crop is None or width < 1 or height < 1 or self.pattern is None:
            return None
        if len(crop) < width * height:
            return None
        pw = self.pattern_w
        ph = self.pattern_h
        if width == pw and height == ph:
            aligned = crop
        else:
            aligned = resize_nearest(crop, width, height, pw, ph)
        moments = None
        if self.appearance is not None:
            moments = self.appearance.moments
        reading = survey(
            aligned,
            pw,
            ph,
            self.pattern,
            self.mask,
            moments,
            self.frozen,
            self.frozen_mask,
            self.frozen_moments,
            self._band(),
        )
        return self.finish(reading, lo, hi)

    def _track_contrast(self, crop, side=None):
        """Fold this crop into the saved means. Polarity stays on the locked side.

        side is the band mean when the caller already walked the crop.
        """
        band = self._band()
        if band is None or self.appearance is None:
            return
        bg = self.border_mean
        if bg is None:
            bg = self.appearance.bg_mean
        if side is None:
            side = side_mean(crop, band[0], band[1])
        self.appearance.note_contrast(side, bg)

    def _learned_mask(self, crop, box_w, box_h, foreground):
        """Object mask on the pattern grid. One band pass while the mask is still open."""
        band = self._band()
        if band is None:
            return mask_on_grid(
                crop, box_w, box_h, self.appearance.mask, self.pattern_w, self.pattern_h, foreground
            )
        flags = band_mask(crop, band[0], band[1])
        if box_w == self.pattern_w and box_h == self.pattern_h:
            return flags
        return resize_nearest(flags, box_w, box_h, self.pattern_w, self.pattern_h)

    def _learn(self, crop, box_w, box_h, cx, cy, temp_c, foreground=None, side=None):
        if self.appearance is None:
            return
        if crop is None:
            self.appearance.see(cx, cy)
            self.appearance._note_size(box_w, box_h)
            return
        if self.distant:
            self.clear_seen += 1
            if self.clear_seen % CALIBRATE_EVERY == 0 and self._calibrate(
                crop, box_w, box_h, cx, cy, temp_c
            ):
                return
        if self.appearance.samples < LEARN_FRAMES:
            grid = self._learned_mask(crop, box_w, box_h, foreground)
        else:
            grid = self.appearance.mask
        self._track_contrast(crop, side)
        if box_w == self.pattern_w and box_h == self.pattern_h:
            gray = crop
        else:
            gray = resize_nearest(crop, box_w, box_h, self.pattern_w, self.pattern_h)
        self.appearance.record_grid(gray, grid, box_w, box_h, cx, cy, temp_c)
        self.pattern = self.appearance.pattern
        self.mask = self.appearance.mask

    def _calibrate(self, crop, box_w, box_h, cx, cy, temp_c):
        """Refresh the live size and shape. The anchor and the file stay.

        A side that has moved by two pixels or more becomes the new pattern
        grid. A smaller change refreshes the mask and keeps the blend.
        """
        if len(crop) != box_w * box_h:
            return False
        flags = self._band_flags(crop)
        if flags is None:
            return False
        dw = box_w - self.pattern_w
        if dw < 0:
            dw = -dw
        dh = box_h - self.pattern_h
        if dh < 0:
            dh = -dh
        if dw >= 2 or dh >= 2:
            self.appearance.adopt(crop, flags, box_w, box_h)
            self.appearance.see(cx, cy)
            if self.appearance.samples < LEARN_FRAMES:
                self.appearance._accumulate(crop, self.appearance.mask)
                self.appearance.samples += 1
            self._track_contrast(crop)
            self.pattern = self.appearance.pattern
            self.mask = self.appearance.mask
            self.pattern_w = self.appearance.width
            self.pattern_h = self.appearance.height
            return True
        grid = resize_nearest(flags, box_w, box_h, self.pattern_w, self.pattern_h)
        if not _any_on(grid):
            return False
        self.appearance.mask = grid
        self.appearance._bind()
        self._track_contrast(crop)
        if box_w == self.pattern_w and box_h == self.pattern_h:
            gray = crop
        else:
            gray = resize_nearest(crop, box_w, box_h, self.pattern_w, self.pattern_h)
        self.appearance.record_grid(gray, grid, box_w, box_h, cx, cy, temp_c)
        self.pattern = self.appearance.pattern
        self.mask = self.appearance.mask
        return True

    def _band_flags(self, crop):
        """Locked-band mask of this crop, or None when the band is empty."""
        band = self._band()
        if band is None:
            return None
        flags = band_mask(crop, band[0], band[1])
        if not _any_on(flags):
            return None
        return flags

    def _delete(self):
        import os

        for path in self.paths:
            try:
                os.remove(path)
            except OSError:
                pass


def _parse(data):
    lines = []
    start = 0
    index = 0
    limit = len(data)
    while len(lines) < 3 and index < limit:
        if data[index] == 10:
            lines.append(data[start:index])
            start = index + 1
        index += 1
    if len(lines) < 3:
        return None
    try:
        sensor_w, sensor_h = _ints(lines[0], 2)
        box = _ints(lines[1], 4)
        pattern_w, pattern_h = _ints(lines[2], 2)
    except ValueError:
        return None
    raw = data[start:]
    need = pattern_w * pattern_h
    if pattern_w < 1 or pattern_h < 1 or len(raw) != need * 2:
        return None
    pattern = []
    mask = []
    index = 0
    while index < need:
        pattern.append(raw[index])
        index += 1
    while index < need * 2:
        mask.append(1 if raw[index] else 0)
        index += 1
    return (sensor_w, sensor_h, pattern_w, pattern_h, pattern, mask, box)


def _ints(line, count):
    parts = line.split()
    if len(parts) != count:
        raise ValueError("short header")
    values = []
    for part in parts:
        values.append(int(part))
    return values
