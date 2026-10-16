# Confidence that the locked object is the one in the box.
#
# Each name in CUES is one algorithm. Drop a name from that tuple to stop
# using it. evaluate() times every cue that runs. format_line() prints them
# in that order, each score and its cost as name=value:
#
#   appearance=84 appearance_us=40 unique=90 unique_us=12 motion=na motion_us=8 size=100 size_us=5 temperature=100 temperature_us=6 anchor=88 anchor_us=30 hottest=na hottest_us=4 sole=na sole_us=4
#
# The score is 0..100, or na when the cue abstains. name_us is microseconds.
# combine() is the weighted average of the cues that voted. The numbers
# are in weights.txt on the card. Thermal noise makes the silhouette
# unreliable, so appearance and the frozen anchor weigh the least.
# Proximity to the prediction weighs the most.
# A trace of recent centers pulls that proximity onto the line the object
# has been walking. The temperature cue also compares a 4 by 4 relative
# grid when one has been learned, and keeps the lower score.
# A hotter distractor does not vote: hottest abstains unless this object holds
# the frame maximum, and then it only raises the score.
# sole votes only when the single warm object never left. That frame's
# confidence is then at least SOLE_SURE.

CUES = ("appearance", "unique", "motion", "size", "temperature", "anchor", "hottest", "sole")
SOLE_SURE = 80


class Cue(object):
    def __init__(self, name, value, us):
        self.name = name
        self.value = value
        self.us = us


class Facts(object):
    """One sighting. Unused fields stay empty and the cue abstains."""

    def __init__(self):
        self.score = None
        self.second = None
        self.cx = 0
        self.cy = 0
        self.predicted = None
        self.reach = 16
        self.width = 0
        self.height = 0
        self.last_w = 0
        self.last_h = 0
        self.temp_c = None
        self.temp_mean = None
        self.temp_spread = None
        self.crop = None
        self.crop_w = 0
        self.crop_h = 0
        self.frozen = None
        self.frozen_w = 0
        self.frozen_h = 0
        self.frozen_mask = None
        self.obj_max = None
        self.frame_max = None
        self.present = False
        self.hidden = False
        self.trace = None
        self.structure = None
        self.anchor_known = False
        self.anchor_score = None


def appearance(score):
    """Correlation of the chosen box. The search that found the box is not repeated."""
    if score is None:
        return None
    value = int(score)
    if value < 0:
        return 0
    if value > 100:
        return 100
    return value


def unique(score, second):
    """How far the runner-up sits behind the leader. Abstains when it was not measured."""
    if score is None or second is None or score <= 0:
        return None
    gap = 100 - (int(second) * 100) // int(score)
    if gap < 0:
        return 0
    if gap > 100:
        return 100
    return gap


def motion(cx, cy, predicted, reach, trace=None):
    """Agreement with the constant-velocity prediction, 100 on it and 0 at the reach.

    With three or more recent centers, a candidate off that line scores lower
    than the same distance along it. A short trace leaves the radial score.
    """
    radial = _radial(cx, cy, predicted, reach)
    if radial is None or trace is None or len(trace) < 3:
        return radial
    crossed = _cross(cx, cy, predicted, reach, trace)
    if crossed is None:
        return radial
    return radial * crossed // 100


def _radial(cx, cy, predicted, reach):
    if predicted is None:
        return None
    if reach < 1:
        reach = 1
    dx = cx - predicted[0]
    dy = cy - predicted[1]
    dist2 = dx * dx + dy * dy
    if dist2 >= reach * reach:
        return 0
    dist = dist2 ** 0.5
    score = int(100 - (100.0 * dist) / reach)
    if score < 0:
        return 0
    if score > 100:
        return 100
    return score


def _cross(cx, cy, predicted, reach, trace):
    """100 on the recent line, 0 once the perpendicular error leaves its limit.

    A stopped trace has no line and returns None so the radial score stands.
    """
    count = len(trace) - 1
    if count < 1:
        return None
    sx = 0
    sy = 0
    index = 1
    while index < len(trace):
        sx += trace[index][0] - trace[index - 1][0]
        sy += trace[index][1] - trace[index - 1][1]
        index += 1
    sx = sx / float(count)
    sy = sy / float(count)
    length2 = sx * sx + sy * sy
    if length2 < 1.0:
        return None
    length = length2 ** 0.5
    ux = sx / length
    uy = sy / length
    cross_sum = 0.0
    index = 1
    while index < len(trace):
        dx = trace[index][0] - trace[index - 1][0]
        dy = trace[index][1] - trace[index - 1][1]
        cross = dx * uy - dy * ux
        if cross < 0:
            cross = -cross
        cross_sum += cross
        index += 1
    limit = cross_sum / float(count) * 3.0
    if limit < 4.0:
        limit = 4.0
    if limit > reach:
        limit = float(reach)
    ox = cx - predicted[0]
    oy = cy - predicted[1]
    cross = ox * uy - oy * ux
    if cross < 0:
        cross = -cross
    if cross >= limit:
        return 0
    score = int(100 - (100.0 * cross) / limit)
    if score < 0:
        return 0
    if score > 100:
        return 100
    return score


# The larger axis-scale may sit this far above the smaller one and still be
# the same object. That is the 15 percent step the search already takes,
# measured between the two sides rather than against the size at lock.
ASPECT_PERCENT = 15


def size(width, height, last_w, last_h):
    """100 when this box is a uniform scale of the latest one.

    Approaching the object grows both sides together and stays at 100, even
    when the box is much larger than it was at lock. The score falls as the
    two sides disagree. Abstains until a previous box exists.
    """
    compared = _aspect(width, height, last_w, last_h)
    if compared is None:
        return None
    diff = compared[0]
    span = compared[1]
    score = 100 - (100 * diff) // span
    if score < 0:
        return 0
    return score


def same_aspect(width, height, last_w, last_h):
    """True when both sides scaled by about the same factor."""
    compared = _aspect(width, height, last_w, last_h)
    if compared is None:
        return False
    left, right = compared[2], compared[3]
    if left > right:
        return left * 100 <= right * (100 + ASPECT_PERCENT)
    return right * 100 <= left * (100 + ASPECT_PERCENT)


def temperature(temp_c, mean, spread, structure=None):
    """Object Celsius against the learned band. Abstains until the band exists.

    structure is the agreement of the 4 by 4 relative grid, 0..100. The cue
    keeps the lower of the two, so a matching mean with a different shape
    does not score as this object.
    """
    if temp_c is None or mean is None:
        return None
    limit = 2.0
    if spread is not None and spread * 3 > limit:
        limit = spread * 3
    if temp_c < mean:
        delta = mean - temp_c
    else:
        delta = temp_c - mean
    if delta >= limit:
        score = 0
    else:
        score = int(100 - (100.0 * delta) / limit)
        if score < 0:
            score = 0
    if structure is None or structure >= score:
        return score
    if structure < 0:
        return 0
    return structure


def grid_agreement(learned, learned_valid, sample, sample_valid):
    """Cosine of two 4 by 4 relative grids on the cells both filled, 0..100.

    Fewer than four shared cells abstains. Two flat grids abstain. One flat
    grid against a shaped one scores 0.
    """
    if learned is None or sample is None or learned_valid is None or sample_valid is None:
        return None
    num = 0.0
    den_l = 0.0
    den_s = 0.0
    count = 0
    limit = len(learned)
    if len(sample) < limit:
        limit = len(sample)
    index = 0
    while index < limit:
        if learned_valid[index] and sample_valid[index]:
            lv = learned[index]
            sv = sample[index]
            num += lv * sv
            den_l += lv * lv
            den_s += sv * sv
            count += 1
        index += 1
    if count < 4:
        return None
    if den_l <= 0 and den_s <= 0:
        return None
    if den_l <= 0 or den_s <= 0:
        return 0
    ratio = num / ((den_l ** 0.5) * (den_s ** 0.5))
    if ratio <= 0:
        return 0
    score = int(ratio * 100 + 0.5)
    if score > 100:
        return 100
    return score


def sole(present, second, hidden):
    """100 when the only warm object is still in view.

    Abstains when the sighting is missing, the object was hidden, or another
    warm object was measured. Those frames keep the ordinary average.
    """
    if not present or hidden or second is not None:
        return None
    return 100


def hottest(obj_max, frame_max):
    """100 when the object holds the hottest reading in the frame.

    Otherwise abstain. A cooler target is still the target, so a hotter
    reading somewhere else does not lower the score.
    """
    if obj_max is None or frame_max is None:
        return None
    if obj_max >= frame_max:
        return 100
    return None


def anchor(crop, crop_w, crop_h, frozen, frozen_w, frozen_h, mask):
    """Correlation of this crop with the pattern saved at lock."""
    if crop is None or frozen is None or crop_w < 1 or crop_h < 1 or frozen_w < 1 or frozen_h < 1:
        return None
    if crop_w != frozen_w or crop_h != frozen_h:
        aligned = _resize(crop, crop_w, crop_h, frozen_w, frozen_h)
    else:
        aligned = crop
    return _match(aligned, frozen, mask)


def evaluate(facts, clock=None, active=None):
    """Run the active cues. `clock`, if passed, is called before and after each one."""
    if active is None:
        active = CUES
    results = []
    index = 0
    while index < len(active):
        results.append(_timed(active[index], facts, clock))
        index += 1
    return results


# Defaults used when weights.txt is missing. The file on the card replaces
# any name it sets. 0 keeps the cue in the log and out of the average.
_WEIGHT_NAMES = (
    "appearance",
    "unique",
    "motion",
    "size",
    "temperature",
    "anchor",
    "hottest",
    "sole",
)
_WEIGHT_DEFAULTS = (1, 3, 8, 4, 3, 1, 5, 5)
_WEIGHT_VALUES = [1, 3, 8, 4, 3, 1, 5, 5]


def _apply_weight_text(text):
    """Overlay names from one weights file. Unknown lines are skipped."""
    if text is None:
        return
    lines = text.split("\n")
    row = 0
    while row < len(lines):
        line = lines[row]
        row += 1
        mark = line.find("#")
        if mark >= 0:
            line = line[:mark]
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = int(parts[1])
        except ValueError:
            continue
        if value < 0:
            continue
        index = 0
        while index < len(_WEIGHT_NAMES):
            if _WEIGHT_NAMES[index] == parts[0]:
                _WEIGHT_VALUES[index] = value
                break
            index += 1


def load_weights(paths=None):
    """Read cue weights. A missing file leaves the defaults in place.

    The tracker looks on the card. Pass a path to read that file.
    Pass an empty list to put the defaults back.
    """
    index = 0
    while index < len(_WEIGHT_NAMES):
        _WEIGHT_VALUES[index] = _WEIGHT_DEFAULTS[index]
        index += 1
    if paths is None:
        paths = ("/sdcard/weights.txt", "/sd/weights.txt")
    slot = 0
    while slot < len(paths):
        path = paths[slot]
        slot += 1
        try:
            handle = open(path)
        except OSError:
            continue
        try:
            text = handle.read()
        finally:
            handle.close()
        _apply_weight_text(text)
        return


def weight(name):
    """How much one cue moves the fused score. Unknown names weigh 1."""
    index = 0
    while index < len(_WEIGHT_NAMES):
        if _WEIGHT_NAMES[index] == name:
            return _WEIGHT_VALUES[index]
        index += 1
    return 1


load_weights()


def combine(results):
    """Weighted average of the cues that voted, in 0..100."""
    total = 0
    mass = 0
    index = 0
    while index < len(results):
        value = results[index].value
        if value is not None:
            share = weight(results[index].name)
            total += value * share
            mass += share
        index += 1
    if mass < 1:
        return 0
    return total // mass


def with_sole(fused, sole_value):
    """The only warm object still in view is at least SOLE_SURE.

    A higher average is kept. A missing sole vote means the object was gone,
    hidden, or not alone, and the average is unchanged.
    """
    if sole_value is None:
        return fused
    if fused > SOLE_SURE:
        return fused
    return SOLE_SURE


def format_line(results):
    parts = []
    index = 0
    while index < len(results):
        cue = results[index]
        if cue.value is None:
            shown = "na"
        else:
            shown = "%d" % cue.value
        parts.append("%s=%s" % (cue.name, shown))
        parts.append("%s_us=%d" % (cue.name, cue.us))
        index += 1
    return " ".join(parts)


def _value(name, facts):
    if name == "appearance":
        return appearance(facts.score)
    if name == "unique":
        return unique(facts.score, facts.second)
    if name == "motion":
        return motion(facts.cx, facts.cy, facts.predicted, facts.reach, facts.trace)
    if name == "size":
        return size(facts.width, facts.height, facts.last_w, facts.last_h)
    if name == "temperature":
        return temperature(facts.temp_c, facts.temp_mean, facts.temp_spread, facts.structure)
    if name == "hottest":
        return hottest(facts.obj_max, facts.frame_max)
    if name == "sole":
        return sole(facts.present, facts.second, facts.hidden)
    if name == "anchor":
        if facts.anchor_known:
            return facts.anchor_score
        return anchor(
            facts.crop,
            facts.crop_w,
            facts.crop_h,
            facts.frozen,
            facts.frozen_w,
            facts.frozen_h,
            facts.frozen_mask,
        )
    return None


def _timed(name, facts, clock):
    if clock is None:
        import time

        ticks = getattr(time, "ticks_us", None)
        diff = getattr(time, "ticks_diff", None)
        if ticks is None or diff is None:
            return Cue(name, _value(name, facts), 0)
        start = ticks()
        value = _value(name, facts)
        elapsed = diff(ticks(), start)
    else:
        start = clock()
        value = _value(name, facts)
        elapsed = clock() - start
    if elapsed < 0:
        elapsed = 0
    return Cue(name, value, elapsed)


def _aspect(width, height, last_w, last_h):
    """Cross products of this box and the latest one, or None when a side is empty.

    The pair is (difference, larger, left, right). Equal sides mean a uniform
    scale: the camera moved closer or farther, and the aspect did not change.
    """
    if width < 1 or height < 1 or last_w < 1 or last_h < 1:
        return None
    left = width * last_h
    right = height * last_w
    if left < right:
        return (right - left, right, left, right)
    return (left - right, left, left, right)


def _resize(src, src_w, src_h, dst_w, dst_h):
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


def _match(sample, pattern, mask):
    """One integer walk. The ratio is the same one logic._zncc uses."""
    try:
        import logic
    except ImportError:
        from src import logic

    count = 0
    sum_s = 0
    sum_p = 0
    sum_ss = 0
    sum_pp = 0
    sum_sp = 0
    index = 0
    limit = len(pattern)
    while index < limit:
        if mask is None or mask[index]:
            left = sample[index]
            right = pattern[index]
            sum_s += left
            sum_p += right
            sum_ss += left * left
            sum_pp += right * right
            sum_sp += left * right
            count += 1
        index += 1
    return logic._ratio(count, sum_s, sum_p, sum_ss, sum_pp, sum_sp)
