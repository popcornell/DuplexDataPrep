"""Turn-level segment handling shared by every corpus."""

from __future__ import annotations


def is_alignable(text: str) -> bool:
    """True if the text has at least one token that could carry a phone."""
    return any(ch.isalpha() or ch.isdigit() for ch in text)


def resegment_on_words(start, duration, words, max_dur=20.0, min_gap=0.30):
    """Split an over-long supervision at its widest inter-word pauses.

    `words` is Lhotse's alignment format: [[symbol, start, duration, score], ...]
    in absolute recording time. Returns [(start, end, text), ...]. Falls back to
    the original span when there is nothing to split on.

    Needed by any corpus whose turn table aggregates runs of backchannels into
    single turns (CANDOR's worst is ~52 minutes), which forced alignment cannot
    handle. Existing word times are used purely to pick cut points -- the
    alignment the aligner then produces is entirely its own.
    """
    end = start + duration
    if duration <= max_dur or not words:
        return [(start, end, None)]

    items = sorted(
        ((float(w[1]), float(w[1]) + float(w[2]), str(w[0])) for w in words),
        key=lambda t: t[0],
    )

    groups, cur = [], [items[0]]
    for prev, item in zip(items, items[1:]):
        gap = item[0] - prev[1]
        span = item[1] - cur[0][0]
        if gap >= min_gap and span > max_dur:
            groups.append(cur)
            cur = [item]
        else:
            cur.append(item)
    groups.append(cur)

    # A group can still exceed max_dur if the speaker never paused; split it on
    # the largest remaining gaps rather than dropping it.
    out = []
    for g in groups:
        while (g[-1][1] - g[0][0]) > max_dur and len(g) > 1:
            gaps = [(g[i + 1][0] - g[i][1], i) for i in range(len(g) - 1)]
            # cut at the widest gap in the first max_dur seconds of the group
            head = [(gp, i) for gp, i in gaps if g[i][1] - g[0][0] <= max_dur]
            _, cut = max(head or gaps)
            out.append(g[: cut + 1])
            g = g[cut + 1 :]
        out.append(g)

    segs = []
    for g in out:
        if not g:
            continue
        s, e = max(g[0][0], start), min(g[-1][1], end)
        if e - s > 0.02:
            segs.append((s, e, " ".join(w[2] for w in g)))
    return segs or [(start, end, None)]
