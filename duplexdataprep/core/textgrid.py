"""Praat TextGrid writing and reading.

The corpus layout is the standard MFA "file + TextGrid" form:

    corpus/<dataset>/shard000/<file_id>.flac      (symlink or extracted audio)
    corpus/<dataset>/shard000/<file_id>.TextGrid  (one IntervalTier per speaker)

MFA takes the tier *name* as the speaker id and each non-empty interval as one
utterance, so long recordings never need to be chopped up on disk.
"""

from __future__ import annotations

import re
from pathlib import Path

# Labels MFA emits for non-speech; never folded back into a supervision.
SILENCE = {"", "<eps>", "sil", "sp", "spn_sil"}


def _tidy(intervals, xmax, min_dur=0.03):
    """Sort, de-overlap and clip intervals so Praat/praatio will accept them."""
    out, prev_end = [], 0.0
    for s, e, txt in sorted(intervals, key=lambda t: (t[0], t[1])):
        s, e = max(float(s), prev_end), min(float(e), xmax)
        if e - s < min_dur:
            continue
        out.append((s, e, txt))
        prev_end = e
    return out


def _fill(intervals, xmax):
    """Insert empty intervals so the tier tiles [0, xmax] with no holes."""
    filled, t = [], 0.0
    for s, e, txt in intervals:
        if s - t > 1e-6:
            filled.append((t, s, ""))
        filled.append((s, e, txt))
        t = e
    if xmax - t > 1e-6:
        filled.append((t, xmax, ""))
    return filled or [(0.0, xmax, "")]


# praatio splits long-form TextGrids into tiers with re.split(r"item ?\[")
# (utilities/textgrid_io.py), so a text field containing "item [" -- e.g. Fisher's
# "an item [lipsmack]" -- fabricates a tier and the file fails to parse. MFA then
# drops it at corpus load with no warning. Angle brackets are in the `brackets`
# config too, so MFA still maps the token to `spn`; only the literal changes.
_PRAATIO_TIER_SPLIT = re.compile(r"item ?\[")


def _praatio_safe(text: str) -> str:
    if not _PRAATIO_TIER_SPLIT.search(text):
        return text
    return re.sub(r"\[([^\[\]]*)\]", r"<\1>", text)


def _esc(text: str) -> str:
    return _praatio_safe(text).replace('"', '""')


def write_textgrid(path: Path, xmax: float, tiers: dict[str, list]) -> int:
    """Write a long-form Praat TextGrid. Returns the number of spoken intervals.

    `tiers` maps speaker id -> [(start, end, text), ...] in seconds.
    """
    prepared, n_spoken = {}, 0
    for name, ivals in tiers.items():
        ivals = _fill(_tidy(ivals, xmax), xmax)
        n_spoken += sum(1 for _, _, t in ivals if t)
        prepared[name] = ivals

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {xmax:.6f}",
        "tiers? <exists>",
        f"size = {len(prepared)}",
        "item []:",
    ]
    for i, (name, ivals) in enumerate(prepared.items(), start=1):
        lines += [
            f"    item [{i}]:",
            '        class = "IntervalTier"',
            f'        name = "{_esc(name)}"',
            "        xmin = 0",
            f"        xmax = {xmax:.6f}",
            f"        intervals: size = {len(ivals)}",
        ]
        for j, (s, e, txt) in enumerate(ivals, start=1):
            lines += [
                f"        intervals [{j}]:",
                f"            xmin = {s:.6f}",
                f"            xmax = {e:.6f}",
                f'            text = "{_esc(txt)}"',
            ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n_spoken


def read_tiers(path):
    """Return {'words': [(label, start, end)], 'phones': [...]} from an aligned TextGrid.

    Imports praatio lazily: building a corpus does not need it, only collecting
    alignments back does.
    """
    from praatio import textgrid as praat_tg

    tg = praat_tg.openTextgrid(str(path), includeEmptyIntervals=False)
    out = {"words": [], "phones": []}
    for name in tg.tierNames:
        # MFA names tiers "words"/"phones" when a file has one speaker, and
        # "<speaker> - words"/"<speaker> - phones" when it has several.
        lower = name.lower()
        if lower in ("words", "phones"):
            kind = lower
        elif lower.endswith(" - words"):
            kind = "words"
        elif lower.endswith(" - phones"):
            kind = "phones"
        else:
            continue
        for entry in tg.getTier(name).entries:
            label = entry.label.strip()
            if label in SILENCE:
                continue
            out[kind].append((label, float(entry.start), float(entry.end)))
    for k in out:
        out[k].sort(key=lambda t: t[1])
    return out
