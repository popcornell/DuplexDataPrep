"""MFA supervisions + recordings -> Lhotse CutSets, in FastMSS's convention.

Ported from `MFA/scripts/build_cutsets.py`. Writes

    <prefix>_cuts_mfa.jsonl.gz       one MultiCut per recording, every supervision
    <prefix>_cuts_mfa_utt.jsonl.gz   one single-speaker MonoCut per utterance

**FastMSS convention for the per-utterance cuts.**
`fastmss/utils.py::split_monocut_at_pauses` rebases word times itself
(`item.start - monocut.start`), so it wants *absolute* alignment times next to a
cut-relative `supervision.start` of 0. `CutSet.trim_to_supervisions()` makes the
alignment cut-relative instead, which FastMSS would then offset a second time;
hence the cuts are built by hand.

`resample` points the cuts at a lazily resampled view of the recording
(`Recording.resample`), so a 24 kHz Sidon output feeds a 16 kHz simulation
without a second copy of the audio on disk.

Supervisions are grouped by recording as they stream, so peak memory is one
conversation. Needs lhotse (the `fastmss` env on Delta).
"""

from __future__ import annotations

import json
from pathlib import Path


def _grouped(paths):
    """(recording_id, [SupervisionSegment]) across one or more manifests, in
    order. A recording's supervisions must be contiguous (collect preserves
    the source manifest's order, which is grouped); a repeat is an error."""
    from lhotse import SupervisionSet

    seen, group, current = set(), [], None
    for p in paths:
        for s in SupervisionSet.from_jsonl_lazy(str(p)):
            if s.recording_id != current:
                if group:
                    yield current, group
                if s.recording_id in seen:
                    raise ValueError(f"supervisions of {s.recording_id} are not contiguous")
                seen.add(s.recording_id)
                current, group = s.recording_id, []
            group.append(s)
    if group:
        yield current, group


def build_cuts(recordings, supervisions, out_dir, prefix, resample=None,
               blocked=(), report=None, log=print) -> dict:
    """`blocked` is a set of (recording_id, channel) whose utterance cuts are
    dropped (they stay in the full-recording MultiCuts)."""
    from lhotse import MonoCut, MultiCut, RecordingSet
    from lhotse.serialization import SequentialJsonlWriter
    from lhotse.utils import fastcopy

    recs = {r.id: r for r in RecordingSet.from_file(str(recordings))}
    if resample:
        recs = {k: r.resample(int(resample)) for k, r in recs.items()}
    blocked = set(blocked)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cuts_path = out / f"{prefix}_cuts_mfa.jsonl.gz"
    utt_path = out / f"{prefix}_cuts_mfa_utt.jsonl.gz"

    n = {"recordings": 0, "supervisions": 0, "utterances": 0, "no_recording": 0,
         "no_word_alignment": 0, "blocked_channel": 0}
    hours = 0.0
    speakers = set()
    with SequentialJsonlWriter(cuts_path) as cw, SequentialJsonlWriter(utt_path) as uw:
        for rid, segs in _grouped(supervisions):
            rec = recs.get(rid)
            if rec is None:
                n["no_recording"] += 1
                continue
            segs.sort(key=lambda s: (s.start, s.channel))
            cw.write(MultiCut(id=rid, start=0.0, duration=rec.duration,
                              channel=list(rec.channel_ids), recording=rec,
                              supervisions=segs))
            n["recordings"] += 1
            n["supervisions"] += len(segs)
            for s in segs:
                if (rid, s.channel) in blocked:
                    n["blocked_channel"] += 1
                    continue
                if not (s.alignment or {}).get("word"):
                    n["no_word_alignment"] += 1
                    continue
                uw.write(MonoCut(id=s.id, start=s.start, duration=s.duration,
                                 channel=s.channel, recording=rec,
                                 supervisions=[fastcopy(s, start=0.0)]))
                n["utterances"] += 1
                hours += s.duration / 3600
                speakers.add(s.speaker)

    stats = {**n, "utterance_hours": round(hours, 3), "speakers": len(speakers),
             "resample": resample, "outputs": [str(cuts_path), str(utt_path)]}
    log(f"[cuts] {n['recordings']} recordings -> {n['utterances']} utterance cuts "
        f"({stats['utterance_hours']} h, {len(speakers)} speakers)", flush=True)
    if report:
        Path(report).write_text(json.dumps(stats, indent=2))
    return stats
