#!/usr/bin/env python3
"""Synthesise a miniature NOTSOFAR-1 release tree for the smoke test.

A fixture keeps the contract test fast and hermetic, and lets it assert on
cases a real subset only happens to contain. It is built to the layout the
challenge repo documents (`<root>/<subset>/<version>/MTG/<session>/...`) and to
the fields the real `gt_transcription.json` carries, including `word_timing`.
Audio is silence -- prep only reads durations from the wav headers, never
samples.

Aliases are deliberately reused across sessions, as the real release does, so
the speaker-scoping rule is exercised rather than assumed.

    python tools/make_notsofar_fixture.py <root> [--sessions 2] [--speakers 3]
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

RATE = 16000
VERSION = "240208.2_fixture"


def write_silence(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(RATE)
        fh.writeframes(b"\x00\x00" * int(seconds * RATE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--split", default="train_set")
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--speakers", type=int, default=3)
    ap.add_argument("--duration", type=float, default=60.0)
    a = ap.parse_args()

    mtg = Path(a.root) / a.split / VERSION / "MTG"
    for s in range(a.sessions):
        sess = mtg / f"MTG_{s:04d}"
        aliases = {}
        segments = []
        for p in range(a.speakers):
            spk = f"spk{p}"
            ct_name = f"{spk}_ct.wav"
            aliases[spk] = ct_name
            write_silence(sess / "close_talk" / ct_name, a.duration)

            # four ordinary turns (covering every marker shape the release
            # uses), two marker-only turns (must be dropped as non-lexical),
            # one over-long turn (must be counted, not dropped)
            base = 1.0 + p * 0.5
            for i, (start, dur, text) in enumerate([
                (base, 2.0, "hello everyone <ST/> thanks for joining"),
                (base + 5.0, 1.5, "yeah <FILL/> that works for me"),
                (base + 10.0, 3.0, "let's take that offline [laughter]"),
                (base + 13.0, 1.2, "thanks <PName> Bert </PName> <ISSUE/>"),
                (base + 15.0, 0.8, "<UNKNOWN/>"),
                (base + 17.0, 0.6, "<BA/>"),
                (base + 20.0, 25.0, "and then " + "the plan " * 40),
            ]):
                seg = {
                    "speaker_id": spk,
                    "start_time": start,
                    "end_time": start + dur,
                    "text": text,
                    "ct_wav_file_name": f"close_talk/{ct_name}",
                }
                # Real segments carry per-word times, in absolute recording
                # seconds as [symbol, start, end], with marker tokens present.
                words = text.split()
                if words:
                    step = dur / len(words)
                    seg["word_timing"] = [
                        [w.lower(), round(start + j * step, 2),
                         round(start + (j + 1) * step, 2)]
                        for j, w in enumerate(words)
                        if not (w.startswith("<") and w.endswith(">"))
                    ]
                segments.append(seg)

        # a segment whose close-talk device was not released with the session
        segments.append({
            "speaker_id": "spk_absent",
            "start_time": 40.0,
            "end_time": 41.0,
            "text": "this speaker has no headset file",
            "ct_wav_file_name": "close_talk/missing_ct.wav",
        })

        # far-field devices exist in the release and must be ignored by prep
        write_silence(sess / "mc_device0" / "ch0.wav", a.duration)

        (sess / "gt_transcription.json").write_text(json.dumps(segments, indent=1))
        (sess / "gt_meeting_metadata.json").write_text(json.dumps(
            {"ParticipantAliasToCtDevice": aliases}, indent=1))

    print(f"fixture: {a.sessions} sessions x {a.speakers} speakers under {mtg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
