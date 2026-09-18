"""nsf_chorus: NOTSOFAR-1 close-talk, Sidon-restored to 24 kHz.

The same meetings as `corpora.notsofar`, but a different artifact and a
different job:

| | `notsofar` | `nsf_chorus` |
|---|---|---|
| audio | raw close-talk, 16 kHz, dev only | Sidon-restored, 24 kHz, train+dev+eval |
| word times | from the release, per channel | from the release (dev) + CHiME-8 DASR (train/eval) |
| needs MFA | yes -- that is what `prep` builds for | no, the manifests ship aligned |

So this corpus exists for its **manifest stage**, not for `prep`. It is the
analogue of the Sidon-restored Fisher the simulated corpora are built from, and
because every split already carries word times, it never needs an aligner run.

    msprep manifest nsf_chorus            # all three splits + the dev+train merge
    msprep manifest nsf_chorus --split dev

`iter_channels` is still implemented, so the corpus can be inspected and, if it
is ever wanted, aligned like any other -- one speaker's restored wav is exactly
one speaker channel.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from ...core.manifests import channels_of, load_jsonl
from ...core.segments import is_alignable
from ..base import Channel, ChannelSource, register

#: NOTSOFAR ground truth carries annotator markup inline, and the two fields
#: spell it differently: `text` has `<ST/> <FILL/> <FILLlaugh/> <UNKNOWN/>
#: <BA/> <ISSUE/> <SN/> <PAUSE/>` and the `<PName>…</PName>` pair around a
#: person's name, while `word_timing` carries the lower-cased self-closing forms
#: only (`<fill/> <filllaugh/> <unknown/>`). So strip *any* tag rather than a
#: fixed list: every self-closing marker disappears, and `<PName>` and its close
#: disappear while the name between them stays a word. Nothing else in this
#: corpus uses angle brackets -- of 3,558 distinct word symbols in dev, exactly
#: those three are markup.
TAG = re.compile(r"<[^>]*>")


def clean_text(text: str) -> str:
    return " ".join(TAG.sub(" ", text).split())


@register
class NsfChorus(ChannelSource):
    name = "nsf_chorus"

    def configure(self) -> None:
        cfg = self.config
        self.root = Path(cfg["root"])
        self.split = cfg.get("split", "all")
        m = cfg["manifests"]
        d = Path(m["dir"])
        self.recordings_path = d / m["recordings"]
        # `all` is the dev+train merge; a named split is its own file.
        self.supervisions_path = (
            d / (m["merged"] if self.split in ("all", "merged")
                 else m["supervisions"].format(split=self.split)))

    # ---------------------------------------------------------------- text

    def normalize(self, text: str) -> str:
        """Drop every angle-bracket tag, keep what sits between a paired one."""
        return clean_text(text)

    # ------------------------------------------------------- manifest stage

    def build_manifests(self, **kwargs) -> dict:
        from .manifest import build_manifests
        return build_manifests(self, **kwargs)

    # ------------------------------------------------------------ channels

    def iter_channels(self, limit: int = 0):
        if not self.recordings_path.exists():
            raise FileNotFoundError(
                f"{self.recordings_path} not found -- run "
                f"`msprep manifest nsf_chorus` first")

        recs = {}
        for i, r in enumerate(load_jsonl(self.recordings_path)):
            if limit and len(recs) >= limit:
                break
            recs[r["id"]] = {"duration": r["duration"],
                             "channels": channels_of(r), "index": i}
        print(f"[nsf_chorus] {len(recs)} meetings ({self.split})", flush=True)

        turns: dict[tuple, list] = defaultdict(list)
        speakers: dict[tuple, str] = {}
        n_sup = n_dropped = 0
        for s in load_jsonl(self.supervisions_path):
            rid = s["recording_id"]
            if rid not in recs:
                continue
            n_sup += 1
            text = self.normalize(s["text"])
            if not is_alignable(text):
                n_dropped += 1
                continue
            key = (rid, s["channel"])
            turns[key].append((s["start"], s["start"] + s["duration"], text))
            speakers.setdefault(key, str(s.get("speaker") or f"{rid}_{s['channel']}"))
        print(f"[nsf_chorus] {n_sup} supervisions "
              f"({n_dropped} dropped as non-lexical)", flush=True)

        self.stats = {"split": self.split, "recordings": len(recs),
                      "supervisions": n_sup, "dropped_non_lexical": n_dropped}

        for rid, rec in recs.items():
            for ch, audio in sorted(rec["channels"].items()):
                key = (rid, ch)
                if key not in turns:
                    continue
                yield Channel(
                    file_id=f"{rid}-c{ch}",
                    audio_path=Path(audio),
                    speaker=speakers[key],
                    duration=rec["duration"],
                    turns=turns[key],
                    group=rid,
                    group_index=rec["index"],
                )

    # ------------------------------------------------------------- collect

    @property
    def supervisions(self) -> str:
        return str(self.supervisions_path)

    def stem_key(self, stem: str) -> tuple[str, int]:
        # MTG_30830-c0 -> (MTG_30830, 0)
        rid, _, ch = stem.rpartition("-c")
        return rid, int(ch)
