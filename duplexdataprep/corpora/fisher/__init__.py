"""Fisher (sidon-restored) -> speaker channels.

Fisher ships one mono FLAC per conversation side, so no audio is rewritten --
each side is symlinked into its shard directory with a single-tier TextGrid
beside it.

    corpus/fisher/shardNNN/fe_03_00001-A.flac      -> fisher_wavs_sidon_24k/...
    corpus/fisher/shardNNN/fe_03_00001-A.TextGrid  tier "2602", one interval/turn
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from ...core.manifests import channels_of, load_jsonl
from ...core.segments import is_alignable
from ..base import Channel, ChannelSource, register

# Fisher writes acronyms as "u._s." / "t._v." -- one letter per spoken token.
_ACRONYM = re.compile(r"\b(?:[a-z]\._)+[a-z]\.?")
# "(( maybe this ))" marks a low-confidence stretch; keep the words, drop the marks.
_UNCERTAIN = re.compile(r"\(\(|\)\)")
# Bracketed non-speech events Fisher uses; MFA maps bracketed tokens to `spn`,
# which is what we want -- the audio under them is real, just not lexical.
EVENTS = {
    "[laughter]", "[laugh]", "[noise]", "[mn]", "[sigh]",
    "[lipsmack]", "[cough]", "[breath]", "[sneeze]", "[pause]",
}


@register
class Fisher(ChannelSource):
    name = "fisher"

    def configure(self) -> None:
        m = self.config["manifests"]
        self.recordings_path = Path(m["recordings"])
        self.supervisions_path = Path(m["supervisions"])
        self.text_cfg = self.config.get("text", {})

    # ---------------------------------------------------------------- text

    def normalize(self, text: str) -> str:
        """Fisher transcript -> MFA-friendly text.

        Conservative on purpose: MFA already lowercases, strips punctuation and
        routes out-of-vocabulary / bracketed tokens to `spn`, so only the things
        it would otherwise get wrong are fixed here. With `keep_events` (the
        default) bracketed events and partial words (`th-`, `-cause`) are left
        alone -- `spn` keeps the audio under them accounted for instead of
        letting it be absorbed into a neighbouring word.
        """
        if self.text_cfg.get("strip_uncertainty", True):
            text = _UNCERTAIN.sub(" ", text)
        if self.text_cfg.get("expand_acronyms", True):
            text = _ACRONYM.sub(
                lambda m: " ".join(c for c in m.group(0) if c.isalpha()), text)
        if not self.text_cfg.get("keep_events", True):
            text = " ".join(w for w in text.split() if w.lower() not in EVENTS)
        return " ".join(text.split())

    # ------------------------------------------------------------ channels

    def iter_channels(self, limit: int = 0):
        recs = {}
        for i, r in enumerate(load_jsonl(self.recordings_path)):
            if limit and len(recs) >= limit:
                break
            recs[r["id"]] = {
                "duration": r["duration"],
                "channels": channels_of(r),
                "index": i,
            }
        print(f"[fisher] {len(recs)} recordings", flush=True)

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
        print(f"[fisher] {n_sup} supervisions ({n_dropped} dropped as non-lexical)",
              flush=True)

        self.stats = {
            "recordings": len(recs),
            "supervisions": n_sup,
            "dropped_non_lexical": n_dropped,
        }

        for rid, rec in recs.items():
            for ch, audio in sorted(rec["channels"].items()):
                key = (rid, ch)
                if key not in turns:
                    continue
                audio = Path(audio)
                yield Channel(
                    file_id=audio.stem,          # fe_03_00001-A
                    audio_path=audio,
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
        # fe_03_00001-A -> (fe_03_00001, 0)
        rid, side = stem.rsplit("-", 1)
        return rid, {"A": 0, "B": 1}.get(side, 0)
