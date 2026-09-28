"""CANDOR -> speaker channels.

CANDOR ships one 48 kHz *stereo* mp3 per conversation, channel 0 / channel 1
carrying one speaker each. `duplexdataprep extract candor` splits it into two 16 kHz
mono FLACs up front so MFA gets exact seeks and no crosstalk; this adapter then
writes a single-tier TextGrid per side.

    audio/candor/<cid>-ch0.flac
    corpus/candor/shardNNN/<cid>-ch0.flac      -> ../../../audio/candor/...
    corpus/candor/shardNNN/<cid>-ch0.TextGrid  tier "<user_id>"

Turns longer than `resegment.max_dur` are cut at their widest inter-word pauses;
CANDOR's turn tables otherwise contain aggregated backchannel runs up to ~52
minutes long, which forced alignment cannot handle.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ...core.manifests import load_jsonl
from ...core.segments import is_alignable, resegment_on_words
from ..base import Channel, ChannelSource, register


@register
class Candor(ChannelSource):
    name = "candor"

    def configure(self) -> None:
        m = self.config["manifests"]
        root = Path(m["root"])
        self.variant = self.config.get("variant", "audiophile")
        self.recordings_path = root / m["recordings"]
        self.supervisions_path = root / m["supervisions"].format(variant=self.variant)

        audio = self.config["audio"]
        self.audio_dir = Path(audio["dir"])
        self.audio_ext = audio.get("ext", ".flac")

        reseg = self.config.get("resegment", {})
        self.max_dur = float(reseg.get("max_dur", 20.0))
        self.min_gap = float(reseg.get("min_gap", 0.30))

    # ---------------------------------------------------------------- text

    def normalize(self, text: str) -> str:
        """CANDOR transcript -> MFA-friendly text.

        CANDOR is cased and punctuated ASR output. MFA's own cleanup handles
        both, so this mostly guards against empty/whitespace-only turns.
        """
        return " ".join(text.split())

    def extract(self, **kwargs) -> dict:
        from .extract import extract
        return extract(self, **kwargs)

    # ------------------------------------------------------------ channels

    def iter_channels(self, limit: int = 0):
        recs = {}
        for i, r in enumerate(load_jsonl(self.recordings_path)):
            if limit and len(recs) >= limit:
                break
            recs[r["id"]] = {"duration": r["duration"], "index": i}
        print(f"[candor] {len(recs)} recordings", flush=True)

        turns: dict[tuple, list] = defaultdict(list)
        speakers: dict[tuple, str] = {}
        n_sup = n_dropped = n_split = n_split_out = 0
        for s in load_jsonl(self.supervisions_path):
            rid = s["recording_id"]
            if rid not in recs:
                continue
            n_sup += 1
            key = (rid, s["channel"])
            speakers.setdefault(key, str(s.get("speaker") or f"{rid}_{s['channel']}"))

            words = (s.get("alignment") or {}).get("word") or []
            pieces = resegment_on_words(s["start"], s["duration"], words,
                                        max_dur=self.max_dur, min_gap=self.min_gap)
            if len(pieces) > 1:
                n_split += 1
                n_split_out += len(pieces)

            for start, end, sub_text in pieces:
                text = self.normalize(sub_text if sub_text is not None else s["text"])
                if not is_alignable(text):
                    n_dropped += 1
                    continue
                turns[key].append((start, end, text))

        print(f"[candor] {n_sup} supervisions -> {sum(len(v) for v in turns.values())} "
              f"utterances ({n_split} split into {n_split_out}, "
              f"{n_dropped} dropped as non-lexical)", flush=True)

        self.stats = {
            "variant": self.variant,
            "recordings": len(recs),
            "supervisions": n_sup,
            "resegmented_supervisions": n_split,
            "resegmented_into": n_split_out,
            "dropped_non_lexical": n_dropped,
        }

        for rid, rec in recs.items():
            for ch in (0, 1):
                key = (rid, ch)
                if key not in turns:
                    continue
                yield Channel(
                    file_id=f"{rid}-ch{ch}",
                    audio_path=self.audio_dir / f"{rid}-ch{ch}{self.audio_ext}",
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
        # <cid>-ch0 -> (<cid>, 0)
        rid, ch = stem.rsplit("-ch", 1)
        return rid, int(ch)
