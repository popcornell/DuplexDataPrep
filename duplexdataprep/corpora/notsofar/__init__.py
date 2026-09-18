"""NOTSOFAR-1 -> speaker channels (close-talk track).

Unlike Fisher and CANDOR, NOTSOFAR ships no Lhotse manifests, so there is an
extra stage first:

    duplexdataprep manifest notsofar --split train_set    # raw tree -> JSONL manifests
    duplexdataprep prep     notsofar --corpus-dir corpus/notsofar

One close-talk wav is one speaker channel, so the corpus fits the same shape as
the others once the manifests exist:

    corpus/notsofar/shardNNN/train_set__<session>__<ct_stem>.wav       (symlink)
    corpus/notsofar/shardNNN/train_set__<session>__<ct_stem>.TextGrid  tier "<speaker_id>"

**Only the close-talk track.** A far-field array channel carries every speaker
at once; aligning it would need separation first. It is an evaluation target,
not a ChannelSource, and `mic: far` raises rather than pretending otherwise.

Leakage: gating a channel to its own oracle segmentation (planned
`core.gating`) removes *between-turn* leakage from the other participants in the
room. Leakage during a speaker's own turn -- which in a real meeting room is the
dominant kind -- stays, so channel-quality filtering is still needed downstream.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from ...core.manifests import channels_of, load_jsonl
from ...core.segments import is_alignable
from ..base import Channel, ChannelSource, register

# NOTSOFAR annotates with angle-bracket tags of two shapes: standalone markers
# (`<ST/>`, `<FILL/>`, `<FILLlaugh/>`, `<UNKNOWN/>`, `<BA/>`, `<ISSUE/>`,
# `<SN/>`, `<PAUSE/>`) and the paired `<PName>Bert</PName>`, which wraps a real
# spoken name. Dropping every tag while keeping what sits between them handles
# both: the name survives as a word, the annotation does not. Square-bracket
# events are deliberately left alone -- MFA maps those to `spn`, which keeps the
# audio under them accounted for.
MARKER_RE = re.compile(r"<[^<>]*>")


@register
class Notsofar(ChannelSource):
    name = "notsofar"

    def configure(self) -> None:
        cfg = self.config
        self.split = cfg.get("split", "train_set")
        self.mic = cfg.get("mic", "near")
        m = cfg["manifests"]
        d = Path(m["dir"])
        self.recordings_path = d / m["recordings"].format(split=self.split)
        self.supervisions_path = d / m["supervisions"].format(split=self.split)
        text_cfg = cfg.get("text", {})
        self.strip_markers = list(text_cfg.get("strip_markers", []))
        self.strip_all_markers = bool(text_cfg.get("strip_all_markers", True))
        self.warn_turn_over = float(cfg.get("warn_turn_over", 20.0))

    # ---------------------------------------------------------------- text

    def normalize(self, text: str) -> str:
        """NOTSOFAR ground truth -> MFA-friendly text.

        Every angle-bracket tag is annotation rather than a word, so all of them
        go (`strip_all_markers`), keeping any text between a paired tag --
        `<PName>Bert</PName>` is the spoken name "Bert". An explicit
        `strip_markers` list is still applied afterwards for anything literal a
        release adds that is not tag-shaped.

        Listing markers one by one was the earlier rule and it under-matched: a
        dev_set run left 508 `<FILLlaugh/>`, 197 `<BA/>`, 143 `<ISSUE/>` and 189
        `<PName>` pairs in the TextGrids, where MFA reads them as bracketed
        tokens and maps each to `spn` -- silently charging real words' audio to a
        non-speech phone.

        Square-bracket events and partial words are left alone on purpose, so
        MFA routes them to `spn` and the audio under them stays accounted for
        instead of being absorbed into a neighbouring word.
        """
        if self.strip_all_markers:
            text = MARKER_RE.sub(" ", text)
        for marker in self.strip_markers:
            text = text.replace(marker, " ")
        return " ".join(text.split())

    # ------------------------------------------------------- manifest stage

    def build_manifests(self, **kwargs) -> dict:
        from .manifest import build_manifests
        return build_manifests(self, **kwargs)

    # ------------------------------------------------------------ channels

    def iter_channels(self, limit: int = 0):
        if self.mic != "near":
            raise ValueError(
                "duplexdataprep only prepares the NOTSOFAR close-talk track (`mic: near`). "
                "Far-field channels are mixtures and cannot be force aligned.")
        if not self.recordings_path.exists():
            raise FileNotFoundError(
                f"{self.recordings_path} not found -- run "
                f"`duplexdataprep manifest notsofar --split {self.split}` first")

        recs = {}
        for i, r in enumerate(load_jsonl(self.recordings_path)):
            if limit and len(recs) >= limit:
                break
            recs[r["id"]] = {
                "duration": r["duration"],
                "channels": channels_of(r),
                "index": i,
            }
        print(f"[notsofar] {len(recs)} close-talk channels ({self.split})", flush=True)

        turns: dict[tuple, list] = defaultdict(list)
        speakers: dict[tuple, str] = {}
        n_sup = n_dropped = n_long = 0
        for s in load_jsonl(self.supervisions_path):
            rid = s["recording_id"]
            if rid not in recs:
                continue
            n_sup += 1
            text = self.normalize(s["text"])
            if not is_alignable(text):
                n_dropped += 1
                continue
            if s["duration"] > self.warn_turn_over:
                n_long += 1
            key = (rid, s["channel"])
            turns[key].append((s["start"], s["start"] + s["duration"], text))
            speakers.setdefault(key, str(s.get("speaker") or f"{rid}_{s['channel']}"))
        print(f"[notsofar] {n_sup} supervisions ({n_dropped} dropped as non-lexical, "
              f"{n_long} longer than {self.warn_turn_over}s)", flush=True)

        self.stats = {
            "split": self.split,
            "recordings": len(recs),
            "supervisions": n_sup,
            "dropped_non_lexical": n_dropped,
            "turns_over_warn_length": n_long,
        }

        for rid, rec in recs.items():
            for ch, audio in sorted(rec["channels"].items()):
                key = (rid, ch)
                if key not in turns:
                    continue
                yield Channel(
                    file_id=rid,          # split__session__ct_stem
                    audio_path=Path(audio),
                    speaker=speakers[key],
                    duration=rec["duration"],
                    turns=turns[key],
                    group=rid,            # one speaker per recording already
                    group_index=rec["index"],
                )

    # ------------------------------------------------------------- collect

    @property
    def supervisions(self) -> str:
        return str(self.supervisions_path)

    def stem_key(self, stem: str) -> tuple[str, int]:
        # The file stem *is* the recording id, and a close-talk wav is mono.
        return stem, 0
