"""otoSpeech (full-duplex-processed-141h) -> speaker channels.

A duplex release packaged as WebDataset: every session is one *stereo* 44.1 kHz
FLAC plus a JSON whose `speakers[].channel` says which side each speaker is on.
The release has no transcripts; the Parakeet v3 pseudo labels published with
Lhotse's `oto_speech` recipe (`seglst.json`) supply per-word times. The chain:

    duplexdataprep extract  otospeech --shard i --n-shards N   # tar -> per-speaker mono FLAC
    duplexdataprep manifest otospeech                          # word times -> manifests
    duplexdataprep restore  otospeech --shard i --n-shards N   # Sidon -> 24 kHz (GPU)
    duplexdataprep prep     otospeech --corpus-dir corpus/otospeech
    # ... mfa align corpus/otospeech/shardNNN ...
    duplexdataprep collect  otospeech --aligned-dir out/otospeech --output ...

    audio/otospeech/<sid>-ch0.flac               extract output, native rate
    audio/otospeech_sidon24k/<sid>-ch0.flac      restore output
    corpus/otospeech/shardNNN/<sid>-ch0.TextGrid tier "<speaker_id>"

Download (gated HF dataset, needs HF_TOKEN and gdown) with Lhotse:

    python -c "from lhotse.recipes import download_oto_speech; download_oto_speech('<root>')"

Speaker ids are corpus-wide hashes (165 people across 538 labelled sessions).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ...core.manifests import load_jsonl
from ...core.segments import is_alignable
from ..base import Channel, ChannelSource, register

CHANNELS = (0, 1)


@register
class OtoSpeech(ChannelSource):
    name = "otospeech"

    def configure(self) -> None:
        cfg = self.config
        self.shards_dir = Path(cfg["shards_dir"])
        self.labels_path = Path(cfg["labels"])
        m = cfg["manifests"]
        self.manifest_dir = Path(m["dir"])
        self.manifest_names = {
            "recordings": m["recordings"],
            "recordings_restored": m.get("recordings_restored",
                                         "otospeech_recordings_sidon24k.jsonl.gz"),
            "supervisions": m["supervisions"],
        }
        self.recordings_path = self.manifest_dir / m["recordings"]
        self.supervisions_path = self.manifest_dir / m["supervisions"]

        seg = cfg.get("segment", {})
        self.segment_cfg = {
            "pause_split": float(seg.get("pause_split", 1.0)),
            "max_dur": float(seg.get("max_dur", 20.0)),
            "min_gap": float(seg.get("min_gap", 0.30)),
            "pad": float(seg.get("pad", 0.30)),
        }
        self.keep_word_timing = bool(cfg.get("keep_word_timing", True))
        self.drop_redacted = bool(cfg.get("drop_redacted", True))

        audio = cfg.get("audio", {})
        self.audio_use = audio.get("use", "restored")
        if self.audio_use not in ("restored", "raw"):
            raise ValueError(f"audio.use must be restored|raw, not {self.audio_use!r}")
        self.raw_dir = Path(audio.get("raw_dir", "audio/otospeech"))
        self.restored_dir = Path(audio.get("restored_dir", "audio/otospeech_sidon24k"))
        self.restored_ext = audio.get("ext", ".flac")
        self.restore_cfg = cfg.get("restore", {})
        self.dnsmos_cfg = cfg.get("dnsmos", {})

    # ---------------------------------------------------------------- paths

    def meta_path(self, sid: str) -> Path:
        return self.raw_dir / f"{sid}.json"

    def raw_path(self, sid: str, ch: int) -> Path:
        return self.raw_dir / f"{sid}-ch{ch}.flac"

    def restored_path(self, sid: str, ch: int) -> Path:
        return self.restored_dir / f"{sid}-ch{ch}{self.restored_ext}"

    # ---------------------------------------------------------------- text

    def normalize(self, text: str) -> str:
        """Cased, punctuated ASR words; MFA's own cleanup handles both."""
        return " ".join(text.split())

    # -------------------------------------------------- extract / manifest

    def extract(self, **kwargs) -> dict:
        from .extract import extract
        return extract(self, **kwargs)

    def build_manifests(self, **kwargs) -> dict:
        from .manifest import build_manifests
        return build_manifests(self, **kwargs)

    # ------------------------------------------------------------ channels

    def iter_channels(self, limit: int = 0, audio: str | None = None):
        """`audio` overrides `audio.use` -- restore reads `raw`, DNSMOS can
        score either."""
        use = audio or self.audio_use
        if not self.recordings_path.exists():
            raise FileNotFoundError(
                f"{self.recordings_path} not found -- run `duplexdataprep manifest "
                "otospeech` first")

        recs = {}
        for i, r in enumerate(load_jsonl(self.recordings_path)):
            if limit and len(recs) >= limit:
                break
            recs[r["id"]] = {"duration": r["duration"], "index": i}
        print(f"[otospeech] {len(recs)} sessions", flush=True)

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

        self.stats = {
            "recordings": len(recs),
            "supervisions": n_sup,
            "dropped_non_lexical": n_dropped,
            "audio": use,
        }

        for rid, rec in recs.items():
            for ch in CHANNELS:
                key = (rid, ch)
                if key not in turns:
                    continue
                path = (self.restored_path(rid, ch) if use == "restored"
                        else self.raw_path(rid, ch))
                yield Channel(
                    file_id=f"{rid}-ch{ch}",
                    audio_path=path,
                    speaker=speakers[key],
                    duration=rec["duration"],
                    turns=turns[key],
                    group=rid,
                    group_index=rec["index"],
                )

    # -------------------------------------------------------------- restore

    def restore_items(self, limit: int = 0):
        """(channel, src_path, dst_path) for `duplexdataprep restore`."""
        for ch in self.iter_channels(limit=limit, audio="raw"):
            rid, c = self.stem_key(ch.file_id)
            yield ch, ch.audio_path, self.restored_path(rid, c)

    # ------------------------------------------------------------- collect

    @property
    def supervisions(self) -> str:
        return str(self.supervisions_path)

    def stem_key(self, stem: str) -> tuple[str, int]:
        # <sid>-ch0 -> (<sid>, 0)
        rid, ch = stem.rsplit("-ch", 1)
        return rid, int(ch)
