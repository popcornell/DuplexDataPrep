"""OpenYAP-1K -> speaker channels.

A duplex release: every conversation is two mono 48 kHz FLACs on one timeline,
one per speaker, so each side is already a speaker channel -- no extraction, no
separation. What it lacks is a turn table; `duplexdataprep manifest openyap` builds
utterances from the ASR word times first. The full chain:

    duplexdataprep manifest openyap                       # word times -> manifests
    duplexdataprep restore  openyap --shard i --n-shards N   # Sidon, 48 kHz -> 24 kHz (GPU)
    duplexdataprep prep     openyap --corpus-dir corpus/openyap
    # ... mfa align corpus/openyap/shardNNN ...
    duplexdataprep collect  openyap --aligned-dir out/openyap --output ...
    duplexdataprep dnsmos   openyap --shard i --n-shards N   # restored-audio quality

    audio/openyap_sidon24k/<cid>-a.flac
    corpus/openyap/shardNNN/<cid>-a.flac       -> ../../../audio/openyap_sidon24k/...
    corpus/openyap/shardNNN/<cid>-a.TextGrid   tier "<speaker_a_id>"

Speaker ids (`spk_<hex>`) are corpus-wide: 239 people across 865
conversations, so unlike NOTSOFAR nothing needs scoping.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ...core.manifests import load_jsonl
from ...core.segments import is_alignable
from ..base import Channel, ChannelSource, register

SIDE_OF_CHANNEL = {0: "a", 1: "b"}
CHANNEL_OF_SIDE = {v: k for k, v in SIDE_OF_CHANNEL.items()}


@register
class OpenYap(ChannelSource):
    name = "openyap"

    def configure(self) -> None:
        cfg = self.config
        self.root = Path(cfg["root"])
        m = cfg["manifests"]
        self.manifest_dir = Path(m["dir"])
        self.manifest_names = {
            "recordings": m["recordings"],
            "recordings_restored": m.get("recordings_restored",
                                         "openyap_recordings_sidon24k.jsonl.gz"),
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

        audio = cfg.get("audio", {})
        self.audio_use = audio.get("use", "restored")
        if self.audio_use not in ("restored", "raw"):
            raise ValueError(f"audio.use must be restored|raw, not {self.audio_use!r}")
        self.restored_dir = Path(audio.get("restored_dir", "audio/openyap_sidon24k"))
        self.restored_ext = audio.get("ext", ".flac")
        self.restore_cfg = cfg.get("restore", {})
        self.dnsmos_cfg = cfg.get("dnsmos", {})

    # ---------------------------------------------------------------- paths

    def raw_path(self, cid: str, side: str) -> Path:
        return self.root / cid / f"speaker_{side}.flac"

    def restored_path(self, cid: str, side: str) -> Path:
        return self.restored_dir / f"{cid}-{side}{self.restored_ext}"

    # ---------------------------------------------------------------- text

    def normalize(self, text: str) -> str:
        """Cased, punctuated ASR words; MFA's own cleanup handles both."""
        return " ".join(text.split())

    # ------------------------------------------------------- manifest stage

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
                f"{self.recordings_path} not found -- run `duplexdataprep manifest openyap` first")

        recs = {}
        for i, r in enumerate(load_jsonl(self.recordings_path)):
            if limit and len(recs) >= limit:
                break
            recs[r["id"]] = {"duration": r["duration"], "index": i}
        print(f"[openyap] {len(recs)} conversations", flush=True)

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
            for ch, side in SIDE_OF_CHANNEL.items():
                key = (rid, ch)
                if key not in turns:
                    continue
                path = (self.restored_path(rid, side) if use == "restored"
                        else self.raw_path(rid, side))
                yield Channel(
                    file_id=f"{rid}-{side}",
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
            cid, side = ch.file_id.rsplit("-", 1)
            yield ch, ch.audio_path, self.restored_path(cid, side)

    # ------------------------------------------------------------- collect

    @property
    def supervisions(self) -> str:
        return str(self.supervisions_path)

    def stem_key(self, stem: str) -> tuple[str, int]:
        # <cid>-a -> (<cid>, 0)
        rid, side = stem.rsplit("-", 1)
        return rid, CHANNEL_OF_SIDE[side]
