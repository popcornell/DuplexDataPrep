"""NOTSOFAR-1 raw release -> Lhotse-style manifests for the close-talk track.

The download lays out (see the challenge repo's `utils/azure_storage.py`):

    <root>/<subset>/<version>/MTG/<session>/
        close_talk/*.wav          one headset per participant   <- what we use
        mc*/ sc*/ *.wav           far-field array devices
        gt_transcription.json     [{speaker_id, start_time, end_time, text,
                                    ct_wav_file_name}, ...]
        gt_meeting_metadata.json  {"ParticipantAliasToCtDevice": {...}, ...}

Only the close-talk track becomes a ChannelSource. A far-field array channel
carries every speaker at once, so it cannot be force aligned without separation
first -- it is an evaluation target, not a source. `mic: far` is refused rather
than silently producing one "speaker channel" per array microphone.

    msprep manifest notsofar --split train_set

Durations are read from the wav headers with the stdlib `wave` module, so this
stage needs no audio library.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

from ...core.manifests import write_jsonl
from . import MARKER_RE


def _seg_text(seg: dict) -> str:
    """The transcript field. `gt_transcription.json` calls it `text`; the
    chime-utils reformat of the same data calls it `words`."""
    v = seg.get("text")
    return v if v is not None else (seg.get("words") or "")


def _seg_speaker(seg: dict) -> str:
    """Likewise `speaker_id` in the raw release, `speaker` in the reformat."""
    v = seg.get("speaker_id")
    return str(v if v is not None else seg.get("speaker") or "")


def speaker_id(session: str, alias: str, scope: str = "session") -> str:
    """Participant alias -> an id that identifies one *person*.

    NOTSOFAR aliases are per-meeting cast names ("Bert", "Ernie", "Olivia"),
    reused across sessions by different people: the dev_set's 201 close-talk
    channels carry only 12 distinct alias strings, and "Bert" alone is 31
    different speakers in 31 meetings. Left unscoped they collapse into one
    identity, which silently corrupts anything that samples distinct speakers --
    FastMSS would build "meetings" whose participants are the same person, and
    speaker-conditioned models would train on a 31-way blend.

    The alias is unique *within* a meeting, so scoping by session recovers the
    true identity. `scope: global` keeps the raw alias for a release that
    genuinely has corpus-wide speaker ids.
    """
    if scope == "global":
        return alias
    return f"{session}__{alias}"


def _word_alignment(seg: dict) -> list:
    """`word_timing` -> Lhotse alignment items [symbol, start, duration].

    The release ships per-word times (98.1 % of dev_set segments have them) in
    absolute recording seconds as [symbol, start, end]; the chime-utils reformat
    writes the numbers as strings. Marker tokens (`<fill/>`, `<unknown/>`) are
    dropped so the alignment carries the same words the normalized text does.
    """
    out = []
    for tok in seg.get("word_timing") or []:
        if len(tok) < 3:
            continue
        symbol = str(tok[0])
        if MARKER_RE.fullmatch(symbol):
            continue
        start, end = float(tok[1]), float(tok[2])
        out.append([symbol, round(start, 4), round(end - start, 4)])
    return out


def _wav_info(path: Path) -> tuple[int, int]:
    """(num_samples, sampling_rate) from a RIFF header."""
    with wave.open(str(path), "rb") as fh:
        return fh.getnframes(), fh.getframerate()


def _resolve_version(split_dir: Path, version: str | None) -> Path:
    """`<split>/<version>/MTG`, with `version: auto` picking the only one there."""
    if version and version != "auto":
        return split_dir / version / "MTG"
    versions = sorted(d for d in split_dir.iterdir() if d.is_dir())
    if not versions:
        raise FileNotFoundError(f"no version directory under {split_dir}")
    if len(versions) > 1:
        raise ValueError(
            f"{split_dir} holds {len(versions)} versions ({[v.name for v in versions]}); "
            "set `version:` in the config to pick one")
    return versions[0] / "MTG"


def session_id(split: str, session: str, ct_stem: str) -> str:
    """Recording id == MFA file stem. Flat, filename-safe, and its own inverse."""
    for part in (split, session, ct_stem):
        if "__" in part:
            raise ValueError(
                f"'__' is the id separator but appears in {part!r}; "
                "set `id_separator` in the config to something else")
    return f"{split}__{session}__{ct_stem}"


def build_manifests(source, split=None, out_dir=None, limit=0, report=None,
                    log=print) -> dict:
    """Write `<out_dir>/notsofar_{recordings,supervisions}_<split>.jsonl.gz`."""
    cfg = source.config
    if cfg.get("mic", "near") != "near":
        raise ValueError(
            "msprep only prepares the NOTSOFAR close-talk track (`mic: near`). "
            "Far-field channels are mixtures and cannot be force aligned.")

    root = Path(cfg["root"])
    split = split or cfg.get("split", "train_set")
    out_dir = Path(out_dir or cfg["manifests"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    scope = cfg.get("speaker_scope", "session")
    keep_word_timing = bool(cfg.get("keep_word_timing", True))

    mtg = _resolve_version(root / split, cfg.get("version", "auto"))
    sessions = sorted(d for d in mtg.iterdir() if d.is_dir())
    if limit:
        sessions = sessions[:limit]
    log(f"[notsofar] {len(sessions)} sessions under {mtg}", flush=True)

    recordings, supervisions = [], []
    stats = {"split": split, "sessions": len(sessions), "sessions_no_close_talk": 0,
             "sessions_no_transcript": 0, "channels": 0, "segments": 0,
             "segments_unmatched_ct": 0, "speakers": set(), "speaker_aliases": set(),
             "segments_with_word_timing": 0}

    for sess in sessions:
        ct_dir = sess / "close_talk"
        if not ct_dir.is_dir():
            stats["sessions_no_close_talk"] += 1
            continue
        gt = sess / "gt_transcription.json"
        if not gt.is_file():
            stats["sessions_no_transcript"] += 1
            continue

        by_ct_name = {}
        for wav in sorted(ct_dir.glob("*.wav")):
            rid = session_id(split, sess.name, wav.stem)
            n_samples, rate = _wav_info(wav)
            recordings.append({
                "id": rid,
                "sources": [{"type": "file", "channels": [0], "source": str(wav)}],
                "sampling_rate": rate,
                "num_samples": n_samples,
                "duration": round(n_samples / rate, 6),
                "channel_ids": [0],
            })
            by_ct_name[wav.name] = rid
            stats["channels"] += 1

        with open(gt, "rt", encoding="utf-8") as fh:
            segments = json.load(fh)

        for i, seg in enumerate(segments):
            ct_name = Path(seg.get("ct_wav_file_name") or "").name
            rid = by_ct_name.get(ct_name)
            if rid is None:
                # A segment whose close-talk device was not released with the
                # session; it belongs to no single channel, so it is dropped.
                stats["segments_unmatched_ct"] += 1
                continue
            start = float(seg["start_time"])
            alias = _seg_speaker(seg)
            spk = speaker_id(sess.name, alias, scope)
            sup = {
                "id": f"{rid}-{i:05d}",
                "recording_id": rid,
                "start": round(start, 4),
                "duration": round(float(seg["end_time"]) - start, 4),
                "channel": 0,
                "language": "English",
                "speaker": spk,
                "text": _seg_text(seg),
            }
            if keep_word_timing:
                words = _word_alignment(seg)
                if words:
                    sup["alignment"] = {"word": words}
                    stats["segments_with_word_timing"] += 1
            supervisions.append(sup)
            stats["segments"] += 1
            stats["speakers"].add(spk)
            stats["speaker_aliases"].add(alias)

    rec_path = out_dir / f"notsofar_recordings_{split}.jsonl.gz"
    sup_path = out_dir / f"notsofar_supervisions_{split}.jsonl.gz"
    write_jsonl(rec_path, recordings)
    write_jsonl(sup_path, supervisions)

    stats["speakers"] = len(stats["speakers"])
    stats["speaker_aliases"] = len(stats["speaker_aliases"])
    stats.update({"recordings": str(rec_path), "supervisions": str(sup_path),
                  "speaker_scope": scope})
    log(f"[notsofar] {stats['channels']} close-talk channels, "
        f"{stats['segments']} segments, {stats['speakers']} speakers", flush=True)
    if report:
        Path(report).write_text(json.dumps(stats, indent=2))
    return stats
