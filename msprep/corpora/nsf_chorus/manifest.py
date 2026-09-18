"""nsf_chorus (Sidon-restored NOTSOFAR-1 close-talk) -> Lhotse-style manifests.

The audio is ``nsf_chorus``: the NOTSOFAR close-talk headset tracks restored to
24 kHz by the same Sidon pipeline Fisher uses (w2v-BERT feature predictor + DAC
vocoder). One mono ``<Speaker>.wav`` per participant, which is exactly the
speaker-channel shape this package is built around -- each speaker becomes one
channel of a single multi-source recording.

Word times are the one thing ``nsf_chorus`` does not carry: its manifest has
utterance-level ``[start, end, text]`` only. They are taken from two annotation
copies that sit on the *same* timeline:

  * **dev** from the raw release's ``gt_transcription.json``, which carries
    ``word_timing`` for 98.1 % of utterances.
  * **train** and **eval** from the CHiME-8 DASR transcriptions
    (``full_dasr_with_gt_new``, the speaker-label-corrected copy).

No forced alignment is involved, so this corpus never goes through `msprep prep`
/ MFA / `msprep collect` -- the manifests it writes are already word-aligned.

Ported from ``HEAT-TTS/scripts/build_notsofar_manifests.py``. That script built
lhotse objects; this one writes the same JSON directly, because lhotse is not
installed in the `mfa` env and every other `msprep` manifest stage needs nothing
beyond the standard library plus PyYAML. `tools/parity_nsf_chorus.sh` diffs the
output against the manifests the original produced.

    msprep manifest nsf_chorus
"""

from __future__ import annotations

import json
import wave
from collections import Counter, defaultdict
from pathlib import Path

from ...core.manifests import write_jsonl
from . import TAG, clean_text

#: Splits whose word times come from CHiME-8 rather than the raw release.
CHIME8_SPLITS = ("train", "eval")


def _wav_info(path: Path) -> tuple[int, int]:
    """(num_samples, sampling_rate) from a RIFF header.

    The restored wavs are 16-bit PCM, so the stdlib reads them and this stage
    needs no audio library.
    """
    with wave.open(str(path), "rb") as fh:
        return fh.getnframes(), fh.getframerate()


def _align_end(start: float, duration: float) -> float:
    """Lhotse's `AlignmentItem.end`, reproduced so the port matches byte for byte."""
    return round(start + duration, 8)


def build_recordings(nsf_root: Path, limit: int = 0):
    """One recording per meeting; one channel per speaker, ordered by name.

    Returns ``(recordings, {(meeting, speaker|ct_device): channel}, meta)``.
    """
    meetings = [json.loads(line) for line in open(nsf_root / "manifest.jsonl")
                if line.strip()]
    print(f"[nsf_chorus] {len(meetings)} meetings in the nsf_chorus manifest",
          flush=True)
    ids = [m["meeting_id"] for m in meetings]
    if len(set(ids)) != len(ids):
        raise ValueError("meeting_id is not unique across splits; ids would collide")
    if limit:
        meetings = meetings[:limit]

    recordings, channel_of, missing, meta = [], {}, [], {}
    for m in meetings:
        mid = m["meeting_id"]
        sources, num_samples, sampling_rate = [], None, None
        for i, name in enumerate(sorted(m["speakers"])):
            spk = m["speakers"][name]
            wav = nsf_root / spk["wav"]
            if not wav.exists():
                missing.append(str(wav))
                continue
            # Read the header rather than trusting the upstream manifest, and
            # take the shortest channel: a meeting's speakers must share one
            # timeline, so the recording is only as long as its shortest track.
            frames, rate = _wav_info(wav)
            num_samples = frames if num_samples is None else min(num_samples, frames)
            sampling_rate = rate
            sources.append({"type": "file", "channels": [i], "source": str(wav)})
            channel_of[(mid, name)] = i
            channel_of[(mid, spk["ct_device"])] = i
        if not sources:
            continue
        recordings.append({
            "id": mid,
            "sources": sources,
            "sampling_rate": sampling_rate,
            "num_samples": num_samples,
            "duration": num_samples / sampling_rate,
            "channel_ids": list(range(len(sources))),
        })
        # The split is what everything downstream filters on, and a recording
        # manifest has nowhere to put it, so it goes in a sidecar.
        meta[mid] = {"split": m["split"], "room": m.get("room"),
                     "n_speakers": m["n_speakers"], "corpus": "notsofar"}
    if missing:
        print(f"[nsf_chorus] !! {len(missing)} wav files missing, "
              f"e.g. {missing[0]}", flush=True)
    return recordings, channel_of, meta


def raw_ground_truth(chime_mtg: Path):
    """(meeting, entries) from the raw-release dev `gt_transcription.json`."""
    for gt_file in sorted(chime_mtg.glob("*/gt_transcription.json")):
        yield gt_file.parent.name, json.load(open(gt_file))


def chime8_ground_truth(transcriptions: Path):
    """(meeting, entries) from CHiME-8 DASR transcriptions, one per meeting.

    CHiME-8 ships one transcript per far-field device session (S30830107,
    S30830207, ... for MTG_30830). They are the same annotation on the same
    timeline -- every utterance and first-word time agrees across the sessions
    of a meeting -- so the first session stands in for the meeting. Note that
    CHiME-8's `train` holds NOTSOFAR's train *and* dev meetings; the caller's
    `known` set decides which are kept.
    """
    first: dict[str, Path] = {}
    for f in sorted(transcriptions.glob("S*.json")):
        first.setdefault(f"MTG_{f.stem[1:6]}", f)
    for mid in sorted(first):
        yield mid, json.load(open(first[mid]))


def build_supervisions(ground_truth, known: set, channel_of: dict):
    """Word-timed supervisions on the nsf_chorus timeline.

    Reads both ground-truth schemas: the raw release (`speaker_id`, `text`) and
    CHiME-8 (`speaker`, `words`). The segment's own `start_time` / `end_time`
    are used only for ordering -- the span written is the one the *words* span,
    so the supervision cannot claim audio no word covers.
    """
    segments: list[dict] = []
    no_wt = unmatched = sessions = empty_utts = 0
    stripped: Counter = Counter()

    for mid, entries in ground_truth:
        if mid not in known:
            continue    # other split, or ground truth with no nsf_chorus audio
        sessions += 1
        per_speaker = defaultdict(list)
        for e in entries:
            wt = e.get("word_timing") or []
            if not wt:
                no_wt += 1
                continue
            ct = Path(e.get("ct_wav_file_name", "")).stem  # close_talk/CT_21.wav
            speaker = e.get("speaker_id") or e.get("speaker")
            ch = channel_of.get((mid, speaker))
            if ch is None:
                ch = channel_of.get((mid, ct))
            if ch is None:
                unmatched += 1
                continue
            per_speaker[(ch, speaker)].append((e, wt))

        for (ch, spk), items in sorted(per_speaker.items()):
            items.sort(key=lambda p: float(p[0]["start_time"]))
            for i, (e, wt) in enumerate(items):
                words = []
                for w in wt:
                    raw = str(w[0])
                    symbol = clean_text(raw)
                    if not symbol:
                        stripped[raw] += 1
                        continue
                    start, end = float(w[1]), float(w[2])
                    words.append([symbol, start, max(end - start, 1e-3), None])
                if not words:
                    empty_utts += 1   # markup only, e.g. a lone <filllaugh/>
                    continue
                start = min(w[1] for w in words)
                end = max(_align_end(w[1], w[2]) for w in words)
                segments.append({
                    "id": f"{mid}-c{ch}-{i:05d}",
                    "recording_id": mid,
                    "start": start,
                    "duration": end - start,
                    "channel": ch,
                    "text": clean_text(e.get("text", e.get("words", ""))),
                    "language": "English",
                    "speaker": spk,
                    "alignment": {"word": words},
                })

    print(f"[nsf_chorus]   {no_wt} utterances had no word_timing, "
          f"{unmatched} unmatched speakers", flush=True)
    print(f"[nsf_chorus]   {sum(stripped.values())} markup tokens dropped "
          f"{dict(stripped)}", flush=True)
    print(f"[nsf_chorus]   {empty_utts} utterances were markup only and were dropped",
          flush=True)
    # Consumers that stream supervisions flush a recording when the id changes,
    # so they must be contiguous per recording.
    segments.sort(key=lambda s: (s["recording_id"], s["channel"], s["start"]))
    return segments, sessions


def build_manifests(source, split=None, out_dir=None, limit=0, report=None,
                    log=print) -> dict:
    """Write recordings, the split sidecar, and per-split supervisions."""
    cfg = source.config
    nsf_root = Path(cfg["root"])
    chime_mtg = Path(cfg["dev_ground_truth"])
    chime8 = Path(cfg["chime8_transcriptions"])
    out = Path(out_dir or cfg["manifests"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    names = cfg["manifests"]

    recordings, channel_of, meta = build_recordings(nsf_root, limit=limit)
    rec_path = out / names["recordings"]
    write_jsonl(rec_path, recordings)
    log(f"[nsf_chorus] {len(recordings)} recordings -> {rec_path}", flush=True)

    meta_path = out / names["meta"]
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True))
    counts = Counter(v["split"] for v in meta.values())
    log(f"[nsf_chorus] split sidecar -> {meta_path}  {dict(counts)}", flush=True)

    sources = {
        "dev": raw_ground_truth(chime_mtg),
        "train": chime8_ground_truth(chime8 / "train"),
        "eval": chime8_ground_truth(chime8 / "eval"),
    }
    wanted = [split] if split and split != "all" else list(sources)

    rec_ids = {r["id"] for r in recordings}
    stats: dict = {"recordings": len(recordings),
                   "channels": sum(len(r["channel_ids"]) for r in recordings),
                   "meetings_by_split": dict(counts), "per_split": {}}
    per_split: dict[str, list] = {}

    for name in wanted:
        known = {m for m in rec_ids if meta[m]["split"] == name}
        sups, sessions = build_supervisions(sources[name], known, channel_of)
        sup_path = out / names["supervisions"].format(split=name)
        write_jsonl(sup_path, sups)
        n_words = sum(len(s["alignment"]["word"]) for s in sups)
        log(f"[nsf_chorus] {len(sups)} supervisions from {sessions}/{len(known)} "
            f"{name} meetings -> {sup_path}", flush=True)
        log(f"[nsf_chorus]   {n_words} aligned words", flush=True)
        per_split[name] = sups
        stats["per_split"][name] = {
            "supervisions": len(sups), "aligned_words": n_words,
            "meetings": len({s["recording_id"] for s in sups}),
            "output": str(sup_path),
        }

    # The default corpus is dev + train; eval is the benchmark split and is
    # deliberately kept out of it.
    if all(k in per_split for k in ("dev", "train")):
        merged = sorted(per_split["dev"] + per_split["train"],
                        key=lambda s: (s["recording_id"], s["channel"], s["start"]))
        merged_path = out / names["merged"]
        write_jsonl(merged_path, merged)
        log(f"[nsf_chorus] {len(merged)} dev+train supervisions -> {merged_path}",
            flush=True)
        stats["merged"] = {"supervisions": len(merged), "output": str(merged_path)}

    # Every supervision must sit inside its recording and name a real channel.
    by_id = {r["id"]: r for r in recordings}
    bad = 0
    for rows in per_split.values():
        for s in rows:
            r = by_id[s["recording_id"]]
            if (s["start"] < -1e-6
                    or s["start"] + s["duration"] > r["duration"] + 1e-6
                    or s["channel"] not in r["channel_ids"]):
                bad += 1
    log(f"[nsf_chorus]   {bad} supervisions fall outside their recording "
        f"or channel set", flush=True)
    stats["supervisions_out_of_bounds"] = bad
    stats.update({"recordings_output": str(rec_path), "meta": str(meta_path)})

    if report:
        Path(report).write_text(json.dumps(stats, indent=2))
    return stats
