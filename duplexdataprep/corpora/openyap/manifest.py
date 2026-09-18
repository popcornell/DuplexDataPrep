"""OpenYAP-1K release -> Lhotse-style manifests.

The release has no turn table: each speaker channel ships a flat list of ASR
word times. Utterances are built here, once, because `collect` later buckets
MFA's words back into exactly these spans -- the supervision manifest *is* the
segmentation, and prep must not re-derive it.

    duplexdataprep manifest openyap                 # every complete conversation
    duplexdataprep manifest openyap --limit 5       # pilot

Writes three files under `manifests.dir`:

    recordings            one recording per conversation, channel 0 = speaker_a
                          and channel 1 = speaker_b, pointing at the raw 48 kHz
    recordings_restored   the same, pointing at `duplexdataprep restore` output. Sidon
                          output is length-matched, so supervisions are shared
    supervisions          one per utterance, with the ASR word times attached as
                          `alignment.word` ([symbol, start, duration, 1.0])

Sample counts come from the FLAC STREAMINFO block, so this stage needs no audio
library. A conversation with a missing meta, transcript or audio file is skipped
and reported (the 2026-09-11 release has one: `conv_45e5bf954591`).
"""

from __future__ import annotations

import json
from pathlib import Path

from ...core.manifests import write_jsonl
from ...core.segments import resegment_on_words

SIDES = ("a", "b")          # speaker_a -> channel 0, speaker_b -> channel 1


def flac_info(path: Path) -> tuple[int, int, int]:
    """(sample_rate, channels, total_samples) from a FLAC's STREAMINFO block."""
    with open(path, "rb") as fh:
        if fh.read(4) != b"fLaC":
            raise ValueError(f"{path}: not a FLAC file")
        header = fh.read(4)
        if header[0] & 0x7F != 0:
            raise ValueError(f"{path}: first metadata block is not STREAMINFO")
        info = fh.read(34)
    # bytes 10..17: 20 bits rate | 3 bits channels-1 | 5 bits bps-1 | 36 bits samples
    packed = int.from_bytes(info[10:18], "big")
    rate = packed >> 44
    channels = ((packed >> 41) & 0x7) + 1
    total = packed & ((1 << 36) - 1)
    return rate, channels, total


def utterances(words, duration, pause_split, max_dur, min_gap, pad):
    """ASR words -> [(start, end, text, word_alignment)] in recording time.

    A pause of `pause_split` or more ends an utterance; over-long ones are cut
    at their widest pauses by `resegment_on_words`. Each span is then padded by
    `pad`, but never past the midpoint of the gap to its neighbour, so the
    intervals of one tier cannot overlap.
    """
    items = []
    for w in words:
        if w.get("type", "word") != "word":
            continue
        text = str(w.get("word", "")).strip()
        if not text:
            continue
        s, e = float(w["start"]), float(w["end"])
        items.append((s, max(e, s), text))
    items.sort(key=lambda t: t[0])
    if not items:
        return []

    # ASR word times overlap occasionally, so a pause is measured from the
    # furthest end seen so far, not from the previous word's end.
    groups, cur, reach = [], [items[0]], items[0][1]
    for item in items[1:]:
        if item[0] - reach >= pause_split:
            groups.append(cur)
            cur = [item]
        else:
            cur.append(item)
        reach = max(reach, item[1])
    groups.append(cur)

    raw = []
    for g in groups:
        start, end = g[0][0], max(x[1] for x in g)
        lhotse_words = [[t, s, e - s] for s, e, t in g]
        for s, e, sub in resegment_on_words(start, end - start, lhotse_words,
                                            max_dur=max_dur, min_gap=min_gap):
            inside = [x for x in g if s <= (x[0] + x[1]) / 2 <= e] or g
            text = sub if sub is not None else " ".join(x[2] for x in inside)
            raw.append((s, e, text, inside))

    out = []
    for i, (s, e, text, inside) in enumerate(raw):
        lo = max(0.0, s - pad)
        hi = min(duration, e + pad)
        if i > 0:
            lo = max(lo, (raw[i - 1][1] + s) / 2)
        if i + 1 < len(raw):
            hi = min(hi, (e + raw[i + 1][0]) / 2)
        if hi - lo <= 0.02:
            continue
        align = [[t, round(ws, 3), round(we - ws, 3), 1.0] for ws, we, t in inside]
        out.append((round(lo, 3), round(hi, 3), text, align))
    return out


def build_manifests(source, split=None, out_dir=None, limit=0, report=None) -> dict:
    root = source.root
    out = Path(out_dir) if out_dir else source.manifest_dir
    seg = source.segment_cfg

    only = source.config.get("conversations")
    if only:                        # pilot subset: skip scanning the whole release
        convs = [root / c for c in sorted(only) if (root / c).is_dir()]
    else:
        convs = sorted(d for d in root.iterdir() if d.is_dir())
    recs, recs_restored, sups = [], [], []
    skipped: dict[str, str] = {}
    n_words = 0

    for conv in convs:
        if limit and len(recs) >= limit:
            break
        cid = conv.name
        need = ["meta.json"] + [f"speaker_{x}{suf}" for x in SIDES
                                for suf in (".flac", "_transcript.json")]
        missing = [n for n in need if not (conv / n).exists()]
        if missing:
            skipped[cid] = f"missing {', '.join(missing)}"
            continue

        meta = json.loads((conv / "meta.json").read_text())
        infos = [flac_info(conv / f"speaker_{x}.flac") for x in SIDES]
        rates = {i[0] for i in infos}
        if len(rates) != 1 or any(i[1] != 1 for i in infos):
            skipped[cid] = f"channel mismatch {infos}"
            continue
        sr = rates.pop()
        n = max(i[2] for i in infos)
        duration = n / sr

        base = {"id": cid, "sampling_rate": sr, "num_samples": n,
                "duration": duration, "channel_ids": [0, 1]}
        recs.append({**base, "sources": [
            {"type": "file", "channels": [ch], "source": str(conv / f"speaker_{x}.flac")}
            for ch, x in enumerate(SIDES)]})
        out_sr = source.restore_cfg.get("out_sr", 24000)
        n_out = int(round(n * out_sr / sr))
        recs_restored.append({**base, "sampling_rate": out_sr, "num_samples": n_out,
                              "duration": n_out / out_sr, "sources": [
            {"type": "file", "channels": [ch],
             "source": str(source.restored_path(cid, x))}
            for ch, x in enumerate(SIDES)]})

        for ch, x in enumerate(SIDES):
            tr = json.loads((conv / f"speaker_{x}_transcript.json").read_text())
            words = tr.get("words") or []
            n_words += len(words)
            speaker = meta.get(f"speaker_{x}_id") or f"{cid}_{x}"
            utts = utterances(words, duration, seg["pause_split"], seg["max_dur"],
                              seg["min_gap"], seg["pad"])
            for k, (s, e, text, align) in enumerate(utts):
                sup = {"id": f"{cid}-{x}-{k:05d}", "recording_id": cid,
                       "start": s, "duration": round(e - s, 3), "channel": ch,
                       "text": text, "language": "English", "speaker": speaker}
                if source.keep_word_timing:
                    sup["alignment"] = {"word": align}
                sups.append(sup)

    paths = {
        "recordings": out / source.manifest_names["recordings"],
        "recordings_restored": out / source.manifest_names["recordings_restored"],
        "supervisions": out / source.manifest_names["supervisions"],
    }
    write_jsonl(paths["recordings"], recs)
    write_jsonl(paths["recordings_restored"], recs_restored)
    write_jsonl(paths["supervisions"], sups)

    durs = [s["duration"] for s in sups]
    stats = {
        "conversations_seen": len(convs) if not limit else None,
        "recordings": len(recs),
        "skipped": skipped,
        "hours_conversation": round(sum(r["duration"] for r in recs) / 3600, 2),
        "asr_words": n_words,
        "supervisions": len(sups),
        "supervision_hours": round(sum(durs) / 3600, 2),
        "longest_supervision": round(max(durs), 2) if durs else 0.0,
        "segment": seg,
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    print(f"[openyap] {len(recs)} conversations ({len(skipped)} skipped) -> "
          f"{len(sups)} utterances, {stats['supervision_hours']} h of speech", flush=True)
    if report:
        Path(report).write_text(json.dumps(stats, indent=2))
    return stats
