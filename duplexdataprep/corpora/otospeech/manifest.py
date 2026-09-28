"""otoSpeech extract output + Parakeet pseudo labels -> Lhotse-style manifests.

The labels (`seglst.json`, from Lhotse's `oto_speech` recipe) are a flat list of
ASR segments `{session_id, speaker, start_time, end_time, words,
word_alignment: [[word, start, end], ...]}` in session time. They carry the
speaker id but not the channel; the channel comes from each session's own
`speakers[].channel`. (Lhotse's recipe puts every segment on channel 0, which
for this stereo, one-speaker-per-channel release is wrong half the time.)

Utterances are rebuilt from the word times with OpenYAP's rules -- same
`segment` knobs, same padding -- rather than taken from the ASR segments,
so the two duplex corpora are segmented identically. A few pseudo-label word
times are broken (144 of 1.26M words claim > 5 s, the worst 98.9 s for
"mostly."); a word no pause can split would become one utterance of that
length, so words longer than `max_word_dur` are dropped first.

    duplexdataprep manifest otospeech               # every extracted session
    duplexdataprep manifest otospeech --limit 5     # pilot

Writes three files under `manifests.dir`:

    recordings            one recording per session, channel 0/1 pointing at the
                          extracted per-speaker FLACs (native rate)
    recordings_restored   the same, pointing at `duplexdataprep restore` output
    supervisions          one per utterance, ASR word times as `alignment.word`

A session is skipped and reported when its audio or metadata is missing, it
has no labels, or a labelled speaker is absent from its channel map.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ...core.manifests import write_jsonl
from ..openyap.manifest import flac_info, utterances


def load_labels(path: Path, max_word_dur: float) -> tuple[dict, int]:
    """seglst.json -> ({session: {speaker: [{word, start, end}, ...]}}, n_dropped).

    Words longer than `max_word_dur` are broken timestamps and are dropped.
    """
    with open(path, encoding="utf-8") as fh:
        segs = json.load(fh)
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    n_dropped = 0
    for seg in segs:
        words = seg.get("word_alignment") or [[seg["words"], seg["start_time"],
                                               seg["end_time"]]]
        dst = out[seg["session_id"]][seg["speaker"]]
        for w, s, e in words:
            if float(e) - float(s) > max_word_dur:
                n_dropped += 1
                continue
            dst.append({"word": w, "start": float(s), "end": float(e)})
    return out, n_dropped


def overlaps(s: float, e: float, spans: list[tuple[float, float]]) -> bool:
    return any(s < b and a < e for a, b in spans)


def build_manifests(source, split=None, out_dir=None, limit=0, report=None) -> dict:
    out = Path(out_dir) if out_dir else source.manifest_dir
    seg = source.segment_cfg

    labels, n_long_words = load_labels(source.labels_path, source.max_word_dur)
    metas = sorted(source.raw_dir.glob("*.json"))
    print(f"[otospeech] {len(metas)} extracted sessions, {len(labels)} labelled",
          flush=True)

    recs, recs_restored, sups = [], [], []
    skipped: dict[str, str] = {}
    n_words = n_redacted = 0
    redacted_s = 0.0

    for meta_path in metas:
        if limit and len(recs) >= limit:
            break
        sid = meta_path.stem
        paths = [source.raw_path(sid, ch) for ch in (0, 1)]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            skipped[sid] = f"missing {', '.join(missing)}"
            continue
        if sid not in labels:
            skipped[sid] = "no pseudo labels"
            continue

        meta = json.loads(meta_path.read_text())
        chan_of = {str(s["speaker_id"]): int(s["channel"])
                   for s in meta.get("speakers", [])}
        unmapped = sorted(set(labels[sid]) - set(chan_of))
        if unmapped:
            skipped[sid] = f"speakers not in channel map: {unmapped}"
            continue
        if len(set(chan_of.values())) != len(chan_of):
            skipped[sid] = f"two speakers on one channel: {chan_of}"
            continue

        infos = [flac_info(p) for p in paths]
        rates = {i[0] for i in infos}
        if len(rates) != 1 or any(i[1] != 1 for i in infos):
            skipped[sid] = f"channel mismatch {infos}"
            continue
        sr = rates.pop()
        n = max(i[2] for i in infos)
        duration = n / sr

        base = {"id": sid, "sampling_rate": sr, "num_samples": n,
                "duration": duration, "channel_ids": [0, 1]}
        recs.append({**base, "sources": [
            {"type": "file", "channels": [ch], "source": str(p)}
            for ch, p in enumerate(paths)]})
        out_sr = source.restore_cfg.get("out_sr", 24000)
        n_out = int(round(n * out_sr / sr))
        recs_restored.append({**base, "sampling_rate": out_sr, "num_samples": n_out,
                              "duration": n_out / out_sr, "sources": [
            {"type": "file", "channels": [ch],
             "source": str(source.restored_path(sid, ch))}
            for ch in (0, 1)]})

        # Redacted spans are muted in the audio; an utterance across one would
        # make MFA align words to digital silence.
        redacted = [(float(r["start_sec"]), float(r["end_sec"]))
                    for r in meta.get("redacted_segments") or []]
        redacted_s += sum(b - a for a, b in redacted)

        for speaker, words in sorted(labels[sid].items()):
            ch = chan_of[speaker]
            n_words += len(words)
            utts = utterances(words, duration, seg["pause_split"], seg["max_dur"],
                              seg["min_gap"], seg["pad"])
            for k, (s, e, text, align) in enumerate(utts):
                if source.drop_redacted and overlaps(s, e, redacted):
                    n_redacted += 1
                    continue
                sup = {"id": f"{sid}-ch{ch}-{k:05d}", "recording_id": sid,
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
        "sessions_extracted": len(metas),
        "sessions_labelled": len(labels),
        "recordings": len(recs),
        "skipped": skipped,
        "hours_session": round(sum(r["duration"] for r in recs) / 3600, 2),
        "asr_words": n_words,
        "dropped_long_words": n_long_words,
        "supervisions": len(sups),
        "supervision_hours": round(sum(durs) / 3600, 2),
        "dropped_redacted": n_redacted,
        "redacted_hours": round(redacted_s / 3600, 2),
        "longest_supervision": round(max(durs), 2) if durs else 0.0,
        "segment": seg,
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    print(f"[otospeech] {len(recs)} sessions ({len(skipped)} skipped) -> "
          f"{len(sups)} utterances, {stats['supervision_hours']} h of speech", flush=True)
    if report:
        Path(report).write_text(json.dumps(stats, indent=2))
    return stats
