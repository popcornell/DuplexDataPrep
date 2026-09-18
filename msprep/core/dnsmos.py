"""DNSMOS P.835 per speaker channel, over its most speech-dense windows.

Ported from `MFA/scripts/dnsmos_sessions.py`. Scored per *channel*, not per
utterance: DNSMOS's input is 9.01 s, meaningless on a backchannel, and the
failure being hunted -- one badly recorded side of a conversation -- is a
property of the channel. Windows are placed where the channel's own turns are,
because scoring silence measures the noise floor, not the speech.

Protocol cost is not linear in window length: DNSMOS scores 9.01 s segments at
a 1 s hop, so a 10 s window is 1 model pass and a 20 s window is 11. Scores
from different protocols (`windows` x `win_sec`) are not comparable; keep one
per corpus.

Output is one JSON per shard, `{file_id: {OVRL, SIG, BAK, P808_MOS, *_min,
n_windows, windows, ...}}`, checkpointed as it goes and resumable.

Needs espnet2's `DNSMOS_local` + onnxruntime (the urgent_baselines env on Delta).
"""

from __future__ import annotations

import json
from pathlib import Path

from .sharding import shard_of

CHECKPOINT_EVERY = 10


def pick_windows(speech, total_dur, win, k):
    """Up to k non-overlapping windows, ranked by how much speech they hold."""
    if total_dur < win:
        return [(0.0, total_dur)] if total_dur > 1.0 else []
    cands, t = [], 0.0
    while t + win <= total_dur:
        lo, hi = t, t + win
        covered = sum(min(hi, e) - max(lo, s) for s, e in speech if e > lo and s < hi)
        cands.append((covered, t))
        t += win / 2
    cands.sort(reverse=True)
    chosen = []
    for covered, t in cands:
        if covered < win * 0.25:          # a quarter of the window must be speech
            continue
        if any(abs(t - c) < win for c in chosen):
            continue
        chosen.append(t)
        if len(chosen) >= k:
            break
    return [(t, t + win) for t in sorted(chosen)]


def score(source, out, shard=0, n_shards=1, limit=0, audio=None, device="cuda",
          resume=True, log=print) -> dict:
    import numpy as np
    import soundfile as sf
    from espnet2.enh.layers.dnsmos import DNSMOS_local

    cfg = source.config.get("dnsmos", {})
    model_dir = Path(cfg["model_dir"])
    windows, win_sec = int(cfg.get("windows", 12)), float(cfg.get("win_sec", 10.0))
    model = DNSMOS_local(str(model_dir / "sig_bak_ovr.onnx"),
                         str(model_dir / "model_v8.onnx"),
                         use_gpu=("cuda" in device))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if resume and out.exists() else {}
    log(f"[dnsmos] {source.name} shard {shard}/{n_shards} {windows}x{win_sec:g}s "
        f"({len(results)} already scored)", flush=True)

    def flush():
        tmp = out.with_name(out.name + ".part")
        tmp.write_text(json.dumps(results, indent=1))
        tmp.replace(out)

    kw = {"audio": audio} if audio else {}
    n_new = 0
    for ch in source.iter_channels(limit=limit, **kw):
        if shard_of(ch.group_index, n_shards) != shard or ch.file_id in results:
            continue
        path = Path(ch.audio_path)
        if not path.exists():
            results[ch.file_id] = {"error": "missing_audio", "path": str(path)}
            continue
        info = sf.info(str(path))
        dur = info.frames / info.samplerate
        speech = sorted((s, e) for s, e, _ in ch.turns)
        wins = pick_windows(speech, dur, win_sec, windows)
        if not wins:
            results[ch.file_id] = {"error": "no_window", "duration": round(dur, 1)}
            continue

        scores = []
        for lo, hi in wins:
            try:
                x, fs = sf.read(str(path), start=int(lo * info.samplerate),
                                stop=int(hi * info.samplerate), dtype="float32")
            except sf.LibsndfileError as exc:   # damaged seektable: drop the window
                log(f"  {ch.file_id}: window {lo:.0f}s unreadable ({exc})", flush=True)
                continue
            if x.ndim > 1:
                x = x[:, 0]
            s = model(x, fs)
            scores.append({k: float(v) for k, v in s.items()
                           if isinstance(v, (int, float, np.floating))})
        if not scores:
            results[ch.file_id] = {"error": "unreadable", "duration": round(dur, 1)}
            continue

        agg = {}
        for k in scores[0]:
            vals = [s[k] for s in scores]
            agg[k] = round(float(np.mean(vals)), 3)
            agg[k + "_min"] = round(float(np.min(vals)), 3)
        agg.update(n_windows=len(scores), duration=round(dur, 1),
                   speech_seconds=round(sum(e - s for s, e in speech), 1),
                   audio=str(path),
                   windows=[{"start": round(lo, 1), **s} for (lo, _), s in zip(wins, scores)])
        results[ch.file_id] = agg
        n_new += 1
        log(f"  {ch.file_id} OVRL={agg.get('OVRL', float('nan')):.2f} "
            f"SIG={agg.get('SIG', float('nan')):.2f} "
            f"BAK={agg.get('BAK', float('nan')):.2f}", flush=True)
        if n_new % CHECKPOINT_EVERY == 0:
            flush()
    flush()

    ok = sorted(v["OVRL"] for v in results.values() if "OVRL" in v)
    stats = {"shard": shard, "scored": len(ok), "new": n_new,
             "errors": sum("error" in v for v in results.values()), "output": str(out)}
    if ok:
        stats["OVRL"] = {"min": ok[0], "p25": ok[len(ok) // 4],
                         "median": ok[len(ok) // 2], "max": ok[-1]}
    return stats
