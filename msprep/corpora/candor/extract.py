"""Split every CANDOR stereo mp3 into two 16 kHz mono FLACs, one per speaker.

One ffmpeg decode per conversation produces both sides. Output goes to
`<audio.dir>/<cid>-ch0.flac` and `-ch1.flac`, matching the channel indices in
the recordings manifest (channel 0 == channel_map.json "L"). Decoding is
sample-exact against the manifest durations (verified: 0.000 s drift).

    msprep extract candor --shard 0 --n-shards 8 --jobs 8
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...core.manifests import load_jsonl


def _ffmpeg_cmd(src, tmp0, tmp1, audio):
    sr = str(audio.get("sample_rate", 16000))
    comp = str(audio.get("compression_level", 5))
    fmt = audio.get("sample_fmt", "s16")
    side = lambda tag, out: [  # noqa: E731
        "-map", tag, "-ar", sr, "-ac", "1", "-c:a", "flac",
        "-compression_level", comp, "-sample_fmt", fmt, "-f", "flac", str(out),
    ]
    return [
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
        "-filter_complex", "[0:a]channelsplit=channel_layout=stereo[L][R]",
        # -f flac is explicit because the temp files carry a .part extension
        *side("[L]", tmp0), *side("[R]", tmp1),
    ]


def extract_one(rid, src, out_dir: Path, audio: dict, overwrite: bool) -> dict:
    ext = audio.get("ext", ".flac")
    out0, out1 = out_dir / f"{rid}-ch0{ext}", out_dir / f"{rid}-ch1{ext}"
    if not overwrite and out0.exists() and out1.exists():
        return {"id": rid, "status": "cached"}
    if not Path(src).exists():
        return {"id": rid, "status": "missing_source", "source": str(src)}

    tmp0, tmp1 = out0.with_suffix(f"{ext}.part"), out1.with_suffix(f"{ext}.part")
    proc = subprocess.run(_ffmpeg_cmd(src, tmp0, tmp1, audio),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        for t in (tmp0, tmp1):
            t.unlink(missing_ok=True)
        return {"id": rid, "status": "ffmpeg_error", "stderr": proc.stderr[-500:]}
    tmp0.replace(out0)
    tmp1.replace(out1)
    return {"id": rid, "status": "ok"}


def extract(source, shard=0, n_shards=1, jobs=8, limit=0, overwrite=False,
            report=None, log=print) -> dict:
    """Run the stereo split over one shard of the corpus. Returns stats."""
    audio = source.config["audio"]
    out_dir = Path(audio["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for i, r in enumerate(load_jsonl(source.recordings_path)):
        if limit and i >= limit:
            break
        if i % n_shards == shard:
            todo.append((r["id"], r["sources"][0]["source"]))
    log(f"[candor] extract shard {shard}/{n_shards}: {len(todo)} conversations",
        flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(extract_one, rid, src, out_dir, audio, overwrite)
                   for rid, src in todo]
        for n, fut in enumerate(futures, 1):
            results.append(fut.result())
            if n % 25 == 0 or n == len(futures):
                log(f"[candor] extract {n}/{len(futures)}", flush=True)

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    bad = [r for r in results if r["status"] not in ("ok", "cached")]
    log(f"[candor] extract {counts}", flush=True)
    for r in bad[:5]:
        print(f"[candor] FAILED {r}", file=sys.stderr, flush=True)

    stats = {"shard": shard, "n_shards": n_shards, "counts": counts,
             "failures": bad[:50], "audio_dir": str(out_dir)}
    if report:
        Path(report).write_text(json.dumps(stats, indent=2))
    return stats
