"""Unpack the otoSpeech WebDataset shards into per-speaker mono FLACs.

Each shard is a tar of `<sid>.flac` (stereo, one speaker per channel) and
`<sid>.json` (session metadata, incl. `speakers[].channel`). One ffmpeg decode
per session writes both sides at the native rate -- Sidon resamples on its
own, so nothing is lost before restoration -- and the JSON is kept next to
them for `duplexdataprep manifest otospeech`.

    duplexdataprep extract otospeech --shard 0 --n-shards 8 --jobs 8

Sharding is over tar files, so a task never reads another task's shard.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...core.sharding import shard_of


def _ffmpeg_cmd(tmp0, tmp1, audio):
    comp = str(audio.get("compression_level", 5))
    side = lambda tag, out: [  # noqa: E731
        "-map", tag, "-ac", "1", "-c:a", "flac", "-compression_level", comp,
        "-f", "flac", str(out),
    ]
    return [
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", "pipe:0",
        "-filter_complex", "[0:a]channelsplit=channel_layout=stereo[L][R]",
        # -f flac is explicit because the temp files carry a .part extension
        *side("[L]", tmp0), *side("[R]", tmp1),
    ]


def split_one(source, sid: str, flac: bytes, meta: bytes, audio: dict,
              overwrite: bool) -> dict:
    out0, out1 = source.raw_path(sid, 0), source.raw_path(sid, 1)
    meta_out = source.meta_path(sid)
    if not overwrite and out0.exists() and out1.exists() and meta_out.exists():
        return {"id": sid, "status": "cached"}

    tmp0, tmp1 = out0.with_suffix(".flac.part"), out1.with_suffix(".flac.part")
    proc = subprocess.run(_ffmpeg_cmd(tmp0, tmp1, audio), input=flac,
                          capture_output=True)
    if proc.returncode != 0:
        for t in (tmp0, tmp1):
            t.unlink(missing_ok=True)
        return {"id": sid, "status": "ffmpeg_error",
                "stderr": proc.stderr.decode(errors="replace")[-500:]}
    tmp0.replace(out0)
    tmp1.replace(out1)
    meta_out.write_bytes(meta)
    return {"id": sid, "status": "ok"}


def iter_sessions(tar_path: Path):
    """Yield (sid, flac bytes, json bytes) from one WebDataset shard.

    WebDataset groups a sample's files by key (the name up to the first dot);
    both members of a sample are adjacent in the tar.
    """
    pending: dict[str, dict] = {}
    with tarfile.open(tar_path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name).name
            sid, _, ext = name.partition(".")
            if ext not in ("flac", "json"):
                continue
            pending.setdefault(sid, {})[ext] = tar.extractfile(member).read()
            if len(pending[sid]) == 2:
                got = pending.pop(sid)
                yield sid, got["flac"], got["json"]
    for sid, got in pending.items():
        yield sid, got.get("flac"), got.get("json")


def extract(source, shard=0, n_shards=1, jobs=8, limit=0, overwrite=False,
            report=None, log=print) -> dict:
    """Unpack one shard of the tar list. Returns stats."""
    audio = source.config.get("audio", {})
    source.raw_dir.mkdir(parents=True, exist_ok=True)

    tars = sorted(source.shards_dir.glob("*.tar"))
    if not tars:
        raise FileNotFoundError(f"no *.tar under {source.shards_dir}")
    mine = [t for i, t in enumerate(tars) if shard_of(i, n_shards) == shard]
    log(f"[otospeech] extract shard {shard}/{n_shards}: {len(mine)}/{len(tars)} tars",
        flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = []
        for tar_path in mine:
            for sid, flac, meta in iter_sessions(tar_path):
                if limit and len(futures) >= limit:
                    break
                if flac is None or meta is None:
                    results.append({"id": sid, "status": "incomplete_sample",
                                    "tar": tar_path.name})
                    continue
                futures.append(pool.submit(split_one, source, sid, flac, meta,
                                           audio, overwrite))
        for n, fut in enumerate(futures, 1):
            results.append(fut.result())
            if n % 25 == 0 or n == len(futures):
                log(f"[otospeech] extract {n}/{len(futures)}", flush=True)

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    bad = [r for r in results if r["status"] not in ("ok", "cached")]
    log(f"[otospeech] extract {counts}", flush=True)
    for r in bad[:5]:
        print(f"[otospeech] FAILED {r}", file=sys.stderr, flush=True)

    stats = {"shard": shard, "n_shards": n_shards, "tars": [t.name for t in mine],
             "counts": counts, "failures": bad[:50], "raw_dir": str(source.raw_dir)}
    if report:
        Path(report).write_text(json.dumps(stats, indent=2))
    return stats
