"""Corpus construction: ChannelSource -> sharded MFA corpus on disk.

Corpus-agnostic. Every decision that differs between corpora (which audio, how
text is normalized, how turns are grouped) has already been made by the adapter
by the time a Channel arrives here.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from .sharding import shard_name, shard_of
from .textgrid import write_textgrid


def build_corpus(source, corpus_dir, n_shards=60, limit=0, report=None,
                 log=print) -> dict:
    """Write `<corpus_dir>/shardNNN/{<file_id>.<ext>, <file_id>.TextGrid}`.

    Audio is symlinked, never copied -- MFA resamples internally and the source
    corpora are large enough that a second copy is not worth the disk.
    """
    corpus = Path(corpus_dir).resolve()
    corpus.mkdir(parents=True, exist_ok=True)

    n_files = n_intervals = 0
    missing_audio: list[str] = []
    per_shard: dict[str, int] = defaultdict(int)

    for ch in source.iter_channels(limit=limit):
        sd = corpus / shard_name(shard_of(ch.group_index, n_shards))
        sd.mkdir(exist_ok=True)

        audio = Path(ch.audio_path)
        if not audio.exists():
            missing_audio.append(str(audio))
            continue

        link = sd / f"{ch.file_id}{audio.suffix}"
        if not link.is_symlink() and not link.exists():
            link.symlink_to(audio)

        n_intervals += write_textgrid(
            sd / f"{ch.file_id}.TextGrid", ch.duration, {ch.speaker: ch.turns}
        )
        n_files += 1
        per_shard[sd.name] += 1

    log(f"[{source.name}] wrote {n_files} files / {n_intervals} utterances "
        f"across {len(per_shard)} shards", flush=True)
    if missing_audio:
        print(f"[{source.name}] WARNING: {len(missing_audio)} missing audio files, "
              f"e.g. {missing_audio[:3]}", file=sys.stderr, flush=True)

    stats = dict(source.stats)
    stats.update({
        "files": n_files,
        "utterances": n_intervals,
        "shards": sorted(per_shard),
        "missing_audio": missing_audio[:50],
        "n_missing_audio": len(missing_audio),
    })
    if report:
        Path(report).write_text(json.dumps(stats, indent=2))
    return stats
