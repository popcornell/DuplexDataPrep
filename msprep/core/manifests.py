"""Reading and writing Lhotse-style JSONL manifests."""

from __future__ import annotations

import gzip
import json
from pathlib import Path


def load_jsonl(path):
    """Yield dicts from a .jsonl or .jsonl.gz file."""
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, items):
    """Write dicts to a .jsonl or .jsonl.gz file. Returns the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    n = 0
    with opener(str(path), "wt", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")
            n += 1
    return n


def channels_of(recording: dict) -> dict[int, str]:
    """recording manifest entry -> {channel index: audio path}."""
    chans = {}
    for src in recording["sources"]:
        for ch in src["channels"]:
            chans[ch] = src["source"]
    return chans
