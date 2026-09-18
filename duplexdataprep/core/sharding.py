"""Round-robin sharding.

Sharding is keyed on the *recording* ordinal, not the channel, so every channel
of one conversation lands in the same shard -- MFA's speaker adaptation and any
per-conversation debugging both want the sides together. Round-robin (rather
than contiguous blocks) makes any single shard a representative sample, so
`--array=0-2` is a valid smaller test of a 60-shard corpus.
"""

from __future__ import annotations


def shard_of(index: int, n_shards: int) -> int:
    return index % n_shards


def shard_name(i: int) -> str:
    return f"shard{i:03d}"
