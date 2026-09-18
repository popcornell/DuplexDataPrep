"""Fold an aligner's output TextGrids back into Lhotse supervision manifests.

MFA writes one TextGrid per corpus file with `<speaker> - words` and
`<speaker> - phones` tiers in absolute recording time. Each word (and optionally
each phone) is attached to the original supervision whose span contains the
word's midpoint, so the output manifest is the input manifest plus `alignment`.

Corpus-agnostic: the adapter supplies the source manifest and the stem -> key
rule, which is the only thing that ever differed between corpora here.
"""

from __future__ import annotations

import gzip
import json
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from .manifests import load_jsonl
from .textgrid import read_tiers


def assign(items, spans):
    """Bucket (label, start, end) items into supervision spans by midpoint.

    `spans` is a sorted list of (start, end, sup_index). Returns
    {sup_index: [[label, start, duration], ...]} plus the count of unassigned.
    """
    starts = [s for s, _, _ in spans]
    buckets = defaultdict(list)
    unassigned = 0
    for label, st, en in items:
        mid = (st + en) / 2.0
        i = bisect_right(starts, mid) - 1
        placed = False
        # spans can be adjacent/nested; check the candidate and its neighbours
        for j in (i, i - 1, i + 1):
            if 0 <= j < len(spans):
                s, e, idx = spans[j]
                if s - 1e-6 <= mid <= e + 1e-6:
                    buckets[idx].append([label, round(st, 3), round(en - st, 3)])
                    placed = True
                    break
        if not placed:
            unassigned += 1
    return buckets, unassigned


def collect(source, aligned_dir, output, phones=False, report=None, log=print) -> dict:
    """Write `<source supervisions> + alignment` to `output`. Returns stats."""
    aligned, out_path = Path(aligned_dir), Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grids = sorted(aligned.rglob("*.TextGrid"))
    log(f"[{source.name}] {len(grids)} aligned TextGrids under {aligned}", flush=True)
    if not grids:
        raise FileNotFoundError(f"no aligned TextGrids under {aligned}")

    # ---- supervisions grouped by (recording, channel), keeping input order
    sups = list(load_jsonl(source.supervisions))
    by_key: dict[tuple, list] = defaultdict(list)
    for i, s in enumerate(sups):
        by_key[(s["recording_id"], s["channel"])].append(
            (s["start"], s["start"] + s["duration"], i))
    for v in by_key.values():
        v.sort()
    log(f"[{source.name}] {len(sups)} supervisions in the source manifest", flush=True)

    word_align: dict[int, list] = {}
    phone_align: dict[int, list] = {}
    stats = {"grids": len(grids), "grids_no_match": 0, "words": 0,
             "words_unassigned": 0, "unk_words": 0, "phones": 0}
    unmatched_files: list[str] = []

    for n, g in enumerate(grids, 1):
        key = source.stem_key(g.stem)
        spans = by_key.get(key)
        if not spans:
            stats["grids_no_match"] += 1
            unmatched_files.append(g.name)
            continue
        tiers = read_tiers(g)
        stats["words"] += len(tiers["words"])
        stats["unk_words"] += sum(1 for w, _, _ in tiers["words"] if w == "<unk>")
        w_buckets, w_un = assign(tiers["words"], spans)
        stats["words_unassigned"] += w_un
        word_align.update(w_buckets)
        if phones:
            stats["phones"] += len(tiers["phones"])
            p_buckets, _ = assign(tiers["phones"], spans)
            phone_align.update(p_buckets)
        if n % 500 == 0 or n == len(grids):
            log(f"[{source.name}] parsed {n}/{len(grids)}", flush=True)

    # ---- write out: source supervision + alignment
    n_written = n_with_words = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        for i, s in enumerate(sups):
            words = word_align.get(i)
            if words is None and i not in phone_align:
                # This side was not aligned (shard not run, or MFA skipped it).
                continue
            s = dict(s)
            alignment = {}
            if words:
                alignment["word"] = words
                n_with_words += 1
            if phones and phone_align.get(i):
                alignment["phone"] = phone_align[i]
            s["alignment"] = alignment
            fh.write(json.dumps(s) + "\n")
            n_written += 1

    covered_keys = {source.stem_key(g.stem) for g in grids} & set(by_key)
    n_expected = sum(len(by_key[k]) for k in covered_keys)
    stats.update({
        "supervisions_total": len(sups),
        "supervisions_in_aligned_files": n_expected,
        "supervisions_written": n_written,
        "supervisions_with_words": n_with_words,
        "word_coverage_pct": round(100 * n_with_words / max(n_expected, 1), 3),
        "oov_pct": round(100 * stats["unk_words"] / max(stats["words"], 1), 3),
        "unmatched_files": unmatched_files[:20],
        "output": str(out_path),
    })
    if report:
        Path(report).write_text(json.dumps(stats, indent=2))
    return stats
