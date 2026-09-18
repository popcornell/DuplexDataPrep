"""Command line: `python -m msprep <stage> <corpus> ...`.

Every stage takes a corpus name; paths and defaults come from that corpus's
YAML config, which `--config` can point elsewhere.
"""

from __future__ import annotations

import argparse
import json
import sys

from .core.collect import collect
from .core.prep import build_corpus
from .corpora import available, get_source


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="msprep")
    sub = ap.add_subparsers(dest="stage", required=True)

    def corpus_args(p):
        p.add_argument("corpus", choices=available())
        p.add_argument("--config", default=None,
                       help="YAML config for this corpus (default: packaged)")

    p = sub.add_parser("prep", help="build a sharded MFA corpus")
    corpus_args(p)
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--n-shards", type=int, default=None,
                   help="default: prep.n_shards from the config")
    p.add_argument("--limit", type=int, default=0,
                   help="only the first N recordings (pilot)")
    p.add_argument("--report", default=None, help="write a JSON summary here")

    c = sub.add_parser("collect", help="fold aligned TextGrids back into supervisions")
    corpus_args(c)
    c.add_argument("--aligned-dir", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--phones", action="store_true", help="also emit phone alignments")
    c.add_argument("--report", default=None)

    e = sub.add_parser("extract", help="materialise per-speaker audio (corpus-specific)")
    corpus_args(e)
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--n-shards", type=int, default=1)
    e.add_argument("--jobs", type=int, default=8)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--overwrite", action="store_true")
    e.add_argument("--report", default=None)

    m = sub.add_parser("manifest",
                       help="build Lhotse-style manifests from a raw release "
                            "(only for corpora that ship none)")
    corpus_args(m)
    m.add_argument("--split", default=None,
                   help="default: split from the config; `all` where a corpus "
                        "builds every split at once")
    m.add_argument("--out-dir", default=None)
    m.add_argument("--limit", type=int, default=0, help="only the first N sessions")
    m.add_argument("--report", default=None)

    s = sub.add_parser("config", help="print the resolved config for a corpus")
    corpus_args(s)

    a = ap.parse_args(argv)
    source = get_source(a.corpus, config=a.config)

    if a.stage == "prep":
        n_shards = a.n_shards or source.config.get("prep", {}).get("n_shards", 60)
        stats = build_corpus(source, a.corpus_dir, n_shards=n_shards,
                             limit=a.limit, report=a.report)
    elif a.stage == "collect":
        stats = collect(source, a.aligned_dir, a.output,
                        phones=a.phones, report=a.report)
    elif a.stage == "manifest":
        try:
            stats = source.build_manifests(split=a.split, out_dir=a.out_dir,
                                           limit=a.limit, report=a.report)
        except NotImplementedError as exc:
            print(f"[msprep] {exc}", file=sys.stderr)
            return 2
    elif a.stage == "extract":
        if a.corpus != "candor":
            print(f"[msprep] {a.corpus} audio needs no extraction step",
                  file=sys.stderr)
            return 2
        from .corpora.candor.extract import extract
        stats = extract(source, shard=a.shard, n_shards=a.n_shards, jobs=a.jobs,
                        limit=a.limit, overwrite=a.overwrite, report=a.report)
    else:
        stats = source.config

    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
