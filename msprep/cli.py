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

    r = sub.add_parser("restore", help="Sidon speech restoration (GPU)")
    corpus_args(r)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--n-shards", type=int, default=1)
    r.add_argument("--limit", type=int, default=0, help="only the first N recordings")
    r.add_argument("--device", default="cuda")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--report", default=None)

    d = sub.add_parser("dnsmos", help="DNSMOS P.835 per speaker channel")
    corpus_args(d)
    d.add_argument("--out", required=True, help="JSON results (resumable)")
    d.add_argument("--shard", type=int, default=0)
    d.add_argument("--n-shards", type=int, default=1)
    d.add_argument("--limit", type=int, default=0, help="only the first N recordings")
    d.add_argument("--audio", default=None,
                   help="corpus-specific audio variant, e.g. openyap `raw`|`restored`")
    d.add_argument("--device", default="cuda")
    d.add_argument("--no-resume", action="store_true")

    k = sub.add_parser("cuts", help="MFA supervisions -> FastMSS-ready CutSets (lhotse)")
    corpus_args(k)
    k.add_argument("--supervisions", nargs="+", required=True,
                   help="collected MFA supervision manifest(s), e.g. one per shard")
    k.add_argument("--recordings", required=True)
    k.add_argument("--out-dir", required=True)
    k.add_argument("--prefix", default=None, help="default: the corpus name")
    k.add_argument("--resample", type=int, default=None,
                   help="lazily resample recordings to this rate (e.g. 16000)")
    k.add_argument("--blocklist", default=None,
                   help="JSON list of channel file_ids to drop from the utterance cuts")
    k.add_argument("--report", default=None)

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
    elif a.stage == "restore":
        from .core.restore import restore
        try:
            stats = restore(source, shard=a.shard, n_shards=a.n_shards, limit=a.limit,
                            overwrite=a.overwrite, device=a.device, report=a.report)
        except NotImplementedError as exc:
            print(f"[msprep] {exc}", file=sys.stderr)
            return 2
        if stats["failures"]:
            print(json.dumps(stats, indent=2), flush=True)
            return 1
    elif a.stage == "dnsmos":
        from .core.dnsmos import score
        stats = score(source, a.out, shard=a.shard, n_shards=a.n_shards, limit=a.limit,
                      audio=a.audio, device=a.device, resume=not a.no_resume)
    elif a.stage == "cuts":
        from .core.cuts import build_cuts
        blocked = set()
        if a.blocklist:
            with open(a.blocklist) as fh:
                blocked = {source.stem_key(f) for f in json.load(fh)}
        stats = build_cuts(a.recordings, a.supervisions, a.out_dir,
                           a.prefix or a.corpus, resample=a.resample,
                           blocked=blocked, report=a.report)
    else:
        stats = source.config

    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
