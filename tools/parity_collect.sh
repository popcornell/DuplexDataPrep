#!/usr/bin/env bash
# Smoke test: `duplexdataprep collect <corpus>` must reproduce scripts/collect_alignments.py.
#
#   tools/parity_collect.sh <fisher|candor> [ALIGNED_DIR] [WORKDIR]
#
# Needs praatio + PyYAML -> run inside the `mfa` conda env.
set -euo pipefail

CORPUS=${1:?usage: parity_collect.sh <fisher|candor> [ALIGNED_DIR] [WORKDIR]}
MFA_ROOT=${MFA_ROOT:-/scratch/bbjs/cornell2/MFA}
ALIGNED=${2:-$MFA_ROOT/out/$CORPUS/shard000}
WORK=${3:-/tmp/duplexdataprep-collect-parity-$CORPUS}
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$WORK"

echo "== legacy: scripts/collect_alignments.py $CORPUS"
python3 "$MFA_ROOT/scripts/collect_alignments.py" "$CORPUS" \
    --aligned-dir "$ALIGNED" --output "$WORK/old.jsonl.gz" \
    --phones --report "$WORK/old.json" >/dev/null

echo "== package: duplexdataprep collect $CORPUS"
PYTHONPATH="$PKG_ROOT" python3 -m duplexdataprep collect "$CORPUS" \
    --aligned-dir "$ALIGNED" --output "$WORK/new.jsonl.gz" \
    --phones --report "$WORK/new.json" >/dev/null

echo "== manifests (uncompressed bytes)"
a=$(zcat "$WORK/old.jsonl.gz" | md5sum | cut -d' ' -f1)
b=$(zcat "$WORK/new.jsonl.gz" | md5sum | cut -d' ' -f1)
[ "$a" = "$b" ] || { echo "MANIFEST MISMATCH: $a != $b" >&2; exit 1; }
echo "identical ($a, $(zcat "$WORK/new.jsonl.gz" | wc -l) supervisions)"

echo "== reports (ignoring the output path)"
python3 - "$WORK/old.json" "$WORK/new.json" <<'PY'
import json, sys
old, new = (json.load(open(p)) for p in sys.argv[1:3])
for d in (old, new):
    d.pop("output", None)          # differs by construction
bad = {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
       if old.get(k) != new.get(k)}
if bad:
    print("MISMATCH:", bad)
    raise SystemExit(1)
print(f"identical ({len(old)} keys, word coverage {old['word_coverage_pct']}%)")
PY
echo "PARITY OK"
