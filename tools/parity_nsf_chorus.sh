#!/usr/bin/env bash
# Parity check for the nsf_chorus manifest stage.
#
#   tools/parity_nsf_chorus.sh [WORKDIR] [REFERENCE_DIR]
#
# The port replaced lhotse object construction with plain JSON writing, so the
# thing to prove is that it produces the *same manifests* as the script it came
# from (HEAT-TTS/scripts/build_notsofar_manifests.py). This rebuilds them into a
# scratch directory and compares the uncompressed bytes against the reference
# copies that script already wrote.
set -euo pipefail

WORK=${1:-/tmp/msprep-nsf-parity}
REF=${2:-/scratch/bbjs/cornell2/MFA/out/manifests}
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rm -rf "$WORK"; mkdir -p "$WORK"

python3 -c "import yaml" 2>/dev/null || {
    echo "PyYAML missing -- run inside the mfa env" >&2; exit 2; }

export PYTHONPATH="$PKG_ROOT"
echo "== rebuilding into $WORK"
python3 -m msprep manifest nsf_chorus --out-dir "$WORK" \
    --report "$WORK/report.json" >"$WORK/stdout"
grep -E '^\[' "$WORK/stdout"

echo "== comparing against $REF"
status=0
for f in notsofar_recordings_24k.jsonl.gz \
         notsofar_supervisions_dev.jsonl.gz \
         notsofar_supervisions_train.jsonl.gz \
         notsofar_supervisions_eval.jsonl.gz \
         notsofar_supervisions.jsonl.gz; do
    if [ ! -f "$REF/$f" ]; then
        echo "  -- $f: no reference, skipped"; continue
    fi
    # Compare decompressed bytes: gzip headers carry mtimes and would differ.
    if cmp -s <(gzip -cd "$WORK/$f") <(gzip -cd "$REF/$f"); then
        echo "  ok  $f identical"
    else
        echo "  !!  $f DIFFERS"
        diff <(gzip -cd "$REF/$f") <(gzip -cd "$WORK/$f") | head -4 || true
        status=1
    fi
done

# The sidecar is plain JSON written with the same indent/sort, so compare directly.
if [ -f "$REF/notsofar_meta.json" ]; then
    if cmp -s "$WORK/notsofar_meta.json" "$REF/notsofar_meta.json"; then
        echo "  ok  notsofar_meta.json identical"
    else
        echo "  !!  notsofar_meta.json DIFFERS"; status=1
    fi
fi

[ $status -eq 0 ] && echo "PARITY OK" || echo "PARITY FAILED"
exit $status
