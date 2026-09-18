#!/usr/bin/env bash
# Smoke test: `msprep prep <corpus>` must reproduce the legacy scripts/prep_<corpus>.py
# byte-for-byte on the same subset.
#
#   tools/parity_prep.sh <fisher|candor> [N_RECORDINGS] [N_SHARDS] [WORKDIR]
#
# Needs PyYAML -> run inside the `mfa` conda env.
set -euo pipefail

CORPUS=${1:?usage: parity_prep.sh <fisher|candor> [N] [SHARDS] [WORKDIR]}
N=${2:-20}
SHARDS=${3:-4}
WORK=${4:-/tmp/msprep-parity-$CORPUS}
MFA_ROOT=${MFA_ROOT:-/scratch/bbjs/cornell2/MFA}
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 -c "import yaml" 2>/dev/null || {
    echo "PyYAML missing -- run inside the mfa env:" >&2
    echo "  source /scratch/bbjs/cornell2/miniconda3/etc/profile.d/conda.sh && conda activate mfa" >&2
    exit 2
}

OLD="$WORK/old"; NEW="$WORK/new"
rm -rf "$OLD" "$NEW"; mkdir -p "$OLD" "$NEW"

echo "== legacy: scripts/prep_$CORPUS.py"
case "$CORPUS" in
    candor) extra=(--audio-dir "$MFA_ROOT/audio/candor") ;;
    *)      extra=() ;;
esac
python3 "$MFA_ROOT/scripts/prep_$CORPUS.py" \
    --corpus-dir "$OLD" --n-shards "$SHARDS" --limit "$N" \
    --report "$WORK/old.json" "${extra[@]}" >"$WORK/old.stdout"
tail -2 "$WORK/old.stdout"

echo "== package: msprep prep $CORPUS"
PYTHONPATH="$PKG_ROOT" python3 -m msprep prep "$CORPUS" \
    --corpus-dir "$NEW" --n-shards "$SHARDS" --limit "$N" \
    --report "$WORK/new.json" >"$WORK/new.stdout"
grep -E '^\[' "$WORK/new.stdout" | tail -2

echo "== diff (content + symlink targets)"
if diff -r --no-dereference "$OLD" "$NEW"; then
    echo "corpus trees identical ($(find "$NEW" -name '*.TextGrid' | wc -l) TextGrids)"
else
    echo "PARITY FAILED" >&2
    exit 1
fi

echo "== reports"
python3 - "$WORK/old.json" "$WORK/new.json" <<'PY'
import json, sys
old, new = (json.load(open(p)) for p in sys.argv[1:3])
shared = sorted(set(old) & set(new))
bad = {k: (old[k], new[k]) for k in shared if old[k] != new[k]}
print("compared:", ", ".join(shared))
print("only in legacy:", sorted(set(old) - set(new)) or "-")
print("only in package:", sorted(set(new) - set(old)) or "-")
if bad or set(old) != set(new):
    print("MISMATCH:", bad)
    raise SystemExit(1)
print("report values match")
PY
echo "PARITY OK"
