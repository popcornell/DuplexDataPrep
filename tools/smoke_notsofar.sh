#!/usr/bin/env bash
# Smoke test for the NOTSOFAR adapter against a synthetic release tree.
#
#   tools/smoke_notsofar.sh [WORKDIR]
#
# There is no legacy script to diff against -- NOTSOFAR was never in the MFA
# repo -- so this checks the contract instead: manifests build from the raw
# layout, prep turns them into one TextGrid per close-talk channel, every
# angle-bracket marker shape is stripped while `<PName>` content survives,
# aliases are scoped to their session so reused cast names stay distinct
# speakers, word times are carried into the supervisions, marker-only turns are
# dropped, over-long turns are counted, and far-field devices are ignored.
set -euo pipefail

WORK=${1:-/tmp/duplexdataprep-notsofar-smoke}
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rm -rf "$WORK"; mkdir -p "$WORK"

python3 -c "import yaml" 2>/dev/null || {
    echo "PyYAML missing -- run inside the mfa env" >&2; exit 2; }

python3 "$PKG_ROOT/tools/make_notsofar_fixture.py" "$WORK/data" \
    --sessions 2 --speakers 3

cat > "$WORK/notsofar.yaml" <<YAML
name: notsofar
root: $WORK/data
split: train_set
version: auto
mic: near
manifests:
  dir: $WORK/manifests
  recordings: "notsofar_recordings_{split}.jsonl.gz"
  supervisions: "notsofar_supervisions_{split}.jsonl.gz"
prep:
  n_shards: 2
audio:
  mode: in_place
speaker_scope: session
keep_word_timing: true
text:
  strip_all_markers: true
  strip_markers: []
warn_turn_over: 20.0
YAML

export DUPLEXDATAPREP_NOTSOFAR_CONFIG="$WORK/notsofar.yaml"
export PYTHONPATH="$PKG_ROOT"

echo "== manifest"
python3 -m duplexdataprep manifest notsofar --report "$WORK/manifest.json" >"$WORK/manifest.stdout"
tail -1 "$WORK/manifest.stdout" >/dev/null; grep -E '^\[' "$WORK/manifest.stdout"

echo "== prep"
python3 -m duplexdataprep prep notsofar --corpus-dir "$WORK/corpus" \
    --report "$WORK/prep.json" >"$WORK/prep.stdout"
grep -E '^\[' "$WORK/prep.stdout"

echo "== assertions"
python3 - "$WORK" <<'PY'
import json, sys
from pathlib import Path
work = Path(sys.argv[1])
man = json.load(open(work / "manifest.json"))
prep = json.load(open(work / "prep.json"))

def check(label, got, want):
    assert got == want, f"{label}: got {got!r}, want {want!r}"
    print(f"  ok  {label} == {want!r}")

check("close-talk channels", man["channels"], 6)          # 2 sessions x 3 speakers
check("segments", man["segments"], 42)                    # 6 channels x 7 turns
check("segments with no headset file", man["segments_unmatched_ct"], 2)
# The fixture reuses spk0-2 in both sessions, as the real release reuses its
# cast names. Unscoped that is 3 "speakers" for 6 distinct people.
check("speaker aliases in the release", man["speaker_aliases"], 3)
check("speakers after session scoping", man["speakers"], 6)
check("word times carried through", man["segments_with_word_timing"], 30)
check("corpus files", prep["files"], 6)
check("marker-only turns dropped", prep["dropped_non_lexical"], 12)
check("over-long turns counted", prep["turns_over_warn_length"], 6)
check("utterances written", prep["utterances"], 30)       # 42 - 12 dropped

grids = sorted((work / "corpus").rglob("*.TextGrid"))
check("TextGrids on disk", len(grids), 6)
wavs = sorted((work / "corpus").rglob("*.wav"))
check("audio symlinks", len(wavs), 6)
assert all(w.is_symlink() for w in wavs), "audio must be symlinked, not copied"
assert not any("ch0" in w.name for w in wavs), "far-field device leaked into the corpus"
print("  ok  far-field devices ignored")

text = grids[0].read_text()
for marker in ("<ST/>", "<FILL/>", "<UNKNOWN/>", "<BA/>", "<ISSUE/>", "<PName>"):
    assert marker not in text, f"marker {marker} survived normalisation"
print("  ok  every angle-bracket marker shape stripped")
assert 'text = "hello everyone thanks for joining"' in text
print("  ok  turn text intact after stripping")
assert 'text = "thanks Bert"' in text, "<PName> content must survive its tags"
print("  ok  <PName> tags dropped, the name kept")
assert '[laughter]' in text
print("  ok  bracketed events left for MFA to map to spn")

# The tier name is the speaker id, so an unscoped alias would put two different
# people on identically-named tiers in different files.
tiers = sorted({ln.split('=', 1)[1].strip().strip('"')
                for g in grids for ln in g.read_text().splitlines()
                if ln.strip().startswith("name =")})
check("distinct tier names", len(tiers), 6)
assert all("__" in t for t in tiers), f"tier names not session-scoped: {tiers}"
print(f"  ok  tier names session-scoped, e.g. {tiers[0]}")

import gzip
sups = [json.loads(ln) for ln in gzip.open(
    next((work / "manifests").glob("*supervisions*.jsonl.gz")), "rt") if ln.strip()]
aligned = [s for s in sups if s.get("alignment", {}).get("word")]
check("supervisions carrying word times", len(aligned), 30)
assert all(not w[0].startswith("<") for s in aligned
           for w in s["alignment"]["word"]), "marker tokens leaked into alignment"
print("  ok  marker tokens excluded from word alignment")

stem = grids[0].stem
assert stem.count("__") == 2, f"file stem {stem} is not split__session__ct_stem"
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
print(f"  ok  file stem shape: {stem}")
PY
echo "SMOKE OK"
