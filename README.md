# msprep

Multi-speaker corpus preprocessing for forced alignment and FastMSS meeting
simulation. Extracted from the MFA repo so the same pipeline can run over any
corpus that can be expressed as *single-speaker channels* — Fisher, CANDOR,
LibriSpeech, NOTSOFAR-1.

FastMSS itself stays an unmodified external dependency. This package is the
pre/post work around it.

## Layout

```
msprep/
  core/                 corpus-agnostic stages
    manifests.py  segments.py  textgrid.py  sharding.py
    config.py     prep.py      collect.py
  corpora/              one directory per corpus
    base.py             Channel, ChannelSource, the registry
    fisher/
      __init__.py       the adapter + its normalisation
      fisher.yaml       paths and parameters
    candor/
      __init__.py       the adapter + its resegmentation wiring
      candor.yaml
      extract.py        stereo mp3 -> per-speaker 16 kHz FLAC
    notsofar/
      __init__.py       the adapter (close-talk track only)
      notsofar.yaml
      manifest.py       raw release tree -> Lhotse-style manifests
tools/                  parity_prep.sh  parity_collect.sh
                        smoke_notsofar.sh  make_notsofar_fixture.py
```

Nothing in `core/` knows a corpus name. Everything a corpus does differently is
either in its directory or in its YAML — that split is what the extraction was
for.

## The one abstraction

```python
@dataclass
class Channel:
    file_id: str      # MFA file stem, e.g. fe_03_00001-A
    audio_path: Path  # existing mono audio (symlinked into the shard)
    speaker: str      # TextGrid tier name == speaker id
    duration: float   # recording duration -> TextGrid xmax
    turns: list[tuple[float, float, str]]   # (start, end, normalized text)
    group: str        # recording id; channels of one recording share a shard
    group_index: int  # recording ordinal -> round-robin shard
```

A corpus adapter yields these and supplies three small rules: `normalize(text)`,
`supervisions` (the manifest alignments attach back to), and `stem_key(stem) ->
(recording_id, channel)`. Config arrives as `self.config`; `configure()` is the
hook that turns it into attributes.

**"Speaker channel", not "recording channel."** A far-field array channel is not
a `ChannelSource`: it carries every speaker at once and cannot be force aligned
without separation first. For NOTSOFAR-1 that means the close-talk track is the
pipeline input; the far-field mixture is an evaluation target, not a source.

## Config

Paths and parameters live in `corpora/<name>/<name>.yaml`, never in code.
Resolution order, first hit wins:

1. `--config /path/to.yaml`
2. `$MSPREP_<CORPUS>_CONFIG` — e.g. `MSPREP_CANDOR_CONFIG`
3. `$MSPREP_CONFIG_DIR/<corpus>.yaml` — a whole directory of overrides
4. the packaged default

Every string is passed through `expandvars`/`expanduser`, so a config can be
written against `$MY_DATA_ROOT` and moved between machines unedited. Keys a
config omits fall back to the adapter's defaults, so an override file only needs
the lines it changes.

```bash
msprep config candor                       # print the resolved config
msprep config candor --config mine.yaml
```

CANDOR's `variant` (`audiophile` | `cliffhanger` | `backbiter`) is substituted
into the supervisions filename, so switching turn tables is a one-line edit.

## Use

```bash
# CANDOR only: split stereo mp3 -> per-speaker 16 kHz FLAC (SLURM array friendly)
msprep extract candor --shard 0 --n-shards 8 --jobs 8

# NOTSOFAR only: raw release tree -> manifests (it ships none)
msprep manifest notsofar --split train_set

# build a sharded MFA corpus (audio symlinked, one TextGrid per speaker channel)
msprep prep fisher   --corpus-dir corpus/fisher    # n_shards from the config
msprep prep candor   --corpus-dir corpus/candor
msprep prep notsofar --corpus-dir corpus/notsofar

# ... run MFA over corpus/<ds>/shardNNN ...

# fold the aligned TextGrids back into supervisions
msprep collect fisher --aligned-dir out/fisher \
    --output out/manifests/fisher_supervisions_mfa.jsonl.gz --phones
```

Without an install, `PYTHONPATH=<this dir> python3 -m msprep ...` works the same.

`prep` and `extract` need only PyYAML; `collect` also needs `praatio`, imported
lazily so corpus construction never pays for it. On Delta both are in the `mfa`
conda env.

## Parity

The port is validated against the scripts it replaces, not just tested:

```bash
tools/parity_prep.sh    fisher 20 4     # corpus, N recordings, N shards
tools/parity_prep.sh    candor 20 4
tools/parity_collect.sh fisher          # one aligned shard
tools/parity_collect.sh candor
tools/smoke_notsofar.sh                 # fixture-based; no legacy script exists
```

`parity_prep.sh` builds the same subset with `scripts/prep_<corpus>.py` and with
`msprep`, then diffs the corpus trees with `--no-dereference` (so symlink
targets count, not just their presence) and compares the JSON reports key by
key. `parity_collect.sh` runs both collectors over an already-aligned shard and
compares uncompressed manifest bytes plus every report key except `output`.

Measured on this tree:

| check | result |
|---|---|
| prep fisher, 20 rec / 4 shards | 40 TextGrids identical, reports match |
| prep fisher, 200 rec / 8 shards | 400 TextGrids identical, reports match |
| prep candor, 20 rec / 4 shards | 40 TextGrids identical, 21,459 utterances, 155 turns resegmented into 549 |
| collect fisher, `out/fisher/shard000` (390 grids) | 36,939 supervisions, md5 `1e50c142…`, 99.593 % word coverage |
| collect candor, `out/candor/shard000` (84 grids) | 35,017 supervisions, md5 `745f0d24…`, 99.5 % word coverage |
| smoke notsofar, 2 sessions × 3 speakers (fixture) | 22 contract assertions, all pass |
| notsofar `dev_set`, real data | 201 TextGrids, 9,674 utterances, 0 markers surviving, 201 distinct tiers |

Any corpus added later should land with the same kind of check against whatever
it replaces.

## Status

| stage | state |
|---|---|
| `core.prep` — manifests → sharded MFA corpus | ported, parity-checked on Fisher + CANDOR |
| `core.collect` — TextGrids → supervisions + alignment | ported, parity-checked on Fisher + CANDOR |
| `core.config` — per-corpus YAML | new |
| `corpora.fisher` | ported |
| `corpora.candor` (+ `extract.py`) | ported |
| `corpora.notsofar` (+ `manifest.py`) | ported, run on the real `dev_set` (36 sessions / 201 channels) |
| `corpora.librispeech` | next |
| `core.gating` — oracle-segmentation gate | not written; see below |
| cut construction / filter policy / FastMSS staging | still in the MFA repo's `scripts/` |

The MFA repo's `scripts/` are still what the production pipeline runs; nothing
was switched over.

## Design notes carried over

- **TextGrid writing is deliberately hand-rolled.** praatio splits long-form
  files on `re.split(r"item ?\[")`, so a transcript containing "item [" — Fisher's
  "an item [lipsmack]" — fabricates a tier and the file silently fails to load in
  MFA. `_praatio_safe` rewrites those brackets to angle brackets, which are in
  MFA's `brackets` config too, so the token still maps to `spn`.
- **Sharding is round-robin over recordings**, so any single shard is a
  representative sample and `--array=0-2` is a valid smaller test. It keys on the
  recording ordinal, not the channel, so both sides of a conversation stay
  together.
- **One MFA "file" is one speaker's channel**, with a single-tier TextGrid, so
  long recordings are never chopped up on disk.
- **CANDOR resegmentation uses existing ASR word times only to pick cut points.**
  Every alignment in the output is MFA's own.
- **Filtering is not part of prep.** Fisher's junk is textual (`[laughter]` is
  acoustically invisible — normal SNR, better alignment likelihood) and CANDOR's
  is acoustic (crosstalk), so the filter policy is measured per corpus and lives
  downstream.

## NOTSOFAR-1

**The data is on this cluster**, as a staging copy made for chime-utils:

    /scratch/bbjs/cornell2/CHiME6/tmp_chimeutils/notsofar1_gt_dev_old/
        dev_set/240130.1_dev_with_GT_delme/MTG/MTG_*/

That path is named `_old` / `_delme` and is not a durable home — re-download to
a permanent root and repoint `root:` before anything depends on it. `dev_set` is
also the only split here with audio: the chime-utils reformat under
`new_notsofar_train/` carries the train transcriptions (390 sessions, 103,505
segments) but its `audio/train` is empty.

Measured on the real `dev_set` (`msprep manifest` then `msprep prep`):

| | |
|---|---|
| sessions / close-talk channels | 36 / 201 |
| close-talk audio | 20.93 h, 16 kHz mono |
| segments | 10,473 → 9,674 kept, 799 dropped as pure annotation |
| speakers | 201 (from 12 alias strings — see below) |
| turns over `warn_turn_over` | 0; longest turn 8.0 s |
| segments unmatched to a headset | 0 |

Three things differ from Fisher and CANDOR:

- **It ships no manifests**, so `manifest.py` builds them from the raw tree
  first. Durations come from the wav headers via the stdlib `wave` module, so
  that stage needs no audio library.
- **Only the close-talk track is a corpus.** `mic: far` raises instead of
  emitting one "speaker channel" per array microphone — a far-field channel
  carries every speaker at once. Recording id, and therefore the MFA file stem,
  is `<split>__<session>__<ct_stem>`, which makes `stem_key` its own inverse;
  `manifest.py` refuses ids whose parts contain the separator rather than
  producing a stem it cannot invert.
- **Participant ids are per-meeting cast names, not people.** "Bert", "Ernie",
  "Olivia" are reused across sessions by different speakers: the 201 dev_set
  channels carry just 12 distinct alias strings, and "Bert" alone is 31
  different people in 31 meetings. Unscoped they collapse into one identity,
  which silently corrupts anything that samples distinct speakers — FastMSS
  would build "meetings" whose participants are the same person. `speaker_scope:
  session` (the default) scopes the alias to `<session>__<alias>`, which is the
  real identity because an alias *is* unique within a meeting.

### Text markers

NOTSOFAR annotates with angle-bracket tags in two shapes, and both are stripped
(`text.strip_all_markers`, on by default):

- standalone markers — `<ST/>` `<FILL/>` `<FILLlaugh/>` `<UNKNOWN/>` `<BA/>`
  `<ISSUE/>` `<SN/>` `<PAUSE/>`
- the paired `<PName>Bert</PName>`, which wraps a **real spoken name** — the
  tags go, the name stays a word

Listing markers one at a time was the earlier rule and it under-matched: a
dev_set run left 508 `<FILLlaugh/>`, 197 `<BA/>`, 189 `<PName>` pairs and 143
`<ISSUE/>` in the TextGrids, where MFA reads each as a bracketed token and maps
it to `spn` — charging real words' audio to a non-speech phone. Square-bracket
events are still left alone on purpose, so `spn` keeps that audio accounted for.

Turns that are *only* annotation become empty and are dropped as non-lexical
(799 of 10,473 in dev_set).

### Word times

The release **does** ship per-word times, as `word_timing` on each segment
(`[symbol, start, end]`, absolute recording seconds). Every one of the 9,674
lexical turns carries them. They are not used for alignment — MFA produces its
own — but `keep_word_timing` (default on) converts them to Lhotse's
`[symbol, start, duration]` and attaches them to the supervisions, where cut
construction and FastMSS's pause splitting can use them; marker tokens are
excluded so the alignment carries the same words the normalized text does.

They are *not* needed for resegmentation: NOTSOFAR turns are short (longest 8.0 s
in dev_set, 8.7 s in the train transcriptions, none over 20 s), so nothing is
split. Turns longer than `warn_turn_over` are counted and reported
(`turns_over_warn_length`) rather than silently handed to MFA.

## Planned: `core.gating`

Zero (taper, really) the regions outside each speaker's oracle segmentation
before alignment, so a restored close-talk channel stops carrying its
neighbours. Three things it has to get right:

1. **Not literal zero.** Log-energy features floor out on exact silence and a
   hard edge is a wideband click the acoustic model can read as a spurious
   phone. 5–10 ms cosine ramps into a ~-60 dBFS floor.
2. **Between-turn leakage only.** Another speaker bleeding through *during* your
   turn is inside your segment and survives gating. Channel-quality filtering is
   still needed.
3. **Say which audio the cuts point at.** Gated audio is the right FastMSS
   source; that makes it canonical and the manifest `sources` must reflect it.

Restore (Sidon) first, then gate — a restoration model fed digital silence
invents content in it, and gating afterwards removes leakage the restorer just
enhanced.
