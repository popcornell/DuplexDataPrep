"""The one interface every corpus implements.

A corpus, for this pipeline's purposes, is a stream of **single-speaker
channels**: one mono audio file, one speaker, and that speaker's turns in
absolute recording time. Fisher is two conversation sides, CANDOR is a stereo
mp3 split in two, NOTSOFAR-1 is the close-talk mic per participant. Everything
downstream -- TextGrid writing, sharding, alignment, folding alignments back,
cut construction, FastMSS simulation -- is corpus-agnostic once a corpus can be
expressed this way.

Note "speaker channel", not "recording channel". A far-field array channel is
not a ChannelSource: it carries every speaker at once and cannot be force
aligned without separation first.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..core.config import load_config


@dataclass
class Channel:
    """One speaker's channel, ready to be written as an MFA corpus file."""

    file_id: str
    """MFA file stem, e.g. `fe_03_00001-A`. Identifies the file everywhere
    downstream: TextGrid name, quality tables, blocklists."""

    audio_path: Path
    """Existing mono audio. Adapters that must cut or decode audio (CANDOR's
    stereo split) do so before yielding, so this always points at a real file."""

    speaker: str
    """TextGrid tier name -- MFA reads the tier name as the speaker id."""

    duration: float
    """Recording duration; becomes the TextGrid `xmax`."""

    turns: list[tuple[float, float, str]]
    """(start, end, normalized_text) in absolute recording time."""

    group: str
    """Recording id. Channels sharing a group land in the same shard."""

    group_index: int
    """Ordinal of the recording in the source manifest; drives round-robin
    sharding, so both sides of a conversation stay together."""


class ChannelSource(ABC):
    """Adapter protocol: turn a corpus into a stream of speaker channels.

    Paths and parameters come from the corpus's YAML config (see
    `msprep.core.config`); only behaviour lives in the subclass.
    """

    name: str

    def __init__(self, config=None, **overrides) -> None:
        pkg_dir = Path(inspect.getfile(type(self))).resolve().parent
        self.config = load_config(self.name, pkg_dir, explicit=config,
                                  overrides=overrides or None)
        self.stats: dict = {}
        self.configure()

    def configure(self) -> None:
        """Hook: derive attributes from `self.config` after it is loaded."""

    @abstractmethod
    def iter_channels(self, limit: int = 0) -> Iterator[Channel]:
        """Yield every speaker channel. `limit` caps *recordings*, not channels.

        Implementations should record `recordings`, `supervisions` and
        `dropped_non_lexical` in `self.stats` as they go.
        """

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Transcript convention -> aligner-friendly text."""

    def build_manifests(self, **kwargs) -> dict:
        """Raw release tree -> Lhotse-style manifests.

        Only for corpora whose release ships none, like NOTSOFAR-1. Fisher
        and CANDOR already have theirs, so the default refuses.
        """
        raise NotImplementedError(
            f"{self.name} already ships Lhotse manifests; `msprep manifest` is "
            "only for corpora whose release has none")

    # ---- used when folding alignments back (core.collect)

    @property
    @abstractmethod
    def supervisions(self) -> str:
        """Path to the source supervision manifest the alignments attach to."""

    @abstractmethod
    def stem_key(self, stem: str) -> tuple[str, int]:
        """TextGrid stem -> (recording_id, channel). Inverse of `file_id`."""


# ------------------------------------------------------------------ registry

_REGISTRY: dict[str, type[ChannelSource]] = {}


def register(cls: type[ChannelSource]) -> type[ChannelSource]:
    _REGISTRY[cls.name] = cls
    return cls


def get_source(name: str, config=None, **overrides) -> ChannelSource:
    if name not in _REGISTRY:
        raise KeyError(f"unknown corpus {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name](config=config, **overrides)


def available() -> list[str]:
    return sorted(_REGISTRY)
