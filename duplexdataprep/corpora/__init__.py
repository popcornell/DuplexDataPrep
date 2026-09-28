"""Corpus packages. Importing this module registers every built-in corpus.

One directory per corpus:

    corpora/<name>/__init__.py   the ChannelSource subclass and its functions
    corpora/<name>/<name>.yaml   its paths and parameters
    corpora/<name>/*.py          anything only that corpus needs (e.g. CANDOR's
                                 stereo extraction)
"""

from __future__ import annotations

from .base import Channel, ChannelSource, available, get_source, register  # noqa: F401
from . import candor, fisher, notsofar, openyap, otospeech  # noqa: F401,E402  (registers each corpus)

__all__ = ["Channel", "ChannelSource", "available", "get_source", "register"]
