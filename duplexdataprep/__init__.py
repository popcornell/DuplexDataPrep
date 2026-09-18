"""duplexdataprep -- multi-speaker corpus preprocessing for forced alignment and FastMSS.

Corpus-specific knowledge lives in `duplexdataprep.corpora.<name>` (one directory per
corpus: its ChannelSource subclass, its YAML config, its own helpers);
everything else (`duplexdataprep.core`) is shared. See README.md.
"""

from __future__ import annotations

__version__ = "0.3.0"

from .corpora import available, get_source  # noqa: F401
