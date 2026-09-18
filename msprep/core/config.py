"""Corpus configuration: paths and parameters live in YAML, not in code.

Each corpus ships `msprep/corpora/<name>/<name>.yaml` with its manifest paths
and its knobs. Resolution order, first hit wins:

    1. an explicit path (CLI `--config`)
    2. $MSPREP_<CORPUS>_CONFIG            e.g. MSPREP_CANDOR_CONFIG
    3. $MSPREP_CONFIG_DIR/<corpus>.yaml   a whole directory of overrides
    4. the packaged default

String values are passed through `expandvars`/`expanduser`, so a config can be
written against `$MSPREP_DATA_ROOT` and moved between machines without edits.
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ImportError(
            "msprep reads corpus configs from YAML and needs PyYAML. "
            "On Delta: `conda activate mfa`, or `pip install pyyaml`."
        ) from exc
    with open(path, "rt", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def expand(obj):
    """Recursively expand $VARS and ~ in every string of a config tree."""
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    if isinstance(obj, dict):
        return {k: expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand(v) for v in obj]
    return obj


def deep_update(base: dict, other: dict) -> dict:
    """Merge `other` into a copy of `base`, recursing into nested dicts."""
    out = dict(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def config_path(corpus: str, package_dir: Path, explicit=None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get(f"MSPREP_{corpus.upper()}_CONFIG")
    if env:
        return Path(env)
    config_dir = os.environ.get("MSPREP_CONFIG_DIR")
    if config_dir and (Path(config_dir) / f"{corpus}.yaml").exists():
        return Path(config_dir) / f"{corpus}.yaml"
    return Path(package_dir) / f"{corpus}.yaml"


def load_config(corpus: str, package_dir: Path, explicit=None,
                overrides: dict | None = None) -> dict:
    path = config_path(corpus, package_dir, explicit)
    if not path.exists():
        raise FileNotFoundError(f"no config for corpus {corpus!r} at {path}")
    cfg = expand(_load_yaml(path))
    if overrides:
        cfg = deep_update(cfg, overrides)
    cfg["_config_path"] = str(path)
    return cfg
