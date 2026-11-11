"""Load config.yaml from the repository root."""
import math
import re
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
REQUIRED = ("share_root", "aois")


class ConfigError(ValueError, yaml.YAMLError):
    """The config is not valid. It is a ValueError and a YAMLError."""


class _UniqueLoader(yaml.SafeLoader):
    """A safe loader that refuses a duplicate key in a mapping."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if isinstance(key, (list, dict, set)):
                continue  # the base class refuses an unhashable key
            if key in seen:
                raise ConfigError(f"duplicate key {key!r} (line {key_node.start_mark.line + 1})")
            seen.add(key)
        return super().construct_mapping(node, deep)


_CHUNK = re.compile(r"r(?:0|[1-9][0-9]*)_c(?:0|[1-9][0-9]*)", re.ASCII)
MIN_FREE_FLOOR_MB = 8000  # D50: the lowest allowed `segment.min_free_mb` (the memory stop cannot be off)
_NUMBERS = ("simplify_tolerance_m", "min_area_m2")


def non_negative_number(value):
    """True for a finite int or float (not a bool) that is 0 or more."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


LABEL_MODEL_IDS = ("sam2-hiera-tiny", "sam2-hiera-small", "sam2-hiera-base-plus", "sam2-hiera-large")
LABEL_DEFAULTS = {"model_id": "sam2-hiera-large", "box_threshold": 0.25, "text_threshold": 0.25,
                  "min_overlap": 0.5}


def _real(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value))


def label_settings(section):
    """Return the `label` settings (D51 item 7) as a dict of the 4 keys. A missing key takes its default.

    `section` is the `label` mapping or None. Raise ValueError for a value that is not valid.
    """
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("config label must be a mapping")
    out = {**LABEL_DEFAULTS, **{k: v for k, v in section.items() if k in LABEL_DEFAULTS}}
    for key in ("box_threshold", "text_threshold"):
        if not _real(out[key]) or not 0 < out[key] <= 1:
            raise ValueError(f"config label.{key} must be a number with 0 < value <= 1, got {out[key]!r}")
    if not _real(out["min_overlap"]) or not 0 <= out["min_overlap"] <= 1:
        raise ValueError(f"config label.min_overlap must be a number from 0 to 1, got {out['min_overlap']!r}")
    if out["model_id"] not in LABEL_MODEL_IDS:
        raise ValueError(f"config label.model_id must be one of {LABEL_MODEL_IDS}, got {out['model_id']!r}")
    return out


def _check_label(path, section):
    """Check the optional `label` section (D51)."""
    try:
        label_settings(section)
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _check_segment(path, seg):
    """Check the optional `segment` section (D47). A missing key is allowed."""
    if not isinstance(seg, dict):
        raise ConfigError(f"{path}: segment must be a mapping")
    free = seg.get("min_free_mb", MIN_FREE_FLOOR_MB)
    if type(free) is not int or free < MIN_FREE_FLOOR_MB:
        raise ConfigError(f"{path}: segment.min_free_mb must be an integer of {MIN_FREE_FLOOR_MB} or more (D50)")
    for key in _NUMBERS:
        if key in seg and not non_negative_number(seg[key]):
            raise ConfigError(f"{path}: segment.{key} must be a finite number of 0 or more")
    model = seg.get("model_id", "x")
    if not isinstance(model, str) or not model.strip():
        raise ConfigError(f"{path}: segment.model_id must be a non-empty string")


def _check_demo(path, demo):
    """Check the optional `demo` section (D46)."""
    if not isinstance(demo, dict):
        raise ConfigError(f"{path}: demo must be a mapping")
    pid, chunks = demo.get("pid"), demo.get("chunks")
    if type(pid) is not int or pid < 1:
        raise ConfigError(f"{path}: demo.pid must be an integer of 1 or more")
    if (not isinstance(chunks, list) or not chunks
            or any(not isinstance(c, str) or _CHUNK.fullmatch(c) is None for c in chunks)
            or len(set(chunks)) != len(chunks)):
        raise ConfigError(f"{path}: demo.chunks must be a non-empty list of unique chunk IDs")


def _check(path, data):
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the top level must be a mapping")
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise ConfigError(f"{path}: missing keys {missing}")
    root = data["share_root"]
    if not isinstance(root, str) or not root or not Path(root).is_absolute():
        raise ConfigError(f"{path}: share_root must be a non-empty absolute path string")
    aois = data["aois"]
    if (not isinstance(aois, list) or not aois
            or any(type(a) is not int for a in aois)):
        raise ConfigError(f"{path}: aois must be a non-empty list of integers")
    if "segment" in data:
        _check_segment(path, data["segment"])
    if "label" in data:
        _check_label(path, data["label"])
    if "demo" in data:
        _check_demo(path, data["demo"])


def load_config(path=None):
    """Return the config mapping. Raise ValueError if it is not valid."""
    path = Path(path) if path is not None else DEFAULT_PATH
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.load(text, Loader=_UniqueLoader)  # noqa: S506 (SafeLoader subclass)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    _check(path, data)
    return data
