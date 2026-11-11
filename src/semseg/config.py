"""Load config.yaml from the repository root."""
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
