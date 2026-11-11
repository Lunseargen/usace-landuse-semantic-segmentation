"""config.yaml loader (task 0.1)."""
from pathlib import Path

import pytest

from semseg.config import load_config

ROOT = Path(__file__).resolve().parent.parent
D23 = "/run/user/1001/gvfs/smb-share:server=100.102.218.98,share=lab/data/usace-landuse-semseg"


def test_config_file_at_repository_root():
    assert (ROOT / "config.yaml").is_file()


def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert cfg["share_root"] == D23
    assert 77 in cfg["aois"] and 392 in cfg["aois"]


def test_load_default_equals_explicit_root_file():
    assert load_config() == load_config(ROOT / "config.yaml")


def test_load_explicit_str_and_path(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("share_root: /x\naois: [1, 2]\nextra: 5\n")
    for p in (f, str(f)):
        assert load_config(p) == {"share_root": "/x", "aois": [1, 2], "extra": 5}


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize("text", ["- a\n- b\n", "just a string\n", "42\n"])
def test_non_mapping(tmp_path, text):
    f = tmp_path / "c.yaml"
    f.write_text(text)
    with pytest.raises(ValueError):
        load_config(f)


def test_empty_file(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("")
    with pytest.raises(ValueError):
        load_config(f)


def test_comment_only_file(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("# nothing\n")
    with pytest.raises(ValueError):
        load_config(f)


@pytest.mark.parametrize("text", ["aois: [77, 392]\n", "share_root: /x\n", "other: 1\n"])
def test_missing_required_key(tmp_path, text):
    f = tmp_path / "c.yaml"
    f.write_text(text)
    with pytest.raises(ValueError):
        load_config(f)
