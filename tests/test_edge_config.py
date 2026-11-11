"""Edge attacks on load_config (task 0.1, new rows FM-22 and FM-23).

load_config says "Raise ValueError if it is not valid". A test marked xfail(strict=True)
records an open fault. Remove the mark when the implementer fixes it.
"""
import os

import pytest
import yaml

from semseg.config import load_config


def _cfg(tmp_path, text, name="c.yaml", mode="w", **kw):
    f = tmp_path / name
    f.write_text(text, encoding="utf-8", **kw) if mode == "w" else f.write_bytes(text)
    return f


# ---------------------------------------------------------------- safe

def test_edge_directory_as_path_fails_loud(tmp_path):
    with pytest.raises(IsADirectoryError):
        load_config(tmp_path)


def test_edge_non_utf8_fails_loud(tmp_path):
    with pytest.raises(UnicodeDecodeError):
        load_config(_cfg(tmp_path, b"share_root: \xff\naois: [1]\n", mode="b"))


@pytest.mark.skipif(os.getuid() == 0, reason="root ignores file modes")
def test_edge_unreadable_file_fails_loud(tmp_path):
    f = _cfg(tmp_path, "share_root: /x\naois: [1]\n")
    f.chmod(0)
    try:
        with pytest.raises(PermissionError):
            load_config(f)
    finally:
        f.chmod(0o644)


def test_edge_python_tag_is_refused(tmp_path):
    # safe_load must not run code from the file.
    f = _cfg(tmp_path, "share_root: !!python/object/apply:os.system ['true']\naois: [1]\n")
    with pytest.raises(yaml.YAMLError):
        load_config(f)


def test_edge_multi_document_refused(tmp_path):
    with pytest.raises(yaml.YAMLError):
        load_config(_cfg(tmp_path, "share_root: /x\naois: [1]\n---\nshare_root: /y\n"))


def test_edge_bom_crlf_unicode_accepted(tmp_path):
    f = _cfg(tmp_path, "﻿share_root: /x/日本\U0001F642\r\naois: [1]\r\n")
    assert load_config(f) == {"share_root": "/x/日本\U0001F642", "aois": [1]}


def test_edge_extra_and_non_string_keys_pass_through(tmp_path):
    cfg = load_config(_cfg(tmp_path, "1: a\nshare_root: /x\naois: [1]\nzzz: {k: v}\n"))
    assert cfg[1] == "a" and cfg["zzz"] == {"k": "v"}


def test_edge_empty_aois_list_is_refused(tmp_path):
    # FM-22 fix: an empty aois list is refused (it was "accepted as is" before).
    with pytest.raises(ValueError):
        load_config(_cfg(tmp_path, "share_root: /x\naois: []\n"))


def test_edge_unhashable_key_still_refused(tmp_path):
    with pytest.raises(yaml.YAMLError):
        load_config(_cfg(tmp_path, "? [a]\n: 1\nshare_root: /x\naois: [1]\n"))


def test_edge_error_names_the_file_and_is_both_error_types(tmp_path):
    f = _cfg(tmp_path, "share_root: [unclosed\n")
    with pytest.raises(ValueError, match="c.yaml") as ei:
        load_config(f)
    assert isinstance(ei.value, yaml.YAMLError)


# ---------------------------------------------------------------- open faults

@pytest.mark.parametrize("value", ["null", "5", "yes", "[a]", "''", "{a: b}"])
def test_edge_share_root_must_be_nonempty_string(tmp_path, value):
    with pytest.raises(ValueError):
        load_config(_cfg(tmp_path, f"share_root: {value}\naois: [1]\n"))


@pytest.mark.parametrize("value", ["", "77", "'77'", "['a','b']", "[1, true]", "[2.5]",
                                   "[.nan]", "{a: 1}"])
def test_edge_aois_must_be_list_of_int(tmp_path, value):
    with pytest.raises(ValueError):
        load_config(_cfg(tmp_path, f"share_root: /x\naois: {value}\n"))


def test_edge_share_root_must_be_absolute(tmp_path):
    with pytest.raises(ValueError):
        load_config(_cfg(tmp_path, "share_root: rel/path\naois: [1]\n"))


def test_edge_duplicate_key_refused(tmp_path):
    with pytest.raises(ValueError):
        load_config(_cfg(tmp_path, "share_root: /a\nshare_root: /b\naois: [1]\n"))


def test_edge_malformed_yaml_is_value_error(tmp_path):
    with pytest.raises(ValueError):
        load_config(_cfg(tmp_path, "share_root: [unclosed\n"))
