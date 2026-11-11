"""Edge attacks on the foreign-import scanner (task 0.1, FM-16 and new rows).

A test marked xfail(strict=True) records an open fault. When the implementer fixes the
fault, the strict mark fails the test: remove the mark then.
"""
import importlib.util
import os
from pathlib import Path

import pytest

from semseg._independence import find_foreign_imports

ROOT = Path(__file__).resolve().parent.parent


def _plant(tmp_path, name, data):
    f = tmp_path / name
    if isinstance(data, bytes):
        f.write_bytes(data)
    else:
        f.write_text(data, encoding="utf-8")
    return f


def _modules(path):
    return [h[2] for h in find_foreign_imports([path])]


# ---------------------------------------------------------------- safe: caught or loud

@pytest.mark.parametrize("name,text,expected", [
    ("alias", "import fine_grid as fg\n", ["fine_grid"]),
    ("from_sub", "from fine_grid.x import y\n", ["fine_grid.x"]),
    ("star", "from usace_veg import *\n", ["usace_veg"]),
    ("deep", "import usace_veg.a.b.c\n", ["usace_veg.a.b.c"]),
    ("two", "import fine_grid.a, usace_veg.b\n", ["fine_grid.a", "usace_veg.b"]),
    ("try", "try:\n    import fine_grid\nexcept ImportError:\n    pass\n", ["fine_grid"]),
    ("typing", "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import fine_grid\n",
     ["fine_grid"]),
    ("builtins", "import builtins\nbuiltins.__import__('fine_grid')\n", ["fine_grid"]),
    ("importlib_attr", "import importlib\nimportlib.import_module('usace_veg.x')\n",
     ["usace_veg.x"]),
    ("crlf", "import os\r\nimport fine_grid\r\n", ["fine_grid"]),
])
def test_edge_caught_forms(tmp_path, name, text, expected):
    assert _modules(_plant(tmp_path, name + ".py", text)) == expected


@pytest.mark.parametrize("text", [
    "import fine_gridx\n",
    "import usace_vegetation\n",
    "import usace_veg2\n",
    "import xfine_grid\n",
    "import finegrid\n",
    "from . import fine_grid\n",
    "from .fine_grid import x\n",
    "from ..usace_veg import x\n",
    "from other.fine_grid import x\n",
    "x = 'fine_grid'\n",
    "__import__('fine_gridx')\n",
])
def test_edge_near_miss_names_not_flagged(tmp_path, text):
    assert _modules(_plant(tmp_path, "m.py", text)) == []


def test_edge_import_module_other_arg_types_do_not_crash(tmp_path):
    f = _plant(tmp_path, "m.py", "__import__(1)\n__import__(b'x')\n__import__(*['a'])\n"
               "d = {}\nd['k']('fine_grid_other')\n(lambda: 1)()()\n")
    assert _modules(f) == []


def test_edge_empty_file_and_empty_dir(tmp_path):
    f = _plant(tmp_path, "empty.py", "")
    assert _modules(f) == []
    d = tmp_path / "d"
    d.mkdir()
    assert find_foreign_imports([d]) == []


def test_edge_file_without_py_suffix_scanned_when_named(tmp_path):
    f = _plant(tmp_path, "script", "import fine_grid\n")
    assert _modules(f) == ["fine_grid"]


def test_edge_non_py_in_directory_skipped(tmp_path):
    _plant(tmp_path, "readme", "import fine_grid\n")
    assert find_foreign_imports([tmp_path]) == []


@pytest.mark.parametrize("name,data,exc", [
    ("syntax.py", "def (:\n", SyntaxError),
    ("py2.py", "print 'x'\n", SyntaxError),
    ("nul.py", b"import fine_grid\x00\n", (SyntaxError, ValueError)),
    ("nonutf.py", b"# \xff\xfe\nimport fine_grid\n", UnicodeDecodeError),
])
def test_edge_unreadable_file_fails_loud_not_silent(tmp_path, name, data, exc):
    # A guard must never return "clean" for a file that it cannot read.
    f = _plant(tmp_path, name, data)
    with pytest.raises(exc):
        find_foreign_imports([f])


def test_edge_nonexistent_path_fails_loud(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_foreign_imports([tmp_path / "nope"])


def test_edge_bad_argument_types_fail_loud():
    with pytest.raises(TypeError):
        find_foreign_imports(None)


def test_edge_broken_symlink_fails_loud(tmp_path):
    (tmp_path / "broken.py").symlink_to(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        find_foreign_imports([tmp_path])


@pytest.mark.skipif(os.getuid() == 0, reason="root ignores file modes")
def test_edge_unreadable_permission_fails_loud(tmp_path):
    f = _plant(tmp_path, "p.py", "import fine_grid\n")
    f.chmod(0)
    try:
        with pytest.raises(PermissionError):
            find_foreign_imports([f])
    finally:
        f.chmod(0o644)


def test_edge_symlink_loop_terminates(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "loop").symlink_to(tmp_path, target_is_directory=True)
    _plant(tmp_path / "a", "m.py", "import fine_grid\n")
    assert len(find_foreign_imports([tmp_path])) == 1


def test_edge_symlinked_file_is_scanned(tmp_path):
    (tmp_path / "src").mkdir()
    real = _plant(tmp_path / "src", "real.py", "import usace_veg\n")
    scan = tmp_path / "scan"
    scan.mkdir()
    (scan / "link.py").symlink_to(real)
    assert [h[2] for h in find_foreign_imports([scan])] == ["usace_veg"]


def test_edge_duplicate_path_arguments_report_each_time(tmp_path):
    f = _plant(tmp_path, "a.py", "import fine_grid\n")
    assert len(find_foreign_imports([f, f])) == 2


def test_edge_unicode_source_and_names(tmp_path):
    f = _plant(tmp_path, "u.py", "# 日本語 \U0001F642\nimport usace_veg\n")
    assert _modules(f) == ["usace_veg"]


# ---------------------------------------------------------------- open faults

@pytest.mark.parametrize("text", [
    "import importlib\nimportlib.import_module(name='fine_grid')\n",
    "__import__(name='usace_veg')\n",
])
def test_edge_keyword_argument_import(tmp_path, text):
    assert len(_modules(_plant(tmp_path, "m.py", text))) == 1


@pytest.mark.parametrize("text", [
    "import importlib\nimportlib.import_module(f'fine_grid')\n",
    "import importlib\nimportlib.import_module(f'fine_grid.{1}')\n",
    "import importlib\nimportlib.import_module('fine_' + 'grid')\n",
    "import importlib\nimportlib.import_module('fine_grid' + '.x')\n",
    "import importlib\nimportlib.import_module('fine_grid.%s' % 'x')\n",
])
def test_edge_built_name_import(tmp_path, text):
    assert len(_modules(_plant(tmp_path, "m.py", text))) == 1


@pytest.mark.parametrize("text", [
    "from importlib import import_module as im\nim('fine_grid')\n",
    "import importlib\nimportlib.import_module('.core', package='fine_grid')\n",
])
def test_edge_aliased_import_call(tmp_path, text):
    assert len(_modules(_plant(tmp_path, "m.py", text))) == 1


@pytest.mark.parametrize("data", [
    b"\xef\xbb\xbfimport fine_grid\n",
    b"# -*- coding: latin-1 -*-\nx = '\xe9'\nimport fine_grid\n",
])
def test_edge_valid_python_encodings(tmp_path, data):
    assert _modules(_plant(tmp_path, "m.py", data)) == ["fine_grid"]


def test_edge_directory_named_like_python_file(tmp_path):
    d = tmp_path / "dir.py"
    d.mkdir()
    _plant(d, "z.py", "import usace_veg\n")
    assert [h[2] for h in find_foreign_imports([tmp_path])] == ["usace_veg"]


def test_edge_symlinked_directory_is_scanned(tmp_path):
    ext = tmp_path / "ext"
    ext.mkdir()
    _plant(ext, "ev.py", "import usace_veg\n")
    scan = tmp_path / "scan"
    scan.mkdir()
    (scan / "link").symlink_to(ext, target_is_directory=True)
    assert len(find_foreign_imports([scan])) == 1


@pytest.mark.parametrize("name", ["w.pyw", "t.pyi"])
def test_edge_other_python_suffixes(tmp_path, name):
    _plant(tmp_path, name, "import fine_grid\n")
    assert len(find_foreign_imports([tmp_path])) == 1


# ---------------------------------------------------------------- pyproject helper

def _toml_helper():
    spec = importlib.util.spec_from_file_location("_t_indep", ROOT / "tests" / "test_independent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._toml_foreign_names


def _toml(tmp_path, text):
    f = tmp_path / "p.toml"
    f.write_text(text, encoding="utf-8")
    return _toml_helper()(f)


def test_edge_toml_dependency_group_and_hyphen_caught(tmp_path):
    assert _toml(tmp_path, '[project]\nname="x"\ndependencies=[]\n'
                 '[dependency-groups]\ndev=["usace-veg>=1", {include-group="o"}]\n'
                 ) == ["usace_veg"]


def test_edge_toml_url_dependency_caught(tmp_path):
    assert _toml(tmp_path, '[project]\nname="x"\n'
                 'dependencies=["fine-grid @ git+https://example.org/x.git"]\n'
                 ) == ["fine_grid"]


@pytest.mark.parametrize("text", [
    '[project]\nname="x"\n[project.optional-dependencies]\nveg=["usace-veg"]\n',
    '[project]\nname="x"\n[tool.uv]\ndev-dependencies=["fine-grid"]\n',
    '[project]\nname="x"\n[tool.uv.workspace]\nmembers=["../usace_veg"]\n',
    '[project]\nname="x"\ndependencies=["helper"]\n'
    '[tool.uv.sources]\nhelper={ path = "../usace-landuse/fine_grid" }\n',
    '[project]\nname="x"\ndependencies=["helper"]\n'
    '[tool.uv.sources]\nhelper={ git = "https://example.org/usace_veg.git" }\n',
])
def test_edge_toml_other_sections(tmp_path, text):
    assert _toml(tmp_path, text) != []


def test_edge_toml_near_miss_name_not_flagged(tmp_path):
    # FM-16 fix: the match is on word boundaries, so a near-miss name is not flagged.
    assert _toml(tmp_path, '[project]\nname="x"\ndependencies=["usace-vegetation", "fine_gridx"]\n') == []


# ---------------------------------------------------------------- fix round extras

def test_edge_variable_name_is_a_documented_non_goal(tmp_path):
    # Non-goal (FM-19): a name built from a variable cannot be resolved statically.
    f = _plant(tmp_path, "m.py", "import importlib\nn = 'fine_grid'\nimportlib.import_module(n)\n"
               "importlib.import_module(n + '.x')\nimportlib.import_module(f'{n}')\n")
    assert _modules(f) == []


@pytest.mark.parametrize("text,count", [
    ("from builtins import __import__ as imp\nimp('usace_veg')\n", 1),
    ("import importlib as il\nil.import_module('fine_grid')\n", 1),
    ("import importlib\nimportlib.import_module('.x', 'fine_grid')\n", 1),
    ("import importlib\nimportlib.import_module('.x')\n", 0),
    ("import importlib\nimportlib.import_module('.x', package=v)\n", 0),
    ("import importlib\nimportlib.import_module()\n", 0),
    ("import importlib\nimportlib.import_module(f'{1:>3}')\n", 0),
    ("import importlib\nimportlib.import_module(f'{v}')\n", 0),
    ("import importlib\nimportlib.import_module('a' + v)\n", 0),
    ("import importlib\nimportlib.import_module(v + 'a')\n", 0),
    ("import importlib\nimportlib.import_module('%d' % 'x')\n", 0),
    ("import importlib\nimportlib.import_module('%s' % v)\n", 0),
    ("import importlib\nimportlib.import_module(v % 1)\n", 0),
    ("import importlib\nimportlib.import_module('fine_%s' % ('grid',))\n", 1),
    ("import importlib\nimportlib.import_module('a' - 'b')\n", 0),
    ("from importlib import import_module\nimport_module('fine_grid')\n", 1),
    ("from importlib import util\nutil.x('fine_grid')\n", 0),
])
def test_edge_call_forms(tmp_path, text, count):
    assert len(_modules(_plant(tmp_path, "m.py", text))) == count


def test_edge_fifo_in_directory_is_skipped(tmp_path):
    os.mkfifo(tmp_path / "pipe.py")
    _plant(tmp_path, "ok.py", "import fine_grid\n")
    assert len(find_foreign_imports([tmp_path])) == 1


def test_edge_unreadable_subdirectory_fails_loud(tmp_path):
    if os.getuid() == 0:
        pytest.skip("root ignores file modes")
    d = tmp_path / "locked"
    d.mkdir()
    d.chmod(0)
    try:
        with pytest.raises(PermissionError):
            find_foreign_imports([tmp_path])
    finally:
        d.chmod(0o755)


def test_edge_toml_find_foreign_deps_direct(tmp_path):
    from semseg._independence import find_foreign_deps
    f = tmp_path / "p.toml"
    f.write_text('[a]\nk = [{ x = "Fine.Grid" }]\nn = 5\n[b]\n"usace-veg" = 1\n')
    assert find_foreign_deps(f) == ["fine_grid", "usace_veg"]
