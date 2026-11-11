"""R4 / D3: no module imports or copies fine_grid or usace_veg (FM-16)."""
from pathlib import Path

from semseg._independence import find_foreign_deps, find_foreign_imports

ROOT = Path(__file__).resolve().parent.parent
FOREIGN = ("fine_grid", "usace_veg")


def _toml_foreign_names(pyproject: Path) -> list[str]:
    """Return the foreign names in the whole TOML file."""
    return find_foreign_deps(pyproject)


def test_no_foreign_imports():
    hits = find_foreign_imports([ROOT / "src", ROOT / "tests"])
    assert hits == []
    assert _toml_foreign_names(ROOT / "pyproject.toml") == []


def test_scanner_scans_real_trees_that_exist():
    # The scan must not pass because it saw no file.
    assert any((ROOT / "src" / "semseg").glob("*.py"))
    assert (ROOT / "tests" / "test_independent.py").is_file()


def _plant(tmp_path, name, text):
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return f


def _names(hits):
    return sorted(h[2].split(".")[0] for h in hits)


def test_scanner_catches_planted_imports(tmp_path):
    cases = {
        "a_import.py": "import fine_grid\n",
        "b_import_sub.py": "import fine_grid.core.x\n",
        "c_import_as.py": "import usace_veg as uv\n",
        "d_from.py": "from fine_grid import x\n",
        "e_from_sub.py": "from usace_veg.sub import y\n",
        "f_importlib.py": "import importlib\nm = importlib.import_module('fine_grid.x')\n",
        "g_dunder.py": "m = __import__('usace_veg')\n",
        "h_nested.py": "def f():\n    if True:\n        import usace_veg\n",
        "i_multi.py": "import os, fine_grid\n",
    }
    for name, text in cases.items():
        f = _plant(tmp_path, name, text)
        hits = find_foreign_imports([f])
        assert len(hits) == 1, name
        path, lineno, mod = hits[0]
        assert Path(path) == f, name
        assert isinstance(lineno, int) and lineno >= 1, name
        assert mod.split(".")[0] in FOREIGN, name


def test_scanner_reports_line_number(tmp_path):
    f = _plant(tmp_path, "m.py", "import os\n\n\nimport usace_veg\n")
    assert [h[1] for h in find_foreign_imports([f])] == [4]


def test_scanner_walks_directories_recursively(tmp_path):
    sub = tmp_path / "pkg" / "deep"
    sub.mkdir(parents=True)
    _plant(sub, "x.py", "from fine_grid import z\n")
    _plant(tmp_path, "y.py", "import usace_veg\n")
    assert _names(find_foreign_imports([tmp_path])) == ["fine_grid", "usace_veg"]


def test_scanner_accepts_str_paths(tmp_path):
    f = _plant(tmp_path, "s.py", "import fine_grid\n")
    assert len(find_foreign_imports([str(f)])) == 1


def test_scanner_clean_files_give_empty_list(tmp_path):
    _plant(tmp_path, "ok.py", "import os\nfrom pathlib import Path\nimport finegrid\n"
           "import fine_grid_other\nfrom . import sibling\nfrom .x import y\n"
           "import importlib\nimportlib.import_module('numpy')\n__import__('json')\n"
           "x = 'import fine_grid'\n# import usace_veg\n")
    assert find_foreign_imports([tmp_path]) == []


def test_scanner_empty_inputs(tmp_path):
    assert find_foreign_imports([]) == []
    assert find_foreign_imports([tmp_path]) == []


def test_scanner_ignores_non_python_files(tmp_path):
    _plant(tmp_path, "notes.txt", "import fine_grid\n")
    assert find_foreign_imports([tmp_path]) == []


def test_scanner_counts_each_hit(tmp_path):
    f = _plant(tmp_path, "two.py", "import fine_grid\nimport usace_veg\n")
    assert len(find_foreign_imports([f])) == 2


def test_toml_scan_catches_planted_names(tmp_path):
    bad_dep = tmp_path / "a.toml"
    bad_dep.write_text('[project]\nname="x"\ndependencies = ["fine_grid>=1"]\n')
    assert _toml_foreign_names(bad_dep) == ["fine_grid"]
    bad_src = tmp_path / "b.toml"
    bad_src.write_text('[project]\nname="x"\ndependencies = []\n'
                       '[tool.uv.sources]\nusace-veg = { path = "../usace_veg", editable = true }\n')
    assert _toml_foreign_names(bad_src) == ["usace_veg"]
    good = tmp_path / "c.toml"
    good.write_text('[project]\nname="x"\ndependencies = ["numpy"]\n'
                    '[tool.uv.sources]\ntorch = { index = "i" }\n')
    assert _toml_foreign_names(good) == []
