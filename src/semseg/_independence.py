"""Find foreign packages (R4, D3). The scan uses the AST, not text."""
import ast
import os
import re
import tokenize
import tomllib
from io import BytesIO
from pathlib import Path

FOREIGN = ("fine_grid", "usace_veg")
SUFFIXES = (".py", ".pyi", ".pyw")
IMPORT_FUNCS = ("import_module", "__import__")


def _is_foreign(name):
    return name.split(".")[0] in FOREIGN


def _fold(node):
    """Return the string value of a constant expression, or None if it is not constant."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _fold(node.left), _fold(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return _fold_percent(node)
    if isinstance(node, ast.JoinedStr):
        parts = [_fold_fpart(v) for v in node.values]
        return None if None in parts else "".join(parts)
    return None


def _fold_fpart(part):
    if isinstance(part, ast.Constant):
        return part.value
    simple = (isinstance(part, ast.FormattedValue) and part.conversion == -1
              and part.format_spec is None and isinstance(part.value, ast.Constant))
    return str(part.value.value) if simple else None


def _fold_percent(node):
    left = _fold(node.left)
    right = node.right
    items = right.elts if isinstance(right, ast.Tuple) else [right]
    if left is None or not all(isinstance(i, ast.Constant) for i in items):
        return None
    try:
        return left % tuple(i.value for i in items)
    except (TypeError, ValueError):
        return None


def _aliases(tree):
    """Return the names bound to import_module or __import__ by a from-import."""
    names = set(IMPORT_FUNCS)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("importlib", "builtins"):
            names.update(a.asname for a in node.names
                         if a.name in IMPORT_FUNCS and a.asname)
    return names


def _call_target(node, aliases):
    """Return the module name that an import call names, or None."""
    func = node.func
    name = getattr(func, "id", None) or getattr(func, "attr", None)
    if name not in aliases:
        return None
    kw = {k.arg: k.value for k in node.keywords if k.arg}
    first = node.args[0] if node.args else kw.get("name")
    pkg = node.args[1] if len(node.args) > 1 else kw.get("package")
    target = _fold(first) if first is not None else None
    if target is not None and target.startswith("."):
        base = _fold(pkg) if pkg is not None else None
        target = base + target if base is not None else None
    return target


def _decode(data):
    """Decode as Python does (BOM, coding line). Bad bytes raise UnicodeDecodeError."""
    try:
        encoding, _ = tokenize.detect_encoding(BytesIO(data).readline)
    except SyntaxError:  # the first two lines are not valid UTF-8
        encoding = "utf-8"
    return data.decode(encoding)


def _scan_file(path):
    tree = ast.parse(_decode(path.read_bytes()), filename=str(path))
    aliases = _aliases(tree)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [(path, node.lineno, a.name) for a in node.names if _is_foreign(a.name)]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and _is_foreign(node.module):
                hits.append((path, node.lineno, node.module))
        elif isinstance(node, ast.Call):
            name = _call_target(node, aliases)
            if name is not None and _is_foreign(name):
                hits.append((path, node.lineno, name))
    return hits


def _walk_files(root):
    """Yield the Python files under root. Follow symbolic links; stop on a loop."""
    def fail(exc):
        raise exc

    seen = set()
    for top, dirs, files in os.walk(root, followlinks=True, onerror=fail):
        st = os.stat(top)
        key = (st.st_dev, st.st_ino)
        if key in seen:
            dirs[:] = []
            continue
        seen.add(key)
        for f in sorted(files):
            p = Path(top, f)
            if p.suffix in SUFFIXES and (p.is_file() or not p.exists()):
                yield p


def find_foreign_imports(paths):
    """Return (path, lineno, module) for each foreign import in the files or folders."""
    hits = []
    for p in paths:
        p = Path(p)
        for f in (_walk_files(p) if p.is_dir() else [p]):
            hits += _scan_file(f)
    return sorted(hits, key=lambda h: (str(h[0]), h[1]))


def _strings(node):
    """Yield every key and string value in a parsed TOML value."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item)


def _foreign_pattern():
    names = "|".join(n.replace("_", "-") for n in FOREIGN)
    return re.compile(rf"(?<![a-z0-9])({names})(?![a-z0-9])")


def find_foreign_deps(pyproject_path):
    """Return the sorted foreign names in any key or string value of a TOML file.

    A name is normalised as in PEP 503 and matched on word boundaries.
    """
    data = tomllib.loads(Path(pyproject_path).read_text(encoding="utf-8"))
    pattern = _foreign_pattern()
    found = set()
    for text in _strings(data):
        norm = re.sub(r"[-_.]+", "-", text.lower())
        found.update(m.replace("-", "_") for m in pattern.findall(norm))
    return sorted(found)
