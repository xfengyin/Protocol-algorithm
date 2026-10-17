"""examples/ 与 src/ 里的「关键字透传」必须真的落到形参上。

## 为什么需要这个测试

2026-10-17 发现 `examples/run_ai_leach.py` 里写着：

    AITrainer(model_type="sklearn", model_type_rf="rf", n_estimators=100)

`AITrainer.__init__` 的签名是 `(model_type, **kwargs)`，它把 `**kwargs` 原样转给
`SklearnClusterSelector(**kwargs)`。而 `SklearnClusterSelector` 从来没有
`model_type_rf` 这个参数 —— **这段示例一调用就 TypeError**。

现有门禁全部看不见它：

- `examples/` 只走 black / isort / flake8 / mypy，**不进 pytest**；
- mypy 不报错，因为 `SklearnClusterSelector(**kwargs)` 里 `kwargs` 是 `Any`，
  透传目标无法静态确定；
- 运行时才会炸，而示例脚本没有人跑。

所以这里补一道**能红的**检查：把「调用点写了哪些关键字」和
「透传目标真的接受哪些关键字」对起来。

## 判据（刻意保守，宁可少报也不误报）

只检查**能在本仓解析出定义**的类：
1. 取出 `src/` 下所有类的 `__init__` 形参；
2. 对 `examples/`、`src/`、`tests/` 里出现的 `Cls(...)` 调用，收集字面关键字；
3. 如果 `Cls.__init__` 有 `**kwargs`，就把「本仓里被它转发的目标类」也算进来，
   条件是：`**kwargs` 的参数名出现在某个 `Target(**kwargs)` 调用里；
4. 一个字面关键字必须被 `Cls` 自己**或**某个转发目标接受，否则报错。
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ("src", "examples", "tests")
SKIP_DIRS = {".venv", "__pycache__", ".git", "build", "dist"}


# ── 解析工具 ──────────────────────────────────────────────────────────────
def py_files():
    for d in SCAN_DIRS:
        for p in (ROOT / d).rglob("*.py"):
            if not any(part in SKIP_DIRS for part in p.parts):
                yield p


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def class_inits(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    """类名 → __init__ 的 FunctionDef。同名类只保留第一个（本仓没有同名不同物的实例）。"""
    out: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    out.setdefault(node.name, item)
    return out


def params(fn: ast.FunctionDef) -> tuple[list[str], str | None]:
    """(具名形参, **kwargs 的名字)。"""
    named = [a.arg for a in fn.args.args[1:]]  # 去掉 self
    named += [a.arg for a in fn.args.kwonlyargs]
    return named, (fn.args.kwarg.arg if fn.args.kwarg else None)


def forwarded_targets(fn: ast.FunctionDef, kwarg_name: str) -> set[str]:
    """`Target(**kwargs)` 形式出现在 fn 体内的目标类名。"""
    out: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if not name:
            continue
        for k in node.keywords:
            if k.arg is None and isinstance(k.value, ast.Name) and k.value.id == kwarg_name:
                out.add(name)
    return out


def build_index():
    inits: dict[str, ast.FunctionDef] = {}
    for p in py_files():
        inits.update(class_inits(parse(p)))
    return inits


def accepted_keywords(cls: str, inits, seen=None) -> tuple[set[str], bool]:
    """该类（含其转发目标）接受的具名关键字，以及是否接受任意关键字。"""
    seen = seen or set()
    if cls in seen or cls not in inits:
        return set(), False
    seen.add(cls)
    fn = inits[cls]
    named, kw = params(fn)
    keys = set(named)
    if kw:
        for target in forwarded_targets(fn, kw):
            more, _ = accepted_keywords(target, inits, seen)
            keys |= more
    return keys, kw is not None


def call_sites():
    """所有 `Cls(...)` 调用点里的字面关键字。"""
    for p in py_files():
        tree = parse(p)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if not name:
                continue
            kws = [k.arg for k in node.keywords if k.arg is not None]
            if kws:
                yield p, node.lineno, name, kws


# ── 判据 ──────────────────────────────────────────────────────────────────
def violations(inits) -> list[str]:
    bad = []
    for path, lineno, cls, kws in call_sites():
        if cls not in inits:  # 不是本仓定义的类，无法判定，跳过
            continue
        ok_keys, tolerant = accepted_keywords(cls, inits)
        if not tolerant:
            # 没有 **kwargs 的类，mypy 自己就会报，这里不重复
            continue
        unknown = [k for k in kws if k not in ok_keys]
        if unknown:
            rel = path.relative_to(ROOT)
            bad.append(
                "%s:%d  %s(%s) —— %s 既不在 %s 的形参里，也不在它转发目标的形参里"
                % (rel, lineno, cls, ", ".join("%s=..." % k for k in kws), ", ".join(unknown), cls)
            )
    return bad


def test_examples_wiring_has_no_unknown_keywords():
    inits = build_index()
    assert inits, "没有解析到任何类的 __init__，检查扫描目录"
    bad = violations(inits)
    assert not bad, "关键字透传目标不接受的参数（会在运行时 TypeError）：\n  " + "\n  ".join(bad)


# ── 负对照：这道门禁必须能红 ──────────────────────────────────────────────
def test_negative_control_catches_the_2026_10_17_bug():
    """把当年的错法构造出来，判据必须报出来 —— 否则这测试只是装饰。"""
    inits = build_index()
    # 当年 examples/run_ai_leach.py:139 的写法
    fake = 'AITrainer(model_type="sklearn", model_type_rf="rf", n_estimators=100)'
    node = ast.parse(fake).body[0].value
    assert isinstance(node, ast.Call)
    cls = "AITrainer"
    assert cls in inits, "AITrainer 不在索引里"
    ok_keys, tolerant = accepted_keywords(cls, inits)
    assert tolerant, "AITrainer 应当有 **kwargs"
    kws = [k.arg for k in node.keywords if k.arg]
    unknown = [k for k in kws if k not in ok_keys]
    assert unknown == [
        "model_type_rf"
    ], "负对照失效：期望恰好报出 model_type_rf，实际 %r（接受的关键字=%s）" % (
        unknown,
        sorted(ok_keys),
    )


def test_sklearn_selector_really_takes_estimator():
    """改名后的形参确实存在，且旧的 model_type 不在其中。"""
    inits = build_index()
    named, _ = params(inits["SklearnClusterSelector"])
    assert "estimator" in named
    assert "model_type" not in named, "旧名不应再作为形参存在"
