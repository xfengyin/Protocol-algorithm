"""src/benchmark.py 与导入隔离护栏的测试（由 LEACH 仓库合并而来，已适配本仓 API）。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src import benchmark

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_execution_modes_compute_the_same_thing() -> None:
    """串行与两种池跑的是同一个函数：同样的种子必须得到同样的存活节点数。

    这是整个对比有意义的前提——否则比较的是三件不同的事。
    """
    expected = [benchmark._one_run(30, 5, seed) for seed in range(3)]
    again = [benchmark._one_run(30, 5, seed) for seed in range(3)]
    assert expected == again, "同一种子两次结果不同，仿真不可复现，基准没有意义"

    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    for executor_cls in (ThreadPoolExecutor, ProcessPoolExecutor):
        with executor_cls(max_workers=2) as pool:
            got = [pool.submit(benchmark._one_run, 30, 5, seed).result() for seed in range(3)]
        assert got == expected, f"{executor_cls.__name__} 的结果与串行不一致"


def test_benchmark_result_shape_and_no_baked_numbers() -> None:
    rows = benchmark.benchmark(node_counts=(20,), rounds=3, repeats=2, workers=2)
    assert len(rows) == 1
    row = rows[0]
    for key in (
        "node_count",
        "rounds",
        "repeats",
        "serial_seconds",
        "threads_seconds",
        "processes_seconds",
        "serial_over_threads",
        "serial_over_processes",
    ):
        assert key in row
    assert row["node_count"] == 20.0 and row["rounds"] == 3.0 and row["repeats"] == 2.0
    # 耗时必须是真测出来的正数，比值与耗时自洽（不预置数字）。
    assert row["serial_seconds"] > 0 and row["threads_seconds"] > 0 and row["processes_seconds"] > 0
    assert row["serial_over_threads"] == pytest.approx(
        row["serial_seconds"] / row["threads_seconds"]
    )
    assert row["serial_over_processes"] == pytest.approx(
        row["serial_seconds"] / row["processes_seconds"]
    )


def test_format_table_has_header_and_one_line_per_row() -> None:
    rows = [
        {
            "node_count": 20.0,
            "rounds": 3.0,
            "repeats": 2.0,
            "serial_seconds": 1.0,
            "threads_seconds": 2.0,
            "processes_seconds": 0.5,
            "serial_over_threads": 0.5,
            "serial_over_processes": 2.0,
        }
    ]
    text = benchmark.format_table(rows)
    lines = text.splitlines()
    assert "serial_s" in lines[1] and set(lines[2]) == {"-"}
    assert len(lines) == 4 and lines[3].split()[0] == "20"
    assert lines[3].split()[-2:] == ["0.50", "2.00"]


def test_main_accepts_injected_argv(capsys: pytest.CaptureFixture[str]) -> None:
    benchmark.main(["--nodes", "15", "--rounds", "2", "--repeats", "1", "--workers", "1"])
    out = capsys.readouterr().out
    assert "执行方式耗时对比" in out and out.strip().splitlines()[-1].split()[0] == "15"


# --- 导入隔离护栏 -------------------------------------------------------------------------
# 为什么在子进程里判定：pytest 把所有测试收进同一个解释器，同进程里 sys.modules 有没有某个库取决于
# 收集顺序，断言会时真时假。子进程里只 import 一个模块，出现的重库就只可能是它自己拖进来的。
_PROBE = """
import json, sys, importlib
importlib.import_module(sys.argv[1])
heavy = ("matplotlib", "torch", "sklearn")
print(json.dumps(sorted(m for m in sys.modules if m.split(".")[0] in heavy)))
"""

CORE_MODULES = [
    "src.models.network",
    "src.leach.classic",
    "src.energy.radio_model",
    "src.simulation.engine",
]


def _heavy_modules_pulled_by(module: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, module],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"子进程导入 {module} 失败：\n{result.stderr}"
    parsed: list[str] = json.loads(result.stdout)
    return parsed


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_modules_do_not_import_heavy_optional_libs(module: str) -> None:
    """核心仿真模块在导入期不得拖入 matplotlib / torch / sklearn：没装这些可选依赖时核心也必须可用。"""
    leaked = _heavy_modules_pulled_by(module)
    assert leaked == [], f"import {module} 把可选重依赖拖进了导入图：{leaked[:5]}"


def test_guard_is_not_vacuous() -> None:
    """负对照：证明这个探针真的能发现问题——导入明确依赖 matplotlib 的模块，必须报告出来。"""
    leaked = _heavy_modules_pulled_by("src.visualization.comparison")
    assert any(
        m.startswith("matplotlib") for m in leaked
    ), "探针对已知依赖 matplotlib 的模块也没报告，护栏会空转"
