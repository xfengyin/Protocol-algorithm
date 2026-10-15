"""BatchSimulator 的聚合与协议对比（回归测试）

回归来源（2026-10-15）：`ParallelSimulationEngine._aggregate_results` 里
`aggregated["summary"]` 初始化成 `{}` 之后**从未被填充**，只填了 `by_params`；
而 `BatchSimulator.compare_protocols` 读的正是 `["summary"]["network_lifetime"]`
⇒ 每个协议的 lifetime 都取默认值 0，`best_protocol` 恒等于协议列表的第一个
（`max()` 在 key 相同时返回第一个元素），与仿真结果无关。

这个错值是可被用户直接消费的公开输出：
`examples/advanced_usage.py`（README 的示例脚本表里有它）会打印这些 0 与「最优协议」，
`docs/tutorial.md` 也照样演示。所以按「公开输出恒错」修，而不是当遗留项。
"""

from typing import Any, Dict

import pytest

from src.simulation.engine import BatchSimulator, ParallelSimulationEngine


def _fake_sweep(lifetimes: Dict[str, float]) -> Any:
    """构造一个假的 run_parameter_sweep：按 protocol_name 返回不同的 network_lifetime。

    用它而不是真跑仿真，是为了让「best_protocol 必须挑出较优的那个」这条断言与
    物理参数无关（真跑的话谁更优取决于参数，测不出判别力）。
    """

    def _sweep(
        base_config: Any, param_grid: Any, n_runs: int = 1, aggregate: bool = True
    ) -> Dict[str, Any]:
        return {"summary": {"network_lifetime": lifetimes[base_config.protocol_name]}}

    return _sweep


def test_compare_protocols_picks_the_longest_lived(monkeypatch: pytest.MonkeyPatch) -> None:
    """判别力测试：较优协议排在列表**第二个**，断言被选中的是它而不是第一个。"""
    lifetimes = {"leach": 900.0, "leach_ee": 1500.0}
    batch = BatchSimulator()
    monkeypatch.setattr(batch.engine, "run_parameter_sweep", _fake_sweep(lifetimes))

    comparison = batch.compare_protocols(protocols=["leach", "leach_ee"], n_runs=1)

    assert comparison["summary"]["lifetimes"] == lifetimes
    # 修之前这里是 "leach"（列表第一个），因为所有 lifetime 都是 0
    assert comparison["summary"]["best_protocol"] == "leach_ee"


def test_compare_protocols_real_run_has_nonzero_lifetimes() -> None:
    """端到端：真跑两条协议，summary 里必须是有意义的正数，且 best 与 lifetimes 自洽。"""
    batch = BatchSimulator(engine=ParallelSimulationEngine(n_workers=2))

    comparison = batch.compare_protocols(
        n_nodes=30,
        area=(0, 50, 0, 50),
        rounds=200,
        protocols=["leach", "leach_ee"],
        n_runs=1,
    )

    lifetimes = comparison["summary"]["lifetimes"]
    assert all(value > 0 for value in lifetimes.values()), lifetimes
    assert comparison["summary"]["best_protocol"] == max(lifetimes, key=lifetimes.get)
    # 明细没有被 summary 取代
    assert comparison["protocols"]["leach"]["by_params"]
