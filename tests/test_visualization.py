"""可视化模块测试

为什么这个文件存在（2026-11-04）：

`src/visualization/` 的 374 行此前**零单测**，理由是「绘图代码测不了，断言图片字节很脆」。
这个理由只对了一半：测**像素**确实没意义，但这两个函数里除了画图还有**指标计算**，
而那部分是可以断言的 —— 也正因没人测，它错了很久没人发现。

具体地，`ComparisonPlotter._plot_performance_comparison` 原来写的是：

    first_dead = np.argmax(alive < alive[0] + alive[0])      # 恒等于 alive < 2*alive[0]
    half_dead  = np.argmax(alive <= (alive[0] + alive[0]) / 2)  # 恒等于 alive <= alive[0]

`alive[0] + alive[0]` 就是 `2 * alive[0]`，于是两个条件对第 0 个元素都为真，
`np.argmax` 永远返回 0 —— 「性能指标对比」图上的 First Dead 与 Half Dead 两根柱子
**恒为 0**。`compare_protocols` 在 examples 里被调用 5 次，报告里的数字也是 0。

本文件的断言只针对「算出来的指标」与「图上画出来的值」，不针对像素。
"""

import matplotlib

matplotlib.use("Agg")  # CI 无 DISPLAY；必须在 import pyplot 之前设置

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from src.visualization.comparison import ComparisonPlotter  # noqa: E402
from src.visualization.metrics_plots import MetricsPlotter  # noqa: E402


def _results(alive):
    """构造一个最小可用的仿真结果字典（字段与 engine 产出一致）。"""
    n = len(alive)
    initial = alive[0]
    return {
        "rounds": list(range(n)),
        "alive_nodes": list(alive),
        "dead_nodes": [initial - a for a in alive],
        "total_energy": [1000.0 - 10 * i for i in range(n)],
    }


def _bar_heights(ax):
    """读出柱状图上每根柱子的高度（= 被画出来的指标值）。"""
    return [float(p.get_height()) for p in ax.patches]


def _bar_labels(ax):
    return [t.get_text() for t in ax.get_xticklabels()]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestPerformanceComparison:
    """`_plot_performance_comparison` 画的必须是真实指标，不是常数。"""

    def test_first_and_half_dead_are_computed_from_data(self):
        """回归：这两根柱子曾经恒为 0（`alive[0] + alive[0]` 的写法）。"""
        # 第 2 轮开始掉到 98（第一次低于初始值 100）
        # 第 6 轮开始是 50（第一次 <= 100/2）
        alive = [100, 100, 98, 95, 80, 60, 50, 30, 10, 0]
        plotter = ComparisonPlotter()
        fig, ax = plt.subplots()
        plotter._plot_performance_comparison([_results(alive)], ["X"], ax)

        labels = _bar_labels(ax)
        heights = _bar_heights(ax)
        got = dict(zip(labels, heights))

        assert got["First Dead"] == 2, (
            "First Dead 应是第一个存活数低于初始值的轮次（2），实际 %r" % got["First Dead"]
        )
        assert got["Half Dead"] == 6, (
            "Half Dead 应是第一个存活数 <= 初始值一半的轮次（6），实际 %r" % got["Half Dead"]
        )
        assert got["Total Rounds"] == 10

    def test_metrics_are_not_constant_across_different_inputs(self):
        """负对照：换一组数据，指标必须跟着变 —— 否则说明它根本没读输入。

        两组数据的**总轮数刻意取成相同**（都是 10），这样 Total Rounds 这根柱子
        不参与区分，只有 First Dead / Half Dead 能区分。
        第一版我把两组长度写成 11 与 10，于是 Total Rounds 天然不同、断言自动通过 ——
        那个负对照其实抓不到 bug（实测：bug 存在时它仍然绿）。收紧后才有效。
        """
        plotter = ComparisonPlotter()
        shapes = []
        for alive in (
            [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],  # 10 轮，第 1 轮就开始掉
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],  # 10 轮，全程不掉
        ):
            fig, ax = plt.subplots()
            plotter._plot_performance_comparison([_results(alive)], ["X"], ax)
            shapes.append(_bar_heights(ax))
            plt.close(fig)
        assert (
            shapes[0] != shapes[1]
        ), "两组存活曲线完全不同（一组持续掉、一组从不掉）却画出了相同指标 %r：说明指标没读输入" % (
            shapes[0],
        )

    def test_survivor_never_dies_reports_total_rounds(self):
        """全程没死时，First Dead 应等于总轮数（表示「没发生」），而不是 0。"""
        alive = [100] * 10
        plotter = ComparisonPlotter()
        fig, ax = plt.subplots()
        plotter._plot_performance_comparison([_results(alive)], ["X"], ax)
        got = dict(zip(_bar_labels(ax), _bar_heights(ax)))
        assert got["First Dead"] == 10, (
            "全程无死亡时应报总轮数 10（哨兵值），实际 %r" % got["First Dead"]
        )


class TestMetricsPlotterSmoke:
    """`MetricsPlotter` 的三个绘图入口至少要能跑完（它们此前一次都没被执行过）。"""

    def test_plot_all_metrics_runs_and_writes_files(self, tmp_path):
        alive = [100, 98, 95, 80, 60, 50, 30, 10, 0]
        results = _results(alive)
        results["cluster_heads"] = [5] * len(alive)
        results["first_dead_round"] = 1
        plotter = MetricsPlotter()
        plotter.plot_all_metrics(results, output_dir=str(tmp_path), show=False)

        written = sorted(p.name for p in tmp_path.glob("*.png"))
        assert written, "plot_all_metrics 没有产出任何图片文件"

    def test_plot_network_lifetime_survives_empty_first_dead_round(self, tmp_path):
        """`first_dead_round` 缺失时不应崩（原代码用 .get，这里锁住这个行为）。"""
        results = _results([100, 90, 80])
        plotter = MetricsPlotter()
        out = tmp_path / "lifetime.png"
        plotter.plot_network_lifetime(results, save_path=str(out), show=False)
        assert out.exists() and out.stat().st_size > 0
