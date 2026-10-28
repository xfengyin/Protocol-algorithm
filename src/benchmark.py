"""同步 / 并行执行模式基准对比（由 LEACH 仓库合并而来，已适配本仓的 Network API）

LEACH 仓库里这份基准比较的是同一份实现的 run_round() 与 run_round_async()。本仓没有 async 路径
（`grep -rn "async def" src` 为 0），也不该为了基准去造一个：本仓真正存在的两条执行路径是
`ParallelSimulationEngine` 的 **进程池** 与 **线程池**。所以这里比较的是：

    串行（一个进程内顺序跑）  vs  线程池  vs  进程池

用同一组 SimulationConfig，只改执行方式，计时。结果依机器而异，本模块**不预置任何基准数字**，
也不把它们写进 README。

用法:
    python -m src.benchmark --nodes 50,100 --rounds 20 --workers 2
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Dict, List, Optional, Sequence

from .models.network import Network

logger = logging.getLogger(__name__)

DEFAULT_NODE_COUNTS = (50, 100, 200)
DEFAULT_ROUNDS = 20
AREA = (0.0, 100.0, 0.0, 100.0)
BASE_STATION = (50.0, 175.0)


def _one_run(n_nodes: int, rounds: int, seed: int) -> int:
    """跑一次完整仿真，返回仿真结束时的存活节点数（用来证明各执行路径算的是同一件事）。"""
    network = Network(n_nodes, AREA, BASE_STATION, seed=seed)
    for _ in range(rounds):
        network.setup_phase("leach")
        network.steady_phase()
    return network.n_alive


def run_serial(n_nodes: int, rounds: int, repeats: int) -> float:
    start = time.perf_counter()
    for seed in range(repeats):
        _one_run(n_nodes, rounds, seed)
    return time.perf_counter() - start


def run_pool(n_nodes: int, rounds: int, repeats: int, workers: int, use_threads: bool) -> float:
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    start = time.perf_counter()
    with executor_cls(max_workers=workers) as pool:
        futures = [pool.submit(_one_run, n_nodes, rounds, seed) for seed in range(repeats)]
        for future in futures:
            future.result()
    return time.perf_counter() - start


def benchmark(
    node_counts: Sequence[int] = DEFAULT_NODE_COUNTS,
    rounds: int = DEFAULT_ROUNDS,
    repeats: int = 4,
    workers: int = 2,
) -> List[Dict[str, float]]:
    """逐档节点数对比 串行 / 线程池 / 进程池 的耗时。"""
    results: List[Dict[str, float]] = []
    for n_nodes in node_counts:
        serial = run_serial(n_nodes, rounds, repeats)
        threads = run_pool(n_nodes, rounds, repeats, workers, use_threads=True)
        processes = run_pool(n_nodes, rounds, repeats, workers, use_threads=False)
        results.append(
            {
                "node_count": float(n_nodes),
                "rounds": float(rounds),
                "repeats": float(repeats),
                "serial_seconds": serial,
                "threads_seconds": threads,
                "processes_seconds": processes,
                # 大于 1 表示该执行方式比串行快。小规模时进程启动 / 调度开销占主导，可能小于 1。
                "serial_over_threads": serial / threads if threads else 0.0,
                "serial_over_processes": serial / processes if processes else 0.0,
            }
        )
        logger.info(
            "节点数 %s：串行 %.4fs，线程池 %.4fs，进程池 %.4fs", n_nodes, serial, threads, processes
        )
    return results


def format_table(results: List[Dict[str, float]]) -> str:
    lines = [
        "执行方式耗时对比（同一组仿真配置，只改执行方式）",
        f"{'nodes':>7} {'rounds':>7} {'serial_s':>10} {'threads_s':>10} "
        f"{'procs_s':>10} {'s/thr':>7} {'s/proc':>7}",
        "-" * 66,
    ]
    for row in results:
        lines.append(
            f"{int(row['node_count']):>7} {int(row['rounds']):>7} "
            f"{row['serial_seconds']:>10.4f} {row['threads_seconds']:>10.4f} "
            f"{row['processes_seconds']:>10.4f} "
            f"{row['serial_over_threads']:>7.2f} {row['serial_over_processes']:>7.2f}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="串行 / 线程池 / 进程池 执行方式基准对比")
    parser.add_argument(
        "--nodes",
        default=",".join(str(c) for c in DEFAULT_NODE_COUNTS),
        help="逗号分隔的节点数",
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="每次仿真跑多少轮")
    parser.add_argument("--repeats", type=int, default=4, help="每档重复仿真次数（不同随机种子）")
    parser.add_argument("--workers", type=int, default=2, help="池的并发数")
    args = parser.parse_args(argv)
    counts = tuple(int(p) for p in args.nodes.split(",") if p.strip())
    print(format_table(benchmark(counts, args.rounds, args.repeats, args.workers)))


if __name__ == "__main__":
    main()
