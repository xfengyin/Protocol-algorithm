"""性能基准测试"""

import pytest
import time
import numpy as np
from typing import List, Dict, Any

from src.models.network import Network
from src.energy.radio_model import FirstOrderRadioModel


class BenchmarkResults:
    """基准测试结果"""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def add(
        self,
        name: str,
        n_nodes: int,
        n_rounds: int,
        execution_time: float,
        operations_per_second: float
    ):
        """添加结果"""
        self.results.append({
            'name': name,
            'n_nodes': n_nodes,
            'n_rounds': n_rounds,
            'execution_time': execution_time,
            'ops_per_sec': operations_per_second
        })

    def summary(self) -> Dict[str, Any]:
        """生成汇总"""
        if not self.results:
            return {}

        times = [r['execution_time'] for r in self.results]
        ops = [r['ops_per_sec'] for r in self.results]

        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'mean_ops': np.mean(ops),
            'total_runs': len(self.results)
        }

    def __str__(self) -> str:
        summary = self.summary()
        if not summary:
            return "No results"

        lines = [
            "=" * 60,
            "BENCHMARK RESULTS",
            "=" * 60,
            f"Total Runs: {summary['total_runs']}",
            f"Mean Time: {summary['mean_time']:.4f}s ± {summary['std_time']:.4f}s",
            f"Min Time: {summary['min_time']:.4f}s",
            f"Max Time: {summary['max_time']:.4f}s",
            f"Mean Ops/sec: {summary['mean_ops']:.2f}",
            "=" * 60
        ]
        return "\n".join(lines)


class TestPerformanceBenchmarks:
    """性能基准测试"""

    @pytest.fixture
    def benchmark_results(self):
        return BenchmarkResults()

    def test_neighbor_search_performance(self, benchmark_results):
        """邻居查找性能测试"""
        for n_nodes in [100, 500, 1000]:
            network = Network(
                n_nodes=n_nodes,
                area=(0, 200, 0, 200),
                base_station_pos=(100, 100),
                seed=42
            )

            node = network.alive_nodes[0]

            start = time.time()
            for _ in range(100):
                neighbors = network.get_neighbors(node, 30.0)
            elapsed = time.time() - start

            ops = 100 / elapsed
            benchmark_results.add(
                'neighbor_search',
                n_nodes,
                100,
                elapsed,
                ops
            )

            print(f"Nodes: {n_nodes}, Time: {elapsed:.4f}s, Ops: {ops:.2f}/s")

    def test_simulation_performance(self, benchmark_results):
        """仿真性能测试"""
        for n_nodes in [50, 100, 200]:
            network = Network(
                n_nodes=n_nodes,
                area=(0, 100, 0, 100),
                base_station_pos=(50, 50),
                seed=42
            )

            start = time.time()
            results = network.simulate_network(rounds=100, protocol_name='leach')
            elapsed = time.time() - start

            ops = 100 / elapsed
            benchmark_results.add(
                'simulation_round',
                n_nodes,
                100,
                elapsed,
                ops
            )

            print(f"Nodes: {n_nodes}, Time: {elapsed:.4f}s, Rounds: {results['total_rounds_simulated']}")

    def test_vectorized_vs_original(self, benchmark_results):
        """向量化 vs 原始实现性能对比"""
        n_nodes = 100
        n_rounds = 50

        print("\n--- Vectorized Implementation ---")
        network_vec = Network(
            n_nodes=n_nodes,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            seed=42
        )

        start = time.time()
        network_vec.simulate_network(rounds=n_rounds, protocol_name='leach')
        vec_time = time.time() - start

        print(f"Vectorized: {vec_time:.4f}s")

        print("\n--- Original Implementation ---")
        network_orig = Network(
            n_nodes=n_nodes,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            seed=42
        )

        start = time.time()
        for _ in range(n_rounds):
            network_orig.setup_phase('leach')
            network_orig.steady_phase_original()
        orig_time = time.time() - start

        print(f"Original: {orig_time:.4f}s")

        speedup = orig_time / vec_time if vec_time > 0 else 0
        print(f"\nSpeedup: {speedup:.2f}x")

        benchmark_results.add(
            'vectorized_vs_original',
            n_nodes,
            n_rounds,
            vec_time,
            n_rounds / vec_time
        )

    def test_memory_usage_estimate(self):
        """内存使用估算"""
        import sys

        network = Network(
            n_nodes=1000,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            seed=42
        )

        size_bytes = sys.getsizeof(network)
        nodes_size = sum(sys.getsizeof(n) for n in network.nodes)

        print(f"\nNetwork object size: {size_bytes / 1024:.2f} KB")
        print(f"Nodes total size: {nodes_size / 1024:.2f} KB")
        print(f"Metrics history: {len(network.metrics_history)} entries")

    def test_scalability(self, benchmark_results):
        """可扩展性测试"""
        sizes = [50, 100, 200, 500]
        times = []

        for n in sizes:
            network = Network(
                n_nodes=n,
                area=(0, 100, 0, 100),
                base_station_pos=(50, 50),
                seed=42
            )

            start = time.time()
            network.simulate_network(rounds=50, protocol_name='leach')
            elapsed = time.time() - start
            times.append(elapsed)

            print(f"Nodes: {n}, Time: {elapsed:.4f}s")

        if len(times) >= 2:
            ratio_2x = times[1] / times[0] if times[0] > 0 else 0
            print(f"\nTime ratio (2x nodes): {ratio_2x:.2f}x")


class TestEnergyModelComparison:
    """能量模型对比测试"""

    def test_model_energy_calculation_comparison(self):
        """不同能量模型计算对比"""
        from src.energy.models import FirstOrderRadioModel, Mica2Model

        first_order = FirstOrderRadioModel()
        mica2 = Mica2Model()

        distance = 50
        message_size = 4000

        fo_tx = first_order.calc_transmit_energy(distance, message_size)
        m2_tx = mica2.calc_transmit_energy(distance, message_size)

        print(f"\nFirst Order TX Energy: {fo_tx:.10f} J")
        print(f"Mica2 TX Energy: {m2_tx:.10f} J")
        print(f"Ratio: {m2_tx / fo_tx:.2f}x")

        fo_rx = first_order.calc_receive_energy(message_size)
        m2_rx = mica2.calc_receive_energy(message_size)

        print(f"\nFirst Order RX Energy: {fo_rx:.10f} J")
        print(f"Mica2 RX Energy: {m2_rx:.10f} J")


class TestAIInference:
    """AI 推理性能测试"""

    def test_feature_extraction_performance(self):
        """特征提取性能"""
        from src.ai.feature_engineering import AdvancedFeatureExtractor

        network = Network(
            n_nodes=200,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            seed=42
        )

        extractor = AdvancedFeatureExtractor(network)

        start = time.time()
        for _ in range(100):
            features = extractor.extract_batch(network.alive_nodes)
        elapsed = time.time() - start

        print(f"\nFeature extraction: {elapsed:.4f}s for 100 iterations")
        print(f"Per iteration: {elapsed/100*1000:.2f}ms")
        print(f"Feature shape: {features.shape}")


def run_all_benchmarks():
    """运行所有基准测试"""
    import sys

    print("=" * 60)
    print("RUNNING PERFORMANCE BENCHMARKS")
    print("=" * 60)

    benchmarks = TestPerformanceBenchmarks()
    results = BenchmarkResults()

    print("\n1. Neighbor Search Performance")
    print("-" * 40)
    benchmarks.test_neighbor_search_performance(results)

    print("\n2. Simulation Performance")
    print("-" * 40)
    benchmarks.test_simulation_performance(results)

    print("\n3. Vectorized vs Original")
    print("-" * 40)
    benchmarks.test_vectorized_vs_original(results)

    print("\n4. Memory Usage")
    print("-" * 40)
    benchmarks.test_memory_usage_estimate()

    print("\n5. Scalability")
    print("-" * 40)
    benchmarks.test_scalability(results)

    print("\n" + str(results))

    return results


if __name__ == '__main__':
    run_all_benchmarks()
