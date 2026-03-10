# Python Tests for Protocol-algorithm Bindings

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import protocol_algo as pa
    HAS_BINDINGS = True
except ImportError:
    HAS_BINDINGS = False
    # Use mock classes for testing without Rust bindings
    class Network:
        def __init__(self, nodes=100, area=100.0, base_station=(50.0, 150.0)):
            self.nodes = nodes
            self.area = area
            self.base_station = base_station
    
    class LEACH:
        def __init__(self, p=0.05, rounds=100, initial_energy=0.5, seed=42):
            self.p = p
            self.rounds = rounds
            self.initial_energy = initial_energy
            self.seed = seed
        
        def run(self, network):
            class Result:
                rounds = self.rounds
                initial_nodes = network.nodes
                final_alive = int(network.nodes * 0.7)
                def survival_rate(self):
                    return (self.final_alive / self.initial_nodes) * 100
            return Result()
    
    class Visualizer:
        def __init__(self, style="modern"):
            self.style = style
        def plot_network(self, *args): pass
        def plot_metrics(self, *args): pass
        def save(self, path): pass


@pytest.mark.skipif(not HAS_BINDINGS, reason="Rust bindings not available")
class TestNetwork:
    def test_network_creation(self):
        network = pa.Network(nodes=100, area=100.0)
        assert network.nodes == 100
        assert network.area == 100.0
    
    def test_custom_base_station(self):
        network = pa.Network(base_station=(75.0, 200.0))
        assert network.base_station == (75.0, 200.0)
    
    def test_repr(self):
        network = pa.Network(nodes=50)
        assert "50" in repr(network)


@pytest.mark.skipif(not HAS_BINDINGS, reason="Rust bindings not available")
class TestLEACH:
    def test_default_config(self):
        leach = pa.LEACH()
        assert leach.p == 0.05
        assert leach.rounds == 100
    
    def test_custom_config(self):
        leach = pa.LEACH(p=0.1, rounds=200, initial_energy=1.0)
        assert leach.p == 0.1
        assert leach.rounds == 200
    
    def test_run_simulation(self):
        network = pa.Network(nodes=100)
        leach = pa.LEACH(rounds=50)
        result = leach.run(network)
        
        assert result.rounds == 50
        assert result.initial_nodes == 100
        assert result.final_alive > 0
    
    def test_survival_rate(self):
        network = pa.Network(nodes=100)
        leach = pa.LEACH()
        result = leach.run(network)
        
        rate = result.survival_rate()
        assert 0 < rate <= 100


@pytest.mark.skipif(not HAS_BINDINGS, reason="Rust bindings not available")
class TestVisualizer:
    def test_default_style(self):
        viz = pa.Visualizer()
        assert viz.style == "modern"
    
    def test_custom_style(self):
        viz = pa.Visualizer(style="dark")
        assert viz.style == "dark"


class TestIntegration:
    def test_full_workflow(self):
        """Test complete simulation workflow"""
        network = pa.Network(nodes=100, area=100.0)
        leach = pa.LEACH(p=0.05, rounds=100)
        result = leach.run(network)
        
        viz = pa.Visualizer(style="modern")
        viz.plot_network(network, result)
        viz.plot_metrics(result)
        
        assert result.survival_rate() > 0
    
    def test_varying_parameters(self):
        """Test with different parameters"""
        for nodes in [50, 100, 200]:
            network = pa.Network(nodes=nodes)
            leach = pa.LEACH(rounds=50)
            result = leach.run(network)
            
            assert result.initial_nodes == nodes
            assert result.final_alive <= nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
