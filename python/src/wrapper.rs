//! Python API wrapper for Protocol-algorithm

use pyo3::prelude::*;
use protocol_algo_core as core;
use protocol_algo_core::leach::{LEACHConfig, Node};
use protocol_algo_core::network::Topology;
use protocol_algo_core::simulation::Simulation;

/// Network configuration for WSN
#[pyclass]
pub struct Network {
    nodes: usize,
    area: f64,
    base_station: (f64, f64),
}

#[pymethods]
impl Network {
    #[new]
    #[pyo3(signature = (nodes=100, area=100.0, base_station=(50.0, 150.0)))]
    fn new(nodes: usize, area: f64, base_station: (f64, f64)) -> Self {
        Self {
            nodes,
            area,
            base_station,
        }
    }

    fn __repr__(&self) -> String {
        format!("Network(nodes={}, area={}, base_station={:?})", 
                self.nodes, self.area, self.base_station)
    }
}

/// LEACH Protocol configuration
#[pyclass]
pub struct LEACH {
    p: f64,
    rounds: usize,
    initial_energy: f64,
    seed: u64,
}

#[pymethods]
impl LEACH {
    #[new]
    #[pyo3(signature = (p=0.05, rounds=100, initial_energy=0.5, seed=42))]
    fn new(p: f64, rounds: usize, initial_energy: f64, seed: u64) -> Self {
        Self {
            p,
            rounds,
            initial_energy,
            seed,
        }
    }

    /// Run simulation
    fn run(&self, network: &Network) -> SimulationResult {
        let config = LEACHConfig {
            p: self.p,
            rounds: self.rounds,
            initial_energy: self.initial_energy,
            base_station: network.base_station,
            seed: self.seed,
        };

        let topology = Topology::random(network.area, network.area, network.base_station);
        let mut simulation = Simulation::new(config, topology, network.nodes);
        let states = simulation.run();

        SimulationResult {
            rounds: self.rounds,
            initial_nodes: network.nodes,
            final_alive: states.last().map(|s| s.alive_nodes).unwrap_or(0),
        }
    }

    fn __repr__(&self) -> String {
        format!("LEACH(p={}, rounds={}, initial_energy={})", 
                self.p, self.rounds, self.initial_energy)
    }
}

/// Simulation result
#[pyclass]
pub struct SimulationResult {
    rounds: usize,
    initial_nodes: usize,
    final_alive: usize,
}

#[pymethods]
impl SimulationResult {
    fn __repr__(&self) -> String {
        format!("SimulationResult(rounds={}, initial={}, final_alive={})", 
                self.rounds, self.initial_nodes, self.final_alive)
    }

    /// Get survival rate
    fn survival_rate(&self) -> f64 {
        (self.final_alive as f64 / self.initial_nodes as f64) * 100.0
    }
}

/// Visualization helper
#[pyclass]
pub struct Visualizer {
    style: String,
}

#[pymethods]
impl Visualizer {
    #[new]
    #[pyo3(signature = (style="modern"))]
    fn new(style: &str) -> Self {
        Self {
            style: style.to_string(),
        }
    }

    /// Plot network topology (placeholder)
    fn plot_network(&self, _network: &Network, _result: &SimulationResult) {
        println!("Network visualization (style: {})", self.style);
    }

    /// Plot metrics (placeholder)
    fn plot_metrics(&self, _result: &SimulationResult) {
        println!("Metrics visualization");
    }

    /// Save to file
    fn save(&self, path: &str) {
        println!("Saved to {}", path);
    }

    fn __repr__(&self) -> String {
        format!("Visualizer(style='{}')", self.style)
    }
}
