//! Simulation engine

use crate::leach::{LEACH, LEACHConfig, LEACHState, Node};
use crate::network::Topology;

/// Simulation runner
pub struct Simulation {
    leach: LEACH,
    nodes: Vec<Node>,
    topology: Topology,
}

impl Simulation {
    /// Create new simulation
    pub fn new(config: LEACHConfig, topology: Topology, node_count: usize) -> Self {
        let nodes = (0..node_count)
            .map(|i| {
                // Random position for now
                let x = (i as f64 * 17.0) % topology.area_width;
                let y = (i as f64 * 23.0) % topology.area_height;
                Node::new(i, x, y, config.initial_energy)
            })
            .collect();

        let leach = LEACH::new(config);

        Self {
            leach,
            nodes,
            topology,
        }
    }

    /// Run simulation
    pub fn run(&mut self) -> Vec<LEACHState> {
        self.leach.run_simulation(&mut self.nodes, &self.topology)
    }

    /// Get nodes
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Get mutable nodes
    pub fn nodes_mut(&mut self) -> &mut [Node] {
        &mut self.nodes
    }
}
