//! LEACH Protocol Implementation

use crate::leach::{Cluster, EnergyModel, Node};
use crate::network::Topology;
use rand::prelude::*;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

/// LEACH Protocol Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LEACHConfig {
    /// Probability of becoming cluster head
    pub p: f64,
    /// Number of rounds
    pub rounds: usize,
    /// Initial energy per node (J)
    pub initial_energy: f64,
    /// Base station position (x, y)
    pub base_station: (f64, f64),
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for LEACHConfig {
    fn default() -> Self {
        Self {
            p: 0.05,
            rounds: 100,
            initial_energy: 0.5,
            base_station: (50.0, 150.0),
            seed: 42,
        }
    }
}

/// LEACH Protocol State
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LEACHState {
    /// Current round number
    pub current_round: usize,
    /// Number of alive nodes
    pub alive_nodes: usize,
    /// Number of cluster heads
    pub cluster_heads: usize,
    /// Total energy consumed
    pub total_energy_consumed: f64,
}

/// LEACH Protocol Implementation
pub struct LEACH {
    config: LEACHConfig,
    energy_model: EnergyModel,
    rng: Xoshiro256PlusPlus,
}

impl LEACH {
    /// Create new LEACH protocol with config
    pub fn new(config: LEACHConfig) -> Self {
        let rng = Xoshiro256PlusPlus::seed_from_u64(config.seed);
        Self {
            config,
            energy_model: EnergyModel::default(),
            rng,
        }
    }

    /// Create LEACH with custom energy model
    pub fn with_energy_model(config: LEACHConfig, energy_model: EnergyModel) -> Self {
        let rng = Xoshiro256PlusPlus::seed_from_u64(config.seed);
        Self {
            config,
            energy_model,
            rng,
        }
    }

    /// Run one round of LEACH protocol
    pub fn run_round(&mut self, nodes: &mut [Node], topology: &Topology) -> LEACHState {
        let mut rng = self.rng.clone();
        
        // Step 1: Select cluster heads
        self.select_cluster_heads(nodes, &mut rng);
        
        // Step 2: Form clusters
        self.form_clusters(nodes);
        
        // Step 3: Simulate communication
        self.simulate_communication(nodes, topology);
        
        // Step 4: Update state
        let alive_nodes = nodes.iter().filter(|n| n.is_alive).count();
        let cluster_heads = nodes.iter().filter(|n| n.is_cluster_head).count();
        
        LEACHState {
            current_round: self.config.rounds,
            alive_nodes,
            cluster_heads,
            total_energy_consumed: 0.0, // Will be calculated
        }
    }

    /// Select cluster heads using LEACH algorithm
    fn select_cluster_heads(&mut self, nodes: &mut [Node], rng: &mut impl RngCore) {
        // Reset cluster head status
        for node in nodes.iter_mut() {
            node.is_cluster_head = false;
            node.cluster_head_id = None;
        }

        // Select CHs based on probability
        let alive_nodes: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_alive)
            .map(|(i, _)| i)
            .collect();

        let expected_chs = (alive_nodes.len() as f64 * self.config.p).ceil() as usize;
        
        // Random selection with energy consideration
        let mut candidates: Vec<(usize, f64)> = alive_nodes
            .iter()
            .map(|&i| {
                let energy_factor = nodes[i].energy / nodes[i].initial_energy;
                (i, energy_factor)
            })
            .collect();
        
        // Sort by energy (higher energy = higher priority)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Select top candidates as CHs
        for (i, _) in candidates.into_iter().take(expected_chs) {
            nodes[i].is_cluster_head = true;
        }
    }

    /// Form clusters by associating non-CH nodes with nearest CH
    fn form_clusters(&self, nodes: &mut [Node]) {
        let ch_positions: Vec<(usize, f64, f64)> = nodes
            .iter()
            .filter(|n| n.is_alive && n.is_cluster_head)
            .map(|n| (n.id, n.x, n.y))
            .collect();

        for node in nodes.iter_mut() {
            if !node.is_alive || node.is_cluster_head {
                continue;
            }

            // Find nearest CH
            let nearest_ch = ch_positions
                .iter()
                .min_by(|a, b| {
                    let dist_a = node.distance_to_point(a.1, a.2);
                    let dist_b = node.distance_to_point(b.1, b.2);
                    dist_a.partial_cmp(&dist_b).unwrap()
                });

            if let Some((ch_id, _, _)) = nearest_ch {
                node.cluster_head_id = Some(*ch_id);
            }
        }
    }

    /// Simulate communication and energy consumption
    fn simulate_communication(&mut self, nodes: &mut [Node], topology: &Topology) {
        let bs_pos = self.config.base_station;

        for node in nodes.iter_mut() {
            if !node.is_alive {
                continue;
            }

            if node.is_cluster_head {
                // CH: receive from members, aggregate, transmit to BS
                let members: Vec<&Node> = nodes
                    .iter()
                    .filter(|n| n.is_alive && n.cluster_head_id == Some(node.id))
                    .collect();
                
                let member_count = members.len();
                let avg_distance = if member_count > 0 {
                    members.iter().map(|m| node.distance_to(m)).sum::<f64>() / (member_count as f64)
                } else {
                    0.0
                };

                let energy = self.energy_model.ch_energy(member_count, avg_distance);
                node.consume_energy(energy);
            } else if let Some(ch_id) = node.cluster_head_id {
                // Non-CH: transmit to CH
                if let Some(ch) = nodes.iter().find(|n| n.id == ch_id) {
                    let distance = node.distance_to(ch);
                    let energy = self.energy_model.tx_energy(distance);
                    node.consume_energy(energy);
                }
            }
        }
    }

    /// Run complete simulation
    pub fn run_simulation(
        &mut self,
        nodes: &mut [Node],
        topology: &Topology,
    ) -> Vec<LEACHState> {
        let mut states = Vec::new();

        for round in 0..self.config.rounds {
            self.config.rounds = round + 1;
            let state = self.run_round(nodes, topology);
            states.push(state);
        }

        states
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::GridTopology;

    #[test]
    fn test_leach_creation() {
        let config = LEACHConfig::default();
        let leach = LEACH::new(config);
        assert_eq!(leach.config.p, 0.05);
    }

    #[test]
    fn test_cluster_head_selection() {
        let config = LEACHConfig {
            p: 0.1,
            rounds: 1,
            initial_energy: 1.0,
            ..Default::default()
        };
        
        let mut leach = LEACH::new(config);
        let mut nodes = vec![
            Node::new(0, 0.0, 0.0, 1.0),
            Node::new(1, 10.0, 10.0, 1.0),
            Node::new(2, 20.0, 20.0, 1.0),
        ];
        
        let topology = Topology::Grid(GridTopology::new(100.0, 100.0));
        leach.run_round(&mut nodes, &topology);
        
        // At least one CH should be selected
        let ch_count = nodes.iter().filter(|n| n.is_cluster_head).count();
        assert!(ch_count >= 1);
    }
}
