//! Simulation metrics

use serde::{Deserialize, Serialize};

/// Metrics collected during simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationMetrics {
    /// Total rounds completed
    pub rounds: usize,
    /// First node death round
    pub first_death_round: Option<usize>,
    /// Half nodes death round
    pub half_death_round: Option<usize>,
    /// Last node death round
    pub last_death_round: Option<usize>,
    /// Total energy consumed
    pub total_energy_consumed: f64,
    /// Average energy per round
    pub avg_energy_per_round: f64,
    /// Cluster head count per round
    pub ch_count_per_round: Vec<usize>,
    /// Alive nodes per round
    pub alive_nodes_per_round: Vec<usize>,
}

impl SimulationMetrics {
    /// Create empty metrics
    pub fn new() -> Self {
        Self {
            rounds: 0,
            first_death_round: None,
            half_death_round: None,
            last_death_round: None,
            total_energy_consumed: 0.0,
            avg_energy_per_round: 0.0,
            ch_count_per_round: Vec::new(),
            alive_nodes_per_round: Vec::new(),
        }
    }
}

impl Default for SimulationMetrics {
    fn default() -> Self {
        Self::new()
    }
}
