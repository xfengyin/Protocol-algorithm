//! Network topology types

use serde::{Deserialize, Serialize};

/// Type of network topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TopologyType {
    Random,
    Grid,
    Clustered,
    Custom,
}

/// Network topology configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topology {
    pub topology_type: TopologyType,
    pub area_width: f64,
    pub area_height: f64,
    pub base_station: (f64, f64),
}

impl Topology {
    /// Create random topology
    pub fn random(width: f64, height: f64, bs: (f64, f64)) -> Self {
        Self {
            topology_type: TopologyType::Random,
            area_width: width,
            area_height: height,
            base_station: bs,
        }
    }

    /// Create grid topology
    pub fn grid(width: f64, height: f64, bs: (f64, f64)) -> Self {
        Self {
            topology_type: TopologyType::Grid,
            area_width: width,
            area_height: height,
            base_station: bs,
        }
    }
}
