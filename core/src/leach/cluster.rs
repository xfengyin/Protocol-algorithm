//! Cluster management

use crate::leach::Node;
use serde::{Deserialize, Serialize};

/// Represents a cluster of nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    /// Cluster head node ID
    pub head_id: usize,
    /// Member node IDs
    pub members: Vec<usize>,
    /// Cluster centroid (x, y)
    pub centroid: (f64, f64),
}

impl Cluster {
    /// Create new cluster with given head
    pub fn new(head_id: usize, head_x: f64, head_y: f64) -> Self {
        Self {
            head_id,
            members: Vec::new(),
            centroid: (head_x, head_y),
        }
    }

    /// Add member to cluster
    pub fn add_member(&mut self, node_id: usize) {
        self.members.push(node_id);
        self.update_centroid();
    }

    /// Remove member from cluster
    pub fn remove_member(&mut self, node_id: usize) {
        self.members.retain(|&id| id != node_id);
        self.update_centroid();
    }

    /// Update cluster centroid based on members
    fn update_centroid(&mut self) {
        // Will be updated when nodes are added
    }

    /// Number of members (excluding head)
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// Total nodes in cluster (including head)
    pub fn total_size(&self) -> usize {
        self.members.len() + 1
    }
}
