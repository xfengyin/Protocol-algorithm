//! Node representation in WSN

use serde::{Deserialize, Serialize};

/// Represents a sensor node in the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique node identifier
    pub id: usize,
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
    /// Current energy level (Joules)
    pub energy: f64,
    /// Initial energy (Joules)
    pub initial_energy: f64,
    /// Is this node a cluster head?
    pub is_cluster_head: bool,
    /// Cluster head ID this node belongs to (None if this is a CH)
    pub cluster_head_id: Option<usize>,
    /// Is node alive?
    pub is_alive: bool,
}

impl Node {
    /// Create a new node at given position with specified energy
    pub fn new(id: usize, x: f64, y: f64, initial_energy: f64) -> Self {
        Self {
            id,
            x,
            y,
            energy: initial_energy,
            initial_energy,
            is_cluster_head: false,
            cluster_head_id: None,
            is_alive: true,
        }
    }

    /// Check if node is dead
    pub fn is_dead(&self) -> bool {
        self.energy <= 0.0 || !self.is_alive
    }

    /// Consume energy from this node
    pub fn consume_energy(&mut self, amount: f64) {
        self.energy = (self.energy - amount).max(0.0);
        if self.is_dead() {
            self.is_alive = false;
        }
    }

    /// Reset node state for new simulation round
    pub fn reset(&mut self) {
        self.energy = self.initial_energy;
        self.is_cluster_head = false;
        self.cluster_head_id = None;
        self.is_alive = true;
    }

    /// Distance to another node
    pub fn distance_to(&self, other: &Node) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Distance to a point
    pub fn distance_to_point(&self, x: f64, y: f64) -> f64 {
        let dx = self.x - x;
        let dy = self.y - y;
        (dx * dx + dy * dy).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = Node::new(0, 10.0, 20.0, 100.0);
        assert_eq!(node.id, 0);
        assert_eq!(node.x, 10.0);
        assert_eq!(node.y, 20.0);
        assert_eq!(node.energy, 100.0);
        assert!(!node.is_cluster_head);
    }

    #[test]
    fn test_energy_consumption() {
        let mut node = Node::new(0, 0.0, 0.0, 100.0);
        node.consume_energy(30.0);
        assert_eq!(node.energy, 70.0);
        assert!(node.is_alive);
    }

    #[test]
    fn test_node_death() {
        let mut node = Node::new(0, 0.0, 0.0, 100.0);
        node.consume_energy(150.0);
        assert_eq!(node.energy, 0.0);
        assert!(!node.is_alive);
        assert!(node.is_dead());
    }

    #[test]
    fn test_distance() {
        let node1 = Node::new(0, 0.0, 0.0, 100.0);
        let node2 = Node::new(1, 3.0, 4.0, 100.0);
        assert_eq!(node1.distance_to(&node2), 5.0);
    }
}
