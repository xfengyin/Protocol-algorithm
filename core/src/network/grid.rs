//! Grid topology generation

use crate::leach::Node;
use crate::network::Topology;

/// Grid topology generator
pub struct GridTopology {
    rows: usize,
    cols: usize,
    width: f64,
    height: f64,
}

impl GridTopology {
    /// Create grid topology
    pub fn new(width: f64, height: f64) -> Self {
        Self {
            rows: 10,
            cols: 10,
            width,
            height,
        }
    }

    /// Generate nodes in grid pattern
    pub fn generate_nodes(&self, node_count: usize, initial_energy: f64) -> Vec<Node> {
        let mut nodes = Vec::with_capacity(node_count);
        
        let spacing_x = self.width / (self.cols as f64);
        let spacing_y = self.height / (self.rows as f64);
        
        for i in 0..node_count {
            let row = i / self.cols;
            let col = i % self.cols;
            
            let x = (col as f64 + 0.5) * spacing_x;
            let y = (row as f64 + 0.5) * spacing_y;
            
            nodes.push(Node::new(i, x, y, initial_energy));
        }
        
        nodes
    }
}
