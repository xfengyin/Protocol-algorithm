//! Protocol-algorithm Python Bindings
//! 
//! High-performance WSN protocol simulation with beautiful visualizations.

use pyo3::prelude::*;
use protocol_algo_core as core;

mod wrapper;

/// Python module for protocol-algorithm
#[pymodule]
fn protocol_algo(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<wrapper::Network>()?;
    m.add_class::<wrapper::LEACH>()?;
    m.add_class::<wrapper::Visualizer>()?;
    m.add_class::<wrapper::SimulationResult>()?;
    Ok(())
}
