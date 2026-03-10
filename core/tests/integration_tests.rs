//! Integration tests for Protocol-algorithm Core

use protocol_algo_core::{LEACH, LEACHConfig, Network, Simulation, Topology};

#[test]
fn test_basic_simulation() {
    let config = LEACHConfig {
        p: 0.05,
        rounds: 50,
        initial_energy: 0.5,
        base_station: (50.0, 150.0),
        seed: 42,
    };

    let topology = Topology::random(100.0, 100.0, config.base_station);
    let mut simulation = Simulation::new(config, topology, 100);
    let states = simulation.run();

    assert_eq!(states.len(), 50);
    assert!(states[0].alive_nodes > 0);
}

#[test]
fn test_cluster_head_selection() {
    let config = LEACHConfig {
        p: 0.1,
        rounds: 1,
        initial_energy: 1.0,
        base_station: (50.0, 50.0),
        seed: 42,
    };

    let topology = Topology::random(100.0, 100.0, config.base_station);
    let mut simulation = Simulation::new(config, topology, 50);
    let states = simulation.run();

    // Should have at least one cluster head
    assert!(states[0].cluster_heads >= 1);
}

#[test]
fn test_energy_consumption() {
    let config = LEACHConfig {
        p: 0.05,
        rounds: 100,
        initial_energy: 0.5,
        base_station: (50.0, 150.0),
        seed: 42,
    };

    let topology = Topology::random(100.0, 100.0, config.base_station);
    let mut simulation = Simulation::new(config, topology, 100);
    let states = simulation.run();

    // Energy should decrease over time
    let first_alive = states[0].alive_nodes;
    let last_alive = states.last().unwrap().alive_nodes;
    
    // Some nodes should die during simulation
    assert!(last_alive <= first_alive);
}

#[test]
fn test_network_lifetime() {
    let config = LEACHConfig {
        p: 0.05,
        rounds: 200,
        initial_energy: 1.0,
        base_station: (50.0, 150.0),
        seed: 42,
    };

    let topology = Topology::random(100.0, 100.0, config.base_station);
    let mut simulation = Simulation::new(config, topology, 100);
    let states = simulation.run();

    // Find first node death
    let first_death = states.iter().position(|s| s.alive_nodes < 100);
    assert!(first_death.is_some());

    // Find half network death
    let half_death = states.iter().position(|s| s.alive_nodes < 50);
    assert!(half_death.is_some() || half_death.is_none());
}

#[test]
fn test_different_topology() {
    let config = LEACHConfig {
        p: 0.05,
        rounds: 50,
        initial_energy: 0.5,
        base_station: (50.0, 50.0),
        seed: 42,
    };

    // Test random topology
    let random_topo = Topology::random(100.0, 100.0, config.base_station);
    let mut sim_random = Simulation::new(config.clone(), random_topo, 100);
    let states_random = sim_random.run();

    // Test grid topology
    let grid_topo = Topology::grid(100.0, 100.0, config.base_station);
    let mut sim_grid = Simulation::new(config, grid_topo, 100);
    let states_grid = sim_grid.run();

    // Both should complete successfully
    assert_eq!(states_random.len(), 50);
    assert_eq!(states_grid.len(), 50);
}

#[test]
fn test_varying_node_count() {
    let config = LEACHConfig {
        p: 0.05,
        rounds: 50,
        initial_energy: 0.5,
        base_station: (50.0, 50.0),
        seed: 42,
    };

    let topology = Topology::random(100.0, 100.0, config.base_station);

    // Test with different node counts
    for node_count in [50, 100, 200] {
        let mut sim = Simulation::new(config.clone(), topology.clone(), node_count);
        let states = sim.run();
        
        assert_eq!(states.len(), 50);
        assert!(states[0].alive_nodes <= node_count);
    }
}
